"""Qwen2.5-3B-Instruct generation with token-level probability signals."""

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


_SYSTEM_PROMPT = (
    "You are an expert medical researcher. "
    "Answer the question using only the information provided in the context. "
    "Be concise and factual."
)


def build_prompt(question: str, context_text: str) -> list[dict]:
    """Return chat messages list for the tokenizer's apply_chat_template."""
    user_content = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class QwenGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        do_sample: bool = False,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Left-pad so batched generation aligns generated tokens correctly
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CUDA: use auto + bfloat16. Mac/CPU: use CPU + float32
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",  # required for output_attentions=True
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                attn_implementation="eager",  # required for output_attentions=True
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        self.model.eval()

    # ------------------------------------------------------------------
    # Single-example interface (wraps batch for convenience)
    # ------------------------------------------------------------------
    def generate(self, question: str, context_text: str) -> dict[str, Any]:
        return self.generate_batch([(question, context_text)])[0]

    # ------------------------------------------------------------------
    # Batched interface
    # ------------------------------------------------------------------
    def generate_batch(self, examples: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """
        Generate answers for a batch of (question, context_text) pairs.

        Returns a list of dicts (one per example), each containing:
            generated_answer, answer_n_tokens, tokens, token_log_probs,
            top100_logit_values, top100_logit_token_ids, token_entropies,
            context_attention_ratios, mean_input_attention,
            middle_layer_hidden_state, context_start_idx, context_end_idx,
            padding_offset, input_len, uncertainty_features
        """
        from transformers import GenerationConfig

        batch_size = len(examples)

        # --- Build prompts and tokenize ---
        input_texts = []
        for question, context_text in examples:
            messages = build_prompt(question, context_text)
            input_texts.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.model.device)

        # input_len per example = number of real (non-padding) tokens
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()  # list of ints

        # padded_input_len is the same for all examples in the batch
        padded_input_len = inputs["input_ids"].shape[1]

        # --- Context token indices (per example, in unpadded token space) ---
        context_indices = []
        for i, (question, context_text) in enumerate(examples):
            input_text = input_texts[i]
            ctx_marker = "Context:\n"
            ctx_marker_pos = input_text.find(ctx_marker)
            if ctx_marker_pos == -1:
                context_indices.append((0, 0))
            else:
                context_start_char = ctx_marker_pos + len(ctx_marker)
                context_end_char = context_start_char + len(context_text)
                context_start_idx = len(
                    self.tokenizer(
                        input_text[:context_start_char],
                        add_special_tokens=False,
                    )["input_ids"]
                )
                context_end_idx = len(
                    self.tokenizer(
                        input_text[:context_end_char],
                        add_special_tokens=False,
                    )["input_ids"]
                )
                context_indices.append((context_start_idx, context_end_idx))

        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=0.8 if self.do_sample else 1.0,
            top_k=20 if self.do_sample else 0,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                output_scores=True,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # outputs.sequences: (batch, padded_input_len + n_generated)
        # outputs.scores: tuple of n_generated, each (batch, vocab_size)
        # outputs.attentions: tuple of n_generated, each a tuple of n_layers tensors
        #   each layer tensor: (batch, n_heads, 1, seq_len_t) with KV cache
        #   seq_len_t = padded_input_len + t  (grows by 1 each step)
        # outputs.hidden_states: tuple of n_generated, each a tuple of (n_layers+1) tensors
        #   step 0, layer: (batch, 1, hidden_size) with KV cache
        generated_ids_batch = outputs.sequences[:, padded_input_len:]  # (batch, n_gen)

        eos_id = self.tokenizer.eos_token_id
        results = []

        for b in range(batch_size):
            gen_ids = generated_ids_batch[b]  # (n_gen,)

            # Find where EOS first appears (if at all)
            eos_positions = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                n_valid = eos_positions[0].item() + 1  # include EOS token
            else:
                n_valid = len(gen_ids)

            valid_ids = gen_ids[:n_valid]
            tokens = self.tokenizer.convert_ids_to_tokens(valid_ids.tolist())
            generated_answer = self.tokenizer.decode(valid_ids, skip_special_tokens=True).strip()

            # --- Per-token scores ---
            token_log_probs: list[float] = []
            token_probs: list[float] = []
            top2_probs: list[float] = []
            token_entropies: list[float] = []
            top100_logit_values: list[list[float]] = []
            top100_logit_token_ids: list[list[int]] = []

            for step_idx in range(n_valid):
                logits_1d = outputs.scores[step_idx][b]  # (vocab_size,)
                probs = F.softmax(logits_1d, dim=-1)
                token_id = valid_ids[step_idx].item()
                selected_prob = probs[token_id].item()

                # top2: index 0 = top-1 (selected with greedy), index 1 = runner-up
                top2 = torch.topk(probs, k=2)
                top2_prob = top2.values[1].item()

                token_probs.append(selected_prob)
                token_log_probs.append(min(0.0, math.log(selected_prob + 1e-10)))
                top2_probs.append(top2_prob)

                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                token_entropies.append(entropy)

                top100 = torch.topk(logits_1d, k=100)
                top100_logit_values.append(top100.values.cpu().float().tolist())
                top100_logit_token_ids.append(top100.indices.cpu().tolist())

            # --- Padding offset and padded context indices ---
            context_start_idx, context_end_idx = context_indices[b]
            input_len = int(input_lens[b])
            padding_offset = padded_input_len - input_len
            ctx_start_padded = context_start_idx + padding_offset
            ctx_end_padded = context_end_idx + padding_offset

            # --- Mean input attention + context attention ratio ---
            # At step t, attention shape is (batch, n_heads, 1, padded_input_len + t).
            # We keep only the first padded_input_len positions (attention to input tokens)
            # so all steps have a consistent shape that can be averaged.
            # context_attention_ratio: fraction of total attention (including to generated
            # tokens) that falls on context tokens — one scalar per generated token.
            mean_input_attn: np.ndarray | None = None  # (n_layers, n_heads, padded_input_len)
            context_attention_ratios: list[float] = []

            for step_idx in range(n_valid):
                # step_attn_np: (n_layers, n_heads, seq_len_t)
                step_attn_np = np.stack(
                    [layer[b, :, -1, :].cpu().float().numpy()
                     for layer in outputs.attentions[step_idx]],
                    axis=0,
                )

                # Context attention ratio over full seq_len (includes generated tokens)
                ctx_sum = step_attn_np[:, :, ctx_start_padded:ctx_end_padded].sum(axis=-1)
                total_sum = step_attn_np.sum(axis=-1) + 1e-10
                context_attention_ratios.append(float((ctx_sum / total_sum).mean()))

                # Accumulate input-portion attention (running sum → divide at end)
                input_attn = step_attn_np[:, :, :padded_input_len]
                if mean_input_attn is None:
                    mean_input_attn = input_attn.copy()
                else:
                    mean_input_attn += input_attn

            mean_input_attn = mean_input_attn / n_valid  # (n_layers, n_heads, padded_input_len)

            # --- Layer-18 hidden state: genuine mean across all generated tokens ---
            # With KV cache, hidden_states[step][layer] has shape (batch, 1, hidden_size).
            # Use [b, -1, :] to extract the single generated token's hidden state at each step.
            mid_layer_hs = np.stack([
                outputs.hidden_states[step_idx][18][b, -1, :].cpu().float().numpy()
                for step_idx in range(n_valid)
            ], axis=0).mean(axis=0)  # (2048,)

            uncertainty_features = _compute_uncertainty(token_probs, top2_probs, token_entropies)

            results.append({
                "generated_answer": generated_answer,
                "answer_n_tokens": n_valid,
                "tokens": tokens,
                "token_log_probs": token_log_probs,
                "top100_logit_values": top100_logit_values,
                "top100_logit_token_ids": top100_logit_token_ids,
                "token_entropies": token_entropies,
                "context_attention_ratios": context_attention_ratios,   # list[float], len=n_valid
                "mean_input_attention": mean_input_attn,                # np.ndarray (n_layers, 16, padded_input_len)
                "middle_layer_hidden_state": mid_layer_hs,              # np.ndarray (2048,)
                "context_start_idx": context_start_idx,
                "context_end_idx": context_end_idx,
                "padding_offset": padding_offset,
                "input_len": input_len,
                "uncertainty_features": uncertainty_features,
            })

        return results


def _compute_uncertainty(
    token_probs: list[float],
    top2_probs: list[float],
    token_entropies: list[float],
) -> dict[str, float]:
    """Compute aggregate uncertainty features from per-token probabilities."""
    if not token_probs:
        return {
            "mean_token_prob": 0.0,
            "min_token_prob": 0.0,
            "std_token_prob": 0.0,
            "max_prob_gap": 0.0,
            "mean_entropy": 0.0,
        }

    n = len(token_probs)
    mean_p = sum(token_probs) / n
    min_p = min(token_probs)

    variance = sum((p - mean_p) ** 2 for p in token_probs) / n
    std_p = math.sqrt(variance)

    # max_prob_gap: mean margin between top-1 and top-2 probability across tokens.
    # With greedy decoding selected == top-1, so this measures decisiveness.
    # Previously this was (top1 - selected) which is always 0 under greedy — now fixed.
    prob_gaps = [s - t2 for s, t2 in zip(token_probs, top2_probs)]
    max_prob_gap = sum(prob_gaps) / n

    mean_entropy = sum(token_entropies) / len(token_entropies) if token_entropies else 0.0

    return {
        "mean_token_prob": round(mean_p, 6),
        "min_token_prob": round(min_p, 6),
        "std_token_prob": round(std_p, 6),
        "max_prob_gap": round(max_prob_gap, 6),
        "mean_entropy": round(mean_entropy, 6),
    }


def verify_output(record: dict) -> bool:
    """
    Sanity-check a generate() return dict.
    Prints a warning for each failed check but does not raise.
    Returns True if all checks pass, False otherwise.
    """
    ok = True

    # token_log_probs: non-empty list of non-positive floats
    tlp = record.get("token_log_probs", [])
    if not tlp:
        print("[verify_output] WARNING: token_log_probs is empty")
        ok = False
    elif any(v > 0 for v in tlp):
        print(f"[verify_output] WARNING: token_log_probs has positive value(s) (max={max(tlp):.8f})")
        ok = False

    # mean_input_attention: shape (n_layers, 16, padded_input_len) where all dims > 0
    attn = record.get("mean_input_attention")
    if attn is None:
        print("[verify_output] WARNING: mean_input_attention is None")
        ok = False
    elif not isinstance(attn, np.ndarray) or attn.ndim != 3 or attn.shape[1] != 16 or attn.shape[2] == 0:
        print(
            f"[verify_output] WARNING: mean_input_attention has unexpected shape "
            f"{getattr(attn, 'shape', type(attn))}, expected (n_layers, 16, padded_input_len>0)"
        )
        ok = False

    # middle_layer_hidden_state: shape (2048,)
    hs = record.get("middle_layer_hidden_state")
    if hs is None:
        print("[verify_output] WARNING: middle_layer_hidden_state is None")
        ok = False
    elif not isinstance(hs, np.ndarray) or hs.shape != (2048,):
        print(f"[verify_output] WARNING: middle_layer_hidden_state shape {getattr(hs, 'shape', type(hs))}, expected (2048,)")
        ok = False

    # top100_logit_values: exactly 100 entries per token
    t100 = record.get("top100_logit_values", [])
    if not t100:
        print("[verify_output] WARNING: top100_logit_values is empty")
        ok = False
    elif any(len(v) != 100 for v in t100):
        bad = [i for i, v in enumerate(t100) if len(v) != 100]
        print(f"[verify_output] WARNING: top100_logit_values has != 100 entries at token indices {bad[:5]}")
        ok = False

    # entropy values: all non-negative
    ents = record.get("token_entropies", [])
    if not ents:
        print("[verify_output] WARNING: token_entropies is empty")
        ok = False
    elif any(e < 0 for e in ents):
        print("[verify_output] WARNING: some token entropy is negative")
        ok = False

    # context_attention_ratios: same length as tokens, all in [0, 1]
    car = record.get("context_attention_ratios", [])
    n_tokens = record.get("answer_n_tokens", len(record.get("tokens", [])))
    if not car:
        print("[verify_output] WARNING: context_attention_ratios is empty")
        ok = False
    elif len(car) != n_tokens:
        print(f"[verify_output] WARNING: context_attention_ratios length {len(car)} != answer_n_tokens {n_tokens}")
        ok = False
    elif any(r < 0 or r > 1 for r in car):
        print("[verify_output] WARNING: some context_attention_ratio is outside [0, 1]")
        ok = False

    # max_prob_gap: should be >= 0 (top1 - top2 is always non-negative)
    gap = record.get("uncertainty_features", {}).get("max_prob_gap", -1)
    if gap < 0:
        print(f"[verify_output] WARNING: max_prob_gap={gap:.4f} is negative (should be >= 0)")
        ok = False

    # context index constraint: context_start_idx < context_end_idx < input_len
    csi = record.get("context_start_idx", 0)
    cei = record.get("context_end_idx", 0)
    il = record.get("input_len", 0)
    if not (csi < cei < il):
        print(
            f"[verify_output] WARNING: context index constraint violated: "
            f"context_start_idx={csi} < context_end_idx={cei} < input_len={il}"
        )
        ok = False

    if ok:
        print("[verify_output] All checks passed.")
    return ok
