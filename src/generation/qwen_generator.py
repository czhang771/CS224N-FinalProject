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
            generated_answer, tokens, token_log_probs, top100_logit_values,
            top100_logit_token_ids, token_entropies, final_token_attention,
            middle_layer_hidden_state, context_start_idx, context_end_idx,
            input_len, uncertainty_features
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

        # --- Context token indices (per example) ---
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

        # outputs.sequences: (batch, prompt_len_padded + n_generated)
        # The padded prompt length is inputs["input_ids"].shape[1]
        padded_input_len = inputs["input_ids"].shape[1]

        # --- Final token attention: last generation step, all layers ---
        # outputs.attentions[-1]: tuple of 36 tensors, each (batch, n_heads, seq_len, seq_len)
        # We take the last row ([-1] in seq dim) → (batch, n_heads, seq_len)
        # Stack across layers → (36, batch, n_heads, seq_len), then split per example
        final_step_attns = outputs.attentions[-1]  # tuple of 36
        # Stack: (36, batch, 2, seq_len)
        final_attn_stacked = np.stack(
            [layer.cpu().float().numpy() for layer in final_step_attns], axis=0
        )  # (36, batch, 2, seq_len, seq_len)
        # Last row of each attention matrix → (36, batch, 2, seq_len)
        final_attn_last_row = final_attn_stacked[:, :, :, -1, :]  # (36, batch, 2, seq_len)

        # --- Middle layer hidden state: layer 18, last generation step ---
        # outputs.hidden_states[-1]: tuple of 37 tensors, each (batch, seq_len, 2048)
        # Take layer index 18, mean-pool over seq_len → (batch, 2048)
        final_step_hs = outputs.hidden_states[-1]  # tuple of 37
        middle_hs = final_step_hs[18]  # (batch, seq_len, 2048)
        middle_hs_pooled = middle_hs.mean(dim=1).cpu().float().numpy()  # (batch, 2048)

        # --- Per-token signals from scores ---
        # outputs.scores: tuple of n_generated, each (batch, vocab_size)
        # We need to demux per example. Since all examples in the batch may have
        # generated different numbers of tokens (padding stops at EOS), we track
        # generated ids per example to know valid steps.
        generated_ids_batch = outputs.sequences[:, padded_input_len:]  # (batch, n_gen)

        # Build per-example lists of scores up to their EOS
        # For simplicity we process scores for all steps and mask out post-EOS
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

            token_log_probs: list[float] = []
            token_probs: list[float] = []
            top1_probs: list[float] = []
            token_entropies: list[float] = []
            top100_logit_values: list[list[float]] = []
            top100_logit_token_ids: list[list[int]] = []

            for step_idx in range(n_valid):
                logits_1d = outputs.scores[step_idx][b]  # (vocab_size,)
                probs = F.softmax(logits_1d, dim=-1)
                token_id = valid_ids[step_idx].item()
                selected_prob = probs[token_id].item()
                top1_prob = probs.max().item()

                token_probs.append(selected_prob)
                token_log_probs.append(math.log(selected_prob + 1e-10))
                top1_probs.append(top1_prob)

                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                token_entropies.append(entropy)

                top100 = torch.topk(logits_1d, k=100)
                top100_logit_values.append(top100.values.cpu().float().tolist())
                top100_logit_token_ids.append(top100.indices.cpu().tolist())

            # final_token_attention for this example: (36, 2, seq_len)
            final_token_attn = final_attn_last_row[:, b, :, :]  # (36, 2, seq_len)

            # middle_layer_hidden_state for this example: (2048,)
            middle_hs_example = middle_hs_pooled[b]  # (2048,)

            context_start_idx, context_end_idx = context_indices[b]
            input_len = int(input_lens[b])

            uncertainty_features = _compute_uncertainty(token_probs, top1_probs, token_entropies)

            results.append({
                "generated_answer": generated_answer,
                "tokens": tokens,
                "token_log_probs": token_log_probs,
                "top100_logit_values": top100_logit_values,
                "top100_logit_token_ids": top100_logit_token_ids,
                "token_entropies": token_entropies,
                "final_token_attention": final_token_attn,          # np.ndarray (36, 2, seq_len)
                "middle_layer_hidden_state": middle_hs_example,     # np.ndarray (2048,)
                "context_start_idx": context_start_idx,
                "context_end_idx": context_end_idx,
                "input_len": input_len,
                "uncertainty_features": uncertainty_features,
            })

        return results


def _compute_uncertainty(
    token_probs: list[float],
    top1_probs: list[float],
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

    prob_gaps = [t - s for t, s in zip(top1_probs, token_probs)]
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

    # token_log_probs: non-empty list of negative floats
    tlp = record.get("token_log_probs", [])
    if not tlp:
        print("[verify_output] WARNING: token_log_probs is empty")
        ok = False
    elif any(v >= 0 for v in tlp):
        print(f"[verify_output] WARNING: token_log_probs has non-negative value(s) (max={max(tlp):.4f})")
        ok = False

    # final_token_attention: shape (36, 2, seq_len) where seq_len > 0
    attn = record.get("final_token_attention")
    if attn is None:
        print("[verify_output] WARNING: final_token_attention is None")
        ok = False
    elif not isinstance(attn, np.ndarray) or attn.ndim != 3 or attn.shape[0] != 36 or attn.shape[1] != 2 or attn.shape[2] == 0:
        print(
            f"[verify_output] WARNING: final_token_attention has unexpected shape "
            f"{getattr(attn, 'shape', type(attn))}, expected (36, 2, seq_len>0)"
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
