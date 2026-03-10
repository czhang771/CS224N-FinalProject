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

        # CUDA: use auto + bfloat16. Mac/CPU: use CPU + float32 (avoids MPS bfloat16 + disk offload issues)
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

    def generate(self, question: str, context_text: str) -> dict[str, Any]:
        """
        Generate an answer and extract token-level probability signals.

        Returns a dict with:
            generated_answer, tokens, token_log_probs, top100_logit_values,
            top100_logit_token_ids, token_entropies, attention_matrices,
            mean_pooled_hidden_states, context_start_idx, context_end_idx,
            input_len, uncertainty_features
        """
        messages = build_prompt(question, context_text)
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        # --- Context token indices ---
        # Locate the context_text character span within input_text, then
        # count tokens for the prefix up to each boundary.
        # add_special_tokens=False because apply_chat_template embeds special
        # tokens (e.g. <|im_start|>) directly in the text string.
        ctx_marker = "Context:\n"
        ctx_marker_pos = input_text.find(ctx_marker)
        if ctx_marker_pos == -1:
            context_start_idx = 0
            context_end_idx = 0
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

        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=0.8 if self.do_sample else 1.0,  # 1.0 = no nucleus filter for greedy
            top_k=20 if self.do_sample else 0,      # 0 = no top-k for greedy
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

        # Generated token ids (excluding the prompt)
        generated_ids = outputs.sequences[0, input_len:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids.tolist())
        generated_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # --- Per-token signals from scores ---
        # outputs.scores: tuple of n_generated, each (1, vocab_size) — raw logits
        token_log_probs: list[float] = []
        token_probs: list[float] = []
        top1_probs: list[float] = []
        token_entropies: list[float] = []
        top100_logit_values: list[list[float]] = []
        top100_logit_token_ids: list[list[int]] = []

        for step_idx, logits in enumerate(outputs.scores):
            logits_1d = logits[0]  # (vocab_size,)
            probs = F.softmax(logits_1d, dim=-1)
            token_id = generated_ids[step_idx].item()
            selected_prob = probs[token_id].item()
            top1_prob = probs.max().item()

            token_probs.append(selected_prob)
            token_log_probs.append(math.log(selected_prob + 1e-10))
            top1_probs.append(top1_prob)

            # Entropy: H = -sum(p * log(p + eps))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            token_entropies.append(entropy)

            # Top-100 logits (stored before softmax for flexibility)
            top100 = torch.topk(logits_1d, k=100)
            top100_logit_values.append(top100.values.cpu().float().tolist())
            top100_logit_token_ids.append(top100.indices.cpu().tolist())

        # --- Attention last rows ---
        # outputs.attentions: tuple of n_gen_tokens
        #   each: tuple of num_hidden_layers (36) tensors
        #   each tensor: (batch=1, num_kv_heads=2, seq_len, seq_len)
        # We store only the last row (current token's attention over all positions):
        #   layer[0, :, -1, :] → (2, seq_len), stacked across layers → (36, 2, seq_len)
        # seq_len grows by 1 each step, so this stays a list of arrays.
        attention_last_rows: list[np.ndarray] = []
        if outputs.attentions is not None:
            for step_attns in outputs.attentions:
                # step_attns: tuple of 36 layer tensors, each (1, 2, seq_len, seq_len)
                stacked = np.stack(
                    [layer[0, :, -1, :].cpu().float().numpy() for layer in step_attns],
                    axis=0,
                )  # (36, 2, seq_len)
                attention_last_rows.append(stacked)

        # --- Hidden states ---
        # outputs.hidden_states: tuple of n_gen_tokens
        #   each: tuple of num_hidden_layers+1 (37) tensors, incl. embedding layer
        #   each tensor: (batch=1, seq_len, hidden_size=2048)
        # Mean-pool over seq_len per layer, then average across generated tokens.
        # Final shape: (37, 2048)
        mean_pooled_hidden_states: np.ndarray | None = None
        if outputs.hidden_states is not None:
            pooled_per_token: list[np.ndarray] = []
            for step_hs in outputs.hidden_states:
                # step_hs: tuple of 37 tensors, each (1, seq_len, 2048)
                step_pooled = np.stack(
                    [layer[0].mean(dim=0).cpu().float().numpy() for layer in step_hs],
                    axis=0,
                )  # (37, 2048)
                pooled_per_token.append(step_pooled)
            mean_pooled_hidden_states = np.stack(pooled_per_token, axis=0).mean(axis=0)  # (37, 2048)

        uncertainty_features = _compute_uncertainty(token_probs, top1_probs, token_entropies)

        return {
            "generated_answer": generated_answer,
            "tokens": tokens,
            "token_log_probs": token_log_probs,
            "top100_logit_values": top100_logit_values,
            "top100_logit_token_ids": top100_logit_token_ids,
            "token_entropies": token_entropies,
            "attention_last_rows": attention_last_rows,        # list of np.ndarray, each (36, 2, seq_len)
            "mean_pooled_hidden_states": mean_pooled_hidden_states,  # np.ndarray (37, 2048)
            "context_start_idx": context_start_idx,
            "context_end_idx": context_end_idx,
            "input_len": input_len,
            "uncertainty_features": uncertainty_features,
        }


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

    # attention_last_rows: list of (36, 2, seq_len) arrays
    attn = record.get("attention_last_rows", [])
    if not attn:
        print("[verify_output] WARNING: attention_last_rows is empty")
        ok = False
    else:
        for i, a in enumerate(attn):
            if not isinstance(a, np.ndarray) or a.ndim != 3 or a.shape[0] != 36 or a.shape[1] != 2:
                print(
                    f"[verify_output] WARNING: attention_last_rows[{i}] has unexpected shape "
                    f"{getattr(a, 'shape', type(a))}, expected (36, 2, seq_len)"
                )
                ok = False
                break

    # mean_pooled_hidden_states: shape (37, 2048)
    hs = record.get("mean_pooled_hidden_states")
    if hs is None:
        print("[verify_output] WARNING: mean_pooled_hidden_states is None")
        ok = False
    elif not isinstance(hs, np.ndarray) or hs.shape != (37, 2048):
        print(f"[verify_output] WARNING: mean_pooled_hidden_states shape {getattr(hs, 'shape', type(hs))}, expected (37, 2048)")
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
