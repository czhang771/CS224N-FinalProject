"""Qwen2.5-3B-Instruct generation with token-level probability signals."""

import math
from typing import Any

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
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        self.model.eval()

    def generate(self, question: str, context_text: str) -> dict[str, Any]:
        """
        Generate an answer and extract token-level probability signals.

        Returns a dict with:
            generated_answer, tokens, token_log_probs, uncertainty_features
        """
        messages = build_prompt(question, context_text)
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=0.8 if self.do_sample else 1.0,  # 1.0 = no nucleus filter for greedy
            top_k=20 if self.do_sample else 0,    # 0 = no top-k for greedy
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Generated token ids (excluding the prompt)
        generated_ids = outputs.sequences[0, input_len:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids.tolist())
        generated_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Per-token probabilities from scores
        # outputs.scores is a tuple of length n_generated, each (1, vocab_size)
        token_log_probs: list[float] = []
        token_probs: list[float] = []
        top1_probs: list[float] = []

        for step_idx, logits in enumerate(outputs.scores):
            probs = F.softmax(logits[0], dim=-1)  # (vocab_size,)
            token_id = generated_ids[step_idx].item()
            selected_prob = probs[token_id].item()
            top1_prob = probs.max().item()

            token_probs.append(selected_prob)
            token_log_probs.append(math.log(selected_prob + 1e-10))
            top1_probs.append(top1_prob)

        uncertainty_features = _compute_uncertainty(token_probs, top1_probs)

        return {
            "generated_answer": generated_answer,
            "tokens": tokens,
            "token_log_probs": token_log_probs,
            "uncertainty_features": uncertainty_features,
        }


def _compute_uncertainty(
    token_probs: list[float], top1_probs: list[float]
) -> dict[str, float]:
    """Compute aggregate uncertainty features from per-token probabilities."""
    if not token_probs:
        return {
            "mean_token_prob": 0.0,
            "min_token_prob": 0.0,
            "std_token_prob": 0.0,
            "max_prob_gap": 0.0,
        }

    n = len(token_probs)
    mean_p = sum(token_probs) / n
    min_p = min(token_probs)

    variance = sum((p - mean_p) ** 2 for p in token_probs) / n
    std_p = math.sqrt(variance)

    prob_gaps = [t - s for t, s in zip(top1_probs, token_probs)]
    max_prob_gap = sum(prob_gaps) / n

    return {
        "mean_token_prob": round(mean_p, 6),
        "min_token_prob": round(min_p, 6),
        "std_token_prob": round(std_p, 6),
        "max_prob_gap": round(max_prob_gap, 6),
    }
