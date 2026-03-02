"""Gemini 2.5 Flash LLM judge for context faithfulness."""

import json
import os
import time
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

_JUDGE_PROMPT_TEMPLATE = """\
You are a medical fact-checking expert. Your task is to judge whether a generated answer \
contains claims that are NOT supported by the provided context passages.

Context:
{context_text}

Question:
{question}

Generated Answer:
{generated_answer}

Instructions:
- Judge ONLY context faithfulness: does the answer make claims not supported by the context?
- Do NOT penalize for incomplete answers; only flag unsupported claims.
- Respond with valid JSON only, no additional text.

Respond with this exact JSON structure:
{{
  "hallucination_label": "hallucinated" or "not_hallucinated",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}}
"""


class GeminiJudge:
    def __init__(self, model_name: str = "gemini-2.5-flash", max_retries: int = 3):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment or .env file")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max_retries

    def judge(
        self, question: str, context_text: str, generated_answer: str
    ) -> dict[str, Any]:
        """
        Judge whether generated_answer is faithful to context_text.

        Returns a dict with: judge_label, judge_confidence, judge_reasoning
        """
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            context_text=context_text,
            question=question,
            generated_answer=generated_answer,
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                result = _parse_judge_response(response.text)
                return result
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                print(f"  [GeminiJudge] Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        # All retries exhausted — return a sentinel
        print(f"  [GeminiJudge] All retries failed: {last_error}")
        return {
            "judge_label": "error",
            "judge_confidence": 0.0,
            "judge_reasoning": f"Judge failed after {self.max_retries} retries: {last_error}",
        }


def _parse_judge_response(text: str) -> dict[str, Any]:
    """Parse and validate the JSON response from Gemini."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(text)

    label = data.get("hallucination_label", "").lower()
    if label not in ("hallucinated", "not_hallucinated"):
        raise ValueError(f"Unexpected hallucination_label: {label!r}")

    confidence = float(data.get("confidence", 0.0))
    reasoning = str(data.get("reasoning", ""))

    return {
        "judge_label": label,
        "judge_confidence": round(confidence, 4),
        "judge_reasoning": reasoning,
    }
