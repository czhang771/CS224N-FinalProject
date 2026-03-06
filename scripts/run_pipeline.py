#!/usr/bin/env python3
"""End-to-end pipeline: PubMedQA → Qwen generation → Gemini judge → JSONL output."""

import argparse
import os
import sys

from tqdm import tqdm

# Allow running from project root without installing as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.load_pubmedqa import load_pubmedqa
from src.generation.qwen_generator import QwenGenerator
from src.judge.gemini_judge import GeminiJudge
from src.utils.io import append_jsonl, load_completed_pubids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hallucination detection pipeline on PubMedQA"
    )
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of PubMedQA examples to process"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for dataset shuffle"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/outputs", help="Directory for JSONL output"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="pipeline_output.jsonl",
        help="Output JSONL filename",
    )
    parser.add_argument(
        "--skip_judge",
        action="store_true",
        help="Skip the Gemini judge step (generation only)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max tokens to generate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Qwen model (use Qwen/Qwen2.5-0.5B-Instruct for faster CPU testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = os.path.join(args.output_dir, args.output_filename)

    # --- Load data ---
    print(f"Loading {args.n_samples} PubMedQA examples (seed={args.seed})...")
    examples = load_pubmedqa(n_samples=args.n_samples, seed=args.seed)
    print(f"  Loaded {len(examples)} examples.")

    # --- Checkpoint: skip already-completed pubids ---
    completed = load_completed_pubids(output_path)
    if completed:
        print(f"  Resuming: {len(completed)} already completed, skipping.")
    remaining = [ex for ex in examples if str(ex["pubid"]) not in completed]
    print(f"  {len(remaining)} examples to process.")

    if not remaining:
        print("All examples already processed. Exiting.")
        return

    # --- Load Qwen generator ---
    print(f"Loading Qwen generator: {args.model}")
    generator = QwenGenerator(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )

    # --- Load Gemini judge ---
    judge: GeminiJudge | None = None
    if not args.skip_judge:
        print("Initializing Gemini judge...")
        judge = GeminiJudge()

    # --- Run pipeline ---
    print(f"\nRunning pipeline on {len(remaining)} examples...")
    for example in tqdm(remaining, desc="Pipeline"):
        record = dict(example)  # copy base fields

        # Generation
        try:
            gen_result = generator.generate(
                question=example["question"],
                context_text=example["context_text"],
            )
            record.update(gen_result)
        except Exception as e:
            print(f"\n  [Generator] Error on pubid={example['pubid']}: {e}")
            record.update(
                {
                    "generated_answer": "",
                    "tokens": [],
                    "token_log_probs": [],
                    "uncertainty_features": {},
                }
            )

        # Judge
        if judge is not None and record.get("generated_answer"):
            try:
                judge_result = judge.judge(
                    question=example["question"],
                    context_text=example["context_text"],
                    generated_answer=record["generated_answer"],
                )
                record.update(judge_result)
            except Exception as e:
                print(f"\n  [Judge] Error on pubid={example['pubid']}: {e}")
                record.update(
                    {
                        "judge_label": "error",
                        "judge_confidence": 0.0,
                        "judge_reasoning": str(e),
                    }
                )
        elif not args.skip_judge:
            record.update(
                {
                    "judge_label": "skipped_empty_answer",
                    "judge_confidence": 0.0,
                    "judge_reasoning": "",
                }
            )

        append_jsonl(record, output_path)

    print(f"\nDone. Output written to: {output_path}")


if __name__ == "__main__":
    main()
