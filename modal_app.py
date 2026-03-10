"""
Modal app for the hallucination detection pipeline.

Usage:
    # Store your Vertex service account JSON as a Modal secret (one-time setup):
    modal secret create gcp-vertex-secret SERVICE_ACCOUNT_JSON="$(cat ~/Downloads/YOUR_SERVICE_ACCOUNT_KEY.json)"

    # Small test (run first):
    modal run modal_app.py --n-samples 5

    # Medium test (run second):
    modal run modal_app.py --n-samples 50

    # Full run (only when both above look correct):
    modal run modal_app.py --n-samples 1000

    # Skip judge for generation-only testing:
    modal run modal_app.py --n-samples 5 --skip-judge

    # Download output after run:
    modal volume get pipeline-outputs pipeline_output.jsonl ./data/outputs/
"""

import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# Use a CUDA base so transformers can use the GPU. Install deps explicitly
# so Modal can cache the layer — avoid `pip_install_from_requirements` because
# that busts the cache on every requirements.txt touch.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        # Core ML — on Modal we always have CUDA, so no upper bound needed
        "torch>=2.1.0",
        "transformers>=4.40.0,<5.0.0",
        "accelerate>=0.27.0",
        # Data
        "datasets>=2.18.0",
        # Judge
        "google-genai>=1.0.0",
        # Utils
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    )
    # Copy local src/ into the image (replaces Mount in Modal 1.x)
    .add_local_dir("src", remote_path="/root/src")
)

# ---------------------------------------------------------------------------
# Persistent volume — stores the JSONL output across runs
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("pipeline-outputs", create_if_missing=True)
VOLUME_PATH = "/outputs"
OUTPUT_FILENAME = "pipeline_output.jsonl"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("hallucination-pipeline", image=image)


@app.function(
    gpu="A10G",
    timeout=14400,  # 4 hours — covers 1,000 samples with attention/hidden-state logging
    secrets=[modal.Secret.from_name("gcp-vertex-secret")],
    volumes={VOLUME_PATH: volume},
)
def run_pipeline(
    n_samples: int = 100,
    seed: int = 42,
    max_new_tokens: int = 512,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    skip_judge: bool = False,
):
    import os
    import sys
    import numpy as np
    import torch
    from pathlib import Path

    sys.path.insert(0, "/root")

    creds_path = Path("/tmp/gcp-vertex-secret.json")
    creds_path.write_text(os.environ["SERVICE_ACCOUNT_JSON"])

    import json as _json
    project_id = _json.loads(os.environ["SERVICE_ACCOUNT_JSON"]).get("project_id")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

    from tqdm import tqdm

    from src.data.load_pubmedqa import load_pubmedqa
    from src.generation.qwen_generator import QwenGenerator
    from src.judge.gemini_judge import GeminiJudge
    from src.utils.io import append_jsonl, load_completed_pubids

    output_path = f"{VOLUME_PATH}/{OUTPUT_FILENAME}"

    # --- Load data ---
    print(f"Loading {n_samples} PubMedQA examples (seed={seed})...")
    examples = load_pubmedqa(n_samples=n_samples, seed=seed)
    print(f"  Loaded {len(examples)} examples.")

    # --- Checkpoint: skip already-completed pubids ---
    completed = load_completed_pubids(output_path)
    if completed:
        print(f"  Resuming: {len(completed)} already completed, skipping.")
    remaining = [ex for ex in examples if str(ex["pubid"]) not in completed]
    print(f"  {len(remaining)} examples to process.")

    if not remaining:
        print("All examples already processed.")
        return output_path

    # --- Load Qwen generator ---
    print(f"Loading Qwen generator: {model_name}")
    generator = QwenGenerator(model_name=model_name, max_new_tokens=max_new_tokens)

    # --- Load Gemini judge ---
    judge = None
    if not skip_judge:
        print("Initializing Gemini judge...")
        judge = GeminiJudge()

    # --- Run pipeline ---
    print(f"\nRunning pipeline on {len(remaining)} examples...")
    for example in tqdm(remaining, desc="Pipeline"):
        record = dict(example)

        # Generation
        try:
            gen_result = generator.generate(
                question=example["question"],
                context_text=example["context_text"],
            )
            # Convert numpy arrays to lists for JSON serialization before storing
            if isinstance(gen_result.get("attention_last_rows"), list):
                gen_result["attention_last_rows"] = [
                    a.tolist() if isinstance(a, np.ndarray) else a
                    for a in gen_result["attention_last_rows"]
                ]
            if isinstance(gen_result.get("mean_pooled_hidden_states"), np.ndarray):
                gen_result["mean_pooled_hidden_states"] = gen_result["mean_pooled_hidden_states"].tolist()
            record.update(gen_result)
        except Exception as e:
            print(f"\n  [Generator] Error on pubid={example['pubid']}: {e}")
            record.update({
                "generated_answer": "",
                "tokens": [],
                "token_log_probs": [],
                "top100_logit_values": [],
                "top100_logit_token_ids": [],
                "token_entropies": [],
                "attention_last_rows": [],
                "mean_pooled_hidden_states": [],
                "context_start_idx": 0,
                "context_end_idx": 0,
                "input_len": 0,
                "uncertainty_features": {},
            })

        # Free GPU memory after each example
        torch.cuda.empty_cache()

        # Judge
        if judge is not None and record.get("generated_answer"):
            try:
                record.update(
                    judge.judge(
                        question=example["question"],
                        context_text=example["context_text"],
                        generated_answer=record["generated_answer"],
                    )
                )
            except Exception as e:
                print(f"\n  [Judge] Error on pubid={example['pubid']}: {e}")
                record.update({"judge_label": "error", "judge_confidence": 0.0, "judge_reasoning": str(e)})
        elif not skip_judge:
            record.update({"judge_label": "skipped_empty_answer", "judge_confidence": 0.0, "judge_reasoning": ""})

        append_jsonl(record, output_path)
        volume.commit()  # flush write to volume after each record

    print(f"\nDone. Output at: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Local entrypoint — called when you run `modal run modal_app.py`
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    n_samples: int = 100,
    seed: int = 42,
    max_new_tokens: int = 512,
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    skip_judge: bool = False,
):
    print(f"Launching pipeline on Modal (n_samples={n_samples}, seed={seed}, gpu=A10G)...")
    output_path = run_pipeline.remote(
        n_samples=n_samples,
        seed=seed,
        max_new_tokens=max_new_tokens,
        model_name=model,
        skip_judge=skip_judge,
    )
    print(f"\nPipeline complete. Output saved to Modal volume at: {output_path}")
    print("To download:")
    print(f"  modal volume get pipeline-outputs {OUTPUT_FILENAME} ./data/outputs/")
