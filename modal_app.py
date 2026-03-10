"""
Modal app for the hallucination detection pipeline.

Usage:
    # Store your Vertex service account JSON as a Modal secret (one-time setup):
    modal secret create gcp-vertex-secret SERVICE_ACCOUNT_JSON="$(cat ~/Downloads/YOUR_SERVICE_ACCOUNT_KEY.json)"

    # Generation only (no judge):
    modal run modal_app.py --n-samples 8

    # Judge pass (run after generation):
    modal run modal_app.py::judge

    # Skip judge during generation (same as default now):
    modal run modal_app.py --n-samples 8

    # Download output after run:
    modal volume get pipeline-outputs pipeline_output.jsonl ./data/outputs/
"""

import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0,<5.0.0",
        "accelerate>=0.27.0",
        "datasets>=2.18.0",
        "google-genai>=1.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    )
    .add_local_dir("src", remote_path="/root/src")
)

# ---------------------------------------------------------------------------
# Persistent volume
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("pipeline-outputs", create_if_missing=True)
VOLUME_PATH = "/outputs"
OUTPUT_FILENAME = "pipeline_output.jsonl"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("hallucination-pipeline", image=image)


# ---------------------------------------------------------------------------
# Generation function (no judge)
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=14400,
    secrets=[modal.Secret.from_name("gcp-vertex-secret")],
    volumes={VOLUME_PATH: volume},
)
def run_pipeline(
    n_samples: int = 100,
    seed: int = 42,
    max_new_tokens: int = 512,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    batch_size: int = 4,
):
    import sys
    import numpy as np
    import torch
    from pathlib import Path

    sys.path.insert(0, "/root")

    from tqdm import tqdm

    from src.data.load_pubmedqa import load_pubmedqa
    from src.generation.qwen_generator import QwenGenerator
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

    # --- Run pipeline in batches ---
    print(f"\nRunning generation on {len(remaining)} examples (batch_size={batch_size})...")

    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

    for batch in tqdm(batches, desc="Batches"):
        pairs = [(ex["question"], ex["context_text"]) for ex in batch]

        try:
            gen_results = generator.generate_batch(pairs)
        except Exception as e:
            print(f"\n  [Generator] Batch error: {e}. Falling back to per-example.")
            gen_results = []
            for ex in batch:
                try:
                    gen_results.append(generator.generate(ex["question"], ex["context_text"]))
                except Exception as e2:
                    print(f"\n  [Generator] Error on pubid={ex['pubid']}: {e2}")
                    gen_results.append(_empty_gen_result())

        for ex, gen_result in zip(batch, gen_results):
            record = dict(ex)

            # Serialize numpy arrays
            if isinstance(gen_result.get("final_token_attention"), np.ndarray):
                gen_result["final_token_attention"] = gen_result["final_token_attention"].tolist()
            if isinstance(gen_result.get("middle_layer_hidden_state"), np.ndarray):
                gen_result["middle_layer_hidden_state"] = gen_result["middle_layer_hidden_state"].tolist()

            record.update(gen_result)

            # Judge fields absent during generation pass
            record["judge_label"] = None
            record["judge_confidence"] = None
            record["judge_reasoning"] = None

            append_jsonl(record, output_path)
            volume.commit()

        torch.cuda.empty_cache()

    print(f"\nDone. Output at: {output_path}")
    return output_path


def _empty_gen_result() -> dict:
    return {
        "generated_answer": "",
        "tokens": [],
        "token_log_probs": [],
        "top100_logit_values": [],
        "top100_logit_token_ids": [],
        "token_entropies": [],
        "final_token_attention": [],
        "middle_layer_hidden_state": [],
        "context_start_idx": 0,
        "context_end_idx": 0,
        "input_len": 0,
        "uncertainty_features": {},
    }


# ---------------------------------------------------------------------------
# Judge function (separate pass)
# ---------------------------------------------------------------------------
@app.function(
    timeout=7200,
    secrets=[modal.Secret.from_name("gcp-vertex-secret")],
    volumes={VOLUME_PATH: volume},
)
def run_judge():
    import os
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")

    from src.judge.gemini_judge import GeminiJudge
    from src.utils.io import append_jsonl

    creds_path = Path("/tmp/gcp-vertex-secret.json")
    creds_path.write_text(os.environ["SERVICE_ACCOUNT_JSON"])

    project_id = json.loads(os.environ["SERVICE_ACCOUNT_JSON"]).get("project_id")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

    output_path = f"{VOLUME_PATH}/{OUTPUT_FILENAME}"

    volume.reload()
    records = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    to_judge = [r for r in records if r.get("judge_label") is None and r.get("generated_answer")]
    print(f"Found {len(records)} records, {len(to_judge)} need judging.")

    if not to_judge:
        print("Nothing to judge.")
        return

    judge = GeminiJudge()

    for r in to_judge:
        try:
            result = judge.judge(
                question=r["question"],
                context_text=r["context_text"],
                generated_answer=r["generated_answer"],
            )
            r.update(result)
        except Exception as e:
            print(f"  [Judge] Error on pubid={r.get('pubid')}: {e}")
            r["judge_label"] = "error"
            r["judge_confidence"] = 0.0
            r["judge_reasoning"] = str(e)

    # Rewrite the full file with updated records
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    import shutil
    shutil.move(tmp_path, output_path)
    volume.commit()

    print(f"Done. Judged {len(to_judge)} records.")


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    n_samples: int = 100,
    seed: int = 42,
    max_new_tokens: int = 512,
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    batch_size: int = 4,
):
    print(f"Launching generation on Modal (n_samples={n_samples}, seed={seed}, batch_size={batch_size}, gpu=A10G)...")
    output_path = run_pipeline.remote(
        n_samples=n_samples,
        seed=seed,
        max_new_tokens=max_new_tokens,
        model_name=model,
        batch_size=batch_size,
    )
    print(f"\nGeneration complete. Output saved to Modal volume at: {output_path}")
    print("To run judge pass:")
    print("  modal run modal_app.py::judge")
    print("To download:")
    print(f"  modal volume get pipeline-outputs {OUTPUT_FILENAME} ./data/outputs/")


@app.local_entrypoint()
def judge():
    print("Launching judge pass on Modal...")
    run_judge.remote()
    print("Judge pass complete.")
    print("To download:")
    print(f"  modal volume get pipeline-outputs {OUTPUT_FILENAME} ./data/outputs/")
