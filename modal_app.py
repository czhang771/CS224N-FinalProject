"""
Modal app for the hallucination detection pipeline.

Usage:
    # --- Generation (each team member runs their training shard) ---
    modal run modal_app.py::main --shard 0   # examples 0–232   (~233 samples)
    modal run modal_app.py::main --shard 1   # examples 233–465 (~233 samples)
    modal run modal_app.py::main --shard 2   # examples 466–699 (~234 samples)

    # --- Val/test (one person runs both after training is done) ---
    modal run modal_app.py::eval_set --split val
    modal run modal_app.py::eval_set --split test

    # --- Judge pass (run after generation for each shard/split) ---
    modal run modal_app.py::judge --shard 0
    modal run modal_app.py::judge --split val

    # --- Merge all training shards + val + test into one file ---
    modal run modal_app.py::merge

    # --- Download ---
    modal volume get pipeline-outputs pipeline_output_train_shard0.jsonl ./data/outputs/
    modal volume get pipeline-outputs pipeline_output_val.jsonl ./data/outputs/
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
MERGED_FILENAME = "pipeline_output.jsonl"

# All output files that get merged
SHARD_FILENAMES = [
    "pipeline_output_train_shard0.jsonl",
    "pipeline_output_train_shard1.jsonl",
    "pipeline_output_train_shard2.jsonl",
    "pipeline_output_val.jsonl",
    "pipeline_output_test.jsonl",
]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("hallucination-pipeline", image=image)


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    timeout=14400,
    secrets=[modal.Secret.from_name("gcp-vertex-secret")],
    volumes={VOLUME_PATH: volume},
)
def run_pipeline(
    data_split: str = "train_shard_0",
    seed: int = 42,
    max_new_tokens: int = 512,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    batch_size: int = 4,
    max_samples: int = 0,  # 0 = no limit; set >0 for smoke tests
):
    import sys
    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/root")

    from src.data.load_pubmedqa import load_pubmedqa
    from src.generation.qwen_generator import QwenGenerator
    from src.utils.io import append_jsonl, load_completed_pubids

    # Determine output filename from split name
    if data_split in ("val", "test"):
        output_path = f"{VOLUME_PATH}/pipeline_output_{data_split}.jsonl"
    else:
        shard_idx = data_split.split("_")[-1]
        output_path = f"{VOLUME_PATH}/pipeline_output_train_shard{shard_idx}.jsonl"

    print(f"Loading split='{data_split}' (seed={seed})...")
    examples = load_pubmedqa(split=data_split, seed=seed)
    print(f"  {len(examples)} examples in this split.")

    # --- Checkpoint ---
    completed = load_completed_pubids(output_path)
    if completed:
        print(f"  Resuming: {len(completed)} already completed, skipping.")
    remaining = [ex for ex in examples if str(ex["pubid"]) not in completed]
    if max_samples > 0:
        remaining = remaining[:max_samples]
    print(f"  {len(remaining)} examples to process.")

    if not remaining:
        print("All examples already processed.")
        return output_path

    print(f"Loading Qwen generator: {model_name}")
    generator = QwenGenerator(model_name=model_name, max_new_tokens=max_new_tokens)

    print(f"\nRunning generation (batch_size={batch_size})...")
    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

    for batch in tqdm(batches, desc=data_split):
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

            if isinstance(gen_result.get("mean_input_attention"), np.ndarray):
                gen_result["mean_input_attention"] = gen_result["mean_input_attention"].tolist()
            if isinstance(gen_result.get("middle_layer_hidden_state"), np.ndarray):
                gen_result["middle_layer_hidden_state"] = gen_result["middle_layer_hidden_state"].tolist()

            record.update(gen_result)
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
        "answer_n_tokens": 0,
        "tokens": [],
        "token_log_probs": [],
        "top100_logit_values": [],
        "top100_logit_token_ids": [],
        "token_entropies": [],
        "context_attention_ratios": [],
        "mean_input_attention": [],
        "middle_layer_hidden_state": [],
        "context_start_idx": 0,
        "context_end_idx": 0,
        "padding_offset": 0,
        "input_len": 0,
        "uncertainty_features": {},
    }


# ---------------------------------------------------------------------------
# Judge function (per shard or split)
# ---------------------------------------------------------------------------
@app.function(
    timeout=7200,
    secrets=[modal.Secret.from_name("gcp-vertex-secret")],
    volumes={VOLUME_PATH: volume},
)
def run_judge(data_split: str = "train_shard_0"):
    import os
    import json
    import sys
    import shutil
    from pathlib import Path

    sys.path.insert(0, "/root")

    from src.judge.gemini_judge import GeminiJudge

    creds_path = Path("/tmp/gcp-vertex-secret.json")
    creds_path.write_text(os.environ["SERVICE_ACCOUNT_JSON"])
    project_id = json.loads(os.environ["SERVICE_ACCOUNT_JSON"]).get("project_id")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

    if data_split in ("val", "test"):
        output_path = f"{VOLUME_PATH}/pipeline_output_{data_split}.jsonl"
    else:
        shard_idx = data_split.split("_")[-1]
        output_path = f"{VOLUME_PATH}/pipeline_output_train_shard{shard_idx}.jsonl"

    volume.reload()
    records = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    to_judge = [r for r in records if r.get("judge_label") is None and r.get("generated_answer")]
    print(f"Split '{data_split}': {len(records)} records, {len(to_judge)} need judging.")

    if not to_judge:
        print("Nothing to judge.")
        return

    judge = GeminiJudge()
    for r in to_judge:
        try:
            r.update(judge.judge(
                question=r["question"],
                context_text=r["context_text"],
                generated_answer=r["generated_answer"],
            ))
        except Exception as e:
            print(f"  [Judge] Error on pubid={r.get('pubid')}: {e}")
            r["judge_label"] = "error"
            r["judge_confidence"] = 0.0
            r["judge_reasoning"] = str(e)

    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    shutil.move(tmp_path, output_path)
    volume.commit()
    print(f"Done. Judged {len(to_judge)} records.")


# ---------------------------------------------------------------------------
# Merge function
# ---------------------------------------------------------------------------
@app.function(
    timeout=600,
    volumes={VOLUME_PATH: volume},
)
def run_merge():
    import json
    import os

    volume.reload()
    merged_path = f"{VOLUME_PATH}/{MERGED_FILENAME}"
    total = 0

    with open(merged_path, "w") as out_f:
        for filename in SHARD_FILENAMES:
            path = f"{VOLUME_PATH}/{filename}"
            if not os.path.exists(path):
                print(f"  WARNING: {filename} not found, skipping.")
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out_f.write(line + "\n")
                        total += 1
            print(f"  Merged {filename}.")

    volume.commit()
    print(f"Done. {total} total records → {MERGED_FILENAME}.")


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    shard: int = 0,
    seed: int = 42,
    max_new_tokens: int = 512,
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    batch_size: int = 4,
    max_samples: int = 0,
):
    """Run generation for a training shard (0, 1, or 2)."""
    data_split = f"train_shard_{shard}"
    print(f"Launching generation: split={data_split}, batch_size={batch_size}, gpu=A10G...")
    run_pipeline.remote(
        data_split=data_split,
        seed=seed,
        max_new_tokens=max_new_tokens,
        model_name=model,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    print(f"\nGeneration complete.")
    print(f"To run judge:  modal run modal_app.py::judge --shard {shard}")
    print(f"To download:   modal volume get pipeline-outputs pipeline_output_{data_split}.jsonl ./data/outputs/")


@app.local_entrypoint()
def eval_set(
    split: str = "val",
    seed: int = 42,
    max_new_tokens: int = 512,
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    batch_size: int = 4,
):
    """Run generation for val or test split."""
    if split not in ("val", "test"):
        raise ValueError("--split must be 'val' or 'test'")
    print(f"Launching generation: split={split}, batch_size={batch_size}, gpu=A10G...")
    run_pipeline.remote(
        data_split=split,
        seed=seed,
        max_new_tokens=max_new_tokens,
        model_name=model,
        batch_size=batch_size,
    )
    print(f"\nGeneration complete.")
    print(f"To run judge:  modal run modal_app.py::judge --split {split}")
    print(f"To download:   modal volume get pipeline-outputs pipeline_output_{split}.jsonl ./data/outputs/")


@app.local_entrypoint()
def judge(
    shard: int = -1,
    split: str = "",
):
    """Run judge for a training shard (--shard 0/1/2) or eval split (--split val/test)."""
    if split in ("val", "test"):
        data_split = split
    elif shard >= 0:
        data_split = f"train_shard_{shard}"
    else:
        raise ValueError("Provide --shard 0/1/2 or --split val/test")
    print(f"Launching judge pass for split={data_split}...")
    run_judge.remote(data_split=data_split)
    print("Judge complete.")


@app.local_entrypoint()
def merge():
    """Merge all shard and eval files into pipeline_output.jsonl."""
    print("Merging all splits into pipeline_output.jsonl...")
    run_merge.remote()
    print(f"Merge complete.")
    print(f"To download: modal volume get pipeline-outputs {MERGED_FILENAME} ./data/outputs/")
