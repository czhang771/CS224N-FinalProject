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
    .add_local_file("extract_features.py", remote_path="/root/extract_features.py")
    .add_local_file("extract_features_v2.py", remote_path="/root/extract_features_v2.py")
)

# Lightweight image for ablation (no GPU needed)
ablation_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    )
    .add_local_file("results/v3/ablation_study_v3.py", remote_path="/root/ablation_study_v3.py")
    .add_local_file("results/v4/ablation_study_v4.py", remote_path="/root/ablation_study_v4.py")
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


@app.function(
    timeout=1800,
    volumes={VOLUME_PATH: volume},
)
def run_feature_extraction(data_split: str):
    import sys
    import json
    import numpy as np
    from pathlib import Path

    sys.path.insert(0, "/root")
    from extract_features import extract_record

    # Input JSONL
    if data_split in ("val", "test"):
        input_path = f"{VOLUME_PATH}/pipeline_output_{data_split}.jsonl"
        output_path = f"{VOLUME_PATH}/features_{data_split}.npz"
    else:
        shard_idx = data_split.split("_")[-1]
        input_path  = f"{VOLUME_PATH}/pipeline_output_train_shard{shard_idx}.jsonl"
        output_path = f"{VOLUME_PATH}/features_train_shard{shard_idx}.npz"

    volume.reload()
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {input_path}")

    all_scalars, all_hidden, all_labels, all_pubids = [], [], [], []
    feature_names = None
    n_skipped = 0

    for i, record in enumerate(records):
        try:
            scalar, hidden, label, pubid = extract_record(record)
        except Exception as e:
            print(f"  WARNING: skipping record {i} (pubid={record.get('pubid')}): {e}")
            n_skipped += 1
            continue
        if feature_names is None:
            feature_names = list(scalar.keys())
        all_scalars.append(list(scalar.values()))
        all_hidden.append(hidden)
        all_labels.append(label)
        all_pubids.append(pubid)

    X        = np.array(all_scalars, dtype=np.float32)
    X_hidden = np.array(all_hidden,  dtype=np.float32)
    y        = np.array(all_labels,  dtype=np.int32)
    pubids   = np.array(all_pubids,  dtype=object)

    np.savez_compressed(
        output_path,
        scalar_features=X,
        hidden_states=X_hidden,
        y=y,
        pubids=pubids,
        feature_names=np.array(feature_names, dtype=object),
    )
    volume.commit()

    label_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"Done. {len(all_scalars)} extracted, {n_skipped} skipped.")
    print(f"  scalar_features={X.shape}  hidden_states={X_hidden.shape}")
    print(f"  Labels — hallucinated(1): {label_counts.get(1,0)}, faithful(0): {label_counts.get(0,0)}, unknown(-1): {label_counts.get(-1,0)}")
    print(f"  Saved: {output_path}")


@app.local_entrypoint()
def extract(
    shard: int = -1,
    split: str = "",
):
    """Extract features from a judged shard or eval split into a compact .npz on the volume.

    Usage:
        modal run modal_app.py::extract --shard 0   # train shard
        modal run modal_app.py::extract --split val  # val/test split
    """
    if split in ("val", "test"):
        data_split = split
    elif shard >= 0:
        data_split = f"train_shard_{shard}"
    else:
        raise ValueError("Provide --shard 0/1/2 or --split val/test")
    print(f"Extracting features for split={data_split}...")
    run_feature_extraction.remote(data_split=data_split)
    if split in ("val", "test"):
        fname = f"features_{split}.npz"
    else:
        fname = f"features_train_shard{shard}.npz"
    print(f"Done. To download: modal volume get pipeline-outputs {fname} ./data/features/")


# ---------------------------------------------------------------------------
# V2 feature extraction (adds per_head_ctx: (N, 36, 16) to output)
# ---------------------------------------------------------------------------
@app.function(
    timeout=1800,
    volumes={VOLUME_PATH: volume},
)
def run_feature_extraction_v2(data_split: str):
    import sys
    import json
    import numpy as np

    sys.path.insert(0, "/root")
    from extract_features_v2 import extract_record

    if data_split in ("val", "test"):
        input_path  = f"{VOLUME_PATH}/pipeline_output_{data_split}.jsonl"
        output_path = f"{VOLUME_PATH}/features_v2_{data_split}.npz"
    else:
        shard_idx   = data_split.split("_")[-1]
        input_path  = f"{VOLUME_PATH}/pipeline_output_train_shard{shard_idx}.jsonl"
        output_path = f"{VOLUME_PATH}/features_v2_train_shard{shard_idx}.npz"

    volume.reload()
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {input_path}")

    all_scalars, all_hidden, all_per_head, all_labels, all_pubids = [], [], [], [], []
    feature_names = None
    n_skipped = 0

    for i, record in enumerate(records):
        try:
            scalar, hidden, per_head_ctx, label, pubid = extract_record(record)
        except Exception as e:
            print(f"  WARNING: skipping record {i} (pubid={record.get('pubid')}): {e}")
            n_skipped += 1
            continue
        if feature_names is None:
            feature_names = list(scalar.keys())
        all_scalars.append(list(scalar.values()))
        all_hidden.append(hidden)
        all_per_head.append(per_head_ctx)
        all_labels.append(label)
        all_pubids.append(pubid)

    X          = np.array(all_scalars,  dtype=np.float32)
    X_hidden   = np.array(all_hidden,   dtype=np.float32)
    X_per_head = np.array(all_per_head, dtype=np.float32)
    y          = np.array(all_labels,   dtype=np.int32)
    pubids     = np.array(all_pubids,   dtype=object)

    np.savez_compressed(
        output_path,
        scalar_features=X,
        hidden_states=X_hidden,
        per_head_ctx=X_per_head,
        y=y,
        pubids=pubids,
        feature_names=np.array(feature_names, dtype=object),
    )
    volume.commit()

    label_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"Done. {len(all_scalars)} extracted, {n_skipped} skipped.")
    print(f"  scalar_features={X.shape}  hidden_states={X_hidden.shape}  per_head_ctx={X_per_head.shape}")
    print(f"  Labels — hallucinated(1): {label_counts.get(1,0)}, faithful(0): {label_counts.get(0,0)}, unknown(-1): {label_counts.get(-1,0)}")
    print(f"  Saved: {output_path}")


@app.local_entrypoint()
def extract_v2(
    shard: int = -1,
    split: str = "",
):
    """Extract v2 features (adds per_head_ctx (N,36,16)) from a judged shard or eval split.

    Usage:
        modal run modal_app.py::extract_v2 --shard 0
        modal run modal_app.py::extract_v2 --split val
    """
    if split in ("val", "test"):
        data_split = split
        fname = f"features_v2_{split}.npz"
    elif shard >= 0:
        data_split = f"train_shard_{shard}"
        fname = f"features_v2_train_shard{shard}.npz"
    else:
        raise ValueError("Provide --shard 0/1/2 or --split val/test")
    print(f"Extracting v2 features for split={data_split}...")
    run_feature_extraction_v2.remote(data_split=data_split)
    print(f"Done. To download: modal volume get pipeline-outputs {fname} ./data/features/")


@app.local_entrypoint()
def merge():
    """Merge all shard and eval files into pipeline_output.jsonl."""
    print("Merging all splits into pipeline_output.jsonl...")
    run_merge.remote()
    print(f"Merge complete.")
    print(f"To download: modal volume get pipeline-outputs {MERGED_FILENAME} ./data/outputs/")


# ---------------------------------------------------------------------------
# Ablation study v3
# ---------------------------------------------------------------------------
@app.function(
    image=ablation_image,
    timeout=1800,
    volumes={VOLUME_PATH: volume},
)
def run_ablation_v3():
    import os
    import shutil
    import subprocess

    volume.reload()

    # Copy feature files from volume to a local temp dir accessible to the script
    tmp_data = "/tmp/features"
    os.makedirs(tmp_data, exist_ok=True)
    for fname in ("features_v2_train_all.npz", "features_v2_val.npz", "features_v2_test.npz"):
        src = f"{VOLUME_PATH}/{fname}"
        dst = f"{tmp_data}/{fname}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"Copied {fname}")

    tmp_out = "/tmp/ablation_out"
    os.makedirs(tmp_out, exist_ok=True)

    env = os.environ.copy()
    env["FEATURE_DATA_DIR"] = tmp_data
    env["ABLATION_OUT_DIR"] = tmp_out

    import subprocess
    subprocess.run(["python", "/root/ablation_study_v3.py"], check=True, env=env)

    # Persist outputs back to volume
    for fname in os.listdir(tmp_out):
        src = os.path.join(tmp_out, fname)
        dst = f"{VOLUME_PATH}/ablation_v3_{fname}"
        shutil.copy2(src, dst)
        print(f"Saved to volume: ablation_v3_{fname}")
    volume.commit()


@app.local_entrypoint()
def ablation_v3():
    """Run ablation study v3 on Modal (CPU only).

    Usage:
        modal run modal_app.py::ablation_v3

    Download results:
        modal volume get pipeline-outputs ablation_v3_results_ablation_v3.csv ./results/v3/
        modal volume get pipeline-outputs ablation_v3_roc_curves_v3.png ./results/v3/
        modal volume get pipeline-outputs ablation_v3_pr_curves_v3.png ./results/v3/
    """
    print("Launching ablation study v3 on Modal...")
    run_ablation_v3.remote()
    print("\nDone. Download results with:")
    print("  modal volume get pipeline-outputs ablation_v3_results_ablation_v3.csv ./results/v3/")
    print("  modal volume get pipeline-outputs ablation_v3_roc_curves_v3.png ./results/v3/")
    print("  modal volume get pipeline-outputs ablation_v3_pr_curves_v3.png ./results/v3/")


# ---------------------------------------------------------------------------
# Ablation study v4 (PCA on hidden states)
# ---------------------------------------------------------------------------
@app.function(
    image=ablation_image,
    timeout=1800,
    volumes={VOLUME_PATH: volume},
)
def run_ablation_v4():
    import os
    import shutil
    import subprocess

    volume.reload()

    tmp_data = "/tmp/features"
    os.makedirs(tmp_data, exist_ok=True)
    for fname in ("features_v2_train_all.npz", "features_v2_val.npz", "features_v2_test.npz"):
        src = f"{VOLUME_PATH}/{fname}"
        dst = f"{tmp_data}/{fname}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"Copied {fname}")

    tmp_out = "/tmp/ablation_out_v4"
    os.makedirs(tmp_out, exist_ok=True)

    env = os.environ.copy()
    env["FEATURE_DATA_DIR"] = tmp_data
    env["ABLATION_OUT_DIR"] = tmp_out

    subprocess.run(["python", "/root/ablation_study_v4.py"], check=True, env=env)

    for fname in os.listdir(tmp_out):
        src = os.path.join(tmp_out, fname)
        dst = f"{VOLUME_PATH}/ablation_v4_{fname}"
        shutil.copy2(src, dst)
        print(f"Saved to volume: ablation_v4_{fname}")
    volume.commit()


@app.local_entrypoint()
def ablation_v4():
    """Run PCA ablation study (v4) on Modal (CPU only).

    Usage:
        modal run modal_app.py::ablation_v4

    Download results:
        modal volume get pipeline-outputs ablation_v4_results_ablation_v4.csv ./results/v4/
        modal volume get pipeline-outputs ablation_v4_pca_auc_v4.png ./results/v4/
        modal volume get pipeline-outputs ablation_v4_roc_curves_v4.png ./results/v4/
    """
    print("Launching PCA ablation study v4 on Modal...")
    run_ablation_v4.remote()
    print("\nDone. Download results with:")
    print("  modal volume get pipeline-outputs ablation_v4_results_ablation_v4.csv ./results/v4/")
    print("  modal volume get pipeline-outputs ablation_v4_pca_auc_v4.png ./results/v4/")
    print("  modal volume get pipeline-outputs ablation_v4_roc_curves_v4.png ./results/v4/")
