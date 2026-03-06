"""
Modal app for the hallucination detection pipeline.

Usage:
    # Store your Gemini API key as a Modal secret (one-time setup):
    modal secret create gemini-secret GEMINI_API_KEY=<your-key>

    # Run the full pipeline (100 samples) on a GPU:
    modal run modal_app.py

    # Custom run:
    modal run modal_app.py --n-samples 10 --seed 0 --skip-judge

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
    timeout=7200,  # 2 hours — enough for 100 samples
    secrets=[modal.Secret.from_name("gemini-secret")],
    volumes={VOLUME_PATH: volume},
)
def run_pipeline(
    n_samples: int = 100,
    seed: int = 42,
    max_new_tokens: int = 512,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    skip_judge: bool = False,
):
    import sys
    sys.path.insert(0, "/root")

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
            record.update(
                generator.generate(
                    question=example["question"],
                    context_text=example["context_text"],
                )
            )
        except Exception as e:
            print(f"\n  [Generator] Error on pubid={example['pubid']}: {e}")
            record.update({"generated_answer": "", "tokens": [], "token_log_probs": [], "uncertainty_features": {}})

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
