"""
Feature extraction script for hallucination detection pipeline.

Each teammate runs this on their own downloaded shard JSONL after judge pass completes.

Usage:
    python extract_features.py --input ./data/outputs/pipeline_output_train_shard1.jsonl \
                               --output ./data/features/features_train_shard1.npz

Output .npz contains:
    scalar_features  (N, 91)    float32  — all scalar/per-layer features
    hidden_states    (N, 2048)  float32  — layer-18 mean hidden state (for probe)
    y                (N,)       int32    — judge label: 1=hallucinated, 0=faithful, -1=unknown
    pubids           (N,)       str      — pubmed IDs
    feature_names    (91,)      str      — names for each column in scalar_features

Scalar features (91 total):
  Token-level (12):
    mean_token_prob, min_token_prob, std_token_prob, max_prob_gap, mean_entropy,
    max_entropy, std_entropy, early_divergence_index, answer_n_tokens,
    top1_top2_logit_gap, top10_mass_ratio, top100_entropy_mean
  Attention trajectory (5):
    mean_context_ratio, min_context_ratio, std_context_ratio,
    attention_ratio_slope, early_divergence_index_attention
  Per-layer (36 + 36 = 72 = 2 summary + 70 per-layer):
    attention_entropy_mean, attention_entropy_std,
    layer{00-35}_context_concentration, layer{00-35}_attention_entropy
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _token_prob_features(record: dict) -> dict[str, float]:
    """Features from token_log_probs, token_entropies, top100_logit_values."""
    uf = record.get("uncertainty_features", {})
    entropies = np.array(record["token_entropies"], dtype=np.float64)
    n = len(entropies)

    # Early divergence index: first step where entropy > mean + std, normalized by n
    mean_ent = entropies.mean()
    std_ent = entropies.std() + 1e-10
    diverge_mask = entropies > (mean_ent + std_ent)
    early_div = float(np.argmax(diverge_mask) / n) if diverge_mask.any() else 1.0

    # Top-100 logit distributional shape
    top100 = np.array(record["top100_logit_values"], dtype=np.float64)  # (n, 100)
    top100_shifted = top100 - top100.max(axis=1, keepdims=True)
    top100_exp = np.exp(top100_shifted)
    top100_probs = top100_exp / top100_exp.sum(axis=1, keepdims=True)  # (n, 100)

    top1_top2_logit_gap = float((top100[:, 0] - top100[:, 1]).mean())
    top10_mass_ratio = float(top100_probs[:, :10].sum(axis=1).mean())
    top100_entropy = float(
        (-np.sum(top100_probs * np.log(top100_probs + 1e-10), axis=1)).mean()
    )

    return {
        "mean_token_prob":        float(uf.get("mean_token_prob", 0.0)),
        "min_token_prob":         float(uf.get("min_token_prob", 0.0)),
        "std_token_prob":         float(uf.get("std_token_prob", 0.0)),
        "max_prob_gap":           float(uf.get("max_prob_gap", 0.0)),
        "mean_entropy":           float(uf.get("mean_entropy", 0.0)),
        "max_entropy":            float(entropies.max()),
        "std_entropy":            float(entropies.std()),
        "early_divergence_index": early_div,
        "answer_n_tokens":        float(record.get("answer_n_tokens", n)),
        "top1_top2_logit_gap":    top1_top2_logit_gap,
        "top10_mass_ratio":       top10_mass_ratio,
        "top100_entropy_mean":    top100_entropy,
    }


def _attention_features(record: dict) -> dict[str, float]:
    """Features from context_attention_ratios and mean_input_attention."""
    ratios = np.array(record["context_attention_ratios"], dtype=np.float64)
    n = len(ratios)
    mean_ratio = float(ratios.mean())
    std_ratio = float(ratios.std()) + 1e-10

    # Linear trend of context attention ratio over generation steps
    if n > 1:
        x = np.arange(n, dtype=np.float64)
        slope = float(np.polyfit(x, ratios, 1)[0])
    else:
        slope = 0.0

    # First step where ratio drops > 1 std below mean (normalized)
    diverge_mask = ratios < (mean_ratio - std_ratio)
    attn_div_idx = float(np.argmax(diverge_mask) / n) if diverge_mask.any() else 1.0

    # mean_input_attention: (n_layers, n_heads, padded_input_len)
    attn = np.array(record["mean_input_attention"], dtype=np.float64)
    n_layers, n_heads, seq_len = attn.shape

    # Normalize each (layer, head) attention row → probability distribution
    attn_norm = attn / (attn.sum(axis=-1, keepdims=True) + 1e-10)

    # Entropy per (layer, head)
    attn_entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-10), axis=-1)  # (n_layers, n_heads)

    # Per-layer context concentration: fraction of attention on context tokens
    ctx_start = max(0, min(record["context_start_idx"] + record["padding_offset"], seq_len))
    ctx_end   = max(0, min(record["context_end_idx"]   + record["padding_offset"], seq_len))
    ctx_attn  = attn[:, :, ctx_start:ctx_end].sum(axis=-1)       # (n_layers, n_heads)
    total_attn = attn.sum(axis=-1) + 1e-10                        # (n_layers, n_heads)
    per_layer_ctx_conc  = (ctx_attn / total_attn).mean(axis=1)   # (n_layers,)
    per_layer_attn_ent  = attn_entropy.mean(axis=1)              # (n_layers,)

    feats: dict[str, float] = {
        "mean_context_ratio":               mean_ratio,
        "min_context_ratio":                float(ratios.min()),
        "std_context_ratio":                float(ratios.std()),
        "attention_ratio_slope":            slope,
        "early_divergence_index_attention": attn_div_idx,
        "attention_entropy_mean":           float(attn_entropy.mean()),
        "attention_entropy_std":            float(attn_entropy.std()),
    }
    for i in range(n_layers):
        feats[f"layer{i:02d}_context_concentration"] = float(per_layer_ctx_conc[i])
    for i in range(n_layers):
        feats[f"layer{i:02d}_attention_entropy"] = float(per_layer_attn_ent[i])

    return feats


def _judge_label_to_int(judge_label) -> int:
    """1=hallucinated, 0=faithful, -1=unknown/missing."""
    if judge_label is None:
        return -1
    s = str(judge_label).lower().strip()
    if s in ("hallucinated", "hallucination", "yes", "1", "true"):
        return 1
    if s in ("faithful", "grounded", "no", "0", "false"):
        return 0
    return -1


def extract_record(record: dict) -> tuple[dict[str, float], np.ndarray, int, str]:
    """
    Returns (scalar_feats, hidden_state, label, pubid).
    Raises on bad/empty records so callers can skip them.
    """
    scalar = {**_token_prob_features(record), **_attention_features(record)}
    hidden = np.array(record["middle_layer_hidden_state"], dtype=np.float32)
    if hidden.shape != (2048,):
        raise ValueError(f"unexpected hidden_state shape {hidden.shape}")
    label  = _judge_label_to_int(record.get("judge_label"))
    pubid  = str(record.get("pubid", ""))
    return scalar, hidden, label, pubid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract features from a pipeline output JSONL shard.")
    parser.add_argument("--input",  required=True, help="Input JSONL file (e.g. pipeline_output_train_shard1.jsonl)")
    parser.add_argument("--output", required=True, help="Output .npz file (e.g. features_train_shard1.npz)")
    args = parser.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.input}")

    all_scalars: list[list[float]] = []
    all_hidden:  list[np.ndarray] = []
    all_labels:  list[int] = []
    all_pubids:  list[str] = []
    feature_names: list[str] | None = None
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

    if not all_scalars:
        print("ERROR: no records extracted successfully.")
        return

    X        = np.array(all_scalars, dtype=np.float32)   # (N, F)
    X_hidden = np.array(all_hidden,  dtype=np.float32)   # (N, 2048)
    y        = np.array(all_labels,  dtype=np.int32)      # (N,)
    pubids   = np.array(all_pubids,  dtype=object)        # (N,)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        scalar_features=X,
        hidden_states=X_hidden,
        y=y,
        pubids=pubids,
        feature_names=np.array(feature_names, dtype=object),
    )

    label_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"Done. {len(all_scalars)} records extracted, {n_skipped} skipped.")
    print(f"  scalar_features: {X.shape}  hidden_states: {X_hidden.shape}")
    print(f"  Labels — hallucinated(1): {label_counts.get(1, 0)}, faithful(0): {label_counts.get(0, 0)}, unknown(-1): {label_counts.get(-1, 0)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()