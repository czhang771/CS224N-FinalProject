"""
Feature extraction script for hallucination detection pipeline.

Each teammate runs this on their own downloaded shard JSONL after judge pass completes.

Usage:
    python extract_features.py --input ./data/outputs/pipeline_output_train_shard1.jsonl \
                               --output ./data/features/features_train_shard1.npz

Output .npz contains:
    scalar_features  (N, F)     float32  — all scalar/per-layer features
    hidden_states    (N, 2048)  float32  — layer-18 mean hidden state (for probe)
    y                (N,)       int32    — judge label: 1=hallucinated, 0=faithful, -1=unknown
    pubids           (N,)       str      — pubmed IDs
    feature_names    (F,)       str      — names for each column in scalar_features

Scalar feature groups:
  Token-level (12):
    mean_token_prob, min_token_prob, std_token_prob, max_prob_gap, mean_entropy,
    max_entropy, std_entropy, early_divergence_index, answer_n_tokens,
    top1_top2_logit_gap, top10_mass_ratio, top100_entropy_mean
  Token-level additional (5):
    first_token_entropy, entropy_first_half, entropy_last_half,
    entropy_spike_count_norm, frac_low_confidence
  Attention trajectory (5):
    mean_context_ratio, min_context_ratio, std_context_ratio,
    attention_ratio_slope, early_divergence_index_attention
  Attention trajectory additional (3):
    context_ratio_first_quarter, context_ratio_last_quarter, context_gradient_late_minus_early
  Per-layer (36 context_concentration + 36 attention_entropy = 72):
    layer{00-35}_context_concentration, layer{00-35}_attention_entropy
  Per-layer additional (36 head_disagreement):
    layer{00-35}_head_disagreement
  Global attention summary additional (2):
    attention_entropy_mean, attention_entropy_std
  Interaction additional (1):
    entropy_context_interaction
  EOS logit additional (2):
    frac_steps_eos_in_top100, mean_eos_logit_pre_final
  Text additional (3):
    answer_starts_with_decision, hedging_word_ratio, answer_word_count
"""

import argparse
import json
from pathlib import Path

import numpy as np

_HEDGING_WORDS = {
    "may", "might", "could", "possibly", "perhaps", "unclear", "uncertain",
    "likely", "unlikely", "suggest", "suggests", "suggested", "appear",
    "appears", "seem", "seems", "approximately", "generally", "typically",
}
_DECISION_WORDS = {"yes", "no", "maybe"}


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _token_prob_features(record: dict) -> dict[str, float]:
    """Features from token_log_probs, token_entropies, top100_logit_values."""
    uf = record.get("uncertainty_features", {})
    entropies = np.array(record["token_entropies"], dtype=np.float64)
    log_probs  = np.array(record["token_log_probs"],  dtype=np.float64)
    n = len(entropies)

    # Early divergence index: first step where entropy > mean + std, normalized by n
    mean_ent = entropies.mean()
    std_ent  = entropies.std() + 1e-10
    diverge_mask = entropies > (mean_ent + std_ent)
    early_div = float(np.argmax(diverge_mask) / n) if diverge_mask.any() else 1.0

    # Top-100 logit distributional shape
    top100 = np.array(record["top100_logit_values"], dtype=np.float64)  # (n, 100)
    top100_shifted = top100 - top100.max(axis=1, keepdims=True)
    top100_exp   = np.exp(top100_shifted)
    top100_probs = top100_exp / top100_exp.sum(axis=1, keepdims=True)   # (n, 100)

    top1_top2_logit_gap = float((top100[:, 0] - top100[:, 1]).mean())
    top10_mass_ratio    = float(top100_probs[:, :10].sum(axis=1).mean())
    top100_entropy      = float(
        (-np.sum(top100_probs * np.log(top100_probs + 1e-10), axis=1)).mean()
    )

    # --- additional features ---
    mid = max(1, n // 2)
    entropy_first_half = float(entropies[:mid].mean())                       # [additional]
    entropy_last_half  = float(entropies[mid:].mean()) if n > mid else float(entropies.mean())  # [additional]
    entropy_spike_count_norm = float(diverge_mask.sum() / n)                 # [additional]
    first_token_entropy      = float(entropies[0]) if n > 0 else 0.0         # [additional]
    frac_low_confidence      = float((log_probs < -0.693).mean())            # [additional] prob < 0.5

    return {
        "mean_token_prob":         float(uf.get("mean_token_prob", 0.0)),
        "min_token_prob":          float(uf.get("min_token_prob", 0.0)),
        "std_token_prob":          float(uf.get("std_token_prob", 0.0)),
        "max_prob_gap":            float(uf.get("max_prob_gap", 0.0)),
        "mean_entropy":            float(uf.get("mean_entropy", 0.0)),
        "max_entropy":             float(entropies.max()),
        "std_entropy":             float(entropies.std()),
        "early_divergence_index":  early_div,
        "answer_n_tokens":         float(record.get("answer_n_tokens", n)),
        "top1_top2_logit_gap":     top1_top2_logit_gap,
        "top10_mass_ratio":        top10_mass_ratio,
        "top100_entropy_mean":     top100_entropy,
        # additional
        "first_token_entropy":       first_token_entropy,
        "entropy_first_half":        entropy_first_half,
        "entropy_last_half":         entropy_last_half,
        "entropy_spike_count_norm":  entropy_spike_count_norm,
        "frac_low_confidence":       frac_low_confidence,
    }


def _attention_features(record: dict) -> dict[str, float]:
    """Features from context_attention_ratios and mean_input_attention."""
    ratios = np.array(record["context_attention_ratios"], dtype=np.float64)
    n = len(ratios)
    mean_ratio = float(ratios.mean())
    std_ratio  = float(ratios.std()) + 1e-10

    # Linear trend of context attention ratio over generation steps
    slope = float(np.polyfit(np.arange(n, dtype=np.float64), ratios, 1)[0]) if n > 1 else 0.0

    # First step where ratio drops > 1 std below mean (normalized)
    diverge_mask = ratios < (mean_ratio - std_ratio)
    attn_div_idx = float(np.argmax(diverge_mask) / n) if diverge_mask.any() else 1.0

    # --- additional: context ratio by quarter ---
    q = max(1, n // 4)
    ctx_ratio_first_q = float(ratios[:q].mean())                            # [additional]
    ctx_ratio_last_q  = float(ratios[-q:].mean())                           # [additional]

    # mean_input_attention: (n_layers, n_heads, padded_input_len)
    attn = np.array(record["mean_input_attention"], dtype=np.float64)
    n_layers, n_heads, seq_len = attn.shape

    # Normalize each (layer, head) attention row → probability distribution
    attn_norm  = attn / (attn.sum(axis=-1, keepdims=True) + 1e-10)
    attn_entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-10), axis=-1)  # (n_layers, n_heads)

    # Per-layer context concentration: fraction of attention on context tokens
    ctx_start = max(0, min(record["context_start_idx"] + record["padding_offset"], seq_len))
    ctx_end   = max(0, min(record["context_end_idx"]   + record["padding_offset"], seq_len))
    ctx_attn       = attn[:, :, ctx_start:ctx_end].sum(axis=-1)              # (n_layers, n_heads)
    total_attn     = attn.sum(axis=-1) + 1e-10                               # (n_layers, n_heads)
    per_layer_ctx  = (ctx_attn / total_attn).mean(axis=1)                    # (n_layers,)
    per_layer_ent  = attn_entropy.mean(axis=1)                               # (n_layers,)

    # --- additional: head disagreement per layer (std across heads) ---
    per_layer_head_disagree = (ctx_attn / total_attn).std(axis=1)           # [additional] (n_layers,)

    # --- additional: late-layer vs early-layer context concentration gradient ---
    mid_layer = n_layers // 2
    ctx_gradient = float(per_layer_ctx[mid_layer:].mean() - per_layer_ctx[:mid_layer].mean())  # [additional]

    feats: dict[str, float] = {
        "mean_context_ratio":               mean_ratio,
        "min_context_ratio":                float(ratios.min()),
        "std_context_ratio":                float(ratios.std()),
        "attention_ratio_slope":            slope,
        "early_divergence_index_attention": attn_div_idx,
        "attention_entropy_mean":           float(attn_entropy.mean()),
        "attention_entropy_std":            float(attn_entropy.std()),
        # additional
        "context_ratio_first_quarter":      ctx_ratio_first_q,
        "context_ratio_last_quarter":       ctx_ratio_last_q,
        "context_gradient_late_minus_early": ctx_gradient,
    }
    for i in range(n_layers):
        feats[f"layer{i:02d}_context_concentration"] = float(per_layer_ctx[i])
    for i in range(n_layers):
        feats[f"layer{i:02d}_attention_entropy"] = float(per_layer_ent[i])
    for i in range(n_layers):
        feats[f"layer{i:02d}_head_disagreement"] = float(per_layer_head_disagree[i])  # [additional]

    return feats


def _interaction_features(record: dict) -> dict[str, float]:
    """[additional] Cross-signal interaction features."""
    entropies = np.array(record["token_entropies"],        dtype=np.float64)
    ratios    = np.array(record["context_attention_ratios"], dtype=np.float64)
    # Align lengths (truncate to shorter; they should already match)
    n = min(len(entropies), len(ratios))
    # High when model is uncertain AND not looking at context — direct hallucination signal
    entropy_context_interaction = float((entropies[:n] * (1.0 - ratios[:n])).mean())  # [additional]
    return {"entropy_context_interaction": entropy_context_interaction}


def _eos_features(record: dict) -> dict[str, float]:
    """[additional] EOS token logit features — how strongly the model considered stopping early."""
    token_ids_per_step = record["top100_logit_token_ids"]   # list[list[int]], len=n_valid
    logit_vals_per_step = record["top100_logit_values"]     # list[list[float]], len=n_valid
    n = len(token_ids_per_step)
    if n == 0:
        return {"frac_steps_eos_in_top100": 0.0, "mean_eos_logit_pre_final": 0.0}

    # Infer EOS token ID from the final generation step (EOS was chosen there)
    # top100_logit_token_ids[-1][0] = highest-logit token at last step = EOS
    eos_id = token_ids_per_step[-1][0]

    # Examine only pre-final steps (last step trivially has EOS)
    pre_final_steps = n - 1
    eos_in_top100 = 0
    eos_logits: list[float] = []

    for step in range(pre_final_steps):
        ids  = token_ids_per_step[step]
        vals = logit_vals_per_step[step]
        if eos_id in ids:
            eos_in_top100 += 1
            idx = ids.index(eos_id)
            eos_logits.append(vals[idx])

    frac = float(eos_in_top100 / pre_final_steps) if pre_final_steps > 0 else 0.0
    mean_logit = float(np.mean(eos_logits)) if eos_logits else 0.0

    return {
        "frac_steps_eos_in_top100":    frac,       # [additional]
        "mean_eos_logit_pre_final":    mean_logit, # [additional]
    }


def _text_features(record: dict) -> dict[str, float]:
    """[additional] Text-level features from generated_answer."""
    answer = str(record.get("generated_answer", "")).strip().lower()
    words  = answer.split()
    n_words = len(words)

    # Does the answer start with yes / no / maybe?
    starts_with_decision = float(bool(words) and words[0] in _DECISION_WORDS)  # [additional]

    # Fraction of words that are hedging language
    hedging_count = sum(1 for w in words if w.rstrip(".,;:!?") in _HEDGING_WORDS)
    hedging_ratio = float(hedging_count / n_words) if n_words > 0 else 0.0     # [additional]

    return {
        "answer_starts_with_decision": starts_with_decision,  # [additional]
        "hedging_word_ratio":          hedging_ratio,          # [additional]
        "answer_word_count":           float(n_words),         # [additional]
    }


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
    """Returns (scalar_feats, hidden_state, label, pubid). Raises on bad records."""
    scalar = {
        **_token_prob_features(record),
        **_attention_features(record),
        **_interaction_features(record),
        **_eos_features(record),
        **_text_features(record),
    }
    hidden = np.array(record["middle_layer_hidden_state"], dtype=np.float32)
    if hidden.shape != (2048,):
        raise ValueError(f"unexpected hidden_state shape {hidden.shape}")
    label = _judge_label_to_int(record.get("judge_label"))
    pubid = str(record.get("pubid", ""))
    return scalar, hidden, label, pubid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract features from a pipeline output JSONL shard.")
    parser.add_argument("--input",  required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output .npz file")
    args = parser.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.input}")

    all_scalars: list[list[float]] = []
    all_hidden:  list[np.ndarray]  = []
    all_labels:  list[int]         = []
    all_pubids:  list[str]         = []
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

    X        = np.array(all_scalars, dtype=np.float32)
    X_hidden = np.array(all_hidden,  dtype=np.float32)
    y        = np.array(all_labels,  dtype=np.int32)
    pubids   = np.array(all_pubids,  dtype=object)

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
    print(f"  Labels — hallucinated(1): {label_counts.get(1,0)}, faithful(0): {label_counts.get(0,0)}, unknown(-1): {label_counts.get(-1,0)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()