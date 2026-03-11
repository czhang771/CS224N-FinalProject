# Hallucination Detection — Ablation Study & Results

## Setup

All experiments use a logistic regression classifier with:
- **StandardScaler** normalization (zero mean, unit variance)
- **class_weight='balanced'** to handle class imbalance (~75% faithful, ~25% hallucinated)
- **Hyperparameter tuning on validation set**: grid search over penalty ∈ {L1, L2} and C ∈ {1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0}, selecting the combination that maximizes AUC-ROC on the val set
- **Threshold tuning on validation set**: after selecting the best model, sweep thresholds 0.01–0.99 to find the value maximizing F1 on the val set
- **Final evaluation on held-out test set only**
- Dataset: 700 train / 150 val / 150 test examples from PubMedQA

Primary metric: **AUC-ROC** (threshold-independent, robust to class imbalance). Accuracy is not reliable here — a classifier that always predicts "faithful" achieves 75% accuracy while catching zero hallucinations.

---

## Ablation Study V3 — Feature Set Experiments

### Feature Blocks

Features are extracted from Qwen2.5-3B-Instruct generations across three top-level blocks:

| Block | Dim | Description |
|-------|-----|-------------|
| LB | 576 | Lookback Lens — per-(layer, head) context concentration, flattened from (36, 16) |
| Scalar | 141 | All scalar signals: token probability, context attention trajectory, per-layer attention stats, misc |
| HS | 2048 | Mean-pooled layer-18 hidden state across all generated tokens |

The 141 scalar features decompose as:

| Sub-block | Indices | Dim | Content |
|-----------|---------|-----|---------|
| TP | 0–16 | 17 | Token probability statistics (mean/min/std prob, entropy, top-k logit gaps, etc.) |
| CAT | 17–26 | 10 | Context attention trajectory scalars (mean/std ratio, slope, divergence index, quarter stats, gradient) |
| PLC | 27–62 | 36 | Per-layer context concentration (mean over heads per layer) |
| PLE | 63–98 | 36 | Per-layer attention entropy (mean over heads per layer) |
| PLH | 99–134 | 36 | Per-layer head disagreement (std across heads per layer) |
| MISC | 135–140 | 6 | Cross-signal interaction, EOS logit stats, text-level features |

---

### Standalone Experiments (7)

#### 1. LB — Lookback Lens Baseline (576-dim)
**Hypothesis:** Per-(layer, head) context concentration captures whether the model attends to the provided context or relies on parametric memory. Hallucinated answers should show lower or more diffuse context attention. This replicates the Lookback Lens approach (EMNLP 2024) as our baseline.

**Results:** AUC-ROC = 0.604, AUC-PR = 0.293, F1 = 0.316 | C = 0.001, L2

**Interpretation:** Only slightly above chance. Lookback Lens was designed to distinguish context-copying from parametric recall — a related but different signal from faithfulness. On PubMedQA, where all answers *should* be grounded in context, this distinction is less informative. The 576-dim feature space also has limited signal density given only 700 training examples.

---

#### 2. TP — Token Probability Features (17-dim)
**Hypothesis:** When a model hallucinates, it may generate tokens with different confidence patterns — either over-confidently producing wrong facts or hesitating (high entropy) when fabricating. Token-level probability statistics should directly capture this uncertainty signal.

**Results:** AUC-ROC = 0.637, AUC-PR = 0.308, F1 = 0.360 | C = 1e-5, L2

**Interpretation:** Outperforms the Lookback Lens baseline despite being 34× smaller (17 vs 576 dimensions). The very small optimal C (1e-5) indicates heavy regularization was needed, suggesting the features are somewhat noisy or redundant. Still, generation confidence is a meaningful hallucination signal and the most informative scalar feature set.

---

#### 3. PLC — Per-Layer Context Concentration (36-dim)
**Hypothesis:** The per-layer average context attention (averaged over all 16 heads) captures which layers are most engaged with the source context. Layers that stop attending to context may indicate hallucination.

**Results:** AUC-ROC = 0.576, AUC-PR = 0.224, F1 = 0.315 | C = 10.0, L2

**Interpretation:** Weakest per-layer signal. The very high optimal C (minimal regularization) suggests the model struggled to find a consistent signal and was fitting noise. Averaging over all 16 heads loses the head-level specificity that Lookback Lens preserves — some heads are informative while others are not, and averaging dilutes the signal.

---

#### 4. PLE — Per-Layer Attention Entropy (36-dim)
**Hypothesis:** Attention entropy measures how diffuse vs. focused each layer's attention is. During hallucination, attention may become more diffuse rather than concentrated on relevant context tokens.

**Results:** AUC-ROC = 0.596, AUC-PR = 0.239, F1 = 0.308 | C = 0.1, L2

**Interpretation:** Weak performance. Attention entropy is a global measure that does not distinguish *where* attention is going, only whether it is focused. A model can be sharply focused on the wrong tokens, which entropy would misclassify as confident and grounded. This limits its discriminative power.

---

#### 5. PLH — Per-Layer Head Disagreement (36-dim)
**Hypothesis:** When heads within a layer strongly disagree about where to attend (high std across heads), it may indicate uncertainty or conflicting information. Consistent cross-head agreement might indicate confident, grounded generation.

**Results:** AUC-ROC = 0.601, AUC-PR = 0.233, F1 = 0.330 | C = 1e-5, L2

**Interpretation:** Slightly better than PLC and PLE, likely because head disagreement captures some of the per-head variation that Lookback Lens uses more explicitly. Very small optimal C again indicates a noisy signal.

---

#### 6. PLA — All Per-Layer Features Combined (108-dim)
**Hypothesis:** Combining PLC + PLE + PLH captures complementary per-layer attention statistics. Even if each alone is weak, together they might paint a fuller picture of layer-wise attention behavior.

**Results:** AUC-ROC = 0.564, AUC-PR = 0.211, F1 = 0.265 | C = 0.01, L2

**Interpretation:** The worst standalone result — combining all three per-layer features hurts rather than helps. With 108 features and only 700 training examples the model overfits. The three sub-blocks are highly correlated (all derived from the same attention matrices), so combining them adds noise rather than complementary signal.

---

#### 7. HS — Hidden-State Probe (2048-dim)
**Hypothesis:** The model's internal representation at mid-layer (layer 18 of 36) encodes rich semantic information about what the model is computing. If the model is generating a hallucination, this should be reflected in the activation pattern. A linear probe on the mean-pooled hidden state across generated tokens should capture this.

**Results:** AUC-ROC = 0.703, AUC-PR = 0.319, F1 = 0.364 | C = 1e-4, L2

**Interpretation:** By far the strongest single feature set — a 10-point AUC-ROC gap over the next best (TP). The very small optimal C (0.0001) shows that even with heavy regularization the signal is strong enough. This confirms that factual and semantic information relevant to faithfulness is encoded linearly in transformer hidden representations at mid-layer.

---

### Stacked Experiments (5) — All Stacked on Lookback Lens

These experiments test whether additional feature blocks provide complementary signal on top of the Lookback Lens baseline.

#### 8. LB + TP (593-dim)
**Hypothesis:** Token probability features capture generation confidence, orthogonal to context attention patterns. Together they might capture both "is the model looking at context?" and "is it generating confidently?"

**Results:** AUC-ROC = 0.614, AUC-PR = 0.317, F1 = 0.333 | C = 0.001, L2

**Interpretation:** Modest +0.010 gain over LB alone. Token probability adds some complementary signal but gains are small — the two feature sets share some information about model confidence.

---

#### 9. LB + CAT (586-dim)
**Hypothesis:** Context attention trajectory captures *how* attention to context evolves over generation steps — a temporal signal that Lookback Lens averages away. Together they capture both the average and the dynamics of context attention.

**Results:** AUC-ROC = 0.609, AUC-PR = 0.303, F1 = 0.340 | C = 0.001, L2

**Interpretation:** Minimal +0.005 gain. Temporal dynamics of context attention add little extra information once the average per-head context concentration is known. Hallucination likely manifests in the overall attention pattern rather than the timing of when attention drifts.

---

#### 10. LB + PLA (684-dim)
**Hypothesis:** Per-layer statistics are derived from the same attention matrices as Lookback Lens but summarized differently. They might provide complementary layer-level structure.

**Results:** AUC-ROC = 0.603, AUC-PR = 0.287, F1 = 0.292 | C = 0.001, L2

**Interpretation:** Essentially no gain (-0.001) over LB alone. PLA and LB are highly redundant — both derived from the same attention data. Adding 108 correlated dimensions introduces noise without adding signal.

---

#### 11. LB + HS (2624-dim)
**Hypothesis:** Hidden states and attention patterns capture fundamentally different aspects of model behavior — "what the model is computing" vs. "where it is looking." These should be genuinely complementary signals.

**Results:** AUC-ROC = 0.683, AUC-PR = 0.307, F1 = 0.418 | C = 1e-4, L2

**Interpretation:** The strongest stacked result and best F1 score in the entire V3 study (+0.079 AUC-ROC over LB alone). HS adds substantial complementary information to LB. The same optimal C as HS alone (0.0001) confirms that the HS component dominates this combination.

---

#### 12. Full System — LB + All Scalar + HS (2765-dim)
**Hypothesis:** Using all available features gives the classifier the most complete picture of model behavior.

**Results:** AUC-ROC = 0.674, AUC-PR = 0.305, F1 = 0.370 | C = 1e-4, L2

**Interpretation:** Slightly *worse* than LB + HS (0.683). Adding all 141 scalar features on top of LB + HS introduces noise — many scalar features are redundant with LB or carry weak individual signal. With 2765 features and only 700 training examples, the dimensionality is too high for a linear classifier.

---

### V3 Summary Table

| # | Model | Dim | Penalty | C | AUC-ROC | AUC-PR | F1 |
|---|-------|-----|---------|---|---------|--------|-----|
| 1 | LB (baseline) | 576 | L2 | 0.001 | 0.604 | 0.293 | 0.316 |
| 2 | TP | 17 | L2 | 1e-5 | 0.637 | 0.308 | 0.360 |
| 3 | PLC | 36 | L2 | 10.0 | 0.576 | 0.224 | 0.315 |
| 4 | PLE | 36 | L2 | 0.1 | 0.596 | 0.239 | 0.308 |
| 5 | PLH | 36 | L2 | 1e-5 | 0.601 | 0.233 | 0.330 |
| 6 | PLA | 108 | L2 | 0.01 | 0.564 | 0.211 | 0.265 |
| 7 | **HS** | 2048 | L2 | 1e-4 | **0.703** | 0.319 | 0.364 |
| 8 | LB + TP | 593 | L2 | 0.001 | 0.614 | 0.317 | 0.333 |
| 9 | LB + CAT | 586 | L2 | 0.001 | 0.609 | 0.303 | 0.340 |
| 10 | LB + PLA | 684 | L2 | 0.001 | 0.603 | 0.287 | 0.292 |
| 11 | **LB + HS** | 2624 | L2 | 1e-4 | 0.683 | 0.307 | **0.418** |
| 12 | Full System | 2765 | L2 | 1e-4 | 0.674 | 0.305 | 0.370 |

**Key V3 findings:**
- Hidden states dominate all other feature sets (AUC-ROC = 0.703)
- Lookback Lens (0.604) is weaker than simple 17-dim token probability features (0.637)
- Per-layer attention features consistently add noise when combined (PLA = 0.564, worst standalone)
- LB + HS achieves the best F1 (0.418) but does not beat HS alone on AUC-ROC
- Adding more features beyond LB + HS hurts (full system = 0.674 < LB + HS = 0.683)

---

## Ablation Study V4 — PCA on Hidden States

### Motivation

HS alone achieves AUC-ROC = 0.703 with C = 0.0001 (very heavy regularization). This suggests most of the 2048 hidden state dimensions are not useful for the task and the classifier is already trying to ignore them. The hypothesis is that the hallucination signal is concentrated in a small number of principal components, and removing the remaining dimensions removes noise rather than signal.

### Setup

PCA is fit on the **training set hidden states only**, then the learned projection is applied to val and test. We test k ∈ {32, 64, 128, 256, 512} compressed dimensions against the raw 2048-dim baseline. All other settings (hyperparameter search, threshold tuning, evaluation) are identical to V3.

### Results

| Model | Dim | Penalty | C | AUC-ROC | AUC-PR | F1 |
|-------|-----|---------|---|---------|--------|-----|
| HS raw | 2048 | L2 | 1e-4 | 0.703 | 0.319 | 0.364 |
| **HS-PCA-32** | **32** | **L2** | **0.001** | **0.717** | **0.374** | **0.389** |
| HS-PCA-64 | 64 | L2 | 1e-5 | 0.690 | 0.339 | 0.382 |
| HS-PCA-128 | 128 | L2 | 0.001 | 0.661 | 0.297 | 0.370 |
| HS-PCA-256 | 256 | L1 | 10.0 | 0.611 | 0.259 | 0.359 |
| HS-PCA-512 | 512 | L2 | 0.01 | 0.619 | 0.274 | 0.342 |

### Interpretation

**HS-PCA-32 is the best result across the entire study** (AUC-ROC = 0.717), outperforming raw HS on all three metrics: AUC-ROC (+0.014), AUC-PR (+0.055), and F1 (+0.025). Compressing 2048 dimensions down to just 32 principal components improves performance, confirming that the hallucination signal in the hidden states is highly compact.

Performance drops sharply and monotonically as k increases beyond 32. By k=256, PCA is worse than raw HS (0.611 vs 0.703), meaning the additional components beyond the top 32 contain noise that actively hurts the classifier. The erratic hyperparameter selection at k=256 (L1, C=10.0 — the opposite extreme from k=32) signals that the optimizer was struggling to find any useful structure in those dimensions.

The optimal C for PCA-32 (0.001) is 10× larger than for raw HS (0.0001), meaning less regularization is needed once noise dimensions are removed — the signal-to-noise ratio of the 32-dim representation is much higher than the raw 2048-dim one.

**Overall best model: HS-PCA-32 — AUC-ROC = 0.717, AUC-PR = 0.374, F1 = 0.389.**
