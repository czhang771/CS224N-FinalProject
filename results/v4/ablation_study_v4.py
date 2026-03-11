"""
Ablation study v4 — PCA on hidden states (standalone HS only).

Builds on v3 results. Tests whether compressing the 2048-dim hidden state
with PCA improves over raw HS standalone (experiment 7 from v3).

PCA is fit on training set only, then applied to val and test.

Experiments:
  Baseline (from v3 for reference):
    A. HS raw              (2048-dim)

  PCA variants (k = 32, 64, 128, 256, 512):
    1. HS-PCA-k            (k-dim)
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("FEATURE_DATA_DIR", "../../data/features")
OUT_DIR  = os.environ.get("ABLATION_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_split(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

train = load_split(os.path.join(DATA_DIR, "features_v2_train_all.npz"))
val   = load_split(os.path.join(DATA_DIR, "features_v2_val.npz"))
test  = load_split(os.path.join(DATA_DIR, "features_v2_test.npz"))

hs_tr = train["hidden_states"]
hs_va = val["hidden_states"]
hs_te = test["hidden_states"]

y_tr = train["y"].astype(int)
y_va = val["y"].astype(int)
y_te = test["y"].astype(int)

print(f"Train: HS={hs_tr.shape}, y={y_tr.shape}")
print(f"Val:   HS={hs_va.shape}, y={y_va.shape}")
print(f"Test:  HS={hs_te.shape}, y={y_te.shape}")
for split_name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  {split_name}: faithful(0)={counts.get(0,0)}, hallucinated(1)={counts.get(1,0)}")
print()

# ---------------------------------------------------------------------------
# Precompute PCA projections for each k
# PCA is fit on train HS only, then applied to val/test
# ---------------------------------------------------------------------------

PCA_COMPONENTS = [32, 64, 128, 256, 512]

hs_pca = {}  # k -> (hs_tr_k, hs_va_k, hs_te_k)
explained = {}

for k in PCA_COMPONENTS:
    pca = PCA(n_components=k, random_state=42)
    tr_k = pca.fit_transform(hs_tr)
    va_k = pca.transform(hs_va)
    te_k = pca.transform(hs_te)
    hs_pca[k] = (tr_k, va_k, te_k)
    explained[k] = pca.explained_variance_ratio_.sum()
    print(f"PCA k={k:4d}: explains {explained[k]*100:.1f}% of HS variance")
print()

# ---------------------------------------------------------------------------
# Build experiments
# ---------------------------------------------------------------------------

experiments = [
    # Baseline (raw HS, reproduced from v3 for direct comparison)
    ("HS raw (2048-dim)", [hs_tr], [hs_va], [hs_te]),
]

for k in PCA_COMPONENTS:
    tr_k, va_k, te_k = hs_pca[k]
    experiments.append((f"HS-PCA-{k}", [tr_k], [va_k], [te_k]))

# ---------------------------------------------------------------------------
# Hyperparameter search (same as v3)
# ---------------------------------------------------------------------------

C_VALUES  = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
PENALTIES = ["l1", "l2"]


def tune_C(X_tr, y_tr, X_va, y_va):
    best_auc = 0.0
    best = dict(C=C_VALUES[0], penalty="l2", scaler=None, clf=None)
    for penalty in PENALTIES:
        for C in C_VALUES:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            clf = LogisticRegression(
                C=C, penalty=penalty, max_iter=2000,
                class_weight="balanced", solver="saga", random_state=42,
            )
            clf.fit(X_tr_s, y_tr)
            proba = clf.predict_proba(scaler.transform(X_va))[:, 1]
            val_auc = roc_auc_score(y_va, proba)
            if val_auc > best_auc:
                best_auc = val_auc
                best.update(C=C, penalty=penalty, scaler=scaler, clf=clf)
    return best["C"], best["penalty"], best["scaler"], best["clf"]


def tune_threshold(clf, scaler, X_va, y_va):
    proba = clf.predict_proba(scaler.transform(X_va))[:, 1]
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_va, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate(clf, scaler, X_te, y_te, threshold):
    proba = clf.predict_proba(scaler.transform(X_te))[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "Accuracy":  accuracy_score(y_te, preds),
        "F1":        f1_score(y_te, preds, zero_division=0),
        "Precision": precision_score(y_te, preds, zero_division=0),
        "Recall":    recall_score(y_te, preds, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_te, proba),
        "AUC-PR":    auc(*precision_recall_curve(y_te, proba)[1::-1]),
        "_proba":    proba,
        "_threshold": threshold,
    }

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

results = []
scalers = {}

for name, Xtr_parts, Xva_parts, Xte_parts in experiments:
    X_tr = np.concatenate(Xtr_parts, axis=1)
    X_va = np.concatenate(Xva_parts, axis=1)
    X_te = np.concatenate(Xte_parts, axis=1)

    print(f"Training: {name}  (dim={X_tr.shape[1]})")

    best_C, best_penalty, scaler, clf = tune_C(X_tr, y_tr, X_va, y_va)
    threshold = tune_threshold(clf, scaler, X_va, y_va)
    metrics   = evaluate(clf, scaler, X_te, y_te, threshold)

    print(f"  penalty={best_penalty}  C={best_C}  thresh={threshold:.2f}"
          f"  F1={metrics['F1']:.4f}  AUC-ROC={metrics['AUC-ROC']:.4f}")

    results.append({
        "Model":   name,
        "dim":     X_tr.shape[1],
        "penalty": best_penalty,
        "C":       best_C,
        **{k: v for k, v in metrics.items() if not k.startswith("_")},
    })
    scalers[name] = (scaler, metrics["_proba"], threshold)

print()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

df = pd.DataFrame(results).set_index("Model")
fmt_cols = ["Accuracy", "F1", "Precision", "Recall", "AUC-ROC", "AUC-PR"]
df_fmt = df[["dim", "penalty", "C"] + fmt_cols].copy()
df_fmt[fmt_cols] = df_fmt[fmt_cols].map(lambda x: f"{x:.4f}")

print("=== Test Set Results ===")
print(df_fmt.to_string())
print()

csv_path = os.path.join(OUT_DIR, "results_ablation_v4.csv")
df[["dim", "penalty", "C"] + fmt_cols].to_csv(csv_path)
print(f"Saved {csv_path}")

# ---------------------------------------------------------------------------
# PCA variance explained vs AUC-ROC plot
# ---------------------------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(9, 5))

ks = PCA_COMPONENTS
hs_aucs  = [df.loc[f"HS-PCA-{k}", "AUC-ROC"] for k in ks]
raw_hs   = df.loc["HS raw (2048-dim)", "AUC-ROC"]
var_exp  = [explained[k] * 100 for k in ks]

ax1.plot(ks, hs_aucs, "o-", color="steelblue", label="HS-PCA")
ax1.axhline(raw_hs, linestyle="--", color="steelblue", alpha=0.5, label=f"HS raw ({raw_hs:.3f})")
ax1.set_xlabel("PCA components (k)")
ax1.set_ylabel("AUC-ROC (test)")
ax1.set_title("PCA on Hidden States: AUC-ROC vs. k")
ax1.legend(loc="lower right", fontsize=9)

ax2 = ax1.twinx()
ax2.plot(ks, var_exp, "^--", color="gray", alpha=0.5, label="Variance explained (%)")
ax2.set_ylabel("Variance explained (%)", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_auc_v4.png"), dpi=150)
plt.close()
print(f"Saved pca_auc_v4.png")

# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 7))
for row in results:
    name = row["Model"]
    _, proba, _ = scalers[name]
    fpr, tpr, _ = roc_curve(y_te, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={row['AUC-ROC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — PCA Ablation (v4)")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curves_v4.png"), dpi=150)
plt.close()
print("Saved roc_curves_v4.png")
