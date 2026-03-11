"""
Ablation study for hallucination detection on PubMedQA.

Classifiers:
  1. Lookback Lens baseline      — per_head_ctx flattened (N, 576)
  2. Scalar features only        — scalar_features (N, 141)
  3. Hidden states only          — hidden_states (N, 2048)
  4. Lookback Lens + Scalar      — (N, 717)
  5. Lookback Lens + Hidden      — (N, 2624)
  6. Full system (ours)          — all three (N, 2765)
"""

import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_split(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

train = load_split("./data/features/features_v2_train_all.npz")
val   = load_split("./data/features/features_v2_val.npz")
test  = load_split("./data/features/features_v2_test.npz")

# Build feature blocks
def get_blocks(data):
    lookback = data["per_head_ctx"].reshape(len(data["per_head_ctx"]), -1)  # (N, 576)
    scalar   = data["scalar_features"]                                       # (N, 141)
    hidden   = data["hidden_states"]                                         # (N, 2048)
    return lookback, scalar, hidden

Xl_tr, Xs_tr, Xh_tr = get_blocks(train)
Xl_va, Xs_va, Xh_va = get_blocks(val)
Xl_te, Xs_te, Xh_te = get_blocks(test)

y_tr = train["y"].astype(int)
y_va = val["y"].astype(int)
y_te = test["y"].astype(int)

feature_names = train["feature_names"]

print("=== Data Shapes ===")
print(f"Train: lookback={Xl_tr.shape}, scalar={Xs_tr.shape}, hidden={Xh_tr.shape}, y={y_tr.shape}")
print(f"Val:   lookback={Xl_va.shape}, scalar={Xs_va.shape}, hidden={Xh_va.shape}, y={y_va.shape}")
print(f"Test:  lookback={Xl_te.shape}, scalar={Xs_te.shape}, hidden={Xh_te.shape}, y={y_te.shape}")
for split_name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  {split_name} labels: faithful(0)={counts.get(0,0)}, hallucinated(1)={counts.get(1,0)}")
print()

# ---------------------------------------------------------------------------
# Define experiments
# ---------------------------------------------------------------------------

experiments = [
    ("1. Lookback Lens (baseline)",  [Xl_tr],             [Xl_va],             [Xl_te]),
    ("2. Scalar features only",      [Xs_tr],             [Xs_va],             [Xs_te]),
    ("3. Hidden states only",        [Xh_tr],             [Xh_va],             [Xh_te]),
    ("4. Lookback + Scalar",         [Xl_tr, Xs_tr],      [Xl_va, Xs_va],      [Xl_te, Xs_te]),
    ("5. Lookback + Hidden",         [Xl_tr, Xh_tr],      [Xl_va, Xh_va],      [Xl_te, Xh_te]),
    ("6. Full system (ours)",        [Xl_tr, Xs_tr, Xh_tr],[Xl_va, Xs_va, Xh_va],[Xl_te, Xs_te, Xh_te]),
]

# ---------------------------------------------------------------------------
# Train, threshold-tune, evaluate
# ---------------------------------------------------------------------------

def tune_threshold(clf, scaler, X_val, y_val):
    """Find threshold maximizing F1 on val set."""
    proba = clf.predict_proba(scaler.transform(X_val))[:, 1]
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh

def evaluate(clf, scaler, X_test, y_test, threshold):
    proba = clf.predict_proba(scaler.transform(X_test))[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "Accuracy":  accuracy_score(y_test, preds),
        "F1":        f1_score(y_test, preds, zero_division=0),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall":    recall_score(y_test, preds, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_test, proba),
        "AUC-PR":    auc(*precision_recall_curve(y_test, proba)[1::-1]),
        "_proba":    proba,
        "_threshold": threshold,
    }

C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]

def tune_C(X_tr, y_tr, X_va, y_va):
    """Select C maximizing val AUC-ROC. Returns (best_C, scaler, clf)."""
    best_auc, best_C, best_scaler, best_clf = 0.0, C_VALUES[0], None, None
    for C in C_VALUES:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        clf = LogisticRegression(
            C=C, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )
        clf.fit(X_tr_s, y_tr)
        proba = clf.predict_proba(scaler.transform(X_va))[:, 1]
        auc_val = roc_auc_score(y_va, proba)
        if auc_val > best_auc:
            best_auc, best_C, best_scaler, best_clf = auc_val, C, scaler, clf
    return best_C, best_scaler, best_clf

results   = []
clfs      = {}
scalers   = {}

for name, Xtr_parts, Xva_parts, Xte_parts in experiments:
    X_tr = np.concatenate(Xtr_parts, axis=1)
    X_va = np.concatenate(Xva_parts, axis=1)
    X_te = np.concatenate(Xte_parts, axis=1)

    print(f"Training: {name}  (feature_dim={X_tr.shape[1]})")

    best_C, scaler, clf = tune_C(X_tr, y_tr, X_va, y_va)

    threshold = tune_threshold(clf, scaler, X_va, y_va)
    metrics   = evaluate(clf, scaler, X_te, y_te, threshold)

    print(f"  best_C={best_C}  threshold={threshold:.2f}  F1={metrics['F1']:.4f}  AUC-ROC={metrics['AUC-ROC']:.4f}")

    results.append({"Model": name, "C": best_C, **{k: v for k, v in metrics.items() if not k.startswith("_")}})
    clfs[name]    = clf
    scalers[name] = (scaler, metrics["_proba"], threshold)

print()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

df = pd.DataFrame(results).set_index("Model")
fmt_cols = ["Accuracy", "F1", "Precision", "Recall", "AUC-ROC", "AUC-PR"]
df_fmt = df[["C"] + fmt_cols].copy()
df_fmt[fmt_cols] = df_fmt[fmt_cols].map(lambda x: f"{x:.4f}")

print("=== Test Set Results ===")
print(df_fmt.to_string())
print()

df[["C"] + fmt_cols].to_csv("results_ablation.csv")
print("Saved results_ablation.csv")

# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
for row in results:
    name = row["Model"]
    _, proba, _ = scalers[name]
    fpr, tpr, _ = roc_curve(y_te, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={row['AUC-ROC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Hallucination Detection (PubMedQA)")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()
print("Saved roc_curves.png")

# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
for row in results:
    name = row["Model"]
    _, proba, _ = scalers[name]
    prec, rec, _ = precision_recall_curve(y_te, proba)
    ax.plot(rec, prec, label=f"{name} (AUC-PR={row['AUC-PR']:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Hallucination Detection (PubMedQA)")
ax.legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig("pr_curves.png", dpi=150)
plt.close()
print("Saved pr_curves.png")

# ---------------------------------------------------------------------------
# Top-20 Lookback Lens features by coefficient magnitude
# ---------------------------------------------------------------------------

lb_name = "1. Lookback Lens (baseline)"
lb_clf  = clfs[lb_name]
coefs   = lb_clf.coef_[0]  # (576,)

top20_idx = np.argsort(np.abs(coefs))[::-1][:20]
print("\n=== Top 20 Lookback Lens Features (by |coefficient|) ===")
for rank, idx in enumerate(top20_idx, 1):
    layer = idx // 16
    head  = idx % 16
    print(f"  {rank:2d}. Layer {layer:02d}, Head {head:02d}  coef={coefs[idx]:+.4f}")

# ---------------------------------------------------------------------------
# Save best model (Full system)
# ---------------------------------------------------------------------------

best_name = "6. Full system (ours)"
best_clf  = clfs[best_name]
best_scaler, _, best_thresh = scalers[best_name]

with open("best_model.pkl", "wb") as f:
    pickle.dump({"clf": best_clf, "scaler": best_scaler, "threshold": best_thresh}, f)
print(f"\nSaved best_model.pkl (threshold={best_thresh:.2f})")
