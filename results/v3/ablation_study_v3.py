"""
Ablation study v3 — granular feature breakdown for hallucination detection on PubMedQA.

scalar_features index map (141 total):
  TP   (token prob)            :   0 – 16   (17 features)
  CAT  (context attn traj)     :  17 – 26   (10 features)
  PLC  (per-layer ctx conc)    :  27 – 62   (36 features)
  PLE  (per-layer attn entropy):  63 – 98   (36 features)
  PLH  (per-layer head disagr) :  99 – 134  (36 features)
  MISC (interaction/EOS/text)  : 135 – 140  ( 6 features)

Experiments
-----------
Standalone (7):
  1.  LB   — Lookback Lens baseline      (576-dim,  per_head_ctx flattened)
  2.  TP   — Token-probability features  ( 17-dim)
  3.  PLC  — Per-layer context conc.     ( 36-dim)
  4.  PLE  — Per-layer attn entropy      ( 36-dim)
  5.  PLH  — Per-layer head disagreement ( 36-dim)
  6.  PLA  — All per-layer (PLC+PLE+PLH) (108-dim)
  7.  HS   — Hidden-state probe          (2048-dim)

Stacked on LB (5):
  8.  LB + TP             (593-dim)
  9.  LB + CAT            (586-dim)
  10. LB + PLA            (684-dim)
  11. LB + HS             (2624-dim)
  12. Full system         (2765-dim, LB + all scalar + HS)
"""

import os
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
# Paths — adjust if running from a different working directory
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Build named feature blocks
# scalar_features index map (141 features total):
#   TP   : 0:17    (17) token-probability features
#   CAT  : 17:27   (10) context attention trajectory scalars
#   PLC  : 27:63   (36) per-layer context concentration
#   PLE  : 63:99   (36) per-layer attention entropy
#   PLH  : 99:135  (36) per-layer head disagreement
#   MISC : 135:141  (6) interaction / EOS / text features
# ---------------------------------------------------------------------------

def get_blocks(data):
    lb  = data["per_head_ctx"].reshape(len(data["per_head_ctx"]), -1)  # (N, 576)
    sc  = data["scalar_features"]                                        # (N, 141)
    hs  = data["hidden_states"]                                          # (N, 2048)

    tp   = sc[:,   0:17]   # token-probability
    cat  = sc[:,  17:27]   # context attention trajectory
    plc  = sc[:,  27:63]   # per-layer context concentration
    ple  = sc[:,  63:99]   # per-layer attention entropy
    plh  = sc[:,  99:135]  # per-layer head disagreement
    pla  = sc[:,  27:135]  # all per-layer (PLC + PLE + PLH)

    return dict(lb=lb, tp=tp, cat=cat, plc=plc, ple=ple, plh=plh, pla=pla, sc=sc, hs=hs)

tr = get_blocks(train)
va = get_blocks(val)
te = get_blocks(test)

y_tr = train["y"].astype(int)
y_va = val["y"].astype(int)
y_te = test["y"].astype(int)

feature_names = train["feature_names"]

print("=== Data Shapes ===")
print(f"Train: LB={tr['lb'].shape}, scalar={tr['sc'].shape}, HS={tr['hs'].shape}, y={y_tr.shape}")
print(f"Val:   LB={va['lb'].shape}, scalar={va['sc'].shape}, HS={va['hs'].shape}, y={y_va.shape}")
print(f"Test:  LB={te['lb'].shape}, scalar={te['sc'].shape}, HS={te['hs'].shape}, y={y_te.shape}")
for split_name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
    counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  {split_name} labels: faithful(0)={counts.get(0,0)}, hallucinated(1)={counts.get(1,0)}")
print()

print("Feature block dims:")
for name, arr in tr.items():
    print(f"  {name:5s}: {arr.shape[1]}")
print()

# Verify feature names align with expected index ranges
print("=== Feature name spot-checks ===")
fn = list(feature_names)
print(f"  [0]   {fn[0]}   (expect: mean_token_prob)")
print(f"  [16]  {fn[16]}  (expect: frac_low_confidence)")
print(f"  [17]  {fn[17]}  (expect: mean_context_ratio)")
print(f"  [26]  {fn[26]}  (expect: context_gradient_late_minus_early)")
print(f"  [27]  {fn[27]}  (expect: layer00_context_concentration)")
print(f"  [62]  {fn[62]}  (expect: layer35_context_concentration)")
print(f"  [63]  {fn[63]}  (expect: layer00_attention_entropy)")
print(f"  [98]  {fn[98]}  (expect: layer35_attention_entropy)")
print(f"  [99]  {fn[99]}  (expect: layer00_head_disagreement)")
print(f"  [134] {fn[134]} (expect: layer35_head_disagreement)")
print(f"  [135] {fn[135]} (expect: entropy_context_interaction)")
print(f"  [140] {fn[140]} (expect: answer_word_count)")
print()

# ---------------------------------------------------------------------------
# Experiment definitions  (name, [train_parts], [val_parts], [test_parts])
# ---------------------------------------------------------------------------

experiments = [
    # --- Standalone ---
    ("1.  LB  (Lookback Lens baseline)",
     [tr["lb"]], [va["lb"]], [te["lb"]]),
    ("2.  TP  (Token probability)",
     [tr["tp"]], [va["tp"]], [te["tp"]]),
    ("3.  PLC (Per-layer ctx concentration)",
     [tr["plc"]], [va["plc"]], [te["plc"]]),
    ("4.  PLE (Per-layer attn entropy)",
     [tr["ple"]], [va["ple"]], [te["ple"]]),
    ("5.  PLH (Per-layer head disagreement)",
     [tr["plh"]], [va["plh"]], [te["plh"]]),
    ("6.  PLA (All per-layer: PLC+PLE+PLH)",
     [tr["pla"]], [va["pla"]], [te["pla"]]),
    ("7.  HS  (Hidden-state probe)",
     [tr["hs"]], [va["hs"]], [te["hs"]]),
    # --- Stacked on LB ---
    ("8.  LB + TP",
     [tr["lb"], tr["tp"]], [va["lb"], va["tp"]], [te["lb"], te["tp"]]),
    ("9.  LB + CAT (Context attn trajectory)",
     [tr["lb"], tr["cat"]], [va["lb"], va["cat"]], [te["lb"], te["cat"]]),
    ("10. LB + PLA",
     [tr["lb"], tr["pla"]], [va["lb"], va["pla"]], [te["lb"], te["pla"]]),
    ("11. LB + HS",
     [tr["lb"], tr["hs"]], [va["lb"], va["hs"]], [te["lb"], te["hs"]]),
    ("12. Full system (LB + all scalar + HS)",
     [tr["lb"], tr["sc"], tr["hs"]], [va["lb"], va["sc"], va["hs"]], [te["lb"], te["sc"], te["hs"]]),
]

# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

C_VALUES  = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
PENALTIES = ["l1", "l2"]


def tune_C(X_tr, y_tr, X_va, y_va):
    """Select penalty + C maximising val AUC-ROC."""
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
    """Find threshold maximising F1 on val set."""
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
# Run all experiments
# ---------------------------------------------------------------------------

results  = []
clfs     = {}
scalers  = {}

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
    clfs[name]    = clf
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

csv_path = os.path.join(OUT_DIR, "results_ablation_v3.csv")
df[["dim", "penalty", "C"] + fmt_cols].to_csv(csv_path)
print(f"Saved {csv_path}")

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
ax.set_title("ROC Curves — Hallucination Detection (PubMedQA) — v3")
ax.legend(fontsize=6.5, loc="lower right")
plt.tight_layout()
roc_path = os.path.join(OUT_DIR, "roc_curves_v3.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"Saved {roc_path}")

# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 7))
for row in results:
    name = row["Model"]
    _, proba, _ = scalers[name]
    prec, rec, _ = precision_recall_curve(y_te, proba)
    ax.plot(rec, prec, label=f"{name} (AUC-PR={row['AUC-PR']:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Hallucination Detection (PubMedQA) — v3")
ax.legend(fontsize=6.5, loc="upper right")
plt.tight_layout()
pr_path = os.path.join(OUT_DIR, "pr_curves_v3.png")
plt.savefig(pr_path, dpi=150)
plt.close()
print(f"Saved {pr_path}")

# ---------------------------------------------------------------------------
# Top-20 LB (Lookback Lens) features by coefficient magnitude
# ---------------------------------------------------------------------------

lb_name = "1.  LB  (Lookback Lens baseline)"
lb_clf  = clfs[lb_name]
coefs   = lb_clf.coef_[0]  # (576,)

top20_idx = np.argsort(np.abs(coefs))[::-1][:20]
print("\n=== Top 20 Lookback Lens Features (by |coefficient|) ===")
for rank, idx in enumerate(top20_idx, 1):
    layer = idx // 16
    head  = idx % 16
    print(f"  {rank:2d}. Layer {layer:02d}, Head {head:02d}  coef={coefs[idx]:+.4f}")

# ---------------------------------------------------------------------------
# Save best model (Full system, exp 12)
# ---------------------------------------------------------------------------

best_name   = "12. Full system (LB + all scalar + HS)"
best_clf    = clfs[best_name]
best_scaler, _, best_thresh = scalers[best_name]

pkl_path = os.path.join(OUT_DIR, "best_model_v3.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump({"clf": best_clf, "scaler": best_scaler, "threshold": best_thresh}, f)
print(f"\nSaved {pkl_path} (threshold={best_thresh:.2f})")
