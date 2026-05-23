"""Step 5 — Three-regime payoff test.

Train RandomForest failure-predictors on regimes A, B, C and evaluate
on the held-out stratified test set produced by Step 4. Compare to
the thresholds pre-registered in preregistration.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Pre-registered settings (DO NOT EDIT after Step-5 results are known)
CLASSIFIER_KWARGS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
)
CLASSIFIER_SEEDS = [0, 1, 2, 3, 4]
THRESH_VALIDATE_MARGIN = 0.10  # B - A
THRESH_ORACLE_GAP      = 0.05  # C - B max for "close to ceiling"
THRESH_PARTIAL_MARGIN  = 0.05


def train_eval(X_train, y_train, X_test, y_test, seed):
    # Guard: if training data has only one class, the classifier
    # collapses to predicting that class; sklearn raises on AUC.
    if len(np.unique(y_train)) < 2:
        only = int(y_train[0])
        pred = np.full_like(y_test, only)
        return {
            "bacc": float(balanced_accuracy_score(y_test, pred)),
            "acc": float(accuracy_score(y_test, pred)),
            "f1_fail": float(f1_score(y_test, pred, pos_label=0, zero_division=0)),
            "recall_fail": float(((pred == 0) & (y_test == 0)).sum() / max(1, (y_test == 0).sum())),
            "recall_succ": float(((pred == 1) & (y_test == 1)).sum() / max(1, (y_test == 1).sum())),
            "auc": float("nan"),
        }
    clf = RandomForestClassifier(random_state=seed, **CLASSIFIER_KWARGS)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    cls = list(clf.classes_)
    i_success = cls.index(1) if 1 in cls else 0
    try:
        auc = float(roc_auc_score(y_test, proba[:, i_success]))
    except ValueError:
        auc = float("nan")
    return {
        "bacc": float(balanced_accuracy_score(y_test, pred)),
        "acc": float(accuracy_score(y_test, pred)),
        "f1_fail": float(f1_score(y_test, pred, pos_label=0, zero_division=0)),
        "recall_fail": float(((pred == 0) & (y_test == 0)).sum() / max(1, (y_test == 0).sum())),
        "recall_succ": float(((pred == 1) & (y_test == 1)).sum() / max(1, (y_test == 1).sum())),
        "auc": auc,
    }


def main() -> dict:
    ds = np.load(DATA_DIR / "step4_training_sets.npz")
    test_X = ds["test_params"]
    test_y = ds["test_outcomes"]
    print(f"Test set: N={len(test_y)}, fail_rate={1-test_y.mean():.3f}")

    regimes = {
        "A": (ds["params_A"], ds["outcomes_A"]),
        "B": (ds["params_B"], ds["outcomes_B"]),
        "C": (ds["params_C"], ds["outcomes_C"]),
    }

    per_seed = {r: [] for r in regimes}
    for seed in CLASSIFIER_SEEDS:
        for r, (X, y) in regimes.items():
            per_seed[r].append(train_eval(X, y, test_X, test_y, seed))

    # Aggregate
    agg = {}
    for r, runs in per_seed.items():
        keys = runs[0].keys()
        agg[r] = {k: float(np.mean([run[k] for run in runs])) for k in keys}
        agg[r]["bacc_std"] = float(np.std([run["bacc"] for run in runs]))

    print()
    print(f"{'regime':>8s}  {'BAcc':>8s} ± {'std':>5s}  {'Acc':>6s}  {'F1_fail':>8s}  {'AUC':>6s}  {'R_fail':>6s}  {'R_succ':>6s}")
    for r in ["A", "B", "C"]:
        a = agg[r]
        print(f"{r:>8s}  {a['bacc']:.4f} ± {a['bacc_std']:.4f}  {a['acc']:.4f}  "
              f"{a['f1_fail']:.4f}  {a['auc']:.4f}  {a['recall_fail']:.4f}  {a['recall_succ']:.4f}")

    margin_BA = agg["B"]["bacc"] - agg["A"]["bacc"]
    oracle_gap = agg["C"]["bacc"] - agg["B"]["bacc"]
    print()
    print(f"BAcc(B) - BAcc(A) = {margin_BA:+.4f}  (pre-registered validate threshold: ≥ {THRESH_VALIDATE_MARGIN:.2f})")
    print(f"BAcc(C) - BAcc(B) = {oracle_gap:+.4f}  (pre-registered ceiling gap: ≤ {THRESH_ORACLE_GAP:.2f})")

    if margin_BA >= THRESH_VALIDATE_MARGIN and oracle_gap <= THRESH_ORACLE_GAP:
        verdict = "VALIDATED"
    elif margin_BA >= THRESH_PARTIAL_MARGIN:
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "REFUTED"
    print(f"\nVerdict: {verdict}")

    out = {
        "per_regime": agg,
        "per_seed": {r: per_seed[r] for r in regimes},
        "margin_B_minus_A": margin_BA,
        "oracle_gap_C_minus_B": oracle_gap,
        "verdict": verdict,
        "thresholds": {
            "validate_margin_B_minus_A": THRESH_VALIDATE_MARGIN,
            "oracle_gap_C_minus_B_max": THRESH_ORACLE_GAP,
            "partial_margin_B_minus_A": THRESH_PARTIAL_MARGIN,
        },
        "n_test": int(len(test_y)),
        "test_fail_rate": float(1 - test_y.mean()),
        "classifier_seeds": CLASSIFIER_SEEDS,
    }
    with open(DATA_DIR / "step5_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {DATA_DIR/'step5_summary.json'}")
    return out


if __name__ == "__main__":
    main()
