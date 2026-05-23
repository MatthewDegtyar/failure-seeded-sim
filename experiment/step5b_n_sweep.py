"""Step 5b (exploratory, NOT pre-registered) — N sweep.

Question: at what training-budget N does local failure-seeding (B) start
to beat global domain randomization (A)? Step 5 found A wins at N=1500.

We resample each regime at smaller N and rerun the classifier comparison.
Test set is unchanged (the held-out stratified set from Step 4).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from sim.grasp import (
    ALL_PARAMS,
    GROUP_A,
    GROUP_B,
    PARAM_BOUNDS,
    params_to_dict,
    run_grasp,
)
from experiment.step4_neighborhood import (
    WINDOW_FRAC,
    sample_group_a,
    sample_group_b_window,
    build_full_params,
    simulate_batch,
)
from experiment.step5_payoff import CLASSIFIER_KWARGS, CLASSIFIER_SEEDS

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def evaluate(params_train, outcomes_train, X_test, y_test):
    baccs = []
    for s in CLASSIFIER_SEEDS:
        if len(np.unique(outcomes_train)) < 2:
            pred = np.full_like(y_test, int(outcomes_train[0]))
        else:
            clf = RandomForestClassifier(random_state=s, **CLASSIFIER_KWARGS)
            clf.fit(params_train, outcomes_train)
            pred = clf.predict(X_test)
        baccs.append(balanced_accuracy_score(y_test, pred))
    return float(np.mean(baccs)), float(np.std(baccs))


def main(n_values: tuple[int, ...] = (50, 100, 200, 500, 1000, 1500), seed: int = 300) -> dict:
    ds4 = np.load(DATA_DIR / "step4_training_sets.npz")
    test_X = ds4["test_params"]; test_y = ds4["test_outcomes"]
    posterior_center = ds4["posterior_center"]
    truth_b = ds4["truth"]

    client = json.load(open(DATA_DIR / "step2_client.json"))
    group_a_center = client["true_group_A"]

    rng = np.random.default_rng(seed)
    results = []
    for n in n_values:
        print(f"\n=== N = {n} ===")
        # Sample A
        params_A = np.zeros((n, len(ALL_PARAMS)))
        for j, k in enumerate(ALL_PARAMS):
            lo, hi = PARAM_BOUNDS[k]
            params_A[:, j] = rng.uniform(lo, hi, size=n)
        outcomes_A = simulate_batch(params_A, "A")
        # Sample B (around posterior center)
        ga_B = sample_group_a(group_a_center, n, rng)
        gb_B = sample_group_b_window(posterior_center, WINDOW_FRAC, n, rng)
        params_B = build_full_params(ga_B, gb_B)
        outcomes_B = simulate_batch(params_B, "B")
        # Sample C (around truth)
        ga_C = sample_group_a(group_a_center, n, rng)
        gb_C = sample_group_b_window(truth_b, WINDOW_FRAC, n, rng)
        params_C = build_full_params(ga_C, gb_C)
        outcomes_C = simulate_batch(params_C, "C")

        bacc_A, std_A = evaluate(params_A, outcomes_A, test_X, test_y)
        bacc_B, std_B = evaluate(params_B, outcomes_B, test_X, test_y)
        bacc_C, std_C = evaluate(params_C, outcomes_C, test_X, test_y)

        print(f"  A: BAcc={bacc_A:.4f}±{std_A:.4f}  (succ_rate={outcomes_A.mean():.3f})")
        print(f"  B: BAcc={bacc_B:.4f}±{std_B:.4f}  (succ_rate={outcomes_B.mean():.3f})")
        print(f"  C: BAcc={bacc_C:.4f}±{std_C:.4f}  (succ_rate={outcomes_C.mean():.3f})")
        print(f"  margin(B-A)={bacc_B-bacc_A:+.4f}  oracle_gap(C-B)={bacc_C-bacc_B:+.4f}")

        results.append({
            "n": n,
            "bacc_A": bacc_A, "bacc_B": bacc_B, "bacc_C": bacc_C,
            "std_A": std_A, "std_B": std_B, "std_C": std_C,
            "succ_rate_A": float(outcomes_A.mean()),
            "succ_rate_B": float(outcomes_B.mean()),
            "succ_rate_C": float(outcomes_C.mean()),
        })

    out = {"results": results, "test_n": int(len(test_y))}
    with open(DATA_DIR / "step5b_n_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {DATA_DIR/'step5b_n_sweep.json'}")
    return out


if __name__ == "__main__":
    main()
