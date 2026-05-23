"""Step 6 — Noise robustness sweep.

Repeat Step 3 with increasing Gaussian noise on the observed trajectory
positions. For each noise level, re-explode regime B (regime A and C
do not depend on the inversion), train classifiers, and report
BAcc(B), BAcc(A) margin, inversion error.

Step 6 is descriptive (no pre-registered threshold).
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
from inversion.sysid import cmaes_invert
from experiment.step4_neighborhood import (
    WINDOW_FRAC,
    TEST_WINDOW_FRAC,
    GROUP_A_JITTER,
    sample_group_a,
    sample_group_b_window,
    build_full_params,
    simulate_batch,
    stratified_test_set,
    axis_lo_hi,
    clip_to_bounds,
)
from experiment.step5_payoff import CLASSIFIER_KWARGS, CLASSIFIER_SEEDS

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def invert_with_noise(obs_pos, obs_quat, group_a, noise_sigma_pos, seed):
    rng = np.random.default_rng(seed)
    pos = obs_pos.copy()
    if noise_sigma_pos > 0:
        pos = pos + rng.normal(0.0, noise_sigma_pos, size=pos.shape)
    res = cmaes_invert(
        group_a, pos, obs_quat,
        sigma0=0.30, popsize=10, maxiter=35, seed=seed, verbose=False,
    )
    # ABC posterior
    sorted_idx = np.argsort(res.all_discrepancies)
    keep_n = min(200, max(40, len(sorted_idx) // 6))
    posterior_pts = res.all_points[sorted_idx[:keep_n]]
    return posterior_pts, res


def evaluate_regime(params_train, outcomes_train, X_test, y_test):
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


def main(
    noise_levels: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01),
    n_per_regime: int = 1500,
    seed: int = 200,
) -> dict:
    obs = np.load(DATA_DIR / "step2_observation.npz")
    true_vec = obs["true_params"]
    true_params = {k: float(true_vec[i]) for i, k in enumerate(ALL_PARAMS)}
    group_a = {k: true_params[k] for k in GROUP_A}
    truth_b = np.array([true_params[k] for k in GROUP_B])

    obs_pos = obs["traj_pos"].astype(np.float64)
    obs_quat = obs["traj_quat"].astype(np.float64)

    lo_b, hi_b = axis_lo_hi(list(GROUP_B))

    # Regime A and the test set are noise-independent; share across sweep.
    ds4 = np.load(DATA_DIR / "step4_training_sets.npz")
    params_A = ds4["params_A"]; outcomes_A = ds4["outcomes_A"]
    test_X = ds4["test_params"]; test_y = ds4["test_outcomes"]

    # Regime C is also noise-independent (oracle uses truth).
    params_C = ds4["params_C"]; outcomes_C = ds4["outcomes_C"]

    bacc_A, _ = evaluate_regime(params_A, outcomes_A, test_X, test_y)
    bacc_C, _ = evaluate_regime(params_C, outcomes_C, test_X, test_y)

    print(f"Reference: BAcc(A)={bacc_A:.4f}  BAcc(C)={bacc_C:.4f}")
    print()

    results = []
    rng = np.random.default_rng(seed)
    for noise in noise_levels:
        t0 = time.time()
        print(f"--- noise σ_pos = {noise*1000:.2f} mm ---")
        post_pts, _res = invert_with_noise(obs_pos, obs_quat, group_a, noise, seed)
        post_mean = post_pts.mean(axis=0)
        # Inversion error vs truth (overall mean relative)
        rel_err = np.abs(post_mean - truth_b) / (hi_b - lo_b)
        inv_err_mean = float(rel_err.mean())
        print(f"  posterior mean = {post_mean.round(4).tolist()}  truth = {truth_b.round(4).tolist()}  rel_err = {rel_err.round(3).tolist()}")
        print(f"  overall inv. error = {inv_err_mean:.3f}")

        # Resample regime B around this posterior
        ga_B = sample_group_a(group_a, n_per_regime, rng)
        gb_B = sample_group_b_window(post_mean, WINDOW_FRAC, n_per_regime, rng)
        params_B = build_full_params(ga_B, gb_B)
        outcomes_B = simulate_batch(params_B, f"B@{noise*1000:.1f}mm")

        bacc_B, bacc_B_std = evaluate_regime(params_B, outcomes_B, test_X, test_y)
        margin = bacc_B - bacc_A
        oracle_gap = bacc_C - bacc_B
        print(f"  BAcc(B)={bacc_B:.4f}±{bacc_B_std:.4f}  margin(B-A)={margin:+.4f}  oracle gap={oracle_gap:+.4f}")
        print(f"  elapsed {time.time()-t0:.1f}s")
        print()

        results.append({
            "noise_sigma_pos_m": float(noise),
            "posterior_mean": post_mean.tolist(),
            "inversion_error": inv_err_mean,
            "per_axis_err": rel_err.tolist(),
            "succ_rate_B": float(outcomes_B.mean()),
            "bacc_B": bacc_B,
            "bacc_B_std": bacc_B_std,
            "margin_B_minus_A": float(margin),
            "oracle_gap_C_minus_B": float(oracle_gap),
        })

    out = {
        "bacc_A_reference": bacc_A,
        "bacc_C_reference": bacc_C,
        "truth_group_b": truth_b.tolist(),
        "results": results,
    }
    with open(DATA_DIR / "step6_noise_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {DATA_DIR/'step6_noise_sweep.json'}")
    return out


if __name__ == "__main__":
    main()
