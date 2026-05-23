"""Step 4 — Explode the neighborhood + assemble training and test sets.

Regimes (equal N each; classifiers will see the same budget):
  A  blind DR     — uniform across full PARAM_BOUNDS
  B  ours         — uniform box around the ABC-posterior mean
  C  oracle       — uniform box around the true hidden params

Group-A jitter is the same across B and C (small calibration window).

The held-out test set is drawn from a window CENTERED ON THE TRUTH,
stratified to be class-balanced (≈50% failures, ≈50% successes) — this
makes the metric sensitive to both kinds of mistakes and prevents a
trivial "always predict fail" baseline from winning.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sim.grasp import (
    ALL_PARAMS,
    GROUP_A,
    GROUP_B,
    PARAM_BOUNDS,
    params_to_dict,
    run_grasp,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

GROUP_A_JITTER = {"obj_x": 0.005, "obj_y": 0.005, "obj_yaw": 0.05}

# Group-B window half-width as a fraction of each axis's range.
# 0.40 ≈ ±44% of friction range, ±16% of mass range, ±8mm of CoM —
# wide enough that the success boundary is reachable on at least one
# side of the truth.
WINDOW_FRAC = 0.40
TEST_WINDOW_FRAC = 0.50  # test window is slightly wider so B isn't 100% in-window
MAX_TEST_SCAN = 25000     # cap on candidate sims when stratifying test set


def axis_lo_hi(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([PARAM_BOUNDS[k][0] for k in names])
    hi = np.array([PARAM_BOUNDS[k][1] for k in names])
    return lo, hi


def clip_to_bounds(arr: np.ndarray, names: list[str]) -> np.ndarray:
    lo, hi = axis_lo_hi(names)
    return np.clip(arr, lo, hi)


def sample_group_a(center: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.zeros((n, len(GROUP_A)))
    for j, k in enumerate(GROUP_A):
        jitter = GROUP_A_JITTER[k]
        lo, hi = PARAM_BOUNDS[k]
        arr[:, j] = np.clip(rng.uniform(center[k] - jitter, center[k] + jitter, size=n), lo, hi)
    return arr


def sample_group_b_window(center: np.ndarray, frac: float, n: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = axis_lo_hi(GROUP_B)
    ranges = hi - lo
    half = ranges * frac
    arr = center[None, :] + rng.uniform(-half, half, size=(n, len(GROUP_B)))
    return clip_to_bounds(arr, list(GROUP_B))


def build_full_params(group_a_arr: np.ndarray, group_b_arr: np.ndarray) -> np.ndarray:
    out = np.zeros((len(group_a_arr), len(ALL_PARAMS)))
    for j, k in enumerate(ALL_PARAMS):
        if k in GROUP_A:
            out[:, j] = group_a_arr[:, GROUP_A.index(k)]
        else:
            out[:, j] = group_b_arr[:, GROUP_B.index(k)]
    return out


def simulate_batch(params_arr: np.ndarray, label: str) -> np.ndarray:
    out = np.zeros(len(params_arr), dtype=np.int8)
    t0 = time.time()
    for i, v in enumerate(params_arr):
        out[i] = 1 if run_grasp(params_to_dict(v), record=False)["success"] else 0
        if (i + 1) % 500 == 0:
            print(f"  [{label}] {i+1}/{len(params_arr)}  "
                  f"elapsed={time.time()-t0:.1f}s  succ_rate={out[:i+1].mean():.3f}")
    return out


def stratified_test_set(
    truth_b: np.ndarray, group_a_center: dict, n_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample scenes around the truth (TEST_WINDOW_FRAC), simulate, and
    keep balanced classes. Returns (params, outcomes).
    """
    successes_params: list[np.ndarray] = []
    failures_params: list[np.ndarray] = []
    scanned = 0
    t0 = time.time()
    while (len(successes_params) < n_per_class or len(failures_params) < n_per_class) \
            and scanned < MAX_TEST_SCAN:
        batch = 200
        gb = sample_group_b_window(truth_b, TEST_WINDOW_FRAC, batch, rng)
        ga = sample_group_a(group_a_center, batch, rng)
        full = build_full_params(ga, gb)
        for v in full:
            scanned += 1
            ok = run_grasp(params_to_dict(v), record=False)["success"]
            if ok and len(successes_params) < n_per_class:
                successes_params.append(v)
            elif (not ok) and len(failures_params) < n_per_class:
                failures_params.append(v)
            if len(successes_params) >= n_per_class and len(failures_params) >= n_per_class:
                break
        elapsed = time.time() - t0
        print(f"  [test] scanned={scanned}  fails={len(failures_params)}  "
              f"succ={len(successes_params)}  elapsed={elapsed:.1f}s")

    if len(successes_params) < n_per_class:
        print(f"  WARNING: only found {len(successes_params)} successes after {scanned} scans")
    if len(failures_params) < n_per_class:
        print(f"  WARNING: only found {len(failures_params)} failures after {scanned} scans")

    params = np.array(successes_params + failures_params)
    outs = np.array([1] * len(successes_params) + [0] * len(failures_params), dtype=np.int8)
    perm = rng.permutation(len(params))
    return params[perm], outs[perm]


def main(n_per_regime: int = 1500, n_test_per_class: int = 400, seed: int = 100) -> dict:
    rng = np.random.default_rng(seed)

    post = np.load(DATA_DIR / "step3_posterior.npz")
    posterior_pts = post["posterior_points"]
    truth_b = post["truth"]
    posterior_center = posterior_pts.mean(axis=0)

    client = json.load(open(DATA_DIR / "step2_client.json"))
    group_a_center = client["true_group_A"]

    # --- Training sets ---
    print("\n=== Regime B (ours): window around posterior mean ===")
    ga_B = sample_group_a(group_a_center, n_per_regime, rng)
    gb_B = sample_group_b_window(posterior_center, WINDOW_FRAC, n_per_regime, rng)
    params_B = build_full_params(ga_B, gb_B)
    outcomes_B = simulate_batch(params_B, "B")

    print("\n=== Regime C (oracle): window around truth ===")
    ga_C = sample_group_a(group_a_center, n_per_regime, rng)
    gb_C = sample_group_b_window(truth_b, WINDOW_FRAC, n_per_regime, rng)
    params_C = build_full_params(ga_C, gb_C)
    outcomes_C = simulate_batch(params_C, "C")

    print("\n=== Regime A (DR): uniform over full bounds ===")
    params_A = np.zeros((n_per_regime, len(ALL_PARAMS)))
    for j, k in enumerate(ALL_PARAMS):
        lo, hi = PARAM_BOUNDS[k]
        params_A[:, j] = rng.uniform(lo, hi, size=n_per_regime)
    outcomes_A = simulate_batch(params_A, "A")

    # --- Test set (stratified, around truth) ---
    print(f"\n=== Stratified test set: {n_test_per_class} per class, around truth ===")
    test_params, test_outcomes = stratified_test_set(
        truth_b, group_a_center, n_test_per_class, rng,
    )

    print()
    for name, p, o in [("A", params_A, outcomes_A),
                       ("B", params_B, outcomes_B),
                       ("C", params_C, outcomes_C),
                       ("TEST", test_params, test_outcomes)]:
        print(f"{name:>4s}: N={len(p)}, success_rate={float(o.mean()):.3f}")

    np.savez(
        DATA_DIR / "step4_training_sets.npz",
        params_A=params_A.astype(np.float32), outcomes_A=outcomes_A,
        params_B=params_B.astype(np.float32), outcomes_B=outcomes_B,
        params_C=params_C.astype(np.float32), outcomes_C=outcomes_C,
        test_params=test_params.astype(np.float32), test_outcomes=test_outcomes,
        param_names=np.array(ALL_PARAMS),
        window_frac=np.float32(WINDOW_FRAC),
        test_window_frac=np.float32(TEST_WINDOW_FRAC),
        posterior_center=posterior_center.astype(np.float32),
        truth=truth_b.astype(np.float32),
    )

    out = {
        "n_per_regime": n_per_regime, "n_test_per_class": n_test_per_class,
        "seed": seed, "window_frac": WINDOW_FRAC, "test_window_frac": TEST_WINDOW_FRAC,
        "succ_rate_A": float(outcomes_A.mean()),
        "succ_rate_B": float(outcomes_B.mean()),
        "succ_rate_C": float(outcomes_C.mean()),
        "succ_rate_test": float(test_outcomes.mean()),
        "test_n": int(len(test_outcomes)),
    }
    with open(DATA_DIR / "step4_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {DATA_DIR/'step4_training_sets.npz'}")
    return out


if __name__ == "__main__":
    main()
