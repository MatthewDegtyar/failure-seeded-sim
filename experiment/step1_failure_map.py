"""Step 1 — Map the failure manifold.

Sample N scenes uniformly across PARAM_BOUNDS, run each grasp, record
success/failure. Check the failure region is sparse and structured by
looking at marginal failure rates and 1-NN purity.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from sim.grasp import (
    ALL_PARAMS,
    PARAM_BOUNDS,
    run_grasp,
    sample_uniform_params,
    params_to_dict,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)


def main(n: int = 2000, seed: int = 1, save_traj_for: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    vecs = sample_uniform_params(rng, n)

    outcomes = np.zeros(n, dtype=np.int8)
    final_z = np.zeros(n, dtype=np.float32)

    t0 = time.time()
    for i, v in enumerate(vecs):
        # Record trajectory for a handful of episodes only (cheaper)
        record = i < save_traj_for
        r = run_grasp(params_to_dict(v), record=record)
        outcomes[i] = 1 if r["success"] else 0
        final_z[i] = r["final_obj_z"]
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n}  elapsed={elapsed:.1f}s  succ_rate={outcomes[:i+1].mean():.3f}")

    elapsed = time.time() - t0
    success_rate = float(outcomes.mean())
    fail_rate = 1.0 - success_rate

    # --- structure check: 1-NN purity ---
    # Normalize each param column to [0,1] before measuring distances.
    norm = vecs.copy().astype(np.float64)
    for j, k in enumerate(ALL_PARAMS):
        lo, hi = PARAM_BOUNDS[k]
        norm[:, j] = (norm[:, j] - lo) / (hi - lo)
    nn = NearestNeighbors(n_neighbors=2).fit(norm)
    _, idx = nn.kneighbors(norm)
    nn_label = outcomes[idx[:, 1]]
    same = (nn_label == outcomes).mean()

    # If outcomes were i.i.d. random, expected same-neighbor rate =
    # p^2 + (1-p)^2. Compare 1-NN purity to that null.
    p = success_rate
    null_purity = p**2 + (1 - p) ** 2
    structure_score = float(same - null_purity)

    # Marginal failure rate per parameter (binned)
    marginals = {}
    for j, k in enumerate(ALL_PARAMS):
        # 5 bins along this axis
        bin_edges = np.linspace(*PARAM_BOUNDS[k], 6)
        bins = np.digitize(vecs[:, j], bin_edges[1:-1])
        per_bin = []
        for b in range(5):
            mask = bins == b
            per_bin.append(float((outcomes[mask] == 0).mean()) if mask.sum() else None)
        marginals[k] = per_bin

    print()
    print(f"Total elapsed: {elapsed:.1f}s  ({elapsed/n*1000:.1f} ms/sim)")
    print(f"Success rate: {success_rate:.3f}  Failure rate: {fail_rate:.3f}")
    print(f"1-NN label agreement: {same:.3f}  (random-baseline={null_purity:.3f})  → structure_score={structure_score:+.3f}")
    print()
    print("Marginal failure rate per param (5 quantile bins from low→high):")
    for k, m in marginals.items():
        line = "  ".join("--" if v is None else f"{v:.2f}" for v in m)
        print(f"  {k:>10s}: [{line}]")

    out = {
        "n": n,
        "seed": seed,
        "success_rate": success_rate,
        "fail_rate": fail_rate,
        "nn_label_agreement": float(same),
        "null_baseline": float(null_purity),
        "structure_score": structure_score,
        "marginal_fail_rates": marginals,
        "elapsed_sec": elapsed,
        "ms_per_sim": elapsed / n * 1000,
    }

    np.savez(
        DATA_DIR / "step1_dataset.npz",
        params=vecs.astype(np.float32),
        outcomes=outcomes,
        final_z=final_z,
        param_names=np.array(ALL_PARAMS),
    )
    with open(DATA_DIR / "step1_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved dataset → {DATA_DIR/'step1_dataset.npz'}")
    print(f"Saved summary → {DATA_DIR/'step1_summary.json'}")
    return out


if __name__ == "__main__":
    main()
