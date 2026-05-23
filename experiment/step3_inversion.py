"""Step 3 — Inversion.

3a: geometric (group A) params given.
3b: CMA-ES over group B to fit the observed trajectory.

Reports per-param inversion error and the recovered posterior
(mean + Gaussian covariance).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sim.grasp import ALL_PARAMS, GROUP_A, GROUP_B, PARAM_BOUNDS
from inversion.sysid import cmaes_invert

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def relative_err(est: np.ndarray, truth: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per-axis absolute error normalized by axis range."""
    return np.abs(est - truth) / (hi - lo)


def main(
    seed: int = 42,
    sigma0: float = 0.30,
    popsize: int = 10,
    maxiter: int = 35,
    obs_noise_pos: float = 0.0,    # std of Gaussian pos noise (meters)
    obs_noise_quat: float = 0.0,   # std added to quat components (then renormalized)
    output_suffix: str = "",
) -> dict:
    obs = np.load(DATA_DIR / "step2_observation.npz")
    true_vec = obs["true_params"]
    true_params = {k: float(true_vec[i]) for i, k in enumerate(ALL_PARAMS)}
    group_a = {k: true_params[k] for k in GROUP_A}
    true_group_b = np.array([true_params[k] for k in GROUP_B])

    pos = obs["traj_pos"].astype(np.float64)
    quat = obs["traj_quat"].astype(np.float64)

    # Optionally add observation noise (used by step 6)
    rng = np.random.default_rng(seed)
    if obs_noise_pos > 0:
        pos = pos + rng.normal(0.0, obs_noise_pos, size=pos.shape)
    if obs_noise_quat > 0:
        quat = quat + rng.normal(0.0, obs_noise_quat, size=quat.shape)
        quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

    print(f"Inversion target (hidden truth, group B):")
    for k, v in zip(GROUP_B, true_group_b):
        print(f"  {k:>10s} = {v:+.4f}")

    t0 = time.time()
    res = cmaes_invert(
        group_a,
        pos,
        quat,
        sigma0=sigma0,
        popsize=popsize,
        maxiter=maxiter,
        seed=seed,
        verbose=False,
    )
    elapsed = time.time() - t0

    lo = np.array([PARAM_BOUNDS[k][0] for k in GROUP_B])
    hi = np.array([PARAM_BOUNDS[k][1] for k in GROUP_B])
    err_mean = relative_err(res.mean, true_group_b, lo, hi)
    err_best = relative_err(res.best, true_group_b, lo, hi)

    print()
    print(f"CMA-ES finished: {res.eval_count} sim evals in {elapsed:.1f}s "
          f"({elapsed/res.eval_count*1000:.0f} ms/eval). "
          f"Best discrepancy: {res.best_discrepancy:.3e}")
    print()
    print("Recovered (CMA-ES posterior mean → best):")
    for i, k in enumerate(GROUP_B):
        std = float(np.sqrt(res.cov[i, i]))
        print(f"  {k:>10s}  truth={true_group_b[i]:+.4f}  "
              f"mean={res.mean[i]:+.4f} (±{std:.4f})  best={res.best[i]:+.4f}")
    print()
    print("Inversion error (|est - truth| / axis_range):")
    for i, k in enumerate(GROUP_B):
        print(f"  {k:>10s}  mean_err={err_mean[i]:.3f}  best_err={err_best[i]:.3f}")

    overall_err = float(err_mean.mean())
    print(f"\nOverall mean relative error (group B): {overall_err:.3f}")

    out = {
        "seed": seed, "popsize": popsize, "maxiter": maxiter, "sigma0": sigma0,
        "obs_noise_pos": obs_noise_pos, "obs_noise_quat": obs_noise_quat,
        "group_b_names": list(GROUP_B),
        "truth": [float(x) for x in true_group_b],
        "mean": [float(x) for x in res.mean],
        "best": [float(x) for x in res.best],
        "cov": res.cov.tolist(),
        "best_discrepancy": float(res.best_discrepancy),
        "err_mean_per_axis": [float(x) for x in err_mean],
        "err_best_per_axis": [float(x) for x in err_best],
        "overall_mean_relative_error": overall_err,
        "eval_count": int(res.eval_count),
        "elapsed_sec": elapsed,
        "history_best": [float(x) for x in res.history_best],
    }

    # Build an ABC-style posterior sample from the lowest-discrepancy
    # visited points. This honestly captures the degeneracy ridge.
    sorted_idx = np.argsort(res.all_discrepancies)
    keep_n = min(200, max(40, len(sorted_idx) // 6))
    posterior_pts = res.all_points[sorted_idx[:keep_n]]
    posterior_disc = res.all_discrepancies[sorted_idx[:keep_n]]
    pmean = posterior_pts.mean(axis=0)
    pstd = posterior_pts.std(axis=0)
    print(f"\nABC posterior ({keep_n} lowest-discrepancy visits):")
    for i, k in enumerate(GROUP_B):
        print(f"  {k:>10s}  truth={true_group_b[i]:+.4f}  "
              f"abc_mean={pmean[i]:+.4f} (±{pstd[i]:.4f})")

    # Also report whether truth lies within the posterior support
    in_support = []
    for i in range(len(GROUP_B)):
        lo_i, hi_i = posterior_pts[:, i].min(), posterior_pts[:, i].max()
        in_support.append(bool(lo_i <= true_group_b[i] <= hi_i))
    print(f"Truth within posterior support per axis: {in_support}")
    posterior_err = relative_err(pmean, true_group_b, lo, hi)
    out["posterior_n"] = int(keep_n)
    out["posterior_mean"] = [float(x) for x in pmean]
    out["posterior_std"]  = [float(x) for x in pstd]
    out["posterior_err_per_axis"] = [float(x) for x in posterior_err]
    out["posterior_overall_mean_relative_error"] = float(posterior_err.mean())
    out["truth_in_posterior_support"] = in_support
    print(f"ABC posterior mean relative error: {float(posterior_err.mean()):.3f}  "
          f"(per-axis: {posterior_err.round(3).tolist()})")

    suffix = output_suffix or ""
    np.savez(
        DATA_DIR / f"step3_posterior{suffix}.npz",
        mean=res.mean, best=res.best, cov=res.cov,
        truth=true_group_b,
        param_names=np.array(GROUP_B),
        posterior_points=posterior_pts,
        posterior_discrepancies=posterior_disc,
        all_points=res.all_points,
        all_discrepancies=res.all_discrepancies,
    )
    with open(DATA_DIR / f"step3_summary{suffix}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    main()
