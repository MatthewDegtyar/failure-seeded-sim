"""System identification via CMA-ES.

Group A (geometric) params are handed in as known. We search only over
group B (friction, mass, com_x), matching the simulated trajectory
against the observed trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass

import cma
import numpy as np

from sim.grasp import (
    GROUP_A,
    GROUP_B,
    PARAM_BOUNDS,
    run_grasp,
)


def quat_angular_dist(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angular distance between two quaternions (per row), in radians."""
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def trajectory_discrepancy(
    traj_a_pos: np.ndarray,
    traj_a_quat: np.ndarray,
    traj_b_pos: np.ndarray,
    traj_b_quat: np.ndarray,
    pos_weight: float = 1.0,
    rot_weight: float = 0.005,  # rad^2 ≈ small numbers vs m^2 of pos
) -> float:
    """Mean squared discrepancy across trajectory frames.

    Includes both position and orientation; orientation contribution
    is downweighted because grasping rotation angles can be large.
    """
    n = min(len(traj_a_pos), len(traj_b_pos))
    pos_err = np.sum((traj_a_pos[:n] - traj_b_pos[:n]) ** 2, axis=1)
    rot_err = quat_angular_dist(traj_a_quat[:n], traj_b_quat[:n]) ** 2
    return float(pos_weight * pos_err.mean() + rot_weight * rot_err.mean())


@dataclass
class InversionResult:
    mean: np.ndarray            # CMA-ES final mean (group B order)
    best: np.ndarray            # best-ever point seen
    cov: np.ndarray             # CMA-ES final C matrix (group B order)
    sigma: float                # final sigma
    eval_count: int
    best_discrepancy: float
    history_best: list          # best discrepancy per generation
    group_b_names: list
    all_points: np.ndarray      # (E, 3) every group-B point evaluated
    all_discrepancies: np.ndarray  # (E,) corresponding fitnesses


def _params_dict(group_a: dict, group_b_vec: np.ndarray) -> dict:
    d = dict(group_a)
    for k, v in zip(GROUP_B, group_b_vec):
        d[k] = float(v)
    return d


def cmaes_invert(
    group_a: dict,
    observed_pos: np.ndarray,
    observed_quat: np.ndarray,
    sigma0: float = 0.30,
    popsize: int = 10,
    maxiter: int = 35,
    seed: int = 0,
    verbose: bool = False,
    noise_sigma_pos: float = 0.0,  # added at evaluation time to candidate? no — added to observation before calling
) -> InversionResult:
    """Run CMA-ES to recover group-B params explaining the observation.

    Optimization is performed in NORMALIZED group-B space (each axis
    rescaled to [0,1]); we de-normalize before simulating.
    """
    lo = np.array([PARAM_BOUNDS[k][0] for k in GROUP_B])
    hi = np.array([PARAM_BOUNDS[k][1] for k in GROUP_B])

    def denorm(x_norm: np.ndarray) -> np.ndarray:
        return lo + np.clip(x_norm, 0.0, 1.0) * (hi - lo)

    def f(x_norm: np.ndarray) -> float:
        gb = denorm(np.asarray(x_norm))
        p = _params_dict(group_a, gb)
        r = run_grasp(p, record=True)
        return trajectory_discrepancy(
            observed_pos, observed_quat, r["traj_pos"], r["traj_quat"]
        )

    all_points_norm: list[np.ndarray] = []
    all_fs: list[float] = []

    history_best: list[float] = []
    best_x_norm: np.ndarray | None = None
    best_f = float("inf")

    # Run an ensemble of CMA-ES from multiple starts so the visited
    # points cover the degeneracy ridge, not just one local mode.
    rng = np.random.default_rng(seed)
    n_restarts = 4
    starts = [np.full(len(GROUP_B), 0.5)]
    for _ in range(n_restarts - 1):
        starts.append(rng.uniform(0.15, 0.85, size=len(GROUP_B)))

    final_es = None
    for r_idx, x0 in enumerate(starts):
        opts = {
            "bounds": [[0.0] * len(GROUP_B), [1.0] * len(GROUP_B)],
            "maxiter": maxiter,
            "popsize": popsize,
            "seed": seed + r_idx,
            "verbose": -9 if not verbose else 1,
        }
        es = cma.CMAEvolutionStrategy(np.asarray(x0), sigma0, opts)
        while not es.stop():
            xs = es.ask()
            fs = [f(np.asarray(x)) for x in xs]
            es.tell(xs, fs)
            for x, fv in zip(xs, fs):
                all_points_norm.append(np.asarray(x).copy())
                all_fs.append(fv)
            history_best.append(min(fs))
            i_best = int(np.argmin(fs))
            if fs[i_best] < best_f:
                best_f = fs[i_best]
                best_x_norm = np.asarray(xs[i_best]).copy()
        # Keep the last es for its mean/cov (it should match best)
        final_es = es

    mean_norm = np.asarray(final_es.mean)
    C_norm = final_es.sigma ** 2 * np.asarray(final_es.C)
    scale = (hi - lo)
    C_b = (scale[:, None] * C_norm * scale[None, :])
    mean_b = denorm(mean_norm)
    best_b = denorm(best_x_norm) if best_x_norm is not None else mean_b
    sigma_b = float(final_es.sigma * np.exp(np.log(scale).mean()))

    all_points_b = denorm(np.asarray(all_points_norm))
    all_fs_arr = np.asarray(all_fs)

    return InversionResult(
        mean=mean_b,
        best=best_b,
        cov=C_b,
        sigma=sigma_b,
        eval_count=int(len(all_fs)),
        best_discrepancy=best_f,
        history_best=history_best,
        group_b_names=list(GROUP_B),
        all_points=all_points_b,
        all_discrepancies=all_fs_arr,
    )
