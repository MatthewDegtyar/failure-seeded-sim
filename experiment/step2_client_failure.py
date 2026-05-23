"""Step 2 — Designate one 'client failure'.

Pick a failed configuration from the Step-1 dataset, treat its parameters
as hidden ground truth, and freeze its observed trajectory as the
multi-view observation the inversion must explain.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim.grasp import ALL_PARAMS, GROUP_A, GROUP_B, params_to_dict, run_grasp

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main(seed: int = 7) -> dict:
    ds = np.load(DATA_DIR / "step1_dataset.npz")
    vecs = ds["params"]
    outs = ds["outcomes"]
    names = list(ds["param_names"])
    assert names == ALL_PARAMS, names

    fail_idx = np.where(outs == 0)[0]
    print(f"{len(fail_idx)} failures in Step-1 dataset")

    rng = np.random.default_rng(seed)

    # Pick one failure that is "interesting": not in an extreme corner of
    # the param space (so inversion has somewhere to converge to that
    # isn't a wall). We require each param to be strictly inside its
    # middle 80% range.
    from sim.grasp import PARAM_BOUNDS

    def interior(v):
        for j, k in enumerate(ALL_PARAMS):
            lo, hi = PARAM_BOUNDS[k]
            pad = 0.10 * (hi - lo)
            if not (lo + pad < v[j] < hi - pad):
                return False
        return True

    interior_fails = [i for i in fail_idx if interior(vecs[i])]
    print(f"{len(interior_fails)} of those are 'interior' (middle 80% of each axis)")
    pick = int(rng.choice(interior_fails))
    chosen_vec = vecs[pick].astype(np.float64)
    chosen = params_to_dict(chosen_vec)

    print(f"\nChosen client failure (Step-1 idx={pick}):")
    for k in ALL_PARAMS:
        marker = " [hidden]" if k in GROUP_B else " [known]"
        print(f"  {k:>10s} = {chosen[k]:+.4f}{marker}")

    # Re-simulate with trajectory recording (deterministic). This is
    # the 'observation' the inversion is allowed to see (alongside the
    # group-A params).
    r = run_grasp(chosen, record=True)
    assert r["success"] is False, "Re-sim should still be a failure (deterministic)."

    out = {
        "step1_index": int(pick),
        "true_params": chosen,
        "true_group_A": {k: chosen[k] for k in GROUP_A},
        "true_group_B": {k: chosen[k] for k in GROUP_B},
        "final_obj_z": r["final_obj_z"],
    }

    np.savez(
        DATA_DIR / "step2_observation.npz",
        traj_pos=r["traj_pos"],
        traj_quat=r["traj_quat"],
        traj_t=r["traj_t"],
        true_params=chosen_vec,
        param_names=np.array(ALL_PARAMS),
    )
    with open(DATA_DIR / "step2_client.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nTrajectory length: {len(r['traj_t'])} samples over {r['traj_t'][-1]:.2f}s")
    print(f"Saved → {DATA_DIR/'step2_observation.npz'}")
    print(f"Saved → {DATA_DIR/'step2_client.json'}")
    return out


if __name__ == "__main__":
    main()
