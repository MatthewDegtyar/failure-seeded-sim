"""Generate the plots referenced in README.md."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def plot_failure_marginals():
    s = json.load(open(DATA / "step1_summary.json"))
    marg = s["marginal_fail_rates"]
    keys = list(marg.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(2.4 * len(keys), 2.6), sharey=True)
    for ax, k in zip(axes, keys):
        vals = [v if v is not None else np.nan for v in marg[k]]
        ax.bar(range(len(vals)), vals, color="#7488d8")
        ax.set_title(k, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"q{i+1}" for i in range(len(vals))], fontsize=7)
    axes[0].set_ylabel("fail rate")
    fig.suptitle(
        f"Step 1: per-axis marginal failure rate  "
        f"(N={s['n']}, fail={s['fail_rate']:.2f}, structure_score={s['structure_score']:+.2f})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "step1_marginal_fail_rates.png", dpi=140)
    plt.close(fig)


def plot_failure_scatter():
    ds = np.load(DATA / "step1_dataset.npz")
    p = ds["params"]
    o = ds["outcomes"]
    names = list(ds["param_names"])
    fr_i = names.index("friction")
    m_i = names.index("mass")
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    fail = o == 0
    ax.scatter(p[~fail, fr_i], p[~fail, m_i], s=4, c="#5bb05b", alpha=0.5, label="success")
    ax.scatter(p[fail, fr_i], p[fail, m_i], s=8, c="#d05050", alpha=0.7, label="failure")
    # Mark client failure and posterior center
    truth = np.load(DATA / "step3_posterior.npz")["truth"]
    post = np.load(DATA / "step3_posterior.npz")["posterior_points"]
    ax.scatter([truth[0]], [truth[1]], marker="*", s=180, c="black", edgecolors="yellow", linewidth=1.2, label="hidden truth")
    ax.scatter(post[:, 0], post[:, 1], s=4, c="#3030aa", alpha=0.3, label="ABC posterior")
    ax.set_xlabel("friction"); ax.set_ylabel("mass")
    ax.set_title("Failure manifold (friction vs mass)\nwith client failure + recovered posterior")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "failure_manifold_with_posterior.png", dpi=140)
    plt.close(fig)


def plot_step5_bars():
    s = json.load(open(DATA / "step5_summary.json"))
    per = s["per_regime"]
    regimes = ["A", "B", "C"]
    bacc = [per[r]["bacc"] for r in regimes]
    std  = [per[r]["bacc_std"] for r in regimes]
    rfail = [per[r]["recall_fail"] for r in regimes]
    rsucc = [per[r]["recall_succ"] for r in regimes]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    x = np.arange(len(regimes))
    w = 0.25
    ax.bar(x - w, bacc, w, yerr=std, label="balanced accuracy", color="#3360c0")
    ax.bar(x, rfail, w, label="recall(fail)", color="#d06060")
    ax.bar(x + w, rsucc, w, label="recall(success)", color="#5bb05b")
    ax.set_xticks(x); ax.set_xticklabels([
        "A: DR\n(uniform)", "B: ours\n(posterior)", "C: oracle\n(truth)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("metric")
    ax.set_title(f"Step 5 payoff test (verdict: {s['verdict']})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "step5_bars.png", dpi=140)
    plt.close(fig)


def plot_step6_noise():
    s = json.load(open(DATA / "step6_noise_sweep.json"))
    res = s["results"]
    noise_mm = [r["noise_sigma_pos_m"] * 1000.0 for r in res]
    inv = [r["inversion_error"] for r in res]
    bb = [r["bacc_B"] for r in res]
    bb_std = [r["bacc_B_std"] for r in res]
    margin = [r["margin_B_minus_A"] for r in res]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))
    ax1.plot(noise_mm, inv, "o-", color="#3360c0", label="inversion error (rel.)")
    ax1.set_xlabel("trajectory pos noise σ (mm)")
    ax1.set_ylabel("overall mean relative error")
    ax1.set_title("Inversion error vs observation noise")
    ax1.grid(True, alpha=0.3)

    ax2.axhline(s["bacc_A_reference"], color="#cf6e6e", linestyle="--", label=f"BAcc(A)={s['bacc_A_reference']:.3f}")
    ax2.axhline(s["bacc_C_reference"], color="#5fbf5f", linestyle="--", label=f"BAcc(C)={s['bacc_C_reference']:.3f}")
    ax2.errorbar(noise_mm, bb, yerr=bb_std, fmt="o-", color="#3360c0", label="BAcc(B)")
    ax2.set_xlabel("trajectory pos noise σ (mm)")
    ax2.set_ylabel("balanced accuracy")
    ax2.set_title("Classifier BAcc(B) vs noise")
    ax2.set_ylim(0.5, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "step6_noise_sweep.png", dpi=140)
    plt.close(fig)


def plot_n_sweep():
    s = json.load(open(DATA / "step5b_n_sweep.json"))
    res = s["results"]
    n = [r["n"] for r in res]
    a = [r["bacc_A"] for r in res]; sa = [r["std_A"] for r in res]
    b = [r["bacc_B"] for r in res]; sb = [r["std_B"] for r in res]
    c = [r["bacc_C"] for r in res]; sc = [r["std_C"] for r in res]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.errorbar(n, a, yerr=sa, fmt="o-", color="#cf6e6e", label="A (uniform DR)")
    ax.errorbar(n, b, yerr=sb, fmt="o-", color="#3360c0", label="B (ours, posterior)")
    ax.errorbar(n, c, yerr=sc, fmt="o-", color="#5fbf5f", label="C (oracle, truth)")
    ax.set_xscale("log")
    ax.set_xlabel("N per regime (training size)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Exploratory: BAcc vs training budget")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "step5b_n_sweep.png", dpi=140)
    plt.close(fig)


def main():
    plot_failure_marginals()
    plot_failure_scatter()
    plot_step5_bars()
    plot_step6_noise()
    plot_n_sweep()
    print(f"Wrote plots → {OUT}")


if __name__ == "__main__":
    main()
