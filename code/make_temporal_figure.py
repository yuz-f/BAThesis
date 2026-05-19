"""
make_temporal_figure.py
========================

Produces a two-panel figure showing how the H1 and H2 metrics emerge over
time, with mean trajectories and 95% confidence bands across seeds for all
three scenarios (PEAKED, BROAD, FLAT).

Panel A: cumulative replication failure rate over 300 steps
Panel B: Gini coefficient of per-domain publication counts over 300 steps

Saves to meta/img/temporal_figure.pdf

Run from any directory:
    .venv/bin/python3 code/make_temporal_figure.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

STEPS = 300
SEEDS = list(range(30))

PEAKED = dict(peak_skill_mean=0.55, peak_skill_std=0.07,
            other_skill_mean=0.25, other_skill_std=0.06,
            selection_interval=40)
BROAD  = dict(peak_skill_mean=0.55, peak_skill_std=0.08,
            other_skill_mean=0.33, other_skill_std=0.08,
            selection_interval=40)
FLAT = dict(peak_skill_mean=0.40, peak_skill_std=0.02,
               other_skill_mean=0.40, other_skill_std=0.02,
               selection_interval=40)
SCENARIOS = {"PEAKED": PEAKED, "BROAD": BROAD, "FLAT": FLAT}

COLORS = {
    "PEAKED":    "#2563EB",
    "BROAD":     "#059669",
    "FLAT": "#9333EA",
}


def _gini(vals):
    vals = sorted(v for v in vals if v > 0)
    n = len(vals)
    if n == 0:
        return 0.0
    s = sum(vals)
    return float((2 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * s)) - (n + 1) / n)


def _worker(task):
    """Run one (scenario, seed) and return per-step (fail_rate, gini) trajectories."""
    label, seed = task
    from world import ScienceWorld

    scenario = SCENARIOS[label]
    w = ScienceWorld(rng=seed, train_threshold=0.30, skill_gain_attempt=0.06,
                     **scenario)

    fail_rate_traj = np.zeros(STEPS)
    gini_traj      = np.zeros(STEPS)

    for step in range(STEPS):
        w.step()
        # cumulative failure rate
        if w.replication_attempts > 0:
            fail_rate_traj[step] = w.replication_failures / w.replication_attempts
        # Gini of per-domain publication counts
        pub_counts = Counter()
        for a in w.agents:
            for d, n in a.domain_pubs.items():
                pub_counts[d] += n
        gini_traj[step] = _gini(list(pub_counts.values()))

    return label, seed, fail_rate_traj, gini_traj


def main():
    tasks = [(label, seed) for label in SCENARIOS for seed in SEEDS]
    print(f"Running {len(tasks)} simulations with per-step tracking…", flush=True)

    with Pool() as pool:
        # imap_unordered to get progress updates as workers finish
        results = []
        for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
            results.append(r)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                print(f"  completed {i+1}/{len(tasks)}", flush=True)

    # Organise into arrays per scenario: shape (n_seeds, STEPS)
    fail_arrays = {lbl: np.zeros((len(SEEDS), STEPS)) for lbl in SCENARIOS}
    gini_arrays = {lbl: np.zeros((len(SEEDS), STEPS)) for lbl in SCENARIOS}
    for label, seed, fail, gini in results:
        idx = SEEDS.index(seed)
        fail_arrays[label][idx] = fail
        gini_arrays[label][idx] = gini

    # ── plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    steps = np.arange(1, STEPS + 1)

    # Panel A: failure rate
    ax_a = axes[0]
    for lbl in ("PEAKED", "BROAD", "FLAT"):
        arr = fail_arrays[lbl]
        mu  = arr.mean(axis=0)
        # 95% CI via 2 × SE
        se  = arr.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
        ax_a.fill_between(steps, mu - 1.96 * se, mu + 1.96 * se,
                           color=COLORS[lbl], alpha=0.18, linewidth=0)
        ax_a.plot(steps, mu, color=COLORS[lbl], lw=1.7,
                   label=lbl.title() if lbl != "FLAT" else "Uniform (control)")
    ax_a.set_xlabel("Simulation step")
    ax_a.set_ylabel("Cumulative replication failure rate")
    ax_a.set_title("A.  H1 — failure rate over time", pad=8, fontsize=11)
    ax_a.set_xlim(0, STEPS)
    ax_a.set_ylim(0.45, 0.70)
    ax_a.legend(loc="lower right", frameon=False, fontsize=9)
    ax_a.grid(axis="y", color="#EEE", linewidth=0.6, zorder=0)

    # Panel B: Gini
    ax_b = axes[1]
    for lbl in ("PEAKED", "BROAD", "FLAT"):
        arr = gini_arrays[lbl]
        mu  = arr.mean(axis=0)
        se  = arr.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
        ax_b.fill_between(steps, mu - 1.96 * se, mu + 1.96 * se,
                           color=COLORS[lbl], alpha=0.18, linewidth=0)
        ax_b.plot(steps, mu, color=COLORS[lbl], lw=1.7,
                   label=lbl.title() if lbl != "FLAT" else "Uniform (control)")
    ax_b.set_xlabel("Simulation step")
    ax_b.set_ylabel("Gini of per-domain publication counts")
    ax_b.set_title("B.  H2 — domain concentration over time", pad=8, fontsize=11)
    ax_b.set_xlim(0, STEPS)
    ax_b.set_ylim(0, 0.20)
    ax_b.legend(loc="upper right", frameon=False, fontsize=9)
    ax_b.grid(axis="y", color="#EEE", linewidth=0.6, zorder=0)

    fig.suptitle("Temporal dynamics of the confirmatory metrics  ($n=30$ seeds per scenario; "
                 "shaded bands $=$ 95% CI)", fontsize=10, y=1.02)
    fig.tight_layout()

    out = os.path.join(HERE, "..", "meta", "img", "temporal_figure.pdf")
    fig.savefig(out)
    print(f"Saved → {out}")
    out_png = out.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
