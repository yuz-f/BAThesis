"""
skill_dynamics.py — 30-seed measurement of skill-frontier trajectories.

Motivation
----------
The mean-constant reparameterisation equates per-researcher mean skill at
*initialisation* (m = 0.32 for all three scenarios). This script measures
what happens to mean skill *over the 300-step run*: a 5-seed diagnostic
showed the scenarios diverge — Peaked climbs to ≈0.40, Broad to ≈0.36,
Flat barely moves (≈0.33). This is an emergent consequence of the
distribution shape (a mediator, not a confound): concentrated competence
produces high-complexity "summit" models that the whole community then
assimilates toward, raising the skill frontier; a flat community has no
summit to climb toward and stagnates.

This run quantifies that divergence at 30 seeds per scenario so it can be
reported as a mechanism-check finding in the thesis.

Output
------
meta/skill_dynamics_results.csv   per (scenario, seed, checkpoint) row
stdout                            formatted trajectory summary
"""
from __future__ import annotations

import sys, os, csv
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))

from experiment import PEAKED, BROAD, FLAT, SHARED, SCENARIOS

STEPS       = 300
SEEDS       = list(range(30))
CHECKPOINTS = [0, 50, 100, 150, 200, 250, 300]


def _skill_stats(world) -> tuple[float, float]:
    """Return (mean skill across all researchers×domains, mean per-researcher peak)."""
    skills = np.array([a.domain_skills for a in world.agents])   # n_researchers × n_domains
    return float(skills.mean()), float(skills.max(axis=1).mean())


def _worker(task: tuple) -> list[dict]:
    label, seed = task
    from world import ScienceWorld

    w = ScienceWorld(rng=seed, **SCENARIOS[label], **SHARED)
    rows = []
    for step in range(STEPS + 1):
        if step in CHECKPOINTS:
            mean_sk, peak_sk = _skill_stats(w)
            rows.append(dict(label=label, seed=seed, step=step,
                             mean_skill=mean_sk, peak_skill=peak_sk))
        if step < STEPS:
            w.step()
    return rows


if __name__ == "__main__":
    LABELS = ("PEAKED", "BROAD", "FLAT")
    tasks  = [(lbl, seed) for lbl in LABELS for seed in SEEDS]

    print(f"Running {len(tasks)} simulations "
          f"({len(LABELS)} scenarios × {len(SEEDS)} seeds × {STEPS} steps)…", flush=True)
    with Pool() as pool:
        nested = pool.map(_worker, tasks)
    results = [r for rows in nested for r in rows]

    out_dir  = os.path.join(os.path.dirname(__file__), "..", "meta")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "skill_dynamics_results.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader()
        wr.writerows(results)
    print(f"\nRaw results saved → {csv_path}\n")

    # --- summary ---
    print("=" * 66)
    print("Mean-skill trajectory (mean across 30 seeds)")
    print("=" * 66)
    header = "  Scenario " + "".join(f"{f'step {c}':>11s}" for c in CHECKPOINTS)
    print(header)
    print("-" * len(header))
    for lbl in LABELS:
        cells = []
        for c in CHECKPOINTS:
            vals = [r["mean_skill"] for r in results if r["label"] == lbl and r["step"] == c]
            cells.append(f"{np.mean(vals):>11.4f}")
        print(f"  {lbl:8s}" + "".join(cells))
    print()
    print("=" * 66)
    print("Per-researcher peak-skill trajectory (mean across 30 seeds)")
    print("=" * 66)
    print(header)
    print("-" * len(header))
    for lbl in LABELS:
        cells = []
        for c in CHECKPOINTS:
            vals = [r["peak_skill"] for r in results if r["label"] == lbl and r["step"] == c]
            cells.append(f"{np.mean(vals):>11.4f}")
        print(f"  {lbl:8s}" + "".join(cells))
