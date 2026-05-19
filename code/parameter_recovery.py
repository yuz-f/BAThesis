"""
parameter_recovery.py
=====================

Two methodological additions to the main experiment:

1. PARAMETER RECOVERY (Tier 3.1):
   Sweeps the manipulated parameter `other_skill_mean` over a grid of values
   spanning the PEAKED and BROAD conditions plus extrapolations, with 15 seeds
   per value. Asks: from the simulation outcome metrics alone (fail_rate,
   per-domain Gini, action fractions, reputation), can the underlying
   parameter be recovered? This tests whether the simulated outcomes carry
   information about the manipulation — a partial-identifiability check.

2. PER-RESEARCHER PUBLICATION GINI (Tier 3.2):
   Computes the Gini coefficient of per-researcher publication counts at
   each run's end-state, for empirical comparison. Real academic publication
   distributions are highly unequal (typical Gini ≈ 0.4–0.6 across long
   career horizons). This adds a second empirical contact point beyond the
   OSC failure-rate comparison.

Saves per-run results to meta/parameter_recovery_results.csv

Run from any directory:
    .venv/bin/python3 code/parameter_recovery.py
"""
from __future__ import annotations
import os
import sys
import csv
import numpy as np
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEPS = 300
SEEDS = list(range(15))     # 15 seeds × 6 values × ≈300s = ≈10 min on multi-core

# Sweep grid: spans PEAKED (.25), BROAD (.33), and extrapolates both directions.
OTHER_SKILL_MEAN_GRID = [0.20, 0.25, 0.30, 0.33, 0.38, 0.43]

# All other parameters fixed to BROAD-style baseline; we manipulate one variable.
FIXED = dict(
    peak_skill_mean=0.55,
    peak_skill_std=0.07,
    other_skill_std=0.06,
    selection_interval=40,
    train_threshold=0.30,
    skill_gain_attempt=0.06,
)


def _gini(vals):
    vals = sorted(v for v in vals if v > 0)
    n = len(vals)
    if n == 0:
        return 0.0
    s = sum(vals)
    return float((2 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * s)) - (n + 1) / n)


def _worker(task):
    other_mean, seed = task
    from world import ScienceWorld

    w = ScienceWorld(rng=seed, other_skill_mean=other_mean, **FIXED)

    action_counts = {"exploit": 0, "explore": 0, "train": 0}
    prev = {}
    for step in range(STEPS):
        prev = {a.unique_id: (a.exploit_steps, a.explore_steps, a.training_steps)
                for a in w.agents}
        w.step()
        for a in w.agents:
            if a.unique_id not in prev:
                continue
            de = a.exploit_steps  - prev[a.unique_id][0]
            dr = a.explore_steps  - prev[a.unique_id][1]
            dt = a.training_steps - prev[a.unique_id][2]
            action_counts["exploit"] += de
            action_counts["explore"] += dr
            action_counts["train"]   += dt

    agents = list(w.agents)
    total_steps = sum(action_counts.values()) or 1

    fail_rate = w.replication_failures / max(w.replication_attempts, 1)

    # Per-domain Gini (H2 metric)
    pub_counts = Counter()
    for a in agents:
        for d, n in a.domain_pubs.items():
            pub_counts[d] += n
    domain_gini = _gini(list(pub_counts.values()))

    # PER-RESEARCHER publication Gini — new for tier 3.2
    researcher_pubs = [a.publications for a in agents]
    researcher_gini = _gini(researcher_pubs)
    mean_pubs_per_researcher = float(np.mean(researcher_pubs))

    # mean reputation
    reps = [a.reputation for a in agents]
    mean_rep = float(np.mean(reps))

    return dict(
        other_skill_mean=other_mean,
        seed=seed,
        fail_rate=fail_rate,
        domain_gini=domain_gini,
        researcher_gini=researcher_gini,
        mean_pubs=mean_pubs_per_researcher,
        mean_reputation=mean_rep,
        exploit_frac=action_counts["exploit"] / total_steps,
        explore_frac=action_counts["explore"] / total_steps,
        train_frac=action_counts["train"]   / total_steps,
    )


if __name__ == "__main__":
    tasks = [(om, seed) for om in OTHER_SKILL_MEAN_GRID for seed in SEEDS]
    print(f"Running {len(tasks)} simulations "
          f"({len(OTHER_SKILL_MEAN_GRID)} values × {len(SEEDS)} seeds × {STEPS} steps)…",
          flush=True)

    with Pool() as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
            results.append(r)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                print(f"  completed {i+1}/{len(tasks)}", flush=True)

    out_dir = os.path.join(HERE, "..", "meta")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "parameter_recovery_results.csv")

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved → {csv_path}", flush=True)
