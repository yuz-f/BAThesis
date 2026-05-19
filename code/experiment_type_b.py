"""
Type A vs Type B comparison experiment.

Crosses the three scenarios (PEAKED, BROAD, FLAT) with the two architectural
modes (Type A — skill-bias action selection, Equation 6; Type B — softmax
over Rescorla-Wagner learned domain utility) and reports H1 (replication
failure rate) and H2 (per-domain Gini concentration) under each combination.

Headline question: does the H2 concentration effect persist under Type B,
or does it collapse once the architectural skill-bias term is removed?

  • If H2 effect size (Peaked vs Broad) remains large under Type B
    → concentration is feedback-emergent (success→reward→utility loop is
       sufficient to produce it)
  • If H2 effect collapses
    → original H2 was largely architectural; the skill-bias rule was doing
       the load-bearing work

Tasks: 2 modes × 3 scenarios × 30 seeds = 180 simulations.
Output: meta/experiment_type_b_results.csv + summary printed to stdout.
"""
from __future__ import annotations

import sys, os, csv
import numpy as np
from multiprocessing import Pool
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from experiment import (
    PEAKED, BROAD, FLAT, SHARED, SCENARIOS,
    STEPS, SEEDS, _gini, _ci95, _cohen_d,
)

MODES = ("A", "B")  # A: skill-bias (Equation 6); B: softmax + Rescorla-Wagner

# Rescorla-Wagner / softmax defaults — also exposed for sensitivity work later
ALPHA_RL = 0.10
BETA_RL  = 3.0


def _worker(task: tuple) -> dict:
    mode, label, seed = task
    from world import ScienceWorld
    from collections import Counter

    scenario = SCENARIOS[label]
    w = ScienceWorld(
        rng=seed,
        **scenario, **SHARED,
        enable_type_b=(mode == "B"),
        alpha_rl=ALPHA_RL,
        beta_rl=BETA_RL,
    )

    action_counts = {"exploit": 0, "explore": 0, "train": 0}
    prev = {}

    for _ in range(STEPS):
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

    pub_counts: Counter = Counter()
    for a in agents:
        for d, n in a.domain_pubs.items():
            pub_counts[d] += n
    gini = _gini(list(pub_counts.values()))

    reputations = [a.reputation for a in agents]
    mean_rep    = float(np.mean(reputations))

    return dict(
        mode=mode, label=label, seed=seed,
        fail_rate=fail_rate, gini=gini,
        mean_reputation=mean_rep,
        exploit_frac=action_counts["exploit"] / total_steps,
        explore_frac=action_counts["explore"] / total_steps,
        train_frac=action_counts["train"]     / total_steps,
        n_models=len(w.scientific_models),
    )


def _summarise(results: list[dict]) -> dict[tuple[str, str], dict]:
    """Group results by (mode, label) and compute mean, 95% CI for each metric."""
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        by_cell.setdefault((r["mode"], r["label"]), []).append(r)
    out: dict[tuple[str, str], dict] = {}
    for cell, rows in by_cell.items():
        d = {}
        for k in ("fail_rate", "gini", "mean_reputation",
                  "exploit_frac", "explore_frac", "train_frac", "n_models"):
            vals = [row[k] for row in rows]
            m, lo, hi = _ci95(vals)
            d[k] = (m, lo, hi)
        out[cell] = d
    return out


def _h1_h2_table(results: list[dict]) -> None:
    """Per-mode Peaked-vs-Broad effect sizes for H1 and H2."""
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        by_cell.setdefault((r["mode"], r["label"]), []).append(r)

    print("=" * 78)
    print("PEAKED vs BROAD effect sizes by mode")
    print("=" * 78)
    print(f"{'Mode':6s}{'Metric':22s}{'Peaked':>10s}{'Broad':>10s}{'d':>8s}{'p':>10s}")
    print("-" * 78)
    for mode in MODES:
        for key, name in [("fail_rate", "H1: fail rate"),
                          ("gini",      "H2: per-domain Gini")]:
            p_vals = [r[key] for r in by_cell[(mode, "PEAKED")]]
            b_vals = [r[key] for r in by_cell[(mode, "BROAD")]]
            _, p_val = stats.mannwhitneyu(p_vals, b_vals, alternative="two-sided")
            d = _cohen_d(p_vals, b_vals)
            print(f"  {mode}   {name:22s}{np.mean(p_vals):>10.4f}{np.mean(b_vals):>10.4f}{d:>8.3f}{p_val:>10.4g}")
        print()


def _ab_contrast_table(results: list[dict]) -> None:
    """Within-scenario A-vs-B contrast: does Type B change the outcome?"""
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        by_cell.setdefault((r["mode"], r["label"]), []).append(r)

    print("=" * 78)
    print("Within-scenario A-vs-B contrast")
    print("=" * 78)
    print(f"{'Scenario':10s}{'Metric':22s}{'A-mean':>10s}{'B-mean':>10s}{'d(A-B)':>10s}{'p':>10s}")
    print("-" * 78)
    for lbl in ("PEAKED", "BROAD", "FLAT"):
        for key, name in [("fail_rate", "H1: fail rate"),
                          ("gini",      "H2: per-domain Gini")]:
            a_vals = [r[key] for r in by_cell[("A", lbl)]]
            b_vals = [r[key] for r in by_cell[("B", lbl)]]
            _, p_val = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            d = _cohen_d(a_vals, b_vals)
            print(f"  {lbl:8s}{name:22s}{np.mean(a_vals):>10.4f}{np.mean(b_vals):>10.4f}{d:>10.3f}{p_val:>10.4g}")
        print()


def _descriptives_table(results: list[dict]) -> None:
    """Per-cell descriptive statistics (mean and 95% CI)."""
    summary = _summarise(results)
    print("=" * 78)
    print("Descriptives (mean [95% CI])")
    print("=" * 78)
    metrics = [
        ("fail_rate",       "H1 fail rate"),
        ("gini",            "H2 Gini"),
        ("mean_reputation", "Mean reputation"),
        ("exploit_frac",    "Exploit fraction"),
        ("explore_frac",    "Explore fraction"),
        ("train_frac",      "Train fraction"),
    ]
    for key, name in metrics:
        print(f"\n  {name}")
        for mode in MODES:
            for lbl in ("PEAKED", "BROAD", "FLAT"):
                m, lo, hi = summary[(mode, lbl)][key]
                print(f"    {mode}-{lbl:7s}  {m:7.4f}  [{lo:7.4f}, {hi:7.4f}]")


if __name__ == "__main__":
    LABELS = ("PEAKED", "BROAD", "FLAT")
    tasks = [(mode, label, seed)
             for mode in MODES
             for label in LABELS
             for seed in SEEDS]

    print(f"Running {len(tasks)} simulations "
          f"({len(MODES)} modes × {len(LABELS)} scenarios × {len(SEEDS)} seeds × {STEPS} steps)…")
    with Pool() as pool:
        results = pool.map(_worker, tasks)

    # save raw results
    out_dir  = os.path.join(os.path.dirname(__file__), "..", "meta")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "experiment_type_b_results.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(results)
    print(f"\nRaw results saved → {csv_path}\n")

    _h1_h2_table(results)
    _ab_contrast_table(results)
    _descriptives_table(results)
