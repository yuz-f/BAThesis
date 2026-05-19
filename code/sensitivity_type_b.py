"""
sensitivity_type_b.py — sensitivity analysis for the Type B Rescorla–Wagner
parameters (α, β).

The Type B H1/H2 effect sizes reported in experiment_type_b.py are conditional
on α = 0.10 (learning rate) and β = 3.0 (softmax inverse temperature). A
defensive examiner will ask: how do those numbers move under different
parameter choices? This script gives a one-at-a-time (OAT) answer.

Design
------
α swept at 5 levels (0.05, 0.075, 0.10, 0.15, 0.20) holding β = 3.0
β swept at 5 levels (1.5, 2.0, 3.0, 4.5, 6.0) holding α = 0.10
Each level × 3 scenarios × 8 seeds × 300 steps = 240 simulations.

For every (param, level, scenario) cell we record H1 (replication failure
rate) and H2 (per-domain Gini). The headline question is whether the
*direction* of the Peaked vs Broad contrast is preserved across the swept
range — if H2 effect size (Peaked vs Broad) stays positive and roughly
in the d ≈ 0.5–1.0 range across all levels, the Type B finding is robust.

Output
------
meta/sensitivity_type_b_results.csv    raw per-sim data
stdout                                  formatted per-parameter summary

OAT systematically under-counts parameter importance in nonlinear systems
[Saltelli2010]; this is a lower-bound robustness check, not a full Sobol
decomposition. The Type B model has only two new free parameters, so OAT
is a reasonable scope here.
"""
from __future__ import annotations

import sys, os, csv
import numpy as np
from multiprocessing import Pool
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from experiment import PEAKED, BROAD, FLAT, SHARED, SCENARIOS, _gini, _ci95, _cohen_d

STEPS    = 300
N_SEEDS  = 8
SEEDS    = list(range(N_SEEDS))

ALPHA_LEVELS = [0.05, 0.075, 0.10, 0.15, 0.20]
BETA_LEVELS  = [1.5, 2.0, 3.0, 4.5, 6.0]

ALPHA_BASE = 0.10
BETA_BASE  = 3.0


def _worker(task: tuple) -> dict:
    param, level, label, seed = task
    from world import ScienceWorld
    from collections import Counter

    scenario = SCENARIOS[label]
    alpha = level if param == "alpha" else ALPHA_BASE
    beta  = level if param == "beta"  else BETA_BASE

    w = ScienceWorld(
        rng=seed,
        **scenario, **SHARED,
        enable_type_b=True,
        alpha_rl=alpha,
        beta_rl=beta,
    )

    for _ in range(STEPS):
        w.step()

    agents = list(w.agents)
    fail_rate = w.replication_failures / max(w.replication_attempts, 1)

    pub_counts: Counter = Counter()
    for a in agents:
        for d, n in a.domain_pubs.items():
            pub_counts[d] += n
    gini = _gini(list(pub_counts.values()))

    return dict(
        param=param, level=level, label=label, seed=seed,
        fail_rate=fail_rate, gini=gini,
        mean_reputation=float(np.mean([a.reputation for a in agents])),
    )


def _summarise_param(results: list[dict], param: str) -> None:
    """Per-level Peaked-vs-Broad effect sizes for one swept parameter."""
    rows = [r for r in results if r["param"] == param]
    levels = sorted(set(r["level"] for r in rows))
    by_cell: dict[tuple[float, str], list[dict]] = {}
    for r in rows:
        by_cell.setdefault((r["level"], r["label"]), []).append(r)

    title = f"Sensitivity sweep on {param} (Type B, β={BETA_BASE} / α={ALPHA_BASE} held constant for the other)"
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    header = f"{'Level':>10s} | {'H1 P':>7s} {'H1 B':>7s} {'H1 F':>7s} {'d(P,B)':>7s} | {'H2 P':>7s} {'H2 B':>7s} {'H2 F':>7s} {'d(P,B)':>7s} | {'p(H2)':>10s}"
    print(header)
    print("-" * len(header))
    for lev in levels:
        p_fr = [r["fail_rate"] for r in by_cell[(lev, "PEAKED")]]
        b_fr = [r["fail_rate"] for r in by_cell[(lev, "BROAD")]]
        f_fr = [r["fail_rate"] for r in by_cell[(lev, "FLAT")]]
        p_g  = [r["gini"]      for r in by_cell[(lev, "PEAKED")]]
        b_g  = [r["gini"]      for r in by_cell[(lev, "BROAD")]]
        f_g  = [r["gini"]      for r in by_cell[(lev, "FLAT")]]
        d_fr = _cohen_d(p_fr, b_fr)
        d_g  = _cohen_d(p_g, b_g)
        _, p_val = stats.mannwhitneyu(p_g, b_g, alternative="two-sided")
        flag = ""
        if lev == (ALPHA_BASE if param == "alpha" else BETA_BASE):
            flag = " ← base"
        print(f"{lev:>10g} | {np.mean(p_fr):>7.4f} {np.mean(b_fr):>7.4f} {np.mean(f_fr):>7.4f} {d_fr:>7.3f} | "
              f"{np.mean(p_g):>7.4f} {np.mean(b_g):>7.4f} {np.mean(f_g):>7.4f} {d_g:>7.3f} | {p_val:>10.4g}{flag}")
    print()


if __name__ == "__main__":
    tasks  = [("alpha", lvl, lbl, seed)
              for lvl in ALPHA_LEVELS
              for lbl in ("PEAKED", "BROAD", "FLAT")
              for seed in SEEDS]
    tasks += [("beta",  lvl, lbl, seed)
              for lvl in BETA_LEVELS
              for lbl in ("PEAKED", "BROAD", "FLAT")
              for seed in SEEDS]

    print(f"Running {len(tasks)} simulations "
          f"({len(ALPHA_LEVELS)}+{len(BETA_LEVELS)} levels × 3 scenarios × {N_SEEDS} seeds × {STEPS} steps)…",
          flush=True)
    with Pool() as pool:
        results = pool.map(_worker, tasks)

    out_dir  = os.path.join(os.path.dirname(__file__), "..", "meta")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sensitivity_type_b_results.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(results)
    print(f"\nRaw results saved → {csv_path}\n")

    _summarise_param(results, "alpha")
    _summarise_param(results, "beta")
