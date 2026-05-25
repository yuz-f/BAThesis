"""
mean_skill_sweep.py — does H1/H2 depend on the absolute mean-skill level?

Motivation
----------
The main experiment uses MEAN_SKILL = 0.32 — a value chosen to produce a
functioning simulation in which Flat researchers sit just above the
training threshold and both heterogeneous scenarios have meaningful below-
and above-peak skill values. A user-raised concern: does H1's reversed
direction (Peaked < Broad failure rate) depend on this particular mean
level, or is it a structural feature of the (mean_skill, gap) parameter-
isation that survives at other absolute mean levels?

This sweep varies MEAN_SKILL at four levels — 0.25, 0.32 (base), 0.45,
0.55 — holding the gap structure of each scenario fixed:
  - Peaked: gap = 0.40 (peak = m + 0.36, other = m - 0.04)
  - Broad:  gap = 0.18 (peak = m + 0.162, other = m - 0.018)
  - Flat:   gap = 0.00 (uniform at m)

If the H1 reversal and H2 confirmation hold at every mean level, the
findings reflect skill-distribution shape rather than the specific calibration
of the absolute mean. If H1 flips at higher mean (e.g. because all
researchers approach the skill ceiling so the gap matters less), that
identifies the operational regime in which the result holds.

Output
------
meta/mean_skill_sweep_results.csv         per (mean, scenario, seed) row
stdout                                     formatted H1/H2 effect-size table
"""
from __future__ import annotations

import sys, os, csv
import numpy as np
from multiprocessing import Pool
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from experiment import make_scenario, SHARED

STEPS  = 300
SEEDS  = list(range(10))                    # 10 seeds per cell
MEANS  = [0.25, 0.32, 0.45, 0.55]           # base 0.32 in the middle
GAPS   = {"PEAKED": 0.40, "BROAD": 0.18, "FLAT": 0.00}


def _gini(counts: np.ndarray) -> float:
    if counts.sum() == 0:
        return 0.0
    s = np.sort(counts)
    n = len(s)
    return float((2 * np.dot(np.arange(1, n + 1), s) - (n + 1) * s.sum())
                 / (n * s.sum()))


def _worker(task: tuple) -> dict:
    mean_skill, label, seed = task
    from world import ScienceWorld

    scen = make_scenario(mean_skill=mean_skill, gap=GAPS[label])
    scen.pop("selection_interval", None)    # supplied via SHARED
    w = ScienceWorld(rng=seed, **scen, **SHARED)
    for _ in range(STEPS):
        w.step()

    fail_rate = (w.replication_failures / w.replication_attempts
                 if w.replication_attempts > 0 else 0.0)
    counts = np.array([sum(1 for m in w.scientific_models if m.domain == d)
                       for d in range(w.n_domains)], dtype=float)
    mean_sk = float(np.mean([a.domain_skills for a in w.agents]))

    return dict(mean_init=mean_skill, label=label, seed=seed,
                fail_rate=float(fail_rate),
                gini=_gini(counts),
                mean_skill_end=mean_sk)


def _cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1)
                     + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


if __name__ == "__main__":
    LABELS = ("PEAKED", "BROAD", "FLAT")
    tasks  = [(m, lbl, seed) for m in MEANS for lbl in LABELS for seed in SEEDS]

    print(f"Running {len(tasks)} simulations "
          f"({len(MEANS)} mean levels × {len(LABELS)} scenarios × {len(SEEDS)} seeds × {STEPS} steps)…",
          flush=True)
    with Pool() as pool:
        results = pool.map(_worker, tasks)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "meta")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "mean_skill_sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader()
        wr.writerows(results)
    print(f"\nRaw results saved → {csv_path}\n")

    def grab(m, label, key):
        return [r[key] for r in results if r["mean_init"] == m
                                        and r["label"] == label]

    print("=" * 86)
    print(f"H1/H2 contrasts across mean-skill levels (n={len(SEEDS)} per cell)")
    print("=" * 86)
    fmt = "  m={:<5}   {:>8}={:>5}   {:>8}={:>5}   {:>8}={:>5}   {:>9}   {:>9}"
    hdr = "  {:<7}   {:<14}   {:<14}   {:<14}   {:>9}   {:>9}"
    print(hdr.format("MeanSkill", "Peaked fail", "Broad fail", "Flat fail",
                       "H1: ΔP-B(d)", "H2: ΔP-B(d)"))
    print("-" * 86)
    for m in MEANS:
        p_fail = grab(m, "PEAKED", "fail_rate")
        b_fail = grab(m, "BROAD",  "fail_rate")
        f_fail = grab(m, "FLAT",   "fail_rate")
        p_gini = grab(m, "PEAKED", "gini")
        b_gini = grab(m, "BROAD",  "gini")

        d1   = _cohens_d(p_fail, b_fail)
        d2   = _cohens_d(p_gini, b_gini)
        u1, p1 = stats.mannwhitneyu(p_fail, b_fail, alternative="two-sided")
        u2, p2 = stats.mannwhitneyu(p_gini, b_gini, alternative="two-sided")
        sig1 = ("***" if p1 < .001 else "**" if p1 < .01 else "*" if p1 < .05 else "")
        sig2 = ("***" if p2 < .001 else "**" if p2 < .01 else "*" if p2 < .05 else "")
        print(f"  m={m:.2f}   "
              f"P={np.mean(p_fail):.3f}      "
              f"B={np.mean(b_fail):.3f}      "
              f"F={np.mean(f_fail):.3f}      "
              f"{d1:+.2f}{sig1:<3}   {d2:+.2f}{sig2}")

    print()
    print("Interpretation:")
    print("  H1 reversed: ΔP-B (failure) should be NEGATIVE at every mean level")
    print("  H2 confirmed: ΔP-B (Gini) should be POSITIVE at every mean level")
    print("  Sign-flips identify the operational regime in which the headlines hold.")
