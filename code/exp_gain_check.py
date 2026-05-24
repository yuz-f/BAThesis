"""
exp_gain_check.py — robustness of H1/H2 under experience-based learning.

Motivation
----------
The default learning rule (Equation 7) is frontier-limited: gain ∝ clip(gap, 0, 1),
where gap = complexity − skill. Researchers grow toward higher-complexity
published work, but they do NOT grow from engaging with work below their
current skill. This rule operationalises "shoulders of giants" / Kuhn-style
cumulative paradigm extension and is the substantive theoretical commitment
behind the H1 ordering — Peaked communities, with high-skill summits, have a
higher published frontier and so a steeper learning gradient for everyone
else.

The user-raised question is whether the H1 effect is partly an artefact of
this particular learning rule rather than a robust feature of the
skill-distribution manipulation. To test it, this script runs both modes:

  - frontier_limited (default):  gain ∝ skill·(1-skill)·4·clip(gap,0,1)·focus
  - experience_based  (variant):  gain ∝ skill·(1-skill)·4·focus

In the experience-based variant, every engagement event produces a small
S-curve-shaped skill increment regardless of the target's complexity. This
removes the frontier-driven asymmetry between Peaked and Broad: in Broad,
where the published frontier is lower, researchers can still grow at the
S-curve rate from practice alone.

If H1's reversed direction (Peaked < Broad failure) survives under
experience-based learning, the H1 effect is robust to the choice of
learning rule. If it shrinks or reverses, the rule choice contributes
materially to the magnitude.

Output
------
meta/exp_gain_check_results.csv          per (scenario, mode, seed) row
stdout                                    H1/H2 effect-size comparison
"""
from __future__ import annotations

import sys, os, csv
import numpy as np
from multiprocessing import Pool
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from experiment import PEAKED, BROAD, FLAT, SHARED, SCENARIOS

STEPS  = 300
SEEDS  = list(range(15))      # 15 seeds × 3 scenarios × 2 modes = 90 runs
MODES  = ("frontier_limited", "experience_based")


def _gini(counts: np.ndarray) -> float:
    if counts.sum() == 0:
        return 0.0
    s = np.sort(counts)
    n = len(s)
    return float((2 * np.dot(np.arange(1, n + 1), s) - (n + 1) * s.sum())
                 / (n * s.sum()))


def _worker(task: tuple) -> dict:
    label, mode, seed = task
    from world import ScienceWorld

    extra = {"enable_experience_gain": (mode == "experience_based")}
    w = ScienceWorld(rng=seed, **SCENARIOS[label], **SHARED, **extra)
    for _ in range(STEPS):
        w.step()

    fail_rate = (w.replication_failures / w.replication_attempts
                 if w.replication_attempts > 0 else 0.0)
    counts = np.array([sum(1 for m in w.scientific_models if m.domain == d)
                       for d in range(w.n_domains)], dtype=float)
    mean_skill = float(np.mean([a.domain_skills for a in w.agents]))

    return dict(label=label, mode=mode, seed=seed,
                fail_rate=float(fail_rate),
                gini=_gini(counts),
                mean_skill=mean_skill)


def _cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1)
                     + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


if __name__ == "__main__":
    LABELS = ("PEAKED", "BROAD", "FLAT")
    tasks  = [(lbl, mode, seed) for lbl in LABELS for mode in MODES
                                for seed in SEEDS]

    print(f"Running {len(tasks)} simulations "
          f"({len(LABELS)} scenarios × {len(MODES)} modes × {len(SEEDS)} seeds × {STEPS} steps)…",
          flush=True)
    with Pool() as pool:
        results = pool.map(_worker, tasks)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "meta")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "exp_gain_check_results.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader()
        wr.writerows(results)
    print(f"\nRaw results saved → {csv_path}\n")

    # --- per-mode H1/H2 contrasts ---
    def grab(mode, label, key):
        return [r[key] for r in results if r["mode"] == mode and r["label"] == label]

    print("=" * 78)
    print(f"H1/H2 contrasts under each learning mode (n={len(SEEDS)} per cell)")
    print("=" * 78)
    fmt = "  {:<22} {:>11} {:>11} {:>11} {:>9}"
    print(fmt.format("Mode", "Peaked", "Broad", "Flat", "ΔP-B (d)"))
    print("-" * 78)
    for mode in MODES:
        for metric, name in [("fail_rate", "fail rate"),
                              ("gini",       "Gini"),
                              ("mean_skill", "mean skill")]:
            p = grab(mode, "PEAKED", metric)
            b = grab(mode, "BROAD",  metric)
            f = grab(mode, "FLAT",   metric)
            d = _cohens_d(p, b)
            u, pv = stats.mannwhitneyu(p, b, alternative="two-sided")
            sig = ""
            if   pv < 0.001: sig = "***"
            elif pv < 0.01:  sig = "**"
            elif pv < 0.05:  sig = "*"
            print(fmt.format(f"{mode} — {name}",
                              f"{np.mean(p):.3f}",
                              f"{np.mean(b):.3f}",
                              f"{np.mean(f):.3f}",
                              f"{d:+.2f}{sig}"))
        print("-" * 78)

    print()
    print("Interpretation:")
    print("  H1 (failure rate): Peaked < Broad → ΔP-B should be NEGATIVE")
    print("  H2 (Gini):         Peaked > Broad → ΔP-B should be POSITIVE")
    print("  If both signs survive in experience_based mode, the H1 reversal")
    print("  and H2 confirmation are robust to the gap-dependent learning rule.")
