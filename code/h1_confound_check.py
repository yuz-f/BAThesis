"""
h1_confound_check.py — disentangle initial-shape from winner-take-all in H1.

Motivation
----------
A user-raised concern: maybe the H1 reversal (Peaked fails less than Broad)
is not about the *initial* peaked skill profile but about the *winner-take-all
convergence* that the model develops over 300 steps. Under that worry, by
step 300 every Peaked researcher has accumulated high skill in a single
"winning" domain (through culling + secondary assimilation), and that
late-stage single-domain proficiency — not the initial shape — drives the
failure-rate reduction.

This script disentangles the two channels:

  (1) Early-vs-late temporal check: report H1 (Peaked vs Broad fail rate) at
      steps {25, 50, 100, 150, 200, 250, 300}. If H1 is already strong at
      step 50 — before the winner-take-all has had time to develop, which
      requires several cull cycles (selection_interval = 40) and many
      secondary-assimilation events — the initial-shape channel must be
      doing the work.

  (2) No-cull condition: set selection_interval beyond STEPS to disable the
      evolutionary cull entirely. The remaining winner-take-all channel
      (secondary assimilation at the SECONDARY_LEARN_FACTOR rate) is much
      weaker without the cull's elite-injection. If H1 still holds with no
      cull, the initial-shape channel is unambiguously dominant.

Output
------
meta/h1_confound_check_results.csv         per (seed, scenario, mode, checkpoint) row
stdout                                      H1 effect size at each checkpoint × mode
"""
from __future__ import annotations

import sys, os, csv
import numpy as np
from multiprocessing import Pool
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from experiment import make_scenario, SHARED

STEPS       = 300
SEEDS       = list(range(15))
CHECKPOINTS = [25, 50, 100, 150, 200, 250, 300]
MODES       = ("default_cull", "no_cull")
GAPS        = {"PEAKED": 0.40, "BROAD": 0.18}
MEAN_SKILL  = 0.32


def _worker(task: tuple) -> list[dict]:
    label, mode, seed = task
    from world import ScienceWorld

    scen = make_scenario(mean_skill=MEAN_SKILL, gap=GAPS[label])
    if mode == "no_cull":
        scen["selection_interval"] = STEPS + 999     # never fires
    shared = {k: v for k, v in SHARED.items() if k != "selection_interval"}
    w = ScienceWorld(rng=seed, **scen, **shared)

    rows = []
    for step in range(1, STEPS + 1):
        w.step()
        if step in CHECKPOINTS:
            fail = (w.replication_failures / w.replication_attempts
                    if w.replication_attempts > 0 else 0.0)
            rows.append(dict(label=label, mode=mode, seed=seed, step=step,
                             fail_rate=float(fail),
                             attempts=int(w.replication_attempts),
                             failures=int(w.replication_failures)))
    return rows


def _cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1)
                     + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


if __name__ == "__main__":
    tasks = [(lbl, mode, seed) for lbl in ("PEAKED", "BROAD")
                                for mode in MODES
                                for seed in SEEDS]
    print(f"Running {len(tasks)} simulations "
          f"(2 scenarios × {len(MODES)} modes × {len(SEEDS)} seeds × {STEPS} steps; "
          f"{len(CHECKPOINTS)} checkpoints per run)…", flush=True)
    with Pool() as pool:
        nested = pool.map(_worker, tasks)
    results = [r for rows in nested for r in rows]

    out_dir = os.path.join(os.path.dirname(__file__), "..", "meta")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "h1_confound_check_results.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader()
        wr.writerows(results)
    print(f"\nRaw results saved → {csv_path}\n")

    def grab(mode, label, step):
        return [r["fail_rate"] for r in results
                if r["mode"] == mode and r["label"] == label and r["step"] == step]

    print("=" * 88)
    print(f"H1 (Peaked vs Broad failure rate) across time and cull conditions (n={len(SEEDS)} per cell)")
    print("=" * 88)
    fmt = "  step {:>3} | {:<12} | Peaked={:.3f}  Broad={:.3f}  ΔP-B d={:+.2f}{}"
    for step in CHECKPOINTS:
        for mode in MODES:
            p = grab(mode, "PEAKED", step)
            b = grab(mode, "BROAD",  step)
            d  = _cohens_d(p, b)
            u, pv = stats.mannwhitneyu(p, b, alternative="two-sided")
            sig = ("***" if pv < .001 else "**" if pv < .01 else
                   "*" if pv < .05 else "")
            print(fmt.format(step, mode, np.mean(p), np.mean(b), d, sig))
        print("-" * 88)

    print()
    print("Interpretation:")
    print("  If H1 (ΔP-B negative) is already strong at step 50 in BOTH modes,")
    print("  the initial-shape channel is dominant — winner-take-all is not")
    print("  the cause of H1, only an amplifier.")
    print("  If H1 holds in no_cull mode at every checkpoint, winner-take-all")
    print("  is dispensable for H1.")
