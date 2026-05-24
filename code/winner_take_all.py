"""
winner_take_all.py — multi-seed measurement of domain convergence.

Motivation
----------
A single-seed inspection (seed=42) showed that in the Peaked scenario, 8 of 10
labs ended the run with their max skill in domain 6, even though only 1 lab
picked domain 6 as its initial peak. This script verifies that this
winner-take-all convergence is a robust pattern, not a seed-42 artefact.

The mechanism: gap-dependent assimilation (Equation 7) lets researchers in
labs whose peak domain differs from the early-winning domain cross-train
at the SECONDARY_LEARN_FACTOR (0.12) rate; evolutionary culling (40-step
intervals) further injects mutated copies of the global elite — who all have
high skill in the early-winning domain — into struggling labs. Together
these dominate the weaker fingerprint drift (0.002/step asymmetric pull
back toward the lab's institutional peak), producing convergence on a
single domain over 300 steps.

For each scenario × seed, this script records:
  - initial lab peak domains (random assignment per lab)
  - end-state global argmax domain
  - count of labs whose end-state max equals the global argmax
  - pairwise cosine similarity of researcher profiles (homogeneity measure)
  - initial frequency of the eventual winning domain (1 means single-lab winner)

Output
------
meta/winner_take_all_results.csv         per (scenario, seed) row
stdout                                    formatted convergence summary
"""
from __future__ import annotations

import sys, os, csv
import numpy as np
from itertools import combinations
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))

from experiment import PEAKED, BROAD, FLAT, SHARED, SCENARIOS

STEPS = 300
SEEDS = list(range(30))


def _worker(task: tuple) -> dict:
    label, seed = task
    from world import ScienceWorld

    w = ScienceWorld(rng=seed, **SCENARIOS[label], **SHARED)
    n_labs    = w.n_labs
    n_domains = w.n_domains
    init_peaks = [int(np.argmax(lab.fingerprint)) for lab in w.labs]
    init_peak_counts = np.bincount(init_peaks, minlength=n_domains)

    for _ in range(STEPS):
        w.step()

    by_lab: dict[int, list[list[float]]] = {}
    for a in w.agents:
        by_lab.setdefault(a.lab_id, []).append(a.domain_skills)

    all_skills   = np.array([a.domain_skills for a in w.agents])
    global_mean  = all_skills.mean(axis=0)
    global_argmax = int(np.argmax(global_mean))

    # How many labs end up with their lab-mean argmax equal to the global argmax?
    lab_end_peaks = []
    for lab_id in sorted(by_lab.keys()):
        skills = np.array(by_lab[lab_id])
        lab_end_peaks.append(int(np.argmax(skills.mean(axis=0))))
    converged = sum(1 for p in lab_end_peaks if p == global_argmax)

    # Pairwise cosine similarity (homogeneity of researcher profiles)
    sims = []
    for r1, r2 in combinations(all_skills, 2):
        sims.append(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
    cosine_mean = float(np.mean(sims))
    cosine_std  = float(np.std(sims))

    # How many labs initially picked the eventual winning domain?
    init_winner_count = int(init_peak_counts[global_argmax])

    # Of the labs that did NOT initially peak on the winner, how many converged?
    non_initial_winners = [lab_id for lab_id in sorted(by_lab.keys())
                           if init_peaks[lab_id] != global_argmax]
    converged_non_initial = sum(
        1 for lab_id in non_initial_winners
        if int(np.argmax(np.array(by_lab[lab_id]).mean(axis=0))) == global_argmax
    )

    return dict(
        label=label, seed=seed,
        winning_domain=global_argmax,
        winning_domain_mean_skill=float(global_mean[global_argmax]),
        init_lab_count_on_winner=init_winner_count,
        labs_converged_on_winner=converged,
        non_initial_labs=len(non_initial_winners),
        non_initial_converged=converged_non_initial,
        cosine_sim_mean=cosine_mean,
        cosine_sim_std=cosine_std,
    )


if __name__ == "__main__":
    LABELS = ("PEAKED", "BROAD", "FLAT")
    tasks  = [(lbl, seed) for lbl in LABELS for seed in SEEDS]

    print(f"Running {len(tasks)} simulations "
          f"({len(LABELS)} scenarios × {len(SEEDS)} seeds × {STEPS} steps)…",
          flush=True)
    with Pool() as pool:
        results = pool.map(_worker, tasks)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "meta")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "winner_take_all_results.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader()
        wr.writerows(results)
    print(f"\nRaw results saved → {csv_path}\n")

    # --- summary ---
    print("=" * 70)
    print("Winner-take-all convergence (30 seeds per scenario)")
    print("=" * 70)
    header = f"  {'Scenario':<9} {'avg labs converged':>20} {'avg of non-initial':>20} {'cosine-sim':>12}"
    print(header)
    print("-" * len(header))
    for lbl in LABELS:
        rs    = [r for r in results if r["label"] == lbl]
        conv  = np.mean([r["labs_converged_on_winner"]   for r in rs])
        nic   = np.mean([r["non_initial_converged"] / max(r["non_initial_labs"], 1)
                          for r in rs])
        cosim = np.mean([r["cosine_sim_mean"]            for r in rs])
        print(f"  {lbl:<9} {conv:>20.2f} {nic*100:>19.1f}% {cosim:>12.3f}")
    print()
    print("Reading:")
    print("  'avg labs converged' = number of labs (of 10) whose end-state argmax")
    print("                         matches the global argmax")
    print("  'avg of non-initial' = % of labs that did NOT start on the winning")
    print("                         domain but converged to it anyway")
    print("  'cosine-sim'         = mean pairwise cosine similarity of researcher")
    print("                         profiles (1 = identical, 0 = orthogonal)")
