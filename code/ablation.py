"""
ablation.py — layer-decomposition ablation study (Tier 4 / addresses ChatGPT
critique cluster A: tautology, landscape-as-meta-driver, complexity attribution).

Three nested model layers are crossed with the SPEC and GEN scenarios:

  L1 — Base dynamics only:
       enable_landscape=False, enable_realism=False
       (Selection, learning, salience decay, drift, breakthrough are all on,
        but no v2 realism extensions and no v3 epistemic landscape.)

  L2 — Base + realism (no landscape):
       enable_landscape=False, enable_realism=True

  L3 — Full v3:
       enable_landscape=True,  enable_realism=True

For each (layer, scenario, seed) we compute the standard outcome metrics.
Aggregating across seeds gives the SPEC vs GEN effect size *per layer*,
which quantifies how much each architectural layer contributes to the
H1 and H2 magnitudes.

Saves: meta/ablation_results.csv
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
SEEDS = list(range(15))

SCENARIOS = {
    "SPEC": dict(peak_skill_mean=0.55, peak_skill_std=0.07,
                 other_skill_mean=0.25, other_skill_std=0.06),
    "GEN":  dict(peak_skill_mean=0.55, peak_skill_std=0.08,
                 other_skill_mean=0.33, other_skill_std=0.08),
}

LAYERS = {
    "L1_base":     dict(enable_landscape=False, enable_realism=False),
    "L2_realism":  dict(enable_landscape=False, enable_realism=True),
    "L3_full":     dict(enable_landscape=True,  enable_realism=True),
}


def _gini(vals):
    vals = sorted(v for v in vals if v > 0)
    n = len(vals)
    if n == 0:
        return 0.0
    s = sum(vals)
    return float((2 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * s)) - (n + 1) / n)


def _worker(task):
    layer_name, scenario_name, seed = task
    from world import ScienceWorld

    layer    = LAYERS[layer_name]
    scenario = SCENARIOS[scenario_name]

    w = ScienceWorld(rng=seed,
                     selection_interval=40,
                     train_threshold=0.30,
                     skill_gain_attempt=0.06,
                     **scenario,
                     **layer)

    for _ in range(STEPS):
        w.step()

    fail_rate = w.replication_failures / max(w.replication_attempts, 1)

    pub_counts = Counter()
    for a in w.agents:
        for d, n in a.domain_pubs.items():
            pub_counts[d] += n
    domain_gini = _gini(list(pub_counts.values()))

    researcher_pubs = [a.publications for a in w.agents]
    researcher_gini = _gini(researcher_pubs)

    reps = [a.reputation for a in w.agents]
    mean_rep = float(np.mean(reps))

    return dict(
        layer=layer_name, scenario=scenario_name, seed=seed,
        fail_rate=fail_rate,
        domain_gini=domain_gini,
        researcher_gini=researcher_gini,
        mean_reputation=mean_rep,
    )


if __name__ == "__main__":
    tasks = [(layer, scen, seed)
             for layer in LAYERS
             for scen in SCENARIOS
             for seed in SEEDS]
    print(f"Running {len(tasks)} ablation simulations "
          f"({len(LAYERS)} layers × {len(SCENARIOS)} scenarios × {len(SEEDS)} seeds × {STEPS} steps)…",
          flush=True)

    with Pool() as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
            results.append(r)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                print(f"  completed {i+1}/{len(tasks)}", flush=True)

    out_dir = os.path.join(HERE, "..", "meta")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ablation_results.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(results)
    print(f"\nResults saved → {csv_path}", flush=True)
