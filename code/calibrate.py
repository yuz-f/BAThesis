"""Fast calibration sweep — uses multiprocessing to test parameter combos."""
from __future__ import annotations

import sys, os, itertools
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))

STEPS = 200
SEEDS = [0, 1]

SPEC = dict(peak_skill_mean=0.55, peak_skill_std=0.07, other_skill_mean=0.25, other_skill_std=0.06, selection_interval=40)
GEN  = dict(peak_skill_mean=0.55, peak_skill_std=0.08, other_skill_mean=0.33, other_skill_std=0.08, selection_interval=40)

TT  = [0.20, 0.30, 0.40]
SGA = [0.04, 0.06, 0.08]


def _worker(task):
    scenario_name, tt, sga, seed = task
    from world import ScienceWorld
    scenario = SPEC if scenario_name == 'SPEC' else GEN
    w = ScienceWorld(rng=seed, train_threshold=tt, skill_gain_attempt=sga, **scenario)
    for _ in range(STEPS):
        w.step()
    fail = w.replication_failures / max(w.replication_attempts, 1)
    return (scenario_name, tt, sga, seed, fail)


if __name__ == "__main__":
    tasks = []
    for tt, sga in itertools.product(TT, SGA):
        for seed in SEEDS:
            tasks.append(('SPEC', tt, sga, seed))
            tasks.append(('GEN',  tt, sga, seed))

    print(f"Running {len(tasks)} simulations on {os.cpu_count()} cores...")

    with Pool() as pool:
        results = pool.map(_worker, tasks)

    # aggregate by (scenario, tt, sga)
    agg = {}
    for sc, tt, sga, seed, fail in results:
        agg.setdefault((sc, tt, sga), []).append(fail)

    print(f"\n{'tt':>5} {'sga':>5} {'spec_fail':>10} {'gen_fail':>10} {'gap':>7} {'in_target':>10}")
    for tt, sga in itertools.product(TT, SGA):
        sf = float(np.mean(agg[('SPEC', tt, sga)]))
        gf = float(np.mean(agg[('GEN',  tt, sga)]))
        gap = sf - gf
        ok = (0.50 <= sf <= 0.65) and (0.50 <= gf <= 0.65) and gap > 0
        print(f"{tt:>5.2f} {sga:>5.3f} {sf:>10.1%} {gf:>10.1%} {gap:>+7.1%} {'YES' if ok else '':>10}")
