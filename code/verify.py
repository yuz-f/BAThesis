"""Verify final calibration with action distribution per phase."""
from __future__ import annotations
import sys, os
import numpy as np
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))

STEPS = 300
SEEDS = [0, 1, 2]

SPEC = dict(peak_skill_mean=0.55, peak_skill_std=0.07, other_skill_mean=0.25, other_skill_std=0.06, selection_interval=40)
GEN  = dict(peak_skill_mean=0.55, peak_skill_std=0.08, other_skill_mean=0.33, other_skill_std=0.08, selection_interval=40)

TT  = 0.30
SGA = 0.06


def gini(vals):
    vals = sorted(v for v in vals if v > 0)
    n = len(vals)
    if n == 0: return 0.0
    return float((2*np.sum(np.arange(1,n+1)*vals)/(n*sum(vals))) - (n+1)/n)


def _worker(task):
    label, seed = task
    from world import ScienceWorld
    scenario = SPEC if label == 'SPEC' else GEN
    w = ScienceWorld(rng=seed, train_threshold=TT, skill_gain_attempt=SGA, **scenario)
    phase = {f'{p}_{a}': 0 for p in ('e','m','l') for a in ('expt','expl','train')}
    for step in range(STEPS):
        before = {a.unique_id:(a.exploit_steps,a.explore_steps,a.training_steps) for a in w.agents}
        w.step()
        for a in w.agents:
            if a.unique_id not in before: continue
            de,dr,dt = a.exploit_steps-before[a.unique_id][0], a.explore_steps-before[a.unique_id][1], a.training_steps-before[a.unique_id][2]
            p = 'e' if step < 30 else ('m' if step < 150 else 'l')
            phase[f'{p}_expt']  += de
            phase[f'{p}_expl']  += dr
            phase[f'{p}_train'] += dt
    fail = w.replication_failures/max(w.replication_attempts,1)
    pub_counts = Counter()
    for a in w.agents:
        for d,n in a.domain_pubs.items(): pub_counts[d]+=n
    return label, seed, fail, gini(list(pub_counts.values())), phase


if __name__ == "__main__":
    tasks = [(label, seed) for label in ('SPEC', 'GEN') for seed in SEEDS]
    with Pool() as pool:
        results = pool.map(_worker, tasks)

    by_label = {}
    for label, seed, fail, g, phase in results:
        by_label.setdefault(label, []).append((fail, g, phase))

    print(f"Calibration: train_threshold={TT}, skill_gain_attempt={SGA}\n")
    for label, runs in by_label.items():
        fail = np.mean([r[0] for r in runs])
        g    = np.mean([r[1] for r in runs])
        keys = runs[0][2].keys()
        ph   = {k: np.mean([r[2][k] for r in runs]) for k in keys}
        print(f"{label}  fail={fail:.1%}  gini={g:.3f}")
        for tag, span in (('e','0-30'), ('m','30-150'), ('l','150-300')):
            tot = ph[f'{tag}_expt'] + ph[f'{tag}_expl'] + ph[f'{tag}_train']
            tot = max(tot, 1)
            print(f"  {span:>8}: exploit={ph[f'{tag}_expt']/tot:.0%}  explore={ph[f'{tag}_expl']/tot:.0%}  train={ph[f'{tag}_train']/tot:.0%}")
        print()
