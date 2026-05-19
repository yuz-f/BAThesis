"""
sensitivity_analysis.py — one-at-a-time sensitivity analysis for the Science Lab ABM.

For each fixed parameter, runs both peaked and broad scenarios at
three levels (base × 0.70, base, base × 1.30) with N_RUNS replications each.
All runs are parallelized across available CPU cores.

Key metrics recorded per run:
  - Replication failure rate (cumulative at end of simulation)
  - Gini coefficient of domain model-count distribution
  - Training fraction (training_steps / all action steps, all agents)
  - Mean best truthfulness (average of per-domain max truthfulness)

The qualitative finding that matters:
  peaked-profile researchers show higher replication failure rate AND higher Gini than broad-profile researchers.
  If the sign of (spec − gen) flips for any parameter level, robustness is compromised.

Usage:
    cd BAThesis/code
    .venv/bin/python3 sensitivity_analysis.py
Output:
    sensitivity_results.csv   — full data table
    console                   — formatted summary
"""

import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STEPS       = 200     # enough to reach quasi-steady-state; saves ~33% vs 300
N_RUNS      = 3       # replications per (parameter, level, scenario) combination
SEED_BASE   = 42      # seeds: SEED_BASE, SEED_BASE+1, ..., SEED_BASE+N_RUNS-1
VARY_FACTOR = 0.30    # ±30% around base value

# Base parameter values (must match run.py defaults)
BASE = {
    'train_threshold':    0.30,   # matches experiment.py freeze (2026-05-05)
    'skill_gain_attempt': 0.06,   # matches experiment.py freeze (2026-05-05)
    'skill_gain_train':   0.08,
    'cull_fraction':      0.25,
    'cap_growth_rate':    0.005,
    'mutation_std':       0.04,
}

# Scenario skill distributions (must match run.py)
PEAKED = dict(
    peak_skill_mean=0.55, peak_skill_std=0.07,
    other_skill_mean=0.25, other_skill_std=0.06,
)
BROAD = dict(
    peak_skill_mean=0.55, peak_skill_std=0.08,
    other_skill_mean=0.33, other_skill_std=0.08,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gini(counts: np.ndarray) -> float:
    if counts.sum() == 0:
        return 0.0
    s = np.sort(counts)
    n = len(s)
    return float(
        (2 * np.dot(np.arange(1, n + 1), s) - (n + 1) * s.sum())
        / (n * s.sum())
    )


def _metrics(world) -> dict:
    fail_rate = (
        world.replication_failures / world.replication_attempts
        if world.replication_attempts > 0 else 0.0
    )

    counts = np.array([
        sum(1 for m in world.scientific_models if m.domain == d)
        for d in range(world.n_domains)
    ], dtype=float)

    agents      = list(world.agents)
    total_steps = sum(
        a.training_steps + a.exploit_steps + a.explore_steps for a in agents
    )
    train_frac = (
        sum(a.training_steps for a in agents) / total_steps
        if total_steps > 0 else 0.0
    )

    mean_truth = float(np.mean([
        max((m.actual_truthfulness for m in world.scientific_models if m.domain == d),
            default=0.0)
        for d in range(world.n_domains)
    ]))

    return {
        'fail_rate':  fail_rate,
        'gini':       _gini(counts),
        'train_frac': train_frac,
        'mean_truth': mean_truth,
    }


# ---------------------------------------------------------------------------
# Module-level worker (must be at top level for multiprocessing 'spawn')
# ---------------------------------------------------------------------------

def _worker(task: tuple) -> dict:
    """Run one simulation and return metrics. Called in a subprocess."""
    from world import ScienceWorld   # local import — each worker loads its own copy
    scenario_kw, model_kw, seed = task
    world = ScienceWorld(**scenario_kw, **model_kw, rng=seed)
    for _ in range(STEPS):
        world.step()
    result = _metrics(world)
    result['_task'] = task          # carry task identity back to the parent
    return result


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_sensitivity() -> pd.DataFrame:
    # Build flat task list: (scenario_kw, model_kw, seed)
    # Each entry is one independent simulation.
    task_meta = []   # parallel list of (param_name, level_name, val, scenario_label)
    tasks     = []

    for param_name, base_val in BASE.items():
        levels = {
            'low':  base_val * (1 - VARY_FACTOR),
            'base': base_val,
            'high': base_val * (1 + VARY_FACTOR),
        }
        for level_name, val in levels.items():
            model_kw = {**BASE, param_name: val}
            for scenario_label, scenario_kw in [('spec', PEAKED),
                                                 ('gen',  BROAD)]:
                for i in range(N_RUNS):
                    tasks.append((scenario_kw, model_kw, SEED_BASE + i))
                    task_meta.append((param_name, level_name, round(val, 6),
                                      scenario_label))

    n_tasks = len(tasks)
    n_cores = min(cpu_count(), n_tasks)
    print(f"Sensitivity analysis")
    print(f"  {len(BASE)} parameters × 3 levels × 2 scenarios × {N_RUNS} runs"
          f" = {n_tasks} simulations × {STEPS} steps")
    print(f"  Running on {n_cores} cores — estimated time: "
          f"~{n_tasks // n_cores * 15 // 60 + 1} min\n")

    raw_results = []
    done = 0
    with Pool(n_cores) as pool:
        for result in pool.imap(_worker, tasks):
            raw_results.append(result)
            done += 1
            if done % (N_RUNS * 2) == 0:       # print after each full condition
                pname, level, val, _ = task_meta[done - 1]
                print(f"  [{done:>3}/{n_tasks}]  {pname}  {level}  {val:.4f}")

    # Aggregate: average over N_RUNS for each (param, level, scenario)
    rows = []
    for idx, (param_name, level_name, val, scenario_label) in enumerate(task_meta):
        raw_results[idx]['param_name']     = param_name
        raw_results[idx]['level_name']     = level_name
        raw_results[idx]['val']            = val
        raw_results[idx]['scenario_label'] = scenario_label

    df_raw = pd.DataFrame([
        {k: v for k, v in r.items() if k != '_task'}
        for r in raw_results
    ])

    for (param_name, level_name, val), grp in df_raw.groupby(
            ['param_name', 'level_name', 'val']):
        spec = grp[grp['scenario_label'] == 'spec']
        gen  = grp[grp['scenario_label'] == 'gen']

        def avg(col, subset): return float(subset[col].mean())

        fail_delta = avg('fail_rate', spec) - avg('fail_rate', gen)
        gini_delta = avg('gini', spec)      - avg('gini', gen)
        robust     = (fail_delta > 0) and (gini_delta > 0)

        rows.append({
            'parameter':  param_name,
            'level':      level_name,
            'value':      round(val,                         4),
            'spec_fail':  round(avg('fail_rate',  spec),     3),
            'gen_fail':   round(avg('fail_rate',  gen),      3),
            'delta_fail': round(fail_delta,                  3),
            'spec_gini':  round(avg('gini',       spec),     3),
            'gen_gini':   round(avg('gini',       gen),      3),
            'delta_gini': round(gini_delta,                  3),
            'spec_train': round(avg('train_frac', spec),     3),
            'gen_train':  round(avg('train_frac', gen),      3),
            'spec_truth': round(avg('mean_truth', spec),     3),
            'gen_truth':  round(avg('mean_truth', gen),      3),
            'robust':     robust,
        })

    df = pd.DataFrame(rows)
    # Sort by parameter (insertion order of BASE) then level
    level_order = {'low': 0, 'base': 1, 'high': 2}
    param_order = {p: i for i, p in enumerate(BASE)}
    df['_po'] = df['parameter'].map(param_order)
    df['_lo'] = df['level'].map(level_order)
    df = df.sort_values(['_po', '_lo']).drop(columns=['_po', '_lo'])

    out = Path(__file__).parent.parent / 'meta' / 'sensitivity_results.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


def print_table(df: pd.DataFrame):
    W = 105
    print("\n" + "=" * W)
    print(
        f"{'Parameter':<22} {'Level':<5} {'Value':>7}  "
        f"{'Fail(S)':>7} {'Fail(G)':>7} {'ΔFail':>7}  "
        f"{'Gini(S)':>7} {'Gini(G)':>7} {'ΔGini':>7}  "
        f"{'Train(S)':>8} {'Truth(S)':>8}  Robust"
    )
    print("-" * W)
    prev_param = None
    for _, row in df.iterrows():
        if prev_param and row['parameter'] != prev_param:
            print()
        prev_param = row['parameter']

        base_marker = " ◄" if row['level'] == 'base' else "  "
        robust_str  = "yes" if row['robust'] else "NO !"
        print(
            f"{row['parameter']:<22} {row['level']:<5} {row['value']:>7.4f}  "
            f"{row['spec_fail']:>7.3f} {row['gen_fail']:>7.3f} {row['delta_fail']:>+7.3f}  "
            f"{row['spec_gini']:>7.3f} {row['gen_gini']:>7.3f} {row['delta_gini']:>+7.3f}  "
            f"{row['spec_train']:>8.3f} {row['spec_truth']:>8.3f}  {robust_str}{base_marker}"
        )
    print("=" * W)

    n_robust = df['robust'].sum()
    n_total  = len(df)
    print(f"\nRobust conditions: {n_robust}/{n_total} ({100*n_robust/n_total:.0f}%)")
    flips = df[~df['robust']]
    if not flips.empty:
        print("Qualitative flips:")
        for _, row in flips.iterrows():
            print(f"  {row['parameter']}  {row['level']}  ({row['value']:.4f})"
                  f"  ΔFail={row['delta_fail']:+.3f}  ΔGini={row['delta_gini']:+.3f}")


if __name__ == "__main__":
    df = run_sensitivity()
    print_table(df)
