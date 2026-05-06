"""
grid_search.py — calibration grid for train_threshold × skill_gain_attempt.

Goal: find parameter combinations where the specialist scenario produces a
steady-state replication failure rate close to the OSC (2015) benchmark of ~60%.
All runs are parallelized across available CPU cores.

Usage:
    cd BAThesis/code
    .venv/bin/python3 grid_search.py

Output:
    grid_search_results.csv              — full data table
    meta/img/debug/grid_search.png       — annotated heatmap
    console                              — target-zone summary
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STEPS     = 200
N_RUNS    = 3
SEED_BASE = 99

TARGET_LOW  = 0.50
TARGET_HIGH = 0.65   # OSC 2015 overall ≈ 0.60

TRAIN_THRESHOLDS    = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
SKILL_GAIN_ATTEMPTS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

BASE_MODEL = dict(
    skill_gain_train=0.08,
    cull_fraction=0.25,
    cap_growth_rate=0.005,
    mutation_std=0.04,
)
SPECIALIST = dict(
    peak_skill_mean=0.55, peak_skill_std=0.07,
    other_skill_mean=0.25, other_skill_std=0.06,
)

IMG_OUT = (Path(__file__).parent.parent
           / 'meta' / 'img' / 'debug' / 'grid_search.png')

# ---------------------------------------------------------------------------
# Module-level worker
# ---------------------------------------------------------------------------

def _worker(task: tuple) -> float:
    """Run one simulation, return failure rate. Called in a subprocess."""
    from world import ScienceWorld
    tt, sga, seed = task
    world = ScienceWorld(
        **SPECIALIST,
        train_threshold=tt,
        skill_gain_attempt=sga,
        **BASE_MODEL,
        rng=seed,
    )
    for _ in range(STEPS):
        world.step()
    return (world.replication_failures / world.replication_attempts
            if world.replication_attempts > 0 else 0.0)


# ---------------------------------------------------------------------------
# Grid run
# ---------------------------------------------------------------------------

def run_grid() -> pd.DataFrame:
    tasks = [
        (tt, sga, SEED_BASE + i)
        for tt  in TRAIN_THRESHOLDS
        for sga in SKILL_GAIN_ATTEMPTS
        for i   in range(N_RUNS)
    ]
    task_keys = [
        (tt, sga)
        for tt  in TRAIN_THRESHOLDS
        for sga in SKILL_GAIN_ATTEMPTS
        for _   in range(N_RUNS)
    ]

    n_cells = len(TRAIN_THRESHOLDS) * len(SKILL_GAIN_ATTEMPTS)
    n_cores = min(cpu_count(), len(tasks))
    print(f"Grid search — {n_cells} cells × {N_RUNS} runs = {len(tasks)} simulations × {STEPS} steps")
    print(f"Running on {n_cores} cores")
    print(f"Target zone: {TARGET_LOW:.0%}–{TARGET_HIGH:.0%}  (OSC 2015 ≈ 60%)\n")

    raw = []
    done = 0
    with Pool(n_cores) as pool:
        for rate in pool.imap(_worker, tasks):
            raw.append(rate)
            done += 1
            if done % N_RUNS == 0:
                tt, sga = task_keys[done - 1]
                run_rates = raw[done - N_RUNS: done]
                mean = float(np.mean(run_rates))
                flag = "  ✓ TARGET" if TARGET_LOW <= mean <= TARGET_HIGH else ""
                print(f"  tt={tt:.2f}  sga={sga:.2f}  →  fail={mean:.3f}{flag}")

    # Aggregate per cell
    rows = []
    idx = 0
    for tt in TRAIN_THRESHOLDS:
        for sga in SKILL_GAIN_ATTEMPTS:
            cell_rates = raw[idx: idx + N_RUNS]
            idx += N_RUNS
            mean = float(np.mean(cell_rates))
            std  = float(np.std(cell_rates))
            rows.append({
                'train_threshold':    tt,
                'skill_gain_attempt': sga,
                'mean_fail_rate':     round(mean, 3),
                'std_fail_rate':      round(std,  3),
                'in_target':          TARGET_LOW <= mean <= TARGET_HIGH,
            })

    df = pd.DataFrame(rows)
    csv_out = Path(__file__).parent / 'grid_search_results.csv'
    df.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")
    return df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame):
    pivot = df.pivot(
        index='train_threshold',
        columns='skill_gain_attempt',
        values='mean_fail_rate',
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        pivot.values,
        cmap='RdYlGn_r',
        vmin=0.30, vmax=0.80,
        aspect='auto',
        origin='lower',
    )

    ax.set_xticks(range(len(SKILL_GAIN_ATTEMPTS)))
    ax.set_xticklabels([f"{v:.2f}" for v in SKILL_GAIN_ATTEMPTS])
    ax.set_yticks(range(len(TRAIN_THRESHOLDS)))
    ax.set_yticklabels([f"{v:.2f}" for v in TRAIN_THRESHOLDS])
    ax.set_xlabel("skill_gain_attempt")
    ax.set_ylabel("train_threshold")
    ax.set_title(
        f"Replication failure rate — specialist scenario\n"
        f"Bold/outlined = target zone ({TARGET_LOW:.0%}–{TARGET_HIGH:.0%},  OSC 2015 ≈ 60%)"
    )

    for i, tt in enumerate(TRAIN_THRESHOLDS):
        for j, sga in enumerate(SKILL_GAIN_ATTEMPTS):
            val = pivot.loc[tt, sga]
            hit = TARGET_LOW <= val <= TARGET_HIGH
            ax.text(
                j, i, f"{val:.2f}",
                ha='center', va='center',
                fontsize=8.5,
                fontweight='bold' if hit else 'normal',
                color='white' if hit else '#333333',
            )

    # outline current base-parameter cell
    base_tt, base_sga = 0.35, 0.05
    if base_tt in TRAIN_THRESHOLDS and base_sga in SKILL_GAIN_ATTEMPTS:
        bi = TRAIN_THRESHOLDS.index(base_tt)
        bj = SKILL_GAIN_ATTEMPTS.index(base_sga)
        ax.add_patch(plt.Rectangle(
            (bj - 0.5, bi - 0.5), 1, 1,
            fill=False, edgecolor='black', linewidth=2.5, linestyle='--',
        ))

    plt.colorbar(im, ax=ax, label='Mean failure rate')
    fig.tight_layout()
    IMG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {IMG_OUT}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    targets = df[df['in_target']].sort_values('std_fail_rate')
    print(f"\nTarget zone ({TARGET_LOW:.0%}–{TARGET_HIGH:.0%}) — "
          f"{len(targets)}/{len(df)} cells:\n")
    if targets.empty:
        print("  (none — the model's failure rate sits outside the OSC 2015 range;")
        print("   see Discussion for interpretation)")
        return

    print(f"  {'train_threshold':>16}  {'skill_gain_attempt':>18}  "
          f"{'mean_fail':>9}  {'std_fail':>8}")
    print("  " + "-" * 58)
    for _, row in targets.iterrows():
        base_flag = (
            " ◄ current base"
            if row['train_threshold'] == 0.35
            and row['skill_gain_attempt'] == 0.05
            else ""
        )
        print(
            f"  {row['train_threshold']:>16.2f}  {row['skill_gain_attempt']:>18.2f}  "
            f"{row['mean_fail_rate']:>9.3f}  {row['std_fail_rate']:>8.3f}{base_flag}"
        )


if __name__ == "__main__":
    df = run_grid()
    plot_heatmap(df)
    print_summary(df)
