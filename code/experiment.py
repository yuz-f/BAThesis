"""
30-seed confirmatory experiment.

MODEL SPECIFICATION — best-model branch (mean-constant reparameterisation)
  Scenarios are defined by (mean_skill, gap) rather than (peak, other)
  directly. With one peak domain and nine non-peak domains:
        peak  = mean_skill + 0.9 · gap
        other = mean_skill - 0.1 · gap
  so (peak + 9·other)/10 == mean_skill exactly, for any gap. This holds
  the per-researcher mean skill CONSTANT across all three scenarios and
  varies only the dispersion (gap), removing the mean-skill confound
  that the earlier (peak, other) parameterisation carried — under which
  PEAKED had a lower mean (0.28) than BROAD (0.35).

  Scenarios (mean_skill = 0.32 for all):
    PEAKED  gap=0.40 → peak=0.68, other=0.28   (concentrated competence)
    BROAD   gap=0.18 → peak=0.482, other=0.302 (distributed competence)
    FLAT    gap=0.00 → peak=other=0.32         (uniform; no preferred domain)

  The Peaked-vs-Broad contrast is therefore now a clean shape-only
  manipulation at constant total competence. The trade-off: holding mean
  constant means peak skill varies between scenarios (Peaked concentrates
  the same budget into a higher peak) — which is the correct
  operationalisation of specialisation as budget concentration.

  Shared params: selection_interval=40, n_labs=10, researchers_per_lab=5,
                 n_domains=10, train_threshold=0.30, skill_gain_attempt=0.06,
                 skill_gain_train=0.08, cull_fraction=0.25, mutation_std=0.04,
                 social_learn_strength=0.30, misconduct_base_rate=0.05

  Mechanics: best-model branch — redesigned epistemic landscape (plateaus +
             peaks, gradient-based stability), gradient-aware debunk
             threshold (Equation 5a), quadratic-in-gradient debunk
             instability boost, optional Type B action selection.

  Steps        : 300 per run
  Seeds        : 0..29 (30 per scenario)
"""
from __future__ import annotations

import sys, os, csv
import numpy as np
from multiprocessing import Pool
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

STEPS  = 300
SEEDS  = list(range(30))

# Common per-researcher mean skill, held constant across all scenarios.
MEAN_SKILL = 0.32


def make_scenario(mean_skill: float, gap: float,
                  peak_std: float = 0.07, other_std: float = 0.06,
                  selection_interval: int = 40) -> dict:
    """
    Build a scenario dict from (mean_skill, gap).

    With one peak domain and nine non-peak domains, the per-researcher
    mean across all ten domains is held exactly equal to mean_skill:
        peak  = mean_skill + 0.9 * gap
        other = mean_skill - 0.1 * gap
        (peak + 9 * other) / 10 == mean_skill   for any gap

    gap is the dispersion of the skill profile: gap=0 gives a flat
    profile, large gap gives a sharply peaked one. Holding mean_skill
    fixed and varying only gap isolates distribution *shape* from
    overall competence.
    """
    return dict(
        peak_skill_mean  = mean_skill + 0.9 * gap,
        peak_skill_std   = peak_std,
        other_skill_mean = mean_skill - 0.1 * gap,
        other_skill_std  = other_std,
        selection_interval = selection_interval,
    )


# Peaked: concentrated competence — one sharp peak, weak elsewhere.
PEAKED = make_scenario(MEAN_SKILL, gap=0.40, peak_std=0.07, other_std=0.06)
# Broad: distributed competence — moderate peak, raised baseline.
BROAD  = make_scenario(MEAN_SKILL, gap=0.18, peak_std=0.07, other_std=0.06)
# Flat: uniform competence — no preferred domain (gap=0). Low std so
# all researchers are near-identical, the no-heterogeneity control.
FLAT   = make_scenario(MEAN_SKILL, gap=0.00, peak_std=0.02, other_std=0.02)

SCENARIOS = {"PEAKED": PEAKED, "BROAD": BROAD, "FLAT": FLAT}

SHARED = dict(
    train_threshold=0.30,
    skill_gain_attempt=0.06,
)


def _gini(vals: list[float]) -> float:
    vals = sorted(v for v in vals if v > 0)
    n = len(vals)
    if n == 0:
        return 0.0
    s = sum(vals)
    return float((2 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * s)) - (n + 1) / n)


def _worker(task: tuple) -> dict:
    label, seed = task
    from world import ScienceWorld
    from collections import Counter

    scenario = SCENARIOS[label]
    w = ScienceWorld(rng=seed, **scenario, **SHARED)

    action_counts = {"exploit": 0, "explore": 0, "train": 0}
    prev = {}

    for step in range(STEPS):
        prev = {a.unique_id: (a.exploit_steps, a.explore_steps, a.training_steps)
                for a in w.agents}
        w.step()
        for a in w.agents:
            if a.unique_id not in prev:
                continue
            de = a.exploit_steps  - prev[a.unique_id][0]
            dr = a.explore_steps  - prev[a.unique_id][1]
            dt = a.training_steps - prev[a.unique_id][2]
            action_counts["exploit"] += de
            action_counts["explore"] += dr
            action_counts["train"]   += dt

    agents = list(w.agents)
    total_steps = sum(action_counts.values()) or 1

    # failure rate
    fail_rate = w.replication_failures / max(w.replication_attempts, 1)

    # Gini of publications per domain
    pub_counts: Counter = Counter()
    for a in agents:
        for d, n in a.domain_pubs.items():
            pub_counts[d] += n
    gini = _gini(list(pub_counts.values()))

    # mean best actual / reported truthfulness per domain
    actual_by_domain = [
        max((m.actual_truthfulness   for m in w.scientific_models if m.domain == d), default=0.0)
        for d in range(w.n_domains)
    ]
    reported_by_domain = [
        max((m.reported_truthfulness for m in w.scientific_models if m.domain == d), default=0.0)
        for d in range(w.n_domains)
    ]
    mean_actual   = float(np.mean(actual_by_domain))
    mean_reported = float(np.mean(reported_by_domain))
    bias_gap      = mean_reported - mean_actual

    # reputation dynamics (H3)
    reputations = [a.reputation for a in agents]
    mean_rep    = float(np.mean(reputations))
    var_rep     = float(np.var(reputations))

    # debunk stability (H4): reputation lost to debunking as fraction of total earned
    rep_lost  = [a.reputation_lost_to_debunk for a in agents]
    mean_lost = float(np.mean(rep_lost))
    # debunk impact rate: lost / (current + lost), i.e. share of career earnings wiped
    debunk_impact = float(np.mean([
        r_lost / max(r_curr + r_lost, 1e-8)
        for r_curr, r_lost in zip(reputations, rep_lost)
    ]))

    # action fractions
    exploit_frac = action_counts["exploit"] / total_steps
    explore_frac = action_counts["explore"] / total_steps
    train_frac   = action_counts["train"]   / total_steps

    return dict(
        label=label, seed=seed,
        fail_rate=fail_rate, gini=gini,
        mean_actual=mean_actual, mean_reported=mean_reported, bias_gap=bias_gap,
        mean_reputation=mean_rep, reputation_variance=var_rep,
        mean_reputation_lost=mean_lost, debunk_impact_rate=debunk_impact,
        exploit_frac=exploit_frac, explore_frac=explore_frac, train_frac=train_frac,
    )


def _ci95(vals: list[float]) -> tuple[float, float, float]:
    """Return (mean, lower_95CI, upper_95CI) using t-distribution."""
    arr = np.array(vals)
    n   = len(arr)
    m   = float(arr.mean())
    se  = float(arr.std(ddof=1) / np.sqrt(n))
    t   = float(stats.t.ppf(0.975, df=n - 1))
    return m, m - t * se, m + t * se


def _cohen_d(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2)
        / (na + nb - 2)
    )
    return float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 1e-10 else 0.0


def _report(label: str, metric: str, vals: list[float]):
    m, lo, hi = _ci95(vals)
    print(f"  {label:5s}  {metric:22s}  mean={m:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    LABELS = ("PEAKED", "BROAD", "FLAT")
    tasks = [(label, seed)
             for label in LABELS
             for seed in SEEDS]

    print(f"Running {len(tasks)} simulations ({len(SEEDS)} seeds × {len(LABELS)} scenarios × {STEPS} steps)…")
    with Pool() as pool:
        results = pool.map(_worker, tasks)

    # save raw results
    out_dir = os.path.join(os.path.dirname(__file__), "..", "meta")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "experiment_results.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nRaw results saved → {csv_path}\n")

    # split by scenario
    by_label: dict[str, list[dict]] = {lbl: [] for lbl in LABELS}
    for r in results:
        by_label[r["label"]].append(r)

    # ── descriptive statistics ────────────────────────────────────────────────
    print("=" * 70)
    print("DESCRIPTIVE STATISTICS  (30 seeds × 300 steps per scenario)")
    print("=" * 70)
    metrics = [
        ("fail_rate",            "Replication fail rate"),
        ("gini",                 "Gini (pub domain conc.)"),
        ("mean_actual",          "Mean actual truthfulness"),
        ("mean_reported",        "Mean reported truthfulness"),
        ("bias_gap",             "Bias gap (reported-actual)"),
        ("mean_reputation",      "Mean researcher reputation"),
        ("reputation_variance",  "Reputation variance"),
        ("mean_reputation_lost", "Mean reputation lost to debunk"),
        ("debunk_impact_rate",   "Debunk impact rate"),
        ("exploit_frac",         "Exploit fraction"),
        ("explore_frac",         "Explore fraction"),
        ("train_frac",           "Train fraction"),
    ]
    for key, name in metrics:
        for lbl in LABELS:
            vals = [r[key] for r in by_label[lbl]]
            _report(lbl, name, vals)
        print()

    # ── inferential statistics ────────────────────────────────────────────────
    inf_metrics = [
        ("fail_rate",          "H1: Replication fail rate"),
        ("gini",               "H2: Gini (pub domain conc.)"),
        ("mean_reputation",    "H3: Mean researcher reputation"),
        ("debunk_impact_rate", "H4: Debunk impact rate"),
        ("mean_actual",        "Exploratory: Mean actual truthfulness"),
        ("bias_gap",           "Exploratory: Bias gap"),
    ]

    def _print_pair(a_lbl: str, b_lbl: str):
        for key, name in inf_metrics:
            a_vals = [r[key] for r in by_label[a_lbl]]
            b_vals = [r[key] for r in by_label[b_lbl]]
            u_stat, p_val = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            d = _cohen_d(a_vals, b_vals)
            direction = f"{a_lbl} > {b_lbl}" if np.mean(a_vals) > np.mean(b_vals) else f"{b_lbl} > {a_lbl}"
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
            print(f"  {name}")
            print(f"    {a_lbl}={np.mean(a_vals):.4f}  {b_lbl}={np.mean(b_vals):.4f}  ({direction})")
            print(f"    Mann-Whitney U={u_stat:.1f}  p={p_val:.4f} {sig}  Cohen's d={d:.3f}")
            print()

    print("=" * 70); print("INFERENTIAL STATISTICS  (PEAKED vs BROAD)"); print("=" * 70)
    _print_pair("PEAKED", "BROAD")

    print("=" * 70); print("CONTROL CHECK  (PEAKED vs FLAT)  — H1/H2 should be amplified"); print("=" * 70)
    _print_pair("PEAKED", "FLAT")

    print("=" * 70); print("CONTROL CHECK  (BROAD vs FLAT)  — H1/H2 should attenuate or vanish"); print("=" * 70)
    _print_pair("BROAD", "FLAT")
