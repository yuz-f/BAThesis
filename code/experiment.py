"""
30-seed confirmatory experiment.

MODEL SPECIFICATION — 2026-05-07  (v3: epistemic landscape, +FLAT control)
  Scenarios    : PEAKED    (peak_skill_mean=0.55, peak_skill_std=0.07,
                          other_skill_mean=0.25, other_skill_std=0.06)
                 BROAD     (peak_skill_mean=0.55, peak_skill_std=0.08,
                          other_skill_mean=0.33, other_skill_std=0.08)
                 FLAT (peak_skill_mean=0.28, peak_skill_std=0.02,
                          other_skill_mean=0.28, other_skill_std=0.02) — control:
                          all researchers have near-identical flat profiles
                          at the PEAKED per-researcher mean (0.28), so the only
                          difference vs. PEAKED is the shape of the skill
                          distribution. Tests whether H1/H2 effects require
                          heterogeneity once mean skill is held constant.
  Shared params: selection_interval=40, n_labs=10, researchers_per_lab=5,
                 n_domains=10, train_threshold=0.30, skill_gain_attempt=0.06,
                 skill_gain_train=0.08, cull_fraction=0.25, mutation_std=0.04,
                 social_learn_strength=0.30, misconduct_base_rate=0.05

  Key formulae:
    success_probability = (sim + avg*(1-sim))
                          * (actual/reported)
                          * (0.75 + 0.25*landscape_stability)

    skill_bias = (domain_skill / mean_skill) ** 0.5

    bias_inflation [normal]      ~ clip(N(0.10 + pressure*0.08 + landscape_pb, 0.05), 0, 0.45)
    bias_inflation [breakthrough]~ clip(N(0.05, 0.03) + landscape_pb,  0, 0.45)
    bias_inflation [misconduct]  ~ clip(N(0.22, 0.06) + landscape_pb, 0, 0.45)

  Mechanics (v1): breakthrough (skill²×0.10; salience shock ×0.35; high truthfulness),
                  expert truth correction (proficiency>0.50 failures erode actual_truthfulness)
  Mechanics (v2): career stages (explore boost decaying over 150 steps),
                  social learning (lab domain success signal, λ=0.30, decay×0.90/step),
                  competitive pressure bias (pressure ∝ 1 − rep/median_rep),
                  misconduct pathway (p_base=0.05, scales with pressure),
                  Matthew effect (reputation amplifies salience in exploit values)
  Mechanics (v3): epistemic landscape per domain — 3 Gaussian valleys + 4 peaks;
                  researcher theory-space positions converge via gradient descent;
                  stability modifies replication probability, debunk vulnerability,
                  salience decay (up to −25%), and publication bias inflation (up to +0.15)

  Steps        : 300 per run
  Seeds        : 0..29 (30 per scenario)
  Last updated : 2026-05-07
"""
from __future__ import annotations

import sys, os, csv
import numpy as np
from multiprocessing import Pool
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

STEPS  = 300
SEEDS  = list(range(30))

PEAKED = dict(
    peak_skill_mean=0.55, peak_skill_std=0.07,
    other_skill_mean=0.25, other_skill_std=0.06,
    selection_interval=40,
)
BROAD = dict(
    peak_skill_mean=0.55, peak_skill_std=0.08,
    other_skill_mean=0.33, other_skill_std=0.08,
    selection_interval=40,
)
# Flat scenario — matched to PEAKED's per-researcher mean skill of 0.28
# (= (0.55 + 9·0.25)/10). This removes the mean-skill confound: any
# H1/H2 difference between PEAKED and Flat is then attributable to the
# *shape* of the skill distribution, not its overall magnitude.
FLAT = dict(
    peak_skill_mean=0.28, peak_skill_std=0.02,
    other_skill_mean=0.28, other_skill_std=0.02,
    selection_interval=40,
)
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
