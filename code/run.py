"""
run.py — entry point for the Science Lab ABM.

Run from the BA/abm/ folder:
    python run.py

Scenarios
---------
LOW  (specialist):  one sharp peak, near-zero everywhere else
                    → labs are trapped in their niche, domain monopolies form

HIGH (generalist):  same peak but wide distribution for other domains
                    → natural secondary peaks emerge, labs can roam freely
"""

from world import ScienceWorld
from visualization import (
    plot_lab_history, plot_scenarios, plot_agent_skills,
    plot_cluster_animation, plot_knowledge_quality, plot_action_over_time,
)

STEPS = 300

# specialist scenario — one clear but moderate peak, weaker elsewhere
# peak ≈ 0.55, other ≈ 0.25 → ratio ~2.2×, gap ~0.30
LOW_PEAK_MEAN  = 0.55
LOW_PEAK_STD   = 0.07
LOW_OTHER_MEAN = 0.25
LOW_OTHER_STD  = 0.06

# generalist scenario — same moderate peak, broader floor with more overlap
# peak ≈ 0.55, other ≈ 0.33 → ratio ~1.7×, gap ~0.22
HIGH_PEAK_MEAN  = 0.55
HIGH_PEAK_STD   = 0.08
HIGH_OTHER_MEAN = 0.33
HIGH_OTHER_STD  = 0.08


def run_scenario(peak_mean, peak_std, other_mean, other_std,
                 steps: int = STEPS, rng=None):
    world = ScienceWorld(
        peak_skill_mean=peak_mean,
        peak_skill_std=peak_std,
        other_skill_mean=other_mean,
        other_skill_std=other_std,
        rng=rng,
    )
    for _ in range(steps):
        world.step()
    mdf = world.datacollector.get_model_vars_dataframe()
    adf = world.datacollector.get_agent_vars_dataframe()
    return world, mdf, adf


def print_domain_diagnostics(world: ScienceWorld, label: str):
    print(f"\n--- {label} ---")
    print(f"  Labs: {world.n_labs}  |  Researchers: {len(list(world.agents))}")
    print("Domain | Cap   | Max Truth | Saturation | Models")
    print("-" * 53)
    for d in range(world.n_domains):
        models = [m for m in world.scientific_models if m.domain == d]
        cap    = world.domain_truthfulness_caps[d]
        max_t  = max((m.actual_truthfulness for m in models), default=0.0)
        sat    = max_t / cap if cap > 0 else 0.0
        print(f"  D{d:<3}  | {cap:.2f} | {max_t:.2f}      | {sat:.0%}        | {len(models)}")

    if world.lab_turnover_events:
        print(f"\n  Lab turnovers ({len(world.lab_turnover_events)}):")
        for step, lab_id, old_peak, new_peak in world.lab_turnover_events:
            print(f"    step {step:>4}: Lab {lab_id}  D{old_peak} → D{new_peak}")


if __name__ == "__main__":
    rng = None   # set to an int for reproducible runs

    print("Running scenarios...")
    world_low,  df_low,  adf_low  = run_scenario(
        LOW_PEAK_MEAN,  LOW_PEAK_STD,  LOW_OTHER_MEAN,  LOW_OTHER_STD,  rng=rng)
    world_high, df_high, adf_high = run_scenario(
        HIGH_PEAK_MEAN, HIGH_PEAK_STD, HIGH_OTHER_MEAN, HIGH_OTHER_STD, rng=rng)

    print_domain_diagnostics(world_low,  "Specialist (μ_other=0.25)")
    print_domain_diagnostics(world_high, "Generalist (μ_other=0.33)")

    n_domains = world_low.n_domains
    n_labs    = world_low.n_labs

    # thesis figures
    plot_scenarios(df_low, df_high, adf_low, adf_high, steps=STEPS)
    plot_knowledge_quality(df_low, df_high)
    plot_action_over_time(adf_low, adf_high)
    plot_agent_skills(adf_low, adf_high, n_domains=n_domains)
    plot_cluster_animation(
        adf_low, adf_high,
        n_labs=n_labs,
        selection_interval=world_low.selection_interval,
    )
    # debug figures (saved to meta/img/debug/)
    plot_lab_history(adf_low, adf_high, n_domains=n_domains)
