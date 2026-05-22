"""
run.py — entry point for the Science Lab ABM.

Run from the BAThesis/code/ folder:
    .venv/bin/python3 run.py

Scenarios (mean-constant reparameterisation, mean_skill = 0.32)
---------
LOW  (peaked):  gap=0.40 → peak=0.68, other=0.28
                    → competence concentrated in one niche; domain monopolies form

HIGH (broad):  gap=0.18 → peak=0.482, other=0.302
                    → same total competence spread across more domains
"""

from world import ScienceWorld
from visualization import (
    plot_lab_history, plot_scenarios, plot_agent_skills,
    plot_cluster_animation, plot_knowledge_quality, plot_action_over_time,
)

STEPS = 300

# Mean-constant reparameterisation: peak = m + 0.9·gap, other = m - 0.1·gap.
MEAN_SKILL = 0.32

# peaked scenario — competence concentrated (gap = 0.40)
LOW_PEAK_MEAN  = MEAN_SKILL + 0.9 * 0.40   # 0.68
LOW_PEAK_STD   = 0.07
LOW_OTHER_MEAN = MEAN_SKILL - 0.1 * 0.40   # 0.28
LOW_OTHER_STD  = 0.06

# broad scenario — competence distributed (gap = 0.18)
HIGH_PEAK_MEAN  = MEAN_SKILL + 0.9 * 0.18  # 0.482
HIGH_PEAK_STD   = 0.07
HIGH_OTHER_MEAN = MEAN_SKILL - 0.1 * 0.18  # 0.302
HIGH_OTHER_STD  = 0.06


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

    print_domain_diagnostics(world_low,  "Peaked (gap=0.40)")
    print_domain_diagnostics(world_high, "Broad (gap=0.18)")

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
