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
from visualization import plot_lab_history, plot_scenarios, plot_agent_skills

STEPS = 200

# specialist scenario — one clear but moderate peak, weaker elsewhere
# peak ≈ 0.55, other ≈ 0.25 → ratio ~2.2×, gap ~0.30
LOW_PEAK_MEAN  = 0.55
LOW_PEAK_STD   = 0.07
LOW_OTHER_MEAN = 0.25
LOW_OTHER_STD  = 0.06

# generalist scenario — same moderate peak, broader floor with more overlap
# peak ≈ 0.55, other ≈ 0.40 → ratio ~1.4×, gap ~0.15
HIGH_PEAK_MEAN  = 0.55
HIGH_PEAK_STD   = 0.08
HIGH_OTHER_MEAN = 0.40
HIGH_OTHER_STD  = 0.10


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
    print("Domain | Cap   | Raises | Max Fidelity | Saturation | Models")
    print("-" * 67)
    for d in range(world.n_domains):
        models  = [m for m in world.scientific_models if m.domain == d]
        cap     = world.domain_truthfulness_caps[d]
        raises  = world.domain_cap_raises[d]
        max_fid = max((m.truthfulness for m in models), default=0.0)
        sat     = max_fid / cap if cap > 0 else 0.0
        print(f"  D{d:<3}  | {cap:.2f} |   {raises}    | {max_fid:.2f}         | {sat:.0%}        | {len(models)}")


if __name__ == "__main__":
    rng = None   # set to an int for reproducible runs

    print("Running scenarios...")
    world_low,  df_low,  adf_low  = run_scenario(
        LOW_PEAK_MEAN,  LOW_PEAK_STD,  LOW_OTHER_MEAN,  LOW_OTHER_STD,  rng=rng)
    world_high, df_high, adf_high = run_scenario(
        HIGH_PEAK_MEAN, HIGH_PEAK_STD, HIGH_OTHER_MEAN, HIGH_OTHER_STD, rng=rng)

    print_domain_diagnostics(world_low,  "Low skill  — specialist (μ_other=0.05)")
    print_domain_diagnostics(world_high, "High skill — generalist (μ_other=0.45)")

    n_domains = world_low.n_domains
    plot_lab_history(adf_low, adf_high, n_domains=n_domains)
    plot_scenarios(df_low, df_high, adf_low, adf_high, steps=STEPS)
    plot_agent_skills(adf_low, adf_high, n_domains=n_domains)
