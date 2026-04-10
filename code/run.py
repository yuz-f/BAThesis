"""
run.py — entry point for the Science Lab ABM.

Run from the BA/ folder:
    python -m abm.run
"""

from world import ScienceWorld
from visualization import plot_lab_history, plot_scenarios

STEPS      = 200
PEAK_SKILL = 0.70

LOW_SKILL     = 0.20   # skill in non-peak domains — siloed scenario
HIGH_SKILL    = 0.40   # skill in non-peak domains — broad scenario


def run_scenario(other_skill_mean: float, steps: int = STEPS, rng=None):
    world = ScienceWorld(
        other_skill_mean=other_skill_mean,
        peak_skill_mean=PEAK_SKILL,
        rng=rng,    
    )
    for _ in range(steps):
        world.step()
    mdf = world.datacollector.get_model_vars_dataframe()
    adf = world.datacollector.get_agent_vars_dataframe()
    return mdf, adf


def print_domain_diagnostics(world: ScienceWorld):
    print("\nDomain | Cap   | Max Fidelity | Saturation | Models")
    print("-" * 58)
    for d in range(world.n_domains):
        models  = [m for m in world.scientific_models if m.domain == d]
        cap     = world.domain_fidelity_caps[d]
        max_fid = max((m.fidelity for m in models), default=0.0)
        sat     = max_fid / cap if cap > 0 else 0.0
        print(f"  D{d:<3}  | {cap:.2f} | {max_fid:.2f}         | {sat:.0%}        | {len(models)}")


if __name__ == "__main__":
    # use a shared seed so both scenarios start from the same random draw
    # set rng=None for different results each run
    rng = None

    print("Running scenarios...")
    world_low  = ScienceWorld(other_skill_mean=LOW_SKILL,  peak_skill_mean=PEAK_SKILL, rng=rng)
    world_high = ScienceWorld(other_skill_mean=HIGH_SKILL, peak_skill_mean=PEAK_SKILL, rng=rng)
    for _ in range(STEPS):
        world_low.step()
        world_high.step()
    df_low  = world_low.datacollector.get_model_vars_dataframe()
    adf_low = world_low.datacollector.get_agent_vars_dataframe()
    df_high  = world_high.datacollector.get_model_vars_dataframe()
    adf_high = world_high.datacollector.get_agent_vars_dataframe()

    print("\n--- Low skill scenario ---")
    print_domain_diagnostics(world_low)
    print("\n--- High skill scenario ---")
    print_domain_diagnostics(world_high)

    n_domains = ScienceWorld().n_domains
    plot_lab_history(adf_low, adf_high, n_domains=n_domains)
    plot_scenarios(df_low, df_high, adf_low, adf_high, steps=STEPS)
