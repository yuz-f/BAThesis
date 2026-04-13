import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_agent_skills(adf_low, adf_high, n_domains: int):
    """
    Two-figure view of agent skill profiles: initial vs final, and trajectories.

    Figure 1 — Skill profile heatmaps (2 rows × 2 cols):
      Row 1: Initial profiles  (step 1, all agents present at start)
      Row 2: Final profiles    (last step, surviving agents only)
      Columns: Specialist | Generalist
      Cell colour = skill level (0–1), numeric value annotated per cell.

    Figure 2 — Skill gain trajectories (line chart, 1 row × 2 cols):
      One line per domain, value = mean skill across surviving agents at each step.
      Shows which domains each population collectively developed over time.
    """
    all_steps  = sorted(adf_low.index.get_level_values("Step").unique())
    first_step = all_steps[0]
    final_step = all_steps[-1]
    cmap_domains = plt.get_cmap("tab20", n_domains)

    def agents_at_step(adf, step):
        return sorted(adf.xs(step, level="Step").index.tolist())

    def skill_matrix(adf, step, lab_ids):
        step_df = adf.xs(step, level="Step")
        return np.array([step_df.loc[lid, "DomainSkills"] for lid in lab_ids])

    initial_low  = agents_at_step(adf_low,  first_step)
    initial_high = agents_at_step(adf_high, first_step)
    final_low    = agents_at_step(adf_low,  final_step)
    final_high   = agents_at_step(adf_high, final_step)

    mat_init_low   = skill_matrix(adf_low,  first_step, initial_low)
    mat_init_high  = skill_matrix(adf_high, first_step, initial_high)
    mat_final_low  = skill_matrix(adf_low,  final_step, final_low)
    mat_final_high = skill_matrix(adf_high, final_step, final_high)

    def _draw_heatmap(ax, mat, lab_ids, title, n_domains):
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
        ax.set_xticks(range(n_domains))
        ax.set_xticklabels([f"D{d}" for d in range(n_domains)], fontsize=7)
        ax.set_yticks(range(len(lab_ids)))
        ax.set_yticklabels([f"L{i}" for i in lab_ids], fontsize=6)
        ax.set_xlabel("Domain", fontsize=8)
        ax.set_ylabel("Lab",    fontsize=8)
        ax.set_title(title,     fontsize=9)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if val > 0.6 else "black")
        return im

    # ── Figure 1: initial vs final heatmaps ───────────────────────────────
    n_rows_low  = max(len(initial_low),  len(final_low))
    n_rows_high = max(len(initial_high), len(final_high))
    fig_h = max(5, n_rows_low * 0.35 + 2)

    fig1, axes = plt.subplots(2, 2, figsize=(14, fig_h * 2), constrained_layout=True)
    fig1.suptitle("Agent Skill Profiles — Initial vs Final", fontsize=11)

    im = _draw_heatmap(axes[0, 0], mat_init_low,   initial_low,
                       f"Specialist — Initial  (step {first_step})", n_domains)
    _draw_heatmap(axes[0, 1], mat_init_high,  initial_high,
                  f"Generalist — Initial  (step {first_step})", n_domains)
    _draw_heatmap(axes[1, 0], mat_final_low,  final_low,
                  f"Specialist — Final  (step {final_step})", n_domains)
    _draw_heatmap(axes[1, 1], mat_final_high, final_high,
                  f"Generalist — Final  (step {final_step})", n_domains)

    fig1.colorbar(im, ax=axes, label="Skill level (0–1)", shrink=0.4, pad=0.02)
    plt.savefig("agent_skill_profiles.png", dpi=150, bbox_inches="tight")

    plt.show()

    # ── Figure 2: per-domain skill trajectories ───────────────────────────
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig2.suptitle("Mean Skill per Domain Over Time\n(surviving labs only)", fontsize=11)

    def domain_trajectories(adf, survivors):
        traj = np.zeros((len(all_steps), n_domains))
        for i, step in enumerate(all_steps):
            step_df = adf.xs(step, level="Step")
            present = [lid for lid in survivors if lid in step_df.index]
            if not present:
                continue
            skills = np.array([step_df.loc[lid, "DomainSkills"] for lid in present])
            traj[i] = skills.mean(axis=0)
        return traj

    for ax, adf, survivors, label in [
        (ax3, adf_low,  final_low,  "Specialist (low-skill)"),
        (ax4, adf_high, final_high, "Generalist (high-skill)"),
    ]:
        traj = domain_trajectories(adf, survivors)
        for d in range(n_domains):
            ax.plot(all_steps, traj[:, d], color=cmap_domains(d),
                    linewidth=1.5, label=f"D{d}", alpha=0.85)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Mean skill")
        ymin, ymax = traj.min(), traj.max()
        margin = (ymax - ymin) * 0.05
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2, loc="upper left")

    plt.savefig("agent_skill_trajectories.png", dpi=150, bbox_inches="tight")

    plt.show()


def plot_lab_history(adf_low, adf_high, n_domains: int):
    """
    Scatter plot of publication events.
    Y axis = lab, X axis = time step, colour = domain published in.
    Each dot is one successful publication.
    """
    cmap = plt.get_cmap("tab20", n_domains)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    for ax, adf, label in [
        (ax1, adf_low,  "Low skill  — specialist (μ_other=0.05)"),
        (ax2, adf_high, "High skill — generalist (μ_other=0.45)"),
    ]:
        pub_steps, pub_labs, pub_domains = [], [], []

        for (step, lab_id), row in adf.iterrows():
            # check if a publication happened at this step for this lab
            prev = adf.loc[(step - 1, lab_id), "Publications"] if (step - 1, lab_id) in adf.index else 0
            if row["Publications"] > prev and pd.notna(row["CurrentDomain"]):
                pub_steps.append(step)
                pub_labs.append(lab_id)
                pub_domains.append(int(row["CurrentDomain"]))

        sc = ax.scatter(
            pub_steps, pub_labs,
            c=pub_domains, cmap=cmap, vmin=0, vmax=n_domains - 1,
            s=20, alpha=0.85, linewidths=0
        )
        all_labs = sorted(adf.index.get_level_values("AgentID").unique())
        ax.set_yticks(all_labs)
        ax.set_yticklabels([f"L{i}" for i in all_labs], fontsize=7)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Lab")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.15)

    cbar = fig.colorbar(sc, ax=[ax1, ax2], label="Domain", shrink=0.8)
    cbar.set_ticks(range(n_domains))
    cbar.set_ticklabels([f"D{d}" for d in range(n_domains)], fontsize=7)
    fig.suptitle("Publication Events per Lab", fontsize=11)

    plt.savefig("lab_history.png", dpi=150, bbox_inches="tight")

    plt.show()


def plot_scenarios(df_low, df_high, adf_low, adf_high, steps: int = 100):
    """
    Four-panel plot comparing low-skill vs high-skill scenarios.
    Panel 1: Models per domain (grouped bar at final step)
    Panel 2: Training steps per lab (grouped bar at final step)
    Panels 3-4: Time series — Avg Domain Capacity and Avg Reputation
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle(
        "Science Lab ABM — Specialist vs Generalist Populations\n"
        "(low-skill: one sharp niche  |  high-skill: broad multi-peak  |  selection every 50 steps)",
        fontsize=11
    )

    # --- panel 1: models per domain at final step ---
    ax = axes[0, 0]
    domain_counts_low  = df_low["Models per Domain"].iloc[-1]
    domain_counts_high = df_high["Models per Domain"].iloc[-1]
    domains = sorted(domain_counts_low.keys())
    x = np.arange(len(domains))
    w = 0.35
    ax.bar(x - w/2, [domain_counts_low[d]  for d in domains],
           width=w, label="Low skill",  color="#e05c5c", alpha=0.85)
    ax.bar(x + w/2, [domain_counts_high[d] for d in domains],
           width=w, label="High skill", color="#4a90d9", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}" for d in domains], fontsize=7)
    ax.set_title("Models per Domain\n(at final step)", fontsize=9)
    ax.set_ylabel("Number of models")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- panel 2: training steps per lab at final step ---
    ax = axes[0, 1]
    final_step = adf_low.index.get_level_values("Step").max()
    train_low  = adf_low.xs(final_step, level="Step")["TrainingSteps"].sort_index()
    train_high = adf_high.xs(final_step, level="Step")["TrainingSteps"].sort_index()
    lab_ids = np.arange(len(train_low))
    w = 0.35
    ax.bar(lab_ids - w/2, train_low.values,
           width=w, label="Low skill",  color="#e05c5c", alpha=0.85)
    ax.bar(lab_ids + w/2, train_high.values,
           width=w, label="High skill", color="#4a90d9", alpha=0.85)
    ax.set_xticks(lab_ids)
    ax.set_xticklabels([f"L{i}" for i in train_low.index], fontsize=7)
    ax.set_title("Training Steps per Lab\n(at final step)", fontsize=9)
    ax.set_ylabel("Training steps")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- panels 3-4: time series ---
    time_metrics = [
        ("Avg Domain Capacity",
         "Avg Domain Truthfulness Cap\n(theoretical ceiling across all domains)"),
        ("Avg Reputation",
         "Avg Reputation per lab\n(cumulative scientific value)"),
    ]
    for ax, (metric, title) in zip([axes[1,0], axes[1,1]], time_metrics):
        ax.plot(df_low.index,  df_low[metric],
                label="Low skill  — specialist", color="#e05c5c", linewidth=2.2)
        ax.plot(df_high.index, df_high[metric],
                label="High skill — generalist", color="#4a90d9", linewidth=2.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        for t in range(10, steps, 10):
            ax.axvline(t, color="grey", alpha=0.12, linewidth=0.8)

    plt.savefig("abm_scenarios.png", dpi=150, bbox_inches="tight")

    plt.show()
