import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

IMG_DIR = Path(__file__).parent.parent / "meta" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

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
    plt.savefig(IMG_DIR / "agent_skill_profiles.png", dpi=150, bbox_inches="tight")

    plt.show()

    # ── Figure 2: per-domain skill trajectories ───────────────────────────
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig2.suptitle("Mean Skill per Domain Over Time\n(all active agents per step)", fontsize=11)

    def domain_trajectories(adf):
        traj = np.zeros((len(all_steps), n_domains))
        for i, step in enumerate(all_steps):
            step_df = adf.xs(step, level="Step")
            if step_df.empty:
                continue
            skills = np.array(step_df["DomainSkills"].tolist())
            traj[i] = skills.mean(axis=0)
        return traj

    for ax, adf, label in [
        (ax3, adf_low,  "Specialist (low-skill)"),
        (ax4, adf_high, "Generalist (high-skill)"),
    ]:
        traj = domain_trajectories(adf)
        for d in range(n_domains):
            ax.plot(all_steps, traj[:, d], color=cmap_domains(d / n_domains),
                    linewidth=1.5, label=f"D{d}", alpha=0.85)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Mean skill")
        ymin, ymax = traj.min(), traj.max()
        margin = (ymax - ymin) * 0.05
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2, loc="upper left")

    plt.savefig(IMG_DIR / "agent_skill_trajectories.png", dpi=150, bbox_inches="tight")

    plt.show()


def plot_lab_history(adf_low, adf_high, n_domains: int):
    """
    Scatter plot of publication events.
    Y axis = researcher, X axis = time step.
    Dot colour = domain published in.
    Dot marker shape = lab (so clustering by lab is visible on the y-axis).
    Researchers from the same lab are adjacent on the y-axis.
    """
    cmap_domain = plt.get_cmap("tab20", n_domains)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    for ax, adf, label in [
        (ax1, adf_low,  "Specialist (low-skill)"),
        (ax2, adf_high, "Generalist (high-skill)"),
    ]:
        # build a sorted researcher list: group by lab, then by agent id
        final_step = adf.index.get_level_values("Step").max()
        final_df   = adf.xs(final_step, level="Step")
        all_agents = sorted(final_df.index.tolist(),
                            key=lambda aid: (final_df.loc[aid, "LabID"], aid))

        # y-position per agent (grouped by lab)
        y_pos = {aid: i for i, aid in enumerate(all_agents)}

        pub_steps, pub_y, pub_domains, pub_labs = [], [], [], []
        for (step, agent_id), row in adf.iterrows():
            prev = adf.loc[(step - 1, agent_id), "Publications"] \
                   if (step - 1, agent_id) in adf.index else 0
            if row["Publications"] > prev and pd.notna(row["CurrentDomain"]):
                if agent_id in y_pos:
                    pub_steps.append(step)
                    pub_y.append(y_pos[agent_id])
                    pub_domains.append(int(row["CurrentDomain"]))
                    pub_labs.append(int(row["LabID"]))

        sc = ax.scatter(
            pub_steps, pub_y,
            c=pub_domains, cmap=cmap_domain, vmin=0, vmax=n_domains - 1,
            s=12, alpha=0.75, linewidths=0
        )

        # y-tick labels: researcher id, prefixed with lab
        ax.set_yticks(range(len(all_agents)))
        ax.set_yticklabels(
            [f"L{int(final_df.loc[aid,'LabID'])}-R{aid}" for aid in all_agents],
            fontsize=6
        )

        # draw faint horizontal bands per lab for readability
        lab_ids  = [int(final_df.loc[aid, "LabID"]) for aid in all_agents]
        prev_lab = None
        band     = False
        for i, lid in enumerate(lab_ids):
            if lid != prev_lab:
                band     = not band
                prev_lab = lid
            if band:
                ax.axhspan(i - 0.5, i + 0.5, color="grey", alpha=0.06, linewidth=0)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Researcher (grouped by lab)")
        ax.set_title(label, fontsize=10)
        ax.grid(axis="x", alpha=0.15)

    cbar = fig.colorbar(sc, ax=[ax1, ax2], label="Domain", shrink=0.7)
    cbar.set_ticks(range(n_domains))
    cbar.set_ticklabels([f"D{d}" for d in range(n_domains)], fontsize=7)
    fig.suptitle("Publication Events per Researcher  (grouped by lab)", fontsize=11)

    plt.savefig(IMG_DIR / "lab_history.png", dpi=150, bbox_inches="tight")
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

    # --- panel 2: action mode distribution (stacked bar) ---
    ax = axes[0, 1]
    final_step = adf_low.index.get_level_values("Step").max()

    def mean_action_fractions(adf, step):
        df     = adf.xs(step, level="Step")
        cols   = ["ExploitSteps", "TrainingSteps", "ExploreSteps", "DebunkSteps"]
        totals = df[cols].sum(axis=1).clip(lower=1)
        fracs  = df[cols].div(totals, axis=0)
        return fracs.mean()

    frac_low  = mean_action_fractions(adf_low,  final_step)
    frac_high = mean_action_fractions(adf_high, final_step)

    mode_cols   = ["ExploitSteps", "TrainingSteps", "ExploreSteps", "DebunkSteps"]
    mode_labels = ["Exploit", "Train", "Explore", "Debunk"]
    mode_colors = ["#4a90d9", "#e05c5c", "#5cb85c", "#f0ad4e"]

    bottoms = [0.0, 0.0]
    for col, label, color in zip(mode_cols, mode_labels, mode_colors):
        vals = [frac_low[col], frac_high[col]]
        ax.bar([0, 1], vals, bottom=bottoms, color=color, alpha=0.85, label=label)
        bottoms = [bottoms[i] + vals[i] for i in range(2)]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Specialist\n(low-skill)", "Generalist\n(high-skill)"], fontsize=9)
    ax.set_title("Action Mode Distribution\n(mean fraction of steps at final step)", fontsize=9)
    ax.set_ylabel("Fraction of steps")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")
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

    plt.savefig(IMG_DIR / "abm_scenarios.png", dpi=150, bbox_inches="tight")

    plt.show()


def plot_cluster_animation(adf_low, adf_high, n_labs: int,
                           selection_interval: int = 25, fps: int = 10):
    """
    Animated PCA scatter showing how agents move through skill-space over time.

    Each frame = one simulation step.
      - Dot colour + marker shape = lab of origin (stable across replacement)
      - Gold outline, larger dot  = agent spawned this step
      - Red ✕ at last position    = agent culled this step (shown for one frame)
      - Title turns dark-red and shows "◀ SELECTION" at culling steps
      - Selection-event frames are held 3× longer than normal frames

    PCA and StandardScaler are fit once on the full dataset so axes are stable.
    Output: meta/img/cluster_anim_specialist.gif + cluster_anim_generalist.gif
    """
    cmap    = plt.get_cmap("tab10", n_labs)
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "8"]

    def _animate(adf, label, filename):
        all_steps = sorted(adf.index.get_level_values("Step").unique())

        # Fit scaler + PCA once on all data so axes stay stable
        all_skills = np.array(adf["DomainSkills"].tolist())
        scaler     = StandardScaler().fit(all_skills)
        all_scaled = np.nan_to_num(scaler.transform(all_skills))
        pca        = PCA(n_components=2, random_state=42).fit(all_scaled)
        var        = pca.explained_variance_ratio_

        # Pre-compute per-step dataframes and PCA coords
        step_dfs    = {}
        step_coords = {}
        for step in all_steps:
            df     = adf.xs(step, level="Step")
            scaled = np.nan_to_num(scaler.transform(np.array(df["DomainSkills"].tolist())))
            step_dfs[step]    = df
            step_coords[step] = pca.transform(scaled)

        # Fixed axis bounds across all steps so movement is interpretable
        all_coords = np.vstack(list(step_coords.values()))
        pad  = 0.5
        xlim = (all_coords[:, 0].min() - pad, all_coords[:, 0].max() + pad)
        ylim = (all_coords[:, 1].min() - pad, all_coords[:, 1].max() + pad)

        # Detect spawns and deaths at each step transition
        spawned_at = {}
        died_at    = {}
        prev_ids   = set()
        for step in all_steps:
            curr_ids        = set(step_dfs[step].index)
            spawned_at[step] = curr_ids - prev_ids
            died_at[step]    = prev_ids - curr_ids
            prev_ids         = curr_ids

        # Frame list — hold selection-event steps 3× longer
        frames = []
        for step in all_steps:
            hold = 3 if int(step) % selection_interval == 0 else 1
            frames.extend([step] * hold)

        fig, ax = plt.subplots(figsize=(9, 7))

        def _draw(step):
            ax.clear()
            df      = step_dfs[step]
            coords  = step_coords[step]
            spawned = spawned_at[step]
            died    = died_at[step]

            # Red ✕ at last position of culled agents
            step_idx = all_steps.index(step)
            if step_idx > 0:
                prev_step   = all_steps[step_idx - 1]
                prev_df     = step_dfs[prev_step]
                prev_coords = step_coords[prev_step]
                for j, aid in enumerate(prev_df.index):
                    if aid in died:
                        px, py = prev_coords[j]
                        ax.scatter(px, py, c="red", marker="x",
                                   s=180, linewidths=2.5, zorder=5)

            # Active agents
            for j, (aid, row) in enumerate(df.iterrows()):
                lab    = int(row["LabID"])
                x, y   = coords[j]
                is_new = aid in spawned
                ax.scatter(x, y,
                           c=[cmap(lab / n_labs)],
                           marker=markers[lab % len(markers)],
                           s=130 if is_new else 55,
                           alpha=1.0 if is_new else 0.78,
                           edgecolors="gold" if is_new else "k",
                           linewidths=2.2 if is_new else 0.4,
                           zorder=4 if is_new else 3)
                ax.annotate(f"L{lab}", (x, y), fontsize=5,
                            ha="center", va="bottom",
                            xytext=(0, 4), textcoords="offset points",
                            alpha=0.75)

            is_sel = int(step) % selection_interval == 0
            ax.set_title(
                f"{label}  —  Step {int(step)}"
                + ("  ◀ SELECTION" if is_sel else ""),
                fontsize=10,
                color="darkred" if is_sel else "black"
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(f"PC1 ({var[0]:.1%} var)", fontsize=8)
            ax.set_ylabel(f"PC2 ({var[1]:.1%} var)", fontsize=8)
            ax.grid(alpha=0.2)

            lab_handles = [
                plt.scatter([], [], c=[cmap(i / n_labs)],
                            marker=markers[i % len(markers)],
                            s=40, label=f"Lab {i}")
                for i in range(n_labs)
            ]
            spawn_h = plt.scatter([], [], c="none", marker="o", s=70,
                                  edgecolors="gold", linewidths=2.2,
                                  label="Spawned")
            death_h = plt.scatter([], [], c="red", marker="x", s=90,
                                  linewidths=2.5, label="Culled")
            ax.legend(handles=lab_handles + [spawn_h, death_h],
                      fontsize=6, ncol=2, loc="upper right",
                      framealpha=0.85)

        anim     = FuncAnimation(fig, _draw, frames=frames,
                                 interval=1000 // fps, blit=False)
        out_path = IMG_DIR / filename
        print(f"Saving {filename} ({len(frames)} frames) …")
        anim.save(str(out_path), writer="pillow", fps=fps, dpi=90)
        plt.close(fig)
        print(f"Saved → {out_path}")

    _animate(adf_low,  "Specialist (low-skill)",  "cluster_anim_specialist.gif")
    _animate(adf_high, "Generalist (high-skill)", "cluster_anim_generalist.gif")
