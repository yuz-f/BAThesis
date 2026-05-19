import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

IMG_DIR       = Path(__file__).parent.parent / "meta" / "img"
IMG_DIR_DEBUG = IMG_DIR / "debug"
IMG_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR_DEBUG.mkdir(parents=True, exist_ok=True)

PEAKED_COL = "#e05c5c"
BROAD_COL  = "#4a90d9"


# ── internal helpers ──────────────────────────────────────────────────────────

def _gini(counts):
    """Gini coefficient of a list of counts (0 = perfectly even, 1 = monopoly)."""
    x = np.array(counts, dtype=float)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)


def _action_fractions_over_time(adf, window=15):
    """
    Compute per-step action fractions from cumulative step counters.
    Returns a DataFrame (index=step) with one column per action mode,
    smoothed with a centred rolling average.
    """
    steps = sorted(adf.index.get_level_values("Step").unique())
    cols  = ["ExploitSteps", "TrainingSteps", "ExploreSteps", "DebunkSteps"]
    records = []
    for i, step in enumerate(steps):
        if i == 0:
            records.append({c: 0.0 for c in cols})
            continue
        prev    = steps[i - 1]
        cur_df  = adf.xs(step, level="Step")
        prev_df = adf.xs(prev, level="Step")
        common  = cur_df.index.intersection(prev_df.index)
        if common.empty:
            records.append(records[-1].copy())
            continue
        delta = (cur_df.loc[common, cols] - prev_df.loc[common, cols]).clip(lower=0)
        total = delta.sum(axis=1).clip(lower=1)
        fracs = delta.div(total, axis=0).mean()
        records.append(fracs.to_dict())
    df = pd.DataFrame(records, index=steps)
    return df.rolling(window, min_periods=1, center=True).mean()


# ── thesis figures ────────────────────────────────────────────────────────────

def plot_scenarios(df_low, df_high, adf_low, adf_high, steps: int = 300):
    """
    Four-panel figure mapping directly to the four research claims (H1–H4).

      H1 (top-left)  : Replication failure rate over time
                        Peaked-profile researchers should show persistently higher failure rates.
      H2 (top-right) : Domain concentration (Gini) over time
                        Peaked-profile researchers should concentrate publications in fewer domains.
      H3 (bot-left)  : Average researcher reputation over time
                        Broad-profile researchers should accumulate more reputation (broader success).
      H4 (bot-right) : Cumulative reputation lost to debunking over time
                        Peaked-profile researchers should be less stable — debunking one domain
                        hits them harder because their reputation is concentrated.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Peaked vs Broad — Four Research Claims (H1–H4)", fontsize=11)

    # H1: replication failure rate over time
    ax = axes[0, 0]
    ax.plot(df_low.index,  df_low["Replication Failure Rate"],
            color=PEAKED_COL, linewidth=2.2, label="Peaked")
    ax.plot(df_high.index, df_high["Replication Failure Rate"],
            color=BROAD_COL,  linewidth=2.2, label="Broad")
    ax.axhline(0.60, color="black", linewidth=1.0, linestyle=":",
               label="OSC 2015 (60%)")
    ax.set_title("H1 — Replication Failure Rate", fontsize=9)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Failure rate  (0–1)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # H2: Gini coefficient over time
    ax = axes[0, 1]
    for df_m, color, lbl in [(df_low, PEAKED_COL, "Peaked"),
                              (df_high, BROAD_COL,  "Broad")]:
        ginis = [_gini(list(row.values())) for row in df_m["Models per Domain"]]
        ax.plot(df_m.index, ginis, color=color, linewidth=2.2, label=lbl)
    ax.set_title("H2 — Domain Concentration  (Gini coefficient)", fontsize=9)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Gini  (0 = even  ·  1 = monopoly)")
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # H3: average researcher reputation over time
    ax = axes[1, 0]
    ax.plot(df_low.index,  df_low["Avg Reputation"],
            color=PEAKED_COL, linewidth=2.2, label="Peaked")
    ax.plot(df_high.index, df_high["Avg Reputation"],
            color=BROAD_COL,  linewidth=2.2, label="Broad")
    ax.set_title("H3 — Average Researcher Reputation", fontsize=9)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean reputation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # H4: cumulative reputation lost to debunking over time
    ax = axes[1, 1]
    ax.plot(df_low.index,  df_low["Avg Reputation Lost to Debunk"],
            color=PEAKED_COL, linewidth=2.2, label="Peaked")
    ax.plot(df_high.index, df_high["Avg Reputation Lost to Debunk"],
            color=BROAD_COL,  linewidth=2.2, label="Broad")
    ax.set_title("H4 — Avg Reputation Lost to Debunking\n"
                 "(higher = less stable under scrutiny)", fontsize=9)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative reputation lost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.savefig(IMG_DIR / "abm_scenarios.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_knowledge_quality(df_low, df_high):
    """
    Four-panel figure showing the dual-truthfulness mechanism and knowledge dynamics.

      P1 (top-left)  : Actual vs reported truthfulness — shows the publication bias gap.
                        Both scenarios should reach the same actual frontier while reported
                        stays persistently inflated above it.
      P2 (top-right) : Average bias gap (reported − actual) over time.
                        Should stabilise near 0.10 (the mean inflation drawn at spawn).
      P3 (bot-left)  : Top-5 model salience band — era dynamics and debunk events.
      P4 (bot-right) : Reputation variance over time — peaked-profile researchers should show higher
                        variance as debunking concentrates gains/losses in one domain.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Knowledge Quality, Publication Bias, and Model Salience", fontsize=11)

    # P1: actual vs reported truthfulness per scenario
    ax = axes[0, 0]
    for df_m, color, lbl in [(df_low, PEAKED_COL, "Peaked"),
                              (df_high, BROAD_COL,  "Broad")]:
        mean_actual = [float(np.mean(list(row.values())))
                       for row in df_m["Best Actual Truthfulness per Domain"]]
        mean_rep    = [float(np.mean(list(row.values())))
                       for row in df_m["Best Reported Truthfulness per Domain"]]
        ax.plot(df_m.index, mean_actual, color=color, linewidth=2.2,
                label=f"{lbl} actual")
        ax.plot(df_m.index, mean_rep,    color=color, linewidth=1.2,
                linestyle="--", alpha=0.65, label=f"{lbl} reported")
    ax.set_title("Actual vs Reported Truthfulness\n"
                 "(solid = actual quality  ·  dashed = published quality)", fontsize=9)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean best truthfulness  (0–1)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # P2: avg bias gap (reported − actual) over time
    ax = axes[0, 1]
    ax.plot(df_low.index,  df_low["Avg Bias Gap"],
            color=PEAKED_COL, linewidth=2.2, label="Peaked")
    ax.plot(df_high.index, df_high["Avg Bias Gap"],
            color=BROAD_COL,  linewidth=2.2, label="Broad")
    ax.axhline(0.10, color="black", linewidth=1.0, linestyle=":",
               label="Expected inflation (μ=0.10)")
    ax.set_title("Publication Bias Gap  (reported − actual)\n"
                 "Persistent gap = systematic inflation in the literature", fontsize=9)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean bias gap")
    ax.set_ylim(0, 0.30)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # P3: salience band (mean + min) of top-5 models
    ax = axes[1, 0]
    for df_m, color, lbl in [(df_low, PEAKED_COL, "Peaked"),
                              (df_high, BROAD_COL,  "Broad")]:
        mean = df_m["Avg Top5 Salience"]
        mn   = df_m["Min Top5 Salience"]
        ax.plot(df_m.index, mean, color=color, linewidth=2.0, label=f"{lbl} mean")
        ax.plot(df_m.index, mn,   color=color, linewidth=1.0,
                linestyle="--", alpha=0.7, label=f"{lbl} min")
        ax.fill_between(df_m.index, mn, mean, color=color, alpha=0.15)
    ax.set_title("Top-5 Model Salience  (mean + min)\nDips = debunk / model turnover events",
                 fontsize=9)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Salience  (0–1)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # P4: reputation variance over time (higher = more inequality / fragility)
    ax = axes[1, 1]
    ax.plot(df_low.index,  df_low["Reputation Variance"],
            color=PEAKED_COL, linewidth=2.2, label="Peaked")
    ax.plot(df_high.index, df_high["Reputation Variance"],
            color=BROAD_COL,  linewidth=2.2, label="Broad")
    ax.set_title("Reputation Variance  (higher = more inequality)\n"
                 "Peaked-profile researchers expected to show more fragile, concentrated reputation", fontsize=9)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Variance of reputation across researchers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.savefig(IMG_DIR / "knowledge_quality.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_action_over_time(adf_low, adf_high):
    """
    Stacked area chart of action mode fractions over time (15-step rolling avg).
    Reveals how researcher activity shifts as domains saturate and selection fires.

    The first 10 steps are trimmed from the display: action mode fractions there
    are dominated by multi-step exploit lock-in (one decision at step 1 inflates
    into ~5 counted exploit_steps as the work cycle plays out), and the centered
    rolling average pulls future values into early time-points. Trimming gives
    an honest view of the post-commitment dynamics.
    """
    BURN_IN = 10
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Action Mode Distribution over Time  (15-step rolling average)", fontsize=11)

    mode_cols   = ["ExploitSteps", "TrainingSteps", "ExploreSteps", "DebunkSteps"]
    mode_labels = ["Exploit", "Train", "Explore", "Debunk"]
    mode_colors = [BROAD_COL, PEAKED_COL, "#5cb85c", "#f0ad4e"]

    for ax, adf, lbl in [(ax1, adf_low, "Peaked"), (ax2, adf_high, "Broad")]:
        fracs   = _action_fractions_over_time(adf)
        fracs   = fracs.iloc[BURN_IN:]
        steps   = fracs.index
        bottoms = np.zeros(len(steps))
        for col, lab, color in zip(mode_cols, mode_labels, mode_colors):
            vals = fracs[col].values
            ax.fill_between(steps, bottoms, bottoms + vals,
                            color=color, alpha=0.75, label=lab)
            bottoms += vals
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Fraction of steps")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)

    plt.savefig(IMG_DIR / "action_modes_over_time.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_agent_skills(adf_low, adf_high, n_domains: int):
    """
    Per-domain mean skill trajectories over time for each scenario.
    One line per domain; shows which domains each population collectively develops.
    """
    all_steps    = sorted(adf_low.index.get_level_values("Step").unique())
    cmap_domains = plt.get_cmap("tab20", n_domains)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Mean Skill per Domain over Time  (all active agents per step)", fontsize=11)

    def _trajectories(adf):
        traj = np.zeros((len(all_steps), n_domains))
        for i, step in enumerate(all_steps):
            step_df = adf.xs(step, level="Step")
            if step_df.empty:
                continue
            traj[i] = np.array(step_df["DomainSkills"].tolist()).mean(axis=0)
        return traj

    for ax, adf, lbl in [(ax1, adf_low, "Peaked"), (ax2, adf_high, "Broad")]:
        traj = _trajectories(adf)
        for d in range(n_domains):
            ax.plot(all_steps, traj[:, d],
                    color=cmap_domains(d / n_domains),
                    linewidth=1.5, label=f"D{d}", alpha=0.85)
        ymin, ymax = traj.min(), traj.max()
        margin = (ymax - ymin) * 0.05
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Mean skill")
        ax.set_title(lbl, fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2, loc="upper left")

    plt.savefig(IMG_DIR / "agent_skill_trajectories.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_cluster_animation(adf_low, adf_high, n_labs: int,
                           selection_interval: int = 25, fps: int = 10):
    """
    Animated PCA biplot showing how agents move through skill-space over time.

    Biplot arrows mark the 5 domains that load most strongly on the PCA axes,
    so movement in the plot can be read as shifts in domain-specific skill.
    Arrows are static (PCA is fitted once); agent positions update each frame.

    Encoding:
      colour + marker  = lab of origin
      gold outline     = agent spawned this step
      red ✕            = agent culled this step (shown for one frame)
      dark-red title   = selection event step (held 3× longer)
    """
    cmap    = plt.get_cmap("tab10", n_labs)
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "8"]

    def _animate(adf, label, filename):
        all_steps  = sorted(adf.index.get_level_values("Step").unique())
        all_skills = np.array(adf["DomainSkills"].tolist())
        scaler     = StandardScaler().fit(all_skills)
        all_scaled = np.nan_to_num(scaler.transform(all_skills))
        pca        = PCA(n_components=2, random_state=42).fit(all_scaled)
        var        = pca.explained_variance_ratio_

        # Biplot: loading vectors (n_domains × 2), pick 5 most influential
        loadings  = pca.components_.T          # (n_domains, 2)
        arrow_mag = np.sqrt((loadings ** 2).sum(axis=1))
        top_d     = np.argsort(arrow_mag)[-5:]

        step_dfs    = {}
        step_coords = {}
        for step in all_steps:
            df     = adf.xs(step, level="Step")
            scaled = np.nan_to_num(
                scaler.transform(np.array(df["DomainSkills"].tolist())))
            step_dfs[step]    = df
            step_coords[step] = pca.transform(scaled)

        all_coords = np.vstack(list(step_coords.values()))
        pad  = 0.5
        xlim = (all_coords[:, 0].min() - pad, all_coords[:, 0].max() + pad)
        ylim = (all_coords[:, 1].min() - pad, all_coords[:, 1].max() + pad)

        # Scale arrows to 30% of the shorter axis span
        x_range     = xlim[1] - xlim[0]
        y_range     = ylim[1] - ylim[0]
        arrow_scale = 0.30 * min(x_range, y_range) / (arrow_mag[top_d].max() + 1e-9)
        scaled_arrows = loadings[top_d] * arrow_scale   # (5, 2)

        spawned_at, died_at = {}, {}
        prev_ids = set()
        for step in all_steps:
            curr_ids          = set(step_dfs[step].index)
            spawned_at[step]  = curr_ids - prev_ids
            died_at[step]     = prev_ids - curr_ids
            prev_ids          = curr_ids

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

            # Culled agents: red ✕ at last known position
            step_idx = all_steps.index(step)
            if step_idx > 0:
                prev_step   = all_steps[step_idx - 1]
                prev_coords = step_coords[prev_step]
                for j, aid in enumerate(step_dfs[prev_step].index):
                    if aid in died:
                        ax.scatter(*prev_coords[j], c="red", marker="x",
                                   s=180, linewidths=2.5, zorder=5)

            # Active agents
            for j, (aid, row) in enumerate(df.iterrows()):
                lab = int(row["LabID"])
                new = aid in spawned
                ax.scatter(*coords[j],
                           c=[cmap(lab / n_labs)],
                           marker=markers[lab % len(markers)],
                           s=130 if new else 55,
                           alpha=1.0 if new else 0.78,
                           edgecolors="gold" if new else "k",
                           linewidths=2.2 if new else 0.4,
                           zorder=4 if new else 3)
                ax.annotate(f"L{lab}", coords[j], fontsize=5,
                            ha="center", va="bottom",
                            xytext=(0, 4), textcoords="offset points",
                            alpha=0.75)

            # Biplot arrows — static per animation, drawn each frame
            for i, d in enumerate(top_d):
                dx, dy = scaled_arrows[i]
                ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color="dimgray",
                                            lw=1.5, alpha=0.75))
                ax.text(dx * 1.18, dy * 1.18, f"D{d}",
                        fontsize=7, color="dimgray",
                        ha="center", va="center", fontweight="bold")

            is_sel = int(step) % selection_interval == 0
            ax.set_title(
                f"{label}  —  Step {int(step)}"
                + ("  ◀ SELECTION" if is_sel else ""),
                fontsize=10, color="darkred" if is_sel else "black"
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(
                f"PC1  ({var[0]:.1%} variance explained)\n"
                f"Arrow labels = domains driving this axis", fontsize=8)
            ax.set_ylabel(
                f"PC2  ({var[1]:.1%} variance explained)", fontsize=8)
            ax.grid(alpha=0.2)

            lab_handles = [
                plt.scatter([], [], c=[cmap(i / n_labs)],
                            marker=markers[i % len(markers)],
                            s=40, label=f"Lab {i}")
                for i in range(n_labs)
            ]
            ax.legend(
                handles=lab_handles + [
                    plt.scatter([], [], c="none", marker="o", s=70,
                                edgecolors="gold", linewidths=2.2, label="Spawned"),
                    plt.scatter([], [], c="red", marker="x", s=90,
                                linewidths=2.5, label="Culled"),
                ],
                fontsize=6, ncol=2, loc="upper right", framealpha=0.85
            )

        anim     = FuncAnimation(fig, _draw, frames=frames,
                                 interval=1000 // fps, blit=False)
        out_path = IMG_DIR / filename
        print(f"Saving {filename}  ({len(frames)} frames) …")
        anim.save(str(out_path), writer="pillow", fps=fps, dpi=90)
        plt.close(fig)
        print(f"Saved → {out_path}")

    _animate(adf_low,  "Peaked", "cluster_anim_peaked.gif")
    _animate(adf_high, "Broad", "cluster_anim_broad.gif")


# ── debug figures (saved to meta/img/debug/) ─────────────────────────────────

def plot_lab_history(adf_low, adf_high, n_domains: int):
    """
    [DEBUG] Scatter plot of publication events per researcher over time.
    Colour = domain published in. Grouped by lab on the y-axis.
    Saved to meta/img/debug/ — not a thesis figure.
    """
    cmap_domain = plt.get_cmap("tab20", n_domains)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    for ax, adf, lbl in [(ax1, adf_low, "Peaked"), (ax2, adf_high, "Broad")]:
        final_step = adf.index.get_level_values("Step").max()
        final_df   = adf.xs(final_step, level="Step")
        all_agents = sorted(final_df.index.tolist(),
                            key=lambda aid: (final_df.loc[aid, "LabID"], aid))
        y_pos = {aid: i for i, aid in enumerate(all_agents)}

        pub_steps, pub_y, pub_domains = [], [], []
        for (step, agent_id), row in adf.iterrows():
            prev = adf.loc[(step - 1, agent_id), "Publications"] \
                   if (step - 1, agent_id) in adf.index else 0
            if row["Publications"] > prev and pd.notna(row["CurrentDomain"]):
                if agent_id in y_pos:
                    pub_steps.append(step)
                    pub_y.append(y_pos[agent_id])
                    pub_domains.append(int(row["CurrentDomain"]))

        sc = ax.scatter(pub_steps, pub_y,
                        c=pub_domains, cmap=cmap_domain,
                        vmin=0, vmax=n_domains - 1,
                        s=12, alpha=0.75, linewidths=0)
        ax.set_yticks(range(len(all_agents)))
        ax.set_yticklabels(
            [f"L{int(final_df.loc[aid, 'LabID'])}-R{aid}" for aid in all_agents],
            fontsize=6)

        prev_lab, band = None, False
        for i, aid in enumerate(all_agents):
            lid = int(final_df.loc[aid, "LabID"])
            if lid != prev_lab:
                band     = not band
                prev_lab = lid
            if band:
                ax.axhspan(i - 0.5, i + 0.5, color="grey", alpha=0.06, linewidth=0)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Researcher  (grouped by lab)")
        ax.set_title(lbl, fontsize=10)
        ax.grid(axis="x", alpha=0.15)

    cbar = fig.colorbar(sc, ax=[ax1, ax2], label="Domain", shrink=0.7)
    cbar.set_ticks(range(n_domains))
    cbar.set_ticklabels([f"D{d}" for d in range(n_domains)], fontsize=7)
    fig.suptitle("[DEBUG] Publication Events per Researcher", fontsize=11)

    plt.savefig(IMG_DIR_DEBUG / "lab_history.png", dpi=150, bbox_inches="tight")
    plt.show()
