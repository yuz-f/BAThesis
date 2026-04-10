import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

   

def plot_lab_history(adf_low, adf_high, n_domains: int):
    """
    Scatter plot of publication events.
    Y axis = lab, X axis = time step, colour = domain published in.
    Each dot is one successful publication.
    """
    cmap = plt.get_cmap("tab20", n_domains)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    for ax, adf, label in [
        (ax1, adf_low,  "Low skill  (μ=0.10)"),
        (ax2, adf_high, "High skill (μ=0.40)"),
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
    print("Saved → lab_history.png")
    plt.show()


def plot_scenarios(df_low, df_high, adf_low, adf_high, steps: int = 100):
    """
    Four-panel plot comparing low-skill vs high-skill scenarios.
    Panel 1: Models per domain (grouped bar at final step)
    Panel 2: Training steps per lab (grouped bar at final step)
    Panels 3-4: Time series — Cross-Lab Pub Rate and Avg Reputation
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle(
        "Science Lab ABM — Skill Level × Cross-Lab Model Adoption\n"
        "(low-skill: siloed start  |  high-skill: broad base  |  selection every 10 steps)",
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
    
    ""
    Clustering
    ""

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
        ("Cross-Lab Pub Rate",
         "Cross-Lab Publication Rate\n(building on other labs' models)"),
        ("Avg Reputation",
         "Avg Reputation per lab\n(cumulative scientific value)"),
    ]
    for ax, (metric, title) in zip([axes[1,0], axes[1,1]], time_metrics):
        ax.plot(df_low.index,  df_low[metric],
                label="Low skill  (μ=0.10)", color="#e05c5c", linewidth=2.2)
        ax.plot(df_high.index, df_high[metric],
                label="High skill (μ=0.40)", color="#4a90d9", linewidth=2.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        for t in range(10, steps, 10):
            ax.axvline(t, color="grey", alpha=0.12, linewidth=0.8)

    plt.savefig("abm_scenarios.png", dpi=150, bbox_inches="tight")
    print("Saved → abm_scenarios.png")
    plt.show()
