"""
make_ablation_figure.py
========================

Figure for the layer-decomposition ablation (Tier 4 / addresses ChatGPT
critique cluster A: tautology, landscape-as-meta-driver, complexity attribution).

Two-panel figure:
  Panel A: H1 effect size (Cohen's d, PEAKED vs BROAD failure rate) at each layer
  Panel B: H2 effect size (Cohen's d, PEAKED vs BROAD per-domain Gini) at each layer

Layers: L1 (base dynamics), L2 (+ realism), L3 (full v3 with landscape)

Reads:  meta/ablation_results.csv
Writes: meta/img/ablation_figure.pdf
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as scistats

HERE = os.path.dirname(os.path.abspath(__file__))

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

CSV = os.path.join(HERE, "..", "meta", "ablation_results.csv")
OUT = os.path.join(HERE, "..", "meta", "img", "ablation_figure.pdf")

LAYER_ORDER  = ["L1_base", "L2_realism", "L3_full"]
LAYER_LABELS = {
    "L1_base":    "L1\nBase\n(no landscape, no realism)",
    "L2_realism": "L2\n+ Realism\n(no landscape)",
    "L3_full":    "L3\nFull v3\n(landscape on)",
}
PEAKED_COL = "#2563EB"
BROAD_COL  = "#059669"


def cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled_sd = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) /
                        (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 1e-10 else 0.0


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} rows from {CSV}")

    summary_rows = []
    for layer in LAYER_ORDER:
        spec = df[(df.layer == layer) & (df.scenario == "PEAKED")]
        gen  = df[(df.layer == layer) & (df.scenario == "BROAD")]

        for metric in ["fail_rate", "domain_gini", "researcher_gini", "mean_reputation"]:
            d_val = cohen_d(spec[metric].values, gen[metric].values)
            u, p  = scistats.mannwhitneyu(spec[metric].values, gen[metric].values,
                                          alternative="two-sided")
            summary_rows.append({
                "layer": layer, "metric": metric,
                "spec_mean": spec[metric].mean(),
                "gen_mean":  gen[metric].mean(),
                "cohen_d":   d_val,
                "u_stat":    float(u),
                "p_value":   float(p),
            })
    summary = pd.DataFrame(summary_rows)
    print("\nSummary:")
    print(summary.to_string(index=False))

    # Save summary CSV
    summary_csv = os.path.join(HERE, "..", "meta", "ablation_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\nSummary saved → {summary_csv}")

    # ── plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: H1 (failure rate) — layer-wise PEAKED and BROAD means with CI
    ax_a = axes[0]
    xs   = np.arange(len(LAYER_ORDER))
    spec_means = [df[(df.layer == L) & (df.scenario == "PEAKED")]["fail_rate"].mean()
                  for L in LAYER_ORDER]
    gen_means  = [df[(df.layer == L) & (df.scenario == "BROAD")]["fail_rate"].mean()
                  for L in LAYER_ORDER]
    spec_se = [df[(df.layer == L) & (df.scenario == "PEAKED")]["fail_rate"].std(ddof=1) / np.sqrt(15)
               for L in LAYER_ORDER]
    gen_se  = [df[(df.layer == L) & (df.scenario == "BROAD")]["fail_rate"].std(ddof=1) / np.sqrt(15)
               for L in LAYER_ORDER]

    w = 0.32
    ax_a.bar(xs - w/2, spec_means, w, yerr=[1.96*x for x in spec_se],
             color=PEAKED_COL, alpha=0.78, capsize=3, label="Peaked", edgecolor="white")
    ax_a.bar(xs + w/2, gen_means, w, yerr=[1.96*x for x in gen_se],
             color=BROAD_COL, alpha=0.78, capsize=3, label="Broad", edgecolor="white")

    # annotate effect size above each pair
    for i, L in enumerate(LAYER_ORDER):
        d  = summary[(summary.layer == L) & (summary.metric == "fail_rate")]["cohen_d"].iloc[0]
        p  = summary[(summary.layer == L) & (summary.metric == "fail_rate")]["p_value"].iloc[0]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        top = max(spec_means[i] + 1.96*spec_se[i], gen_means[i] + 1.96*gen_se[i])
        ax_a.text(i, top + 0.012, f"$d = {d:.2f}$\n{sig}",
                  ha="center", va="bottom", fontsize=8.5)

    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([LAYER_LABELS[L] for L in LAYER_ORDER], fontsize=8.5)
    ax_a.set_ylabel("Replication failure rate")
    ax_a.set_title("A.  H1 across model layers", pad=8, fontsize=11)
    # add headroom above the highest annotation so the legend has clean space
    a_top = max(max(spec_means), max(gen_means)) + 0.10
    ax_a.set_ylim(0.50, a_top)
    ax_a.legend(loc="upper left", frameon=False, fontsize=9)
    ax_a.grid(axis="y", color="#EEE", linewidth=0.5, zorder=0)

    # Panel B: H2 (domain Gini)
    ax_b = axes[1]
    spec_means_g = [df[(df.layer == L) & (df.scenario == "PEAKED")]["domain_gini"].mean()
                    for L in LAYER_ORDER]
    gen_means_g  = [df[(df.layer == L) & (df.scenario == "BROAD")]["domain_gini"].mean()
                    for L in LAYER_ORDER]
    spec_se_g = [df[(df.layer == L) & (df.scenario == "PEAKED")]["domain_gini"].std(ddof=1) / np.sqrt(15)
                 for L in LAYER_ORDER]
    gen_se_g  = [df[(df.layer == L) & (df.scenario == "BROAD")]["domain_gini"].std(ddof=1) / np.sqrt(15)
                 for L in LAYER_ORDER]

    ax_b.bar(xs - w/2, spec_means_g, w, yerr=[1.96*x for x in spec_se_g],
             color=PEAKED_COL, alpha=0.78, capsize=3, label="Peaked", edgecolor="white")
    ax_b.bar(xs + w/2, gen_means_g, w, yerr=[1.96*x for x in gen_se_g],
             color=BROAD_COL, alpha=0.78, capsize=3, label="Broad", edgecolor="white")

    for i, L in enumerate(LAYER_ORDER):
        d  = summary[(summary.layer == L) & (summary.metric == "domain_gini")]["cohen_d"].iloc[0]
        p  = summary[(summary.layer == L) & (summary.metric == "domain_gini")]["p_value"].iloc[0]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        top = max(spec_means_g[i] + 1.96*spec_se_g[i], gen_means_g[i] + 1.96*gen_se_g[i])
        ax_b.text(i, top + 0.005, f"$d = {d:.2f}$\n{sig}",
                  ha="center", va="bottom", fontsize=8.5)

    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([LAYER_LABELS[L] for L in LAYER_ORDER], fontsize=8.5)
    ax_b.set_ylabel("Per-domain publication Gini")
    ax_b.set_title("B.  H2 across model layers", pad=8, fontsize=11)
    # extra headroom so annotations clear the upper bound and the legend
    b_top = max(max(spec_means_g), max(gen_means_g)) + 0.10
    ax_b.set_ylim(0, b_top)
    ax_b.legend(loc="upper left", frameon=False, fontsize=9)
    ax_b.grid(axis="y", color="#EEE", linewidth=0.5, zorder=0)

    fig.suptitle("Layer-decomposition ablation: how H1 and H2 effect sizes change "
                 "as architectural layers are added", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"\nSaved → {OUT}")
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
    print(f"Saved → {OUT.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
