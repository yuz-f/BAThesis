"""
make_validation_figure.py
==========================

Two-panel methodological-validation figure that supports Tier 3 claims:

Panel A — Parameter recovery (Tier 3.1):
  Predicted vs. actual scatter of `gap` (skill-distribution dispersion,
  the manipulated parameter under the mean-constant reparameterisation)
  recovered from outcome metrics via leave-one-out cross-validated
  linear regression. Tests partial identifiability: does the simulation
  output carry information about its own input parameter?

Panel B — Per-researcher publication-Gini distribution (Tier 3.2):
  Across the parameter sweep, the Gini coefficient of per-researcher
  publication counts at each run, with the empirical reference band
  from @Nielsen2021 and @Allison1974 overlaid.
  Tests external validity: does the model produce inequality at the
  same magnitude as documented academic publication distributions?

Reads:  meta/parameter_recovery_results.csv
Writes: meta/img/validation_figure.pdf
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
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

CSV = os.path.join(HERE, "..", "meta", "parameter_recovery_results.csv")
OUT = os.path.join(HERE, "..", "meta", "img", "validation_figure.pdf")

# Empirical reference: per-researcher publication-count Gini
# Nielsen & Andersen (2021): citation Gini 0.65–0.70 in 2000–2015
# Allison & Stewart (1974) and follow-up literature: publication Gini ≈ 0.5–0.7
EMPIRICAL_GINI_LO = 0.50
EMPIRICAL_GINI_HI = 0.70


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} rows from {CSV}")

    # ── Panel A: parameter recovery ─────────────────────────────────────────
    feature_cols = ["fail_rate", "domain_gini", "researcher_gini",
                    "mean_pubs", "mean_reputation",
                    "exploit_frac", "explore_frac", "train_frac"]
    X = df[feature_cols].values
    y = df["gap"].values   # manipulated parameter under (mean_skill, gap) reparam

    # leave-one-out cross-validation
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for tr, te in loo.split(X):
        model = LinearRegression()
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])

    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    r2   = float(1.0 - np.sum((preds - y) ** 2) / np.sum((y - y.mean()) ** 2))
    pearson_r, pearson_p = scistats.pearsonr(preds, y)
    print(f"Parameter recovery — LOO-CV: R² = {r2:.3f}, RMSE = {rmse:.4f}, "
          f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.2e})")

    # ── Panel B: per-researcher Gini distribution ───────────────────────────
    # Aggregate by parameter value
    gini_summary = df.groupby("gap")["researcher_gini"].agg(
        ["mean", "std", "count"]).reset_index()
    print("\nPer-researcher publication Gini by parameter value:")
    print(gini_summary.to_string(index=False))

    overall_mean = df["researcher_gini"].mean()
    overall_std  = df["researcher_gini"].std(ddof=1)
    print(f"\nOverall: mean = {overall_mean:.3f}, SD = {overall_std:.3f}")
    print(f"Empirical reference band: [{EMPIRICAL_GINI_LO}, {EMPIRICAL_GINI_HI}]")
    in_band = ((df["researcher_gini"] >= EMPIRICAL_GINI_LO)
               & (df["researcher_gini"] <= EMPIRICAL_GINI_HI))
    print(f"Runs with Gini in empirical band: {in_band.sum()}/{len(df)} "
          f"({100*in_band.mean():.0f}%)")

    # ── plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: recovery scatter
    ax_a = axes[0]
    ax_a.scatter(y, preds, s=24, c="#2563EB", alpha=0.55,
                 edgecolors="white", linewidths=0.4)
    lim = (y.min() - 0.02, y.max() + 0.02)
    ax_a.plot(lim, lim, "k--", lw=0.8, alpha=0.6, label="$y = x$ (perfect recovery)")
    ax_a.set_xlim(lim); ax_a.set_ylim(lim)
    ax_a.set_xlabel("True skill-distribution gap")
    ax_a.set_ylabel("Recovered $\\widehat{\\mathrm{gap}}$ (LOO-CV)")
    ax_a.set_title(f"A.  Parameter recovery from outcome metrics\n"
                   f"$R^2 = {r2:.2f}$,  RMSE = {rmse:.3f},  $r = {pearson_r:.2f}$",
                   pad=8, fontsize=10)
    ax_a.legend(loc="lower right", frameon=False, fontsize=9)
    ax_a.grid(axis="both", color="#EEE", linewidth=0.5, zorder=0)
    ax_a.set_aspect("equal", adjustable="box")

    # Panel B: per-researcher Gini histogram with empirical band
    ax_b = axes[1]
    ax_b.axvspan(EMPIRICAL_GINI_LO, EMPIRICAL_GINI_HI,
                 color="#10B981", alpha=0.18, zorder=1,
                 label=f"Empirical band [{EMPIRICAL_GINI_LO}, {EMPIRICAL_GINI_HI}]\n(Allison \\& Stewart 1974;\nNielsen \\& Andersen 2021)")
    ax_b.hist(df["researcher_gini"], bins=18, color="#2563EB",
              alpha=0.75, edgecolor="white", linewidth=0.8, zorder=2)
    ax_b.axvline(overall_mean, color="black", lw=1.4, linestyle="--",
                 label=f"Simulated mean = {overall_mean:.2f}", zorder=3)
    ax_b.set_xlabel("Gini coefficient of per-researcher publication counts")
    ax_b.set_ylabel("Number of simulation runs")
    ax_b.set_title(f"B.  Publication-inequality validation\n"
                   f"$n = {len(df)}$ runs across 6 parameter values × 15 seeds",
                   pad=8, fontsize=10)
    ax_b.legend(loc="upper right", frameon=False, fontsize=8)
    ax_b.grid(axis="y", color="#EEE", linewidth=0.5, zorder=0)
    ax_b.set_xlim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"\nSaved → {OUT}")
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
    print(f"Saved → {OUT.replace('.pdf', '.png')}")

    # save a tiny summary file for citation in the thesis
    summary_path = os.path.join(HERE, "..", "meta", "validation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("VALIDATION SUMMARY\n")
        f.write("==================\n\n")
        f.write(f"Parameter recovery (Tier 3.1):\n")
        f.write(f"  R²         = {r2:.4f}\n")
        f.write(f"  RMSE       = {rmse:.4f}\n")
        f.write(f"  Pearson r  = {pearson_r:.4f}  (p = {pearson_p:.2e})\n\n")
        f.write(f"Per-researcher publication Gini (Tier 3.2):\n")
        f.write(f"  Simulated mean        = {overall_mean:.4f}\n")
        f.write(f"  Simulated SD          = {overall_std:.4f}\n")
        f.write(f"  Empirical band        = [{EMPIRICAL_GINI_LO}, {EMPIRICAL_GINI_HI}]\n")
        f.write(f"  Runs in empirical band = {in_band.sum()}/{len(df)} ({100*in_band.mean():.1f}%)\n")
    print(f"Saved → {summary_path}")


if __name__ == "__main__":
    main()
