"""
Generate publication-quality figures for thesis Results section.
Saves to meta/img/  (creates files results_confirmatory.pdf and results_exploratory.pdf)
Run from any directory: python code/make_figures.py
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from scipy import stats

# ── aesthetics ────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

SPEC_COL = "#2563EB"   # blue  – Specialist
GEN_COL  = "#059669"   # green – Generalist
ALPHA    = 0.55
JITTER   = 0.08

# ── load data ─────────────────────────────────────────────────────────────────
here    = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(here, "..", "meta", "experiment_results.csv")
df = pd.read_csv(csv_path)
spec = df[df.label == "SPEC"]
gen  = df[df.label == "GEN"]

img_dir = os.path.join(here, "..", "meta", "img")
os.makedirs(img_dir, exist_ok=True)


def violin_half(ax, data, x, color, side="left"):
    """Draw one half of a split violin at position x."""
    kde   = stats.gaussian_kde(data, bw_method=0.4)
    yvals = np.linspace(data.min() - 0.02*(data.max()-data.min()),
                        data.max() + 0.02*(data.max()-data.min()), 200)
    dens  = kde(yvals)
    dens  = dens / dens.max() * 0.38           # scale half-width

    if side == "left":
        ax.fill_betweenx(yvals, x - dens, x, color=color, alpha=ALPHA)
        ax.plot(x - dens, yvals, color=color, lw=0.8)
    else:
        ax.fill_betweenx(yvals, x, x + dens, color=color, alpha=ALPHA)
        ax.plot(x + dens, yvals, color=color, lw=0.8)

    # IQR box + median
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    ax.plot([x, x], [q1, q3], color=color, lw=2.5, solid_capstyle="round")
    ax.plot(x, med, "o", color=color, ms=5, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)


def jitter_strip(ax, data, x, color, side="left"):
    """Jitter strip on one side of centre."""
    rng  = np.random.default_rng(42)
    sign = -1 if side == "left" else 1
    jx   = x + sign * (rng.uniform(0.02, JITTER, size=len(data)))
    ax.scatter(jx, data, s=12, color=color, alpha=0.65,
               linewidths=0, zorder=4)


def compute_stats(sv, gv, label):
    """Compute Mann-Whitney U, p, and Cohen's d from raw arrays.

    Returns (subtitle_string, significance_marker) — guaranteeing that
    the figure annotation matches the data being plotted.
    """
    u, p = stats.mannwhitneyu(sv, gv, alternative="two-sided")
    pooled_sd = np.sqrt(
        ((len(sv) - 1) * sv.var(ddof=1) + (len(gv) - 1) * gv.var(ddof=1))
        / (len(sv) + len(gv) - 2)
    )
    d = (sv.mean() - gv.mean()) / pooled_sd if pooled_sd > 0 else 0.0
    p_str = "p < .0001" if p < 1e-4 else f"p = {p:.3f}".replace("0.", ".")
    subtitle = f"{label}  ·  U = {u:.1f},  {p_str},  d = {d:.2f}"
    if   p < 0.001: sig = "***"
    elif p < 0.01:  sig = "**"
    elif p < 0.05:  sig = "*"
    else:           sig = "n.s."
    return subtitle, sig


def annotate_p(ax, x, y_top, label, dy=0.03):
    """Significance bracket above the pair."""
    ax.annotate("", xy=(x - 0.5, y_top + dy),
                xytext=(x + 0.5, y_top + dy),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax.text(x, y_top + dy + 0.005, label, ha="center", va="bottom",
            fontsize=8.5)


# ── Figure 1: confirmatory outcomes ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.2))
fig.subplots_adjust(wspace=0.45)

panels = [
    (axes[0], "fail_rate", "H1",
     "Replication Failure Rate",
     MultipleLocator(0.05),
     0.46, 0.72),
    (axes[1], "gini", "H2",
     "Gini Coefficient\n(Publication Domain Concentration)",
     MultipleLocator(0.05),
     0.04, 0.22),
]

for ax, col, hyp, ylabel, locator, ylo, yhi in panels:
    sv = spec[col].values
    gv = gen[col].values
    subtitle, sig = compute_stats(sv, gv, hyp)

    violin_half(ax, sv, 1, SPEC_COL, "left")
    violin_half(ax, gv, 1, GEN_COL,  "right")
    jitter_strip(ax, sv, 1, SPEC_COL, "left")
    jitter_strip(ax, gv, 1, GEN_COL,  "right")

    top = max(sv.max(), gv.max())
    bracket_y = top + 0.012
    ax.annotate("", xy=(0.62, bracket_y), xytext=(1.38, bracket_y),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.7))
    ax.text(1.0, bracket_y + 0.005, sig, ha="center", va="bottom",
            fontsize=10, fontweight="bold")

    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(subtitle, fontsize=7.5, color="0.35", pad=4)
    ax.set_xticks([])
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(ylo, yhi + 0.06)
    ax.yaxis.set_minor_locator(locator)

# shared legend
spec_patch = mpatches.Patch(color=SPEC_COL, alpha=0.75, label="Specialist")
gen_patch  = mpatches.Patch(color=GEN_COL,  alpha=0.75, label="Generalist")
fig.legend(handles=[spec_patch, gen_patch], loc="lower center",
           ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))

out1 = os.path.join(img_dir, "results_confirmatory.pdf")
fig.savefig(out1)
print(f"Saved → {out1}")
plt.close(fig)


# ── Figure 2: exploratory outcomes ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.2))
fig.subplots_adjust(wspace=0.55)

panels2 = [
    (axes[0], "mean_reputation", "H3",
     "Mean Researcher Reputation",
     MultipleLocator(2),
     4.0, 18.0),
    (axes[1], "debunk_impact_rate", "H4",
     "Debunk Impact Rate\n(reputation lost / career earnings)",
     MultipleLocator(0.0005),
     -0.0001, 0.0023),
]

for ax, col, hyp, ylabel, locator, ylo, yhi in panels2:
    sv = spec[col].values
    gv = gen[col].values
    subtitle, sig = compute_stats(sv, gv, hyp)

    violin_half(ax, sv, 1, SPEC_COL, "left")
    violin_half(ax, gv, 1, GEN_COL,  "right")
    jitter_strip(ax, sv, 1, SPEC_COL, "left")
    jitter_strip(ax, gv, 1, GEN_COL,  "right")

    top = max(sv.max(), gv.max())
    pad = (yhi - ylo) * 0.04
    bracket_y = top + pad
    ax.annotate("", xy=(0.62, bracket_y), xytext=(1.38, bracket_y),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.7))
    sig_label = sig if sig == "n.s." else sig
    ax.text(1.0, bracket_y + pad * 0.3, sig_label,
            ha="center", va="bottom",
            fontsize=10 if sig != "n.s." else 8.5,
            fontweight="bold" if sig != "n.s." else "normal")

    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(subtitle, fontsize=7.5, color="0.35", pad=4)
    ax.set_xticks([])
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(ylo, yhi + pad * 2)
    ax.yaxis.set_minor_locator(locator)

fig.legend(handles=[spec_patch, gen_patch], loc="lower center",
           ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))

out2 = os.path.join(img_dir, "results_exploratory.pdf")
fig.savefig(out2)
print(f"Saved → {out2}")
plt.close(fig)

print("Done.")
