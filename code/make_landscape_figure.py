"""
make_landscape_figure.py
========================

Produces a four-panel publication figure showing the epistemic landscape
mechanism and the scenario contrast it produces:

  Panel A: 3D surface plot of one representative domain's epistemic landscape
  Panel B: 2D heatmap of the same surface with seed-model position marked
  Panel C: 2D heatmap with all PEAKED-scenario models at t=300 (size = salience)
  Panel D: 2D heatmap with all BROAD-scenario models at t=300 (size = salience)

The seed is chosen so the landscape is identical across panels (landscapes are
generated from rng before any scenario-specific parameters are consumed), which
isolates the PEAKED/BROAD contrast to model-placement dynamics.

Saves to meta/img/landscape_figure.pdf
Run from any directory: .venv/bin/python3 code/make_landscape_figure.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

# Make code imports work from any cwd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from world import ScienceWorld

# ── aesthetics ────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

PEAKED = dict(peak_skill_mean=0.55, peak_skill_std=0.07,
            other_skill_mean=0.25, other_skill_std=0.06)
BROAD  = dict(peak_skill_mean=0.55, peak_skill_std=0.08,
            other_skill_mean=0.33, other_skill_std=0.08)

STEPS = 300
SEED  = 42                # landscape will be identical between scenarios at same seed

PEAKED_COL = "#2563EB"      # consistent with make_figures.py
BROAD_COL  = "#059669"


# ── helpers ───────────────────────────────────────────────────────────────────

def landscape_grid(landscape, n: int = 80):
    """Sample the landscape height on an n×n grid over [0,1]²."""
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[landscape.height(x, y) for x in xs] for y in ys])
    return X, Y, Z


def pick_representative_domain(world, n_grid: int = 60) -> int:
    """
    Pick the domain whose landscape has the largest height range — the most
    visually informative one (clearest valley/peak contrast).
    """
    ranges = []
    for d in range(world.n_domains):
        _, _, Z = landscape_grid(world.landscapes[d], n=n_grid)
        ranges.append(Z.max() - Z.min())
    return int(np.argmax(ranges))


def run_scenario(scenario: dict, seed: int, steps: int) -> ScienceWorld:
    w = ScienceWorld(rng=seed, **scenario)
    for _ in range(steps):
        w.step()
    return w


def scatter_models(ax, models_in_domain, salience_floor: float = 0.05):
    """
    Two-layer plot showing both the publication history and the active literature.

    Layer 1 (background): all published models, tiny faint dots — the "fossil
        record" of where research has happened in theory-space.
    Layer 2 (foreground): currently-active models (salience >= floor),
        sized by salience, coloured by actual truthfulness:
            red   = high-truthfulness (>= 0.50) active model
            grey  = low-truthfulness  (<  0.50) active model

    Together these convey: where work has been done, and which of it has
    survived as the active literature at t=300.
    """
    if not models_in_domain:
        return

    # Layer 1: all models as faint background
    bg_x = [m.position[0] for m in models_in_domain]
    bg_y = [m.position[1] for m in models_in_domain]
    ax.scatter(bg_x, bg_y, s=4, c="white", alpha=0.18,
               edgecolors="none", zorder=2)

    # Layer 2: active models, two truthfulness tiers
    active = [m for m in models_in_domain if m.salience >= salience_floor]
    low    = [m for m in active if m.actual_truthfulness <  0.50]
    high   = [m for m in active if m.actual_truthfulness >= 0.50]

    if low:
        xs = [m.position[0] for m in low]
        ys = [m.position[1] for m in low]
        sizes = [12 + 80 * m.salience for m in low]
        ax.scatter(xs, ys, s=sizes, c="#9CA3AF", alpha=0.85,
                   edgecolors="black", linewidths=0.5, zorder=3)
    if high:
        xs = [m.position[0] for m in high]
        ys = [m.position[1] for m in high]
        sizes = [16 + 110 * m.salience for m in high]
        ax.scatter(xs, ys, s=sizes, c="#DC2626", alpha=0.92,
                   edgecolors="black", linewidths=0.6, zorder=4)

    # annotation: n_total / n_active
    ax.text(0.98, 0.96,
            f"$n_{{\\mathrm{{total}}}}={len(models_in_domain)}$\n"
            f"$n_{{\\mathrm{{active}}}}={len(active)}$",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Running PEAKED scenario (seed={SEED}, {STEPS} steps)…")
    spec_world = run_scenario(PEAKED, SEED, STEPS)
    print(f"Running BROAD  scenario (seed={SEED}, {STEPS} steps)…")
    gen_world  = run_scenario(BROAD,  SEED, STEPS)

    # Pick the same representative domain from spec_world (landscapes are
    # identical between scenarios at the same seed because EpistemicLandscape
    # is constructed before any scenario-specific parameter consumes rng).
    domain = pick_representative_domain(spec_world)
    print(f"Representative domain selected: D{domain}")

    landscape = spec_world.landscapes[domain]
    X, Y, Z   = landscape_grid(landscape, n=120)

    # Sanity-check: confirm the BROAD landscape for this domain is identical
    _, _, Z_gen = landscape_grid(gen_world.landscapes[domain], n=120)
    assert np.allclose(Z, Z_gen), \
        "Landscapes differ between PEAKED and BROAD at same seed — figure invalid."

    # Models in this domain at t=300, per scenario
    spec_models = [m for m in spec_world.scientific_models if m.domain == domain]
    gen_models  = [m for m in gen_world.scientific_models  if m.domain == domain]
    print(f"Models in D{domain}: PEAKED={len(spec_models)}, BROAD={len(gen_models)}")

    # Find the seed model (uid is small for first-spawned)
    seed_model_pos = None
    for m in spec_world.scientific_models:
        if m.domain == domain and m.parent_uid is None:
            seed_model_pos = m.position
            break

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 9))
    gs  = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22,
                            width_ratios=[1, 1], height_ratios=[1, 1])

    # Panel A: 3D surface
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    surf = ax_a.plot_surface(X, Y, Z, cmap="viridis_r", linewidth=0,
                              antialiased=True, alpha=0.95,
                              rstride=2, cstride=2)
    ax_a.set_xlabel("theory-space $x$", labelpad=4)
    ax_a.set_ylabel("theory-space $y$", labelpad=4)
    ax_a.set_zlabel("instability $h$", labelpad=4)
    ax_a.set_title(f"A.  Epistemic landscape — D{domain}\n(valleys = stable paradigms; peaks = unstable theory)",
                   pad=14, fontsize=10)
    ax_a.view_init(elev=28, azim=-55)
    ax_a.set_zlim(0, 1)
    ax_a.tick_params(axis="both", which="major", labelsize=7, pad=-2)

    # Panel B: 2D heatmap of same surface with seed model marked
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.contourf(X, Y, Z, levels=20, cmap="viridis_r")
    ax_b.contour(X, Y, Z, levels=10, colors="white", alpha=0.25, linewidths=0.5)
    if seed_model_pos is not None:
        ax_b.plot(*seed_model_pos, marker="*", color="white", markersize=18,
                   markeredgecolor="black", markeredgewidth=1.0, zorder=5)
        ax_b.annotate("seed\nmodel", xy=seed_model_pos,
                       xytext=(seed_model_pos[0] + 0.10, seed_model_pos[1] + 0.10),
                       color="white", fontsize=8, ha="left",
                       arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("theory-space $x$"); ax_b.set_ylabel("theory-space $y$")
    ax_b.set_title("B.  Same landscape (top-down view)\nwith seed-model position", pad=10, fontsize=10)
    ax_b.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label("instability $h$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Panel C: PEAKED at t=300
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.contourf(X, Y, Z, levels=20, cmap="viridis_r")
    ax_c.contour(X, Y, Z, levels=10, colors="white", alpha=0.25, linewidths=0.5)
    scatter_models(ax_c, spec_models)
    ax_c.set_xlim(0, 1); ax_c.set_ylim(0, 1)
    ax_c.set_xlabel("theory-space $x$"); ax_c.set_ylabel("theory-space $y$")
    ax_c.set_title(f"C.  Peaked scenario, $t=300$\n({len(spec_models)} total models in D{domain})",
                   color=PEAKED_COL, pad=10, fontsize=10)
    ax_c.set_aspect("equal", adjustable="box")

    # Panel D: BROAD at t=300
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.contourf(X, Y, Z, levels=20, cmap="viridis_r")
    ax_d.contour(X, Y, Z, levels=10, colors="white", alpha=0.25, linewidths=0.5)
    scatter_models(ax_d, gen_models)
    ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1)
    ax_d.set_xlabel("theory-space $x$"); ax_d.set_ylabel("theory-space $y$")
    ax_d.set_title(f"D.  Broad scenario, $t=300$\n({len(gen_models)} total models in D{domain})",
                   color=BROAD_COL, pad=10, fontsize=10)
    ax_d.set_aspect("equal", adjustable="box")

    # legend for dot encoding
    fig.text(0.5, 0.02,
             "Faint white background dots: every model published in this domain ("
             "publication history). "
             "Foreground dots: currently-active models (salience $\\geq 0.05$); "
             "red = high truthfulness ($\\theta_{\\mathrm{actual}} \\geq 0.50$), grey = low truthfulness. "
             "Dot size $\\propto$ salience. Colormap: yellow = peaks (unstable), dark = valleys (stable).",
             ha="center", fontsize=8, style="italic", color="#444", wrap=True)

    out = os.path.join(HERE, "..", "meta", "img", "landscape_figure.pdf")
    fig.savefig(out)
    print(f"Saved → {out}")

    # also save a PNG for quick inspection
    out_png = out.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
