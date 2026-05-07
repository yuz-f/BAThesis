"""
landscape.py — epistemic landscape for a single research domain.

Each domain has a 2D potential-energy surface defined by a mixture of
Gaussian wells (valleys = stable paradigms) and Gaussian bumps (peaks =
unstable/intermediate theories).

Analogy
-------
  valley at low elevation   → Newtonian mechanics (well-established, high stability)
  valley at high elevation  → General Relativity (more comprehensive but harder to reach)
  peaks between valleys     → intermediate theories that failed to consolidate

The landscape height h(x, y) ∈ [0, 1] encodes epistemic stability:
  h ≈ 0  (valley bottom) → maximum stability: models here are hard to debunk,
                           easy to replicate, and stay salient longer.
  h ≈ 1  (peak top)      → maximum instability: easy to debunk, high replication
                           failure, over-reported in the literature.

Stability := 1 − h.

Reported-quality amplification
-------------------------------
The reported-truth map is an amplified version of the same surface: peaks are
pushed up further in perceived quality than their actual quality warrants.
Researchers over-attribute significance to results from unstable theory-space
positions (exciting new territory reads as groundbreaking before it collapses).
`position_bias(x, y)` returns the extra bias inflation from this amplification.

Gradient dynamics
-----------------
Follow-up models (built on a parent) are nudged one gradient-descent step toward
the nearest valley: `step_toward_valley(x, y)`. This operationalises cumulative
research converging on stable theoretical anchors.
"""

from __future__ import annotations
import numpy as np


class EpistemicLandscape:
    """
    2D Gaussian-mixture epistemic landscape for one research domain.

    Parameters
    ----------
    n_valleys : int
        Number of stable attractors (Gaussian wells, lower height).
    n_peaks : int
        Number of unstable regions (Gaussian bumps, higher height).
    rng : numpy Generator
        Seeded numpy random generator for reproducibility.
    """

    def __init__(self,
                 n_valleys: int = 3,
                 n_peaks:   int = 4,
                 rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()

        # --- valleys: Gaussian wells that pull height downward ---
        self.valley_centers = rng.uniform(0.10, 0.90, (n_valleys, 2))
        # depth: how far below baseline the valley floor sits
        self.valley_depths  = rng.uniform(0.25, 0.45, n_valleys)
        # width: narrow wells = sharp paradigms; wide wells = broad domains
        self.valley_widths  = rng.uniform(0.03, 0.10, n_valleys)

        # --- peaks: Gaussian bumps that push height upward ---
        self.peak_centers  = rng.uniform(0.10, 0.90, (n_peaks, 2))
        self.peak_heights  = rng.uniform(0.12, 0.30, n_peaks)
        self.peak_widths   = rng.uniform(0.02, 0.08, n_peaks)

        # Baseline: mid-height so the average position is moderately unstable.
        # Valleys pull below this; peaks push above it.
        self._baseline = 0.50

        # Maximum extra bias inflation added at peak positions (reported-truth
        # landscape amplification).  0.15 means a model at full-peak height
        # gets +0.15 added to its publication-bias draw.
        self.peak_bias_amplifier: float = 0.15

    # ------------------------------------------------------------------
    # Core landscape functions
    # ------------------------------------------------------------------

    def height(self, x: float, y: float) -> float:
        """
        Landscape height at position (x, y) ∈ [0, 1]².
        Low  = valley (stable).
        High = peak   (unstable).
        """
        h = self._baseline
        for i, (cx, cy) in enumerate(self.valley_centers):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            h -= self.valley_depths[i] * np.exp(-d2 / (2.0 * self.valley_widths[i]))
        for i, (cx, cy) in enumerate(self.peak_centers):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            h += self.peak_heights[i] * np.exp(-d2 / (2.0 * self.peak_widths[i]))
        return float(np.clip(h, 0.0, 1.0))

    def stability(self, x: float, y: float) -> float:
        """Stability ∈ [0, 1] — 1 at valley bottoms, 0 at peak tops."""
        return 1.0 - self.height(x, y)

    def gradient(self, x: float, y: float, eps: float = 0.005) -> np.ndarray:
        """
        Numerical gradient of the height function.
        Points uphill; negate to descend toward nearest valley.
        """
        dx = (self.height(x + eps, y) - self.height(x - eps, y)) / (2.0 * eps)
        dy = (self.height(x, y + eps) - self.height(x, y - eps)) / (2.0 * eps)
        return np.array([dx, dy])

    def step_toward_valley(self,
                           x: float, y: float,
                           step_size: float = 0.05) -> tuple[float, float]:
        """
        One gradient-descent step from (x, y) toward the nearest valley.

        Used when spawning follow-up models: each successive publication in
        a research programme moves slightly closer to a stable theoretical
        anchor, operationalising cumulative theoretical refinement.
        """
        grad     = self.gradient(x, y)
        new_pos  = np.array([x, y]) - step_size * grad
        return (float(np.clip(new_pos[0], 0.01, 0.99)),
                float(np.clip(new_pos[1], 0.01, 0.99)))

    # ------------------------------------------------------------------
    # Reported-quality amplification
    # ------------------------------------------------------------------

    def position_bias(self, x: float, y: float) -> float:
        """
        Extra publication-bias inflation at landscape position (x, y).

        The reported-truth landscape is an amplified version of the height
        surface: models published near peaks are systematically over-reported
        because researchers in unstable theory-space tend to frame results as
        more definitive than they are.

          extra_bias = height(x, y) × peak_bias_amplifier
        """
        return self.height(x, y) * self.peak_bias_amplifier
