"""
landscape.py — epistemic landscape for a single research domain (best-model variant).

Semantic redesign (Tier 1.5)
---------------------------
The previous landscape used height as a direct proxy for *instability*:
valleys (low height) were stable paradigms, peaks (high height) were
unstable theories. That coupling conflated two different things — *how
much established truth a region carries* and *how stable that region
is locally*.

In the best-model variant, altitude and stability are decoupled:

  truth(x, y)   ∈ [0, 1]   "holistic truth in a domain" at position (x, y)
                            high = closer to absolute truth in this region

  stability(x, y) ∈ [0, 1]  derived from the local gradient magnitude:
                            flat regions = stable; steep regions = unstable

The surface is a sum of two kinds of attractors:
  - *plateaus*: broad Gaussian bumps with modest peak height.
                Represent well-established paradigms — high truth AND flat.
  - *peaks*:    narrow Gaussian bumps with high peak height.
                Represent cutting-edge cutting-edge but unstable theory —
                high truth AND steep slopes.

Low regions (where neither plateau nor peak contributes) are undiscovered
or refuted territory — low truth.

Analogy (Newtonian/relativity, restated under new semantics)
------------------------------------------------------------
  on a plateau:   Newtonian mechanics — broad, established, flat (stable)
  on a peak:      cutting-edge result — locally elevated, narrow, fragile
  in a valley:    undiscovered or refuted territory — low truth

Key dynamics
------------
  step_toward_stable(x, y) — moves toward higher truth AND lower gradient,
                              i.e., toward the nearest plateau. Replaces the
                              previous step_toward_valley.

  breakthrough_across_ridge(x, y) — repositions to a different attractor
                                     (plateau or peak) than the current one,
                                     used by the breakthrough mechanic in
                                     researcher._explore.

  position_bias(x, y) — extra publication-bias inflation at high-gradient
                         (unstable) positions. Inflation scales with local
                         steepness, not with altitude.
"""

from __future__ import annotations
import numpy as np


class EpistemicLandscape:
    """
    2D Gaussian-mixture epistemic landscape for one research domain
    (best-model variant: truth and stability decoupled).

    Parameters
    ----------
    n_plateaus : int
        Broad elevated regions representing stable paradigms.
    n_peaks : int
        Narrow elevated regions representing unstable cutting-edge theory.
    rng : numpy Generator
        Seeded numpy random generator for reproducibility.
    """

    def __init__(self,
                 n_plateaus: int = 2,
                 n_peaks:    int = 4,
                 rng:        np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng()

        # --- plateaus: broad bumps, moderate altitude, flat at top ---
        # Width and height calibrated so plateaus dominate a ~15% radius
        # region but their additive contributions stay below 1.0 even
        # when two plateaus overlap.
        self.plateau_centers = rng.uniform(0.15, 0.85, (n_plateaus, 2))
        self.plateau_heights = rng.uniform(0.25, 0.40, n_plateaus)
        self.plateau_widths  = rng.uniform(0.04, 0.08, n_plateaus)

        # --- peaks: narrow bumps, high in altitude, sharp slopes ---
        # Higher per-peak amplitude than plateaus but very narrow, so they
        # represent localised high-truth-but-fragile regions.
        self.peak_centers = rng.uniform(0.10, 0.90, (n_peaks, 2))
        self.peak_heights = rng.uniform(0.30, 0.55, n_peaks)
        self.peak_widths  = rng.uniform(0.008, 0.020, n_peaks)

        # Background truth — undiscovered territory away from any attractor.
        self._baseline = 0.05

        # Position-bias amplifier ceiling (kept for parity with v3 mechanics).
        self.peak_bias_amplifier: float = 0.15

        # Stability calibration: stability = 1 / (1 + STABILITY_LAMBDA * |∇truth|).
        # λ = 0.6 chosen so the *mean* stability across the landscape is
        # ≈ 0.5, matching the v3 position-based stability (1 − height) which
        # also had mean ≈ 0.5. This keeps the new mechanics calibrated to
        # the same effective scales as the v3 downstream effects
        # (replication factor 0.75 + 0.25·s; debunk boost 1 + 0.5·(1−s);
        # salience decay × (1 − 0.25·s); stability attraction 1 + 0.30·s),
        # so the headline H1/H2 magnitudes should be approximately
        # comparable across the v3 → best-model landscape redesign.
        self.STABILITY_LAMBDA: float = 0.6

    # ------------------------------------------------------------------
    # Core landscape functions
    # ------------------------------------------------------------------

    def truth(self, x: float, y: float) -> float:
        """
        Holistic truth in the domain at position (x, y) ∈ [0, 1]².
        High = closer to absolute truth (a plateau or a peak).
        Low  = undiscovered or refuted territory.
        """
        t = self._baseline
        for i, (cx, cy) in enumerate(self.plateau_centers):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            t += self.plateau_heights[i] * np.exp(-d2 / (2.0 * self.plateau_widths[i]))
        for i, (cx, cy) in enumerate(self.peak_centers):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            t += self.peak_heights[i] * np.exp(-d2 / (2.0 * self.peak_widths[i]))
        return float(np.clip(t, 0.0, 1.0))

    # Backward-compatible alias: external code that still reads .height(...)
    # gets the same number it would have read under the old semantics, since
    # downstream coupling cares about position-based magnitude. This avoids
    # silent breakage in any callers that haven't been migrated yet.
    def height(self, x: float, y: float) -> float:
        return self.truth(x, y)

    def gradient(self, x: float, y: float, eps: float = 0.005) -> np.ndarray:
        """Numerical gradient of truth(x, y). Points uphill."""
        dx = (self.truth(x + eps, y) - self.truth(x - eps, y)) / (2.0 * eps)
        dy = (self.truth(x, y + eps) - self.truth(x, y - eps)) / (2.0 * eps)
        return np.array([dx, dy])

    def gradient_magnitude(self, x: float, y: float, eps: float = 0.005) -> float:
        g = self.gradient(x, y, eps)
        return float(np.sqrt(g[0] ** 2 + g[1] ** 2))

    def stability(self, x: float, y: float) -> float:
        """
        Local stability ∈ [0, 1] derived from gradient magnitude:
            stability = 1 / (1 + λ |∇truth(x, y)|)
        Flat regions (plateau interiors, low-truth basins) → high stability.
        Steep regions (peak edges, plateau rims) → low stability.
        """
        return float(1.0 / (1.0 + self.STABILITY_LAMBDA * self.gradient_magnitude(x, y)))

    # ------------------------------------------------------------------
    # Movement dynamics
    # ------------------------------------------------------------------

    def step_toward_stable(self,
                           x: float, y: float,
                           step_size: float = 0.05) -> tuple[float, float]:
        """
        Move one step toward higher truth AND lower gradient, i.e., toward
        the nearest plateau interior.

        Replaces the old step_toward_valley. Strategy is a greedy local
        search in the four cardinal directions plus staying still; we pick
        the direction that maximises (truth - λ_score · |gradient|). This
        favours regions that are both more truthful and flatter, so mature
        research programmes drift toward established paradigm plateaus.
        """
        candidates = [(x + step_size, y), (x - step_size, y),
                      (x, y + step_size), (x, y - step_size),
                      (x, y)]
        lam_score = 0.30  # relative weight of flatness vs truth
        best = candidates[0]
        best_score = -np.inf
        for px, py in candidates:
            score = self.truth(px, py) - lam_score * self.gradient_magnitude(px, py)
            if score > best_score:
                best_score = score
                best = (px, py)
        return (float(np.clip(best[0], 0.01, 0.99)),
                float(np.clip(best[1], 0.01, 0.99)))

    # Backward-compatible alias for any code still calling step_toward_valley.
    def step_toward_valley(self,
                           x: float, y: float,
                           step_size: float = 0.05) -> tuple[float, float]:
        return self.step_toward_stable(x, y, step_size=step_size)

    def breakthrough_across_ridge(self,
                                  x: float, y: float,
                                  rng: np.random.Generator) -> tuple[float, float]:
        """
        Sample a new position centred on a different attractor (plateau or
        peak) than the current one. Used by the breakthrough mechanic in
        researcher._explore: a paradigm-shifting result repositions the
        model across the landscape to a different region of theory-space.
        """
        all_centers = list(self.plateau_centers) + list(self.peak_centers)
        if not all_centers:
            return (x, y)
        dists = [(cx - x) ** 2 + (cy - y) ** 2 for cx, cy in all_centers]
        current_idx = int(np.argmin(dists))
        candidates = [c for i, c in enumerate(all_centers) if i != current_idx]
        if not candidates:
            return (x, y)
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        new_x = float(np.clip(chosen[0] + rng.normal(0, 0.05), 0.01, 0.99))
        new_y = float(np.clip(chosen[1] + rng.normal(0, 0.05), 0.01, 0.99))
        return (new_x, new_y)

    # ------------------------------------------------------------------
    # Reported-quality amplification
    # ------------------------------------------------------------------

    def position_bias(self, x: float, y: float) -> float:
        """
        Extra publication-bias inflation at landscape position (x, y).

        Inflation scales with local steepness rather than altitude:
        results from steep, unstable regions of theory-space are
        systematically over-reported because their narrow validity is
        easy to overstate. Plateau interiors and flat low-truth basins
        contribute negligible bias; peak edges contribute up to the cap.

          extra_bias = min(|∇truth(x, y)| × 0.05, peak_bias_amplifier)
        """
        instability = self.gradient_magnitude(x, y)
        # Calibrated so the *mean* position-bias across the landscape is
        # roughly 0.075 (matching v3's height-based mean) and only the
        # steepest peak edges saturate at the 0.15 cap.
        return float(min(instability * 0.05, self.peak_bias_amplifier))
