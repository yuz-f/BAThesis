from __future__ import annotations

import numpy as np


class ScientificModel:
    """
    A passive environment object representing a published scientific model.

    Epistemic landscape integration
    --------------------------------
    Each model has a 2D `position` on its domain's EpistemicLandscape and a
    cached `landscape_stability` score derived from that position:

      stability ≈ 1  (valley)  — well-established paradigm; easy to replicate,
                                  hard to debunk, salience decays slowly.
      stability ≈ 0  (peak)    — unstable intermediate theory; hard to replicate,
                                  easy to debunk, fades quickly.

    The position is set by the world at spawn time from the publishing
    researcher's current theory-space position in that domain.  Follow-up
    models are placed one gradient-descent step closer to the nearest valley,
    so research programmes converge toward stable theoretical anchors.

    Publication bias
    ----------------
    `bias_inflation` is set at spawn and includes a landscape component:
    models published from peak positions are over-reported in perceived quality
    (the reported-truth map has amplified peaks relative to actual quality).

    Salience / decay
    ----------------
    Valley models resist salience decay more strongly: their stability reduces
    the effective decay rate by up to 25%, reflecting that well-established
    results stay salient even without constant new citations.
    """

    RECENCY_WINDOW = 20

    def __init__(self, uid: int, domain: int, complexity: list[float],
                 truthfulness: float, origin_lab_id: int,
                 salience_changes: float = 0.05,
                 parent_uid: int | None = None,
                 bias_inflation: float = 0.0,
                 author_agent_id: int | None = None,
                 position: tuple[float, float] = (0.5, 0.5),
                 landscape_stability: float = 0.5):
        self.uid                  = uid
        self.domain               = domain
        self.complexity           = complexity
        self.actual_truthfulness  = truthfulness
        # Cap raised to 0.45 to accommodate landscape-boosted peak inflation
        self._bias_inflation      = float(np.clip(bias_inflation, 0.0, 0.45))
        self.salience             = 0.5
        self.origin_lab_id        = origin_lab_id
        self.parent_uid           = parent_uid
        self.author_agent_id      = author_agent_id
        self.citations            = 0
        self.steps_since_cited    = 0
        self.salience_rate        = salience_changes
        self.salience_decay       = salience_changes * 0.5

        # Epistemic landscape position and stability
        self.position:             tuple[float, float] = position
        self.landscape_stability:  float = landscape_stability

        # precomputed for fast Pearson correlation in researcher._pearson_cor
        _ca = np.array(complexity, dtype=float)
        self._comp_centered = _ca - _ca.mean()
        self._comp_ssq      = float(np.dot(self._comp_centered, self._comp_centered))

    @property
    def reported_truthfulness(self) -> float:
        """Public perception: actual quality + publication bias inflation."""
        return min(self.actual_truthfulness + self._bias_inflation, 0.95)

    def cite(self):
        self.citations         += 1
        self.steps_since_cited  = 0
        self.salience           = min(1.0, self.salience + self.salience_rate)

    def decay(self):
        """
        Salience decays each step.

        Two decay modifiers:
          1. Truthfulness shield (original): high-truth models resist decay
             while actively cited, fading over RECENCY_WINDOW uncited steps.
          2. Landscape stability (new): valley models decay up to 25% more
             slowly than peak models, reflecting that stable paradigms stay
             relevant without constant new citations.

          effective_decay = base_decay
                          × (1 − reported_truthfulness × recency_factor)
                          × (1 − 0.25 × landscape_stability)
        """
        self.steps_since_cited += 1
        recency_factor  = max(0.0, 1.0 - self.steps_since_cited / self.RECENCY_WINDOW)
        effective_decay = self.salience_decay * (1.0 - self.reported_truthfulness * recency_factor)
        # Valley models persist longer — their stability buffers against obsolescence
        effective_decay *= (1.0 - 0.25 * self.landscape_stability)
        self.salience   = max(0.0, self.salience - effective_decay)
