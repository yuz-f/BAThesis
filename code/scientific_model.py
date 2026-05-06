from __future__ import annotations

import numpy as np


class ScientificModel:
    """
    A passive environment object representing a published scientific model.

    Salience grows with citations (other labs adopting the model) and decays
    each step. High-truthfulness models resist decay — but only while they are
    still being actively used. Once citations stop, the shield fades over a
    recency window and the model decays at its full base rate.

    This creates era dynamics: a superseded model loses its protection and
    fades within a predictable number of steps, making room for newer work.

    Two truthfulness attributes model publication bias:
      actual_truthfulness  — the real quality; drives replication outcomes
      reported_truthfulness — actual + bias_inflation; what the community sees
                              and what drives agent decisions

    bias_inflation is fixed at spawn, representing the inflated effect sizes
    and selective reporting endemic in published literature.
    """

    RECENCY_WINDOW = 20   # steps without a citation before shield fully drops

    def __init__(self, uid: int, domain: int, complexity: list[float],
                 truthfulness: float, origin_lab_id: int,
                 salience_changes: float = 0.05,
                 parent_uid: int | None = None,
                 bias_inflation: float = 0.0,
                 author_agent_id: int | None = None):
        self.uid                  = uid
        self.domain               = domain
        self.complexity           = complexity
        self.actual_truthfulness  = truthfulness
        self._bias_inflation      = float(np.clip(bias_inflation, 0.0, 0.30))
        self.salience             = 0.5
        self.origin_lab_id        = origin_lab_id
        self.parent_uid           = parent_uid        # uid of model this was built upon
        self.author_agent_id      = author_agent_id   # unique_id of the researcher who published it
        self.citations            = 0
        self.steps_since_cited    = 0
        self.salience_rate        = salience_changes
        self.salience_decay       = salience_changes * 0.5

        # precomputed for fast Pearson correlation in researcher._pearson_cor
        _ca = np.array(complexity, dtype=float)
        self._comp_centered = _ca - _ca.mean()
        self._comp_ssq      = float(np.dot(self._comp_centered, self._comp_centered))

    @property
    def reported_truthfulness(self) -> float:
        """Public perception: actual quality + publication bias inflation."""
        return min(self.actual_truthfulness + self._bias_inflation, 0.95)

    def cite(self):
        """Called when a lab successfully uses this model — salience rises, recency resets."""
        self.citations         += 1
        self.steps_since_cited  = 0
        self.salience           = min(1.0, self.salience + self.salience_rate)

    def decay(self):
        """
        Salience decays each step.

        Truthfulness shield: high-truthfulness models resist decay, but only
        while they are still being cited. The shield fades linearly to zero
        over RECENCY_WINDOW uncited steps.

          recency_factor  = max(0, 1 − steps_since_cited / RECENCY_WINDOW)
          effective_decay = base_decay × (1 − truthfulness × recency_factor)

        A model cited this step has full protection.
        After RECENCY_WINDOW uncited steps, protection is gone and it decays
        at the full base rate regardless of how truthful it is.
        """
        self.steps_since_cited += 1
        recency_factor  = max(0.0, 1.0 - self.steps_since_cited / self.RECENCY_WINDOW)
        # Shield based on reported truthfulness — community protects models it perceives as good
        effective_decay = self.salience_decay * (1.0 - self.reported_truthfulness * recency_factor)
        self.salience   = max(0.0, self.salience - effective_decay)
