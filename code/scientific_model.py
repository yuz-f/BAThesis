class ScientificModel:
    """
    A passive environment object representing a published scientific model.

    Salience grows with citations (other labs adopting the model) and decays
    each step. High-truthfulness models resist decay — but only while they are
    still being actively used. Once citations stop, the shield fades over a
    recency window and the model decays at its full base rate.

    This creates era dynamics: a superseded model loses its protection and
    fades within a predictable number of steps, making room for newer work.

    Truthfulness measures how accurate and productive a model is (0–1).
    """

    RECENCY_WINDOW = 20   # steps without a citation before shield fully drops

    def __init__(self, uid: int, domain: int, complexity: list[float],
                 truthfulness: float, origin_lab_id: int,
                 salience_changes: float = 0.05):
        self.uid               = uid
        self.domain            = domain       # domain this model was published in
        self.complexity        = complexity   # per-domain complexity vector, mirrors lab fingerprint
        self.truthfulness      = truthfulness
        self.salience          = 0.5
        self.origin_lab_id     = origin_lab_id
        self.citations         = 0
        self.steps_since_cited = 0            # recency counter — resets on each citation
        self.salience_rate     = salience_changes        # gain per citation
        self.salience_decay    = salience_changes * 0.5  # base decay rate

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
        effective_decay = self.salience_decay * (1.0 - self.truthfulness * recency_factor)
        self.salience   = max(0.0, self.salience - effective_decay)
