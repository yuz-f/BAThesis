class ScientificModel:
    """
    A passive environment object representing a published scientific model.

    Novelty grows with citations (other labs adopting the model)
    and decays each time step if citations stop coming in.
    """

    def __init__(self, uid: int, domain: int, complexity: float,
                 fidelity: float, origin_lab_id: int,
                 novelty_changes: float = 0.05):
        self.uid           = uid
        self.domain        = domain
        self.complexity    = complexity
        self.fidelity      = fidelity
        self.novelty       = 0.5
        self.origin_lab_id = origin_lab_id
        self.citations     = 0
        self.novelty_rate  = novelty_changes         # gain per citation
        self.novelty_decay = novelty_changes * 0.5   # base decay rate

    def cite(self):
        """Called when a lab successfully uses this model — novelty rises."""
        self.citations += 1
        self.novelty = min(1.0, self.novelty + self.novelty_rate)

    def decay(self):
        """
        Novelty decays each step.
        Fidelity shield: high-fidelity models resist decay — × (1 - fidelity)

        effective_decay = base_decay × (1 - fidelity)
        """
        effective_decay = self.novelty_decay * (1.0 - self.fidelity)
        self.novelty = max(0.0, self.novelty - effective_decay)

