import numpy as np
import mesa
from collections import defaultdict
from scientific_model import ScientificModel


class ResearchLab(mesa.Agent):
    """
    A research lab agent with a skill vector across scientific domains.

    The skill vector acts as a fingerprint: the complexity of any model this lab
    spawns is drawn from the lab's skill level in that domain. Labs therefore
    naturally gravitate toward models that match their own profile.

    Each step the lab makes a probabilistic three-way decision:
      - Exploit: commit to implementing an existing model (multi-step)
      - Invest:  train toward a model it cannot yet implement (gap > threshold)
      - Explore: publish a new model in an underexplored domain

    The top-N candidates (by expected value) are assembled and each is given a
    probability proportional to how well its complexity matches the lab's skill
    fingerprint. This means high-value models that are also a good skill match
    get chosen most often, but suboptimal matches can still win occasionally.

    Training speed follows an S-curve — labs with some existing knowledge
    train faster than those starting near zero.
    """

    def __init__(self, model, domain_skills: list[float],
                 train_threshold:    float = 0.2,   # min gap that requires training
                 skill_gain_attempt: float = 0.03,  # skill gained on successful attempt
                 skill_gain_train:   float = 0.06,  # max skill gained per training step
                 n_candidates:       int   = 5):    # how many top models to compare
        super().__init__(model)

        self.domain_skills      = domain_skills   # index = domain id
        self.train_threshold    = train_threshold
        self.skill_gain_attempt = skill_gain_attempt
        self.skill_gain_train   = skill_gain_train
        self.n_candidates       = n_candidates

        # outcome trackers
        self.publications      = 0
        self.reputation        = 0.0
        self.training_steps    = 0
        self.cross_lab_pubs    = 0
        self.current_domain: int | None = None
        self.domain_pubs: defaultdict[int, int] = defaultdict(int)
        self.reputation_at_last_cull             = 0.0

        # work-in-progress state
        self.current_target: ScientificModel | None = None
        self.work_progress:  int = 0

    # --- derived properties ---

    @property
    def home_domain(self) -> int:
        """Domain where this lab has highest skill."""
        return int(np.argmax(self.domain_skills))

    @property
    def mean_skill(self) -> float:
        return float(np.mean(self.domain_skills))

    # --- core mechanics ---

    def _sigmoid_gain(self, current: float, base_rate: float) -> float:
        """
        S-curve learning: gain = base_rate × current × (1 - current) × 4

        - Near 0.0 : very slow  (no foundation)
        - Near 0.5 : fastest    (peak = base_rate)
        - Near 1.0 : very slow  (diminishing returns)
        """
        return base_rate * current * (1.0 - current) * 4.0

    def _work_required(self, m: ScientificModel) -> int:
        """Steps needed to implement a model before publication."""
        return max(1, round(m.complexity * 5))

    def _similarity(self, m: ScientificModel) -> float:
        """
        Fingerprint match between this lab and a model.

        How closely does the lab's skill in m.domain match the model's complexity?
        Returns (0, 1] — higher means the model is a natural fit for this lab.

          similarity = 1 / (1 + |skill - complexity|)

        A lab with skill 0.7 is a near-perfect match for a model of complexity 0.7
        and a poor match for one of complexity 0.2. This steers probabilistic
        selection toward models that suit the lab's profile.
        """
        skill = self.domain_skills[m.domain]
        return 1.0 / (1.0 + abs(skill - m.complexity))

    def success_probability(self, m: ScientificModel) -> float:
        skill = self.domain_skills[m.domain]
        return float(np.clip(skill - m.complexity + 0.5, 0.0, 1.0))

    def _split_candidates(self):
        """
        Partition all available models into:
          exploit — gap ≤ threshold: lab can attempt this now
          invest  — gap > threshold: lab must train first
        """
        exploit, invest = [], []
        for m in self.model.scientific_models:
            gap = m.complexity - self.domain_skills[m.domain]
            if gap <= self.train_threshold:
                exploit.append(m)
            else:
                invest.append(m)
        return exploit, invest

    # --- action values ---

    def _exploit_value(self, m: ScientificModel) -> float:
        """
        Expected gain per step from committing to this model.
          = (success_prob × fidelity × novelty) / (work_required × (1 + own_pubs))
        """
        return (self.success_probability(m) * m.fidelity * m.novelty
                / (self._work_required(m) * (1.0 + self.domain_pubs[m.domain])))

    def _invest_value(self, m: ScientificModel) -> float:
        """
        Expected marginal gain from one step of training toward this model.
          = sigmoid_gain(current_skill) × fidelity × novelty
        """
        current = self.domain_skills[m.domain]
        return self._sigmoid_gain(current, self.skill_gain_train) * m.fidelity * m.novelty

    def _explore_value(self, domain: int) -> float:
        """
        Value of publishing a new model in this domain.
          = skill / (1 + n_models + own_pubs)

        Discounted by how saturated the domain already is and how much
        this lab has already published there.
        """
        n_models = sum(1 for m in self.model.scientific_models if m.domain == domain)
        return self.domain_skills[domain] / (1.0 + n_models + self.domain_pubs[domain])

    # --- actions ---

    def _explore(self, domain: int):
        """Publish a new model in an underexplored domain."""
        self.model.spawn_model(origin_lab_id=self.unique_id, domain=domain)
        self.current_domain = domain
        self.publications  += 1
        self.domain_pubs[domain] += 1
        new_model = self.model.scientific_models[-1]
        self.reputation += new_model.fidelity * new_model.novelty

    def _train(self, domain: int):
        """Spend this step building skill in a domain (S-curve rate)."""
        current = self.domain_skills[domain]
        self.domain_skills[domain] = min(1.0, current + self._sigmoid_gain(current, self.skill_gain_train))
        self.training_steps += 1
        self.current_domain  = domain

    def _attempt(self, target: ScientificModel):
        """Try to implement a model. On success: publish, cite, learn, possibly spawn."""
        self.current_domain = target.domain
        if self.random.random() < self.success_probability(target):
            self.publications += 1
            self.domain_pubs[target.domain] += 1

            # reputation scales with how close to the domain frontier
            domain_max     = max(
                (m.fidelity for m in self.model.scientific_models if m.domain == target.domain),
                default=target.fidelity
            )
            frontier_ratio = target.fidelity / domain_max
            self.reputation += frontier_ratio * target.fidelity * target.novelty
            target.cite()

            if target.origin_lab_id != self.unique_id:
                self.cross_lab_pubs += 1

            # learning by doing (S-curve)
            current = self.domain_skills[target.domain]
            self.domain_skills[target.domain] = min(
                1.0, current + self._sigmoid_gain(current, self.skill_gain_attempt)
            )

            # probabilistic spawn — less likely as domain saturates
            saturation = self.model.domain_saturation(target.domain)
            if self.random.random() < 0.15 * (1.0 - saturation):
                self.model.spawn_model(
                    origin_lab_id=self.unique_id,
                    domain=target.domain,
                    initial_novelty=0.7,
                )
                self.publications += 1
                self.domain_pubs[target.domain] += 1
                new_model = self.model.scientific_models[-1]
                self.reputation += new_model.fidelity * new_model.novelty

    # --- probabilistic decision ---

    def _choose_action(self, exploit: list, invest: list, best_domain: int):
        """
        Assemble the top-N candidates by expected value, then choose
        probabilistically by fingerprint similarity.

        Candidates drawn from:
          - top self.n_candidates exploit/invest models (by value)
          - explore (always included as an option)

        Probability of choosing candidate i:
          P(i) = similarity(i) / sum(similarity(j) for all j)

        For model candidates: similarity = 1 / (1 + |skill - complexity|)
        For explore:          similarity = lab's raw skill in that domain
        """
        scored = []
        for m in exploit:
            scored.append(('exploit', m, self._exploit_value(m)))
        for m in invest:
            scored.append(('invest',  m, self._invest_value(m)))

        # take top-N by value, then always add the explore option
        top = sorted(scored, key=lambda x: x[2], reverse=True)[:self.n_candidates]
        top.append(('explore', best_domain, self._explore_value(best_domain)))

        # similarity weights
        weights = []
        for action, target, _ in top:
            if action == 'explore':
                sim = self.domain_skills[target]
            else:
                sim = self._similarity(target)
            weights.append(max(sim, 1e-6))   # guard against zero

        total = sum(weights)
        probs = [w / total for w in weights]

        idx = self.random.choices(range(len(top)), weights=probs, k=1)[0]
        return top[idx]

    # --- step ---

    def step(self):
        # if already committed to a model, continue working — no new decision
        if self.current_target is not None:
            self.current_domain  = self.current_target.domain
            self.work_progress  += 1
            if self.work_progress >= self._work_required(self.current_target):
                self._attempt(self.current_target)
                self.current_target = None
                self.work_progress  = 0
            return

        exploit, invest = self._split_candidates()
        best_domain     = max(range(len(self.domain_skills)), key=self._explore_value)

        action, target, _ = self._choose_action(exploit, invest, best_domain)

        if action == 'explore':
            self._explore(target)
        elif action == 'invest':
            self._train(target.domain)
        else:
            self.current_target = target
            self.current_domain = target.domain
            self.work_progress  = 1
            if self.work_progress >= self._work_required(target):
                self._attempt(self.current_target)
                self.current_target = None
                self.work_progress  = 0
