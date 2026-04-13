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
                 skill_gain_attempt: float = 0.05,  # skill gained on successful attempt
                 skill_gain_train:   float = 0.08): # max skill gained per training step
        super().__init__(model)

        self.domain_skills      = domain_skills   # index = domain id
        self.train_threshold    = train_threshold
        self.skill_gain_attempt = skill_gain_attempt
        self.skill_gain_train   = skill_gain_train

        # outcome trackers
        self.publications      = 0
        self.reputation        = 0.0
        self.training_steps    = 0
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
        """
        Steps needed to implement a model.

        Base cost scales with primary-domain complexity. A saturation multiplier
        makes it progressively harder to publish as a domain matures: when the
        best existing models already approach the truthfulness cap, each new
        contribution demands more rigour to clear the bar.

          work = max(3, round(complexity × 8)) × (1 + domain_saturation × 0.5)

        At saturation 0.0: no penalty.  At 0.5: +25%.  At 0.9: +45%.
        """
        base       = max(3, round(m.complexity[m.domain] * 8))
        saturation = self.model.domain_saturation(m.domain)
        return max(1, round(base * (1.0 + saturation * 0.5)))

    def _similarity(self, m: ScientificModel) -> float:
        """
        Fingerprint match: proficiency × max(0, COR(complexity, skill))

        COR is the Pearson correlation between the lab's skill vector and the
        model's complexity vector — it measures whether the lab's peaks align
        with the model's complexity peaks, independent of absolute scale.

          - COR = +1 : profiles are perfectly co-aligned (specialist ↔ specialist)
          - COR =  0 : no relationship (specialist ↔ flat generalist model)
          - COR = -1 : anti-aligned (peaks and troughs are swapped)

        Negative correlations are clipped to 0 — a lab won't be actively
        repelled, it simply has no affinity for a mismatched model.

        Proficiency = skill in the model's primary domain. Scales the alignment
        score by whether the lab can actually operate in that domain at all.
        Both factors together put specialist-specialist matches at the top and
        give generalist labs a moderate but broad affinity.
        """
        skill_arr = np.array(self.domain_skills)
        comp_arr  = np.array(m.complexity)
        cor = float(np.corrcoef(skill_arr, comp_arr)[0, 1])
        if np.isnan(cor):   # one vector is uniform → no alignment signal
            cor = 0.0
        proficiency = self.domain_skills[m.domain]
        return max(1e-6, proficiency * max(0.0, cor))

    def success_probability(self, m: ScientificModel) -> float:
        """
        Success probability blends fingerprint similarity with a raw proficiency floor.

          P = max(_similarity, proficiency × 0.5)

        _similarity (proficiency × COR) rewards well-matched specialist labs with
        high probability. The floor ensures a generalist lab that genuinely has
        skill in the model's primary domain always has a meaningful chance —
        without it, broad uniform profiles produce low COR even against similarly
        broad models, suppressing generalist publication rates unfairly.
        """
        proficiency = self.domain_skills[m.domain]
        return float(np.clip(max(self._similarity(m), proficiency * 0.5), 0.0, 1.0))

    def _split_candidates(self):
        """
        Partition all available models into:
          exploit — gap ≤ threshold: lab can attempt this now
          invest  — gap > threshold: lab must train first
        """
        exploit, invest = [], []
        for m in self.model.scientific_models:
            gap = m.complexity[m.domain] - self.domain_skills[m.domain]
            if gap <= self.train_threshold:
                exploit.append(m)
            else:
                invest.append(m)
        return exploit, invest

    # --- action values ---

    def _exploit_value(self, m: ScientificModel) -> float:
        """
        Expected gain per step from committing to this model.
          = (success_prob × truthfulness × salience) / (work_required × (1 + own_pubs))
        """
        return (self.success_probability(m) * m.truthfulness * m.salience
                / (self._work_required(m) * (1.0 + self.domain_pubs[m.domain])))

    def _invest_value(self, m: ScientificModel) -> float:
        """
        Expected marginal gain from one step of training toward this model.
          = sigmoid_gain(current_skill) × truthfulness

        Salience is intentionally excluded: training is a long-term capability
        investment. Whether the field is currently excited about a model should
        not determine whether a lab builds skill toward it — only the model's
        quality (truthfulness) and how fast the lab can grow (sigmoid_gain) matter.
        """
        current = self.domain_skills[m.domain]
        return self._sigmoid_gain(current, self.skill_gain_train) * m.truthfulness

    def _explore_value(self, domain: int) -> float:
        """
        Value of publishing a new model in this domain.
          = skill / (1 + log(1 + n_models) + own_pubs)

        n_models uses log-scaling so a domain with 20 models is only ~3×
        less attractive than an empty domain, not 20× — keeping explore
        competitive as domains fill up. own_pubs still discounts a lab's
        own repeated publishing, preventing monopolisation.
        """
        n_models = sum(1 for m in self.model.scientific_models if m.domain == domain)
        return self.domain_skills[domain] / (1.0 + np.log1p(n_models) + self.domain_pubs[domain])

    # --- actions ---

    def _record_spawn(self, domain: int, count_as_pub: bool = True):
        """
        Update trackers after a model has been spawned.

        count_as_pub=True  (explore):  intentional new publication — counts fully.
        count_as_pub=False (attempt):  opportunistic follow-up finding — earns
                                       reputation but is not a standalone publication.
        """
        if count_as_pub:
            self.publications += 1
            self.domain_pubs[domain] += 1
        new_model = self.model.scientific_models[-1]
        self.reputation += new_model.truthfulness * new_model.salience

    def _explore(self, domain: int):
        """Publish a new model in an underexplored domain."""
        self.model.spawn_model(origin_lab_id=self.unique_id, domain=domain)
        self.current_domain = domain
        self._record_spawn(domain)

    def _assimilate(self, m: ScientificModel, rate: float):
        """
        Nudge the full skill vector toward the model's complexity profile.

        Each domain is updated independently:
          gain_d = sigmoid_gain(skill_d, rate) × min(1, gap_d)

        The gap term scales gain down as the skill approaches the target —
        rapid progress when far behind, near-zero once aligned. Skills only
        move upward; labs do not unlearn domains where they exceed the model.
        """
        for d in range(len(self.domain_skills)):
            gap = m.complexity[d] - self.domain_skills[d]
            if gap > 0.0:
                gain = self._sigmoid_gain(self.domain_skills[d], rate) * min(1.0, gap)
                self.domain_skills[d] = float(np.clip(self.domain_skills[d] + gain, 0.01, 0.95))

    def _profile_alignment(self, m: ScientificModel) -> float:
        """
        Pure profile alignment: max(0, COR(skill_vector, complexity_vector)).

        Unlike _similarity this excludes proficiency — it measures only how
        well the lab's skill shape matches the model's complexity shape,
        independent of absolute level. Used to scale learning rates.

        NaN (uniform vectors with zero variance) → 0 (no alignment signal).
        Negative correlations are clipped to 0 — misalignment slows but
        does not reverse learning. Floor at 0.2 so learning never fully stalls.
        """
        cor = float(np.corrcoef(
            np.array(self.domain_skills),
            np.array(m.complexity)
        )[0, 1])
        if np.isnan(cor):
            cor = 0.0
        return max(max(0.0, cor), 0.2)

    def _train(self, m: ScientificModel):
        """
        Spend this step assimilating toward the model's full complexity profile.

        Rate is scaled by profile alignment: a lab whose skill fingerprint
        already resembles the model's complexity learns fast; one that is
        working far outside its profile learns proportionally slower.

          effective_rate = skill_gain_train × alignment   (floor 0.2)
        """
        effective_rate = self.skill_gain_train * self._profile_alignment(m)
        self._assimilate(m, effective_rate)
        self.training_steps += 1
        self.current_domain = m.domain

    def _attempt(self, target: ScientificModel):
        """Try to implement a model. On success: publish, cite, assimilate, possibly spawn.
        On failure: partial assimilation — the work was not wasted, just insufficient."""
        self.current_domain = target.domain
        if self.random.random() < self.success_probability(target):
            self.publications += 1
            self.domain_pubs[target.domain] += 1

            # reputation = quality × current relevance of the model
            self.reputation += target.truthfulness * target.salience
            target.cite()

            # full assimilation toward the model's complexity profile
            self._assimilate(target, self.skill_gain_attempt)
        else:
            # failed attempt still teaches — partial assimilation scaled by alignment.
            # working on a foreign model teaches less even when you fail at it.
            effective_rate = self.skill_gain_attempt * 0.3 * self._profile_alignment(target)
            self._assimilate(target, effective_rate)

            # opportunistic spawn — probability decreases gradually as the domain
            # fills up, reflecting that follow-up findings become harder to justify
            # the more crowded a field already is.
            #   spawn_prob = 0.15 / (1 + log(1 + n_models))
            # At n=0 → 0.15, at n=10 → ~0.06, at n=20 → ~0.05 (log dampening).
            # Does NOT count as a standalone publication — only reputation is earned.
            n_domain = sum(1 for m in self.model.scientific_models
                           if m.domain == target.domain)
            spawn_prob = 0.15 / (1.0 + np.log1p(n_domain))
            if self.random.random() < spawn_prob:
                self.model.spawn_model(
                    origin_lab_id=self.unique_id,
                    domain=target.domain,
                    initial_salience=0.7,
                )
                self._record_spawn(target.domain, count_as_pub=False)

    # --- probabilistic decision ---

    def _choose_action(self, exploit: list, invest: list, best_domain: int):
        """
        Assemble top candidates by expected value, then choose probabilistically
        using a unified weight per candidate:

          weight(model) = match + salience + truthfulness

        where:
          match       = fingerprint similarity (Pearson COR × proficiency)
          salience    = model's current salience
          truthfulness = model's truthfulness (quality / productivity)

        For the explore option, salience and truthfulness don't exist, so:
          weight(explore) = match (= lab's raw skill in that domain)

        This means a model gets chosen more often when it's a good skill
        match AND currently salient AND high-quality — all three factors
        contribute equally rather than truthfulness being buried in the value
        ranking and similarity being the sole selection criterion.

        Candidates: top 3 models (by expected value) + explore domain.
        Capping at 3 ensures explore always gets a fair share of weight.
        """
        scored = []
        for m in exploit:
            scored.append(('exploit', m, self._exploit_value(m)))
        for m in invest:
            scored.append(('invest',  m, self._invest_value(m)))

        top = sorted(scored, key=lambda x: x[2], reverse=True)[:3]
        top.append(('explore', best_domain, self._explore_value(best_domain)))

        weights = []
        for action, target, _ in top:
            if action == 'explore':
                w = self.domain_skills[target]           # match only (no model to inspect)
            else:
                w = (self._similarity(target)            # match
                     + target.salience                   # salience
                     + target.truthfulness)              # truthfulness
            weights.append(max(w, 1e-6))

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

        # sample explore domain proportional to explore_value so every domain has
        # a nonzero chance — prevents domains from being permanently ignored just
        # because they currently have a lower value than the best option.
        n_domains    = len(self.domain_skills)
        explore_vals = [max(self._explore_value(d), 1e-6) for d in range(n_domains)]
        best_domain  = self.random.choices(range(n_domains), weights=explore_vals, k=1)[0]

        action, target, _ = self._choose_action(exploit, invest, best_domain)

        if action == 'explore':
            self._explore(target)
        elif action == 'invest':
            self._train(target)
        else:
            self.current_target = target
            self.current_domain = target.domain
            self.work_progress  = 1
            if self.work_progress >= self._work_required(target):
                self._attempt(self.current_target)
                self.current_target = None
                self.work_progress  = 0
