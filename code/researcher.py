import numpy as np
import mesa
from collections import defaultdict
from scientific_model import ScientificModel


class Researcher(mesa.Agent):
    """
    An individual researcher agent with a personal skill vector.

    Starts with skills drawn from their home lab's fingerprint, then learns
    independently through assimilation: working on models with different
    complexity profiles gradually shifts the researcher's skills toward
    those profiles — without forgetting what they already know.

    Each step the researcher makes a probabilistic three-way decision:
      - Exploit: commit to implementing an existing model (multi-step)
      - Invest:  train toward a model they cannot yet implement
      - Explore: publish a new model in an underexplored domain

    The researcher belongs to a lab for their entire career — they do not
    switch labs. Diversity in skill profiles emerges organically through
    the models they choose to work on.
    """

    # Fraction of the full assimilation rate applied to domains
    # other than the model's primary domain.
    SECONDARY_LEARN_FACTOR = 0.12

    def __init__(self, model, lab_id: int, domain_skills: list[float],
                 lab_fingerprint:    list[float] | None = None,
                 train_threshold:    float = 0.2,
                 skill_gain_attempt: float = 0.05,
                 skill_gain_train:   float = 0.08):
        super().__init__(model)

        self.lab_id         = lab_id
        self.domain_skills  = domain_skills
        # Fixed institutional anchor — skills drift back toward this baseline
        # for non-peak domains that rise above it (slow forgetting of unused skills).
        self.lab_fingerprint = (list(lab_fingerprint)
                                if lab_fingerprint is not None
                                else list(domain_skills))
        self.train_threshold    = train_threshold
        self.skill_gain_attempt = skill_gain_attempt
        self.skill_gain_train   = skill_gain_train

        # outcome trackers
        self.publications    = 0
        self.reputation      = 0.0
        self.training_steps  = 0
        self.exploit_steps   = 0
        self.explore_steps   = 0
        self.debunk_steps    = 0
        self.current_domain: int | None = None
        self.domain_pubs: defaultdict[int, int] = defaultdict(int)
        self.reputation_at_last_cull = 0.0

        # work-in-progress state
        self.current_target: ScientificModel | None = None
        self.work_progress:  int = 0

    # --- derived properties ---

    @property
    def home_domain(self) -> int:
        """Domain where this researcher has highest skill."""
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
        Scales with complexity and domain saturation; experts work faster.
        """
        base         = max(3, round(m.complexity[m.domain] * 8))
        saturation   = self.model._cached_saturations[m.domain]
        skill_factor = 0.5 + self.domain_skills[m.domain]  # range ~0.51–1.45
        return max(1, round(base * (1.0 + saturation * 0.5) / skill_factor))

    def _pearson_cor(self, m: ScientificModel) -> float:
        """
        Inline Pearson correlation between skill vector and model complexity.
        Faster than np.corrcoef — computes one scalar directly, no matrix overhead.
        Returns 0.0 when either vector has zero variance (uniform profiles).
        """
        skill = np.array(self.domain_skills)
        comp  = np.array(m.complexity)
        sc    = skill - skill.mean()
        cc    = comp  - comp.mean()
        denom = np.sqrt((sc * sc).sum() * (cc * cc).sum())
        return float(np.dot(sc, cc) / denom) if denom > 1e-10 else 0.0

    def _similarity(self, m: ScientificModel) -> float:
        """proficiency × max(0, COR) — how well the researcher fits this model."""
        cor = max(0.0, self._pearson_cor(m))
        return max(1e-6, self.domain_skills[m.domain] * cor)

    def success_probability(self, m: ScientificModel) -> float:
        """
        P = max(_similarity, proficiency × 0.5)
        Floor ensures a researcher with genuine domain skill always has a chance.
        """
        proficiency = self.domain_skills[m.domain]
        return float(np.clip(max(self._similarity(m), proficiency * 0.5), 0.0, 1.0))

    def _split_candidates(self):
        """
        Partition all available models into:
          exploit — gap ≤ threshold: researcher can attempt this now
          invest  — gap > threshold: researcher must train first
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
        """
        current = self.domain_skills[m.domain]
        return self._sigmoid_gain(current, self.skill_gain_train) * m.truthfulness

    def _explore_value(self, domain: int) -> float:
        """
        Value of publishing a new model in this domain.
          = skill × (1 − saturation) / (1 + log(1 + n_models) + own_pubs)
        Saturated domains offer little marginal scientific value.
        """
        n_models   = self.model._cached_domain_counts.get(domain, 0)
        saturation = self.model._cached_saturations[domain]
        return (self.domain_skills[domain] * max(0.0, 1.0 - saturation)
                / (1.0 + np.log1p(n_models) + self.domain_pubs[domain]))


    # --- actions ---

    def _record_spawn(self, domain: int, count_as_pub: bool = True):
        """Update trackers after a model has been spawned."""
        if count_as_pub:
            self.publications += 1
            self.domain_pubs[domain] += 1
        new_model = self.model.scientific_models[-1]
        self.reputation += new_model.truthfulness * new_model.salience

    def _explore(self, domain: int):
        """Publish a new model in an underexplored domain."""
        self.model.spawn_model(
            origin_lab_id=self.lab_id,
            domain=domain,
            researcher_skills=self.domain_skills,
        )
        self.current_domain = domain
        self._record_spawn(domain)

    def _debunk(self, target: ScientificModel):
        """
        Attempt to disprove a published model.

        Triggered as a byproduct of failed high-proficiency exploitation — an
        expert who cannot replicate a model is well-placed to challenge it.

        Success: truthfulness drops (×0.75), salience is dented, researcher
                 earns a publication and reputation proportional to how truthful
                 the model was — the bigger the scalp, the larger the reward.
        Failure: minor assimilation from engaging with the model's complexity.
        """
        self.current_domain = target.domain
        proficiency  = self.domain_skills[target.domain]
        success_prob = proficiency * (1.0 - target.truthfulness)
        if self.random.random() < success_prob:
            reward = target.truthfulness * target.salience
            self.reputation                 += reward
            self.publications               += 1
            self.domain_pubs[target.domain] += 1
            target.truthfulness = max(0.01, target.truthfulness * 0.75)
            target.salience     = max(0.0,  target.salience - 0.15)
            self._assimilate(target, self.skill_gain_attempt * 0.5)
        else:
            self._assimilate(target, self.skill_gain_attempt * 0.2)
        self.debunk_steps += 1

    def _assimilate(self, m: ScientificModel, rate: float):
        """
        Nudge skill vector toward model complexity — focused on the model's
        primary domain, with only weak spillover to secondary domains.

        Domain focus weights:
          primary domain (m.domain) → full rate
          all other domains         → SECONDARY_LEARN_FACTOR × rate

        This prevents the runaway convergence where working on any model
        quietly raises every skill. Specialists keep their spike; generalists
        broaden only through the models they actively choose to work on.
        Skills only move upward.
        """
        skill = np.array(self.domain_skills)
        comp  = np.array(m.complexity)
        gap   = comp - skill
        focus = np.full(len(skill), self.SECONDARY_LEARN_FACTOR)
        focus[m.domain] = 1.0
        gain  = rate * skill * (1.0 - skill) * 4.0 * np.clip(gap, 0.0, 1.0) * focus
        gain  = np.where(gap > 0.0, gain, 0.0)
        self.domain_skills = np.clip(skill + gain, 0.01, 0.95).tolist()

    def _profile_alignment(self, m: ScientificModel) -> float:
        """
        Pure profile alignment: max(0, COR(skill_vector, complexity_vector)).
        Used to scale learning rates — mismatched profiles learn slower.
        Floor at 0.2 so learning never fully stalls.
        Reuses _pearson_cor to avoid a second np.corrcoef call.
        """
        return max(max(0.0, self._pearson_cor(m)), 0.2)

    def _train(self, m: ScientificModel):
        """
        Spend this step assimilating toward the model's full complexity profile.
        Rate is scaled by profile alignment — mismatched profiles train slower.
        """
        effective_rate = self.skill_gain_train * self._profile_alignment(m)
        self._assimilate(m, effective_rate)
        self.training_steps += 1
        self.current_domain = m.domain

    def _attempt(self, target: ScientificModel):
        """
        Try to implement a model.
        Success: publish, cite, assimilate fully, possibly spawn a follow-up.
        Failure: partial assimilation scaled by profile alignment.
        """
        self.current_domain = target.domain
        if self.random.random() < self.success_probability(target):
            self.publications += 1
            self.domain_pubs[target.domain] += 1
            self.reputation += target.truthfulness * target.salience
            target.cite()
            self._assimilate(target, self.skill_gain_attempt)

            # opportunistic follow-up spawn on success — probability decreases
            # as the domain fills up (log dampening). Uses cache — no list scan.
            n_domain   = self.model._cached_domain_counts.get(target.domain, 0)
            spawn_prob = 0.15 / (1.0 + np.log1p(n_domain))
            if self.random.random() < spawn_prob:
                self.model.spawn_model(
                    origin_lab_id=self.lab_id,
                    domain=target.domain,
                    researcher_skills=self.domain_skills,
                    initial_salience=0.7,
                )
                self._record_spawn(target.domain, count_as_pub=False)
        else:
            # failed attempt — partial assimilation, scaled by how foreign the work is
            effective_rate = self.skill_gain_attempt * 0.3 * self._profile_alignment(target)
            self._assimilate(target, effective_rate)
            # skilled researchers who fail replication may identify flaws
            proficiency = self.domain_skills[target.domain]
            if proficiency > 0.65 and self.random.random() < proficiency * 0.15:
                self._debunk(target)

    # --- probabilistic decision ---

    def _choose_action(self, exploit: list, invest: list, best_domain: int):
        """
        Assemble top candidates by expected value, then choose probabilistically.

          weight(model)   = match + salience + truthfulness
          weight(explore) = skill in that domain (no model to inspect)

        Top 3 models by value + explore option — capped to keep explore competitive.
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
                w = self.domain_skills[target]
            else:
                w = (self._similarity(target)
                     + target.salience
                     + target.truthfulness)
            weights.append(max(w, 1e-6))

        probs = [w / sum(weights) for w in weights]
        idx   = self.random.choices(range(len(top)), weights=probs, k=1)[0]
        return top[idx]

    # --- step ---

    def step(self):
        if self.current_target is not None:
            self.current_domain  = self.current_target.domain
            self.work_progress  += 1
            self.exploit_steps  += 1
            if self.work_progress >= self._work_required(self.current_target):
                self._attempt(self.current_target)
                self.current_target = None
                self.work_progress  = 0
            return

        exploit, invest = self._split_candidates()

        n_domains    = len(self.domain_skills)
        explore_vals = [max(self._explore_value(d), 1e-6) for d in range(n_domains)]
        best_domain  = self.random.choices(range(n_domains), weights=explore_vals, k=1)[0]

        action, target, _ = self._choose_action(exploit, invest, best_domain)

        if action == 'explore':
            self._explore(target)
            self.explore_steps += 1
        elif action == 'invest':
            self._train(target)
            # training_steps already incremented inside _train
        else:  # exploit — start multi-step commitment
            self.current_target = target
            self.current_domain = target.domain
            self.work_progress  = 1
            self.exploit_steps  += 1
            if self.work_progress >= self._work_required(target):
                self._attempt(self.current_target)
                self.current_target = None
                self.work_progress  = 0

        # Passive reputation decay — old contributions count for less over time
        self.reputation = max(0.0, self.reputation * 0.998)

        # Fingerprint drift — non-peak skills are attracted toward the lab baseline.
        # Skills above it decay down (0.4%/step); skills below it are gently pulled up
        # (0.2%/step). Peak domain is exempt so specialists can grow their spike freely.
        fp      = np.array(self.lab_fingerprint)
        skills  = np.array(self.domain_skills)
        peak    = int(np.argmax(fp))
        excess  = np.maximum(0.0, skills - fp)   # above baseline → decay
        deficit = np.maximum(0.0, fp - skills)   # below baseline → pull up
        excess[peak]  = 0.0
        deficit[peak] = 0.0
        self.domain_skills = np.clip(
            skills - 0.004 * excess + 0.002 * deficit, 0.01, 0.95
        ).tolist()
