from __future__ import annotations

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

    Realism additions vs. v1:
      - Career stages: young researchers explore more; seniors exploit more
      - Social learning: lab-mates' recent domain successes boost action weights
      - Matthew effect: high-reputation researchers are drawn toward
        high-salience (prominent) models
      - Misconduct: under competitive pressure, researchers inflate reported
        quality beyond the standard publication-bias draw
      - Competitive pressure → publication bias: researchers below the
        median reputation inflate bias at spawn time (via world._median_rep)
    """

    SECONDARY_LEARN_FACTOR = 0.12

    def __init__(self, model, lab_id: int, domain_skills: list[float],
                 lab_fingerprint:    list[float] | None = None,
                 train_threshold:    float = 0.30,
                 skill_gain_attempt: float = 0.06,
                 skill_gain_train:   float = 0.08):
        super().__init__(model)

        self.lab_id         = lab_id
        self.domain_skills  = domain_skills
        self.lab_fingerprint = (list(lab_fingerprint)
                                if lab_fingerprint is not None
                                else list(domain_skills))
        self.train_threshold    = train_threshold
        self.skill_gain_attempt = skill_gain_attempt
        self.skill_gain_train   = skill_gain_train

        # outcome trackers
        self.publications              = 0
        self.reputation                = 0.0
        self.reputation_lost_to_debunk = 0.0
        self.training_steps  = 0
        self.exploit_steps   = 0
        self.explore_steps   = 0
        self.debunk_steps    = 0
        self.current_domain: int | None = None
        self.domain_pubs: defaultdict[int, int] = defaultdict(int)
        self.reputation_at_last_cull = 0.0

        # career stage — incremented every step; drives explore-bias decay
        self.career_age: int = 0

        # work-in-progress state
        self.current_target: ScientificModel | None = None
        self.work_progress:  int = 0

        # per-step Pearson cache — refreshed at start of step()
        _sa = np.array(domain_skills, dtype=float)
        self._sc_cache   = _sa - _sa.mean()
        self._ssq_cache  = float(np.dot(self._sc_cache, self._sc_cache))

    # --- derived properties ---

    @property
    def home_domain(self) -> int:
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
        base         = max(3, round(m.complexity[m.domain] * 8))
        saturation   = self.model._cached_saturations[m.domain]
        skill_factor = 0.5 + self.domain_skills[m.domain]
        return max(1, round(base * (1.0 + saturation * 0.5) / skill_factor))

    def _pearson_cor(self, m: ScientificModel) -> float:
        denom = np.sqrt(self._ssq_cache * m._comp_ssq)
        return float(np.dot(self._sc_cache, m._comp_centered) / denom) if denom > 1e-10 else 0.0

    def _similarity(self, m: ScientificModel) -> float:
        cor = max(0.0, self._pearson_cor(m))
        return max(1e-6, self.domain_skills[m.domain] * cor)

    def success_probability(self, m: ScientificModel) -> float:
        sim = self._similarity(m)
        avg = float(np.mean(self.domain_skills))
        base = sim + avg * (1.0 - sim)
        truth_ratio = m.actual_truthfulness / max(m.reported_truthfulness, 1e-8)
        return float(np.clip(base * truth_ratio, 0.0, 1.0))

    def _split_candidates(self):
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

        Three realism extensions over v1:
          - Social learning: domains where labmates recently succeeded get a
            bonus (Crane 1972 — invisible colleges pull focus)
          - Matthew effect: high-reputation researchers are amplified toward
            high-salience models (Merton 1968)
          - Both factors are bounded (tanh) so they nudge rather than dominate
        """
        # Social learning: labmates' recent success in this domain
        social_signal = (
            self.model.lab_domain_successes
            .get(self.lab_id, {})
            .get(m.domain, 0.0)
        )
        social_bonus = 1.0 + self.model.social_learn_strength * np.tanh(social_signal)

        # Matthew effect: prominent models are even more attractive for elite researchers
        matthew_factor = 1.0 + 0.30 * np.tanh(self.reputation * 0.05)
        effective_salience = m.salience * matthew_factor

        base = (self.success_probability(m) * m.reported_truthfulness * effective_salience
                / (self._work_required(m) * (1.0 + self.domain_pubs[m.domain])))
        return base * social_bonus

    def _invest_value(self, m: ScientificModel) -> float:
        current        = self.domain_skills[m.domain]
        gain_per_step  = max(self._sigmoid_gain(current, self.skill_gain_train), 1e-8)
        gap_above      = max(0.0, m.complexity[m.domain] - current - self.train_threshold)
        steps_needed   = gap_above / gain_per_step
        return self._sigmoid_gain(current, self.skill_gain_train) * m.reported_truthfulness / (1.0 + steps_needed)

    def _explore_value(self, domain: int) -> float:
        n_models   = self.model._cached_domain_counts.get(domain, 0)
        saturation = self.model._cached_saturations[domain]
        return (self.domain_skills[domain] * max(0.0, 1.0 - saturation)
                / (1.0 + np.log1p(n_models) + self.domain_pubs[domain]))

    # --- actions ---

    def _record_spawn(self, domain: int, count_as_pub: bool = True):
        if count_as_pub:
            self.publications += 1
            self.domain_pubs[domain] += 1
        new_model = self.model.scientific_models[-1]
        self.reputation += new_model.reported_truthfulness * new_model.salience

    def _explore(self, domain: int):
        """
        Publish a new Scientific Model in an underexplored domain.

        Misconduct pathway (new in v2):
          Under competitive pressure — when the researcher's reputation falls
          below the field median — there is an elevated probability of
          strategic quality inflation (reporting higher than actual).
          The base misconduct probability (world.misconduct_base_rate) scales
          up with pressure so that low-reputation researchers inflate more
          (Fang et al. 2012: misconduct accounts for 67% of retractions).

        Breakthrough mechanic (unchanged from v1):
          Skill²-scaled probability of a paradigm-shifting publication that
          displaces incumbent models via salience shock.
        """
        skill           = self.domain_skills[domain]
        is_breakthrough = self.random.random() < skill ** 2 * 0.10

        # Competitive pressure → misconduct probability
        median_rep = self.model._median_rep
        if median_rep > 0:
            pressure = float(np.clip(1.0 - self.reputation / median_rep, 0.0, 1.0))
        else:
            pressure = 0.0
        misconduct_prob = self.model.misconduct_base_rate * (1.0 + pressure * 2.0)
        is_misconduct   = (not is_breakthrough) and (self.random.random() < misconduct_prob)

        self.model.spawn_model(
            origin_lab_id=self.lab_id,
            domain=domain,
            researcher_skills=self.domain_skills,
            author_agent_id=self.unique_id,
            breakthrough=is_breakthrough,
            misconduct=is_misconduct,
            author_reputation=self.reputation,
        )
        self.current_domain = domain

        if is_breakthrough:
            new_uid = self.model.scientific_models[-1].uid
            for m in self.model.scientific_models:
                if m.domain == domain and m.uid != new_uid:
                    m.salience = max(0.0, m.salience * 0.35)

        self._record_spawn(domain)

    def _debunk(self, target: ScientificModel):
        """
        Attempt to disprove a published model.

        Triggered as a byproduct of failed moderate-to-high-proficiency
        exploitation.  On success: target loses truthfulness and salience,
        cascade propagates partial damage to derivative models, debunker
        earns a publication, original author is penalised.
        """
        self.current_domain = target.domain
        proficiency  = self.domain_skills[target.domain]
        success_prob = proficiency * (1.0 - target.actual_truthfulness)
        if self.random.random() < success_prob:
            reward = target.actual_truthfulness * target.salience
            self.reputation                 += reward
            self.publications               += 1
            self.domain_pubs[target.domain] += 1
            target.actual_truthfulness = max(0.01, target.actual_truthfulness * 0.75)
            target.salience            = max(0.0,  target.salience - 0.15)
            self._assimilate(target, self.skill_gain_attempt * 0.5)

            if target.author_agent_id is not None:
                author = next(
                    (a for a in self.model.agents
                     if a.unique_id == target.author_agent_id), None
                )
                if author is not None:
                    penalty = reward * 0.5
                    author.reputation              = max(0.0, author.reputation - penalty)
                    author.reputation_lost_to_debunk += penalty

            for m in self.model.scientific_models:
                if m.parent_uid == target.uid:
                    m.actual_truthfulness = max(0.01, m.actual_truthfulness * 0.88)
                    m.salience            = max(0.0,  m.salience - 0.08)
        else:
            self._assimilate(target, self.skill_gain_attempt * 0.2)
        self.debunk_steps += 1

    def _assimilate(self, m: ScientificModel, rate: float):
        """
        Nudge skill vector toward model complexity — focused on the model's
        primary domain, with only weak spillover to secondary domains.
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
        return max(max(0.0, self._pearson_cor(m)), 0.2)

    def _train(self, m: ScientificModel):
        effective_rate = self.skill_gain_train * self._profile_alignment(m)
        self._assimilate(m, effective_rate)
        self.training_steps += 1
        self.current_domain = m.domain

    def _attempt(self, target: ScientificModel):
        """
        Try to implement a model.
        Success: publish, cite, assimilate, record lab domain success,
                 possibly spawn a follow-up.
        Failure: partial assimilation, expert truth correction, possible debunk.
        """
        self.current_domain = target.domain
        self.model.replication_attempts += 1
        self.model.domain_replication_attempts[target.domain] += 1
        if self.random.random() < self.success_probability(target):
            self.publications += 1
            self.domain_pubs[target.domain] += 1
            self.reputation += target.reported_truthfulness * target.salience
            target.cite()
            self._assimilate(target, self.skill_gain_attempt)

            # Social learning: notify world that this lab succeeded in this domain
            self.model.record_domain_success(self.lab_id, target.domain)

            n_domain   = self.model._cached_domain_counts.get(target.domain, 0)
            spawn_prob = 0.15 / (1.0 + np.log1p(n_domain))
            if self.random.random() < spawn_prob:
                self.model.spawn_model(
                    origin_lab_id=self.lab_id,
                    domain=target.domain,
                    researcher_skills=self.domain_skills,
                    initial_salience=0.7,
                    parent_uid=target.uid,
                    author_agent_id=self.unique_id,
                    author_reputation=self.reputation,
                )
                self._record_spawn(target.domain, count_as_pub=False)
        else:
            self.model.replication_failures += 1
            self.model.domain_replication_failures[target.domain] += 1
            effective_rate = self.skill_gain_attempt * 0.3 * self._profile_alignment(target)
            self._assimilate(target, effective_rate)

            proficiency = self.domain_skills[target.domain]

            if proficiency > 0.50:
                gap_inflation = target.reported_truthfulness - target.actual_truthfulness
                correction    = proficiency * 0.015 * (1.0 + gap_inflation)
                target.actual_truthfulness = max(0.01,
                                                 target.actual_truthfulness - correction)

            if proficiency > 0.35 and self.random.random() < proficiency * 0.40:
                self._debunk(target)

    # --- probabilistic decision ---

    def _choose_action(self, exploit: list, invest: list, best_domain: int):
        """
        Assemble top candidates by expected value, then choose probabilistically.

        Career stage extension (new in v2):
          Young researchers (low career_age) receive a bonus weight on the
          explore option that decays linearly to zero at career_age = 150.
          This reflects empirical career-trajectory data showing early-career
          researchers take larger exploratory risks (Petersen et al. 2012).

          weight(explore) = 1.5 × (1 + stage_boost) × skill² / mean_skill
          stage_boost = max(0, 1 − career_age / 150)

        Skill-relative bias (unchanged from v1):
          weight(model) = (match + salience + truthfulness) × (skill / mean_skill)^0.5
        """
        scored = []
        for m in exploit:
            scored.append(('exploit', m, self._exploit_value(m)))
        for m in invest:
            scored.append(('invest',  m, self._invest_value(m)))

        top = sorted(scored, key=lambda x: x[2], reverse=True)[:2]
        top.append(('explore', best_domain, self._explore_value(best_domain)))

        # Career stage: fade explore bonus from 1.0 (at birth) to 0.0 (step 150+)
        stage_boost = max(0.0, 1.0 - self.career_age / 150.0)

        mean_sk = float(np.mean(self.domain_skills)) + 1e-8
        weights = []
        for action, target, _ in top:
            domain     = target if action == 'explore' else target.domain
            skill_bias = (self.domain_skills[domain] / mean_sk) ** 0.5
            if action == 'explore':
                w = 1.5 * (1.0 + stage_boost) * self.domain_skills[target] * skill_bias
            else:
                w = (self._similarity(target)
                     + target.salience
                     + target.reported_truthfulness) * skill_bias
            weights.append(max(w, 1e-6))

        probs = [w / sum(weights) for w in weights]
        idx   = self.random.choices(range(len(top)), weights=probs, k=1)[0]
        return top[idx]

    # --- step ---

    def step(self):
        # refresh Pearson cache — skills may have changed since last step
        _sa = np.array(self.domain_skills, dtype=float)
        self._sc_cache  = _sa - _sa.mean()
        self._ssq_cache = float(np.dot(self._sc_cache, self._sc_cache))

        self.career_age += 1

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
        else:
            self.current_target = target
            self.current_domain = target.domain
            self.work_progress  = 1
            self.exploit_steps  += 1
            if self.work_progress >= self._work_required(target):
                self._attempt(self.current_target)
                self.current_target = None
                self.work_progress  = 0

        # Passive reputation decay
        self.reputation = max(0.0, self.reputation * 0.998)

        # Fingerprint drift
        fp      = np.array(self.lab_fingerprint)
        skills  = np.array(self.domain_skills)
        peak    = int(np.argmax(fp))
        excess  = np.maximum(0.0, skills - fp)
        deficit = np.maximum(0.0, fp - skills)
        excess[peak]  = 0.0
        deficit[peak] = 0.0
        self.domain_skills = np.clip(
            skills - 0.002 * excess + 0.001 * deficit, 0.01, 0.95
        ).tolist()
