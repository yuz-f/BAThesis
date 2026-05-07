from __future__ import annotations

import numpy as np
import mesa
from collections import defaultdict
from scientific_model import ScientificModel


class Researcher(mesa.Agent):
    """
    An individual researcher agent with a personal skill vector.

    Epistemic landscape integration (v3)
    ----------------------------------------
    Each researcher tracks a 2D theory-space position per domain:
      self.domain_positions[d]  →  np.ndarray([x, y]) ∈ [0,1]²

    This position represents where the researcher is currently "working"
    on the domain's epistemic landscape.  It evolves as follows:

      - On successful exploitation: position assimilates 10 % toward the
        target model's landscape position (slow convergence toward where
        established work lives).
      - On exploration: the new model is placed near the researcher's
        current position (+noise that shrinks with experience), so experts
        cluster tightly while novices scatter broadly.
      - Follow-up models (after a successful exploit): placed one gradient-
        descent step closer to the nearest valley, operationalising the
        tendency of mature research programmes to converge on stable anchors.

    Stability directly modifies the three key action outcomes:
      1. success_probability: valley models replicate more easily
         (× 0.75 + 0.25 × stability)
      2. debunk success: peak models are more vulnerable
         (× 1 + 0.5 × instability)
      3. exploit attractiveness: stable models look better to work on
         (stability_attract = 1 + 0.30 × stability)

    Other additions (v2, unchanged)
    ----------------------------------------
    - Career stages, social learning, Matthew effect, competitive pressure
      bias, misconduct pathway — see docstrings in each method.
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

        # career stage
        self.career_age: int = 0

        # Theory-space positions on each domain's epistemic landscape.
        # Initialised uniformly — researchers start at unknown positions
        # and converge toward valleys as they accumulate experience.
        n_domains = len(domain_skills)
        self.domain_positions: dict[int, np.ndarray] = {
            d: self.model.rng.uniform(0.05, 0.95, 2)
            for d in range(n_domains)
        }

        # work-in-progress state
        self.current_target: ScientificModel | None = None
        self.work_progress:  int = 0

        # per-step Pearson cache
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
        """
        P(success) = (sim + avg × (1−sim)) × truth_ratio × stability_factor

        Landscape extension:
          stability_factor = 0.75 + 0.25 × m.landscape_stability
          Valley models (stability→1): factor = 1.00 — no penalty.
          Peak models   (stability→0): factor = 0.75 — 25 % harder to replicate.

        The mechanism operationalises the intuition that a model in unstable
        theory-space is harder to reproduce: the conditions required to get it
        to work are narrow, sensitive, and poorly understood.
        """
        sim = self._similarity(m)
        avg = float(np.mean(self.domain_skills))
        base = sim + avg * (1.0 - sim)
        truth_ratio      = m.actual_truthfulness / max(m.reported_truthfulness, 1e-8)
        stability_factor = 0.75 + 0.25 * m.landscape_stability
        return float(np.clip(base * truth_ratio * stability_factor, 0.0, 1.0))

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

        Three multipliers beyond the base value:
          1. Social learning bonus  (v2) — labmate success signal
          2. Matthew effect         (v2) — rep amplifies salience visibility
          3. Stability attraction   (v3) — valley models are preferred
             stability_attract = 1 + 0.30 × landscape_stability
             Valley (1): × 1.30   Peak (0): × 1.00
        """
        # Social learning
        social_signal = (
            self.model.lab_domain_successes
            .get(self.lab_id, {})
            .get(m.domain, 0.0)
        )
        social_bonus = 1.0 + self.model.social_learn_strength * np.tanh(social_signal)

        # Matthew effect
        matthew_factor     = 1.0 + 0.30 * np.tanh(self.reputation * 0.05)
        effective_salience = m.salience * matthew_factor

        # Stability attraction: agents prefer work in stable theory-space
        stability_attract = 1.0 + 0.30 * m.landscape_stability

        base = (self.success_probability(m) * m.reported_truthfulness * effective_salience
                / (self._work_required(m) * (1.0 + self.domain_pubs[m.domain])))
        return base * social_bonus * stability_attract

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
        Publish a new Scientific Model.

        Landscape positioning (v3):
          The model is placed near the researcher's current theory-space
          position in this domain.  The placement noise shrinks with experience:
            noise_scale = max(0.04, 0.20 − 0.01 × own_pubs_in_domain)
          A researcher publishing their 16th paper in a domain scatters ±0.04
          around their current position; a newcomer scatters ±0.20.  This means
          experts converge on a theory-space cluster (which may lie in a valley
          if their programme has matured) while novices enter from random angles.

        Misconduct and breakthrough mechanics unchanged from v2.
        """
        skill           = self.domain_skills[domain]
        is_breakthrough = self.random.random() < skill ** 2 * 0.10

        # --- landscape position for the new model ---
        current_pos = self.domain_positions.get(
            domain,
            self.model.rng.uniform(0.05, 0.95, 2)
        )
        # novices scatter widely; experts place tightly around current position
        noise_scale = max(0.04, 0.20 - 0.01 * self.domain_pubs[domain])
        raw_pos = current_pos + self.model.rng.normal(0.0, noise_scale, 2)
        position = (
            float(np.clip(raw_pos[0], 0.01, 0.99)),
            float(np.clip(raw_pos[1], 0.01, 0.99)),
        )

        # Competitive pressure → misconduct probability
        median_rep = self.model._median_rep
        pressure   = float(np.clip(1.0 - self.reputation / median_rep, 0.0, 1.0)) \
                     if median_rep > 0 else 0.0
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
            position=position,
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

        Landscape extension (v3):
          Peak models are more vulnerable to debunking — they occupy unstable
          theory-space where the conditions for the result are narrow and poorly
          characterised.

            instability_boost = 1.0 + 0.5 × (1 − landscape_stability)
            Valley (stability=1): boost = 1.00 — standard difficulty
            Peak   (stability=0): boost = 1.50 — 50 % more debunkable
        """
        self.current_domain = target.domain
        proficiency       = self.domain_skills[target.domain]
        instability_boost = 1.0 + 0.5 * (1.0 - target.landscape_stability)
        success_prob      = proficiency * (1.0 - target.actual_truthfulness) * instability_boost
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

        Landscape extensions (v3):
          On success:
            - Researcher's theory position in the domain assimilates 10 %
              toward the model's landscape position (slow convergence).
            - Follow-up model (if spawned) is placed one gradient-descent step
              closer to the nearest valley: research programmes mature toward
              stable anchors.
          On failure:
            - No position update (failed replication gives no new theoretical
              footing).
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

            # Social learning signal
            self.model.record_domain_success(self.lab_id, target.domain)

            # Theory-position assimilation toward model's landscape position
            current_pos = self.domain_positions.get(
                target.domain, np.array([0.5, 0.5])
            )
            self.domain_positions[target.domain] = (
                current_pos + 0.10 * (np.array(target.position) - current_pos)
            )

            # Follow-up spawn — gradient-descent position toward valley
            n_domain   = self.model._cached_domain_counts.get(target.domain, 0)
            spawn_prob = 0.15 / (1.0 + np.log1p(n_domain))
            if self.random.random() < spawn_prob:
                landscape  = self.model.landscapes[target.domain]
                next_pos   = landscape.step_toward_valley(*target.position, step_size=0.05)
                self.model.spawn_model(
                    origin_lab_id=self.lab_id,
                    domain=target.domain,
                    researcher_skills=self.domain_skills,
                    initial_salience=0.7,
                    parent_uid=target.uid,
                    author_agent_id=self.unique_id,
                    author_reputation=self.reputation,
                    position=next_pos,
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
        Career stage boost on explore weight (v2, unchanged).
        Stability attraction is already baked into _exploit_value (v3).
        """
        scored = []
        for m in exploit:
            scored.append(('exploit', m, self._exploit_value(m)))
        for m in invest:
            scored.append(('invest',  m, self._invest_value(m)))

        top = sorted(scored, key=lambda x: x[2], reverse=True)[:2]
        top.append(('explore', best_domain, self._explore_value(best_domain)))

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

        self.reputation = max(0.0, self.reputation * 0.998)

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
