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

        # Type B: per-domain learned utility for softmax action selection.
        # Updated by Rescorla–Wagner after each domain-locating action
        # (exploit, train, explore, debunk). Initialised at 0.5 so all
        # domains start equally attractive — concentration must then emerge
        # from the success→reward→utility loop rather than from any
        # hardwired skill preference.
        self.domain_utility = np.full(n_domains, 0.5)

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

    def _update_utility(self, domain: int, reward: float) -> None:
        """
        Type B Rescorla–Wagner update on the researcher's per-domain
        learned utility:

            U_d ← U_d + α (R - U_d)

        Called after every domain-locating action with the observed
        reward — for exploit success the reward is reputation gained
        (reported_truthfulness × salience), for failure it is 0, for
        training a small positive value, for explore the spawned
        model's actual truthfulness. No-op when enable_type_b is False
        so Type A behaviour is preserved bit-identical.
        """
        if not self.model.enable_type_b:
            return
        alpha = self.model.alpha_rl
        self.domain_utility[domain] += alpha * (reward - self.domain_utility[domain])

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

        # Matthew effect (gated by enable_realism)
        if self.model.enable_realism:
            matthew_factor = 1.0 + 0.30 * np.tanh(self.reputation * 0.05)
        else:
            matthew_factor = 1.0
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

        # Best-model variant: a breakthrough sometimes repositions the new
        # model across the landscape to a different attractor (plateau or
        # peak) rather than placing it near the researcher's current position.
        # Operationalises paradigm-shifting results that establish a new
        # research region rather than refining the current one.
        if is_breakthrough and self.model.enable_landscape:
            if self.random.random() < 0.5:
                landscape = self.model.landscapes[domain]
                position  = landscape.breakthrough_across_ridge(
                    current_pos[0], current_pos[1], self.model.rng
                )

        # Competitive pressure → misconduct probability (gated by enable_realism)
        if self.model.enable_realism:
            median_rep = self.model._median_rep
            pressure   = float(np.clip(1.0 - self.reputation / median_rep, 0.0, 1.0)) \
                         if median_rep > 0 else 0.0
            misconduct_prob = self.model.misconduct_base_rate * (1.0 + pressure * 2.0)
            is_misconduct   = (not is_breakthrough) and (self.random.random() < misconduct_prob)
        else:
            is_misconduct   = False

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

        # Type B: explore yields a reward proportional to the new model's
        # actual quality × initial salience. High-skill domains will tend to
        # produce higher-truthfulness models (gain follows the domain cap and
        # researcher skill), so the utility update naturally rewards
        # productive domains without referencing skill directly.
        new_model = self.model.scientific_models[-1]
        self._update_utility(domain, new_model.actual_truthfulness * new_model.salience)

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
            self._update_utility(target.domain, reward)
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
            self._update_utility(target.domain, 0.0)
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
        # Type B: training has no immediate reputation reward, but the agent
        # is making forward progress toward a future exploit. Treat it as a
        # small positive signal scaled by profile alignment, so domains where
        # training is productive (well-matched models exist) are not
        # mistakenly down-weighted by the RW rule.
        self._update_utility(m.domain, 0.1 * self._profile_alignment(m))

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
            reward = target.reported_truthfulness * target.salience
            self.reputation += reward
            self._update_utility(target.domain, reward)
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

            # Follow-up spawn — step toward a stable, high-truth plateau
            # (only if landscape on). Replaces step_toward_valley under the
            # best-model landscape semantics: each cumulative publication in
            # a research programme drifts toward higher truth and lower
            # gradient (an established paradigm region).
            n_domain   = self.model._cached_domain_counts.get(target.domain, 0)
            spawn_prob = 0.15 / (1.0 + np.log1p(n_domain))
            if self.random.random() < spawn_prob:
                if self.model.enable_landscape:
                    landscape = self.model.landscapes[target.domain]
                    next_pos  = landscape.step_toward_stable(*target.position, step_size=0.05)
                else:
                    # neutralised: place follow-up at the parent position (no drift toward stability)
                    next_pos  = target.position
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
            self._update_utility(target.domain, 0.0)
            effective_rate = self.skill_gain_attempt * 0.3 * self._profile_alignment(target)
            self._assimilate(target, effective_rate)

            proficiency = self.domain_skills[target.domain]

            if proficiency > 0.50:
                gap_inflation = target.reported_truthfulness - target.actual_truthfulness
                correction    = proficiency * 0.015 * (1.0 + gap_inflation)
                target.actual_truthfulness = max(0.01,
                                                 target.actual_truthfulness - correction)

            # Best-model variant: debunk trigger threshold is gradient-aware
            # rather than a fixed magic 0.35. Stable plateau models require
            # high proficiency to challenge (rigour bar for established
            # paradigms); unstable peak edges can be challenged at lower
            # proficiency (cutting-edge results are structurally easier to
            # overturn). The threshold scales linearly with the target
            # model's landscape stability:
            #   threshold = 0.20 + 0.40 × stability(x, y)
            # Stable target (s=1): threshold = 0.60 (only experts can attempt)
            # Unstable target (s=0): threshold = 0.20 (low bar)
            # Mean stability ≈ 0.57 → mean threshold ≈ 0.43 (close to the
            # old fixed 0.35 in aggregate, but now sensitive to where on
            # the landscape the target model sits).
            debunk_threshold = 0.20 + 0.40 * target.landscape_stability
            if proficiency > debunk_threshold and self.random.random() < proficiency * 0.40:
                self._debunk(target)

    # --- probabilistic decision ---

    def _choose_action(self, exploit: list, invest: list, best_domain: int):
        """
        Choose among the top two exploit/invest candidates plus one explore option.

        Two architectures are supported:

        Type A (default, enable_type_b=False): the classical model. Each
        candidate's weight is multiplied by the parameter-free skill-bias term

            skill_bias_d = (skill_d / mean_skill) ** 0.5         (Equation 6)

        which directly biases selection toward the researcher's higher-skill
        domains. This is the architecturally directional rule whose
        contribution to H2's concentration result is the main thing the
        Type B reformulation is designed to test.

        Type B (enable_type_b=True): the best-model variant. The skill-bias
        term is removed entirely. Selection weight is instead modulated by

            softmax_d = exp(β · (U_d − mean U))

        where U_d is the agent's Rescorla–Wagner-learned utility for domain
        d, updated after every domain-locating action with the observed
        reward. Skill no longer enters action selection directly; it enters
        only through (a) success_probability, which determines whether an
        exploit attempt succeeds and therefore the realised reward, and
        (b) the explore_value scoring. Domain concentration in Type B must
        therefore emerge through the success → reward → utility feedback
        loop rather than from a hardwired skill preference, providing an
        empirical test of how much of H2 is architecturally guaranteed.
        Similarity is also removed from the exploit weight in Type B so
        that no direct skill term enters the selection step.
        """
        scored = []
        for m in exploit:
            scored.append(('exploit', m, self._exploit_value(m)))
        for m in invest:
            scored.append(('invest',  m, self._invest_value(m)))

        top = sorted(scored, key=lambda x: x[2], reverse=True)[:2]
        top.append(('explore', best_domain, self._explore_value(best_domain)))

        # Career stages (gated by enable_realism)
        stage_boost = (max(0.0, 1.0 - self.career_age / 150.0)
                       if self.model.enable_realism else 0.0)

        if self.model.enable_type_b:
            # --- Type B: softmax over learned domain utility ---
            # Explore-weight base: 0.6 matches the typical magnitude of the
            # Type A explore weight (1.5 × skill_d × skill_bias_d ≈ 0.6 with
            # typical skill ~0.4 and skill_bias ~1.0), so the explore vs
            # exploit balance is calibrated, not over-inflated. Without this
            # rescaling, Type B agents over-publish because no skill term
            # dampens the base 1.5 constant — see smoke-test note in commit
            # message.
            beta    = self.model.beta_rl
            u_mean  = float(self.domain_utility.mean())
            weights = []
            for action, target, _ in top:
                domain      = target if action == 'explore' else target.domain
                util_weight = float(np.exp(beta * (self.domain_utility[domain] - u_mean)))
                if action == 'explore':
                    w = 0.6 * (1.0 + stage_boost) * util_weight
                else:
                    w = (target.salience + target.reported_truthfulness) * util_weight
                weights.append(max(w, 1e-6))
        else:
            # --- Type A: skill-bias action selection (Equation 6) ---
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
