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

    The researcher belongs to a lab for their entire career — they do not
    switch labs. Diversity in skill profiles emerges organically through
    the models they choose to work on.
    """

    # Fraction of the full assimilation rate applied to domains
    # other than the model's primary domain.
    SECONDARY_LEARN_FACTOR = 0.12

    def __init__(self, model, lab_id: int, domain_skills: list[float],
                 lab_fingerprint:    list[float] | None = None,
                 train_threshold:    float = 0.30,
                 skill_gain_attempt: float = 0.06,
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
        self.publications              = 0
        self.reputation                = 0.0
        self.reputation_lost_to_debunk = 0.0   # cumulative reputation lost when own models debunked
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

        # per-step Pearson cache — refreshed at start of step()
        _sa = np.array(domain_skills, dtype=float)
        self._sc_cache   = _sa - _sa.mean()
        self._ssq_cache  = float(np.dot(self._sc_cache, self._sc_cache))

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
        Pearson correlation between skill vector and model complexity.
        Uses per-step cached skill stats and pre-computed model stats to avoid
        redundant array allocations across the many calls within one step.
        Returns 0.0 when either vector has zero variance (uniform profiles).
        """
        denom = np.sqrt(self._ssq_cache * m._comp_ssq)
        return float(np.dot(self._sc_cache, m._comp_centered) / denom) if denom > 1e-10 else 0.0

    def _similarity(self, m: ScientificModel) -> float:
        """proficiency × max(0, COR) — how well the researcher fits this model."""
        cor = max(0.0, self._pearson_cor(m))
        return max(1e-6, self.domain_skills[m.domain] * cor)

    def success_probability(self, m: ScientificModel) -> float:
        """
        P = (sim + avg × (1 − sim)) × (actual / reported)

        Profile alignment (sim) drives success when the researcher's skill
        distribution matches the model's complexity profile. Average competence
        (avg) raises the floor for generalists. The truth ratio (actual /
        reported) discounts success by the publication bias gap — a model
        inflated 20% above its actual quality is ~20% more likely to fail
        independent replication, regardless of researcher skill.
        """
        sim = self._similarity(m)
        avg = float(np.mean(self.domain_skills))
        base = sim + avg * (1.0 - sim)
        truth_ratio = m.actual_truthfulness / max(m.reported_truthfulness, 1e-8)
        return float(np.clip(base * truth_ratio, 0.0, 1.0))

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
          = (success_prob × reported_truthfulness × salience) / (work_required × (1 + own_pubs))
        Agents perceive reported_truthfulness when evaluating potential payoff.
        """
        return (self.success_probability(m) * m.reported_truthfulness * m.salience
                / (self._work_required(m) * (1.0 + self.domain_pubs[m.domain])))

    def _invest_value(self, m: ScientificModel) -> float:
        """
        Expected marginal gain from training toward this model, discounted by
        the time cost to make it exploitable.

          steps_needed = gap_above_threshold / gain_per_step
          value = sigmoid_gain × truthfulness / (1 + steps_needed)

        A model just above the threshold costs ~1 extra step and is barely
        discounted. A model 0.30 above the threshold might cost 10+ steps
        and is heavily discounted — making nearby exploit candidates more
        attractive by comparison.
        """
        current        = self.domain_skills[m.domain]
        gain_per_step  = max(self._sigmoid_gain(current, self.skill_gain_train), 1e-8)
        gap_above      = max(0.0, m.complexity[m.domain] - current - self.train_threshold)
        steps_needed   = gap_above / gain_per_step
        return self._sigmoid_gain(current, self.skill_gain_train) * m.reported_truthfulness / (1.0 + steps_needed)

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
        # Reputation from publishing reflects perceived (reported) quality
        self.reputation += new_model.reported_truthfulness * new_model.salience

    def _explore(self, domain: int):
        """
        Publish a new model in an underexplored domain.

        Breakthrough mechanic: with a small skill-dependent probability, the
        published model is a paradigm-shifting breakthrough.  The probability
        scales with skill² so only genuinely expert researchers produce them
        (≈3% at skill=0.55, ≈8% at skill=0.90, <1% at skill=0.25).

        A breakthrough also triggers a salience shock to the domain: all
        existing models in that domain lose 65% of their current salience as
        the field's attention pivots to the new result.  This creates era
        dynamics — trend cycles where a dominant paradigm is periodically
        displaced rather than accumulating indefinitely.
        """
        skill          = self.domain_skills[domain]
        is_breakthrough = self.random.random() < skill ** 2 * 0.10

        self.model.spawn_model(
            origin_lab_id=self.lab_id,
            domain=domain,
            researcher_skills=self.domain_skills,
            author_agent_id=self.unique_id,
            breakthrough=is_breakthrough,
        )
        self.current_domain = domain

        if is_breakthrough:
            # Salience shock: incumbent models in this domain fade as the
            # community pivots toward the new paradigm.
            new_uid = self.model.scientific_models[-1].uid
            for m in self.model.scientific_models:
                if m.domain == domain and m.uid != new_uid:
                    m.salience = max(0.0, m.salience * 0.35)

        self._record_spawn(domain)

    def _debunk(self, target: ScientificModel):
        """
        Attempt to disprove a published model.

        Triggered as a byproduct of failed moderate-to-high-proficiency
        exploitation. On success, the target model loses truthfulness and
        salience, and a cascade propagates partial damage to any models that
        were built directly on the debunked model (parent_uid == target.uid).
        This simulates paradigm disruption: when a foundational result is
        overturned, derivative work is also undermined.
        """
        self.current_domain = target.domain
        proficiency  = self.domain_skills[target.domain]
        # Debunk success depends on how wrong the model actually is
        success_prob = proficiency * (1.0 - target.actual_truthfulness)
        if self.random.random() < success_prob:
            # Reward reflects actual quality exposed — discovering a real flaw
            reward = target.actual_truthfulness * target.salience
            self.reputation                 += reward
            self.publications               += 1
            self.domain_pubs[target.domain] += 1
            target.actual_truthfulness = max(0.01, target.actual_truthfulness * 0.75)
            target.salience            = max(0.0,  target.salience - 0.15)
            self._assimilate(target, self.skill_gain_attempt * 0.5)

            # penalise original author — they lose half of what the debunker gains
            if target.author_agent_id is not None:
                author = next(
                    (a for a in self.model.agents
                     if a.unique_id == target.author_agent_id), None
                )
                if author is not None:
                    penalty = reward * 0.5
                    author.reputation              = max(0.0, author.reputation - penalty)
                    author.reputation_lost_to_debunk += penalty

            # cascade: derivative models lose partial truthfulness and salience
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
        self.model.replication_attempts += 1
        self.model.domain_replication_attempts[target.domain] += 1
        if self.random.random() < self.success_probability(target):
            self.publications += 1
            self.domain_pubs[target.domain] += 1
            # Reputation reflects reported quality — what the community values
            self.reputation += target.reported_truthfulness * target.salience
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
                    parent_uid=target.uid,
                    author_agent_id=self.unique_id,
                )
                self._record_spawn(target.domain, count_as_pub=False)
        else:
            self.model.replication_failures += 1
            self.model.domain_replication_failures[target.domain] += 1
            # failed attempt — partial assimilation, scaled by how foreign the work is
            effective_rate = self.skill_gain_attempt * 0.3 * self._profile_alignment(target)
            self._assimilate(target, effective_rate)

            proficiency = self.domain_skills[target.domain]

            # Expert truth correction: sufficiently skilled researchers who fail
            # replication have enough domain knowledge to recognise that the
            # failure reflects the model's flaw, not their own incompetence.
            # Each expert failure erodes actual_truthfulness by a small amount
            # proportional to expertise and the publication-bias inflation gap.
            # This creates a slow, continuous truth-revelation process distinct
            # from the large discrete correction of a formal debunk.
            #
            # The correction is larger when the gap between reported and actual
            # is wider — highly inflated models are exposed faster.
            if proficiency > 0.50:
                gap_inflation = target.reported_truthfulness - target.actual_truthfulness
                correction    = proficiency * 0.015 * (1.0 + gap_inflation)
                target.actual_truthfulness = max(0.01,
                                                 target.actual_truthfulness - correction)

            # Discrete debunk: high-proficiency researchers may mount a formal
            # challenge (larger, rarer correction than the continuous erosion above)
            if proficiency > 0.35 and self.random.random() < proficiency * 0.40:
                self._debunk(target)

    # --- probabilistic decision ---

    def _choose_action(self, exploit: list, invest: list, best_domain: int):
        """
        Assemble top candidates by expected value, then choose probabilistically.

          weight(model)   = (match + salience + truthfulness) × (skill / mean_skill)
          weight(explore) = 1.5 × skill² / mean_skill

        The skill-relative bias (skill / mean_skill) is a parameter-free home-domain
        pull derived directly from the skill vector. Specialists (~2× on their peak)
        are pulled strongly toward familiar work; generalists (~1× everywhere) have
        no directional bias.
        """
        scored = []
        for m in exploit:
            scored.append(('exploit', m, self._exploit_value(m)))
        for m in invest:
            scored.append(('invest',  m, self._invest_value(m)))

        top = sorted(scored, key=lambda x: x[2], reverse=True)[:2]
        top.append(('explore', best_domain, self._explore_value(best_domain)))

        mean_sk = float(np.mean(self.domain_skills)) + 1e-8
        weights = []
        for action, target, _ in top:
            domain     = target if action == 'explore' else target.domain
            skill_bias = (self.domain_skills[domain] / mean_sk) ** 0.5
            if action == 'explore':
                w = 1.5 * self.domain_skills[target] * skill_bias
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
            skills - 0.002 * excess + 0.001 * deficit, 0.01, 0.95
        ).tolist()
