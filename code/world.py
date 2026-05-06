from __future__ import annotations

import numpy as np
import mesa
from scientific_model import ScientificModel
from lab import Lab
from researcher import Researcher


class ScienceWorld(mesa.Model):
    """
    The simulation environment.

    Structure:
      - 10 Labs:  fixed fingerprint institutions.
      - 50 Researchers (5 per lab): Mesa agents that learn, publish, and get
                  selected.

    Realism additions vs. v1
    ------------------------
    social_learn_strength
        Weight of the lab social-signal term in exploit action values.
        When labmates recently succeeded in domain d, all researchers in that
        lab receive a bonus for choosing work in d.  The signal decays
        exponentially (×0.90 per step) so only recent activity matters.
        (Crane 1972 — invisible colleges concentrate domain focus.)

    misconduct_base_rate
        Base probability per publication that a researcher engages in
        strategic quality inflation (bias drawn from a higher distribution).
        Scales up multiplicatively with competitive pressure: researchers
        whose reputation falls below the field median inflate more.
        (Fang et al. 2012 — misconduct accounts for 67 % of retractions.)

    Competitive pressure → publication bias
        At spawn time, if the author's reputation is below the current field
        median, the bias_inflation draw is shifted upward by up to +0.08
        (proportional to how far below median they are).
        (Smaldino & McElreath 2016 — publication pressure drives inflation.)

    Matthew effect
        Implemented on the researcher side (_exploit_value): high-reputation
        researchers receive a salience amplification factor when evaluating
        models, pulling them toward prominent work.
        (Merton 1968.)

    Career stages
        Implemented on the researcher side (_choose_action): explore weight
        is boosted for young researchers and decays linearly over 150 steps.
        (Petersen et al. 2012 on early-career risk-taking.)
    """

    def __init__(self,
                 n_labs:               int   = 10,
                 researchers_per_lab:  int   = 5,
                 n_domains:            int   = 10,
                 peak_skill_mean:      float = 0.55,
                 peak_skill_std:       float = 0.07,
                 other_skill_mean:     float = 0.25,
                 other_skill_std:      float = 0.06,
                 selection_interval:   int   = 40,
                 cull_fraction:        float = 0.25,
                 mutation_std:         float = 0.04,
                 cap_growth_rate:      float = 0.005,
                 lab_close_threshold:  float = 0.5,
                 train_threshold:      float = 0.30,
                 skill_gain_attempt:   float = 0.06,
                 skill_gain_train:     float = 0.08,
                 social_learn_strength: float = 0.30,
                 misconduct_base_rate:  float = 0.05,
                 rng: int | None             = None):
        super().__init__(rng=rng)
        self.n_domains           = n_domains
        self.n_labs              = n_labs
        self.selection_interval  = selection_interval
        self.cull_fraction       = cull_fraction
        self.mutation_std        = mutation_std
        self.cap_growth_rate     = cap_growth_rate
        self.lab_close_threshold = lab_close_threshold
        self._train_threshold    = train_threshold
        self._skill_gain_attempt = skill_gain_attempt
        self._skill_gain_train   = skill_gain_train
        self._peak_skill_mean    = peak_skill_mean
        self._peak_skill_std     = peak_skill_std
        self._other_skill_mean   = other_skill_mean
        self._other_skill_std    = other_skill_std

        # realism parameters
        self.social_learn_strength = social_learn_strength
        self.misconduct_base_rate  = misconduct_base_rate

        self.lab_turnover_events: list[tuple] = []
        self.scientific_models: list[ScientificModel] = []
        self.replication_attempts        = 0
        self.replication_failures        = 0
        self.domain_replication_attempts = {d: 0 for d in range(n_domains)}
        self.domain_replication_failures = {d: 0 for d in range(n_domains)}
        self._model_counter = 0
        self._step_count    = 0

        # step-level caches
        self._cached_saturations:   list[float] = [0.0] * n_domains
        self._cached_domain_counts: dict[int, int] = {d: 0 for d in range(n_domains)}

        # competitive pressure: current field median reputation
        # updated at the start of every step, before agents act
        self._median_rep: float = 0.0

        # social learning: exponentially-weighted recent success count
        # lab_domain_successes[lab_id][domain] — decayed ×0.90 per step
        self.lab_domain_successes: dict[int, dict[int, float]] = {
            i: {d: 0.0 for d in range(n_domains)}
            for i in range(n_labs)
        }

        self.domain_truthfulness_caps  = [float(self.rng.uniform(0.35, 0.60)) for _ in range(n_domains)]
        self.domain_difficulty_floors  = [float(self.rng.uniform(0.05, 0.40)) for _ in range(n_domains)]

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Models per Domain": lambda m: {
                    d: sum(1 for s in m.scientific_models if s.domain == d)
                    for d in range(m.n_domains)
                },
                "Best Actual Truthfulness per Domain": lambda m: {
                    d: max(
                        (s.actual_truthfulness for s in m.scientific_models if s.domain == d),
                        default=0.0
                    )
                    for d in range(m.n_domains)
                },
                "Best Reported Truthfulness per Domain": lambda m: {
                    d: max(
                        (s.reported_truthfulness for s in m.scientific_models if s.domain == d),
                        default=0.0
                    )
                    for d in range(m.n_domains)
                },
                "Avg Bias Gap": lambda m: float(np.mean([
                    s.reported_truthfulness - s.actual_truthfulness
                    for s in m.scientific_models
                ])) if m.scientific_models else 0.0,
                "Avg Top5 Salience": lambda m: float(np.mean(
                    sorted([s.salience for s in m.scientific_models], reverse=True)[:5]
                )) if m.scientific_models else 0.0,
                "Min Top5 Salience": lambda m: float(min(
                    sorted([s.salience for s in m.scientific_models], reverse=True)[:5]
                )) if m.scientific_models else 0.0,
                "Replication Failure Rate": lambda m: float(
                    m.replication_failures / m.replication_attempts
                ) if m.replication_attempts > 0 else 0.0,
                "Domain Failure Rates": lambda m: {
                    d: m.domain_replication_failures[d] / m.domain_replication_attempts[d]
                    if m.domain_replication_attempts[d] > 0 else 0.0
                    for d in range(m.n_domains)
                },
                "Avg Reputation": lambda m: float(np.mean(
                    [a.reputation for a in m.agents]
                )),
                "Reputation Variance": lambda m: float(np.var(
                    [a.reputation for a in m.agents]
                )),
                "Avg Reputation Lost to Debunk": lambda m: float(np.mean(
                    [a.reputation_lost_to_debunk for a in m.agents]
                )),
                "Median Reputation": lambda m: float(np.median(
                    [a.reputation for a in m.agents]
                )),
            },
            agent_reporters={
                "LabID":         "lab_id",
                "MeanSkill":     "mean_skill",
                "PeakSkill":     lambda a: a.domain_skills[a.home_domain],
                "Publications":  "publications",
                "TrainingSteps": "training_steps",
                "ExploitSteps":  "exploit_steps",
                "ExploreSteps":  "explore_steps",
                "DebunkSteps":   "debunk_steps",
                "CareerAge":     "career_age",
                "CurrentDomain": "current_domain",
                "DomainSkills":             lambda a: list(a.domain_skills),
                "ReputationLostToDebunk":   "reputation_lost_to_debunk",
                "Reputation":               "reputation",
            }
        )

        # --- create labs ---
        self.labs: list[Lab] = []
        for i in range(n_labs):
            peak = int(self.rng.integers(0, n_domains))
            fingerprint = [
                float(np.clip(
                    self.rng.normal(
                        peak_skill_mean if d == peak else other_skill_mean,
                        peak_skill_std  if d == peak else other_skill_std,
                    ), 0.01, 0.95
                ))
                for d in range(n_domains)
            ]
            self.labs.append(Lab(lab_id=i, fingerprint=fingerprint))

        # --- create researchers ---
        for lab in self.labs:
            for _ in range(researchers_per_lab):
                skills = [
                    float(np.clip(
                        self.rng.normal(lab.fingerprint[d], 0.03),
                        0.01, 0.95
                    ))
                    for d in range(n_domains)
                ]
                Researcher(self, lab_id=lab.lab_id, domain_skills=skills,
                           lab_fingerprint=lab.fingerprint,
                           train_threshold=train_threshold,
                           skill_gain_attempt=skill_gain_attempt,
                           skill_gain_train=skill_gain_train)

        # --- seed one model per domain ---
        for d in range(n_domains):
            lab = self.labs[d % n_labs]
            self.spawn_model(
                origin_lab_id=lab.lab_id,
                domain=d,
                researcher_skills=lab.fingerprint,
            )

    # --- queries ---

    def domain_saturation(self, domain: int) -> float:
        models = [m for m in self.scientific_models if m.domain == domain]
        if not models:
            return 0.0
        return max(m.actual_truthfulness for m in models) / self.domain_truthfulness_caps[domain]

    def get_lab(self, lab_id: int) -> Lab | None:
        return next((l for l in self.labs if l.lab_id == lab_id), None)

    # --- social learning ---

    def record_domain_success(self, lab_id: int, domain: int):
        """
        Called by a researcher when they successfully exploit a model.
        Increments the lab's social signal for that domain, making
        labmates more likely to choose work in the same area next step.
        """
        if lab_id in self.lab_domain_successes:
            self.lab_domain_successes[lab_id][domain] = (
                self.lab_domain_successes[lab_id].get(domain, 0.0) + 1.0
            )

    # --- evolutionary selection ---

    def _evolve(self):
        researchers = list(self.agents)
        if len(researchers) < 4:
            return
        ranked  = sorted(researchers, key=lambda a: a.reputation - a.reputation_at_last_cull)
        n_cull  = max(1, int(len(ranked) * self.cull_fraction))
        losers  = ranked[:n_cull]
        elite   = ranked[-n_cull:]

        for loser in losers:
            parent = elite[int(self.rng.integers(0, len(elite)))]
            child_skills = [
                float(np.clip(self.rng.normal(s, self.mutation_std), 0.01, 0.95))
                for s in parent.domain_skills
            ]
            lab_id = loser.lab_id
            lab    = self.get_lab(lab_id)
            loser.remove()
            Researcher(self, lab_id=lab_id, domain_skills=child_skills,
                       lab_fingerprint=lab.fingerprint if lab else None,
                       train_threshold=self._train_threshold,
                       skill_gain_attempt=self._skill_gain_attempt,
                       skill_gain_train=self._skill_gain_train)

        self._maybe_replace_lab()

        for r in self.agents:
            r.reputation_at_last_cull = r.reputation

    def _maybe_replace_lab(self):
        researchers = list(self.agents)
        if not researchers:
            return

        lab_recent: dict[int, float] = {}
        lab_counts: dict[int, int]   = {}
        for r in researchers:
            gain = r.reputation - r.reputation_at_last_cull
            lab_recent[r.lab_id] = lab_recent.get(r.lab_id, 0.0) + gain
            lab_counts[r.lab_id] = lab_counts.get(r.lab_id, 0) + 1

        lab_mean     = {lid: lab_recent[lid] / lab_counts[lid] for lid in lab_recent}
        overall_mean = sum(lab_mean.values()) / len(lab_mean)

        if overall_mean <= 0:
            return

        worst_id = min(lab_mean, key=lab_mean.get)
        if lab_mean[worst_id] >= self.lab_close_threshold * overall_mean:
            return

        lab = self.get_lab(worst_id)
        if lab is None:
            return

        peak = int(self.rng.integers(0, self.n_domains))
        new_fp = [
            float(np.clip(
                self.rng.normal(
                    self._peak_skill_mean  if d == peak else self._other_skill_mean,
                    self._peak_skill_std   if d == peak else self._other_skill_std,
                ), 0.01, 0.95
            ))
            for d in range(self.n_domains)
        ]
        old_peak = int(np.argmax(lab.fingerprint))
        lab.fingerprint = new_fp
        for r in researchers:
            if r.lab_id == worst_id:
                r.lab_fingerprint = list(new_fp)

        self.lab_turnover_events.append(
            (self._step_count, worst_id, old_peak, peak)
        )

    # --- model spawning ---

    def spawn_model(self, origin_lab_id: int, domain: int,
                    researcher_skills: list[float],
                    initial_salience:   float     = 0.5,
                    parent_uid:         int | None = None,
                    author_agent_id:    int | None = None,
                    breakthrough:       bool       = False,
                    misconduct:         bool       = False,
                    author_reputation:  float | None = None):
        """
        Publish a new ScientificModel.

        Bias inflation logic (v2):
          1. breakthrough=True  → lower bias (more scrutiny before publication)
          2. misconduct=True    → higher bias draw (strategic inflation)
          3. otherwise          → standard draw, shifted up by competitive
                                  pressure when author is below field median
        """
        cap       = self.domain_truthfulness_caps[domain]
        max_truth = max(
            (m.actual_truthfulness for m in self.scientific_models if m.domain == domain),
            default=0.1
        )
        gap = cap - max_truth

        if breakthrough:
            alpha          = 12.0 + (max_truth * 10)
            gain           = float(self.rng.beta(alpha, 2)) * gap * 0.85
            bias_inflation = float(np.clip(self.rng.normal(0.05, 0.03), 0.0, 0.15))
            initial_salience = 0.90
        elif misconduct:
            # Strategic inflation: mean shifted to 0.22, wider variance
            # Models here are more fragile and more likely to be debunked
            alpha          = 2.0 + (max_truth * 10)
            gain           = float(self.rng.beta(alpha, 5)) * gap * 0.5
            bias_inflation = float(np.clip(self.rng.normal(0.22, 0.06), 0.05, 0.40))
        else:
            alpha = 2.0 + (max_truth * 10)
            gain  = float(self.rng.beta(alpha, 5)) * gap * 0.5
            # Competitive pressure: authors below field median inflate more
            if author_reputation is not None and self._median_rep > 0:
                pressure = float(np.clip(
                    1.0 - author_reputation / self._median_rep, 0.0, 1.0
                ))
            else:
                pressure = 0.0
            bias_mean      = 0.10 + pressure * 0.08
            bias_inflation = float(np.clip(self.rng.normal(bias_mean, 0.05), 0.0, 0.35))

        truthfulness = float(np.clip(max_truth + gain, 0.01, cap - 1e-4))

        complexity = [
            float(np.clip(
                self.rng.normal(researcher_skills[d], 0.05),
                self.domain_difficulty_floors[d],
                0.95
            ))
            for d in range(self.n_domains)
        ]

        m = ScientificModel(
            uid=self._model_counter,
            domain=domain,
            complexity=complexity,
            truthfulness=truthfulness,
            origin_lab_id=origin_lab_id,
            parent_uid=parent_uid,
            bias_inflation=bias_inflation,
            author_agent_id=author_agent_id,
        )
        m.salience = initial_salience
        self.scientific_models.append(m)
        self._model_counter += 1

    # --- main loop ---

    def step(self):
        self._step_count += 1

        # refresh domain caches
        for d in range(self.n_domains):
            self._cached_saturations[d] = self.domain_saturation(d)
        self._cached_domain_counts = {d: 0 for d in range(self.n_domains)}
        for m in self.scientific_models:
            self._cached_domain_counts[m.domain] = (
                self._cached_domain_counts.get(m.domain, 0) + 1
            )

        # update field median reputation — used by spawn_model for pressure calc
        reps = [a.reputation for a in self.agents]
        self._median_rep = float(np.median(reps)) if reps else 0.0

        self.datacollector.collect(self)
        self.agents.shuffle_do("step")

        # model salience decay and saturation penalties
        saturations = self._cached_saturations
        for m in self.scientific_models:
            m.decay()
            sat = saturations[m.domain]
            if sat >= 0.90:
                m.salience = max(0.0, m.salience - 0.075)
            elif sat >= 0.70:
                m.salience = max(0.0, m.salience - 0.025)

        # truthfulness cap growth
        for d in range(self.n_domains):
            self.domain_truthfulness_caps[d] = min(
                0.95,
                self.domain_truthfulness_caps[d]
                + self.cap_growth_rate * (0.95 - self.domain_truthfulness_caps[d])
            )

        # decay social learning signals (exponential, half-life ≈ 6 steps)
        for lab_id in self.lab_domain_successes:
            for d in self.lab_domain_successes[lab_id]:
                self.lab_domain_successes[lab_id][d] *= 0.90

        if self._step_count % self.selection_interval == 0:
            self._evolve()
