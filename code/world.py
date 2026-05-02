import numpy as np
import mesa
from scientific_model import ScientificModel
from lab import Lab
from researcher import Researcher


class ScienceWorld(mesa.Model):
    """
    The simulation environment.

    Structure:
      - 10 Labs:  fixed fingerprint institutions. Define the research identity
                  of their members and seed the initial model landscape.
      - 50 Researchers (5 per lab): Mesa agents that learn, publish, and get
                  selected. Start with skills drawn from their lab's fingerprint;
                  diverge over time through assimilation.

    Evolutionary selection every `selection_interval` steps: bottom-performing
    researchers are replaced by mutated copies of elite researchers. The
    replacement researcher joins the culled researcher's lab slot, preserving
    lab population balance.

    Era dynamics:
      - Domain salience decays faster as saturation grows (tiered: ≥0.70 mild, ≥0.90 hard)
      - Truthfulness cap can only be raised max_cap_raises times per domain
        → domains eventually hit a true ceiling and become unattractive
    """

    def __init__(self,
                 n_labs:              int   = 10,
                 researchers_per_lab: int   = 5,
                 n_domains:           int   = 10,
                 peak_skill_mean:     float = 0.55,
                 peak_skill_std:      float = 0.07,
                 other_skill_mean:    float = 0.25,
                 other_skill_std:     float = 0.06,
                 selection_interval:  int   = 25,
                 cull_fraction:       float = 0.25,
                 mutation_std:        float = 0.04,
                 max_cap_raises:      int   = 2,
                 lab_close_threshold: float = 0.5,
                 rng: int | None            = None):
        super().__init__(rng=rng)
        self.n_domains          = n_domains
        self.n_labs             = n_labs
        self.selection_interval = selection_interval
        self.cull_fraction      = cull_fraction
        self.mutation_std       = mutation_std
        self.max_cap_raises     = max_cap_raises
        self.lab_close_threshold = lab_close_threshold
        # stored so _maybe_replace_lab can draw new fingerprints consistently
        self._peak_skill_mean   = peak_skill_mean
        self._peak_skill_std    = peak_skill_std
        self._other_skill_mean  = other_skill_mean
        self._other_skill_std   = other_skill_std
        self.lab_turnover_events: list[tuple] = []  # (step, lab_id, old_peak, new_peak)
        self.scientific_models: list[ScientificModel] = []
        self._model_counter     = 0
        self._step_count        = 0

        # step-level caches — recomputed once at the start of every step
        # and read by all researchers, avoiding redundant full-list scans
        self._cached_saturations:   list[float] = [0.0] * n_domains
        self._cached_domain_counts: dict[int, int] = {d: 0 for d in range(n_domains)}

        self.domain_truthfulness_caps = [float(self.rng.uniform(0.50, 0.80)) for _ in range(n_domains)]
        self.domain_difficulty_floors = [float(self.rng.uniform(0.05, 0.40)) for _ in range(n_domains)]
        self.domain_cap_raises        = [0] * n_domains

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Models per Domain": lambda m: {
                    d: sum(1 for s in m.scientific_models if s.domain == d)
                    for d in range(m.n_domains)
                },
                "Avg Domain Capacity": lambda m: float(np.mean(m.domain_truthfulness_caps)),
                "Avg Reputation":      lambda m: float(np.mean(
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
                "CurrentDomain": "current_domain",
                "DomainSkills":  lambda a: list(a.domain_skills),
            }
        )

        # --- create labs with fixed fingerprints ---
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

        # --- create researchers — skills drawn from their lab's fingerprint ---
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
                           lab_fingerprint=lab.fingerprint)

        # --- seed one model per domain using lab fingerprints ---
        for d in range(n_domains):
            lab = self.labs[d % n_labs]
            self.spawn_model(
                origin_lab_id=lab.lab_id,
                domain=d,
                researcher_skills=lab.fingerprint,
            )

    # --- queries ---

    def domain_saturation(self, domain: int) -> float:
        """How close is this domain to its truthfulness cap? Returns [0, 1]."""
        models = [m for m in self.scientific_models if m.domain == domain]
        if not models:
            return 0.0
        return max(m.truthfulness for m in models) / self.domain_truthfulness_caps[domain]

    def get_lab(self, lab_id: int) -> Lab | None:
        return next((l for l in self.labs if l.lab_id == lab_id), None)

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
            lab_id = loser.lab_id   # replacement joins the same lab slot
            lab    = self.get_lab(lab_id)
            loser.remove()
            Researcher(self, lab_id=lab_id, domain_skills=child_skills,
                       lab_fingerprint=lab.fingerprint if lab else None)

        self._maybe_replace_lab()

        for r in self.agents:
            r.reputation_at_last_cull = r.reputation

    def _maybe_replace_lab(self):
        """
        Replace the weakest lab's fingerprint if it falls far enough below average.

        Evaluated after each cull using reputation gained since the previous cull,
        so it reflects recent performance rather than career totals.

        A new random fingerprint is drawn and pushed to all current researchers
        in that lab — they keep their skills but now drift toward the new direction.
        Lab count and population size stay fixed.
        """
        researchers = list(self.agents)
        if not researchers:
            return

        lab_recent: dict[int, float] = {}
        lab_counts: dict[int, int]   = {}
        for r in researchers:
            gain = r.reputation - r.reputation_at_last_cull
            lab_recent[r.lab_id] = lab_recent.get(r.lab_id, 0.0) + gain
            lab_counts[r.lab_id] = lab_counts.get(r.lab_id, 0) + 1

        lab_mean    = {lid: lab_recent[lid] / lab_counts[lid] for lid in lab_recent}
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
                    initial_salience: float = 0.5):
        """
        Publish a new ScientificModel.

        Complexity is drawn from the researcher's current skill profile — not
        the lab's fingerprint — so models reflect individual expertise, which
        may have diverged from the lab baseline through assimilation.
        """
        if (self.domain_saturation(domain) >= 0.98
                and self.domain_cap_raises[domain] < self.max_cap_raises):
            self.domain_truthfulness_caps[domain] = min(1.0,
                self.domain_truthfulness_caps[domain] + 0.02)
            self.domain_cap_raises[domain] += 1

        cap       = self.domain_truthfulness_caps[domain]
        max_truth = max(
            (m.truthfulness for m in self.scientific_models if m.domain == domain),
            default=0.1
        )
        # Asymptotic approach to cap: each new model closes a fraction of the
        # remaining gap, so the cap is approached but never reached.
        alpha        = 2.0 + (max_truth * 10)
        gap          = cap - max_truth
        gain         = float(self.rng.beta(alpha, 5)) * gap * 0.5
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
        )
        m.salience = initial_salience
        self.scientific_models.append(m)
        self._model_counter += 1

    # --- main loop ---

    def step(self):
        self._step_count += 1

        # refresh caches once — all researchers read from these this step
        for d in range(self.n_domains):
            self._cached_saturations[d] = self.domain_saturation(d)
        self._cached_domain_counts = {d: 0 for d in range(self.n_domains)}
        for m in self.scientific_models:
            self._cached_domain_counts[m.domain] = self._cached_domain_counts.get(m.domain, 0) + 1

        self.datacollector.collect(self)
        self.agents.shuffle_do("step")

        saturations = self._cached_saturations
        for m in self.scientific_models:
            m.decay()
            sat = saturations[m.domain]
            if sat >= 0.90:
                m.salience = max(0.0, m.salience - 0.075)
            elif sat >= 0.70:
                m.salience = max(0.0, m.salience - 0.025)

        if self._step_count % self.selection_interval == 0:
            self._evolve()
