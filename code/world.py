import numpy as np
import mesa
from scientific_model import ScientificModel
from research_lab import ResearchLab


class ScienceWorld(mesa.Model):
    """
    The simulation environment.

    Manages labs, scientific models, data collection, and evolutionary selection.
    Every `selection_interval` steps the bottom-performing labs are replaced
    by offspring of the elite, inheriting their domain skill vector with mutation.

    Skill profiles:
      - Specialist scenario: high peak_skill, low other_skill_mean + tight other_skill_std
        → one clear peak, near-zero elsewhere
      - Generalist scenario: high other_skill_mean + wide other_skill_std
        → natural secondary peaks emerge, higher average across domains

    Era dynamics:
      - Domain salience decays faster as saturation grows (tiered: ≥0.70 mild, ≥0.90 hard)
      - Truthfulness cap can only be raised max_cap_raises times per domain
        → domains eventually hit a true ceiling and become unattractive
    """

    def __init__(self,
                 n_labs:             int   = 20,
                 n_domains:          int   = 10,
                 n_seed_models:      int   = 12,
                 peak_skill_mean:    float = 0.75,
                 peak_skill_std:     float = 0.08,
                 other_skill_mean:   float = 0.15,
                 other_skill_std:    float = 0.05,
                 selection_interval: int   = 50,
                 cull_fraction:      float = 0.25,
                 mutation_std:       float = 0.04,
                 max_cap_raises:     int   = 2,
                 rng: int | None           = None):
        super().__init__(rng=rng)
        self.n_domains           = n_domains
        self.selection_interval  = selection_interval
        self.cull_fraction       = cull_fraction
        self.mutation_std        = mutation_std
        self.max_cap_raises      = max_cap_raises
        self.scientific_models: list[ScientificModel] = []
        self._model_counter     = 0
        self._step_count        = 0

        # domain-level structural properties — lists indexed by domain id
        # truthfulness_cap:      ceiling on model quality; once reached, salience collapses
        # difficulty_floor:  minimum complexity any model in this domain can have
        # cap_raises:        how many times the cap has already been pushed up
        self.domain_truthfulness_caps     = [float(self.rng.uniform(0.50, 0.80)) for _ in range(n_domains)]
        self.domain_difficulty_floors = [float(self.rng.uniform(0.05, 0.40)) for _ in range(n_domains)]
        self.domain_cap_raises        = [0] * n_domains

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Models per Domain": lambda m: {
                    d: sum(1 for s in m.scientific_models if s.domain == d)
                    for d in range(m.n_domains)
                },
                "Avg Domain Capacity": lambda m: float(np.mean(m.domain_truthfulness_caps)),
                "Avg Reputation":     lambda m: float(np.mean(
                    [a.reputation for a in m.agents]
                )),
            },
            agent_reporters={
                "MeanSkill":     "mean_skill",
                "PeakSkill":     lambda a: a.domain_skills[a.home_domain],
                "Publications":  "publications",
                "TrainingSteps": "training_steps",
                "CurrentDomain": "current_domain",
                "DomainSkills":  lambda a: list(a.domain_skills),
            }
        )

        # spawn labs — skill vector is a fingerprint (index = domain)
        # specialist profile:  tight peak, near-zero elsewhere (small other_skill_std)
        # generalist profile:  same peak but wide other_skill distribution,
        #                      allowing natural secondary peaks to emerge
        for _ in range(n_labs):
            peak = int(self.rng.integers(0, n_domains))
            domain_skills = [
                float(np.clip(
                    self.rng.normal(
                        peak_skill_mean if d == peak else other_skill_mean,
                        peak_skill_std  if d == peak else other_skill_std,
                    ), 0.01, 0.95
                ))
                for d in range(n_domains)
            ]
            ResearchLab(self, domain_skills=domain_skills)

        # seed one model per domain so no domain starts empty.
        # owner rotates across labs; domain is assigned explicitly so every
        # domain is reachable from step 1 regardless of lab peak distribution.
        lab_list = list(self.agents)
        for d in range(n_domains):
            owner = lab_list[d % len(lab_list)]
            self.spawn_model(origin_lab_id=owner.unique_id, domain=d)

    # --- queries ---

    def domain_saturation(self, domain: int) -> float:
        """How close is this domain to its truthfulness cap? Returns [0, 1]."""
        models = [m for m in self.scientific_models if m.domain == domain]
        if not models:
            return 0.0
        return max(m.truthfulness for m in models) / self.domain_truthfulness_caps[domain]

    # --- evolutionary selection ---

    def _evolve(self):
        lab_list = list(self.agents)
        if len(lab_list) < 4:
            return
        ranked = sorted(lab_list, key=lambda a: a.reputation - a.reputation_at_last_cull)
        n_cull = max(1, int(len(ranked) * self.cull_fraction))
        losers = ranked[:n_cull]
        elite  = ranked[-n_cull:]

        for loser in losers:
            parent = elite[int(self.rng.integers(0, len(elite)))]
            child_skills = [
                float(np.clip(self.rng.normal(s, self.mutation_std), 0.01, 0.95))
                for s in parent.domain_skills
            ]
            loser.remove()
            ResearchLab(self, domain_skills=child_skills)

        # reset reputation baseline for relative fitness tracking
        for lab in self.agents:
            lab.reputation_at_last_cull = lab.reputation

    # --- model spawning ---

    def spawn_model(self, origin_lab_id: int, domain: int, initial_salience: float = 0.5):
        # push the truthfulness ceiling only if the domain is almost fully saturated
        # and still under the raise limit. Threshold raised to 0.98 and increment
        # halved to 0.02 so labs must sustain near-total saturation before the cap
        # budges — making the ceiling feel genuinely hard to push through.
        if (self.domain_saturation(domain) >= 0.98
                and self.domain_cap_raises[domain] < self.max_cap_raises):
            self.domain_truthfulness_caps[domain] = min(1.0,
                self.domain_truthfulness_caps[domain] + 0.02)
            self.domain_cap_raises[domain] += 1

        # truthfulness builds on the current state of the art in this domain.
        # alpha scales with the best existing truthfulness rather than model count
        # so both specialist and generalist labs benefit equally from a mature domain.
        #   fresh domain  (max ≈ 0.1) → alpha ≈ 3  → Beta mean ≈ 0.38
        #   mature domain (max ≈ 0.7) → alpha ≈ 9  → Beta mean ≈ 0.64
        max_truth  = max(
            (m.truthfulness for m in self.scientific_models if m.domain == domain),
            default=0.1
        )
        alpha      = 2.0 + (max_truth * 10)
        truthfulness = float(min(
            self.rng.beta(alpha, 5),
            self.domain_truthfulness_caps[domain]
        ))

        # complexity vector inherited from the lab's full skill fingerprint —
        # each entry is the complexity requirement in that domain, drawn from the
        # lab's skill level there (with small noise). Labs with a matching profile
        # will find this model a natural fit across all dimensions.
        origin_lab = next((a for a in self.agents if a.unique_id == origin_lab_id), None)
        lab_skills = origin_lab.domain_skills if origin_lab else [0.5] * self.n_domains
        complexity = [
            float(np.clip(
                self.rng.normal(lab_skills[d], 0.05),
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
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")

        # precompute saturations once — avoids O(models²) recalculation
        saturations = [self.domain_saturation(d) for d in range(self.n_domains)]

        for m in self.scientific_models:
            m.decay()
            sat = saturations[m.domain]
            # tiered extra salience decay drives era transitions:
            #   ≥ 0.90 saturation → hard collapse (-0.15/step)
            #   ≥ 0.70 saturation → building pressure (-0.05/step)
            if sat >= 0.90:
                m.salience = max(0.0, m.salience - 0.15)
            elif sat >= 0.70:
                m.salience = max(0.0, m.salience - 0.05)

        if self._step_count % self.selection_interval == 0:
            self._evolve()
