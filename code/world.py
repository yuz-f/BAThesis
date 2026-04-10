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
    """

    def __init__(self,
                 n_labs:             int   = 20,
                 n_domains:          int   = 10,
                 n_seed_models:      int   = 12,
                 peak_skill_mean:    float = 0.70,
                 other_skill_mean:   float = 0.15,
                 selection_interval: int   = 50,
                 cull_fraction:      float = 0.25,
                 mutation_std:       float = 0.04,
                 rng: int | None           = None):
        super().__init__(rng=rng)
        self.n_domains           = n_domains
        self.selection_interval  = selection_interval
        self.cull_fraction       = cull_fraction
        self.mutation_std        = mutation_std
        self.scientific_models: list[ScientificModel] = []
        self._model_counter     = 0
        self._step_count        = 0

        # domain-level structural properties — randomly assigned, fixed for the run
        # fidelity_cap:      ceiling on model quality; once reached, novelty collapses
        # difficulty_floor:  minimum complexity regardless of fidelity
        self.domain_fidelity_caps     = {
            d: float(self.rng.uniform(0.5, 0.80)) for d in range(n_domains)
        }
        self.domain_difficulty_floors = {
            d: float(self.rng.uniform(0.05, 0.40)) for d in range(n_domains)
        }

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Models per Domain":          lambda m: {
                    d: sum(1 for s in m.scientific_models if s.domain == d)
                    for d in range(m.n_domains)
                },
                "Cross-Lab Pub Rate":         self._cross_lab_rate,
                "Avg Reputation":             lambda m: float(np.mean(
                    [a.reputation for a in m.agents]
                )),
            },
            agent_reporters={
                "MeanSkill":       "mean_skill",
                "PeakSkill":       lambda a: a.domain_skills[a.home_domain],
                "Publications":    "publications",
                "TrainingSteps":   "training_steps",
                "CrossLabPubs":    "cross_lab_pubs",
                "CurrentDomain":   "current_domain",
            }
        )

        # spawn labs — each gets a randomly chosen peak domain
        # domain_skills is a list (index = domain) acting as a fingerprint vector
        for _ in range(n_labs):
            peak = int(self.rng.integers(0, n_domains))
            domain_skills = [
                float(np.clip(
                    self.rng.normal(
                        peak_skill_mean if d == peak else other_skill_mean,
                        0.10 if d == peak else 0.05
                    ), 0.01, 0.95
                ))
                for d in range(n_domains)
            ]
            ResearchLab(self, domain_skills=domain_skills)

        # seed models from each lab's current peak domain
        lab_list = list(self.agents)
        for i in range(n_seed_models):
            owner = lab_list[i % len(lab_list)]
            self.spawn_model(origin_lab_id=owner.unique_id,
                             domain=owner.home_domain)

    def domain_saturation(self, domain: int) -> float:
        """
        How close is this domain to its fidelity cap? Returns [0, 1].
        0 = no models or far from cap, 1 = at the ceiling.
        """
        models = [m for m in self.scientific_models if m.domain == domain]
        if not models:
            return 0.0
        max_fidelity = max(m.fidelity for m in models)
        return max_fidelity / self.domain_fidelity_caps[domain]

    # --- reporters ---

    def _cross_lab_rate(self) -> float:
        total = sum(a.publications for a in self.agents)
        cross = sum(a.cross_lab_pubs for a in self.agents)
        return cross / total if total > 0 else 0.0

    # --- evolutionary selection ---

    def _evolve(self):
        lab_list = list(self.agents)
        if len(lab_list) < 4:
            return
        ranked = sorted(lab_list,
                        key=lambda a: a.reputation - a.reputation_at_last_cull)
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

        # reset baseline for all surviving labs
        for lab in self.agents:
            lab.reputation_at_last_cull = lab.reputation

    # --- helpers ---

    def spawn_model(self, origin_lab_id: int, domain: int, initial_novelty: float = 0.5):
        # if the domain is saturated, exploration pushes the frontier —
        # representing a paradigm shift that opens new research possibilities
        if self.domain_saturation(domain) >= 0.95:
            self.domain_fidelity_caps[domain] = min(1.0,
                self.domain_fidelity_caps[domain] + 0.05)

        # fidelity increases with domain maturity, capped by domain ceiling
        n_existing = sum(1 for m in self.scientific_models if m.domain == domain)
        alpha      = 2.0 + (n_existing * 0.8)
        fidelity   = float(min(
            self.rng.beta(alpha, 5),
            self.domain_fidelity_caps[domain]
        ))

        # complexity inherited from the origin lab's skill fingerprint in this domain —
        # expert labs create complex models, novice labs create simpler ones.
        # This means labs with a similar profile will find this model a natural fit.
        origin_lab = next((a for a in self.agents if a.unique_id == origin_lab_id), None)
        lab_skill  = origin_lab.domain_skills[domain] if origin_lab else 0.5
        complexity = float(np.clip(
            self.rng.normal(lab_skill, 0.05),
            self.domain_difficulty_floors[domain],
            0.95
        ))

        m = ScientificModel(
            uid=self._model_counter,
            domain=domain,
            complexity=complexity,
            fidelity=fidelity,
            origin_lab_id=origin_lab_id,
        )
        m.novelty = initial_novelty
        self.scientific_models.append(m)
        self._model_counter += 1

    def step(self):
        self._step_count += 1
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")
        for m in self.scientific_models:
            m.decay()
            # models at the fidelity cap lose novelty faster —
            # the field is exhausted, nothing exciting left to discover
            if m.fidelity >= self.domain_fidelity_caps[m.domain] * 0.95:
                m.novelty = max(0.0, m.novelty - 0.08)
        if self._step_count % self.selection_interval == 0:
            self._evolve()
