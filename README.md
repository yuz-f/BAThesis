# Of the Standards of Science
### Computational Modelling as Psychology's Common Language

**Bachelor Thesis** · Yuzan Ilario Flury · Department of Psychology, University of Zurich

---

## Overview

This thesis investigates whether differences in lab structure — specifically the degree to which researchers specialise versus generalise across domains — affect the replication failure rate of a scientific community. The core argument is that psychology's replication crisis is not only a product of individual researcher behaviour or incentive structures, but of the communicative and epistemic infrastructure of the field as a whole. Computational formalisation is proposed as a remedy: by making theoretical commitments explicit and mechanistically comparable across labs, formal models enable the kind of cumulative science that verbal theories structurally cannot.

The empirical vehicle is an agent-based model (ABM) of a stylised scientific community, calibrated to the Open Science Collaboration's (~60%) replication failure rate. Two scenarios are compared — **specialist labs** (sharp skill peaks, narrow domain coverage) and **generalist labs** (broad overlapping competence) — to examine how lab structure shapes knowledge accumulation and replication reliability.

---

## Repository Structure

```
BAThesis/
├── thesis.qmd              # main thesis document (Quarto, renders to PDF/HTML)
├── thesis.pdf              # compiled thesis
├── code/
│   ├── world.py            # ScienceWorld — Mesa Model, main simulation environment
│   ├── researcher.py       # Researcher — Mesa Agent, skill learning and decision logic
│   ├── scientific_model.py # ScientificModel — passive knowledge object (salience, truthfulness)
│   ├── lab.py              # Lab — institutional fingerprint anchoring researcher skill drift
│   ├── run.py              # entry point: runs both scenarios and generates all figures
│   ├── visualization.py    # all thesis figures (matplotlib)
│   ├── sensitivity_analysis.py  # one-at-a-time sensitivity analysis across 6 parameters
│   └── grid_search.py      # calibration grid search (train_threshold × skill_gain_attempt)
├── meta/
│   ├── references.bib      # bibliography (APA 7)
│   ├── sensitivity_results.csv
│   ├── grid_search_results.csv
│   └── img/                # all generated figures
└── extension/
    ├── apa_preamble.tex    # LaTeX header for APA formatting
    └── APA.css             # HTML stylesheet
```

---

## The Model

The simulation runs 10 labs of 5 researchers each (50 agents total) over 300 steps. Each agent holds a **skill vector** across 10 scientific domains and decides each step whether to:

- **Exploit** — implement an existing published model (multi-step commitment)
- **Invest** — train toward a model currently out of reach
- **Explore** — publish a new model in an underexplored domain

Researchers choose probabilistically, with weights shaped by expected value and a **home-domain bias** derived from their own skill distribution: the weight for any action is multiplied by √(skill_in_domain / mean_skill), so specialists are pulled toward their expertise while generalists face no directional pressure. This mechanism is parameter-free and grounded in bibliometric evidence that scientists explore local knowledge spaces conservatively (Jia et al., 2019; Liu et al., 2024).

**Replication failure** occurs when a researcher attempts to implement an existing model but their domain skill falls below a random draw — a direct operationalisation of the gap between methodological demand and researcher competence. The target failure rate (~60%) is calibrated to the Open Science Collaboration (2015).

**Evolutionary selection** every 40 steps replaces the bottom 25% of researchers (by recent reputation gain) with mutated copies of the elite, preserving lab population balance.

---

## Running the Simulation

Requires Python 3.10+ and [Mesa](https://mesa.readthedocs.io/) 3.5+.

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install mesa numpy matplotlib

# Run both scenarios and generate figures
cd code
python run.py

# Calibration and sensitivity
python grid_search.py
python sensitivity_analysis.py
```

Figures are saved to `meta/img/`. Results CSVs go to `meta/`.

---

## Key Design Decisions

| Choice | Rationale |
|---|---|
| Skill-vector home-domain bias (√-scaled) | Parameter-free; emerges from agent state; empirically supported |
| Success probability = domain skill only | Clean, interpretable; avoids circular use of Pearson COR |
| Truthfulness cap grows asymptotically | Domains retain marginal value throughout the run |
| Salience shield fades after 20 uncited steps | Era dynamics: superseded models decay predictably |
| Fingerprint drift (0.2%/step pull, 0.4%/step decay) | Institutional anchoring without erasing individual learning |

---

## References (selected)

- Open Science Collaboration (2015). Estimating the reproducibility of psychological science. *Science*, 349.
- Smaldino, P. E., & McElreath, R. (2016). The natural selection of bad science. *Royal Society Open Science*, 3.
- Devezer, B., et al. (2019). Scientific discovery in a model-centric framework. *PLOS ONE*, 14.
- Jia, T., et al. (2019). Increasing trend of scientists to switch between topics. *Nature Communications*, 10.
- Eronen, M. I., & Bringmann, L. F. (2021). The theory crisis in psychology. *Perspectives on Psychological Science*, 16.
