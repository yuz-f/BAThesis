from __future__ import annotations


class Lab:
    """
    A research institution with a domain fingerprint.

    The fingerprint defines the lab's research identity. At simulation start
    it determines the initial skill profiles of researchers seeded into the
    lab; thereafter it acts as a passive attractor for those researchers'
    non-peak skills via fingerprint drift (Researcher.step). Labs do not learn
    or publish; their institutional character is maintained through this drift
    rather than through hiring — when an evolutionary replacement occurs, the
    new researcher inherits its skills from a mutated elite parent, not from
    the lab fingerprint, and only the subsequent per-step drift pulls those
    skills toward the lab's identity over time.

    Fingerprint replacement (lab turnover):
        ScienceWorld._maybe_replace_lab() overwrites the fingerprint when a
        lab's recent per-researcher reputation gain falls far below the field
        mean. Existing researchers keep their personal skills but begin
        drifting toward the new institutional direction, representing a change
        in the lab's research identity rather than incremental learning.
    """

    def __init__(self, lab_id: int, fingerprint: list[float]):
        self.lab_id      = lab_id
        self.fingerprint = fingerprint   # domain skill profile; may be replaced by lab turnover
