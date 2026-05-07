from __future__ import annotations


class Lab:
    """
    A research institution with a domain fingerprint.

    The fingerprint defines the lab's research identity and determines the
    initial skill profiles of researchers hired into it, as well as the
    direction of passive fingerprint drift for existing researchers.  Labs do
    not learn or publish; their institutional character is maintained through
    hiring.

    Fingerprint replacement (lab turnover):
        ScienceWorld._maybe_replace_lab() overwrites the fingerprint when a
        lab's recent per-researcher reputation gain falls far below the field
        mean.  Existing researchers keep their personal skills but begin
        drifting toward the new institutional direction, representing a change
        in the lab's research identity rather than incremental learning.
    """

    def __init__(self, lab_id: int, fingerprint: list[float]):
        self.lab_id      = lab_id
        self.fingerprint = fingerprint   # domain skill profile; may be replaced by lab turnover
