from __future__ import annotations


class Lab:
    """
    A research institution with a fixed domain fingerprint.

    The fingerprint defines the lab's research identity and determines
    the initial skill profiles of researchers hired into it. Labs do not
    learn or act — their character is preserved through hiring: new
    researchers drawn to fill vacancies start from this fingerprint
    (with small noise), maintaining institutional continuity over time.
    """

    def __init__(self, lab_id: int, fingerprint: list[float]):
        self.lab_id      = lab_id
        self.fingerprint = fingerprint   # fixed domain skill profile — never changes
