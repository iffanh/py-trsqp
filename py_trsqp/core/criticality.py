"""Procedure CRIT and the criticality contract (Algorithm 2, Def. 6).

Crit asks whether the iterate *appears* near-critical (psi~_k small) while
the radius is still too large to certify it, in which case the radius
should shrink and the sample-set geometry should be repaired (Repoise)
*without moving and without evaluating the objective*. This whole mechanism
is entirely absent from the old ad-hoc algorithm, which had no criticality
measure at all -- meaning its radius/stopping heuristics had no relationship
to an actual first-order certificate.
"""

from __future__ import annotations

PASS = "pass"
CRITICALITY = "criticality"


def crit(psi_tilde: float, poisedness_value: float, radius: float,
         mu: float, lambda_th: float, epsilon_c: float) -> str:
    """Algorithm 2. Returns PASS or CRITICALITY."""
    if psi_tilde > epsilon_c:
        return PASS
    if poisedness_value <= lambda_th and radius <= mu * psi_tilde:
        return PASS
    return CRITICALITY
