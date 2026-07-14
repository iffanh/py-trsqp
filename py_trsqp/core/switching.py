"""The switching condition (Def. 9, Fletcher & Leyffer 2002).

Classifies a *compatible* iteration as optimality-driven (f-iteration, the
ratio test applies) or feasibility-driven (h-iteration, the ratio test is
waived and acceptance is by the filter alone). Without this, an iteration
with small chi~_k (little descent available) but large vartheta~_k would
fail the ratio test and get its radius needlessly shrunk even though the
iterate is far from critical -- Remark 6 of the paper.
"""

from __future__ import annotations

ITERATION_F = "f"
ITERATION_H = "h"
ITERATION_RESTORATION = "restoration"


def classify_iteration(is_compatible: bool, delta_m: float, vartheta_tilde: float,
                        kappa_theta: float, s_theta: float) -> str:
    if not is_compatible:
        return ITERATION_RESTORATION
    if delta_m >= kappa_theta * (vartheta_tilde ** s_theta):
        return ITERATION_F
    return ITERATION_H
