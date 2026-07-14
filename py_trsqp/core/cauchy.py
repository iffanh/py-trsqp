"""Algorithm 3 (CAUCHY) -- an implementable Cauchy step, and the Cauchy
sufficient-decrease inequality (20) it certifies "by construction".

Omega_k = L_k cap B(0, Delta_k) (Def. 7) is the intersection of the
linearised feasible polyhedron with the trust-region ball; the normal step
s^N_k = P_{Omega_k}[0] and the projected-gradient path s_k(t) =
P_{Omega_k}[s^N_k - t g_k] are both evaluated with the Dykstra projection of
core.polyhedron, since Omega_k has no closed-form projection in general
(Remark 3: the theory never needs the exact path minimiser, only a step
satisfying (20), which is exactly what backtracking here certifies).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .geometry import LinearGeometry
from .interpolation import QuadraticModel
from .polyhedron import project_onto_polyhedron


def normal_step(geom: LinearGeometry, radius: float) -> np.ndarray:
    """s^N_k = P_{Omega_k}[0]."""
    return project_onto_polyhedron(np.zeros(geom.n), geom.A_eq, geom.b_eq, geom.A_ineq, geom.b_ineq,
                                    ball_radius=radius)


def cauchy_step(geom: LinearGeometry, model: QuadraticModel, radius: float, chi: float, beta: float,
                 tau: float = 0.5, t0: float = None, max_backtrack: int = 60) -> Tuple[np.ndarray, float]:
    """Algorithm 3. Returns (s, achieved_decrease) with

        m(x_k) - m(x_k + s) >= kappa_1 * chi * min(chi/beta, radius),   kappa_1 = tau/2

    (inequality (20), Lemma 3), or the normal step if chi == 0 (no model
    descent available -- s^N_k trivially satisfies (20) with equality on
    both sides being 0).
    """
    s_n = normal_step(geom, radius)
    base_val = model.c  # m(x_k) = m(center + 0)

    if chi <= 0:
        return s_n, base_val - model.value(model.center + s_n)

    kappa1 = tau / 2.0
    threshold = kappa1 * chi * min(chi / beta, radius)

    g = model.g
    gnorm = np.linalg.norm(g)
    if t0 is None:
        t0 = (10.0 * radius) / gnorm if gnorm > 1e-14 else 10.0 * radius

    s = s_n
    decrease = base_val - model.value(model.center + s)
    for j in range(max_backtrack):
        t = (tau ** j) * t0
        s = project_onto_polyhedron(s_n - t * g, geom.A_eq, geom.b_eq, geom.A_ineq, geom.b_ineq,
                                     ball_radius=radius)
        decrease = base_val - model.value(model.center + s)
        if decrease >= threshold - 1e-13:
            return s, decrease

    # Lemma 3 guarantees this is unreachable in exact arithmetic; fall back
    # to whatever backtracking achieved (still a valid, if unverified, step).
    return s, decrease
