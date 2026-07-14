"""Euclidean projection onto a polyhedron, optionally intersected with a ball.

This is the one piece of "constrained geometry" machinery shared by:
 - core.geometry: dist(0, L_k) for the compatibility test (Def. 4) --
   project the origin onto L_k with no ball.
 - core.cauchy: the normal step s^N_k = P_{Omega_k}[0] and the projected
   gradient path s_k(t) = P_{Omega_k}[s^N_k - t g_k], Omega_k = L_k cap
   B(0; Delta_k) (Def. 7).
 - core.restore: the analogous normal/Cauchy steps for the merit function.

Exact projection onto an intersection of halfspaces and a ball has no closed
form in general, so this uses Dykstra's alternating-projection algorithm:
cyclically project onto each set (the linear-equality subspace once in
closed form, then each inequality halfspace, then the ball), each time
correcting with the increment left over from that set's previous projection.
Dykstra's algorithm converges to the exact Euclidean projection onto the
intersection whenever it is nonempty (Boyle & Dykstra 1986); the paper's own
Remark 3 explicitly does not require the exact projection or path
minimizer, only a step satisfying the Cauchy-decrease inequality (20), so an
iterative-but-converging projection is exactly what the theory asks for.
"""

from __future__ import annotations

import numpy as np


def _project_affine_factory(A_eq: np.ndarray, b_eq: np.ndarray):
    A_pinv = np.linalg.pinv(A_eq)

    def project(x: np.ndarray) -> np.ndarray:
        return x - A_pinv @ (A_eq @ x - b_eq)

    return project


def _project_halfspace_factory(a: np.ndarray, b: float):
    denom = float(a @ a)

    def project(x: np.ndarray) -> np.ndarray:
        val = float(a @ x) - b
        if val >= 0 or denom < 1e-300:
            return x
        return x - (val / denom) * a

    return project


def _project_ball_factory(center: np.ndarray, radius: float):
    def project(x: np.ndarray) -> np.ndarray:
        d = x - center
        norm = np.linalg.norm(d)
        if norm <= radius:
            return x
        return center + (radius / norm) * d

    return project


def build_projectors(n: int, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None,
                      ball_center=None, ball_radius=None):
    projectors = []
    if A_eq is not None and A_eq.shape[0] > 0:
        projectors.append(_project_affine_factory(A_eq, b_eq))
    if A_ineq is not None and A_ineq.shape[0] > 0:
        for a, b in zip(A_ineq, b_ineq):
            projectors.append(_project_halfspace_factory(a, float(b)))
    if ball_radius is not None:
        center = ball_center if ball_center is not None else np.zeros(n)
        projectors.append(_project_ball_factory(center, ball_radius))
    return projectors


def project_onto_polyhedron(x0: np.ndarray, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None,
                             ball_center=None, ball_radius=None,
                             max_iter: int = 500, tol: float = 1e-11) -> np.ndarray:
    """Dykstra projection of x0 onto {A_eq d = b_eq, A_ineq d >= b_ineq} cap B(ball_center, ball_radius).

    Any of the equality/inequality/ball pieces may be omitted (None). Assumes
    the intersection is nonempty; use core.geometry's feasibility checks
    beforehand if that is not already known.
    """
    n = x0.shape[0]
    projectors = build_projectors(n, A_eq, b_eq, A_ineq, b_ineq, ball_center, ball_radius)
    if not projectors:
        return x0.copy()

    x = x0.copy()
    increments = [np.zeros(n) for _ in projectors]
    for _ in range(max_iter):
        x_prev_cycle = x.copy()
        for i, proj in enumerate(projectors):
            y = proj(x + increments[i])
            increments[i] = x + increments[i] - y
            x = y
        if np.linalg.norm(x - x_prev_cycle) < tol * max(1.0, np.linalg.norm(x_prev_cycle)):
            break
    return x
