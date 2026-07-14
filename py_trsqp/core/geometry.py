"""Linearised feasible sets, consistency/compatibility, and the criticality
measure chi -- Definitions 2-5 of the paper.

L_k = { d : c~_i,k(x_k) + A_i,k^T d = 0 (i in E), c~_i,k(x_k) + A_i,k^T d >= 0 (i in I) }

is a polyhedron in *step space* (d is a displacement from x_k). Three
questions get asked of it, each with its own exact numeric test here instead
of the informal/heuristic checks the old code made no attempt to compute at
all:

 - consistency (Def. 3, scale-free): L_k cap B_infty != empty -- an LP
   feasibility check (box bounds [-1,1]^n).
 - compatibility (Def. 4): L_k cap B(0, Delta_k) != empty, which the paper
   notes is *equivalent* to dist(0, L_k) <= Delta_k -- so this is answered by
   projecting the origin onto L_k (core.polyhedron) and comparing distances,
   no ball-constrained feasibility solve needed.
 - the criticality measure chi~_k = |min_{d in L_k cap B_infty} g_k^T d|
   (Def. 5) -- an LP (linear objective over the same box-constrained
   polyhedron), solved with scipy's HiGHS backend.

Also provides the practical Assumption-4 (H4) monitor of Remark 13: the
smallest singular value of the active constraint gradients stacked with the
coordinate rows, reported as a run-time diagnostic (not used to gate control
flow -- H4 is a standing assumption, not something the algorithm enforces).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import linprog

from .interpolation import QuadraticModel
from .polyhedron import project_onto_polyhedron


@dataclass
class LinearGeometry:
    n: int
    A_eq: np.ndarray
    b_eq: np.ndarray
    A_ineq: np.ndarray
    b_ineq: np.ndarray


def build_geometry(n: int, eq_models: List[QuadraticModel], ineq_models: List[QuadraticModel]) -> LinearGeometry:
    if eq_models:
        A_eq = np.array([m.g for m in eq_models])
        b_eq = np.array([-m.c for m in eq_models])
    else:
        A_eq = np.zeros((0, n))
        b_eq = np.zeros(0)

    if ineq_models:
        A_ineq = np.array([m.g for m in ineq_models])
        b_ineq = np.array([-m.c for m in ineq_models])
    else:
        A_ineq = np.zeros((0, n))
        b_ineq = np.zeros(0)

    return LinearGeometry(n=n, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq)


def _linprog_feasible(geom: LinearGeometry, lo, hi) -> bool:
    n = geom.n
    A_ub = -geom.A_ineq if geom.A_ineq.shape[0] else None
    b_ub = -geom.b_ineq if geom.A_ineq.shape[0] else None
    A_eq = geom.A_eq if geom.A_eq.shape[0] else None
    b_eq = geom.b_eq if geom.A_eq.shape[0] else None
    bounds = [(lo, hi)] * n
    res = linprog(np.zeros(n), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return bool(res.success)


def is_consistent(geom: LinearGeometry) -> bool:
    """L_k cap B_infty != empty (Def. 3)."""
    return _linprog_feasible(geom, -1.0, 1.0)


@dataclass
class Compatibility:
    is_compatible: bool
    dist0: float
    d0: Optional[np.ndarray]


def compatibility(geom: LinearGeometry, radius: float) -> Compatibility:
    """L_k cap B(0, radius) != empty, via the equivalence dist(0,L_k) <= radius (Def. 4)."""
    if not _linprog_feasible(geom, None, None):
        return Compatibility(False, np.inf, None)
    d0 = project_onto_polyhedron(np.zeros(geom.n), geom.A_eq, geom.b_eq, geom.A_ineq, geom.b_ineq)
    dist0 = float(np.linalg.norm(d0))
    return Compatibility(dist0 <= radius + 1e-9, dist0, d0)


def criticality_measure(geom: LinearGeometry, g: np.ndarray) -> float:
    """chi~_k = |min_{d in L_k cap B_infty} g_k^T d| (Def. 5). Caller must
    already know the iteration is consistent."""
    n = geom.n
    A_ub = -geom.A_ineq if geom.A_ineq.shape[0] else None
    b_ub = -geom.b_ineq if geom.A_ineq.shape[0] else None
    A_eq = geom.A_eq if geom.A_eq.shape[0] else None
    b_eq = geom.b_eq if geom.A_eq.shape[0] else None
    bounds = [(-1.0, 1.0)] * n
    res = linprog(g, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return 0.0
    return abs(float(res.fun))


def violation(eq_values: np.ndarray, ineq_values: np.ndarray) -> float:
    """vartheta(x) = max(0, max_i |c_i(x)| (i in E), max_i -c_i(x) (i in I)) (Eq. 5)."""
    v = 0.0
    if eq_values is not None and len(eq_values):
        v = max(v, float(np.max(np.abs(eq_values))))
    if ineq_values is not None and len(ineq_values):
        v = max(v, float(np.max(np.maximum(0.0, -np.asarray(ineq_values)))))
    return v


def regularity_margin(geom: LinearGeometry, tol_active: float = 1e-6) -> float:
    """Smallest singular value of the active constraint gradients stacked
    with the coordinate rows (Remark 13's practical H4 monitor)."""
    rows = []
    if geom.A_eq.shape[0]:
        rows.append(geom.A_eq)
    if geom.A_ineq.shape[0]:
        active = np.abs(geom.b_ineq) <= tol_active
        if np.any(active):
            rows.append(geom.A_ineq[active])
    rows.append(np.eye(geom.n))
    M = np.vstack(rows)
    svals = np.linalg.svd(M, compute_uv=False)
    return float(svals[-1])
