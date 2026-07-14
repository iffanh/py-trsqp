"""Algorithm 4 (RESTORE) -- the restoration phase.

Runs whenever the outer iteration is incompatible (Lemma 2(b): the model
violation is large relative to the radius). It is a derivative-free
trust-region method in its own right, minimizing the smooth merit
phi(x) = 0.5*||c-bar(x)||_2^2 (c-bar stacking the currently-violated
constraints), with its own radius, terminating either at filter acceptance
(success) or at a point stationary for phi with vartheta > 0 (Lemma 5 /
Theorem 1 case (a): an infeasible stationary point).

Simplification vs. the literal Algorithm 4: rather than repoising a fresh
Lambda_th-poised sample set around every inner iterate z_t (its line 3), this
reuses the quadratic constraint models already fitted at the outer iteration
-- they are genuine quadratic *functions* of x, not just values at x_k, so
they stay evaluable at any nearby z_t. Restoration's own step is then a
trust-region step on their Gauss-Newton merit model (g = A^T c-bar,
H = A^T A, the standard least-squares trust-region linearisation), solved
exactly via the same TRS core.trs uses, compared against a plain
steepest-descent candidate the same way Definition 8 compares TRQCQP against
the Cauchy step. This keeps Restore's simulation cost to exactly one
evaluation per inner iteration (matching Remark 11) while preserving the
part of Algorithm 4 the convergence proof actually leans on: its own
shrinking radius, a dominated sufficient-decrease step, filter-acceptance as
the success exit, and gradient-stationarity as the failure exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .interpolation import QuadraticModel
from .filter import Filter
from . import trs


@dataclass
class RestoreResult:
    success: bool
    x: Optional[np.ndarray]
    f: Optional[float]
    vartheta: Optional[float]
    n_inner_iters: int
    stationary: bool = False


def _violated_rows(eq_models: List[QuadraticModel], ineq_models: List[QuadraticModel], z: np.ndarray):
    grads = []
    vals = []
    for m in eq_models:
        vals.append(m.value(z))
        grads.append(m.gradient(z))
    for m in ineq_models:
        val = m.value(z)
        if val < 0.0:
            vals.append(val)
            grads.append(m.gradient(z))
    return grads, vals


def _merit_value(eq_models: List[QuadraticModel], ineq_models: List[QuadraticModel], z: np.ndarray) -> float:
    total = 0.0
    for m in eq_models:
        total += m.value(z) ** 2
    for m in ineq_models:
        val = m.value(z)
        if val < 0.0:
            total += val ** 2
    return 0.5 * total


def _gauss_newton(eq_models: List[QuadraticModel], ineq_models: List[QuadraticModel], z: np.ndarray, n: int):
    grads, vals = _violated_rows(eq_models, ineq_models, z)
    if not grads:
        return np.zeros(n), np.zeros((n, n)), 0.0
    A = np.array(grads)
    c = np.array(vals)
    return A.T @ c, A.T @ A, 0.5 * float(c @ c)


def restore(xk: np.ndarray,
            eq_models: List[QuadraticModel],
            ineq_models: List[QuadraticModel],
            evaluate: Callable[[np.ndarray], Tuple[float, float]],
            filt: Filter,
            f_k: float,
            vartheta_k: float,
            delta0: float,
            gamma0: float = 0.5,
            gamma1: float = 1.5,
            eta1: float = 0.1,
            eta2: float = 0.75,
            grad_tol: float = 1e-8,
            radius_floor: float = 1e-14,
            max_inner_iter: int = 60) -> RestoreResult:
    """`evaluate(x) -> (f(x), vartheta(x))` is the *real* simulation, called
    once per inner iteration (on the trial point only)."""
    n = xk.shape[0]
    z = xk.copy()
    delta = delta0
    extra_pair = [(vartheta_k, f_k)]

    for t in range(max_inner_iter):
        g, H, phi_z = _gauss_newton(eq_models, ineq_models, z, n)
        grad_norm = float(np.linalg.norm(g))

        if phi_z > 0 and grad_norm <= grad_tol:
            return RestoreResult(False, z, None, None, t, stationary=True)

        if grad_norm == 0.0:
            r = np.zeros(n)
        else:
            r_gn = trs.solve_tr_subproblem(g, H, delta)
            r_sd = -delta * g / grad_norm
            if _merit_value(eq_models, ineq_models, z + r_sd) < _merit_value(eq_models, ineq_models, z + r_gn):
                r = r_sd
            else:
                r = r_gn

        f_trial, vartheta_trial = evaluate(z + r)

        if filt.is_acceptable(f_trial, vartheta_trial, extra=extra_pair):
            return RestoreResult(True, z + r, f_trial, vartheta_trial, t + 1)

        phi_trial = _merit_value(eq_models, ineq_models, z + r)
        pred = phi_z - (phi_z + g @ r + 0.5 * r @ H @ r)
        pred = max(pred, 1e-16)
        rho = (phi_z - phi_trial) / pred

        if rho >= eta1:
            z = z + r
            if rho >= eta2:
                delta = gamma1 * delta
        else:
            delta = gamma0 * delta

        if delta < radius_floor:
            return RestoreResult(False, z, None, None, t + 1, stationary=False)

    return RestoreResult(False, z, None, None, max_inner_iter, stationary=False)
