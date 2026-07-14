"""TRQCQP(x_k, Delta_k) (Eq. 6) and the Cauchy-domination safeguard (Def. 8).

TRQCQP is solved by IPOPT (CasADi) exactly as the paper specifies -- "solved
by an interior-point method, which returns a local KKT point" -- built
directly from the numeric (c, g, H) model coefficients (no symbolic
simplification, unlike the old ``utils/trqp.py`` which rebuilt/normalized a
full symbolic NLP from scratch every call and had no fallback when IPOPT
returned a merely-local point).

Because IPOPT gives no global-optimality or even reliable local-KKT
guarantee on this nonconvex QCQP, the required sufficient decrease is never
assumed from the solver: Definition 8 compares the IPOPT candidate against
the Cauchy step (core.cauchy) *on the model*, at zero extra simulation cost
(Remark 5), and keeps whichever is feasible-and-better. Lemma 10 guarantees
the Cauchy step always qualifies, so this never fails to produce a step.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import casadi as ca

from .interpolation import QuadraticModel
from .geometry import violation


def _quad_expr(s, model: QuadraticModel):
    return model.c + ca.dot(model.g, s) + 0.5 * ca.dot(s, ca.mtimes(model.H, s))


def solve_trqcqp(model_cf: QuadraticModel, eq_models: List[QuadraticModel], ineq_models: List[QuadraticModel],
                  radius: float, warm_start: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], bool]:
    n = model_cf.g.shape[0]
    s = ca.SX.sym("s", n)

    f_expr = _quad_expr(s, model_cf)

    g_exprs = []
    lbg: List[float] = []
    ubg: List[float] = []
    for m in eq_models:
        g_exprs.append(_quad_expr(s, m))
        lbg.append(0.0)
        ubg.append(0.0)
    for m in ineq_models:
        g_exprs.append(_quad_expr(s, m))
        lbg.append(0.0)
        ubg.append(float(ca.inf))
    g_exprs.append(ca.dot(s, s))  # Euclidean trust region ||s|| <= radius
    lbg.append(0.0)
    ubg.append(radius ** 2)

    nlp = {"x": s, "f": f_expr, "g": ca.vertcat(*g_exprs)}
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
    try:
        solver = ca.nlpsol("trqcqp", "ipopt", nlp, opts)
    except Exception:
        return None, False

    x0 = warm_start if warm_start is not None else np.zeros(n)
    try:
        sol = solver(x0=x0, lbg=lbg, ubg=ubg)
        ok = bool(solver.stats().get("success", False))
        s_val = np.asarray(sol["x"]).reshape(-1)
        return s_val, ok
    except Exception:
        return None, False


def model_violation(eq_models: List[QuadraticModel], ineq_models: List[QuadraticModel], x: np.ndarray) -> float:
    eq_vals = np.array([m.value(x) for m in eq_models]) if eq_models else np.zeros(0)
    ineq_vals = np.array([m.value(x) for m in ineq_models]) if ineq_models else np.zeros(0)
    return violation(eq_vals, ineq_vals)


def select_step(model_cf: QuadraticModel, eq_models: List[QuadraticModel], ineq_models: List[QuadraticModel],
                 cauchy_s: np.ndarray, ipopt_s: Optional[np.ndarray], ipopt_ok: bool,
                 radius: float, c0: float) -> np.ndarray:
    """Def. 8's admissible step set S_k and the argmin over it. Lemma 10
    guarantees cauchy_s is always admissible, so a candidate always exists."""
    candidates = []

    if np.linalg.norm(cauchy_s) <= radius + 1e-8:
        v = model_violation(eq_models, ineq_models, model_cf.center + cauchy_s)
        if v <= c0 * radius ** 2 + 1e-9:
            candidates.append(cauchy_s)

    if ipopt_ok and ipopt_s is not None and np.linalg.norm(ipopt_s) <= radius + 1e-6:
        v = model_violation(eq_models, ineq_models, model_cf.center + ipopt_s)
        if v <= c0 * radius ** 2 + 1e-9:
            candidates.append(ipopt_s)

    if not candidates:
        candidates = [cauchy_s]

    values = [model_cf.value(model_cf.center + s) for s in candidates]
    return candidates[int(np.argmin(values))]
