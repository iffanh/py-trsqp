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

Because the inner loop reuses a single model fit once at the outer iterate
rather than repoising at every inner z_t, that model can be locally
uninformative at a given z -- e.g. its gradient is ~0 simply because the
outer sample set's points sit along a direction the constraints happen to be
structurally flat on at that particular x_k, not because x_k is actually
near-feasible. `_refit_models` guards against declaring "infeasible
stationary point" on a model like that: when the inner loop looks stuck
(gradient vanished, or its own radius has collapsed without making
progress), it re-samples a fresh, randomly-oriented simplex around the
current z and refits eq_models/ineq_models there before concluding z is
truly stationary, costing at most `max_repoise` extra rounds of (n+1)
simulations rather than one per inner iteration. This mirrors how the outer
loop handles the analogous situation for the objective/model-building step
(`IllPoisedModel` -> `repoise()`).

The Gauss-Newton direction (`_gauss_newton`) row-scales each violated
constraint by 1/||its own gradient|| before combining them: without this,
a single constraint whose gradient happens to be orders of magnitude
steeper than the others would dominate the combined search direction,
leaving the rest of the violated constraints essentially unaddressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .interpolation import QuadraticModel, fit_lagrange_set
from .filter import Filter
from .sampling import initial_sample_set
from . import trs
from ..exceptions import FailedSimulation, IllPoisedModel, PoisednessIsZeroException


@dataclass
class RestoreResult:
    success: bool
    x: Optional[np.ndarray]
    f: Optional[float]
    vartheta: Optional[float]
    n_inner_iters: int
    stationary: bool = False
    n_simulation_failures: int = 0


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
    """Row-scaled (diagonally preconditioned) Gauss-Newton direction, plus the
    true (unscaled) gradient/Hessian for judging progress.

    Each violated constraint's row is scaled by 1/||its own gradient|| before
    the rows are stacked into the normal equations (standard diagonal
    preconditioning for badly-scaled nonlinear least-squares). Without this,
    when violated constraints have gradients differing by orders of
    magnitude at the same z, the unscaled normal equations A^T@A / A^T@c are
    dominated entirely by whichever constraint happens to have the steepest
    gradient, leaving the rest of the violated constraints essentially
    unaddressed by the resulting direction.

    g_scaled/H_scaled are a linearization of a *reweighted* merit, not the
    true one, so they must only be used to pick a step direction -- `pred`
    (the model's predicted decrease, compared against the actual decrease via
    rho) and the "stationary" declaration need g_true/H_true instead, or they
    would judge progress against the wrong quantity.

    Returns (g_scaled, H_scaled, phi, g_true, H_true).
    """
    grads, vals = _violated_rows(eq_models, ineq_models, z)
    if not grads:
        z2 = np.zeros((n, n))
        return np.zeros(n), z2, 0.0, np.zeros(n), z2
    A = np.array(grads)
    c = np.array(vals)
    phi = 0.5 * float(c @ c)
    g_true = A.T @ c
    H_true = A.T @ A
    row_norms = np.linalg.norm(A, axis=1)
    w = 1.0 / np.maximum(row_norms, 1e-12)
    A_scaled = A * w[:, None]
    c_scaled = c * w
    return A_scaled.T @ c_scaled, A_scaled.T @ A_scaled, phi, g_true, H_true


def _rotated_sample_set(z: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Like `initial_sample_set`, but the simplex directions are rotated by a
    fresh random orthogonal matrix each call. `initial_sample_set` itself is
    a *fixed* deterministic simplex (same directions every time for a given
    n), so repoising with it around a z that hasn't moved would regenerate
    the exact same sample directions and therefore the exact same
    (possibly uninformative) model. Randomizing the directions gives each
    repoise attempt a chance at seeing the constraints from an angle the
    previous sample set didn't."""
    n = z.shape[0]
    if n == 1:
        Q = np.array([[1.0]])
    else:
        A = rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(A)
    unit = initial_sample_set(np.zeros(n), 1.0)[:, 1:]  # (n, n) unit directions
    return np.hstack([z[:, None], z[:, None] + radius * (Q @ unit)])


def _refit_models(z: np.ndarray, radius: float, evaluate_full: Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]],
                   n_eq: int, n_ineq: int, rng: np.random.Generator) -> Tuple[List[QuadraticModel], List[QuadraticModel]]:
    """Rebuilds eq_models/ineq_models on a fresh, randomly-oriented,
    well-poised simplex around `z`, mirroring DFTRFilter._build_models but
    scoped to Restore's own (constraint-only) models. Raises
    IllPoisedModel/PoisednessIsZeroException on the rare degenerate sample
    set, same as the outer loop's model build."""
    Y = _rotated_sample_set(z, radius, rng)
    F_eq = np.zeros((n_eq, Y.shape[1]))
    F_ineq = np.zeros((n_ineq, Y.shape[1]))
    for j in range(Y.shape[1]):
        _, eqs, ineqs = evaluate_full(Y[:, j])
        if n_eq:
            F_eq[:, j] = eqs
        if n_ineq:
            F_ineq[:, j] = ineqs
    lset = fit_lagrange_set(Y, radius=radius)
    eq_models = [lset.build_model(F_eq[i]) for i in range(n_eq)]
    ineq_models = [lset.build_model(F_ineq[i]) for i in range(n_ineq)]
    return eq_models, ineq_models


def restore(xk: np.ndarray,
            eq_models: List[QuadraticModel],
            ineq_models: List[QuadraticModel],
            evaluate: Callable[[np.ndarray], Tuple[float, float]],
            filt: Filter,
            f_k: float,
            vartheta_k: float,
            delta0: float,
            evaluate_full: Optional[Callable[[np.ndarray], Tuple[float, np.ndarray, np.ndarray]]] = None,
            n_eq: int = 0,
            n_ineq: int = 0,
            gamma0: float = 0.5,
            gamma1: float = 1.5,
            eta1: float = 0.1,
            eta2: float = 0.75,
            grad_tol: float = 1e-8,
            radius_floor: float = 1e-14,
            max_inner_iter: int = 60,
            max_repoise: int = 4,
            rng: Optional[np.random.Generator] = None) -> RestoreResult:
    """`evaluate(x) -> (f(x), vartheta(x))` is the *real* simulation, called
    once per inner iteration (on the trial point only). `evaluate_full(x) ->
    (f(x), eqs(x), ineqs(x))`, if given, lets the inner loop refit its
    (otherwise frozen) constraint models on a fresh, randomly-oriented
    sample set around the current z when it looks stuck, instead of
    immediately concluding z is an infeasible stationary point -- see the
    module docstring."""
    n = xk.shape[0]
    z = xk.copy()
    delta = delta0
    extra_pair = [(vartheta_k, f_k)]
    repoises_left = max_repoise if evaluate_full is not None else 0
    rng = rng if rng is not None else np.random.default_rng()
    t = 0
    n_failures = 0

    while t < max_inner_iter:
        # g/H are row-scaled (for picking a step direction not dominated by
        # whichever violated constraint has the steepest gradient); g_true/
        # H_true are the real, unscaled linearization, used wherever we judge
        # progress or stationarity instead -- see _gauss_newton's docstring.
        g, H, phi_z, g_true, H_true = _gauss_newton(eq_models, ineq_models, z, n)
        grad_norm = float(np.linalg.norm(g))
        true_grad_norm = float(np.linalg.norm(g_true))
        stuck = phi_z > 0 and true_grad_norm <= grad_tol

        if not stuck:
            if grad_norm == 0.0:
                r = np.zeros(n)
            else:
                r_gn = trs.solve_tr_subproblem(g, H, delta)
                r_sd = -delta * g / grad_norm
                if _merit_value(eq_models, ineq_models, z + r_sd) < _merit_value(eq_models, ineq_models, z + r_gn):
                    r = r_sd
                else:
                    r = r_gn

            # `evaluate` (DFTRFilter._evaluate) already turns a non-finite
            # cf/eqcs/ineqcs result into a large-but-finite sentinel and
            # counts it itself -- safe here since this value only ever feeds
            # the scalar filter check below, never a model gradient.
            f_trial, vartheta_trial = evaluate(z + r)

            if filt.is_acceptable(f_trial, vartheta_trial, extra=extra_pair):
                return RestoreResult(True, z + r, f_trial, vartheta_trial, t + 1,
                                      n_simulation_failures=n_failures)

            phi_trial = _merit_value(eq_models, ineq_models, z + r)
            pred = phi_z - (phi_z + g_true @ r + 0.5 * r @ H_true @ r)
            pred = max(pred, 1e-16)
            rho = (phi_z - phi_trial) / pred

            if rho >= eta1:
                z = z + r
                if rho >= eta2:
                    delta = gamma1 * delta
            else:
                delta = gamma0 * delta

            t += 1
            stuck = delta < radius_floor
            if not stuck:
                continue

        # Stuck: either the frozen model's gradient vanished at z, or its own
        # radius collapsed to the floor without ever making progress. Before
        # concluding z is really an infeasible stationary point, try refitting
        # the constraint models on a fresh simplex around z -- the current
        # models were fit once, back at the outer iterate, and may simply be
        # uninformative this far from there (see module docstring).
        if repoises_left <= 0 or evaluate_full is None:
            return RestoreResult(False, z, None, None, t, stationary=(phi_z > 0 and true_grad_norm <= grad_tol),
                                  n_simulation_failures=n_failures)

        repoises_left -= 1
        # Use the outer iteration's own trusted radius, not restore()'s own
        # `delta` -- delta can grow well past the scale the constraint models
        # were ever validated at (repeated gamma1 growth off a single
        # near-noise-level "successful" step), and sampling a fresh simplex
        # that wide overshoots into the same kind of badly-conditioned
        # territory this refit exists to escape.
        fresh_radius = max(delta0, 10 * radius_floor)
        # ineq_models[n_ineq:] are the exact (not fitted) bound-constraint
        # models DFTRFilter._bound_ineq_models built -- affine identities
        # like x_i - lb_i, so they're already exact at *any* z and never go
        # stale; only the first n_ineq (fitted from real cf/ineqcs values)
        # need refitting.
        bound_models = ineq_models[n_ineq:]
        try:
            eq_models, fitted_ineq = _refit_models(z, fresh_radius, evaluate_full, n_eq, n_ineq, rng)
        except FailedSimulation:
            # A point in the fresh simplex itself was NaN/Inf (e.g. the same
            # domain restriction the outer loop can hit) -- fabricating a
            # value here would corrupt the refit exactly like the frozen
            # model this refit exists to replace, so treat it as just another
            # failed attempt: loop back and, if budget remains, try again
            # with a newly-rotated simplex rather than giving up immediately.
            n_failures += 1
            continue
        except (IllPoisedModel, PoisednessIsZeroException):
            # Rare with a freshly-randomized simplex; also just worth another
            # rotation rather than surrendering the rest of the repoise budget.
            continue
        ineq_models = fitted_ineq + bound_models
        delta = fresh_radius
        t += n + 1  # charge the repoise's simplex simulations against the inner budget

    return RestoreResult(False, z, None, None, t, stationary=False, n_simulation_failures=n_failures)
