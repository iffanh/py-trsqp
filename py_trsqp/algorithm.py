"""Algorithm 5 (DFTR-Filter) -- the complete method.

Ties together every core.* module into the loop of Fig. 3 / Sec. 4.8:
Step 0 (evaluate + build models) -> Step 1 (Crit) -> Step 2 (step, with
Cauchy domination, or Restore) -> Step 3 (evaluate + classify + accept) ->
Step 4 (stopping tests). All bookkeeping happens in *normalized* coordinates
(core.problem.Problem.normalize), matching the paper's normalisation
convention (Eq. 3) so every subproblem is solved on a numerically
well-scaled domain regardless of the user's native units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .core.problem import Problem
from .core.interpolation import fit_lagrange_set, QuadraticModel
from .core import geometry
from .core import criticality
from .core import switching
from .core import cauchy
from .core import trqcqp
from .core import restore as restore_mod
from .core.filter import Filter
from .core.repoise import repoise
from .core.sampling import initial_sample_set
from .exceptions import IllPoisedModel, PoisednessIsZeroException, FailedSimulation
from ._blas import cap_blas_threads


@dataclass
class Constants:
    eta2: float = 0.75
    gamma0: float = 0.5
    gamma1: float = 2.0
    gamma_vartheta: float = 1e-4
    kappa_theta: float = 1e-3
    s_theta: float = 1.1
    mu: float = 100.0
    omega: float = 0.5
    lambda_th: float = 10.0
    epsilon_c: float = 1e-3
    tau: float = 0.5
    init_radius: float = 1.0
    max_radius: float = 2.0
    min_radius: float = 1e-8
    epsilon_tol: float = 1e-6
    max_points: Optional[int] = None


@dataclass
class IterationRecord:
    k: int
    x: np.ndarray
    f: Optional[float]
    vartheta: Optional[float]
    radius: float
    it_type: str
    n_evals: int
    psi_tilde: Optional[float] = None
    chi_tilde: Optional[float] = None
    kappa_licq: Optional[float] = None


class DFTRFilter:
    """The DFTR-Filter method (Algorithm 5) for

        min f(x)  s.t.  c_i(x) = 0 (i in E), c_i(x) >= 0 (i in I), lb <= x <= ub.
    """

    def __init__(self, x0, cf: Callable, ub, lb,
                 eqcs: Optional[List[Callable]] = None, ineqcs: Optional[List[Callable]] = None,
                 constants: Optional[dict] = None):
        cap_blas_threads()
        self.problem = Problem(x0, cf, ub, lb, eqcs, ineqcs)
        c = Constants(**(constants or {}))
        if c.max_points is None:
            c.max_points = min(2 * self.problem.n + 1, (self.problem.n + 1) * (self.problem.n + 2) // 2)
        self.c = c
        self.filter = Filter(c.gamma_vartheta)
        self.records: List[IterationRecord] = []
        self.termination_status: Optional[str] = None

        self.radius = c.init_radius
        self.Y = initial_sample_set(self.problem.x0n, self.radius)
        self._cache = {}
        self.best: Optional[IterationRecord] = None

        # Tracks the worst (largest-magnitude) f and vartheta seen across every
        # successful evaluation, and how many non-finite results have been
        # recovered from. Used to scale _failure_sentinel() to the problem's
        # own units instead of a fixed constant, and surfaced in
        # termination_status so a run's outcome shows whether it had to
        # recover from any non-finite cf/eqcs/ineqcs results.
        self._worst_f = 0.0
        self._worst_vartheta = 0.0
        self._simulation_failures = 0

    # -- evaluation / caching -------------------------------------------------
    #
    # A single cache keyed by the (normalized) point's bytes backs every
    # evaluation path -- Step 0's lazy model-building, the ordinary step's
    # trial-point evaluation, and Restore's calls all funnel through
    # `_get_full`, so a point already evaluated as part of the sample set
    # (or vice versa) is never re-run through the user's black box, matching
    # Remark 11's "one simulation per iteration" accounting.
    #
    # `_get_full` *raises* FailedSimulation when cf/eqcs/ineqcs returns a
    # non-finite value. Callers that use the result to *fit* a model
    # (_build_models, restore.py's _refit_models) need that exception to
    # propagate rather than receive a fabricated number: Lagrange
    # interpolation coefficients are a linear function of the sample values,
    # so one synthetic value would corrupt the whole fitted gradient, not
    # just mark that one point as bad. Those callers repoise and retry
    # instead (see optimize()'s FailedSimulation branch and restore()'s
    # repoise-on-stall branch).
    #
    # `_evaluate` wraps `_get_full` for the other kind of call: single
    # trial-point evaluations (the ordinary step's candidate, Restore's
    # candidate) that only ever feed a scalar filter accept/reject decision,
    # never a gradient. There a large-but-finite sentinel is safe to return
    # instead of raising -- the filter naturally rejects it, and the existing
    # radius-shrink/repoise machinery for a rejected step takes it from there.

    def _get_full(self, xn: np.ndarray):
        key = xn.tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        f, eqs, ineqs = self.problem.evaluate(xn)
        self._cache[key] = (f, eqs, ineqs)
        self._worst_f = max(self._worst_f, abs(f))
        self._worst_vartheta = max(self._worst_vartheta, float(geometry.violation(eqs, ineqs)))
        return f, eqs, ineqs

    def _failure_sentinel(self):
        """A (f, vartheta) pair guaranteed worse than anything legitimately
        observed so far, scaled to the problem rather than a fixed constant
        -- used only where the caller just needs the filter to reject the
        point, never where a value feeds model-fitting."""
        penalty = 2.0 * max(self._worst_f, self._worst_vartheta, 1.0)
        return penalty, penalty

    def _evaluate(self, xn: np.ndarray):
        try:
            f, eqs, ineqs = self._get_full(xn)
        except FailedSimulation:
            self._simulation_failures += 1
            return self._failure_sentinel()
        return f, geometry.violation(eqs, ineqs)

    def _build_models(self, Y: np.ndarray):
        F_cf = np.empty(Y.shape[1])
        F_eq = np.zeros((self.problem.n_eq, Y.shape[1]))
        F_ineq = np.zeros((self.problem.n_ineq, Y.shape[1]))
        for j in range(Y.shape[1]):
            f, eqs, ineqs = self._get_full(Y[:, j])
            F_cf[j] = f
            if self.problem.n_eq:
                F_eq[:, j] = eqs
            if self.problem.n_ineq:
                F_ineq[:, j] = ineqs

        lset = fit_lagrange_set(Y, radius=self.radius)
        model_cf = lset.build_model(F_cf)
        eq_models = [lset.build_model(F_eq[i]) for i in range(self.problem.n_eq)]
        ineq_models = [lset.build_model(F_ineq[i]) for i in range(self.problem.n_ineq)]
        return lset, model_cf, eq_models, ineq_models

    # -- sample-set capacity management --------------------------------------

    def _prune_far_points(self, Y: np.ndarray, factor: float = 4.0) -> np.ndarray:
        """Drop sample points that have fallen far outside the current trust
        region (relative to `factor * radius`). Without this, points kept
        from earlier iterations -- when the incumbent has since moved and
        the radius changed -- can sit well outside B(x_k, Delta_k); a
        minimum-Frobenius-norm quadratic fit still interpolates them
        exactly, but the resulting gradient/Hessian describe the surface
        through those distant points rather than local behaviour near x_k,
        producing wild, useless steps. CSV09-style methods keep the sample
        set geometrically relevant to the current ball for exactly this
        reason; Crit/Repoise then only need to fix up poisedness, not
        relevance."""
        center = Y[:, 0]
        dists = np.linalg.norm(Y - center[:, None], axis=0)
        keep = dists <= factor * self.radius
        keep[0] = True
        min_points = self.problem.n + 1
        if np.sum(keep) < min_points:
            # Too few points are still relevant to the (shrunk) trust region.
            # Keeping the "closest of the far ones" regardless of distance
            # would silently pass poisedness checks while the fitted model
            # still describes the surface through stale, distant points --
            # exactly the bug this method exists to prevent. Rebuild a fresh
            # local simplex around the incumbent instead (a few new
            # simulations, spent deliberately, same as the old code's
            # "need_rebuild" path).
            return initial_sample_set(center, self.radius)
        return Y[:, keep]

    def _cap_sample_set(self, Y: np.ndarray) -> np.ndarray:
        max_points = self.c.max_points
        while Y.shape[1] > max_points:
            try:
                lset = fit_lagrange_set(Y, radius=self.radius)
                poised = lset.poisedness(radius=self.radius)
                order = np.argsort(-poised.values)
                idx = next((int(i) for i in order if i != 0), Y.shape[1] - 1)
            except (IllPoisedModel, PoisednessIsZeroException):
                idx = Y.shape[1] - 1
            Y = np.delete(Y, idx, axis=1)
        return Y

    # -- bound constraints, subsumed into I (Sec. 1: "Bound constraints are
    # subsumed into I") -------------------------------------------------------

    def _bound_ineq_models(self, center: np.ndarray) -> List[QuadraticModel]:
        """Exact (not fitted) linear models for lb <= x <= ub, in normalized
        coordinates: x_i - lb_i >= 0 and ub_i - x_i >= 0. These are folded
        into the same inequality-model list used for L_k, TRQCQP, and the
        violation measure everywhere else in the loop, so bounds are
        actually enforced by the optimizer rather than only used for
        reporting/normalization."""
        n = self.problem.n
        zero_H = np.zeros((n, n))
        models = []
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            models.append(QuadraticModel(c=float(center[i] - self.problem.lbn[i]), g=e.copy(),
                                          H=zero_H, center=center.copy()))
            models.append(QuadraticModel(c=float(self.problem.ubn[i] - center[i]), g=-e.copy(),
                                          H=zero_H, center=center.copy()))
        return models

    # -- curvature bound estimate (Assumption 3) -----------------------------

    @staticmethod
    def _kappa_bhm(models: List[QuadraticModel]) -> float:
        best = 0.0
        for m in models:
            if m.H.size:
                best = max(best, float(np.linalg.norm(m.H, ord=2)))
        return max(best, 1e-8)

    # -- main loop ------------------------------------------------------------

    def optimize(self, max_iter: int = 200, budget: Optional[int] = None) -> "DFTRFilter":
        budget = budget if budget is not None else 10 * max_iter * (self.problem.n + 1)
        p = self.problem
        Y = self.Y

        for k in range(max_iter):
            if self.radius < self.c.min_radius:
                self.termination_status = "Minimum radius reached"
                break
            if p.n_cf_evals >= budget:
                self.termination_status = "Budget exceeded"
                break

            Y = self._prune_far_points(Y)

            try:
                lset, model_cf, eq_models, ineq_models = self._build_models(Y)
            except IllPoisedModel:
                Y = repoise(Y, self.radius, self.c.lambda_th)
                self.radius *= self.c.gamma0
                continue
            except FailedSimulation:
                # A sample point returned a non-finite value -- e.g. a domain
                # restriction (log/sqrt) the box bounds don't encode. Handled
                # exactly like IllPoisedModel: repoise the set and shrink
                # toward the incumbent, rather than fitting a model from a
                # fabricated value. The `self.radius < self.c.min_radius`
                # check at the top of the loop is the circuit breaker -- if
                # every point this close to the incumbent is invalid, radius
                # shrinks until it trips and the run ends with "Minimum
                # radius reached" instead of raising.
                self._simulation_failures += 1
                Y = repoise(Y, self.radius, self.c.lambda_th)
                self.radius *= self.c.gamma0
                continue

            center = Y[:, 0]
            ineq_models = ineq_models + self._bound_ineq_models(center)
            eq_vals = np.array([m.c for m in eq_models])
            ineq_vals = np.array([m.c for m in ineq_models])
            vartheta_tilde = geometry.violation(eq_vals, ineq_vals)

            geom = geometry.build_geometry(p.n, eq_models, ineq_models)
            consistent = geometry.is_consistent(geom)
            if consistent:
                chi_tilde = geometry.criticality_measure(geom, model_cf.g)
                psi_tilde = chi_tilde + vartheta_tilde
            else:
                chi_tilde = None
                psi_tilde = vartheta_tilde

            try:
                poised = lset.poisedness(radius=self.radius)
                lambda_k = poised.max_poisedness()
            except PoisednessIsZeroException:
                lambda_k = np.inf

            kappa_licq = geometry.regularity_margin(geom) if (eq_models or ineq_models) else None

            # Step 1: Crit
            status = criticality.crit(psi_tilde, lambda_k, self.radius, self.c.mu, self.c.lambda_th, self.c.epsilon_c)
            if status == criticality.CRITICALITY:
                if lambda_k > self.c.lambda_th:
                    new_radius = self.radius
                else:
                    new_radius = self.c.omega * self.radius
                Y = repoise(Y, new_radius, self.c.lambda_th)
                self.radius = new_radius
                self._record(k, center, None, None, "criticality", psi_tilde, chi_tilde, kappa_licq)
                continue

            f_center, _ = self._evaluate(center)

            # Step 2 + 3
            compat = geometry.compatibility(geom, self.radius)
            if compat.is_compatible:
                Y, it_type, accepted_x = self._ordinary_step(
                    Y, lset, model_cf, eq_models, ineq_models, geom, center,
                    f_center, vartheta_tilde, chi_tilde,
                )
            else:
                Y, it_type, accepted_x = self._restoration_step(
                    Y, eq_models, ineq_models, center, f_center, vartheta_tilde,
                )
                if it_type == "stopped":
                    self._record(k, center, f_center, vartheta_tilde, it_type, psi_tilde, chi_tilde, kappa_licq)
                    self._update_best()
                    self.termination_status = "Infeasible stationary point (restoration failed)"
                    break

            f_rec, v_rec = self._evaluate(Y[:, 0])
            self._record(k, Y[:, 0], f_rec, v_rec, it_type, psi_tilde, chi_tilde, kappa_licq)
            self._update_best()

            # Step 4: stopping tests
            if psi_tilde <= self.c.epsilon_tol and lambda_k <= self.c.lambda_th:
                self.termination_status = "Approximate KKT point"
                break
            if k == max_iter - 1:
                self.termination_status = "Maximum iterations"

        if self.termination_status is None:
            self.termination_status = "Maximum iterations"
        if self._simulation_failures:
            self.termination_status += (
                f" (recovered from {self._simulation_failures} non-finite simulation result(s))"
            )
        self.Y = Y
        return self

    def _ordinary_step(self, Y, lset, model_cf, eq_models, ineq_models, geom, center,
                        f_center, vartheta_tilde, chi_tilde):
        beta = 1.0 + self._kappa_bhm([model_cf] + eq_models + ineq_models)
        chi = chi_tilde if chi_tilde is not None else 0.0

        s_cauchy, _ = cauchy.cauchy_step(geom, model_cf, self.radius, chi, beta, tau=self.c.tau)
        s_ipopt, ipopt_ok = trqcqp.solve_trqcqp(model_cf, eq_models, ineq_models, self.radius, warm_start=s_cauchy)
        c0 = 0.5 * self._kappa_bhm([model_cf] + eq_models + ineq_models)
        s = trqcqp.select_step(model_cf, eq_models, ineq_models, s_cauchy, s_ipopt, ipopt_ok, self.radius, c0)

        delta_m = model_cf.c - model_cf.value(center + s)
        x_next = center + s
        f_next, v_next = self._evaluate(x_next)

        it_type = switching.classify_iteration(True, delta_m, vartheta_tilde, self.c.kappa_theta, self.c.s_theta)

        if it_type == switching.ITERATION_F:
            rho = (f_center - f_next) / delta_m if delta_m > 1e-300 else -np.inf
            accept = self.filter.is_acceptable(f_next, v_next) and rho >= self.c.eta2
        else:
            accept = self.filter.is_acceptable(f_next, v_next)

        if accept:
            if it_type == switching.ITERATION_H:
                self.filter.add(f_next, v_next)
            Y_new = np.concatenate([x_next[:, None], Y], axis=1)
            Y_new = self._cap_sample_set(Y_new)
            step_norm = float(np.linalg.norm(s))
            self.radius = min(max(self.radius, self.c.gamma1 * step_norm), self.c.max_radius)
            return Y_new, it_type, x_next
        else:
            if lset.poisedness(radius=self.radius).max_poisedness() <= self.c.lambda_th:
                self.radius = self.c.gamma0 * self.radius
            else:
                Y = repoise(Y, self.radius, self.c.lambda_th)
            return Y, it_type + "-rejected", center

    def _restoration_step(self, Y, eq_models, ineq_models, center, f_center, vartheta_tilde):
        # `evaluate=self._evaluate` absorbs a non-finite result into a
        # sentinel internally (counted in self._simulation_failures) since
        # it's only ever a trial-point check against the filter, never a
        # model-fitting input. `evaluate_full=self._get_full` is the raw,
        # exception-raising path restore()'s own model refit needs (see
        # restore()'s docstring); any failures there come back via `result`
        # and are folded into the same counter below.
        result = restore_mod.restore(
            center, eq_models, ineq_models,
            evaluate=self._evaluate,
            filt=self.filter,
            f_k=f_center,
            vartheta_k=vartheta_tilde,
            delta0=self.radius,
            evaluate_full=self._get_full,
            n_eq=self.problem.n_eq,
            n_ineq=self.problem.n_ineq,
            gamma0=self.c.gamma0,
            gamma1=self.c.gamma1,
        )
        self._simulation_failures += result.n_simulation_failures
        if not result.success:
            return Y, "stopped", center

        self.filter.add(f_center, vartheta_tilde)
        Y_new = np.concatenate([result.x[:, None], Y], axis=1)
        Y_new = self._cap_sample_set(Y_new)
        step_norm = float(np.linalg.norm(result.x - center))
        self.radius = min(max(self.radius, self.c.gamma1 * step_norm), self.c.max_radius)
        return Y_new, "restoration", result.x

    def _record(self, k, xn, f, vartheta, it_type, psi_tilde, chi_tilde, kappa_licq):
        rec = IterationRecord(
            k=k, x=self.problem.denormalize(xn), f=f, vartheta=vartheta, radius=self.radius,
            it_type=it_type, n_evals=self.problem.n_cf_evals,
            psi_tilde=psi_tilde, chi_tilde=chi_tilde, kappa_licq=kappa_licq,
        )
        self.records.append(rec)

    def _update_best(self):
        rec = self.records[-1]
        if rec.f is None:
            return
        if self.best is None:
            self.best = rec
            return
        if (rec.vartheta or 0.0) < (self.best.vartheta or 0.0) - 1e-10:
            self.best = rec
        elif abs((rec.vartheta or 0.0) - (self.best.vartheta or 0.0)) <= 1e-10 and rec.f < self.best.f:
            self.best = rec
