"""Benchmark derivative-free solvers on CUTEst test problems via pycutest.

Requires a working CUTEst + SIFDecode + pycutest installation (ARCHDEFS,
SIFDECODE, CUTEST, MASTSIF environment variables set, `pycutest` importable).
See DEVELOPMENT.md, section "CUTEst benchmark setup", for how to build that
toolchain from scratch. Every entry point here degrades gracefully -
`cutest_available()` returns False and callers can skip - when the toolchain
isn't present, so this module is safe to import unconditionally.

Any subset of the following solvers can be run on the selected problems,
chosen with --solvers:

  - dftr        DFTR-Filter, the method implemented in this repository
  - cobyla      (scipy) linearizes constraints; native inequalities
  - cobyqa      (scipy) model-based successor to COBYLA; native bounds + eq/ineq
  - pybobyqa    (Py-BOBYQA) bound-constrained trust-region DFO
  - cma         (cma) CMA-ES evolutionary strategy
  - nelder-mead (scipy) simplex baseline

cobyla/cobyqa take the constraints natively (cobyla has no equality support,
so each equality is split into two inequalities). pybobyqa/cma/nelder-mead
have no nonlinear-constraint support, so on constrained problems they are
driven with a quadratic-penalty objective. Regardless of how a solver handles
constraints, its reported (f, vartheta) is always recomputed from the true
cf/eqcs/ineqcs at the point it returned - never trusted from a penalized
objective value - so the comparison stays fair.

Each solver's result is judged against a gradient-based SciPy reference
(SLSQP if constrained, else L-BFGS-B, using CUTEst's exact derivatives - the
derivative-free solvers never see them). A solver "succeeds" on a problem
when it is itself feasible and its objective is within --tol of the
reference objective. When the reference solve is itself non-finite (it can
hit the same NaN-producing domain restriction the black box has) the problem
is "not judged" for every solver rather than counted as a failure.

Each problem is run in its own subprocess with a wall-clock --timeout, so one
pathological problem can't hang or crash the whole sweep.

CLI usage:
    python benchmarks/cutest_benchmark.py --solvers dftr --n-max 20 --count 40
    python benchmarks/cutest_benchmark.py --solvers dftr cobyla cobyqa pybobyqa cma nelder-mead
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# py_trsqp isn't pip-installed in most dev setups (see DEVELOPMENT.md); make
# sure the repo root is importable regardless of how this file is invoked
# (as a script, as `python -m benchmarks.cutest_benchmark`, or imported by
# pytest). Forked subprocesses inherit this via copy-on-write.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

UNBOUNDED = 1e19
DEFAULT_BOUND = 50.0
DEFAULT_UNBOUND_SCALE = 2.0
DEFAULT_SUCCESS_TOL = 0.1   # relative gap to the reference objective counted as success
FEASIBLE_TOL = 1e-4         # vartheta at/below which a point is treated as feasible
DEFAULT_MAX_ITER = 200      # DFTR-Filter iteration cap
DEFAULT_MAX_NFEV = 500      # function-evaluation cap for the other solvers

# Small, fast, curated set used by tests/test_cutest_benchmark.py as a
# regression smoke test. Mixes unconstrained/bound problems with equality-
# and inequality-constrained ones, all cheap to decode/compile/solve.
# (AKIVA is deliberately excluded: its objective uses LOG() over a domain
# that isn't expressible as box bounds, so any bound-constrained DFO method
# hits NaNs there regardless of implementation - not a regression signal.)
SMOKE_PROBLEMS = [
    "ROSENBR",   # unconstrained, n=2
    "BEALE",     # unconstrained, n=2
    "BARD",      # unconstrained, n=3
    "BOX3",      # unconstrained, n=3
    "HS21",      # bound + 1 linear inequality, n=2
    "HS100",     # unconstrained-bounds, 4 inequalities, n=7
    "HS78",      # 3 equalities, n=5
]


def cutest_available() -> bool:
    """Whether pycutest and the CUTEst toolchain look usable in this process."""
    if not all(os.environ.get(v) for v in ("ARCHDEFS", "SIFDECODE", "CUTEST", "MASTSIF")):
        return False
    try:
        import pycutest  # noqa: F401
    except ImportError:
        return False
    return True


# -- problem translation --------------------------------------------------

def _build_dftr_inputs(p, bound: float = DEFAULT_BOUND, unbound_scale: float = DEFAULT_UNBOUND_SCALE):
    """Translate a pycutest problem into a solver-agnostic (cf, eqcs, ineqcs, lb, ub, x0).

    CUTEst represents every constraint as cl[i] <= c_i(x) <= cu[i]; this
    splits each row into separate equality (=0) and inequality (>=0)
    black-box callables depending on which bounds are finite.

    Variables the *problem itself* leaves unbounded (+-1e20) need a synthetic
    finite box -- DFTR-Filter requires one to build its normalized
    interpolation space -- but a single constant box for every problem is
    the wrong default: DFTR-Filter's initial trust-region radius spans half
    the box width (Problem.normalize maps x=ub to xn=+1), so an oversized box
    makes the very first interpolation sample span a region the function may
    be wildly nonlinear over. Each *unbounded* coordinate instead gets its
    own box sized off the problem's own starting point, capped at `bound` as
    a safety ceiling: half-width_i = min(bound, unbound_scale * max(|x0_i|, 1)).
    Variables the problem already bounds keep their real (finite) bl/bu untouched.
    """
    n = p.n
    bl = np.asarray(p.bl, dtype=float)
    bu = np.asarray(p.bu, dtype=float)
    x0_raw = np.asarray(p.x0, dtype=float)

    half_width = np.minimum(bound, unbound_scale * np.maximum(np.abs(x0_raw), 1.0))
    lb = np.where(bl <= -UNBOUNDED, x0_raw - half_width, bl)
    ub = np.where(bu >= UNBOUNDED, x0_raw + half_width, bu)
    x0 = np.clip(x0_raw, lb, ub)

    def cf(x):
        return float(p.obj(np.asarray(x, dtype=float)))

    eqcs, ineqcs = [], []
    m = p.m
    if m > 0:
        cl = np.asarray(p.cl, dtype=float)
        cu = np.asarray(p.cu, dtype=float)
        for i in range(m):
            lo, hi = cl[i], cu[i]
            if lo == hi:
                eqcs.append(lambda x, i=i, lo=lo: float(p.cons(np.asarray(x, dtype=float))[i]) - lo)
            else:
                if lo > -UNBOUNDED:
                    ineqcs.append(lambda x, i=i, lo=lo: float(p.cons(np.asarray(x, dtype=float))[i]) - lo)
                if hi < UNBOUNDED:
                    ineqcs.append(lambda x, i=i, hi=hi: hi - float(p.cons(np.asarray(x, dtype=float))[i]))

    return cf, eqcs, ineqcs, lb, ub, x0, n, m


def _reference_vartheta(eqcs, ineqcs, x) -> float:
    """Same violation convention DFTR-Filter itself uses (py_trsqp.core.geometry.violation):
    0 if feasible, else the largest constraint violation at x. Used to score
    every solver's returned point uniformly."""
    from py_trsqp.core import geometry

    eq_vals = np.array([c(x) for c in eqcs]) if eqcs else np.array([])
    ineq_vals = np.array([c(x) for c in ineqcs]) if ineqcs else np.array([])
    return float(geometry.violation(eq_vals, ineq_vals))


def _reference_solve(cf, eqcs, ineqcs, lb, ub, x0, p) -> Tuple[Optional[float], Optional[float]]:
    """Gradient-based SciPy reference (SLSQP if constrained, else L-BFGS-B).

    Uses CUTEst's exact derivatives for speed/reliability; the derivative-free
    solvers never see them. Used purely as an independent yardstick for
    "success".

    Returns (f_ref, vartheta_ref) -- the reference's own constraint violation
    at the point it stopped at, computed with the same convention as every
    solver, so a "better" f_ref that is actually infeasible (SLSQP can stop
    short of full feasibility) is visible rather than silently treated as
    ground truth.

    Returns (None, None) if the reference solver itself couldn't produce a
    finite objective (e.g. it wanders into the same NaN-producing domain
    restriction the black box has and aborts). Callers must treat None as
    "not judged", not as a failure.
    """
    from scipy.optimize import minimize

    def obj_and_grad(x):
        f, g = p.obj(np.asarray(x, dtype=float), gradient=True)
        return float(f), np.asarray(g, dtype=float)

    bounds = list(zip(lb.tolist(), ub.tolist()))

    if not eqcs and not ineqcs:
        res = minimize(obj_and_grad, x0, jac=True, bounds=bounds, method="L-BFGS-B")
        fun = float(res.fun)
        return (fun, 0.0) if np.isfinite(fun) else (None, None)

    constraints = []
    if p.m > 0:
        cl = np.asarray(p.cl, dtype=float)
        cu = np.asarray(p.cu, dtype=float)
        for i in range(p.m):
            lo, hi = cl[i], cu[i]
            if lo == hi:
                constraints.append({
                    "type": "eq",
                    "fun": lambda x, i=i, lo=lo: float(p.cons(np.asarray(x, dtype=float))[i]) - lo,
                })
            else:
                if lo > -UNBOUNDED:
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda x, i=i, lo=lo: float(p.cons(np.asarray(x, dtype=float))[i]) - lo,
                    })
                if hi < UNBOUNDED:
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda x, i=i, hi=hi: hi - float(p.cons(np.asarray(x, dtype=float))[i]),
                    })
    res = minimize(cf, x0, jac=False, bounds=bounds, constraints=constraints, method="SLSQP")
    fun = float(res.fun)
    if not np.isfinite(fun):
        return None, None
    return fun, _reference_vartheta(eqcs, ineqcs, np.asarray(res.x, dtype=float))


# -- constraint adapters for solvers without native nonlinear-constraint support

def _cobyla_constraints(eqcs, ineqcs):
    """Inequality-only constraint list (COBYLA has no equality support, so
    each equality c=0 is split into c>=0 and -c>=0)."""
    constraints = [{"type": "ineq", "fun": c} for c in ineqcs]
    for c in eqcs:
        constraints.append({"type": "ineq", "fun": c})
        constraints.append({"type": "ineq", "fun": lambda x, c=c: -c(x)})
    return constraints


def _dict_constraints(eqcs, ineqcs):
    """type='eq'/'ineq' constraints as scipy's newer solvers (COBYQA) accept directly."""
    constraints = [{"type": "ineq", "fun": c} for c in ineqcs]
    constraints += [{"type": "eq", "fun": c} for c in eqcs]
    return constraints


def _penalized_objective(cf, eqcs, ineqcs, penalty: float = 1e6):
    """Quadratic-penalty objective for solvers with no nonlinear-constraint
    support (pybobyqa, cma, nelder-mead). Only ever used to *drive* those
    solvers; their final (f, vartheta) is always recomputed from the true,
    unpenalized cf/eqcs/ineqcs afterwards."""
    def wrapped(x):
        x = np.asarray(x, dtype=float)
        f = cf(x)
        if not np.isfinite(f):
            f = 1e10
        v = 0.0
        for c in eqcs:
            v += float(c(x)) ** 2
        for c in ineqcs:
            cv = float(c(x))
            if cv < 0.0:
                v += cv ** 2
        return f + penalty * v
    return wrapped


# -- solver registry ------------------------------------------------------
#
# Each adapter has signature
#     solve(cf, eqcs, ineqcs, lb, ub, x0, cfg) -> (x, nfev, termination)
# returning the final point, the number of objective evaluations used, and a
# termination string (only meaningful for dftr; None otherwise). Adapters may
# raise -- e.g. an optional dependency isn't installed, or the solver failed
# -- and the caller records that as an errored outcome for that one solver
# without sinking the rest of the sweep.

def _solve_dftr(cf, eqcs, ineqcs, lb, ub, x0, cfg):
    from py_trsqp import DFTRFilter
    opt = DFTRFilter(x0=list(x0), cf=cf, lb=list(lb), ub=list(ub),
                     eqcs=eqcs, ineqcs=ineqcs, constants=dict(cfg.dftr_constants))
    opt.optimize(max_iter=cfg.max_iter, budget=cfg.dftr_budget)
    if opt.best is None:
        return None, opt.problem.n_cf_evals, opt.termination_status
    return np.asarray(opt.best.x, dtype=float), opt.problem.n_cf_evals, opt.termination_status


def _solve_cobyla(cf, eqcs, ineqcs, lb, ub, x0, cfg):
    from scipy.optimize import minimize
    bounds = list(zip(lb.tolist(), ub.tolist()))
    res = minimize(cf, x0, method="COBYLA", bounds=bounds,
                    constraints=_cobyla_constraints(eqcs, ineqcs),
                    options={"maxiter": cfg.max_nfev, "tol": 1e-10})
    return np.asarray(res.x, dtype=float), int(res.nfev), None


def _solve_cobyqa(cf, eqcs, ineqcs, lb, ub, x0, cfg):
    from scipy.optimize import minimize
    bounds = list(zip(lb.tolist(), ub.tolist()))
    res = minimize(cf, x0, method="COBYQA", bounds=bounds,
                    constraints=_dict_constraints(eqcs, ineqcs),
                    options={"maxfev": cfg.max_nfev})
    return np.asarray(res.x, dtype=float), int(res.nfev), None


def _solve_pybobyqa(cf, eqcs, ineqcs, lb, ub, x0, cfg):
    import pybobyqa
    obj = cf if not (eqcs or ineqcs) else _penalized_objective(cf, eqcs, ineqcs)
    res = pybobyqa.solve(obj, x0, bounds=(lb, ub), maxfun=cfg.max_nfev,
                          seek_global_minimum=False, do_logging=False)
    return np.asarray(res.x, dtype=float), int(res.nf), None


def _solve_cma(cf, eqcs, ineqcs, lb, ub, x0, cfg):
    import cma
    obj = cf if not (eqcs or ineqcs) else _penalized_objective(cf, eqcs, ineqcs)
    sigma0 = max(0.3 * float(np.min(ub - lb)), 1e-3)
    es = cma.fmin(obj, list(x0), sigma0,
                  options={"bounds": [lb.tolist(), ub.tolist()], "maxfevals": cfg.max_nfev,
                           "verbose": -9, "verb_disp": 0})
    return np.asarray(es[0], dtype=float), int(es[3]), None


def _solve_nelder_mead(cf, eqcs, ineqcs, lb, ub, x0, cfg):
    from scipy.optimize import minimize
    obj = cf if not (eqcs or ineqcs) else _penalized_objective(cf, eqcs, ineqcs)
    bounds = list(zip(lb.tolist(), ub.tolist()))
    res = minimize(obj, x0, method="Nelder-Mead", bounds=bounds,
                    options={"maxfev": cfg.max_nfev, "xatol": 1e-8, "fatol": 1e-10})
    return np.asarray(res.x, dtype=float), int(res.nfev), None


SOLVERS: Dict[str, Callable] = {
    "dftr": _solve_dftr,
    "cobyla": _solve_cobyla,
    "cobyqa": _solve_cobyqa,
    "pybobyqa": _solve_pybobyqa,
    "cma": _solve_cma,
    "nelder-mead": _solve_nelder_mead,
}


# -- run configuration + result records -----------------------------------

@dataclass
class RunConfig:
    solvers: Tuple[str, ...] = ("dftr",)
    n_max: int = 20
    m_max: int = 15
    max_iter: int = DEFAULT_MAX_ITER
    max_nfev: int = DEFAULT_MAX_NFEV
    dftr_budget: Optional[int] = None
    bound: float = DEFAULT_BOUND
    unbound_scale: float = DEFAULT_UNBOUND_SCALE
    tol: float = DEFAULT_SUCCESS_TOL
    use_reference: bool = True
    dftr_constants: dict = field(default_factory=dict)


@dataclass
class SolverOutcome:
    solver: str
    status: str  # "ok" or "error"
    f: Optional[float] = None
    vartheta: Optional[float] = None
    nfev: Optional[int] = None
    wall_time: Optional[float] = None
    termination: Optional[str] = None
    success: Optional[bool] = None  # feasible & within tol of reference; None = not judged
    error: Optional[str] = None


@dataclass
class ProblemResult:
    name: str
    status: str  # "ok", "timeout", "error", "skipped"
    n: Optional[int] = None
    m: Optional[int] = None
    f_ref: Optional[float] = None
    ref_vartheta: Optional[float] = None
    outcomes: Dict[str, SolverOutcome] = field(default_factory=dict)
    detail: Optional[str] = None


def _judge(f: float, vartheta: float, f_ref: Optional[float], tol: float) -> Optional[bool]:
    """Success = feasible AND objective within tol of the reference. None if
    there is no usable reference to judge against."""
    if f_ref is None:
        return None
    feasible = vartheta <= FEASIBLE_TOL
    return bool(feasible and f <= f_ref + tol * max(1.0, abs(f_ref)))


def _run_one_in_process(name: str, cfg: RunConfig, queue: "mp.Queue") -> None:
    """Runs every chosen solver on one problem; always puts one ProblemResult."""
    try:
        import pycutest

        p = pycutest.import_problem(name, quiet=True)
        if p.n > cfg.n_max or p.m > cfg.m_max:
            queue.put(ProblemResult(name=name, status="skipped", n=p.n, m=p.m,
                                     detail=f"n={p.n} or m={p.m} exceeds requested cap"))
            return

        cf, eqcs, ineqcs, lb, ub, x0, n, m = _build_dftr_inputs(p, cfg.bound, cfg.unbound_scale)

        f_ref, ref_vartheta = (None, None)
        if cfg.use_reference:
            f_ref, ref_vartheta = _reference_solve(cf, eqcs, ineqcs, lb, ub, x0, p)

        outcomes: Dict[str, SolverOutcome] = {}
        for sname in cfg.solvers:
            adapter = SOLVERS[sname]
            t0 = time.perf_counter()
            try:
                x, nfev, term = adapter(cf, eqcs, ineqcs, lb, ub, x0, cfg)
                wall = time.perf_counter() - t0
                if x is None:
                    # Solver ran but produced no usable point (e.g. DFTR's
                    # "Simulation failed" before any point was recorded).
                    outcomes[sname] = SolverOutcome(sname, "ok", nfev=nfev, wall_time=wall,
                                                    termination=term, success=False)
                    continue
                x = np.asarray(x, dtype=float)
                f = float(cf(x))
                vartheta = _reference_vartheta(eqcs, ineqcs, x)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite objective at returned x ({f})")
                success = _judge(f, vartheta, f_ref, cfg.tol) if cfg.use_reference else None
                outcomes[sname] = SolverOutcome(sname, "ok", f=f, vartheta=vartheta, nfev=nfev,
                                                wall_time=wall, termination=term, success=success)
            except Exception as e:  # noqa: BLE001 - one bad solver shouldn't sink the problem
                outcomes[sname] = SolverOutcome(sname, "error", wall_time=time.perf_counter() - t0,
                                                error=f"{type(e).__name__}: {e}")

        queue.put(ProblemResult(name=name, status="ok", n=n, m=m,
                                 f_ref=f_ref, ref_vartheta=ref_vartheta, outcomes=outcomes))
    except Exception as e:  # noqa: BLE001 - report, don't crash the sweep
        queue.put(ProblemResult(name=name, status="error", detail=f"{type(e).__name__}: {e}"))


def run_problem(name: str, cfg: RunConfig, timeout: float = 60.0) -> ProblemResult:
    """Runs all of cfg.solvers on one CUTEst problem in a subprocess with a
    wall-clock timeout covering the whole problem."""
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_run_one_in_process, args=(name, cfg, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return ProblemResult(name=name, status="timeout", detail=f"exceeded {timeout}s")
    try:
        return queue.get_nowait()
    except Exception:
        return ProblemResult(name=name, status="error", detail="process exited without a result")


def select_problems(n_max: int = 20, m_max: int = 15, count: Optional[int] = None,
                     include_constrained: bool = True, seed: int = 0) -> List[str]:
    """Picks CUTEst problem names via classification, deterministically subsampled.

    find_problems() filters on the *declared* default size, which occasionally
    disagrees with the actual instantiated n/m (e.g. GENROSE declares ~20 but
    defaults to n=500) - run_problem() re-checks the real p.n/p.m after import
    and marks those "skipped" rather than trusting this pre-filter blindly.
    """
    import pycutest

    names = set(pycutest.find_problems(
        constraints="unconstrained bound", regular=True, n=[2, n_max], userN=False,
    ))
    if include_constrained:
        names |= set(pycutest.find_problems(
            constraints="linear quadratic other", regular=True,
            n=[2, n_max], m=[1, m_max], userN=False, userM=False,
        ))

    names = sorted(names)
    if count is not None and count < len(names):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(names), size=count, replace=False)
        names = [names[i] for i in sorted(idx)]
    return names


def run_suite(problems: List[str], solvers: Tuple[str, ...] = ("dftr",),
              n_max: int = 20, m_max: int = 15, max_iter: int = DEFAULT_MAX_ITER,
              max_nfev: int = DEFAULT_MAX_NFEV, dftr_budget: Optional[int] = None,
              bound: float = DEFAULT_BOUND, unbound_scale: float = DEFAULT_UNBOUND_SCALE,
              tol: float = DEFAULT_SUCCESS_TOL, use_reference: bool = True,
              dftr_constants: Optional[dict] = None, timeout: float = 60.0,
              verbose: bool = True) -> List[ProblemResult]:
    unknown = [s for s in solvers if s not in SOLVERS]
    if unknown:
        raise ValueError(f"unknown solver(s) {unknown}; choose from {sorted(SOLVERS)}")
    cfg = RunConfig(solvers=tuple(solvers), n_max=n_max, m_max=m_max, max_iter=max_iter,
                    max_nfev=max_nfev, dftr_budget=dftr_budget, bound=bound,
                    unbound_scale=unbound_scale, tol=tol, use_reference=use_reference,
                    dftr_constants=dict(dftr_constants or {}))

    results = []
    for i, name in enumerate(problems, 1):
        r = run_problem(name, cfg, timeout=timeout)
        results.append(r)
        if verbose:
            if r.status != "ok":
                tags = f"{r.status}  {r.detail or ''}"
            else:
                parts = []
                for s in cfg.solvers:
                    o = r.outcomes.get(s)
                    if o is None or o.status == "error":
                        parts.append(f"{s}:ERR")
                    elif o.success is None:
                        parts.append(f"{s}:N/A")
                    else:
                        parts.append(f"{s}:{'OK' if o.success else 'FAIL'}")
                tags = "  ".join(parts)
            print(f"[{i}/{len(problems)}] {name:15s} n={r.n or '?':>4} m={r.m or '?':>3}  {tags}",
                  flush=True)
    return results


def summarize(results: List[ProblemResult]) -> dict:
    """Per-solver aggregate stats plus per-problem 'winner' counts.

    A solver's success rate is taken over the problems where it produced a
    result AND a reference was available to judge against ("judged"); problems
    with no usable reference are excluded from the denominator rather than
    counted as failures.
    """
    ok_problems = [r for r in results if r.status == "ok"]
    solver_names = []
    for r in ok_problems:
        for s in r.outcomes:
            if s not in solver_names:
                solver_names.append(s)

    per_solver = {}
    for s in solver_names:
        outs = [r.outcomes[s] for r in ok_problems if s in r.outcomes]
        ran = [o for o in outs if o.status == "ok"]
        judged = [o for o in ran if o.success is not None]
        succ = [o for o in judged if o.success]
        nfevs = [o.nfev for o in ran if o.nfev is not None]
        walls = [o.wall_time for o in ran if o.wall_time is not None]
        per_solver[s] = {
            "completed": len(ran),
            "errored": len(outs) - len(ran),
            "judged": len(judged),
            "no_reference": len(ran) - len(judged),
            "succeeded": len(succ),
            "failed": len(judged) - len(succ),
            "success_rate": (len(succ) / len(judged)) if judged else float("nan"),
            "mean_nfev": float(np.mean(nfevs)) if nfevs else float("nan"),
            "mean_wall_s": float(np.mean(walls)) if walls else float("nan"),
        }

    # Per-problem winner: lowest objective among solvers that were feasible.
    wins = {s: 0 for s in solver_names}
    for r in ok_problems:
        feasible = {s: o.f for s, o in r.outcomes.items()
                    if o.status == "ok" and o.f is not None
                    and (o.vartheta or 0.0) <= FEASIBLE_TOL}
        if feasible:
            wins[min(feasible, key=feasible.get)] += 1

    return {
        "total": len(results),
        "completed_problems": len(ok_problems),
        "timeout": len([r for r in results if r.status == "timeout"]),
        "error": len([r for r in results if r.status == "error"]),
        "skipped": len([r for r in results if r.status == "skipped"]),
        "solvers": solver_names,
        "per_solver": per_solver,
        "wins": wins,
    }


def print_summary(summary: dict) -> None:
    print("\n=== CUTEst benchmark summary ===")
    print(f"  problems attempted : {summary['total']}")
    print(f"  completed / timeout / error / skipped : "
          f"{summary['completed_problems']} / {summary['timeout']} / "
          f"{summary['error']} / {summary['skipped']}")
    print(f"\n  {'solver':12s} {'compl':>6} {'judged':>7} {'succ':>5} {'fail':>5} "
          f"{'rate':>7} {'meanNfev':>9} {'meanWall':>9} {'wins':>5}")
    for s in summary["solvers"]:
        st = summary["per_solver"][s]
        rate = "n/a" if st["judged"] == 0 else f"{st['success_rate']:.0%}"
        mnf = "n/a" if st["completed"] == 0 else f"{st['mean_nfev']:.0f}"
        mw = "n/a" if st["completed"] == 0 else f"{st['mean_wall_s']:.2f}s"
        print(f"  {s:12s} {st['completed']:>6} {st['judged']:>7} {st['succeeded']:>5} "
              f"{st['failed']:>5} {rate:>7} {mnf:>9} {mw:>9} {summary['wins'][s]:>5}")
    print("\n  (rate = success over judged problems; wins = times this solver had the "
          "best feasible objective)")


def write_csv(results: List[ProblemResult], path: str) -> None:
    """Long format: one row per (problem, solver)."""
    fields = ["name", "n", "m", "problem_status", "f_ref", "ref_vartheta",
              "solver", "solver_status", "f", "vartheta", "nfev", "wall_time",
              "success", "termination", "error", "detail"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            base = {"name": r.name, "n": r.n, "m": r.m, "problem_status": r.status,
                    "f_ref": r.f_ref, "ref_vartheta": r.ref_vartheta, "detail": r.detail}
            if not r.outcomes:
                writer.writerow(base)
                continue
            for o in r.outcomes.values():
                row = dict(base)
                row.update({"solver": o.solver, "solver_status": o.status, "f": o.f,
                            "vartheta": o.vartheta, "nfev": o.nfev, "wall_time": o.wall_time,
                            "success": o.success, "termination": o.termination, "error": o.error})
                writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solvers", nargs="+", default=["dftr"], choices=sorted(SOLVERS),
                        help="which optimizer(s) to run on each problem (default: dftr)")
    parser.add_argument("--n-max", type=int, default=20)
    parser.add_argument("--m-max", type=int, default=15)
    parser.add_argument("--count", type=int, default=40, help="number of problems to sample (None = all)")
    parser.add_argument("--no-constrained", action="store_true", help="unconstrained/bound problems only")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="DFTR-Filter iteration cap")
    parser.add_argument("--max-nfev", type=int, default=DEFAULT_MAX_NFEV,
                        help="function-evaluation cap for the non-DFTR solvers")
    parser.add_argument("--budget", type=int, default=None, help="DFTR-Filter evaluation budget (optional)")
    parser.add_argument("--bound", type=float, default=DEFAULT_BOUND,
                        help="safety-ceiling half-width for originally-unbounded variables")
    parser.add_argument("--unbound-scale", type=float, default=DEFAULT_UNBOUND_SCALE,
                        help="for originally-unbounded variables, box half-width = "
                             "min(--bound, unbound-scale * max(|x0_i|, 1))")
    parser.add_argument("--tol", type=float, default=DEFAULT_SUCCESS_TOL,
                        help="relative gap to the reference objective counted as success")
    parser.add_argument("--no-reference", action="store_true",
                        help="skip the gradient-based reference solve (no success judging, just raw results)")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-problem wall-clock timeout (s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=str, default=None, help="optional path to write per-(problem,solver) rows")
    args = parser.parse_args(argv)

    if not cutest_available():
        print("CUTEst/pycutest is not set up in this environment - see DEVELOPMENT.md, "
              "'CUTEst benchmark setup'. Nothing to run.", file=sys.stderr)
        return 1

    problems = select_problems(n_max=args.n_max, m_max=args.m_max, count=args.count,
                                include_constrained=not args.no_constrained, seed=args.seed)
    print(f"Selected {len(problems)} CUTEst problems "
          f"(n<={args.n_max}, m<={args.m_max}, constrained={not args.no_constrained}).")
    print(f"Solvers: {', '.join(args.solvers)}")

    results = run_suite(problems, solvers=tuple(args.solvers), n_max=args.n_max, m_max=args.m_max,
                        max_iter=args.max_iter, max_nfev=args.max_nfev, dftr_budget=args.budget,
                        bound=args.bound, unbound_scale=args.unbound_scale, tol=args.tol,
                        use_reference=not args.no_reference, timeout=args.timeout)
    if args.csv:
        write_csv(results, args.csv)
        print(f"\nWrote per-(problem, solver) results to {args.csv}")
    print_summary(summarize(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
