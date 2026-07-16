"""Compare DFTR-Filter against 5 other open-source derivative-free solvers
on CUTEst problems, and judge it against the BEST of them, not a single one.

Alternatives (all pip-installable, no compiled toolchain beyond CUTEst
itself needed):
  - COBYLA     (scipy)       -- native linear/nonlinear inequality constraints
  - COBYQA     (scipy)       -- Ragonneau's newer model-based successor to
                                 COBYLA; native bounds + nonlinear eq/ineq
  - Py-BOBYQA  (NAG)         -- bound-constrained trust-region DFO, same
                                 algorithmic family as DFTR-Filter itself
  - CMA-ES     (`cma`)       -- evolutionary, fundamentally different class
  - Nelder-Mead (scipy)      -- classic derivative-free simplex baseline

COBYLA/COBYQA get the constraints natively (COBYLA has no equality support,
so each equality is split into two inequalities, same trick used elsewhere
in this repo). Py-BOBYQA/CMA-ES/Nelder-Mead have no nonlinear-constraint
support, so constrained problems are handed a quadratic-penalty objective
instead -- but every solver's *reported* (f, vartheta) is recomputed from
the true cf/eqcs/ineqcs at its returned x, never trusted from the
(possibly-penalized) objective value, so the comparison stays fair
regardless of which mechanism a solver used to handle constraints.

"Best of alternatives" = the lowest f among alternatives that are
themselves feasible (vartheta <= 1e-4); if none of the 5 are feasible on a
given problem, falls back to whichever has the smallest violation, and
DFTR-Filter is only counted as "beating" that when it is itself feasible.

Usage:
    python benchmarks/multi_solver_compare.py --n-max 20 --count 40
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarks.cutest_benchmark import (  # noqa: E402
    DEFAULT_BOUND,
    DEFAULT_UNBOUND_SCALE,
    _build_dftr_inputs,
    _reference_vartheta,
    cutest_available,
    select_problems,
)

FEASIBLE_TOL = 1e-4
DEFAULT_SUCCESS_TOL = 0.1
DEFAULT_MAX_NFEV = 500

SOLVER_NAMES = ["COBYLA", "COBYQA", "Py-BOBYQA", "CMA-ES", "Nelder-Mead"]


# -- constraint adapters -------------------------------------------------

def _cobyla_constraints(eqcs, ineqcs):
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
    """For solvers with no nonlinear-constraint support (Py-BOBYQA, CMA-ES,
    Nelder-Mead): quadratic penalty added to the objective. Only ever used
    to *drive* those solvers; their final (f, vartheta) is always
    recomputed from the true, unpenalized cf/eqcs/ineqcs afterwards."""
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


# -- solver adapters: each returns (final x, actual evaluations used) or raises

def _solve_cobyla(cf, eqcs, ineqcs, lb, ub, x0, max_nfev):
    from scipy.optimize import minimize
    bounds = list(zip(lb.tolist(), ub.tolist()))
    res = minimize(cf, x0, method="COBYLA", bounds=bounds,
                    constraints=_cobyla_constraints(eqcs, ineqcs),
                    options={"maxiter": max_nfev, "tol": 1e-10})
    return np.asarray(res.x, dtype=float), int(res.nfev)


def _solve_cobyqa(cf, eqcs, ineqcs, lb, ub, x0, max_nfev):
    from scipy.optimize import minimize
    bounds = list(zip(lb.tolist(), ub.tolist()))
    res = minimize(cf, x0, method="COBYQA", bounds=bounds,
                    constraints=_dict_constraints(eqcs, ineqcs),
                    options={"maxfev": max_nfev})
    return np.asarray(res.x, dtype=float), int(res.nfev)


def _solve_pybobyqa(cf, eqcs, ineqcs, lb, ub, x0, max_nfev):
    import pybobyqa
    obj = cf if not (eqcs or ineqcs) else _penalized_objective(cf, eqcs, ineqcs)
    res = pybobyqa.solve(obj, x0, bounds=(lb, ub), maxfun=max_nfev,
                          seek_global_minimum=False, do_logging=False)
    return np.asarray(res.x, dtype=float), int(res.nf)


def _solve_cmaes(cf, eqcs, ineqcs, lb, ub, x0, max_nfev):
    import cma
    obj = cf if not (eqcs or ineqcs) else _penalized_objective(cf, eqcs, ineqcs)
    sigma0 = max(0.3 * float(np.min(ub - lb)), 1e-3)
    es = cma.fmin(obj, list(x0), sigma0,
                  options={"bounds": [lb.tolist(), ub.tolist()], "maxfevals": max_nfev,
                           "verbose": -9, "verb_disp": 0})
    return np.asarray(es[0], dtype=float), int(es[3])


def _solve_nelder_mead(cf, eqcs, ineqcs, lb, ub, x0, max_nfev):
    from scipy.optimize import minimize
    obj = cf if not (eqcs or ineqcs) else _penalized_objective(cf, eqcs, ineqcs)
    bounds = list(zip(lb.tolist(), ub.tolist()))
    res = minimize(obj, x0, method="Nelder-Mead", bounds=bounds,
                    options={"maxfev": max_nfev, "xatol": 1e-8, "fatol": 1e-10})
    return np.asarray(res.x, dtype=float), int(res.nfev)


_SOLVERS: dict[str, Callable] = {
    "COBYLA": _solve_cobyla,
    "COBYQA": _solve_cobyqa,
    "Py-BOBYQA": _solve_pybobyqa,
    "CMA-ES": _solve_cmaes,
    "Nelder-Mead": _solve_nelder_mead,
}


@dataclass
class MultiSolverResult:
    name: str
    status: str  # "ok", "timeout", "error", "skipped"
    n: Optional[int] = None
    m: Optional[int] = None
    dftr_f: Optional[float] = None
    dftr_vartheta: Optional[float] = None
    dftr_termination: Optional[str] = None
    dftr_evals: Optional[int] = None
    best_alt_name: Optional[str] = None
    best_alt_f: Optional[float] = None
    best_alt_vartheta: Optional[float] = None
    best_alt_nfev: Optional[int] = None
    any_alt_feasible: Optional[bool] = None
    success: Optional[bool] = None  # DFTR feasible AND DFTR.f <= best_alt.f + tol
    solver_f: dict = field(default_factory=dict)
    solver_vartheta: dict = field(default_factory=dict)
    solver_nfev: dict = field(default_factory=dict)
    solver_error: dict = field(default_factory=dict)
    detail: Optional[str] = None


def _run_one_in_process(name: str, n_max: int, m_max: int, max_iter: int, max_nfev: int,
                         bound: float, unbound_scale: float, tol: float, queue: "mp.Queue") -> None:
    try:
        import pycutest
        from py_trsqp import DFTRFilter

        p = pycutest.import_problem(name, quiet=True)
        if p.n > n_max or p.m > m_max:
            queue.put(MultiSolverResult(name=name, status="skipped", n=p.n, m=p.m,
                                         detail=f"n={p.n} or m={p.m} exceeds requested cap"))
            return

        cf, eqcs, ineqcs, lb, ub, x0, n, m = _build_dftr_inputs(p, bound, unbound_scale)

        opt = DFTRFilter(x0=list(x0), cf=cf, lb=list(lb), ub=list(ub), eqcs=eqcs, ineqcs=ineqcs)
        opt.optimize(max_iter=max_iter)
        dftr_f = opt.best.f if opt.best is not None else None
        dftr_vartheta = opt.best.vartheta if opt.best is not None else None

        solver_f, solver_vartheta, solver_nfev, solver_error = {}, {}, {}, {}
        for sname, solve_fn in _SOLVERS.items():
            try:
                x_final, nfev = solve_fn(cf, eqcs, ineqcs, lb, ub, x0, max_nfev)
                f_final = float(cf(np.asarray(x_final, dtype=float)))
                v_final = _reference_vartheta(eqcs, ineqcs, np.asarray(x_final, dtype=float))
                if not np.isfinite(f_final):
                    raise ValueError(f"non-finite objective at returned x ({f_final})")
                solver_f[sname] = f_final
                solver_vartheta[sname] = v_final
                solver_nfev[sname] = nfev
            except Exception as e:  # noqa: BLE001 - one bad solver shouldn't sink the comparison
                solver_error[sname] = f"{type(e).__name__}: {e}"

        feasible_alts = {s: f for s, f in solver_f.items() if solver_vartheta[s] <= FEASIBLE_TOL}
        any_alt_feasible = len(feasible_alts) > 0
        if feasible_alts:
            best_alt_name = min(feasible_alts, key=feasible_alts.get)
        elif solver_vartheta:
            best_alt_name = min(solver_vartheta, key=solver_vartheta.get)
        else:
            best_alt_name = None

        best_alt_f = solver_f.get(best_alt_name) if best_alt_name else None
        best_alt_vartheta = solver_vartheta.get(best_alt_name) if best_alt_name else None
        best_alt_nfev = solver_nfev.get(best_alt_name) if best_alt_name else None

        success = None
        if dftr_f is not None and best_alt_f is not None:
            dftr_feasible = (dftr_vartheta or 0.0) <= FEASIBLE_TOL
            abs_tol = tol * max(1.0, abs(best_alt_f))
            success = bool(dftr_feasible and any_alt_feasible and dftr_f <= best_alt_f + abs_tol)

        queue.put(MultiSolverResult(
            name=name, status="ok", n=n, m=m,
            dftr_f=dftr_f, dftr_vartheta=dftr_vartheta,
            dftr_termination=opt.termination_status, dftr_evals=opt.problem.n_cf_evals,
            best_alt_name=best_alt_name, best_alt_f=best_alt_f, best_alt_vartheta=best_alt_vartheta,
            best_alt_nfev=best_alt_nfev, any_alt_feasible=any_alt_feasible, success=success,
            solver_f=solver_f, solver_vartheta=solver_vartheta, solver_nfev=solver_nfev,
            solver_error=solver_error,
        ))
    except Exception as e:  # noqa: BLE001
        queue.put(MultiSolverResult(name=name, status="error", detail=f"{type(e).__name__}: {e}"))


def run_problem(name: str, n_max: int = 20, m_max: int = 15, max_iter: int = 500,
                max_nfev: int = DEFAULT_MAX_NFEV, bound: float = DEFAULT_BOUND,
                unbound_scale: float = DEFAULT_UNBOUND_SCALE, tol: float = DEFAULT_SUCCESS_TOL,
                timeout: float = 60.0) -> MultiSolverResult:
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_run_one_in_process,
                        args=(name, n_max, m_max, max_iter, max_nfev, bound, unbound_scale, tol, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return MultiSolverResult(name=name, status="timeout", detail=f"exceeded {timeout}s")
    try:
        return queue.get_nowait()
    except Exception:
        return MultiSolverResult(name=name, status="error", detail="process exited without a result")


def run_suite(problems: list[str], n_max: int = 20, m_max: int = 15, max_iter: int = 500,
              max_nfev: int = DEFAULT_MAX_NFEV, bound: float = DEFAULT_BOUND,
              unbound_scale: float = DEFAULT_UNBOUND_SCALE, tol: float = DEFAULT_SUCCESS_TOL,
              timeout: float = 60.0, verbose: bool = True) -> list[MultiSolverResult]:
    results = []
    for i, name in enumerate(problems, 1):
        r = run_problem(name, n_max=n_max, m_max=m_max, max_iter=max_iter, max_nfev=max_nfev,
                         bound=bound, unbound_scale=unbound_scale, tol=tol, timeout=timeout)
        results.append(r)
        if verbose:
            if r.status != "ok":
                tag = "--- "
            elif r.success is None:
                tag = "N/A "
            else:
                tag = "OK " if r.success else "FAIL"
            best = f"best={r.best_alt_name}" if r.best_alt_name else "best=none"
            print(f"[{i}/{len(problems)}] {name:15s} n={r.n or '?':>4} m={r.m or '?':>3}  "
                  f"{r.status:8s} {tag}  {best}  {r.detail or ''}", flush=True)
    return results


def summarize(results: list[MultiSolverResult]) -> dict:
    ok = [r for r in results if r.status == "ok"]
    judged = [r for r in ok if r.success is not None]
    succ = [r for r in judged if r.success]
    failed = [r for r in judged if not r.success]

    per_solver_wins = {s: 0 for s in SOLVER_NAMES}
    for r in ok:
        if r.best_alt_name:
            per_solver_wins[r.best_alt_name] = per_solver_wins.get(r.best_alt_name, 0) + 1

    return {
        "total": len(results),
        "completed": len(ok),
        "judged": len(judged),
        "succeeded": len(succ),
        "failed": len(failed),
        "timeout": len([r for r in results if r.status == "timeout"]),
        "error": len([r for r in results if r.status == "error"]),
        "skipped": len([r for r in results if r.status == "skipped"]),
        "success_rate_vs_best_alt": (len(succ) / len(judged)) if judged else float("nan"),
        "best_alt_wins": per_solver_wins,
    }


def print_summary(summary: dict) -> None:
    print("\n=== DFTR-Filter vs. best-of-5-alternatives summary ===")
    print(f"  total problems attempted : {summary['total']}")
    print(f"  completed                : {summary['completed']}")
    print(f"  judged (both sides had a result): {summary['judged']}")
    print(f"  DFTR-Filter beat/matched the best alternative: {summary['succeeded']}")
    print(f"  DFTR-Filter lost to the best alternative      : {summary['failed']}")
    print(f"  timed out / errored / skipped: {summary['timeout']} / {summary['error']} / {summary['skipped']}")
    print(f"  success rate vs. best-of-5   : {summary['success_rate_vs_best_alt']:.1%}")
    print("  how often each solver *was* the best alternative:")
    for s, c in summary["best_alt_wins"].items():
        print(f"    {s:12s}: {c}")


_PER_SOLVER_DICT_FIELDS = ("solver_f", "solver_vartheta", "solver_nfev", "solver_error")


def write_csv(results: list[MultiSolverResult], path: str) -> None:
    if not results:
        return
    fields = [f for f in asdict(results[0]).keys() if f not in _PER_SOLVER_DICT_FIELDS]
    fields += ([f"f_{s}" for s in SOLVER_NAMES] + [f"vartheta_{s}" for s in SOLVER_NAMES]
               + [f"nfev_{s}" for s in SOLVER_NAMES] + [f"error_{s}" for s in SOLVER_NAMES])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in asdict(r).items() if k not in _PER_SOLVER_DICT_FIELDS}
            for s in SOLVER_NAMES:
                row[f"f_{s}"] = r.solver_f.get(s)
                row[f"vartheta_{s}"] = r.solver_vartheta.get(s)
                row[f"nfev_{s}"] = r.solver_nfev.get(s)
                row[f"error_{s}"] = r.solver_error.get(s)
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-max", type=int, default=20)
    parser.add_argument("--m-max", type=int, default=15)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--no-constrained", action="store_true")
    parser.add_argument("--max-iter", type=int, default=500, help="DFTR-Filter's max_iter")
    parser.add_argument("--max-nfev", type=int, default=DEFAULT_MAX_NFEV, help="budget for each alternative solver")
    parser.add_argument("--bound", type=float, default=DEFAULT_BOUND)
    parser.add_argument("--unbound-scale", type=float, default=DEFAULT_UNBOUND_SCALE)
    parser.add_argument("--tol", type=float, default=DEFAULT_SUCCESS_TOL)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args(argv)

    if not cutest_available():
        print("CUTEst/pycutest is not set up in this environment.", file=sys.stderr)
        return 1

    problems = select_problems(n_max=args.n_max, m_max=args.m_max, count=args.count,
                                include_constrained=not args.no_constrained, seed=args.seed)
    print(f"Selected {len(problems)} CUTEst problems "
          f"(n<={args.n_max}, m<={args.m_max}, constrained={not args.no_constrained}).")
    print(f"Alternatives: {', '.join(SOLVER_NAMES)}")

    results = run_suite(problems, n_max=args.n_max, m_max=args.m_max, max_iter=args.max_iter,
                         max_nfev=args.max_nfev, bound=args.bound, unbound_scale=args.unbound_scale,
                         tol=args.tol, timeout=args.timeout)
    if args.csv:
        write_csv(results, args.csv)
        print(f"\nWrote {args.csv}")
    print_summary(summarize(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
