"""Benchmark DFTR-Filter against CUTEst test problems via pycutest.

Requires a working CUTEst + SIFDecode + pycutest installation (ARCHDEFS,
SIFDECODE, CUTEST, MASTSIF environment variables set, `pycutest` importable).
See DEVELOPMENT.md, section "CUTEst benchmark setup", for how to build that
toolchain from scratch. Every entry point here degrades gracefully -
`cutest_available()` returns False and callers can skip - when the toolchain
isn't present, so this module is safe to import unconditionally.

CLI usage:
    python benchmarks/cutest_benchmark.py --n-max 15 --count 40

Runs each selected problem in a separate subprocess (bounded by --timeout
seconds) so that a single pathological CUTEst problem cannot hang or crash
the whole sweep, then prints a summary table of success rate, function
evaluations, and wall time, comparing DFTR-Filter's best objective against a
gradient-based SciPy reference solve (SLSQP/L-BFGS-B, using CUTEst's exact
derivatives - DFTR-Filter itself never sees them).
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

# py_trsqp isn't pip-installed in most dev setups (see DEVELOPMENT.md); make
# sure the repo root is importable regardless of how this file is invoked
# (as a script, as `python -m benchmarks.cutest_benchmark`, or imported by
# pytest). Forked subprocesses inherit this via copy-on-write; a "spawn"
# context would re-run this top-level code anyway.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

UNBOUNDED = 1e19
DEFAULT_BOUND = 50.0
DEFAULT_UNBOUND_SCALE = 2.0

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

DEFAULT_SUCCESS_TOL = 0.1  # relative gap to the SciPy reference objective


def cutest_available() -> bool:
    """Whether pycutest and the CUTEst toolchain look usable in this process."""
    if not all(os.environ.get(v) for v in ("ARCHDEFS", "SIFDECODE", "CUTEST", "MASTSIF")):
        return False
    try:
        import pycutest  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class ProblemResult:
    name: str
    status: str  # "ok", "timeout", "error", "skipped"
    n: Optional[int] = None
    m: Optional[int] = None
    f_best: Optional[float] = None
    f_ref: Optional[float] = None
    vartheta: Optional[float] = None
    ref_vartheta: Optional[float] = None
    n_cf_evals: Optional[int] = None
    wall_time: Optional[float] = None
    termination_status: Optional[str] = None
    success: Optional[bool] = None
    detail: Optional[str] = None


def _build_dftr_inputs(p, bound: float = DEFAULT_BOUND, unbound_scale: float = DEFAULT_UNBOUND_SCALE):
    """Translate a pycutest problem into DFTRFilter's (cf, eqcs, ineqcs, lb, ub, x0).

    CUTEst represents every constraint as cl[i] <= c_i(x) <= cu[i]. DFTR-Filter
    wants separate equality (=0) and inequality (>=0) callables, so each row is
    split into 1 or 2 black-box callables depending on which bounds are finite.

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
    0 if feasible, else the largest constraint violation at x."""
    from py_trsqp.core import geometry

    eq_vals = np.array([c(x) for c in eqcs]) if eqcs else np.array([])
    ineq_vals = np.array([c(x) for c in ineqcs]) if ineqcs else np.array([])
    return float(geometry.violation(eq_vals, ineq_vals))


def _reference_solve(cf, eqcs, ineqcs, lb, ub, x0, p) -> Tuple[Optional[float], Optional[float]]:
    """Gradient-based SciPy reference (SLSQP if constrained, else L-BFGS-B).

    Uses CUTEst's exact derivatives for speed/reliability; DFTR-Filter never
    sees them. Used purely as an independent yardstick for "success".

    Returns (f_ref, vartheta_ref) - the reference solver's own constraint
    violation at the point it stopped at, computed with the same convention
    DFTR-Filter uses, so a "better" f_ref that's actually infeasible (SLSQP,
    like DFTR-Filter's own restoration phase, can stop short of full
    feasibility) is visible rather than silently treated as ground truth.

    Returns (None, None) if the reference solver itself couldn't produce a
    finite objective (e.g. it wanders into the same NaN-producing domain
    restriction the black box has and aborts). Comparing against NaN would
    silently fail every success check, mislabeling "no usable reference" as
    "DFTR-Filter failed" - callers must treat None as "not judged", not as a
    failure.
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


def _run_one_in_process(name: str, n_max: int, m_max: int, max_iter: int, budget: Optional[int],
                         bound: float, unbound_scale: float, tol: float, queue: "mp.Queue") -> None:
    """Executes a single problem; always puts exactly one ProblemResult on `queue`."""
    try:
        import pycutest
        from py_trsqp import DFTRFilter

        t0 = time.time()
        p = pycutest.import_problem(name, quiet=True)
        if p.n > n_max or p.m > m_max:
            queue.put(ProblemResult(name=name, status="skipped", n=p.n, m=p.m,
                                     detail=f"n={p.n} or m={p.m} exceeds requested cap"))
            return

        cf, eqcs, ineqcs, lb, ub, x0, n, m = _build_dftr_inputs(p, bound, unbound_scale)
        opt = DFTRFilter(x0=list(x0), cf=cf, lb=list(lb), ub=list(ub), eqcs=eqcs, ineqcs=ineqcs)
        opt.optimize(max_iter=max_iter, budget=budget)
        wall = time.time() - t0

        if opt.best is None:
            # e.g. "Simulation failed": cf/constraints returned NaN/Inf on the
            # very first poised set (domain restrictions CUTEst doesn't encode
            # as box bounds, e.g. sqrt/log terms) before any point was recorded.
            queue.put(ProblemResult(
                name=name, status="ok", n=n, m=m, n_cf_evals=opt.problem.n_cf_evals,
                wall_time=wall, termination_status=opt.termination_status, success=False,
            ))
            return

        f_ref, ref_vartheta = _reference_solve(cf, eqcs, ineqcs, lb, ub, x0, p)
        if f_ref is None:
            # SciPy's own gradient-based solver also couldn't produce a
            # finite objective here (e.g. it hit the same NaN-producing
            # domain restriction) - there's no valid yardstick to judge
            # DFTR-Filter's result against. success=None marks this as
            # "not judged", distinct from a real pass/fail (see summarize()).
            queue.put(ProblemResult(
                name=name, status="ok", n=n, m=m,
                f_best=opt.best.f, f_ref=None, vartheta=opt.best.vartheta,
                n_cf_evals=opt.problem.n_cf_evals, wall_time=wall,
                termination_status=opt.termination_status, success=None,
                detail="no valid reference: SciPy's own solve was non-finite too",
            ))
            return

        # The reference is only a fair yardstick where it's itself feasible;
        # a "better" f_ref reached by an infeasible SLSQP point (it can stop
        # short of full feasibility just like DFTR-Filter's own restoration
        # phase) isn't ground truth, so success requires OUR feasibility, not
        # a comparison to a possibly-infeasible reference point.
        abs_tol = tol * max(1.0, abs(f_ref))
        feasible = (opt.best.vartheta or 0.0) <= 1e-4
        success = feasible and (opt.best.f <= f_ref + abs_tol)

        queue.put(ProblemResult(
            name=name, status="ok", n=n, m=m,
            f_best=opt.best.f, f_ref=f_ref, vartheta=opt.best.vartheta, ref_vartheta=ref_vartheta,
            n_cf_evals=opt.problem.n_cf_evals, wall_time=wall,
            termination_status=opt.termination_status, success=success,
        ))
    except Exception as e:  # noqa: BLE001 - report, don't crash the sweep
        queue.put(ProblemResult(name=name, status="error", detail=f"{type(e).__name__}: {e}"))


def run_problem(name: str, n_max: int = 20, m_max: int = 15, max_iter: int = 100,
                 budget: Optional[int] = None, bound: float = DEFAULT_BOUND,
                 unbound_scale: float = DEFAULT_UNBOUND_SCALE,
                 tol: float = DEFAULT_SUCCESS_TOL, timeout: float = 30.0) -> ProblemResult:
    """Runs one CUTEst problem in a subprocess with a wall-clock timeout."""
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_run_one_in_process,
                        args=(name, n_max, m_max, max_iter, budget, bound, unbound_scale, tol, queue))
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
                     include_constrained: bool = True, seed: int = 0) -> list[str]:
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


def run_suite(problems: list[str], n_max: int = 20, m_max: int = 15, max_iter: int = 100,
              budget: Optional[int] = None, bound: float = DEFAULT_BOUND,
              unbound_scale: float = DEFAULT_UNBOUND_SCALE,
              tol: float = DEFAULT_SUCCESS_TOL, timeout: float = 30.0,
              verbose: bool = True) -> list[ProblemResult]:
    results = []
    for i, name in enumerate(problems, 1):
        r = run_problem(name, n_max=n_max, m_max=m_max, max_iter=max_iter,
                         budget=budget, bound=bound, unbound_scale=unbound_scale,
                         tol=tol, timeout=timeout)
        results.append(r)
        if verbose:
            if r.status != "ok":
                tag = "--- "
            elif r.success is None:
                tag = "N/A "   # e.g. no valid reference to judge against (see _reference_solve)
            else:
                tag = "OK " if r.success else "FAIL"
            print(f"[{i}/{len(problems)}] {name:15s} n={r.n or '?':>4} m={r.m or '?':>3}  "
                  f"{r.status:8s} {tag}  {r.detail or ''}", flush=True)
    return results


def summarize(results: list[ProblemResult]) -> dict:
    ok = [r for r in results if r.status == "ok"]
    # r.success is None means "no valid reference to judge against" (see
    # _reference_solve) - distinct from an actual pass/fail, and excluded
    # from the success-rate denominator rather than silently counted as a
    # failure.
    no_reference = [r for r in ok if r.success is None]
    judged = [r for r in ok if r.success is not None]
    succ = [r for r in judged if r.success]
    failed = [r for r in judged if not r.success]
    timeouts = [r for r in results if r.status == "timeout"]
    errors = [r for r in results if r.status == "error"]
    skipped = [r for r in results if r.status == "skipped"]

    summary = {
        "total": len(results),
        "completed": len(ok),
        "no_reference": len(no_reference),
        "judged": len(judged),
        "succeeded": len(succ),
        "failed": len(failed),
        "timeout": len(timeouts),
        "error": len(errors),
        "skipped": len(skipped),
        "success_rate_of_completed": (len(succ) / len(judged)) if judged else float("nan"),
        "mean_cf_evals": float(np.mean([r.n_cf_evals for r in ok])) if ok else float("nan"),
        "median_cf_evals": float(np.median([r.n_cf_evals for r in ok])) if ok else float("nan"),
        "mean_wall_time_s": float(np.mean([r.wall_time for r in ok])) if ok else float("nan"),
        "total_wall_time_s": float(np.sum([r.wall_time for r in ok])) if ok else float("nan"),
    }
    return summary


def print_summary(summary: dict) -> None:
    print("\n=== CUTEst benchmark summary ===")
    print(f"  total problems attempted : {summary['total']}")
    print(f"  completed (not error/timeout/skipped): {summary['completed']}")
    print(f"  no valid reference (excluded from rate): {summary['no_reference']}")
    print(f"  succeeded (feasible & within tol of SciPy reference): {summary['succeeded']}")
    print(f"  failed (completed but not within tol): {summary['failed']}")
    print(f"  timed out                : {summary['timeout']}")
    print(f"  errored                  : {summary['error']}")
    print(f"  skipped (size cap)       : {summary['skipped']}")
    print(f"  success rate (of judged) : {summary['success_rate_of_completed']:.1%}")
    print(f"  mean/median function evals : {summary['mean_cf_evals']:.1f} / {summary['median_cf_evals']:.1f}")
    print(f"  mean wall time per problem : {summary['mean_wall_time_s']:.2f}s")
    print(f"  total wall time            : {summary['total_wall_time_s']:.1f}s")


def write_csv(results: list[ProblemResult], path: str) -> None:
    fields = list(asdict(results[0]).keys()) if results else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-max", type=int, default=20)
    parser.add_argument("--m-max", type=int, default=15)
    parser.add_argument("--count", type=int, default=40, help="number of problems to sample (None = all)")
    parser.add_argument("--no-constrained", action="store_true", help="unconstrained/bound problems only")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--bound", type=float, default=DEFAULT_BOUND,
                         help="safety-ceiling half-width for originally-unbounded variables")
    parser.add_argument("--unbound-scale", type=float, default=DEFAULT_UNBOUND_SCALE,
                         help="for originally-unbounded variables, box half-width = "
                              "min(--bound, unbound-scale * max(|x0_i|, 1)) -- per-problem, not one constant box")
    parser.add_argument("--tol", type=float, default=DEFAULT_SUCCESS_TOL,
                         help="relative gap to the SciPy reference objective counted as success")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-problem wall-clock timeout (s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=str, default=None, help="optional path to write per-problem results")
    args = parser.parse_args(argv)

    if not cutest_available():
        print("CUTEst/pycutest is not set up in this environment - see DEVELOPMENT.md, "
              "'CUTEst benchmark setup'. Nothing to run.", file=sys.stderr)
        return 1

    problems = select_problems(n_max=args.n_max, m_max=args.m_max, count=args.count,
                                include_constrained=not args.no_constrained, seed=args.seed)
    print(f"Selected {len(problems)} CUTEst problems "
          f"(n<={args.n_max}, m<={args.m_max}, constrained={not args.no_constrained}).")

    results = run_suite(problems, n_max=args.n_max, m_max=args.m_max, max_iter=args.max_iter,
                         budget=args.budget, bound=args.bound, unbound_scale=args.unbound_scale,
                         tol=args.tol, timeout=args.timeout)
    if args.csv:
        write_csv(results, args.csv)
        print(f"\nWrote per-problem results to {args.csv}")
    print_summary(summarize(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
