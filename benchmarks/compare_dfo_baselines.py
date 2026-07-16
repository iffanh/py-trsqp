"""Compare DFTR-Filter against other derivative-free SciPy solvers on a
given list of CUTEst problems (by default, a curated set of problems that
are relatively difficult for DFTR-Filter's restoration/convergence
behavior, useful for comparing against other derivative-free methods on
the same class of problem).

Baselines are chosen to be a fair, same-class (derivative-free) comparison,
not the gradient-based reference used elsewhere in cutest_benchmark.py:
  - Nelder-Mead (bounds, no constraints)     for unconstrained/bound problems
  - COBYLA (bounds + inequality constraints) for constrained problems, with
    each equality constraint split into two inequalities (c>=0 and -c>=0),
    since COBYLA has no native equality-constraint support.

For a broader comparison against five alternative solvers at once (COBYLA,
COBYQA, Py-BOBYQA, CMA-ES, Nelder-Mead), see multi_solver_compare.py.

Usage:
    python benchmarks/compare_dfo_baselines.py --max-nfev 500
    python benchmarks/compare_dfo_baselines.py --problems AIRCRFTA BEALENE ...
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from benchmarks.cutest_benchmark import DEFAULT_BOUND, _build_dftr_inputs, cutest_available

DEFAULT_COMPARISON_PROBLEMS = [
    "AIRCRFTA", "BEALENE", "DENSCHNE", "DEVGLA1", "FBRAINLS", "HS17",
    "LOGROS", "MDHOLE", "MGH09", "MISRA1B", "PALMER2A", "PALMER4A",
    "ROSENMMX", "SYNTHES1", "TRIGON1NE",
]


def _solve_with_scipy_dfo(cf, eqcs, ineqcs, lb, ub, x0, max_nfev):
    from scipy.optimize import minimize

    calls = {"n": 0}

    def counted_cf(x):
        calls["n"] += 1
        return cf(x)

    bounds = list(zip(lb.tolist(), ub.tolist()))

    def vartheta(x):
        """Same violation convention as DFTR-Filter: 0 if feasible."""
        v = 0.0
        for c in eqcs:
            v = max(v, abs(c(x)))
        for c in ineqcs:
            v = max(v, max(0.0, -c(x)))
        return v

    if not eqcs and not ineqcs:
        res = minimize(counted_cf, x0, method="Nelder-Mead", bounds=bounds,
                        options={"maxfev": max_nfev, "xatol": 1e-8, "fatol": 1e-10})
        return "Nelder-Mead", res.fun, 0.0, calls["n"]

    constraints = [{"type": "ineq", "fun": c} for c in ineqcs]
    for c in eqcs:
        constraints.append({"type": "ineq", "fun": c})
        constraints.append({"type": "ineq", "fun": lambda x, c=c: -c(x)})
    res = minimize(counted_cf, x0, method="COBYLA", bounds=bounds,
                    constraints=constraints, options={"maxiter": max_nfev, "tol": 1e-10})
    return "COBYLA", res.fun, vartheta(res.x), calls["n"]


def run_one(name, bound, max_nfev):
    import pycutest

    p = pycutest.import_problem(name, quiet=True)
    cf, eqcs, ineqcs, lb, ub, x0, n, m = _build_dftr_inputs(p, bound)
    t0 = time.time()
    try:
        method, f_final, vartheta, nfev = _solve_with_scipy_dfo(cf, eqcs, ineqcs, lb, ub, x0, max_nfev)
        status = "ok"
    except Exception as e:  # noqa: BLE001
        method, f_final, vartheta, nfev = "-", None, None, None
        status = f"error: {type(e).__name__}: {e}"
    wall = time.time() - t0
    return {"name": name, "n": n, "m": m, "method": method, "f_final": f_final,
            "vartheta": vartheta, "nfev": nfev, "wall_time": wall, "status": status}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=DEFAULT_COMPARISON_PROBLEMS)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--bound", type=float, default=DEFAULT_BOUND)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args(argv)

    if not cutest_available():
        print("CUTEst/pycutest is not set up in this environment.")
        return 1

    rows = []
    print(f"{'name':12s} {'n':>4} {'m':>4}  {'method':11s} {'f_final':>18} {'vartheta':>12} {'nfev':>6}  status")
    for name in args.problems:
        r = run_one(name, args.bound, args.max_nfev)
        rows.append(r)
        f_str = f"{r['f_final']:.6g}" if r["f_final"] is not None else "n/a"
        v_str = f"{r['vartheta']:.3g}" if r["vartheta"] is not None else "n/a"
        print(f"{r['name']:12s} {r['n']:>4} {r['m']:>4}  {r['method']:11s} {f_str:>18} {v_str:>12} "
              f"{r['nfev'] if r['nfev'] is not None else '-':>6}  {r['status']}")

    if args.csv:
        import csv as csv_mod
        with open(args.csv, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
