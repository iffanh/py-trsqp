"""Time the interpolation model-building core across problem dimension n.

Measures the two operations DFTRFilter._build_models spends its non-evaluation
time in:
  - fit_lagrange_set(Y, radius): build the polynomial basis matrix on the
    sample set and solve the interpolation system for the Lagrange
    coefficients. This is where essentially all the cost lives.
  - LagrangeSet.build_model(f): turn one right-hand side of sampled values
    into a QuadraticModel (a `f @ Lambda` contraction plus a coefficient
    reshape). Cheap and nearly flat in n; each extra constraint model reuses
    the same fitted LagrangeSet, so it costs only this, not another fit.

Only the interpolation algebra is timed. A real optimize() iteration also
evaluates the black-box objective/constraints at each sample point, but that
cost is user-function-dependent, not intrinsic to the method, so it is
excluded here.

Two point-count policies are reported:
  - the library default, min(2n+1, q), which for n>=1 is 2n+1: a
    minimum-Frobenius-norm degree-2 fit (underdetermined quadratic);
  - the full quadratic basis, q = (n+1)(n+2)/2: a fully determined quadratic
    (square interpolation system). This grows O(n^2) in points and is what
    makes model building expensive at larger n -- the reason the default caps
    at 2n+1.

Usage:
    python benchmarks/model_build_timing.py
    python benchmarks/model_build_timing.py --reps 40 --csv benchmarks/results/model_build_timing.csv
    python benchmarks/model_build_timing.py --ns 2 5 10 20 50
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from py_trsqp.core.interpolation import fit_lagrange_set  # noqa: E402
from py_trsqp.exceptions import IllPoisedModel, PoisednessIsZeroException  # noqa: E402

DEFAULT_NS = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def default_points(n: int) -> int:
    """The library's default interpolation-set size (DFTRFilter.__init__)."""
    return min(2 * n + 1, (n + 1) * (n + 2) // 2)


def full_basis_points(n: int) -> int:
    """Size of the complete quadratic basis for n variables."""
    return (n + 1) * (n + 2) // 2


def _sample_set(n: int, p: int, seed: int) -> np.ndarray:
    """A radius-1 sample set of p points around the origin: the center plus
    p-1 random unit directions. Random directions are generically well-poised
    for a degree-2 fit; `_time_case` retries other seeds if one happens not
    to be."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n, p - 1))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    center = np.zeros(n)
    return np.hstack([center[:, None], center[:, None] + dirs])


@dataclass
class TimingRow:
    n: int
    points: int
    policy: str  # "default" or "full-basis"
    fit_ms: Optional[float]
    build_ms: Optional[float]
    total_ms: Optional[float]
    detail: Optional[str] = None


def _time_case(n: int, p: int, policy: str, reps: int, max_seed: int = 30) -> TimingRow:
    for seed in range(max_seed):
        Y = _sample_set(n, p, seed)
        f = np.random.default_rng(seed).standard_normal(p)
        try:
            fit_lagrange_set(Y, radius=1.0).build_model(f)  # validity + warmup
            break
        except (IllPoisedModel, PoisednessIsZeroException):
            continue
    else:
        return TimingRow(n=n, points=p, policy=policy, fit_ms=None, build_ms=None,
                         total_ms=None, detail=f"no poised sample set in {max_seed} tries")

    fit_times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lset = fit_lagrange_set(Y, radius=1.0)
        fit_times.append(time.perf_counter() - t0)

    lset = fit_lagrange_set(Y, radius=1.0)
    build_times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lset.build_model(f)
        build_times.append(time.perf_counter() - t0)

    fit_ms = float(np.median(fit_times)) * 1e3
    build_ms = float(np.median(build_times)) * 1e3
    return TimingRow(n=n, points=p, policy=policy, fit_ms=fit_ms, build_ms=build_ms,
                     total_ms=fit_ms + build_ms)


def run(ns: List[int], reps: int, full_basis: bool = True) -> List[TimingRow]:
    rows: List[TimingRow] = []
    for n in ns:
        rows.append(_time_case(n, default_points(n), "default", reps))
        if full_basis:
            rows.append(_time_case(n, full_basis_points(n), "full-basis", reps))
    return rows


def print_table(rows: List[TimingRow]) -> None:
    def fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    print(f"\n{'n':>4} {'policy':>11} {'points':>7} {'fit(ms)':>10} {'build(ms)':>10} {'total(ms)':>10}  detail")
    for r in rows:
        print(f"{r.n:>4} {r.policy:>11} {r.points:>7} {fmt(r.fit_ms):>10} "
              f"{fmt(r.build_ms):>10} {fmt(r.total_ms):>10}  {r.detail or ''}")


def write_csv(rows: List[TimingRow], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    print(f"\nwrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS,
                        help="dimensions to time (default: 2..50 sampled)")
    parser.add_argument("--reps", type=int, default=25,
                        help="timed repetitions per case; the median is reported")
    parser.add_argument("--no-full-basis", action="store_true",
                        help="time only the library-default 2n+1 point count")
    parser.add_argument("--csv", type=str, default=None,
                        help="optional path to write the per-case timings")
    args = parser.parse_args(argv)

    rows = run(args.ns, args.reps, full_basis=not args.no_full_basis)
    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
