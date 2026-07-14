"""User-facing problem wrapping: bound normalization and simulation bookkeeping.

x is normalized to xn via a per-coordinate affine map so that xn in
[-1, 1]^n exactly when x is inside [lb, ub] -- this keeps every model-fit
and subproblem solve numerically well-scaled regardless of the problem's
native units, matching the paper's own normalisation convention (Eq. 3)
without depending on it being enforced by the caller.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from ..exceptions import IncorrectInputException, FailedSimulation


def _as_bounds(ub, lb, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(ub, (list, tuple, np.ndarray)) or isinstance(lb, (list, tuple, np.ndarray)):
        if not (isinstance(ub, (list, tuple, np.ndarray)) and isinstance(lb, (list, tuple, np.ndarray))):
            raise IncorrectInputException(f"ub and lb must both be scalars or both be sequences; got {type(ub)}, {type(lb)}")
        ub_arr = np.asarray(ub, dtype=float)
        lb_arr = np.asarray(lb, dtype=float)
        if ub_arr.shape[0] != lb_arr.shape[0]:
            raise IncorrectInputException(f"len(ub)={ub_arr.shape[0]} != len(lb)={lb_arr.shape[0]}")
        if ub_arr.shape[0] != n:
            raise IncorrectInputException(f"len(ub)={ub_arr.shape[0]} != len(x0)={n}")
    else:
        ub_arr = np.full(n, float(ub))
        lb_arr = np.full(n, float(lb))

    if not (np.all(np.isfinite(ub_arr)) and np.all(np.isfinite(lb_arr))):
        raise IncorrectInputException(
            "Bounds must be finite in every coordinate (normalization divides by the box "
            "half-widths); pass a generously large finite box instead of +/-inf."
        )
    if np.any(lb_arr >= ub_arr):
        raise IncorrectInputException("Each lower bound must be strictly less than the matching upper bound.")
    return ub_arr, lb_arr


class Problem:
    """Normalizes x <-> xn in [-1,1]^n and wraps the user's cf/eqcs/ineqcs
    with NaN/Inf checking and an evaluation counter."""

    def __init__(self, x0: np.ndarray, cf: Callable, ub, lb,
                 eqcs: Optional[List[Callable]] = None, ineqcs: Optional[List[Callable]] = None):
        x0 = np.asarray(x0, dtype=float)
        n = x0.shape[0]
        ub_arr, lb_arr = _as_bounds(ub, lb, n)

        self.n = n
        self.ub = ub_arr
        self.lb = lb_arr
        self.center = (ub_arr + lb_arr) / 2.0
        self.half_hi = ub_arr - self.center
        self.half_lo = self.center - lb_arr

        self.x0n = self.normalize(x0)
        self.ubn = self.normalize(ub_arr)
        self.lbn = self.normalize(lb_arr)

        self.cf = cf
        self.eqcs = list(eqcs) if eqcs else []
        self.ineqcs = list(ineqcs) if ineqcs else []
        self.n_eq = len(self.eqcs)
        self.n_ineq = len(self.ineqcs)

        self.n_cf_evals = 0
        self._seen = set()

    def normalize(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        above = x > self.center
        out[above] = (x[above] - self.center[above]) / self.half_hi[above]
        out[~above] = (x[~above] - self.center[~above]) / self.half_lo[~above]
        return out

    def denormalize(self, xn: np.ndarray) -> np.ndarray:
        out = np.empty_like(xn, dtype=float)
        above = xn > 0
        out[above] = xn[above] * self.half_hi[above] + self.center[above]
        out[~above] = xn[~above] * self.half_lo[~above] + self.center[~above]
        return out

    def in_bounds(self, xn: np.ndarray, tol: float = 1e-6) -> bool:
        return bool(np.all(xn >= self.lbn - tol) and np.all(xn <= self.ubn + tol))

    def _check(self, val: float, label: str, x: np.ndarray) -> float:
        val = float(val)
        if not np.isfinite(val):
            raise FailedSimulation(f"{label} returned a non-finite value at x={x}")
        return val

    def evaluate(self, xn: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Returns (f, eq_values, ineq_values) at normalized point xn, counting
        exactly one cost-function evaluation per distinct point (matching
        Remark 11's "one simulation per ordinary iteration" accounting)."""
        x = self.denormalize(xn)
        key = xn.tobytes()
        if key not in self._seen:
            self._seen.add(key)
            self.n_cf_evals += 1
        f = self._check(self.cf(x), "cost function", x)
        eqs = np.array([self._check(fn(x), f"equality constraint {i}", x) for i, fn in enumerate(self.eqcs)])
        ineqs = np.array([self._check(fn(x), f"inequality constraint {i}", x) for i, fn in enumerate(self.ineqcs)])
        return f, eqs, ineqs
