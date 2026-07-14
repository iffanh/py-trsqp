"""Quadratic (or linear) interpolation models built by plain NumPy linear
algebra, replacing the CasADi-symbolic construction of the old
``lagrange_polynomial.LagrangePolynomials``.

Mirrors the paper's "Models" section (Sec. 1) and the fully-linear model
definition (Def. 1, CSV09 Def. 6.1): given a sample set X_k, this builds
m_k(x_k + s) = f~_k(x_k) + g_k^T s + 0.5 s^T H_k s and the accompanying
Lagrange polynomials, needed for poisedness/Repoise (Algorithm 1).

All interpolation conditions are solved in *normalized* coordinates
d = (y - center) / radius (so the sample ball becomes the unit ball, which
keeps the linear systems well conditioned regardless of the problem's
absolute scale), then converted back to original x-space gradients/Hessians
(g = g_n / radius, H = H_n / radius^2) so every other module works directly
in the paper's x-space units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .basis import Basis, make_basis, basis_size
from ..exceptions import IllPoisedModel, PoisednessIsZeroException
from . import trs

_COND_LIMIT = 1e10


@dataclass
class QuadraticModel:
    """m(x) = c + g^T(x - center) + 0.5 (x - center)^T H (x - center)."""

    c: float
    g: np.ndarray
    H: np.ndarray
    center: np.ndarray

    def value(self, x: np.ndarray) -> float:
        d = x - self.center
        return float(self.c + self.g @ d + 0.5 * d @ self.H @ d)

    def gradient(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        if x is None:
            return self.g
        return self.g + self.H @ (x - self.center)

    def hessian(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        return self.H


@dataclass
class LagrangeSet:
    """A fitted Lagrange basis on a sample set, reusable across every
    quantity interpolated on the same points (cost, each constraint, ...) --
    this is the same "compute the geometry once, evaluate many right-hand
    sides" reuse the old code did via ``lpolynomials=cfm.model....``, kept
    here in the numeric representation.
    """

    basis: Basis
    center: np.ndarray
    radius: float
    Y: np.ndarray  # (n, p) sample points in x-space
    Lambda: np.ndarray  # (p, q): row i = normalized-coordinate coefficients of lambda_i

    @property
    def n(self) -> int:
        return self.basis.n

    @property
    def p(self) -> int:
        return self.Y.shape[1]

    def build_model(self, f: np.ndarray) -> QuadraticModel:
        f = np.asarray(f, dtype=float)
        alpha_n = f @ self.Lambda  # (q,)
        c_n, g_n, H_n = self.basis.gradient_hessian(alpha_n)
        g = g_n / self.radius
        H = H_n / (self.radius ** 2)
        return QuadraticModel(c=c_n, g=g, H=H, center=self.center.copy())

    def lagrange_polynomial_normalized(self, i: int):
        """(c_i, g_i, H_i) of lambda_i in normalized coordinates d = (x-center)/radius."""
        return self.basis.gradient_hessian(self.Lambda[i, :])

    def poisedness(self, radius: Optional[float] = None):
        """Lambda-poisedness of the set on B(center, radius) (Def. of Sec. 4.2/6.2).

        Returns a ``Poisedness`` object. Uses the exact TRS-based quadratic
        maximizer (core.trs), never a generic local NLP solve -- this is what
        keeps poisedness values finite and reproducible (no NaNs, no
        SLSQP-failure garbage).
        """
        rad = 1.0 if radius is None else radius / self.radius
        values = np.empty(self.p)
        candidates = np.empty((self.n, self.p))
        for i in range(self.p):
            c_i, g_i, H_i = self.lagrange_polynomial_normalized(i)
            d_star, val = trs.maximize_abs_quadratic(c_i, g_i, H_i, rad)
            values[i] = abs(val)
            candidates[:, i] = self.center + self.radius * d_star

        if not np.any(values > 0):
            raise PoisednessIsZeroException(
                "Every Lagrange polynomial vanishes on the ball; the sample set is degenerate."
            )
        index = int(np.argmax(values))
        return Poisedness(index=index, values=values, candidate_points=candidates)


@dataclass
class Poisedness:
    index: int
    values: np.ndarray
    candidate_points: np.ndarray  # (n, p): column i is the maximizer of |lambda_i|

    def max_poisedness(self) -> float:
        return float(self.values[self.index])

    def point_to_max_poisedness(self) -> np.ndarray:
        return self.candidate_points[:, self.index]


def choose_degree(n: int, p: int) -> int:
    q1 = basis_size(n, 1)
    return 1 if p <= q1 else 2


def _ball_radius(Y: np.ndarray) -> float:
    center = Y[:, [0]]
    if Y.shape[1] == 1:
        return 1.0
    r = float(np.max(np.linalg.norm(Y - center, axis=0)))
    return r if r > 0 else 1.0


def _solve_square(Phi: np.ndarray) -> np.ndarray:
    p = Phi.shape[0]
    cond = np.linalg.cond(Phi)
    if not np.isfinite(cond) or cond > _COND_LIMIT:
        raise IllPoisedModel(f"Interpolation matrix is ill-conditioned (cond={cond:.3e}).")
    try:
        Phi_inv = np.linalg.solve(Phi, np.eye(p))
    except np.linalg.LinAlgError as e:
        raise IllPoisedModel(f"Interpolation matrix is singular: {e}")
    return Phi_inv.T


def _solve_least_norm(Phi: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(Phi).T


def _solve_frobenius(Phi: np.ndarray, n: int) -> np.ndarray:
    """Minimum-Frobenius-norm Lagrange polynomials (CSV09 Ch. 5, eq. 5.7-5.9),
    solved with a single dense linear solve instead of a symbolic matrix
    inverse -- the actual math is unchanged, only the arithmetic backend is.
    """
    p = Phi.shape[0]
    L = n + 1
    Phi_L = Phi[:, :L]
    Phi_Q = Phi[:, L:]

    A = Phi_Q @ Phi_Q.T
    F = np.block([[A, Phi_L], [Phi_L.T, np.zeros((L, L))]])
    rhs = np.vstack([np.eye(p), np.zeros((L, p))])

    cond = np.linalg.cond(F)
    if not np.isfinite(cond) or cond > _COND_LIMIT:
        raise IllPoisedModel(f"Frobenius-norm system is ill-conditioned (cond={cond:.3e}).")

    try:
        X = np.linalg.solve(F, rhs)
    except np.linalg.LinAlgError:
        X, *_ = np.linalg.lstsq(F, rhs, rcond=None)

    M = X[:p, :]
    V = X[p:, :]
    Lambda_L = V.T
    Lambda_Q = M.T @ Phi_Q
    return np.hstack([Lambda_L, Lambda_Q])


def fit_lagrange_set(Y: np.ndarray, radius: Optional[float] = None, degree: Optional[int] = None) -> LagrangeSet:
    """Build a LagrangeSet (Lagrange basis + geometry) on sample points Y.

    Y: (n, p) array, column 0 is the center/incumbent.
    radius: ball radius to normalize by; defaults to the sample set's own
        diameter from the center (matching the old ``SampleSets`` behaviour).
    degree: 1 or 2; defaults to 2 once there are enough points.
    """
    n, p = Y.shape
    center = Y[:, 0].copy()
    rad = radius if radius is not None else _ball_radius(Y)
    if rad <= 0:
        rad = 1.0

    if degree is None:
        degree = choose_degree(n, p)

    basis = make_basis(n, degree)
    q = basis.size
    if p > q:
        raise ValueError(
            f"Too many points ({p}) for a degree-{degree} model in {n} dims "
            f"(max {q}); cap the sample set before fitting."
        )

    D = (Y - center[:, None]) / rad
    Phi = basis.evaluate(D)

    if p == q:
        Lambda = _solve_square(Phi)
    elif degree == 2:
        Lambda = _solve_frobenius(Phi, n)
    else:
        Lambda = _solve_least_norm(Phi)

    return LagrangeSet(basis=basis, center=center, radius=rad, Y=Y.copy(), Lambda=Lambda)
