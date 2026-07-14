"""Exact trust-region subproblem (TRS) solver, and the poisedness maximization
built on top of it.

Given a quadratic q(d) = c + g^T d + 0.5 d^T H d and a Euclidean ball of the
given radius, this solves

    minimize_d   g^T d + 0.5 d^T H d      s.t.  ||d|| <= radius

exactly (global optimum, including the indefinite-H and "hard" cases) via the
classical secular-equation method (Nocedal & Wright, Algorithm 4.3; the
generalisation of the More-Sorensen algorithm). It replaces the old
``scipy.optimize.minimize(..., method='SLSQP')`` used for poisedness: SLSQP is
a generic local NLP solver with no guarantee of finding the global maximiser
of |lambda_i(x)| over the ball, it silently returns whatever it has after a
single retry on failure, and that is the direct cause of the wrong/NaN
poisedness values -- there is no bad-local-solution or non-convergence case
here, the eigen-based method is closed-form up to a single 1-D root find.

Maximizing |q(d)| over the ball -- which is what Lambda-poisedness needs --
is then just two calls to this solver: one to minimize q, one to minimize
-q, keeping whichever has the larger magnitude.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _solve_secular(eigvals: np.ndarray, ghat: np.ndarray, radius: float, tol: float = 1e-12) -> float:
    """Find sigma >= max(0, -lambda_min) with ||d(sigma)|| = radius.

    d(sigma)_i = -ghat_i / (eigvals_i + sigma). Handles the hard case where
    ghat has (numerically) no component along the smallest eigenspace.
    """
    lam_min = eigvals[0]
    lo = max(0.0, -lam_min)

    def norm_at(sigma: float) -> float:
        denom = eigvals + sigma
        # Components at the boundary (denom ~ 0) are handled by the hard-case
        # branch below; here we only evaluate strictly inside the safe region.
        return np.linalg.norm(ghat / denom)

    # Hard case check: sigma -> lo from above. If the components of ghat along
    # eigenvectors with eigenvalue == lam_min are all ~0, d(sigma) stays finite
    # as sigma -> lo, and if that finite norm is still < radius, the boundary
    # eigenspace can absorb any extra length freely (hard case).
    at_boundary = np.abs(eigvals - lam_min) < 1e-10 * max(1.0, abs(lam_min))
    ghat_boundary_norm = np.linalg.norm(ghat[at_boundary])
    if lam_min < 0 and ghat_boundary_norm < 1e-10 * max(1.0, np.linalg.norm(ghat)):
        denom = eigvals[~at_boundary] + lo
        finite_norm_sq = float(np.sum((ghat[~at_boundary] / denom) ** 2)) if denom.size else 0.0
        if finite_norm_sq <= radius ** 2:
            return lo  # hard case: sigma = -lambda_min exactly

    # Otherwise phi(sigma) = 1/||d(sigma)|| - 1/radius is monotonically
    # increasing on (lo, inf) from -inf (or a finite negative value) to 0.
    # Bracket the root by doubling.
    hi = max(lo + 1.0, 2.0 * abs(lo) + 1.0)
    n_at = norm_at(hi)
    while n_at > radius and hi < 1e18:
        hi *= 2.0
        n_at = norm_at(hi)

    eps = max(1e-14, 1e-8 * max(1.0, abs(lo)))
    lo_eval = lo + eps
    while norm_at(lo_eval) < radius and lo_eval < hi:
        # Extremely rare: even just off the boundary the norm undershoots.
        eps *= 10
        lo_eval = lo + eps

    def phi(sigma: float) -> float:
        return norm_at(sigma) - radius

    if phi(lo_eval) <= 0:
        return lo_eval
    return brentq(phi, lo_eval, hi, xtol=tol, maxiter=200)


def solve_tr_subproblem(g: np.ndarray, H: np.ndarray, radius: float) -> np.ndarray:
    """Global solution of min g^T d + 0.5 d^T H d s.t. ||d|| <= radius."""
    n = g.shape[0]
    if radius <= 0:
        return np.zeros(n)
    if n == 1:
        h = H[0, 0]
        if h > 0:
            d = -g[0] / h
            if abs(d) <= radius:
                return np.array([d])
        return np.array([-radius if g[0] > 0 else radius])

    eigvals, eigvecs = np.linalg.eigh((H + H.T) / 2.0)
    ghat = eigvecs.T @ g

    lam_min = eigvals[0]
    if lam_min > 1e-12:
        d_u = -eigvecs @ (ghat / eigvals)
        if np.linalg.norm(d_u) <= radius:
            return d_u

    sigma = _solve_secular(eigvals, ghat, radius)
    denom = eigvals + sigma
    # Guard against near-zero denominators in the hard case: the corresponding
    # ghat components are ~0 there, so 0/~0 is taken as 0 and any remaining
    # budget is put along that null eigenvector to land exactly on the sphere.
    safe = np.abs(denom) > 1e-10
    coeff = np.zeros(n)
    coeff[safe] = -ghat[safe] / denom[safe]
    d = eigvecs @ coeff
    resid = radius ** 2 - float(d @ d)
    if resid > 1e-12:
        null_dirs = eigvecs[:, ~safe]
        if null_dirs.shape[1] > 0:
            d = d + null_dirs[:, 0] * np.sqrt(resid)
    return d


def maximize_abs_quadratic(c: float, g: np.ndarray, H: np.ndarray, radius: float):
    """Maximize |c + g^T d + 0.5 d^T H d| over ||d|| <= radius.

    Returns (d_star, value_at_d_star).
    """
    d_min = solve_tr_subproblem(g, H, radius)
    d_max = solve_tr_subproblem(-g, -H, radius)

    val_min = c + g @ d_min + 0.5 * d_min @ H @ d_min
    val_max = c + g @ d_max + 0.5 * d_max @ H @ d_max

    if abs(val_min) >= abs(val_max):
        return d_min, float(val_min)
    return d_max, float(val_max)
