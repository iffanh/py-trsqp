"""Monomial polynomial basis for degree-1 and degree-2 models, in plain NumPy.

Replaces the CasADi-symbolic basis construction of the old ``lagrange_polynomial``
module. Symbolic construction (``ca.simplify``, symbolic ``ca.inv``) was the
dominant cost of model building and scaled very badly with the number of
variables n: it forced a symbolic matrix inversion of a q x q matrix (q =
O(n^2) for a quadratic basis) even though every entry was a plain number.
Everything here is a numeric array operation instead, and the quadratic-term
bookkeeping is done with precomputed index arrays (np.triu_indices) rather
than a per-pair Python loop, so it stays cheap as n grows.

A degree-2 basis in n variables has q = 1 + n + n(n+1)/2 monomials, ordered
[1, d_1, ..., d_n, d_1^2/2, d_1 d_2, ..., d_2^2/2, ...]. The 1/k! scaling on
pure powers (matching Conn-Scheinberg-Vicente's "natural basis", Ch. 4) is
what lets gradient/Hessian be read directly off the solved coefficients: for
alpha in this basis, m(d) = alpha_0 + g^T d + 0.5 d^T H d with g = alpha[1:n+1]
and H built from the quadratic block with no extra factors.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Basis:
    """A degree-1 or degree-2 monomial basis in n variables."""

    n: int
    degree: int
    rows: np.ndarray = field(repr=False)  # quadratic term row indices i (i <= j)
    cols: np.ndarray = field(repr=False)  # quadratic term col indices j
    diag_mask: np.ndarray = field(repr=False)  # True where i == j

    @property
    def n_quad(self) -> int:
        return self.rows.shape[0]

    @property
    def size(self) -> int:
        if self.degree == 1:
            return 1 + self.n
        return 1 + self.n + self.n_quad

    def evaluate(self, D: np.ndarray) -> np.ndarray:
        """Evaluate all basis functions at points D (n x p) -> Phi (p x q)."""
        n, p = D.shape
        assert n == self.n
        Phi = np.empty((p, self.size))
        Phi[:, 0] = 1.0
        Phi[:, 1:1 + n] = D.T
        if self.degree == 2:
            base = 1 + n
            quad = D[self.rows, :] * D[self.cols, :]  # (n_quad, p)
            quad[self.diag_mask, :] *= 0.5
            Phi[:, base:] = quad.T
        return Phi

    def gradient_hessian(self, alpha: np.ndarray):
        """Split monomial coefficients alpha (length q) into (c, g, H)."""
        c = float(alpha[0])
        g = np.array(alpha[1:1 + self.n], dtype=float)
        H = np.zeros((self.n, self.n))
        if self.degree == 2:
            base = 1 + self.n
            vals = alpha[base:]
            H[self.rows, self.cols] = vals
            H[self.cols, self.rows] = vals
        return c, g, H

    def pack_gradient_hessian(self, c: float, g: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Inverse of gradient_hessian: build monomial coefficients from (c, g, H)."""
        alpha = np.empty(self.size)
        alpha[0] = c
        alpha[1:1 + self.n] = g
        if self.degree == 2:
            base = 1 + self.n
            alpha[base:] = H[self.rows, self.cols]
        return alpha


def make_basis(n: int, degree: int) -> Basis:
    if degree not in (1, 2):
        raise ValueError(f"Only degree 1 or 2 supported, got {degree}")
    if degree == 2:
        rows, cols = np.triu_indices(n)
        diag_mask = rows == cols
    else:
        rows = np.empty(0, dtype=int)
        cols = np.empty(0, dtype=int)
        diag_mask = np.empty(0, dtype=bool)
    return Basis(n=n, degree=degree, rows=rows, cols=cols, diag_mask=diag_mask)


def basis_size(n: int, degree: int) -> int:
    if degree == 1:
        return 1 + n
    return 1 + n + n * (n + 1) // 2
