"""Initial sample-set generation.

The old ``generate_poised_sets``/``model_improvement_without_feval`` sphere
sampler rebuilt full symbolic ``LagrangePolynomials`` objects and updated
them via repeated CasADi ``Function`` reconstruction for every point of an
n-sphere design -- extremely slow, and only ever used to build an *initial*
set. A minimal (n+1)-point simplex is a perfectly well-poised starting point
for a linear model (CSV09's own recommendation); the algorithm's own
Repoise/step machinery grows and re-poises the set from there as trial
points are accepted, so nothing else needs to happen at start-up.
"""

from __future__ import annotations

import numpy as np


def simplex_directions(n: int) -> np.ndarray:
    """n directions (n, n), positive-spanning together with the origin, via
    the standard Cholesky construction of a regular simplex edge set."""
    if n == 1:
        return np.array([[1.0]])
    a = -1.0 / n
    A = np.full((n, n), a)
    np.fill_diagonal(A, 1.0)
    C = np.linalg.cholesky(A)
    return C.T


def initial_sample_set(center: np.ndarray, radius: float) -> np.ndarray:
    """(n, n+1) sample set: column 0 is `center`, the rest are a well-poised
    simplex of points at distance `radius` from it."""
    n = center.shape[0]
    D = simplex_directions(n)
    Y = center[:, None] + radius * D
    return np.hstack([center[:, None], Y])
