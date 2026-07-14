"""Procedure REPOISE (Algorithm 1) -- purely geometric sample-set repair.

Consumes no simulations: the exchange rule only looks at the *coordinates*
of the sample set (via the Lagrange basis), never function values. The
incumbent (column 0 of Y, i.e. x_k) is never evicted -- Crit/Repoise "spend
geometry, not progress" (Fig. 3's caption): the current iterate must not
move during a criticality/model-improvement pass.
"""

from __future__ import annotations

import numpy as np

from .interpolation import fit_lagrange_set
from ..exceptions import IllPoisedModel, PoisednessIsZeroException


def repoise(Y: np.ndarray, radius: float, lambda_th: float, max_iter: int = 50) -> np.ndarray:
    Y = Y.copy()
    for _ in range(max_iter):
        try:
            lset = fit_lagrange_set(Y, radius=radius)
            poised = lset.poisedness(radius=radius)
        except (IllPoisedModel, PoisednessIsZeroException):
            break

        if poised.max_poisedness() <= lambda_th:
            break

        idx = poised.index
        if idx == 0:
            order = np.argsort(-poised.values)
            candidates = [int(i) for i in order if i != 0]
            if not candidates:
                break
            idx = candidates[0]

        Y[:, idx] = poised.candidate_points[:, idx]

    return Y
