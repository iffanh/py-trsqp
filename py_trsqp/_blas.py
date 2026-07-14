"""Caps BLAS/LAPACK thread pools for this process.

Every linear-algebra call in this package (model fitting, eigendecompositions
in the trust-region subproblem solver, projections, ...) operates on small
dense matrices sized by the number of interpolation points or variables --
never anything close to what would benefit from multi-threaded BLAS. On
several platforms (observed here under WSL with OpenBLAS) the thread-pool
dispatch/sync overhead on such small matrices dominates the actual FLOPs: a
single 100x100 ``numpy.linalg.eigh`` measured ~65ms with OpenBLAS's default
thread count and ~0.5ms capped to one thread -- a 100x difference that has
nothing to do with the algorithm and everything to do with thread
oversubscription. Since DFTR-Filter calls into these routines many times per
iteration (once per Lagrange polynomial for poisedness, for every model
rebuild, ...), this was likely a dominant, invisible cost.

``threadpoolctl`` is used (rather than the OPENBLAS_NUM_THREADS/OMP_NUM_THREADS
environment variables) because it works even when numpy/scipy were already
imported by the host application before this package is loaded -- env vars
only take effect at library load time.
"""

from __future__ import annotations

import os

_LIMIT_ENV = "PY_TRSQP_BLAS_THREADS"
_applied = False


def cap_blas_threads(num_threads: int = 1) -> None:
    """Limit BLAS/LAPACK thread pools to ``num_threads`` (idempotent).

    Set the environment variable ``PY_TRSQP_BLAS_THREADS=0`` before import to
    opt out (e.g. if a caller is fitting unusually large models and has
    profiled that more threads actually help).
    """
    global _applied
    if _applied:
        return
    override = os.environ.get(_LIMIT_ENV)
    if override is not None and int(override) <= 0:
        _applied = True
        return
    if override is not None:
        num_threads = int(override)
    try:
        import threadpoolctl
    except ImportError:
        _applied = True
        return
    threadpoolctl.threadpool_limits(num_threads)
    _applied = True
