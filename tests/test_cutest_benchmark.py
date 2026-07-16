"""Regression smoke test: run DFTR-Filter against a small curated set of
CUTEst problems (via pycutest) and check the success rate hasn't regressed.

Skipped entirely unless a full CUTEst + SIFDecode + pycutest toolchain is
set up (ARCHDEFS/SIFDECODE/CUTEST/MASTSIF env vars + `pycutest` importable).
See DEVELOPMENT.md, "CUTEst benchmark setup", to build that toolchain; for
a broader sweep with performance metrics use benchmarks/cutest_benchmark.py.
"""

import pytest

from benchmarks.cutest_benchmark import (
    SMOKE_PROBLEMS,
    cutest_available,
    run_suite,
    summarize,
)

pytestmark = pytest.mark.skipif(
    not cutest_available(),
    reason="CUTEst/pycutest not set up (see DEVELOPMENT.md 'CUTEst benchmark setup')",
)


def test_cutest_smoke_suite_success_rate():
    results = run_suite(SMOKE_PROBLEMS, n_max=20, m_max=15, max_iter=300,
                         tol=0.1, timeout=60.0, verbose=False)

    errors = [r for r in results if r.status == "error"]
    assert not errors, f"unexpected errors: {[(r.name, r.detail) for r in errors]}"

    summary = summarize(results)
    assert summary["completed"] == len(SMOKE_PROBLEMS)
    assert summary["success_rate_of_completed"] >= 0.55
