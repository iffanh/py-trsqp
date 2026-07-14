from py_trsqp.core.filter import Filter


def test_feasible_point_never_enters_filter():
    """Regression test for the paper's Lemma 4 / Remark 7: a (vartheta=0, f)
    pair must never be inserted, since it would permanently block every
    future point with f' >= f and deadlock the method at a feasible,
    non-optimal point. The old utils/filter.py had no such guard."""
    filt = Filter(gamma_vartheta=1e-4)
    added = filt.add(f=10.0, vartheta=0.0)
    assert added is False
    assert filt.entries == []


def test_infeasible_point_is_added_and_blocks_dominated_points():
    filt = Filter(gamma_vartheta=0.1)
    assert filt.add(f=10.0, vartheta=1.0) is True
    assert len(filt.entries) == 1
    # A point with both worse f and worse vartheta must be rejected.
    assert filt.is_acceptable(f=11.0, vartheta=1.05) is False
    # A point with much better vartheta is accepted regardless of f.
    assert filt.is_acceptable(f=100.0, vartheta=0.05) is True
    # A point with much better f is accepted regardless of vartheta.
    assert filt.is_acceptable(f=1.0, vartheta=1.05) is True


def test_extra_pairs_are_respected_for_restore():
    filt = Filter(gamma_vartheta=0.1)
    # Simulates Restore checking against F cup {(vartheta_k, f_k)} without
    # mutating the real filter: vartheta=0.95 > (1-g)*1.0=0.9 and
    # f=5.0 > f_i - g*vartheta = 4.0 - 0.095 = 3.905, so both disjuncts of
    # (7) fail against the extra pair and the point must be rejected.
    assert filt.is_acceptable(f=5.0, vartheta=0.95, extra=[(1.0, 4.0)]) is False
    assert filt.entries == []
