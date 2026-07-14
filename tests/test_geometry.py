import numpy as np

from py_trsqp.core import geometry
from py_trsqp.core.interpolation import QuadraticModel


def _linear_model(c, g, n):
    return QuadraticModel(c=c, g=np.array(g, dtype=float), H=np.zeros((n, n)), center=np.zeros(n))


def test_consistent_and_compatible_when_feasible_region_reachable():
    n = 2
    # x0 + x1 - 1 >= 0  (one inequality; consistent and compatible for a
    # reasonably sized trust region)
    ineq = [_linear_model(c=-1.0, g=[1.0, 1.0], n=n)]
    geom = geometry.build_geometry(n, eq_models=[], ineq_models=ineq)
    assert geometry.is_consistent(geom)
    result = geometry.compatibility(geom, radius=1.0)
    assert result.is_compatible
    assert result.dist0 >= 0


def test_incompatible_when_violation_far_exceeds_radius():
    n = 2
    # c(x) = x0 + x1 - 10 >= 0 at the origin is a long way from feasible.
    ineq = [_linear_model(c=-10.0, g=[1.0, 1.0], n=n)]
    geom = geometry.build_geometry(n, eq_models=[], ineq_models=ineq)
    result = geometry.compatibility(geom, radius=0.1)
    assert not result.is_compatible


def test_criticality_measure_matches_l1_norm_for_unconstrained():
    n = 3
    geom = geometry.build_geometry(n, eq_models=[], ineq_models=[])
    g = np.array([1.0, -2.0, 0.5])
    chi = geometry.criticality_measure(geom, g)
    assert chi == np.sum(np.abs(g))


def test_regularity_margin_is_one_for_identity_only():
    n = 2
    geom = geometry.build_geometry(n, eq_models=[], ineq_models=[])
    assert geometry.regularity_margin(geom) == 1.0
