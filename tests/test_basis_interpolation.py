import numpy as np
import pytest

from py_trsqp.core.basis import make_basis, basis_size
from py_trsqp.core.interpolation import fit_lagrange_set
from py_trsqp.exceptions import IllPoisedModel

rng = np.random.default_rng(0)


def _kron_error(lset):
    D = (lset.Y - lset.center[:, None]) / lset.radius
    Phi = lset.basis.evaluate(D)
    return np.max(np.abs(Phi @ lset.Lambda.T - np.eye(lset.Y.shape[1])))


def test_square_quadratic_recovers_exact_model():
    n = 4
    q2 = basis_size(n, 2)
    x0 = rng.normal(size=n)
    g_true = rng.normal(size=n)
    A = rng.normal(size=(n, n))
    H_true = A + A.T
    c_true = rng.normal()

    Y = x0[:, None] + np.hstack([np.zeros((n, 1)), rng.normal(size=(n, q2 - 1)) * 0.3])
    f = np.array([c_true + g_true @ (Y[:, i] - x0) + 0.5 * (Y[:, i] - x0) @ H_true @ (Y[:, i] - x0)
                  for i in range(Y.shape[1])])

    lset = fit_lagrange_set(Y)
    assert lset.basis.degree == 2
    assert _kron_error(lset) < 1e-6

    model = lset.build_model(f)
    assert abs(model.c - c_true) < 1e-6
    assert np.max(np.abs(model.gradient() - g_true)) < 1e-6
    assert np.max(np.abs(model.hessian() - H_true)) < 1e-6


def test_underdetermined_quadratic_still_interpolates_exactly():
    n = 5
    p = n + 3
    center = rng.normal(size=n)
    Y = center[:, None] + np.hstack([np.zeros((n, 1)), rng.normal(size=(n, p - 1)) * 0.4])
    f = rng.normal(size=p)

    lset = fit_lagrange_set(Y, degree=2)
    assert _kron_error(lset) < 1e-6

    model = lset.build_model(f)
    reproduced = np.array([model.value(Y[:, i]) for i in range(p)])
    assert np.max(np.abs(reproduced - f)) < 1e-6


def test_linear_model_exact():
    n = 6
    center = rng.normal(size=n)
    Y = center[:, None] + np.hstack([np.zeros((n, 1)), rng.normal(size=(n, n)) * 0.5])
    g_true = rng.normal(size=n)
    c_true = rng.normal()
    f = np.array([c_true + g_true @ (Y[:, i] - center) for i in range(Y.shape[1])])

    lset = fit_lagrange_set(Y)
    assert lset.basis.degree == 1
    model = lset.build_model(f)
    assert abs(model.c - c_true) < 1e-6
    assert np.max(np.abs(model.gradient() - g_true)) < 1e-6


def test_poisedness_has_no_nan_and_identifies_worst_point():
    n = 3
    q2 = basis_size(n, 2)
    center = np.zeros(n)
    Y = center[:, None] + np.hstack([np.zeros((n, 1)), rng.normal(size=(n, q2 - 1)) * 0.6])
    lset = fit_lagrange_set(Y, radius=1.0)
    poised = lset.poisedness()
    assert np.all(np.isfinite(poised.values))
    assert 0 <= poised.index < Y.shape[1]


def test_duplicate_points_raise_illposed_not_nan():
    n = 2
    Y = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])  # last two columns identical
    with pytest.raises(IllPoisedModel):
        fit_lagrange_set(Y)
