import numpy as np
import pytest

from py_trsqp import DFTRFilter
from py_trsqp.exceptions import IncorrectInputException


def simple(x):
    return 10 * (x[0] - 1.0) ** 2 + 4 * (x[1] - 1.0) ** 2


def simple_but_many(x):
    return sum(((10 if i % 2 == 0 else 4) * (x[i] - 1.0) ** 2) for i in range(len(x)))


def eq1(x):
    return x[0] - x[1] - 1


def ineq4(x):
    return x[0] + x[1] - 1


def ineq5(x):
    return -x[0] - x[1] - 1


def test_bound_shape_mismatches_raise():
    with pytest.raises(IncorrectInputException):
        DFTRFilter(x0=[-2.5, -1.0], cf=simple, ub=[3], lb=[-1])
    with pytest.raises(IncorrectInputException):
        DFTRFilter(x0=[-2.5, -1.0], cf=simple, ub=[3], lb=[-1, 1])
    with pytest.raises(IncorrectInputException):
        DFTRFilter(x0=[-2.5, -1.0], cf=simple, ub=3, lb=[-1])


def test_unconstrained_quadratic_converges():
    opt = DFTRFilter(x0=[-2.5, -2.0], cf=simple, ub=3.0, lb=-3.0)
    opt.optimize(max_iter=150)
    assert opt.best.x[0] == pytest.approx(1.0, abs=1e-3)
    assert opt.best.x[1] == pytest.approx(1.0, abs=1e-3)


def test_bound_constrained_optimum_at_corner():
    opt = DFTRFilter(x0=[-2.5, -2.0], cf=simple, ub=-1.9, lb=-5.0)
    opt.optimize(max_iter=100)
    assert opt.best.x[0] == pytest.approx(-1.9, abs=1e-2)
    assert opt.best.x[1] == pytest.approx(-1.9, abs=1e-2)


def test_equality_constraint_is_respected():
    opt = DFTRFilter(x0=[-2.5, -2.0], cf=simple, ub=3.0, lb=-3.0, eqcs=[eq1])
    opt.optimize(max_iter=200)
    sol0, sol1 = 36 / 28, 8 / 28
    assert opt.best.vartheta == pytest.approx(0.0, abs=1e-6)
    assert opt.best.x[0] == pytest.approx(sol0, abs=0.05)
    assert opt.best.x[1] == pytest.approx(sol1, abs=0.05)


def test_infeasible_problem_reports_stationary_point():
    opt = DFTRFilter(x0=[1.0, -1.0], cf=simple, ub=3.0, lb=-3.0, ineqcs=[ineq4, ineq5])
    opt.optimize(max_iter=100)
    assert opt.termination_status == "Infeasible stationary point (restoration failed)"
    assert opt.best.vartheta == pytest.approx(1.0, abs=1e-3)


def test_scales_to_many_variables():
    n = 10
    opt = DFTRFilter(x0=[-2.0] * n, cf=simple_but_many, ub=3.0, lb=-3.0)
    opt.optimize(max_iter=200)
    assert opt.best.f < 1e-6
    for i in range(n):
        assert opt.best.x[i] == pytest.approx(1.0, abs=1e-2)


def test_budget_is_enforced():
    opt = DFTRFilter(x0=[-2.5, -2.0], cf=simple, ub=3.0, lb=-3.0)
    opt.optimize(max_iter=50, budget=5)
    assert opt.termination_status == "Budget exceeded"
    assert opt.problem.n_cf_evals <= 6  # a small overshoot from the in-flight iteration is fine
