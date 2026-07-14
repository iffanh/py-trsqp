"""DFTR-Filter: output-constrained example (runnable script).

Minimises  f(x) = 10 (x1-1)^2 + 4 (x2-1)^2  subject to the unit-disc inequality
c(x) = 1 - x1^2 - x2^2 >= 0. The unconstrained minimiser (1, 1) is infeasible,
so the constrained optimum sits on the constraint boundary -- a good visual test
of constraint handling. Inequalities use the paper's convention c(x) >= 0.

Prints a summary, cross-checks against SciPy SLSQP, and writes two figures
(convergence curves and the path with the feasible region shaded) into
examples/results/.

Run from the repo root or from examples/:
    python examples/constrained_example.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from py_trsqp import DFTRFilter
import viz

OUT = os.path.join(HERE, "results")


def cf(x):
    return 10 * (x[0] - 1.0) ** 2 + 4 * (x[1] - 1.0) ** 2


def disc(x):
    """Inequality constraint c(x) >= 0: the closed unit disc."""
    return 1.0 - (x[0] ** 2 + x[1] ** 2)


def main():
    opt = DFTRFilter(
        x0=[-2.0, -2.0],
        cf=cf,
        lb=-3.0,
        ub=3.0,
        ineqcs=[disc],          # list of black-box inequality constraints
    )
    opt.optimize(max_iter=200)

    print("\n=== constrained result ===")
    print(viz.summary(opt))

    # sanity check against a derivative-based solver
    try:
        from scipy.optimize import minimize
        ref = minimize(cf, [0.8, 0.6], constraints=[{"type": "ineq", "fun": disc}])
        print("\nSciPy SLSQP  :", ref.x, " f =", ref.fun)
        print("DFTR-Filter  :", opt.best.x, " f =", opt.best.f)
        print("|x0^2+x1^2-1| at DFTR best:",
              abs(opt.best.x[0] ** 2 + opt.best.x[1] ** 2 - 1.0))
    except Exception as e:  # scipy optional for the cross-check
        print("(skipped SciPy cross-check:", e, ")")

    print("\nwriting figures:")
    viz.save(viz.plot_convergence(opt, "Constrained convergence"),
             os.path.join(OUT, "constrained_convergence.png"))
    viz.save(viz.plot_trajectory_2d(opt, cf, xlim=(-3, 3), ylim=(-3, 3),
                                    ineqs=[disc],
                                    solution=[0.7945, 0.6073],
                                    title="Constrained path (feasible region shaded)"),
             os.path.join(OUT, "constrained_path.png"))
    plt.close("all")


if __name__ == "__main__":
    main()
