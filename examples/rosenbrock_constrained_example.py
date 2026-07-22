"""DFTR-Filter: output-constrained Rosenbrock example (runnable script).

Minimises the classic Rosenbrock function f(x) = 100 (x2 - x1^2)^2 + (1 - x1)^2
subject to the unit-disc inequality c(x) = 1 - x1^2 - x2^2 >= 0 (the same
constraint constrained_example.py uses, paper convention c(x) >= 0). The
unconstrained minimiser (1, 1) lies outside the disc, so the constrained
optimum sits on the constraint boundary.

The starting point x0 = (1.5, 1.5) is itself infeasible (outside the disc):
this is deliberate, to demonstrate the restoration phase recovering a
feasible, locally optimal point from an infeasible start rather than only
showing the easier case of a feasible starting point.

Prints a summary, cross-checks against SciPy SLSQP, and writes two figures
(convergence curves and the path with the feasible region shaded), plus an
animated GIF of the trust region/sample set/model evolving, into
examples/results/.

Run from the repo root or from examples/:
    python examples/rosenbrock_constrained_example.py
"""

import os
import sys

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
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def disc(x):
    """Inequality constraint c(x) >= 0: the closed unit disc."""
    return 1.0 - (x[0] ** 2 + x[1] ** 2)


def main():
    opt = DFTRFilter(
        x0=[1.5, 1.5],          # infeasible start: disc(1.5, 1.5) = -3.5 < 0
        cf=cf,
        lb=-2.0,
        ub=2.0,
        ineqcs=[disc],          # list of black-box inequality constraints
        record_history=True,    # needed for the trust-region animation below
    )
    opt.optimize(max_iter=500)

    print("\n=== constrained Rosenbrock result ===")
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
    viz.save(viz.plot_convergence(opt, "Rosenbrock: constrained convergence"),
             os.path.join(OUT, "rosenbrock_constrained_convergence.png"))
    viz.save(viz.plot_trajectory_2d(opt, cf, xlim=(-2, 2), ylim=(-2, 2),
                                    ineqs=[disc],
                                    solution=[0.7864, 0.6177],
                                    title="Rosenbrock: constrained path (feasible region shaded)"),
             os.path.join(OUT, "rosenbrock_constrained_path.png"))
    plt.close("all")

    print("\nwriting animation:")
    viz.animate_trust_region(opt, cf, os.path.join(OUT, "rosenbrock_constrained_trust_region.gif"),
                              xlim=(-2, 2), ylim=(-2, 2), ineqs=[disc],
                              solution=[0.7864, 0.6177],
                              title="Rosenbrock (constrained): trust region evolution")


if __name__ == "__main__":
    main()
