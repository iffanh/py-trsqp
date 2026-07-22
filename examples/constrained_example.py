"""DFTR-Filter: output-constrained example (runnable script).

Minimises  f(x) = 10 (x1-1)^2 + 4 (x2-1)^2  subject to the unit-disc inequality
c(x) = 1 - x1^2 - x2^2 >= 0. The unconstrained minimiser (1, 1) is infeasible,
so the constrained optimum sits on the constraint boundary -- a good visual test
of constraint handling. Inequalities use the paper's convention c(x) >= 0.

Both the objective and the constraint are exactly quadratic, so this run sets
max_points to the full quadratic basis size for n=2 (q = (n+1)(n+2)/2 = 6)
rather than the library default of min(2n+1, q) = 5. With the full basis a
quadratic model reproduces a quadratic function exactly, which matters for the
criticality panel of the convergence plot: the model criticality chi~ is a
first-order stationarity estimate built from the fitted gradients, and its
accuracy is only O(radius) when the model is underdetermined (a
minimum-Frobenius-norm fit with fewer points than the basis). That O(radius)
bias makes chi~ decrease erratically and floor well above zero near the
solution. With the complete, exact model chi~ instead tracks the trust-region
radius down in near-lockstep -- the theoretically expected behaviour of a
first-order criticality measure -- and the run reaches the true optimum to
~7 digits. The extra interpolation point costs one more evaluation per model
build, negligible here but O(n^2) rather than O(n) at larger n, which is why
the library default caps at 2n+1 instead.

Prints a summary, cross-checks against SciPy SLSQP, and writes two figures
(convergence curves and the path with the feasible region shaded), plus an
animated GIF of the trust region/sample set/model evolving, into
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
        # Full quadratic basis (see the module docstring): exact models for
        # these quadratic functions, so the criticality panel tracks the
        # radius cleanly instead of flooring on an O(radius) model bias.
        constants={"max_points": 6},
        record_history=True,    # needed for the trust-region animation below
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

    print("\nwriting animation:")
    viz.animate_trust_region(opt, cf, os.path.join(OUT, "constrained_trust_region.gif"),
                              xlim=(-3, 3), ylim=(-3, 3), ineqs=[disc],
                              solution=[0.7945, 0.6073],
                              title="Constrained: trust region evolution")


if __name__ == "__main__":
    main()
