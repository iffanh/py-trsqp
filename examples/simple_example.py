"""DFTR-Filter: unconstrained example (runnable script).

Minimises the boxed quadratic  f(x) = 10 (x1-1)^2 + 4 (x2-1)^2,  -3 <= xi <= 3,
whose minimiser is x* = (1, 1), f* = 0. Prints a summary and writes two figures
(convergence curves and the optimization path), plus an animated GIF of the
trust region/sample set/model evolving, into examples/results/.

Run from the repo root or from examples/:
    python examples/simple_example.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")            # headless: write PNGs, no display needed
import matplotlib.pyplot as plt

# make py_trsqp importable and load the local plotting helpers, regardless of cwd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root -> import py_trsqp
sys.path.insert(0, HERE)                     # examples/  -> import viz

from py_trsqp import DFTRFilter
import viz

OUT = os.path.join(HERE, "results")


def cf(x):
    """Objective. Works on scalars/arrays and on a mesh grid (for plotting)."""
    return 10 * (x[0] - 1.0) ** 2 + 4 * (x[1] - 1.0) ** 2


def main():
    opt = DFTRFilter(
        x0=[-2.5, -2.0],
        cf=cf,
        lb=-3.0,        # scalar bounds broadcast to every coordinate
        ub=3.0,
        record_history=True,   # needed for the trust-region animation below
    )
    opt.optimize(max_iter=200)

    print("\n=== unconstrained result ===")
    print(viz.summary(opt))

    print("\nwriting figures:")
    viz.save(viz.plot_convergence(opt, "Unconstrained convergence"),
             os.path.join(OUT, "simple_convergence.png"))
    viz.save(viz.plot_trajectory_2d(opt, cf, xlim=(-3, 3), ylim=(-3, 3),
                                    solution=[1.0, 1.0],
                                    title="Unconstrained optimization path"),
             os.path.join(OUT, "simple_path.png"))
    plt.close("all")

    print("\nwriting animation:")
    viz.animate_trust_region(opt, cf, os.path.join(OUT, "simple_trust_region.gif"),
                              xlim=(-3, 3), ylim=(-3, 3), solution=[1.0, 1.0],
                              title="Unconstrained: trust region evolution")


if __name__ == "__main__":
    main()
