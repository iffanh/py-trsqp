"""DFTR-Filter: unconstrained Rosenbrock example (runnable script).

Minimises the classic Rosenbrock ("banana") function
    f(x) = 100 (x2 - x1^2)^2 + (1 - x1)^2,   -2 <= xi <= 2,
from the standard starting point x0 = (-1.2, 1.0). The global minimiser is
x* = (1, 1), f* = 0. Rosenbrock's curved, narrow valley is a classic hard
case for quadratic-model trust-region methods: with the default constants,
DFTR-Filter gets trapped following the valley in small steps and only
reaches f ~ 0.3 within a few hundred iterations.

Four constants tuned together get within f ~ 2e-7 of the true minimum in
600 iterations (~20s):
  - max_points=6: uses the full quadratic basis size for n=2 (q = (n+1)(n+2)/2)
    instead of the library default of min(2n+1, q) = 5. The extra
    interpolation point is by far the single biggest lever here -- a more
    complete quadratic model tracks the valley's curvature much better.
  - init_radius=1.5, max_radius=2.0: a larger initial step size covers more
    of the valley per iteration than the default init_radius=1.0.
  - gamma1=4.0: grows the trust-region radius more aggressively after an
    accepted step (default 2.0), compounding the larger initial radius.
  - lambda_th=20.0: a *looser* poisedness threshold than the default 10.0.
    Counterintuitively, tightening this (tried down to 1.5-2.0) makes things
    far worse, not better: the rejected-step path in the main loop only
    shrinks the radius when the current sample set's poisedness is already
    within lambda_th, and otherwise repoises instead -- an overly strict
    threshold means poisedness is judged "good enough" too rarely, so the
    radius never shrinks and the optimizer stalls repoising the same
    sample set instead of making progress.

Prints a summary and writes two figures (convergence curves and the
optimization path) into examples/results/.

Run from the repo root or from examples/:
    python examples/rosenbrock_example.py
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
    """Rosenbrock's function. Works on scalars/arrays and on a mesh grid (for plotting)."""
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def main():
    opt = DFTRFilter(
        x0=[-1.2, 1.0],   # the standard Rosenbrock starting point
        cf=cf,
        lb=-2.0,          # scalar bounds broadcast to every coordinate
        ub=2.0,
        # See the module docstring for why each of these matters here.
        constants={"init_radius": 1.5, "max_radius": 2.0, "gamma1": 4.0,
                   "max_points": 6, "lambda_th": 20.0},
    )
    opt.optimize(max_iter=600)

    print("\n=== unconstrained Rosenbrock result ===")
    print(viz.summary(opt))
    print("known minimiser: [1. 1.], f* = 0")

    print("\nwriting figures:")
    viz.save(viz.plot_convergence(opt, "Rosenbrock: unconstrained convergence"),
             os.path.join(OUT, "rosenbrock_convergence.png"))
    viz.save(viz.plot_trajectory_2d(opt, cf, xlim=(-2, 2), ylim=(-2, 2),
                                    solution=[1.0, 1.0],
                                    title="Rosenbrock: unconstrained optimization path"),
             os.path.join(OUT, "rosenbrock_path.png"))
    plt.close("all")


if __name__ == "__main__":
    main()
