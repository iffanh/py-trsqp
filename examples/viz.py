"""Plotting helpers for the DFTR-Filter example notebooks.

Everything here reads only the public result surface of ``DFTRFilter`` --
``opt.records`` (a list of ``IterationRecord``) and ``opt.best`` -- so the
notebooks never touch algorithm internals. Matplotlib is the only dependency.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm


def iterate_path(opt) -> np.ndarray:
    """(n_points, n) array of the incumbent x at each recorded iteration that
    produced a genuine evaluation (skips pure criticality/geometry steps)."""
    xs = [r.x for r in opt.records if r.f is not None]
    return np.array(xs) if xs else np.empty((0, opt.problem.n))


def save(fig, path: str, dpi: int = 110):
    """Save a figure and report where it went."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  saved {path}")


def plot_convergence(opt, title: str = "Convergence"):
    """2x2 panel: objective, constraint violation, trust-region radius, and
    the model criticality measure psi~ = chi~ + vartheta~, all vs iteration."""
    recs = opt.records
    ks = [r.k for r in recs]

    def series(attr):
        xs, ys = [], []
        for r in recs:
            v = getattr(r, attr)
            if v is not None:
                xs.append(r.k)
                ys.append(v)
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(title, fontsize=14)

    kf, f = series("f")
    ax = axes[0, 0]
    # Shift so a semilog view still works when the optimum is ~0.
    f = np.array(f)
    f_shift = f - f.min() + 1e-12 if len(f) else f
    ax.semilogy(kf, f_shift, "o-", ms=3, color="#1f77b4")
    ax.set_title(r"objective  $f(x_k) - f^\ast$")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

    kv, v = series("vartheta")
    ax = axes[0, 1]
    v = np.array(v)
    ax.semilogy(kv, np.maximum(v, 1e-16), "o-", ms=3, color="#d62728")
    ax.set_title(r"constraint violation  $\vartheta(x_k)$")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(ks, [r.radius for r in recs], "o-", ms=3, color="#2ca02c")
    ax.set_title(r"trust-region radius  $\Delta_k$")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

    kp, psi = series("psi_tilde")
    ax = axes[1, 1]
    ax.semilogy(kp, np.maximum(np.array(psi), 1e-16), "o-", ms=3, color="#9467bd")
    ax.set_title(r"criticality  $\tilde\psi_k = \tilde\chi_k + \tilde\vartheta_k$")
    ax.set_xlabel("iteration")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _feasible_mask(XX, YY, ineqs, bounds):
    mask = np.ones_like(XX, dtype=bool)
    for c in ineqs:
        mask &= c([XX, YY]) >= 0
    if bounds is not None:
        (lb0, lb1), (ub0, ub1) = bounds
        mask &= (XX >= lb0) & (XX <= ub0) & (YY >= lb1) & (YY <= ub1)
    return mask


def plot_trajectory_2d(opt, cf: Callable, xlim=(-3, 3), ylim=(-3, 3),
                       ineqs: Optional[Sequence[Callable]] = None,
                       bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                       solution: Optional[Sequence[float]] = None,
                       levels: Optional[Sequence[float]] = None,
                       title: str = "Optimization path"):
    """Objective contours + the incumbent trajectory (2-variable problems only).

    Optionally shades the feasible region defined by ``ineqs`` (each c(x) >= 0)
    and ``bounds`` = ((lb0, lb1), (ub0, ub1)), and marks a known ``solution``.
    """
    assert opt.problem.n == 2, "trajectory plot is for 2-variable problems"

    XX, YY = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    Z = cf([XX, YY])

    fig, ax = plt.subplots(figsize=(7, 6))
    if levels is None:
        zmin = np.nanmin(Z)
        base = zmin if zmin > 0 else 1e-2
        levels = [base * 10 ** i for i in np.arange(0, 5, 0.5)]
    try:
        CS = ax.contour(XX, YY, Z - np.nanmin(Z) + 1e-6, levels=levels,
                        norm=LogNorm(), cmap=cm.viridis, alpha=0.7)
        ax.clabel(CS, fontsize=7, inline=True, fmt="%.1g")
    except Exception:
        ax.contour(XX, YY, Z, 30, cmap=cm.viridis, alpha=0.7)

    if ineqs or bounds:
        mask = _feasible_mask(XX, YY, ineqs or [], bounds)
        ax.contourf(XX, YY, mask.astype(float), levels=[0.5, 1.5],
                    colors=["#1f77b4"], alpha=0.12)

    path = iterate_path(opt)
    if len(path):
        ax.plot(path[:, 0], path[:, 1], "-", color="0.3", lw=1, alpha=0.7, zorder=2)
        sc = ax.scatter(path[:, 0], path[:, 1], c=np.arange(len(path)),
                        cmap="autumn", s=18, zorder=3, edgecolor="k", linewidth=0.3)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("iteration")
        ax.scatter(*path[0], color="black", s=70, marker="s", zorder=4, label="start")

    if opt.best is not None:
        ax.scatter(opt.best.x[0], opt.best.x[1], color="red", s=90, marker="*",
                   zorder=5, label="best found")
    if solution is not None:
        ax.scatter(solution[0], solution[1], color="lime", s=90, marker="X",
                   edgecolor="k", zorder=5, label="known solution")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def summary(opt) -> str:
    b = opt.best
    lines = [
        f"termination : {opt.termination_status}",
        f"iterations  : {len(opt.records)}",
        f"evaluations : {opt.problem.n_cf_evals}",
    ]
    if b is not None:
        lines += [
            f"best x      : {np.array2string(b.x, precision=6)}",
            f"best f      : {b.f:.6g}",
            f"best viol.  : {b.vartheta:.3g}",
        ]
    return "\n".join(lines)
