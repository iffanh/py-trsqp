"""Plotting helpers for the DFTR-Filter example notebooks.

Most of this reads only the public result surface of ``DFTRFilter`` --
``opt.records`` (a list of ``IterationRecord``) and ``opt.best``.
``animate_trust_region`` additionally reads ``opt.history`` (populated only
when the optimizer was constructed with ``record_history=True``) and
``opt.problem``'s normalization fields, since animating the trust region,
sample set, and fitted model needs the per-iteration state the summary
plots don't. Matplotlib is the only dependency (plus Pillow, to write GIFs).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle


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


def _select_frame_indices(n_total: int, full_prefix: int = 100, budget: int = 200) -> List[int]:
    """Every index for the first `full_prefix` iterations, then a coarser,
    evenly-spaced sample of the rest so the total frame count stays near
    `budget` regardless of how many iterations the run took. Always
    includes the final iteration."""
    if n_total <= full_prefix:
        return list(range(n_total))
    tail_start = full_prefix
    remaining = n_total - tail_start
    tail_budget = max(budget - full_prefix, 10)
    step = max(1, remaining // tail_budget)
    tail = list(range(tail_start, n_total, step))
    if tail[-1] != n_total - 1:
        tail.append(n_total - 1)
    return list(range(full_prefix)) + tail


def _normalize_grid(problem, X: np.ndarray) -> np.ndarray:
    """Vectorized form of Problem.normalize for a (n, N) grid of points,
    rather than the single-point form the algorithm itself uses."""
    center = problem.center[:, None]
    half_hi = problem.half_hi[:, None]
    half_lo = problem.half_lo[:, None]
    above = X > center
    return np.where(above, (X - center) / half_hi, (X - center) / half_lo)


def _model_values(model, Dn: np.ndarray) -> np.ndarray:
    """QuadraticModel.value(), vectorized over Dn = (already-centered,
    normalized) points of shape (n, N)."""
    HD = model.H @ Dn
    quad = np.sum(Dn * HD, axis=0)
    return model.c + model.g @ Dn + 0.5 * quad


def animate_trust_region(opt, cf: Callable, path: str, xlim=(-3, 3), ylim=(-3, 3),
                          ineqs: Optional[Sequence[Callable]] = None,
                          bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                          solution: Optional[Sequence[float]] = None,
                          title: str = "Trust region evolution",
                          full_prefix: int = 100, frame_budget: int = 200,
                          fps: int = 8, dpi: int = 100) -> None:
    """Animate how the trust region, sample set, and fitted model evolve.

    Requires ``opt`` to come from ``DFTRFilter(..., record_history=True)``.
    Each frame is a hybrid contour of the objective: grid points inside the
    current trust-region disk (center x_k, radius Delta_k) are colored by the
    fitted quadratic model's value there, and points outside are colored by
    the true objective ``cf`` -- so the model and the ground truth it's
    approximating are visible in the same picture, spliced at the trust
    region boundary (drawn as a dashed circle). Also plots the interpolation
    sample points, the current center, and the best point found so far at
    each iteration (2-variable problems only). Writes an animated GIF to
    ``path``.

    Assumes an isotropic box (equal half-width on both axes, true for every
    ``lb``/``ub`` pair passed as scalars) so the trust region is a circle in
    the original units; Problem.normalize's per-axis scaling would otherwise
    make it an ellipse.
    """
    assert opt.problem.n == 2, "trust-region animation is for 2-variable problems"
    if not opt.history:
        raise ValueError(
            "opt.history is empty -- construct DFTRFilter(..., record_history=True) "
            "before calling optimize() to record the per-iteration state this needs."
        )

    problem = opt.problem
    frame_indices = _select_frame_indices(len(opt.history), full_prefix, frame_budget)

    XX, YY = np.meshgrid(np.linspace(*xlim, 250), np.linspace(*ylim, 250))
    grid_real = np.stack([XX.ravel(), YY.ravel()])
    Z_true = np.asarray(cf([XX, YY]), dtype=float)

    zmin = np.nanmin(Z_true)
    shift = zmin  # fixed baseline for every frame -- see the render() comment below
    base = zmin if zmin > 0 else 1e-2
    levels = [base * 10 ** i for i in np.arange(0, 5, 0.5)]

    feasible_mask = None
    if ineqs or bounds:
        feasible_mask = _feasible_mask(XX, YY, ineqs or [], bounds)

    # Isotropic-box assumption (see docstring): a single scale factor
    # converts a normalized radius to a real-units radius.
    scale = float(problem.half_hi[0])

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.tight_layout()  # fixed layout computed once; per-frame calls are expensive and unnecessary here

    def render(frame_idx):
        ax.clear()
        snap = opt.history[frame_idx]
        center_real = problem.denormalize(snap.center)
        radius_real = snap.radius * scale

        Dn = _normalize_grid(problem, grid_real) - snap.center[:, None]
        Z_model = _model_values(snap.model_cf, Dn).reshape(XX.shape)
        dist2 = (XX - center_real[0]) ** 2 + (YY - center_real[1]) ** 2
        inside = dist2 <= radius_real ** 2
        Z_hybrid = np.where(inside, Z_model, Z_true)

        # No clabel() here (unlike the static plots): its per-contour label
        # placement is one of the more expensive parts of a frame, and with
        # up to ~200 frames per animation it isn't worth the cost for a
        # preview GIF where the color scale already conveys magnitude.
        #
        # The shift below is fixed (from Z_true's minimum, computed once
        # outside this function) rather than adapted to each frame's own
        # Z_hybrid range -- otherwise the same nominal contour level would
        # correspond to a different absolute objective value in different
        # frames (worse the wilder the model gets inside the trust region),
        # making levels look like they "move" across the animation instead
        # of representing constant objective values throughout. A model
        # dipping below the fixed floor just isn't contoured there (LogNorm
        # treats non-positive shifted values as invalid) rather than forcing
        # a still-consistent-looking picture at the cost of correctness.
        try:
            ax.contour(XX, YY, Z_hybrid - shift,
                       levels=levels, norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
                       cmap=cm.viridis, alpha=0.7)
        except Exception:
            ax.contour(XX, YY, Z_hybrid, 30, cmap=cm.viridis, alpha=0.7)

        if feasible_mask is not None:
            ax.contourf(XX, YY, feasible_mask.astype(float), levels=[0.5, 1.5],
                        colors=["#1f77b4"], alpha=0.12)

        ax.add_patch(Circle(center_real, radius_real, fill=False, linestyle="--",
                             edgecolor="0.2", linewidth=1.2, zorder=3))

        Y_real = np.array([problem.denormalize(snap.Y[:, j]) for j in range(snap.Y.shape[1])])
        ax.scatter(Y_real[:, 0], Y_real[:, 1], color="0.15", s=22, zorder=4,
                   label="sample points")
        ax.scatter(*center_real, color="black", s=60, marker="s", zorder=5, label="center $x_k$")

        if snap.best_x is not None:
            ax.scatter(snap.best_x[0], snap.best_x[1], color="red", s=90, marker="*",
                       zorder=6, label="best so far")
        if solution is not None:
            ax.scatter(solution[0], solution[1], color="lime", s=90, marker="X",
                       edgecolor="k", zorder=6, label="known solution")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        best_f_str = f"{snap.best_f:.4g}" if snap.best_f is not None else "n/a"
        ax.set_title(f"{title}\niteration {snap.k}   radius={radius_real:.3g}   best f={best_f_str}")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    anim = FuncAnimation(fig, render, frames=frame_indices)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"  saved {path} ({len(frame_indices)} frames from {len(opt.history)} iterations)")
