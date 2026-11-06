"""Report figures. Named by role; dpi=300; no rainbow colormaps."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from assemble import grid_from_n, reshape_field

DPI = 300
BLUE = "#1F6F8B"
ORANGE = "#C45C26"
NAVY = "#1B3A5F"


def savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_solution_contour(uh, n, path, title, cbar_label=r"$u_h$"):
    axis, X, Y, h = grid_from_n(n)
    U = reshape_field(uh, n)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cf = ax.contourf(X, Y, U, levels=16, cmap="viridis")
    ax.plot(X.ravel(), Y.ravel(), ".", color="k", markersize=1.2, alpha=0.35)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title)
    fig.colorbar(cf, ax=ax, label=cbar_label)
    savefig(path)


def plot_solution_surface(uh, n, path, title):
    axis, X, Y, h = grid_from_n(n)
    U = reshape_field(uh, n)
    fig = plt.figure(figsize=(5.6, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, U, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$u_h$")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.7, label=r"$u_h$")
    savefig(path)


def plot_spy(A, path, title="Sparsity of $A_h$ after Dirichlet elimination"):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.spy(A, markersize=0.4, color=NAVY)
    ax.set_title(title)
    ax.set_xlabel("column $k$")
    ax.set_ylabel("row $k$")
    savefig(path)


def plot_convergence(
    hs,
    series,
    path,
    title,
    expected=2.0,
    annotation=None,
    ref_index=0,
    fit_indices=None,
):
    """series: list of (errs, label, marker, color)."""
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    hs = np.asarray(hs, dtype=float)
    if fit_indices is None:
        fit_indices = set(range(len(series)))
    for idx, (errs, label, marker, color) in enumerate(series):
        errs = np.asarray(errs, dtype=float)
        ax.loglog(hs, errs, marker + "-", color=color, label=label)
        if idx in fit_indices:
            coef = np.polyfit(np.log(hs), np.log(errs), 1)
            slope = coef[0]
            y_pos = 0.18 if color == BLUE else (0.72 if len(fit_indices) == 1 else 0.08)
            ax.text(
                0.52,
                y_pos,
                f"{label.split()[0]} fit {slope:.2f}",
                transform=ax.transAxes,
                color=color,
                fontsize=8,
            )
    ref_base = np.asarray(series[ref_index][0], dtype=float)[0]
    ref = ref_base * (hs / hs[0]) ** expected
    exp_str = f"{expected:.0f}" if expected != 1.0 else ""
    ref_label = r"reference $O(h)$" if expected == 1.0 else fr"reference $O(h^{{{exp_str}}})$"
    ax.loglog(hs, ref, "k--", label=ref_label)
    ax.set_xlabel(r"mesh size $h$")
    ax.set_ylabel("error")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    if annotation:
        ax.text(0.04, 0.06, annotation, transform=ax.transAxes, fontsize=8)
    savefig(path)


def plot_layered_profile(axis, uh_harm, n, path, exact_fun, uh_arith=None):
    U_h = reshape_field(uh_harm, n)
    mid = n // 2
    x = axis
    xf = np.linspace(0.0, 1.0, 401)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(xf, exact_fun(xf, 0.5), "k-", lw=1.4, label=r"exact $u^*(x)$")
    ax.plot(
        x,
        U_h[:, mid],
        "o",
        color=BLUE,
        markersize=3.4,
        label=r"harmonic $u_h$ ($y\approx 0.5$)",
    )
    if uh_arith is not None:
        U_a = reshape_field(uh_arith, n)
        ax.plot(
            x,
            U_a[:, mid],
            "--",
            color=ORANGE,
            lw=1.6,
            label=r"arithmetic $u_h$ ($y\approx 0.5$)",
        )
    ax.axvline(0.5, color="0.6", ls=":", label=r"interface $x=1/2$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u$")
    ax.set_title("Layered solution along a mid-line")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig(path)


def plot_flux(axis_y, flux_h, flux_a, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(axis_y, flux_h, "-", color=BLUE, lw=2.0, label="harmonic face average", zorder=2)
    ax.plot(axis_y, flux_a, "-", color=ORANGE, lw=1.8, label="arithmetic face average", zorder=2)
    # Draw the exact flux last and leave room below 20/11 so the dashed
    # line is not hidden under the harmonic curve or the bottom spine.
    ax.axhline(20.0 / 11.0, color="k", ls="--", lw=1.4, label=r"exact flux $20/11$", zorder=3)
    ax.set_xlabel(r"$y$ at the interface")
    ax.set_ylabel(r"numerical flux $a_E(U_R-U_L)/h$")
    ax.set_title("Interface flux: harmonic vs arithmetic")
    ax.set_ylim(1.70, 2.52)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    ax.grid(True, alpha=0.3)
    savefig(path)


def plot_residual_history(hist, tol, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.semilogy(np.arange(len(hist)), hist, color=BLUE, label="CG relative residual")
    ax.axhline(tol, color=ORANGE, ls="--", label=fr"stop at relative residual ${tol:.0e}$")
    ax.set_xlabel("CG iteration")
    ax.set_ylabel(r"$\|r_k\|_\infty/\|r_0\|_\infty$")
    ax.set_title("CG residual history (stopping rule is the residual)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    savefig(path)


def plot_cost_accuracy(points, path, target=None):
    """points: list of (cost, error, label, marker, color)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for cost, err, label, marker, color in points:
        ax.loglog(cost, err, marker + "-", color=color, label=label)
    if target is not None:
        ax.axhline(target, color="0.5", ls=":", label=fr"matched target ${target:.1e}$")
    ax.set_xlabel(r"cost (CG iterations, or $1$ for one spsolve)")
    ax.set_ylabel(r"discrete $L^2$ error")
    ax.set_title("Cost vs accuracy at matched residual")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    savefig(path)


def plot_iterations_contrast(contrasts, iters, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.semilogx(contrasts, iters, "o-", color=BLUE)
    ax.set_xlabel(r"contrast $a_R/a_L$")
    ax.set_ylabel("CG iterations to relative residual $10^{-10}$")
    ax.set_title("CG cost grows with coefficient contrast")
    ax.grid(True, which="both", alpha=0.3)
    savefig(path)
