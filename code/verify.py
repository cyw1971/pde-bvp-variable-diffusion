"""Named-norm errors, refinement tables, flux jump, maximum principle."""
import numpy as np

from assemble import assemble, grid_from_n, interface_column_pair, numerical_interface_flux
from model import FLUX_EXACT, LAYERED, P4
from solvers import solve_direct, symmetry_relative


def errors(uh, n, h, problem):
    axis, X, Y, _ = grid_from_n(n)
    exact = np.asarray(problem.exact(X, Y), dtype=float).ravel()
    diff = uh - exact
    error_l2 = float(h * np.linalg.norm(diff))
    error_linf = float(np.linalg.norm(diff, np.inf))
    return error_l2, error_linf, exact


def observed_order(h1, e1, h2, e2):
    return float(np.log(e1 / e2) / np.log(h1 / h2))


def fit_slope(hs, errs):
    hs = np.asarray(hs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    if len(hs) >= 3:
        coef, cov = np.polyfit(np.log(hs), np.log(errs), 1, cov=True)
        slope_se = float(np.sqrt(cov[0, 0]))
    else:
        coef = np.polyfit(np.log(hs), np.log(errs), 1)
        slope_se = float("nan")
    slope = float(coef[0])
    max_dev = float(np.max(np.abs(np.log(errs) - np.polyval(coef, np.log(hs)))))
    return slope, slope_se, max_dev


def refinement_table(ns, problem, face_mode="halfpoint"):
    rows = []
    prev = None
    for n in ns:
        A, b, h, axis = assemble(n, problem, face_mode)
        uh, residual, wall = solve_direct(A, b)
        e2, einf, _ = errors(uh, n, h, problem)
        row = {
            "n": n,
            "N": n + 1,
            "h": h,
            "L2": e2,
            "Linf": einf,
            "residual": residual,
            "nnz": int(A.nnz),
            "wall": wall,
            "symmetry": symmetry_relative(A),
            "order_L2": None,
            "order_Linf": None,
            "uh": uh,
            "A": A,
            "b": b,
        }
        if prev is not None:
            row["order_L2"] = observed_order(prev["h"], prev["L2"], h, e2)
            row["order_Linf"] = observed_order(prev["h"], prev["Linf"], h, einf)
        rows.append(row)
        prev = row
    hs = [r["h"] for r in rows]
    slope_l2, se_l2, dev_l2 = fit_slope(hs, [r["L2"] for r in rows])
    slope_inf, se_inf, dev_inf = fit_slope(hs, [r["Linf"] for r in rows])
    return rows, {
        "L2": (slope_l2, se_l2, dev_l2),
        "Linf": (slope_inf, se_inf, dev_inf),
    }


def layered_diagnostics(n, face_mode):
    A, b, h, axis = assemble(n, LAYERED, face_mode)
    uh, residual, wall = solve_direct(A, b)
    e2, einf, exact = errors(uh, n, h, LAYERED)
    i_left, i_right = interface_column_pair(n, h)
    # Face coefficient used on the interface (same for every y).
    x_l, x_r = axis[i_left], axis[i_right]
    a_face = None
    from assemble import face_coefficient

    a_face = face_coefficient(
        LAYERED.coefficient, x_l, 0.5, 1, 0, h, face_mode, x_r, 0.5
    )
    flux = numerical_interface_flux(uh, n, h, a_face, i_left)
    flux_jump = float(np.max(np.abs(flux - FLUX_EXACT)))
    flux_spread = float(np.max(flux) - np.min(flux))
    gmin = float(np.min(exact))
    gmax = float(np.max(LAYERED.exact(np.array([0.0, 1.0]), np.array([0.0, 0.0]))))
    # Boundary extrema of u* are 0 and 1.
    umin, umax = float(np.min(uh)), float(np.max(uh))
    return {
        "n": n,
        "h": h,
        "mode": face_mode,
        "L2": e2,
        "Linf": einf,
        "residual": residual,
        "a_face": a_face,
        "flux": flux,
        "flux_jump": flux_jump,
        "flux_spread": flux_spread,
        "umin": umin,
        "umax": umax,
        "inside_extrema": (umin >= -1e-12) and (umax <= 1.0 + 1e-12),
        "symmetry": symmetry_relative(A),
        "uh": uh,
        "axis": axis,
        "i_left": i_left,
        "A": A,
        "b": b,
        "wall": wall,
        "nnz": int(A.nnz),
    }


def truncation_discrete_l2(n, problem=P4, face_mode="halfpoint"):
    """h ||b - A u*||_2 : local truncation in the discrete L2 norm."""
    A, b, h, axis = assemble(n, problem, face_mode)
    _, X, Y, _ = grid_from_n(n)
    u_star = np.asarray(problem.exact(X, Y), dtype=float).ravel()
    tau = b - A @ u_star
    return h, float(h * np.linalg.norm(tau)), float(np.linalg.norm(tau, np.inf))
