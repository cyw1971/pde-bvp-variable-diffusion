"""Sparse solves. Stopping rule is the residual, never the iterate difference."""
import inspect
import time

import numpy as np
from scipy.sparse.linalg import cg, eigsh, spsolve


def solve_direct(A, b):
    """Sparse direct solve. Return (uh, residual_inf, wall_time)."""
    t0 = time.perf_counter()
    uh = spsolve(A, b)
    wall = time.perf_counter() - t0
    residual = float(np.linalg.norm(b - A @ uh, np.inf))
    return uh, residual, wall


def _cg_kwargs(tol, maxiter):
    params = inspect.signature(cg).parameters
    if "rtol" in params:
        return {"rtol": tol, "maxiter": maxiter}
    return {"tol": tol, "maxiter": maxiter}


def solve_cg(A, b, tol=1e-10, maxiter=20000, x0=None):
    """Unpreconditioned CG. Return (uh, residual_inf, iters, converged, wall)."""
    count = {"k": 0}

    def tick(_xk):
        count["k"] += 1

    x0 = np.zeros_like(b) if x0 is None else x0
    t0 = time.perf_counter()
    uh, info = cg(A, b, x0=x0, callback=tick, **_cg_kwargs(tol, maxiter))
    wall = time.perf_counter() - t0
    residual = float(np.linalg.norm(b - A @ uh, np.inf))
    return uh, residual, int(count["k"]), bool(info == 0), wall


def residual_history_cg(A, b, tol=1e-10, maxiter=20000):
    """Relative residual ||r_k||_inf / ||r_0||_inf after each CG iteration."""
    x = np.zeros_like(b)
    r0 = float(np.linalg.norm(b - A @ x, np.inf))
    hist = [1.0]
    if r0 == 0.0:
        return np.array([0.0])

    def tick(xk):
        hist.append(float(np.linalg.norm(b - A @ xk, np.inf) / r0))

    cg(A, b, x0=x, callback=tick, **_cg_kwargs(tol, maxiter))
    return np.asarray(hist)


def symmetry_relative(A):
    d = (A - A.T).tocsr()
    num = float(np.abs(d.data).max()) if d.nnz else 0.0
    den = float(np.abs(A.data).max())
    return num / den if den else 0.0


def smallest_eigenvalue(A, k=1):
    """Smallest algebraic eigenvalue (SPD check)."""
    vals = eigsh(A, k=k, which="SA", return_eigenvectors=False)
    return float(np.min(vals))


def condition_estimate(A):
    """kappa_2 estimate from extreme eigenvalues of the SPD interior matrix."""
    lam_max = float(eigsh(A, k=1, which="LM", return_eigenvectors=False)[0])
    lam_min = float(eigsh(A, k=1, which="SA", return_eigenvectors=False)[0])
    return lam_max / lam_min, lam_min, lam_max
