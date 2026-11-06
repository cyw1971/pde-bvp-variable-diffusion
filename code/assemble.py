"""Conservative face-averaged FDM for -div(a grad u) = f.

Indexing (write once): n interior nodes per direction, h = 1/(n+1),
x_i = (i+1)*h, y_j = (j+1)*h, k = i*n + j (meshgrid indexing='ij', axis 0 = x).
Boundary nodes are not unknowns. Dirichlet data enter the RHS.
"""
import numpy as np
from scipy.sparse import coo_matrix

from model import P4, LAYERED, X_INTERFACE


def grid_from_n(n):
    if n < 2:
        raise ValueError("n must be at least 2")
    h = 1.0 / (n + 1)
    axis = np.arange(1, n + 1) * h
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    return axis, X, Y, h


def k_index(i, j, n):
    return i * n + j


def face_coefficient(a_fun, x, y, di, dj, h, mode, x_neigh, y_neigh):
    """Positive shared coefficient on the face between (x,y) and its neighbour."""
    if mode == "halfpoint":
        val = np.asarray(a_fun(x + 0.5 * di * h, y + 0.5 * dj * h), dtype=float)
    else:
        a_here = np.asarray(a_fun(x, y), dtype=float)
        a_there = np.asarray(a_fun(x_neigh, y_neigh), dtype=float)
        if mode == "arithmetic":
            val = 0.5 * (a_here + a_there)
        elif mode == "harmonic":
            val = 2.0 * a_here * a_there / (a_here + a_there)
        else:
            raise ValueError(f"unknown face mode {mode!r}")
    out = np.broadcast_to(val, np.broadcast(x, y).shape).astype(float)
    return float(out) if out.ndim == 0 else out


def assemble(n, problem=P4, face_mode="halfpoint"):
    """Return (A, b, h, axis) for n interior nodes per direction.

    Vectorised COO assembly (no Python loops over grid nodes).
    Stencil:
        (A U)_ij = (1/h^2) [ a_E(U-U_E)+a_W(U-U_W)+a_N(U-U_N)+a_S(U-U_S) ]
    Dirichlet neighbour at (x_e,y_e) with value g:
        b[k] += a_face * g / h^2
        (the neighbour is not a matrix column).
    """
    axis, X, Y, h = grid_from_n(n)
    inv_h2 = 1.0 / h**2
    b_grid = np.broadcast_to(
        np.asarray(problem.forcing(X, Y), dtype=float), (n, n)
    ).copy()
    diag = np.zeros((n, n), dtype=float)
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    K = k_index(I, J, n)
    rows, cols, data = [], [], []
    a_fun = problem.coefficient

    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        X_e = X + di * h
        Y_e = Y + dj * h
        coef = (
            face_coefficient(a_fun, X, Y, di, dj, h, face_mode, X_e, Y_e) * inv_h2
        )
        diag += coef
        II, JJ = I + di, J + dj
        interior = (II >= 0) & (II < n) & (JJ >= 0) & (JJ < n)
        rows.append(K[interior].ravel())
        cols.append(k_index(II[interior], JJ[interior], n).ravel())
        data.append((-coef[interior]).ravel())
        bdry = ~interior
        if np.any(bdry):
            G = np.broadcast_to(
                np.asarray(problem.exact(X_e, Y_e), dtype=float), (n, n)
            )
            b_grid[bdry] += coef[bdry] * G[bdry]

    rows.append(K.ravel())
    cols.append(K.ravel())
    data.append(diag.ravel())

    A = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n * n, n * n),
    ).tocsr()
    return A, b_grid.ravel(), h, axis


def reshape_field(uh, n):
    return np.asarray(uh).reshape((n, n))


def interface_column_pair(n, h, x_star=X_INTERFACE):
    """Interior indices (i_left, i_right) whose shared east face is at x_star.

    For odd N = n+1 the interface bisects a grid edge, as the pack requires.
    """
    axis = np.arange(1, n + 1) * h
    faces = axis + 0.5 * h  # east face of each interior column
    i_left = int(np.argmin(np.abs(faces - x_star)))
    if i_left >= n - 1:
        i_left = n - 2
    return i_left, i_left + 1


def numerical_interface_flux(uh, n, h, a_face, i_left):
    """a_E (U_right - U_left)/h at the interface, one value per y-row."""
    U = reshape_field(uh, n)
    return a_face * (U[i_left + 1, :] - U[i_left, :]) / h
