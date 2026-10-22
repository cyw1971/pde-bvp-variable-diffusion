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
        return float(np.asarray(a_fun(x + 0.5 * di * h, y + 0.5 * dj * h)))
    a_here = float(np.asarray(a_fun(x, y)))
    a_there = float(np.asarray(a_fun(x_neigh, y_neigh)))
    if mode == "arithmetic":
        return 0.5 * (a_here + a_there)
    if mode == "harmonic":
        return 2.0 * a_here * a_there / (a_here + a_there)
    raise ValueError(f"unknown face mode {mode!r}")


def assemble(n, problem=P4, face_mode="halfpoint"):
    """Return (A, b, h, axis) for n interior nodes per direction.

    Stencil:
        (A U)_ij = (1/h^2) [ a_E(U-U_E)+a_W(U-U_W)+a_N(U-U_N)+a_S(U-U_S) ]
    Dirichlet neighbour at (x_e,y_e) with value g:
        b[k] += a_face * g / h^2
        (the neighbour is not a matrix column).
    """
    axis, X, Y, h = grid_from_n(n)
    inv_h2 = 1.0 / h**2
    b = np.asarray(problem.forcing(X, Y), dtype=float).ravel()
    rows, cols, data = [], [], []
    a_fun = problem.coefficient

    for i in range(n):
        for j in range(n):
            kk = k_index(i, j, n)
            x_ij, y_ij = axis[i], axis[j]
            diag = 0.0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x_e = x_ij + di * h
                y_e = y_ij + dj * h
                a_face = face_coefficient(
                    a_fun, x_ij, y_ij, di, dj, h, face_mode, x_e, y_e
                )
                coef = a_face * inv_h2
                diag += coef
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    rows.append(kk)
                    cols.append(k_index(ii, jj, n))
                    data.append(-coef)
                else:
                    g = float(np.asarray(problem.exact(x_e, y_e)))
                    b[kk] += coef * g
            rows.append(kk)
            cols.append(kk)
            data.append(diag)

    A = coo_matrix((data, (rows, cols)), shape=(n * n, n * n)).tocsr()
    return A, b, h, axis


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
