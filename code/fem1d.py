"""1D P1 FEM oracles (Tutorial 6 training + layered independent check).

Project method is FDM. This file is an independent 1D discretisation:
element matrix K^e = (a_e/h) * [[1,-1],[-1,1]] with a_e constant on the element.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from model import A_LEFT, A_RIGHT, X_INTERFACE, u_layered


def mesh(n_elements):
    h = 1.0 / n_elements
    nodes = np.linspace(0.0, 1.0, n_elements + 1)
    elements = np.column_stack(
        (np.arange(n_elements), np.arange(1, n_elements + 1))
    )
    return nodes, elements, h


def element_stiffness(h, a=1.0):
    """int a phi_i' phi_j' dx on one element of length h (a constant)."""
    return (a / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])


def element_load(f, x_left, x_right):
    h = x_right - x_left
    return 0.5 * h * np.array([f(x_left), f(x_right)], dtype=float)


def assemble(nodes, elements, f, a_fun):
    n_nodes = len(nodes)
    rows, cols, data = [], [], []
    F = np.zeros(n_nodes)
    h = nodes[1] - nodes[0]
    for e0, e1 in elements:
        x0, x1 = nodes[e0], nodes[e1]
        xmid = 0.5 * (x0 + x1)
        Ke = element_stiffness(h, float(a_fun(xmid)))
        Fe = element_load(f, x0, x1)
        for a, ia in enumerate((e0, e1)):
            F[ia] += Fe[a]
            for b, ib in enumerate((e0, e1)):
                rows.append(ia)
                cols.append(ib)
                data.append(Ke[a, b])
    K = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tolil()
    u_left = 0.0
    u_right = float(u_layered(1.0, 0.0))
    g = np.zeros(n_nodes)
    g[0] = u_left
    g[-1] = u_right
    # Move non-homogeneous Dirichlet columns into the load, then pin rows.
    F = F - K.tocsr() @ g
    for bnd, val in ((0, u_left), (n_nodes - 1, u_right)):
        K[bnd, :] = 0.0
        K[:, bnd] = 0.0
        K[bnd, bnd] = 1.0
        F[bnd] = val
    return K.tocsr(), F


def solve_poisson_sin(n_elements):
    """Training problem -u''=pi^2 sin(pi x), u(0)=u(1)=0."""
    nodes, elements, h = mesh(n_elements)
    f = lambda x: np.pi**2 * np.sin(np.pi * float(x))
    a_fun = lambda x: 1.0
    # homogeneous Dirichlet
    n_nodes = len(nodes)
    rows, cols, data = [], [], []
    F = np.zeros(n_nodes)
    Ke = element_stiffness(h, 1.0)
    for e0, e1 in elements:
        Fe = element_load(f, nodes[e0], nodes[e1])
        for a, ia in enumerate((e0, e1)):
            F[ia] += Fe[a]
            for b, ib in enumerate((e0, e1)):
                rows.append(ia)
                cols.append(ib)
                data.append(Ke[a, b])
    K = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tolil()
    for bnd in (0, n_nodes - 1):
        K[bnd, :] = 0.0
        K[:, bnd] = 0.0
        K[bnd, bnd] = 1.0
        F[bnd] = 0.0
    uh = spsolve(K.tocsr(), F)
    e = uh - np.sin(np.pi * nodes)
    residual = np.linalg.norm(F - K.tocsr() @ uh, np.inf)
    return h, float(np.sqrt(h) * np.linalg.norm(e)), float(np.linalg.norm(e, np.inf)), residual


def solve_layered_aligned(n_elements=32):
    """Interface-fitted 1D P1: n_elements even so x=1/2 is a node."""
    if n_elements % 2:
        raise ValueError("need even n_elements so x=1/2 is a node")
    nodes, elements, h = mesh(n_elements)

    def a_fun(xmid):
        return A_LEFT if xmid < X_INTERFACE else A_RIGHT

    f = lambda x: 0.0
    K, F = assemble(nodes, elements, f, a_fun)
    uh = spsolve(K, F)
    exact = u_layered(nodes, 0.0)
    residual = float(np.linalg.norm(F - K @ uh, np.inf))
    return nodes, uh, exact, residual, float(np.linalg.norm(uh - exact, np.inf))
