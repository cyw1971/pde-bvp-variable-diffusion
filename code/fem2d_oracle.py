"""Optional 2D P1 FEM oracle for the smooth P4 problem (independent of FDM)."""
import numpy as np


def solve_p4(refinements=4):
    try:
        from skfem import (Basis, BilinearForm, ElementTriP1, Functional,
                           LinearForm, MeshTri, asm, condense, solve)
        from skfem.helpers import dot, grad
    except ImportError:
        return None

    def exact(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    @BilinearForm
    def stiffness(u, v, w):
        a = 1.0 + w.x[0]
        return a * dot(grad(u), grad(v))

    @LinearForm
    def load(v, w):
        x, y = w.x[0], w.x[1]
        u = np.sin(np.pi * x) * np.sin(np.pi * y)
        f = 2.0 * np.pi**2 * (1.0 + x) * u - np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        return f * v

    @Functional
    def squared_l2(w):
        return (w.uh - exact(w.x)) ** 2

    mesh = MeshTri().refined(refinements)
    basis = Basis(mesh, ElementTriP1())
    A = asm(stiffness, basis)
    b = asm(load, basis)
    D = basis.get_dofs().all()
    uh = solve(*condense(A, b, D=D))
    l2 = float(np.sqrt(asm(squared_l2, basis, uh=uh)))
    free = np.setdiff1d(np.arange(basis.N), D)
    residual = float(np.linalg.norm((A @ uh - b)[free], np.inf))
    return {
        "h_max": float(mesh.param()),
        "L2": l2,
        "residual": residual,
        "ndof": int(basis.N),
    }
