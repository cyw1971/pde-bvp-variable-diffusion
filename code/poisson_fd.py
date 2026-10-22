"""Week-5/6 entry point. The project solver is assemble.py; this wrapper keeps
the Tutorial 5 P5 challenge command: python poisson_fd.py
"""
from assemble import assemble
from model import P5
from solvers import solve_direct
from verify import errors


def centre_value(n=15):
    A, b, h, axis = assemble(n, P5, "halfpoint")
    uh, residual, _ = solve_direct(A, b)
    mid = n // 2
    return float(uh[mid * n + mid]), h, residual, uh


if __name__ == "__main__":
    n = 15
    centre, h, residual, uh = centre_value(n)
    A, b, h, axis = assemble(n, P5, "halfpoint")
    e2, einf, _ = errors(uh, n, h, P5)
    print(f"n={n} h={h:.6g} uh(0.5,0.5)={centre:.6f} residual={residual:.3e}")
    print(f"L2={e2:.3e} Linf={einf:.3e}")
    assert abs(centre - 1.65) < 5e-3
    print("P5 centre OK")
