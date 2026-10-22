"""Continuous Direction-3 model: -div(a grad u) = f on (0,1)^2.

Physics (coefficient a, manufactured u*, forcing f, boundary g) lives here.
The grid and matrix live in assemble.py.
"""
import numpy as np

from test_problem_bank import P1, P2, P4, P5

X_INTERFACE = 0.5
A_LEFT = 1.0
A_RIGHT = 10.0
FLUX_EXACT = 20.0 / 11.0  # a u_x = 20/11 on the layered benchmark


def a_smooth(x, y):
    """Smooth conductivity a = 1+x > 0 (bank P4)."""
    return 1.0 + x


def a_layered(x, y, a_left=A_LEFT, a_right=A_RIGHT, x_star=X_INTERFACE):
    """Piecewise-constant conductivity with a jump at x = 1/2.

    The value exactly on x_star is never used by the face-average stencil:
    faces at the interface take a from the two neighbouring nodes.
    """
    return np.where(np.asarray(x) < x_star, a_left, a_right)


def u_layered(x, y):
    """Exact layered solution (independent of y).

    u = (20/11) x                 on [0, 1/2],
    u = (2/11) x + 9/11           on [1/2, 1].
    Continuous, u(0)=0, u(1)=1, flux a u_x = 20/11.
    """
    x = np.asarray(x, dtype=float)
    left = (20.0 / 11.0) * x
    right = (2.0 / 11.0) * x + 9.0 / 11.0
    return np.where(x <= X_INTERFACE, left, right)


def f_layered(x, y):
    return 0.0 * np.asarray(x, dtype=float)


class LayeredProblem:
    name = "layered a=1|10 (Dirichlet)"
    exact = staticmethod(u_layered)
    forcing = staticmethod(f_layered)
    coefficient = staticmethod(a_layered)
    boundary = "g = u*(x) on the whole boundary (independent of y)"
    expected_rate = "nodal exact with harmonic face average; O(1) flux error if arithmetic"
    note = "Problem Pack Direction 3 layered benchmark."


LAYERED = LayeredProblem()
BANK_SMOOTH = (P1, P2, P4, P5)
