"""Sanity tests matching the report's claims. Run: python validation_tests.py"""
import numpy as np

from assemble import assemble
from fem1d import element_stiffness, solve_layered_aligned
from model import LAYERED, P1, P2, P4, P5
from solvers import solve_direct, symmetry_relative, smallest_eigenvalue
from test_problem_bank import verify_bank
from verify import errors, layered_diagnostics, observed_order, refinement_table


def test_bank():
    assert verify_bank() < 2e-3


def test_p2_exact():
    A, b, h, axis = assemble(15, P2, "halfpoint")
    uh, res, _ = solve_direct(A, b)
    e2, einf, _ = errors(uh, 15, h, P2)
    assert einf < 1e-12, einf
    assert res < 1e-10


def test_p5_centre():
    A, b, h, axis = assemble(15, P5, "halfpoint")
    uh, res, _ = solve_direct(A, b)
    centre = float(uh[(15 // 2) * 15 + (15 // 2)])
    assert abs(centre - 1.65) < 5e-3, centre


def test_p1_two_meshes():
    rows, slopes = refinement_table((7, 15), P1, "halfpoint")
    assert rows[1]["order_L2"] > 1.9


def test_p4_order():
    rows, slopes = refinement_table((15, 31, 63), P4, "halfpoint")
    assert slopes["L2"][0] > 1.95
    assert rows[0]["symmetry"] == 0.0
    assert smallest_eigenvalue(rows[0]["A"]) > 0.0


def test_layered_harmonic_nodal():
    d = layered_diagnostics(30, "harmonic")
    assert d["Linf"] < 1e-10, d["Linf"]
    assert d["flux_jump"] < 1e-10, d["flux_jump"]
    assert d["inside_extrema"]


def test_layered_arithmetic_is_wrong():
    d = layered_diagnostics(30, "arithmetic")
    assert d["Linf"] > 1e-4
    assert d["flux_jump"] > 1e-3


def test_hand_matrix_n2():
    A, b, h, axis = assemble(2, P1, "halfpoint")
    assert abs(h - 1.0 / 3.0) < 1e-15
    Ad = A.toarray()
    assert abs(Ad[0, 0] - 4.0 / h**2) < 1e-10
    assert np.count_nonzero(Ad) == 12


def test_fem1d_element():
    Ke = element_stiffness(0.25)
    assert np.allclose(Ke, np.array([[4.0, -4.0], [-4.0, 4.0]]))
    _, _, _, _, einf = solve_layered_aligned(32)
    assert einf < 1e-12, einf


if __name__ == "__main__":
    tests = [
        test_bank,
        test_p2_exact,
        test_p5_centre,
        test_p1_two_meshes,
        test_p4_order,
        test_layered_harmonic_nodal,
        test_layered_arithmetic_is_wrong,
        test_hand_matrix_n2,
        test_fem1d_element,
    ]
    for fn in tests:
        fn()
        print("OK", fn.__name__)
    print("all validation tests passed")
