"""Reproduce every figure in the report. Run: python run_all.py"""
from pathlib import Path
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
FIG = ROOT.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

from assemble import assemble, grid_from_n, reshape_field
from fem1d import element_stiffness, solve_layered_aligned, solve_poisson_sin
from fem2d_oracle import solve_p4 as fem2d_p4
from model import FLUX_EXACT, LAYERED, P1, P2, P4, P5, u_layered
from plots import (
    plot_convergence,
    plot_cost_accuracy,
    plot_cost_meshes,
    plot_flux,
    plot_iterations_contrast,
    plot_iterations_h,
    plot_residual_history,
    plot_solution_surface,
    plot_spy,
    plot_two_surfaces,
)
from solvers import (
    condition_estimate,
    residual_history_cg,
    setup_amg,
    smallest_eigenvalue,
    solve_amg_cg,
    solve_cg,
    solve_direct,
    symmetry_relative,
)
from test_problem_bank import verify_bank
from verify import layered_diagnostics, refinement_table, truncation_discrete_l2


SMOOTH_NS = (15, 31, 63, 127)
LAYERED_NS = (30, 62, 126, 254)  # odd interval counts 31, 63, 127, 255
COST_NS = (31, 63, 127, 255, 511)
TOLS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
MATCH_REL = 1e-3
TIMING_REPEATS = 5
BLUE = "#1F6F8B"
ORANGE = "#C45C26"
NAVY = "#1B3A5F"


def _median_wall(fn, repeats=TIMING_REPEATS):
    """Return (last result, median wall time) over repeated calls."""
    walls, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        walls.append(time.perf_counter() - t0)
    return out, float(np.median(walls))


def _print_table(name, rows, slopes):
    print(f"\n=== {name} ===")
    print(f"{'n':>6} {'h':>10} {'L2':>12} {'Linf':>12} {'residual':>12} {'p_L2':>8} {'p_Linf':>8}")
    for r in rows:
        p2 = "---" if r["order_L2"] is None else f"{r['order_L2']:.2f}"
        pinf = "---" if r["order_Linf"] is None else f"{r['order_Linf']:.2f}"
        print(f"{r['n']:6d} {r['h']:10.6f} {r['L2']:12.3e} {r['Linf']:12.3e} "
              f"{r['residual']:12.3e} {p2:>8} {pinf:>8}")
    sl2, se2, d2 = slopes["L2"]
    slinf, seinf, dinf = slopes["Linf"]
    print(f"fitted L2   {sl2:.3f} ± {se2:.3f}  (max log-dev {d2:.3f})")
    print(f"fitted Linf {slinf:.3f} ± {seinf:.3f}  (max log-dev {dinf:.3f})")


def main():
    results = {}
    print("[bank]", verify_bank())

    # --- sign / boundary ---
    e2, einf, res, *_rest = [None] * 4
    A, b, h, axis = assemble(15, P2, "halfpoint")
    uh, res, wall = solve_direct(A, b)
    from verify import errors
    e2, einf, _ = errors(uh, 15, h, P2)
    print(f"[P2] n=15 Linf={einf:.3e} residual={res:.3e}")
    results["P2_Linf"] = einf
    plot_solution_surface(
        uh, 15, FIG / "fig01_p2_surface.png",
        r"P2 gate: $u_h=x^2-y^2$ (exact on this quadratic), $n=15$",
        zlabel=r"$u_h$",
    )

    A, b, h, axis = assemble(15, P5, "halfpoint")
    uh, res, wall = solve_direct(A, b)
    mid = 15 // 2
    centre = float(uh[mid * 15 + mid])
    print(f"[P5] centre={centre:.6f} residual={res:.3e}")
    results["P5_centre"] = centre
    plot_solution_surface(
        uh, 15, FIG / "fig14_p5_surface.png",
        r"P5 gate: $u_h=e^x\sin(\pi y)$, $n=15$, centre $u_h(1/2,1/2)=1.652$",
        zlabel=r"$u_h$",
    )

    # --- smooth P4 MMS ---
    rows_p4, slopes_p4 = refinement_table(SMOOTH_NS, P4, "halfpoint")
    _print_table("P4 smooth a=1+x", rows_p4, slopes_p4)
    results["P4_slopes"] = {
        "L2": slopes_p4["L2"],
        "Linf": slopes_p4["Linf"],
        "rows": [
            {k: (float(v) if isinstance(v, (float, np.floating)) else v)
             for k, v in r.items() if k not in ("uh", "A", "b")}
            for r in rows_p4
        ],
    }
    uh15 = rows_p4[0]["uh"]
    plot_solution_surface(
        uh15, 15, FIG / "fig02_smooth_surface.png",
        r"P4 manufactured: $a=1+x$, $u_h=\sin(\pi x)\sin(\pi y)$, $n=15$",
        zlabel=r"$u_h$", zlim=(0.0, 1.0),
    )
    plot_spy(rows_p4[0]["A"], FIG / "fig03_spy.png")
    hs = [r["h"] for r in rows_p4]
    plot_convergence(
        hs,
        [([r["L2"] for r in rows_p4], r"discrete $L^2$", "o", BLUE),
         ([r["Linf"] for r in rows_p4], r"$L^\infty$", "s", ORANGE)],
        FIG / "fig04_order_smooth.png",
        r"Smooth P4: observed order $2$ in $L^2$ and $L^\infty$",
        annotation="spsolve residual $10^{-13}$–$10^{-10}$; no solver floor",
    )

    tau_rows = []
    for n in SMOOTH_NS:
        h_t, tau_l2, tau_inf = truncation_discrete_l2(n, P4, "halfpoint")
        tau_rows.append((n, h_t, tau_l2, tau_inf))
        print(f"[truncation P4] n={n} h||tau||_2={tau_l2:.3e} ||tau||_inf={tau_inf:.3e}")
    results["P4_truncation"] = tau_rows

    # --- structure ---
    A15 = rows_p4[0]["A"]
    sym = symmetry_relative(A15)
    lam_min = smallest_eigenvalue(A15)
    kappa, lam_min_c, lam_max = condition_estimate(A15)
    print(f"[SPD P4 n=15] ||A-A.T||/||A||={sym:.1e}  lam_min={lam_min:.3e}  kappa~{kappa:.3e}")
    results["P4_structure"] = {"symmetry": sym, "lam_min": lam_min, "kappa": kappa}

    # --- layered harmonic vs arithmetic ---
    layered_h = []
    layered_a = []
    for n in LAYERED_NS:
        dh = layered_diagnostics(n, "harmonic")
        da = layered_diagnostics(n, "arithmetic")
        layered_h.append(dh)
        layered_a.append(da)
        print(f"[layered n={n}] harm Linf={dh['Linf']:.3e} flux_jump={dh['flux_jump']:.3e} "
              f"| arith Linf={da['Linf']:.3e} flux_jump={da['flux_jump']:.3e} "
              f"maxprin={dh['inside_extrema']}")

    def _lite(d):
        return {k: (float(v) if isinstance(v, (float, np.floating, np.integer)) else v)
                for k, v in d.items() if k not in ("uh", "A", "b", "flux", "axis")}

    results["layered_harmonic"] = [_lite(d) for d in layered_h]
    results["layered_arithmetic"] = [_lite(d) for d in layered_a]

    d_show = layered_h[0]  # n=30, odd N=31 so the kink is visible
    d_show_a = layered_a[0]
    plot_two_surfaces(
        d_show["uh"], r"Harmonic $u_h$ ($x$, $y$ both shown)",
        d_show_a["uh"], r"Arithmetic $u_h$ ($x$, $y$ both shown)",
        d_show["n"], FIG / "fig06_layered_surfaces.png",
        zlabel=r"$u_h$", zlim=(0.0, 1.0), interface=True,
    )
    axis, X, Y, h_g = grid_from_n(d_show["n"])
    U_a = reshape_field(d_show_a["uh"], d_show["n"])
    U_ex = u_layered(X, Y)
    plot_solution_surface(
        (U_a - U_ex).ravel(), d_show["n"], FIG / "fig15_layered_arith_error.png",
        r"Arithmetic error $u_h-u^*$: ridge along $x=1/2$, larger near $y=0,1$",
        zlabel=r"$u_h-u^*$", interface=True, cmap="coolwarm",
    )
    plot_flux(d_show["axis"], d_show["flux"], d_show_a["flux"],
              FIG / "fig07_layered_flux.png")
    hs_L = [d["h"] for d in layered_h]
    plot_convergence(
        hs_L,
        [([d["Linf"] for d in layered_h], "harmonic $L^\\infty$", "o", BLUE),
         ([d["Linf"] for d in layered_a], "arithmetic $L^\\infty$", "s", ORANGE)],
        FIG / "fig08_order_layered.png",
        "Layered benchmark: harmonic is nodally exact; arithmetic is $O(h)$",
        expected=1.0,
        annotation="harmonic errors sit at round-off ($\\sim 10^{-14}$)",
        ref_index=1,
        fit_indices={1},
    )

    # --- FEM 1D oracle ---
    print("[FEM1D] K^e(h=1/4)=\n", element_stiffness(0.25, 1.0))
    for n in (8, 16, 32, 64):
        h1, e2, einf, res = solve_poisson_sin(n)
        print(f"[FEM1D sin] n={n} L2={e2:.3e} Linf={einf:.3e} residual={res:.1e}")
    nodes, uh1, exact1, res1, einf1 = solve_layered_aligned(32)
    print(f"[FEM1D layered aligned] Linf={einf1:.3e} residual={res1:.1e}")
    results["fem1d_layered_Linf"] = einf1

    fem2 = fem2d_p4(4)
    if fem2:
        print(f"[FEM2D P4] h_max={fem2['h_max']:.3e} L2={fem2['L2']:.3e} residual={fem2['residual']:.1e}")
        results["fem2d_p4"] = fem2
    else:
        print("[FEM2D] scikit-fem not available; skipped")

    # --- algebraic vs discretisation: CG sweep on P4 n=31 ---
    n_alg = 31
    A, b, h, axis = assemble(n_alg, P4, "halfpoint")
    uh_dir, res_dir, wall_dir = solve_direct(A, b)
    e2_dir, einf_dir, _ = errors(uh_dir, n_alg, h, P4)
    print(f"[direct n=31] L2={e2_dir:.3e} residual={res_dir:.3e} wall={wall_dir:.3f}s")
    hist = residual_history_cg(A, b, tol=1e-10)
    plot_residual_history(hist, 1e-10, FIG / "fig09_cg_residual.png")
    results["cg_hist_len"] = int(len(hist) - 1)

    tols = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    cg_points_cost, cg_points_err, cg_iters = [], [], []
    for tol in tols:
        uh, res, iters, conv, wall = solve_cg(A, b, tol=tol)
        e2, einf, _ = errors(uh, n_alg, h, P4)
        print(f"[CG tol={tol:.0e}] iters={iters} conv={conv} L2={e2:.3e} res={res:.3e} wall={wall:.3f}s")
        cg_points_cost.append(iters)
        cg_points_err.append(e2)
        cg_iters.append(iters)
    plot_cost_accuracy(
        [
            (np.array(cg_points_cost), np.array(cg_points_err), "CG (iters)", "o", BLUE),
            (np.array([1.0]), np.array([e2_dir]), "spsolve (one factorisation)", "s", ORANGE),
        ],
        FIG / "fig10_cost_accuracy.png",
        target=e2_dir,
    )
    results["alg_vs_disc"] = {
        "direct_L2": e2_dir,
        "direct_residual": res_dir,
        "direct_wall": wall_dir,
        "cg": list(zip(tols, cg_iters, cg_points_err)),
    }

    # --- cost at matched accuracy across meshes (P4) ---
    # Target: discrete L2 within 0.1% of the spsolve value on the same mesh.
    # Each solver gets the loosest tolerance from TOLS that meets it.
    # Not counted: assembly, eigenvalue estimates, plotting, interpreter start.
    cost_rows = []
    for n in COST_NS:
        A, b, h, axis = assemble(n, P4, "halfpoint")
        (uh_d, _, _), t_dir = _median_wall(lambda: solve_direct(A, b))
        e_dir = errors(uh_d, n, h, P4)[0]

        def matched(uh):
            return abs(errors(uh, n, h, P4)[0] - e_dir) <= MATCH_REL * e_dir

        tol_cg = next(t for t in TOLS if matched(solve_cg(A, b, tol=t)[0]))
        (_, _, it_cg, _, _), t_cg = _median_wall(lambda: solve_cg(A, b, tol=tol_cg))

        ml_probe = setup_amg(A)
        tol_amg = next(t for t in TOLS if matched(solve_amg_cg(ml_probe, b, tol=t)[0]))
        ml, t_setup = _median_wall(lambda: setup_amg(A))
        (_, it_amg), t_amg = _median_wall(lambda: solve_amg_cg(ml, b, tol=tol_amg))

        row = {
            "n": n, "h": h, "unknowns": n * n, "L2_target": e_dir,
            "spsolve_wall": t_dir,
            "cg_tol": tol_cg, "cg_iters": it_cg, "cg_wall": t_cg,
            "amg_tol": tol_amg, "amg_iters": it_amg,
            "amg_setup_wall": t_setup, "amg_solve_wall": t_amg,
            "amg_total_wall": t_setup + t_amg,
        }
        cost_rows.append(row)
        print(f"[cost n={n}] N={n*n} L2={e_dir:.3e} | spsolve {t_dir:.4f}s | "
              f"CG tol={tol_cg:.0e} it={it_cg} {t_cg:.4f}s | AMG-CG tol={tol_amg:.0e} "
              f"it={it_amg} setup {t_setup:.4f}s solve {t_amg:.4f}s total {t_setup + t_amg:.4f}s")
    results["cost_meshes"] = cost_rows
    plot_cost_meshes(
        [r["unknowns"] for r in cost_rows],
        [
            ([r["spsolve_wall"] for r in cost_rows], "spsolve (one factorisation)", "s", ORANGE, "-"),
            ([r["cg_wall"] for r in cost_rows], "CG (unpreconditioned)", "o", BLUE, "-"),
            ([r["amg_total_wall"] for r in cost_rows], "AMG-CG (setup + solve)", "^", NAVY, "-"),
            ([r["amg_solve_wall"] for r in cost_rows], "AMG-CG (solve only)", "^", NAVY, ":"),
        ],
        FIG / "fig12_cost_meshes.png",
    )
    plot_iterations_h(
        [r["h"] for r in cost_rows],
        [r["cg_iters"] for r in cost_rows],
        [r["amg_iters"] for r in cost_rows],
        FIG / "fig13_iterations_h.png",
    )

    # --- contrast vs CG iterations (layered, harmonic, n=30) ---
    contrasts = [2.0, 10.0, 50.0, 200.0]
    iters_c = []
    from model import LayeredProblem, a_layered as a_lay

    class ContrastProblem(LayeredProblem):
        def __init__(self, ar):
            self.ar = ar
            self.coefficient = lambda x, y, ar=ar: np.where(np.asarray(x) < 0.5, 1.0, ar)

    for ar in contrasts:
        prob = ContrastProblem(ar)
        # exact 1D solution for general contrast: C=1/(1+ar/2 wait)
        # u_L = A x, A = ar C, u(1)=1 => C + 9/2? General:
        # A=ar*C, D=A/2-C/2, C+D=1 => C + (ar C)/2 - C/2 = 1
        # C (1 + ar/2 - 1/2) = 1 => C (0.5 + ar/2) = 1 => C = 1/(0.5(1+ar)) = 2/(1+ar)
        # A = ar*2/(1+ar), D = A/2 - C/2
        C = 2.0 / (1.0 + ar)
        Acoef = ar * C
        D = 0.5 * Acoef - 0.5 * C

        def exact(x, y, Acoef=Acoef, C=C, D=D):
            x = np.asarray(x, dtype=float)
            return np.where(x <= 0.5, Acoef * x, C * x + D)

        prob.exact = exact
        Amat, bvec, h, axis = assemble(30, prob, "harmonic")
        uh, res, it, conv, wall = solve_cg(Amat, bvec, tol=1e-10)
        print(f"[contrast {ar:g}] CG iters={it} conv={conv} residual={res:.3e}")
        iters_c.append(it)
    plot_iterations_contrast(contrasts, iters_c, FIG / "fig11_iterations_contrast.png")
    results["contrast_iters"] = list(zip(contrasts, iters_c))

    # kappa vs mesh for P4
    kappas = []
    for n in (15, 31, 63):
        A, b, h, axis = assemble(n, P4, "halfpoint")
        kap, lmin, lmax = condition_estimate(A)
        print(f"[kappa P4 n={n}] {kap:.3e}  lam_min={lmin:.3e}")
        kappas.append((n, kap, lmin, lmax))
    results["kappa_P4"] = kappas

    out = FIG / "results.json"

    def _jsonable(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    out.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    print(f"\nwrote figures to {FIG}")
    print("done.")


if __name__ == "__main__":
    main()
