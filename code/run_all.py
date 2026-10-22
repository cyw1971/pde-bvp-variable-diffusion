"""Reproduce every figure in the report. Run: python run_all.py"""
from pathlib import Path
import json
import sys

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
    plot_flux,
    plot_iterations_contrast,
    plot_layered_profile,
    plot_residual_history,
    plot_solution_contour,
    plot_solution_surface,
    plot_spy,
)
from solvers import (
    condition_estimate,
    residual_history_cg,
    smallest_eigenvalue,
    solve_cg,
    solve_direct,
    symmetry_relative,
)
from test_problem_bank import verify_bank
from verify import layered_diagnostics, refinement_table, truncation_discrete_l2


SMOOTH_NS = (15, 31, 63, 127)
LAYERED_NS = (30, 62, 126, 254)  # odd interval counts 31, 63, 127, 255
BLUE = "#1F6F8B"
ORANGE = "#C45C26"


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

    A, b, h, axis = assemble(15, P5, "halfpoint")
    uh, res, wall = solve_direct(A, b)
    mid = 15 // 2
    centre = float(uh[mid * 15 + mid])
    print(f"[P5] centre={centre:.6f} residual={res:.3e}")
    results["P5_centre"] = centre

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
    plot_solution_contour(uh15, 15, FIG / "fig01_smooth_contour.png",
                          r"P4: $u_h$ on $n=15$ (indexing='ij', axis 0 is $x$)")
    plot_solution_surface(uh15, 15, FIG / "fig02_smooth_surface.png",
                          r"P4 manufactured solution, $n=15$")
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

    d_show = layered_h[1]  # n=62
    d_show_a = layered_a[1]
    plot_layered_profile(d_show["axis"], d_show["uh"], d_show["n"],
                         FIG / "fig06_layered_profile.png", u_layered)
    plot_flux(d_show["axis"], d_show["flux"], d_show_a["flux"],
              FIG / "fig07_layered_flux.png")
    hs_L = [d["h"] for d in layered_h]
    plot_convergence(
        hs_L,
        [([d["Linf"] for d in layered_h], "harmonic $L^\\infty$", "o", BLUE),
         ([d["Linf"] for d in layered_a], "arithmetic $L^\\infty$", "s", ORANGE)],
        FIG / "fig08_order_layered.png",
        "Layered benchmark: harmonic is nodally exact; arithmetic is not",
        expected=2.0,
        annotation="harmonic errors sit at round-off; do not read a slope",
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
