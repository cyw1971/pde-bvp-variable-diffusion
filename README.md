# Variable-diffusion elliptic BVP (Project 2) — Team 15

Chuyang Wang, 2363591. Solo.

Conservative face-averaged five-point FDM for \(-\nabla\cdot(a\nabla u)=f\) on the unit square (Problem Pack Direction 3). Smooth manufactured \(a=1+x\), then layered \(a=1|10\) with harmonic versus arithmetic face coefficients.

## Contents (code + report)

| File / folder | Role |
|---|---|
| `chuyangwang_project2.pdf` | Report PDF (Overleaf; AI log + ICS) |
| `report/main.tex` | LaTeX source (compile from `report/` with figures in `../figures/`) |
| `code/` | Assembly, solvers, verification, `run_all.py` |
| `figures/` | Report figures (regenerate with `run_all.py`) |
| `requirements.txt` | Python dependencies |

## Run

```text
python -m pip install -r requirements.txt
python code/validation_tests.py
python code/run_all.py
```

`scikit-fem` is optional: it is an independent 2D P1 oracle for the smooth P4 problem. The project method is FDM.

Cost measure in the report is **wall time at matched discrete \(L^2\)** (within \(0.1\%\) of `spsolve`), with CG / AMG-CG iteration counts as the hardware-independent secondary measure. Seminar slides live in `slides/` locally and are not on GitHub.

## Reproduce each figure

All figures are written by `python code/run_all.py` into `figures/`:

| File | Claim |
|---|---|
| `fig01_p2_surface.png` | P2 gate \(x^2-y^2\) as a surface on \((x,y)\) |
| `fig02_smooth_surface.png` | P4 manufactured \(u_h\) as a surface |
| `fig14_p5_surface.png` | P5 gate \(e^x\sin(\pi y)\) as a surface |
| `fig03_spy.png` | sparsity after Dirichlet elimination |
| `fig04_order_smooth.png` | P4 log-log order 2 |
| `fig06_layered_surfaces.png` | harmonic vs arithmetic \(u_h\) as 3D fields |
| `fig15_layered_arith_error.png` | arithmetic error ridge along \(x=1/2\) |
| `fig07_layered_flux.png` | interface flux, harmonic vs arithmetic |
| `fig08_order_layered.png` | layered \(L^\infty\) vs \(h\) |
| `fig09_cg_residual.png` | CG residual history |
| `fig10_cost_accuracy.png` | error vs CG iterations |
| `fig11_iterations_contrast.png` | CG iters vs \(a_R/a_L\) |
| `fig12_cost_meshes.png` | wall time at matched \(L^2\): spsolve vs CG vs AMG-CG, \(n=31\)–\(511\) |
| `fig13_iterations_h.png` | CG and AMG-CG iterations vs \(h\) |

Index map (also in `code/assemble.py`): \(n\) interior nodes per direction, \(h=1/(n+1)\), \(k=i n+j\).
