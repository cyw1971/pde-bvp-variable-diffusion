# Variable-diffusion elliptic BVP (Project 2) — Team 15

Chuyang Wang, 2363591. Solo.

Conservative face-averaged five-point FDM for \(-\nabla\cdot(a\nabla u)=f\) on the unit square (Problem Pack Direction 3). Smooth manufactured \(a=1+x\), then layered \(a=1|10\) with harmonic versus arithmetic face coefficients.

## Contents (code + report)

| File / folder | Role |
|---|---|
| `chuyang_wang2363591.pdf` | Report PDF (AI log + ICS) |
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

Cost measure is **CG iteration count** (residual stop) or **one sparse factorisation** for `spsolve`. Wall time is secondary and is not the headline cost.

## Reproduce each figure

All figures are written by `python code/run_all.py` into `figures/`:

| File | Claim |
|---|---|
| `fig01_smooth_contour.png` | P4 field + mesh, `indexing='ij'` |
| `fig02_smooth_surface.png` | P4 surface |
| `fig03_spy.png` | sparsity after Dirichlet elimination |
| `fig04_order_smooth.png` | P4 log-log order 2 |
| `fig06_layered_profile.png` | harmonic \(u_h\) vs exact profile |
| `fig07_layered_flux.png` | interface flux, harmonic vs arithmetic |
| `fig08_order_layered.png` | layered \(L^\infty\) vs \(h\) |
| `fig09_cg_residual.png` | CG residual history |
| `fig10_cost_accuracy.png` | error vs CG iterations |
| `fig11_iterations_contrast.png` | CG iters vs \(a_R/a_L\) |

Index map (also in `code/assemble.py`): \(n\) interior nodes per direction, \(h=1/(n+1)\), \(k=i n+j\).
