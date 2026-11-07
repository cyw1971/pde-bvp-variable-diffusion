"""Build the 10-minute Project 2 seminar deck. Run: python slides/build_pptx.py"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
FIG = ROOT.parent / "figures"
OUT = ROOT / "Team15_Project2_slides.pptx"

NAVY = RGBColor(0x1B, 0x3A, 0x5F)
TEAL = RGBColor(0x1F, 0x6F, 0x8B)
ORANGE = RGBColor(0xC4, 0x5C, 0x26)
INK = RGBColor(0x1F, 0x29, 0x37)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CREAM = RGBColor(0xF4, 0xF1, 0xEA)


def set_run(run, size=18, bold=False, color=INK, font="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def add_text_box(slide, l, t, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    set_run(run, size=size, bold=bold, color=color)
    return box


def add_bullets(slide, l, t, w, h, items, size=16):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        p.space_after = Pt(5)
        run = p.add_run()
        run.text = item
        set_run(run, size=size, color=INK)
    return box


def bar(slide, y, color=NAVY, height=0.12):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(y), Inches(13.333), Inches(height)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    return shp


def footer(slide, n, total=12):
    bar(slide, 7.28, color=NAVY, height=0.22)
    add_text_box(
        slide, 0.4, 7.28, 9.8, 0.22,
        "Project 2  |  Variable-diffusion elliptic BVP  |  Chuyang Wang 2363591",
        size=11, color=WHITE,
    )
    add_text_box(slide, 11.4, 7.28, 1.6, 0.22, f"{n} / {total}",
                 size=11, color=WHITE, align=PP_ALIGN.RIGHT)


def takeaway(slide, text):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.4), Inches(6.52), Inches(12.5), Inches(0.68)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = CREAM
    shp.line.color.rgb = ORANGE
    tf = shp.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "Takeaway  |  " + text
    set_run(run, size=15, bold=True, color=ORANGE)


def say_box(slide, text, top=1.02, height=1.15):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.35), Inches(top), Inches(12.6), Inches(height)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = CREAM
    shp.line.color.rgb = TEAL
    tf = shp.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "READ  |  " + text
    set_run(run, size=14, bold=False, color=INK)


def content_title(slide, title, subtitle=None):
    bar(slide, 0, color=NAVY, height=0.92)
    add_text_box(slide, 0.4, 0.18, 12.5, 0.4, title, size=26, bold=True, color=WHITE)
    if subtitle:
        add_text_box(slide, 0.4, 0.55, 12.5, 0.3, subtitle, size=13, color=RGBColor(0xD4, 0xE0, 0xEC))


def picture(slide, name, l, t, w, h):
    path = FIG / name
    if not path.exists():
        raise FileNotFoundError(path)
    slide.shapes.add_picture(str(path), Inches(l), Inches(t), Inches(w), Inches(h))


def title_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    shp.fill.solid()
    shp.fill.fore_color.rgb = NAVY
    shp.line.fill.background()
    accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.18), Inches(7.5))
    accent.fill.solid()
    accent.fill.fore_color.rgb = ORANGE
    accent.line.fill.background()
    add_text_box(s, 0.7, 1.15, 12, 0.35, "Project 2  ·  PDE BVP  ·  10 min + 3 min Q&A",
                 size=16, color=RGBColor(0xC9, 0xD6, 0xE3))
    add_text_box(s, 0.7, 1.6, 12, 1.5, "Numerical Solution of a\nVariable-Diffusion Elliptic BVP",
                 size=32, bold=True, color=WHITE)
    add_text_box(
        s, 0.7, 3.35, 12.2, 2.7,
        "READ  |  Direction 3 from the problem pack: steady heat conduction, "
        "-div(a grad u)=f on the unit square. I discretise with a conservative "
        "face-averaged five-point FDM, not FEM. Smooth manufactured P4 is second order. "
        "On a jump a=1|10, the harmonic face mean is nodally exact; the arithmetic mean "
        "keeps a symmetric M-matrix but the flux is wrong. Cost is CG iterations versus "
        "one sparse factorisation; at n=31 there is no runtime crossover with spsolve.",
        size=17, color=RGBColor(0xC9, 0xD6, 0xE3),
    )
    add_text_box(s, 0.7, 6.25, 12, 0.4, "Start on the next slide. Do not introduce name or ID.",
                 size=14, color=ORANGE)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    total = 12

    title_slide(prs)

    # 2 background
    s = prs.slides.add_slide(prs.slide_layouts[6])
    content_title(s, "Why this problem", "Problem pack Direction 3  ·  FDM path  ·  not a curved-disk Poisson")
    say_box(
        s,
        "A layered composite is the model in which conductivity a(x,y) can jump. "
        "The physically correct operator is divergence form: normal flux a du/dn is continuous. "
        "Expanding a Delta u + grad a · grad u is the wrong starting point once a is discontinuous — "
        "it loses conservation, symmetry, and the M-matrix sign pattern.",
        height=1.25,
    )
    add_bullets(s, 0.45, 2.45, 12.4, 3.8, [
        "Domain Omega = (0,1)^2, Dirichlet on the whole boundary (Gamma_N empty).",
        "PDE:  -div(a grad u) = f  in Omega,   u = g  on dOmega,   a_min > 0.",
        "I chose Direction 3 so the Tutorial 5 five-point machinery stays intact, while the jump forces a modelling choice (harmonic vs arithmetic faces) that smooth Poisson never sees.",
        "Method: FDM only as the project method. scikit-fem P1 is an independent oracle, not a second scheme under test.",
        "Story of this talk: stencil and Dirichlet shift  →  P4 order 2  →  algebraic vs discretisation  →  harmonic vs arithmetic  →  cost.",
    ], size=16)
    takeaway(s, "The brief asked for a real elliptic BVP with evidence. The jump is the numerical content, not decoration.")
    footer(s, 2, total)

    # 3 model
    s = prs.slides.add_slide(prs.slide_layouts[6])
    content_title(s, "Two benchmarks, one operator", "Smooth P4 checks order  ·  layered jump checks flux")
    say_box(
        s,
        "P4: a=1+x, u*=sin(pi x)sin(pi y), g=0. The extra -pi cos(pi x)sin(pi y) in f is grad a · grad u* — a constant-coefficient Laplacian cannot capture it. "
        "Layered: a=1 for x<1/2 and a=10 for x>1/2, f=0, g=u*. Exact profile is piecewise linear, u*(0)=0, u*(1)=1, flux a u_x* = 20/11 continuous.",
        height=1.28,
    )
    add_bullets(s, 0.45, 2.48, 12.4, 3.7, [
        "P4 regularity: u in C^4, a in C^2  →  theory says order 2 in discrete L2 and L-infinity.",
        "Layered regularity: u* is only Lipschitz, a not C^2  →  the order-2 theorem does not apply.",
        "Well-posed: uniform ellipticity + nonempty Dirichlet boundary; no compatibility condition.",
        "Expected signs: layered u* in [0,1] with extrema on the boundary (maximum principle).",
        "Discrete L2 means h ||U||_2. L-infinity is the max nodal |error|. Residual ||b-AU||_infinity is algebraic, not discretisation.",
    ], size=16)
    takeaway(s, "P4 answers “is the stencil second order?”. The jump answers “does symmetry imply correct physics?”.")
    footer(s, 3, total)

    # 4 discretisation
    s = prs.slides.add_slide(prs.slide_layouts[6])
    content_title(s, "Discretisation: shared faces, Dirichlet in b", "n interior nodes/direction,  h=1/(n+1),  k=i n+j")
    say_box(
        s,
        "Conservative five-point: (A_h U)_ij = h^{-2} [ a_E(U-U_E)+a_W(U-U_W)+a_N(U-U_N)+a_S(U-U_S) ]. "
        "The east coefficient of (i,j) is the west coefficient of (i+1,j). That sharing, not the particular average, makes A_h symmetric after elimination. "
        "Boundary nodes are not unknowns: b_k += a_face g / h^2. Replacing a Dirichlet row by e_k and leaving the column destroys symmetry.",
        height=1.35,
    )
    picture(s, "fig03_spy.png", 0.35, 2.52, 4.55, 3.75)
    add_bullets(s, 5.1, 2.52, 7.7, 3.75, [
        "Smooth a: evaluate at the half-point. For a=1+x this is second-order.",
        "Jump: harmonic (resistance) mean  2 a_L a_R/(a_L+a_R).",
        "Why harmonic: integrate u_x=q/a across two half-edges → series resistors.",
        "Arithmetic (a_L+a_R)/2 is a different resistance unless a_L=a_R.",
        "Odd interval counts so x=1/2 bisects a grid edge.",
        "Spy (n=15, 225×225): five-point bandwidth after Dirichlet elimination.",
        "Energy: U^T A U = sum_faces a_face (delta U)^2 / h^2  ≥ a_min |U|_{H^1_h}^2.",
    ], size=14)
    takeaway(s, "Symmetry comes from shared faces. Harmonic averaging is forced by flux continuity, not by taste.")
    footer(s, 4, total)

    # 5 P4 field + order
    s = prs.slides.add_slide(prs.slide_layouts[6])
    content_title(s, "Verification: P4 manufactured solution", "Named norms  ·  four meshes  ·  no round-off floor")
    say_box(
        s,
        "n=15,31,63,127. Discrete L2 and L-infinity both fit order 2.00 (SE of the fit < 10^{-3}). "
        "P5 centre u_h(0.5,0.5)=1.652 checks the Dirichlet sign. P2 (quadratic) is exact to 3.3×10^{-16}. "
        "A 2D P1 FEM oracle on different triangles is only an order-of-magnitude cross-check, not a digit match.",
        height=1.22,
    )
    picture(s, "fig01_smooth_contour.png", 0.3, 2.38, 4.15, 3.9)
    picture(s, "fig04_order_smooth.png", 4.55, 2.38, 4.35, 3.9)
    add_bullets(s, 9.05, 2.38, 3.9, 3.9, [
        "Contour: indexing='ij', axis 0 is x; nodes plotted.",
        "n=15 L2=1.598e-3.",
        "n=31 L2=3.988e-4, p=2.00.",
        "n=127 L2=2.491e-5, p=2.00.",
        "Truncation quarters each doubling.",
        "||A-A^T||_max = 0,  λ_min=28.61.",
    ], size=14)
    takeaway(s, "On smooth data the face-averaged stencil is second order in both named norms.")
    footer(s, 5, total)

    # 6 algebraic vs disc
    s = prs.slides.add_slide(prs.slide_layouts[6])
    content_title(s, "Algebraic error is not discretisation error", "Stop CG on the residual, never on ||x^{k+1}-x^k||")
    say_box(
        s,
        "spsolve has no tolerance: treat it as the exact discrete solution and report ||b-Au||_∞. "
        "On P4, n=31 that residual is 3×10^{-12} while discrete L2 is 3.99×10^{-4}. "
        "CG at relative 10^{-2} still sits above the direct L2; from 10^{-4} the L2 locks. Tightening further does not change the reported order.",
        height=1.22,
    )
    picture(s, "fig09_cg_residual.png", 0.3, 2.38, 6.15, 3.9)
    picture(s, "fig10_cost_accuracy.png", 6.6, 2.38, 6.3, 3.9)
    takeaway(s, "Relative residual 10^{-4} is the loosest CG stop that reproduces the direct L2. The slope is discretisation.")
    footer(s, 6, total)

    # 7 CG vs spsolve numbers
    s = prs.slides.add_slide(prs.slide_layouts[6])
    content_title(s, "Cost: CG iterations vs one factorisation", "Matched target = spsolve L2 = 3.988×10^{-4} on n=31")
    say_box(
        s,
        "Cost named first: CG iteration count to a stated relative residual, versus one sparse factorisation. "
        "Not counted: assembly, eigsh, plotting. At this mesh both finishes are about 2 ms — a negative result for unpreconditioned CG. "
        "On the finest P4 mesh n=127, dropping CG from 10^{-3} to 10^{-4} (one order) locks L2 onto 2.491×10^{-5}.",
        height=1.22,
    )
    add_bullets(s, 0.45, 2.42, 12.4, 3.85, [
        "CG rel. 10^{-2}:  26 iters,  residual 7.5×10^{-1},  L2=4.329×10^{-4}  (still algebraic contamination).",
        "CG rel. 10^{-4}:  39 iters,  residual 5.9×10^{-3},  L2=3.988×10^{-4}  (matches spsolve).",
        "CG rel. 10^{-6} … 10^{-10}:  L2 unchanged. Floor = discretisation.",
        "spsolve:  one factorisation, residual ~10^{-12},  L2=3.988×10^{-4},  wall ≈ 2 ms.",
        "n=127: CG 10^{-3} → L2=2.504×10^{-5};  CG 10^{-4} → L2=2.491×10^{-5} = direct.",
        "This is not an FDM-versus-FEM cost plot. FEM is only the P4 oracle.",
    ], size=16)
    takeaway(s, "At matched discretisation accuracy there is no CG–direct crossover for n≤31.")
    footer(s, 7, total)

    # 8 depth profile
    s = prs.slides.add_slide(prs.slide_layouts[6])
    content_title(s, "Depth comparison: harmonic vs arithmetic", "Same jump, same odd-N grids, same Dirichlet data, same spsolve")
    say_box(
        s,
        "Hypothesis under test: any positive face average that keeps A symmetric is an acceptable interface scheme. "
        "Harmonic averaging is nodally exact (L-infinity ≤ 3×10^{-14}) and the numerical flux equals 20/11. "
        "Arithmetic averaging is still an M-matrix — and the flux jump stays 0.66 under refinement.",
        height=1.22,
    )
    picture(s, "fig06_layered_profile.png", 0.3, 2.38, 6.2, 3.9)
    picture(s, "fig07_layered_flux.png", 6.65, 2.38, 6.25, 3.9)
    takeaway(s, "Symmetry is necessary and not sufficient. Flux continuity selects the harmonic mean.")
    footer(s, 8, total)

    # 9 layered order
    s = prs.slides.add_slide(prs.slide_layouts[6])
    content_title(s, "The jump predicted the order drop", "Deviation from order 2 is theory, not a bug in the table")
    say_box(
        s,
        "On smooth P4, consistency plus M-matrix / energy stability imply order 2. Across the jump that regularity fails. "
        "Harmonic averaging restores the 1D transmission identity, so nodal error sits at round-off. "
        "Arithmetic tracks first-order L-infinity (1.94×10^{-2} down to 2.38×10^{-3} as N=31…255). Both solutions stay in [0,1]: DMP does not pick the physics.",
        height=1.28,
    )
    picture(s, "fig08_order_layered.png", 0.3, 2.48, 6.5, 3.8)
    add_bullets(s, 7.0, 2.48, 5.9, 3.8, [
        "Harmonic L-infinity ~ 10^{-14} on four meshes.",
        "Harmonic flux jump ≤ 5×10^{-13}.",
        "Arithmetic flux jump ≈ 0.66 = O(1), not O(h).",
        "1D P1 FEM with x=1/2 a node: L-infinity 6.7×10^{-16}.",
        "Two methods, neither stencil nor mesh shared, agree on nodal values.",
    ], size=15)
    takeaway(s, "A colour plot of the arithmetic solution still looks stable. The flux plot is the falsifier.")
    footer(s, 9, total)

    # 10 contrast + limitations
    s = prs.slides.add_slide(prs.slide_layouts[6])
    content_title(s, "Contrast raises CG cost; it does not break DMP", "Limitations are part of the grade, not an apology")
    say_box(
        s,
        "Layered harmonic matrix, n=30, residual stop 10^{-10}: CG iterations 125, 227, 401, 615 at contrasts 2, 10, 50, 200. "
        "On P4, kappa_2(A) tracks h^{-2}. Second-order is for smooth a. Harmonic exactness here is a 1D aligned-grid identity, not a theorem for cuts through cells. AMG-CG is the next solver, not a result of this talk.",
        height=1.28,
    )
    picture(s, "fig11_iterations_contrast.png", 0.3, 2.48, 6.6, 3.8)
    add_bullets(s, 7.1, 2.48, 5.8, 3.8, [
        "Cost here = iterations, not wall time.",
        "Contrast is expensive for unpreconditioned CG.",
        "Not a reason to switch to arithmetic faces.",
        "Not implemented: AMG on the FDM matrix.",
        "Not implemented: 2D interface-fitted FEM beyond the P4 oracle.",
        "Cartesian Dirichlet only; a cut-cell disk needs another boundary row.",
    ], size=15)
    takeaway(s, "Unpreconditioned CG will not be the iterative method of choice on finer 2D meshes.")
    footer(s, 10, total)

    # 11 conclusions
    s = prs.slides.add_slide(prs.slide_layouts[6])
    content_title(s, "Conclusions  (end here; backup is Q&A only)")
    say_box(
        s,
        "A conservative face-averaged five-point scheme for -div(a grad u), Dirichlet data in b, is second order on smooth P4. "
        "On the jump, harmonic faces are nodally exact and keep flux 20/11; arithmetic faces give first-order error and an O(1) flux jump. "
        "CG matches spsolve L2 once the relative residual is 10^{-4}; no runtime crossover at n=31. Thank you.",
        height=1.28,
    )
    add_bullets(s, 0.45, 2.48, 12.4, 3.75, [
        "P4: fitted order 2.00 in discrete L2 and L-infinity on four meshes; ||A-A^T||=0; λ_min>0.",
        "Gates: P5 centre 1.652; P2 L-infinity 3.3×10^{-16}; bank self-check 5.8×10^{-6}.",
        "CG n=31: L2 locks at 3.988×10^{-4} from rel. 10^{-4}; spsolve residual ~10^{-12}.",
        "Layered: harmonic L-infinity ~10^{-14}, flux 20/11; arithmetic p≈1, flux jump 0.66.",
        "Reproduce: python code/run_all.py and python code/validation_tests.py in cyw1971/pde-bvp-variable-diffusion.",
    ], size=16)
    takeaway(s, "Trust the order on smooth data and the flux on the jump. Do not trust symmetry alone.")
    footer(s, 11, total)

    # 12 Q&A
    s = prs.slides.add_slide(prs.slide_layouts[6])
    content_title(s, "Backup / Q&A  (do not present this slide)")
    add_bullets(s, 0.4, 1.12, 12.5, 5.95, [
        "Q: FDM or FEM?  A: Project method is FDM. FEM P1 is an oracle (P4 L2=5.18e-3 on a different mesh). No matched FDM–FEM cost table.",
        "Q: bounded / monotone ⇒ accurate?  A: No. Arithmetic is an M-matrix and stays in [0,1], but flux jump is 0.66. Accuracy is named-norm error + flux.",
        "Q: why not pin Dirichlet rows?  A: Leaves columns in place, destroys symmetry, kills CG/energy. Boundary nodes are absent from A.",
        "Q: why odd N?  A: So x=1/2 bisects an edge; harmonic mean is then exact for piecewise-constant a.",
        "Q: residual vs iterate difference?  A: Course rule. ||x^{k+1}-x^k|| can stall while ||b-Ax|| is large.",
        "Q: is 10^{-4} residual ≪ discretisation?  A: Forward L2 no longer changes. Residual 5.9e-3 is not itself the algebraic error in the solution; the locked L2 is the evidence.",
        "Q: modelling vs discretisation vs algebraic?  A: ||u-u*|| model; ||u*-A^{-1}b|| grid; ||A^{-1}b - û|| solver. Fix by better PDE / finer h / tighter tol.",
    ], size=14)
    footer(s, 12, total)

    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
