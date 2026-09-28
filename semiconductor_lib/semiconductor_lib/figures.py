"""
Reproducible figure generators for the Semiconductor Notes pages written
2026-09-28 (crystal structure, Fermi level, dopant ionization, effective
mass, p-n junction, Deal-Grove). Each fig_* function returns a matplotlib
Figure built from the library's own physics modules, so figures and
numbers cannot drift apart.

Size notes learned the hard way (Notion inline attachments are pasted as
text, so SVG size costs tokens and has a hard cap):
  * matplotlib renders mathtext ($E_F$, 10^n log ticks) as glyph OUTLINES even
    with svg.fonttype='none', roughly doubling the file. Use plain Unicode
    text and the plain tick formatters below instead.
  * Sample smooth curves moderately (60-300 points); path.simplify drops
    redundant points on straight segments.
  * save_figure() runs the scour + precision pipeline from plotting.py;
    figures here land at 8-14 KB.

The palette below is the one used in the published notes; it differs from
plotting.COLORS on purpose (kept so regenerated figures match the pages).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from . import bands, carriers, dopants, oxidation, pnjunction
from .plotting import apply_style, full_minify_pipeline

PALETTE = dict(blue="#2b6cb0", orange="#dd6b20", gray="#718096", green="#2f855a",
               red="#c53030", purple="#6b46c1", dark="#1a202c")
C = PALETTE
LOG_FORMATTER = FuncFormatter(lambda v, _: "1e%d" % round(np.log10(v)))
PLAIN_FORMATTER = FuncFormatter(lambda v, _: "%g" % v)


def notes_style():
    """rcParams used for the notes figures (plain text, no mathtext, compact SVG)."""
    apply_style()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.8, "lines.linewidth": 1.5})


def save_figure(fig, path_stem, png=False, minify=True, dpi=110):
    """
    Save fig as <path_stem>.svg (and .png if png=True) and close it. With
    minify=True the SVG is run through plotting.full_minify_pipeline and
    written to <path_stem>_min.svg (returned); the raw SVG is kept too.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path_stem)), exist_ok=True)
    raw = f"{path_stem}.svg"
    fig.savefig(raw, format="svg", bbox_inches="tight")
    if png:
        fig.savefig(f"{path_stem}.png", format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if not minify:
        return raw
    out = f"{path_stem}_min.svg"
    full_minify_pipeline(raw, out)
    return out


# ---------------------------------------------------------------- F1 unit cell
def fig_diamond():
    """Diamond-cubic unit cell: two FCC sublattices, one tetrahedron highlighted, vacant body centre."""
    notes_style()
    A = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)] + \
        [(.5, .5, 0), (.5, .5, 1), (.5, 0, .5), (.5, 1, .5), (0, .5, .5), (1, .5, .5)]
    B = [(.25, .25, .25), (.75, .75, .25), (.75, .25, .75), (.25, .75, .75)]
    A_arr, B_arr = np.array(A, float), np.array(B, float)
    bonds = [(b, a) for b in B_arr for a in A_arr if abs(np.linalg.norm(a - b) - np.sqrt(3) / 4) < 1e-6]
    fig = plt.figure(figsize=(5.4, 4.9))
    ax = fig.add_subplot(111, projection="3d")
    edges = [((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]
    for p1, p2 in edges:
        ax.plot(*zip(p1, p2), color="#a0aec0", lw=0.7, ls=(0, (3, 2)))
    hi = B_arr[0]
    for b, a in bonds:
        is_hi = np.allclose(b, hi)
        ax.plot(*zip(b, a), color=C["orange"] if is_hi else C["gray"], lw=2.6 if is_hi else 1.1,
                alpha=1 if is_hi else 0.75)
    ax.scatter(*A_arr.T, s=110, c=C["blue"], edgecolors="white", linewidths=0.6, depthshade=False,
               label="Sublattice A: FCC (corners + face centres)")
    ax.scatter(*B_arr.T, s=140, c=C["orange"], edgecolors="white", linewidths=0.6, depthshade=False,
               label="Sublattice B: same FCC shifted by (a/4, a/4, a/4)")
    ax.scatter([.5], [.5], [.5], s=150, facecolors="none", edgecolors=C["dark"], linewidths=1.3,
               depthshade=False, label="Body centre (a/2, a/2, a/2): vacant interstitial site, not an atom")
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=17, azim=-62); ax.set_axis_off()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=7.4,
              markerscale=0.6, handletextpad=0.3)
    ax.set_title("Diamond-cubic unit cell of silicon (a = 5.431 Å, 8 atoms)\n"
                 "Thick orange bonds: one atom and its four nearest neighbours (tetrahedron)", fontsize=8)
    return fig


# ---------------------------------------------------------------- F2 Fermi level
def fig_fermi(ND=1e16, T=300.0, n_points=260):
    """n-type Si: bands, DOS, Fermi-Dirac occupation and carrier distribution side by side."""
    notes_style()
    Eg = 1.12; Ec, Ev = Eg, 0.0
    EF = Ec - carriers.ec_minus_ef(ND, dopants.Nc(T), T)
    Ed = Ec - 0.045
    E = np.linspace(-0.35, Eg + 0.35, n_points)
    f = carriers.fermi_dirac(E - EF, T)
    fh = carriers.fermi_dirac(EF - E, T)   # hole occupation computed directly (1-f loses precision near f=1)
    Dc = np.where(E > Ec, carriers.dos_3d(E - Ec, 1.18), np.nan)
    Dv = np.where(E < Ev, carriers.dos_3d(Ev - E, 0.81), np.nan)
    nE, pE = Dc * f, Dv * fh
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 3.3), sharey=True,
                            gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.0, 1.25], wspace=0.12))
    a0, a1, a2, a3 = axs
    a0.fill_between([0, 1], Ec, Ec + 0.35, color=C["blue"], alpha=0.18)
    a0.fill_between([0, 1], Ev, Ev - 0.35, color=C["orange"], alpha=0.18)
    a0.plot([0, 1], [Ec, Ec], color=C["blue"]); a0.plot([0, 1], [Ev, Ev], color=C["orange"])
    a0.plot([0, 1], [EF, EF], color=C["dark"], ls="--"); a0.plot([0.25, 0.75], [Ed, Ed], color=C["gray"], ls=":")
    a0.text(0.5, Ec + 0.12, "conduction band", ha="center", color=C["blue"])
    a0.text(0.5, Ev - 0.2, "valence band", ha="center", color=C["orange"])
    a0.text(0.5, EF - 0.06, "EF (in the gap:\nno states here)", ha="center", va="top", fontsize=7)
    a0.text(0.98, Ed - 0.012, "Ed", ha="right", va="top", color=C["gray"], fontsize=7)   # below the level: no clash with Ec
    a0.text(0.03, Ec - 0.07, "Ec", color=C["blue"]); a0.text(0.03, Ev + 0.03, "Ev", color=C["orange"])
    a0.set_xticks([]); a0.set_ylabel("Energy (eV, Ev = 0)"); a0.set_title("(a) bands", fontsize=8)
    a0.spines["bottom"].set_visible(False)
    a1.plot(Dc / 1e21, E, color=C["blue"]); a1.plot(Dv / 1e21, E, color=C["orange"])
    a1.axhspan(Ev, Ec, color="#edf2f7"); a1.set_xlabel("D(E) (10²¹ cm⁻³eV⁻¹)"); a1.set_title("(b) density of states", fontsize=8)
    a1.text(0.5, 0.5 * Eg, "no states\nin the gap", transform=a1.get_yaxis_transform(), ha="center", va="center", fontsize=7)
    a1.set_xlim(0, None)
    a2.plot(f, E, color=C["dark"]); a2.axhline(EF, color=C["dark"], ls="--", lw=0.8); a2.axvline(0.5, color=C["gray"], lw=0.5, ls=":")
    a2.set_xlabel("f(E)"); a2.set_title("(c) occupation", fontsize=8); a2.set_xlim(-0.05, 1.05)
    a2.text(0.52, EF + 0.03, "f = ½ at EF", fontsize=7, color=C["gray"])
    a3.semilogx(nE, E, color=C["blue"], label="electrons n(E)"); a3.semilogx(pE, E, color=C["orange"], label="holes p(E)")
    a3.set_xlim(1e1, 1e19); a3.xaxis.set_major_formatter(LOG_FORMATTER); a3.xaxis.set_minor_formatter(NullFormatter())
    a3.set_xticks([1e2, 1e6, 1e10, 1e14, 1e18]); a3.set_xlabel("carriers cm⁻³eV⁻¹"); a3.set_title("(d) carriers = (b)×(c)", fontsize=8)
    a3.legend(frameon=False, fontsize=7, loc="center right")
    for a in axs[1:]:
        a.axhline(Ec, color=C["blue"], lw=0.5); a.axhline(Ev, color=C["orange"], lw=0.5)
    a0.set_ylim(-0.35, Eg + 0.35)
    fig.suptitle("n-type Si, ND = 10¹⁶ cm⁻³, 300 K: EF is a parameter of f(E), not a place where electrons sit",
                 fontsize=8.5, y=1.02)
    return fig


# ---------------------------------------------------------------- F3 ionization
def fig_ionization(temps=None):
    """Fraction of dopants ionized versus temperature (phosphorus donor, boron acceptor)."""
    notes_style()
    Ts = np.linspace(25, 500, 56) if temps is None else np.asarray(temps)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    cols = [C["blue"], C["green"], C["orange"], C["red"]]
    for ax, fn, lab in [(axs[0], dopants.solve_donor, "Phosphorus donor, Ec-Ed = 45 meV, g = 2"),
                        (axs[1], dopants.solve_acceptor, "Boron acceptor, Ea-Ev = 45 meV, g = 4")]:
        for N, c in zip([1e15, 1e16, 1e17, 1e18], cols):
            y = np.array([fn(N, T).fraction for T in Ts])
            ax.plot(Ts, y, color=c, ls="-" if N < 1e18 else (0, (4, 2)),
                    label=f"{N:.0e} cm⁻³" + ("  (unreliable)" if N >= 1e18 else ""))
        ax.axvline(77, color="#a0aec0", lw=0.7, ls=":"); ax.axvline(300, color="#a0aec0", lw=0.7, ls=":")
        ax.text(80, 0.05, "77 K", fontsize=7, color=C["gray"]); ax.text(303, 0.05, "300 K", fontsize=7, color=C["gray"])
        ax.set_title(lab, fontsize=8); ax.set_xlabel("Temperature (K)"); ax.set_ylim(0, 1.03)
    axs[0].set_ylabel("Fraction of dopants ionized")
    axs[0].legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.14), title="doping", title_fontsize=7)
    fig.suptitle("Model: charge neutrality + Fermi–Dirac occupancy of the dopant level (non-degenerate, no impurity band)",
                 fontsize=8, y=1.03)
    return fig


# ---------------------------------------------------------------- F4 E-k curvature
def fig_ek(n_points=55):
    """Effective mass as band curvature: one conduction valley and the valence-band top."""
    notes_style()
    k = np.linspace(-0.9, 0.9, n_points)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axs[0]
    ax.plot(k, bands.parabolic_band(k, bands.MT_SI), color=C["blue"], label=f"transverse: mt = {bands.MT_SI} m0 (light)")
    ax.plot(k, bands.parabolic_band(k, bands.ML_SI), color=C["purple"], label=f"along valley axis: ml = {bands.ML_SI} m0 (heavy)")
    ax.set_title("(a) Si conduction-band valley (one of six)", fontsize=8); ax.set_xlabel("k - kmin (nm⁻¹)"); ax.set_ylabel("E - Ec (eV)")
    ax.set_ylim(-0.005, 0.16); ax.legend(frameon=False, fontsize=7, loc="upper center")
    ax = axs[1]
    ax.plot(k, -bands.parabolic_band(k, bands.M_HEAVY_HOLE), color=C["orange"], label=f"heavy hole: {bands.M_HEAVY_HOLE} m0")
    ax.plot(k, -bands.parabolic_band(k, bands.M_LIGHT_HOLE), color=C["red"], label=f"light hole: {bands.M_LIGHT_HOLE} m0")
    ax.plot(k, -bands.SPLIT_OFF_EV - bands.parabolic_band(k, bands.M_SPLIT_OFF), color=C["gray"],
            label=f"split-off: {bands.M_SPLIT_OFF} m0, 44 meV lower")
    ax.set_title("(b) valence-band top at Γ (isotropic approximation)", fontsize=8); ax.set_xlabel("k (nm⁻¹)"); ax.set_ylabel("E - Ev (eV)")
    ax.set_ylim(-0.19, 0.005); ax.legend(frameon=False, fontsize=7, loc="lower center")
    fig.suptitle("Effective mass = curvature: m* = ħ²/(d²E/dk²).  Parabolic approximation near band edges only; "
                 "real hole bands are warped", fontsize=8, y=1.03)
    return fig


# ---------------------------------------------------------------- F5 p-n junction
def fig_pn(NA=1e16, ND=5e16, stride=20):
    """Equilibrium p-n junction: numerical Poisson versus the depletion approximation."""
    notes_style()
    sol = pnjunction.solve_pn_poisson(NA, ND, 0.0)
    dep = pnjunction.depletion_approximation(NA, ND, 0.0)
    xu = sol.x * 1e4
    idx = np.where((xu > -0.7) & (xu < 0.35))[0][::stride]
    xs, xp_um, xn_um = xu[idx], dep["xp"] * 1e4, dep["xn"] * 1e4
    fig, axs = plt.subplots(3, 1, figsize=(5.2, 6.6), sharex=True, gridspec_kw=dict(hspace=0.16))
    a = axs[0]
    a.semilogy(xs, sol.p[idx], color=C["orange"], label="holes p(x)")
    a.semilogy(xs, sol.n[idx], color=C["blue"], label="electrons n(x)")
    a.semilogy([-0.7, 0, 0, 0.35], [NA, NA, ND, ND], color="#a0aec0", ls=":", lw=1, label="net doping (NA | ND)")
    a.axvline(-xp_um, color=C["gray"], ls="--", lw=0.8); a.axvline(xn_um, color=C["gray"], ls="--", lw=0.8)
    a.set_ylim(1e2, 3e17); a.yaxis.set_major_formatter(LOG_FORMATTER); a.yaxis.set_minor_formatter(NullFormatter())
    a.set_yticks([1e2, 1e6, 1e10, 1e14, 1e17]); a.set_ylabel("cm⁻³")
    a.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.62))
    a.set_title("p-n junction in equilibrium: NA = 10¹⁶, ND = 5×10¹⁶ cm⁻³, 300 K (numerical Poisson + Boltzmann carriers)", fontsize=7.6)
    a.text(-xp_um + 0.005, 3e2, "depletion-approx. edges\n(dashed)", fontsize=6.6, color=C["gray"])
    a = axs[1]
    a.plot(xs, sol.E[idx] / 1e3, color=C["dark"], label="numerical")
    a.plot([-0.7, -xp_um, 0, xn_um, 0.35], np.array([0, 0, -dep["Emax"], 0, 0]) / 1e3, color=C["red"], ls="--", lw=1.1,
           label="depletion approximation")
    a.set_ylabel("Field E (kV/cm)"); a.legend(frameon=False, fontsize=7, loc="lower right"); a.axhline(0, color="#cbd5e0", lw=0.6)
    a = axs[2]
    Ei = -sol.psi
    a.plot(xs, (Ei + 0.56)[idx], color=C["blue"], label="Ec"); a.plot(xs, (Ei - 0.56)[idx], color=C["orange"], label="Ev")
    a.axhline(0, color=C["dark"], ls="--", lw=1.0, label="EF (flat)")
    a.set_ylabel("Energy (eV)"); a.set_xlabel("x (µm)   [metallurgical junction at 0; p-type left, n-type right]")
    a.legend(frameon=False, fontsize=7, loc="lower left", ncol=3)
    Ecl = (Ei + 0.56)[np.argmin(np.abs(xu + 0.65))]; Ecr = (Ei + 0.56)[np.argmin(np.abs(xu - 0.32))]
    a.plot([-0.65, 0.34], [Ecl, Ecl], color="#a0aec0", lw=0.6, ls=":"); a.plot([-0.65, 0.34], [Ecr, Ecr], color="#a0aec0", lw=0.6, ls=":")
    a.annotate("", xy=(0.3, Ecr), xytext=(0.3, Ecl), arrowprops=dict(arrowstyle="<->", color=C["gray"], lw=0.8, shrinkA=0, shrinkB=0))
    a.text(0.27, 0.5 * (Ecl + Ecr), "qVbi = %.2f eV" % dep["Vbi"], fontsize=7, color=C["gray"], ha="right", va="center")
    return fig


# ---------------------------------------------------------- F5b Debye length vs. width
def fig_debye_width_scaling(Ns=None):
    """Total equilibrium depletion width W vs. the (smaller-side) Debye
    length L_D, symmetric NA=ND=N, across doping. Shows W/L_D growing
    slowly (8-13x over 1e15-1e18 cm^-3) because it is set by
    2*sqrt(Vbi/Vt), and Vbi grows only logarithmically with N while L_D
    shrinks as 1/sqrt(N)."""
    notes_style()
    Ns = Ns if Ns is not None else np.logspace(14, 19, 60)
    LD = np.array([pnjunction.debye_length(N) for N in Ns]) * 1e7  # nm
    W = np.array([pnjunction.depletion_approximation(N, N, 0.0)["W"] for N in Ns]) * 1e7
    ratio = W / LD
    fig, axs = plt.subplots(2, 1, figsize=(5.0, 5.0), sharex=True, gridspec_kw=dict(hspace=0.12))
    a = axs[0]
    a.loglog(Ns, LD, color=C["blue"], label="Debye length L_D")
    a.loglog(Ns, W, color=C["red"], label="depletion width W")
    a.set_ylabel("nm"); a.legend(frameon=False, fontsize=7, loc="lower left")
    a.set_title("Symmetric p-n junction, NA = ND = N, 300 K: edge scale vs. total width", fontsize=7.6)
    for N in (1e16, 1e18):
        i = np.argmin(np.abs(Ns - N))
        a.annotate("", xy=(Ns[i], W[i]), xytext=(Ns[i], LD[i]),
                   arrowprops=dict(arrowstyle="<->", color=C["gray"], lw=0.8, shrinkA=1, shrinkB=1))
        a.text(Ns[i] * 1.15, np.sqrt(W[i] * LD[i]), "%.0fx" % ratio[i], fontsize=6.6, color=C["gray"])
    a = axs[1]
    a.semilogx(Ns, ratio, color=C["dark"])
    a.set_ylabel("W / L_D"); a.set_xlabel("doping N (cm⁻³)")
    a.set_ylim(0, 14); a.axhline(1, color="#cbd5e0", lw=0.6)
    a.text(2e14, 1.3, "one Debye length (edge only)", fontsize=6.6, color=C["gray"])
    return fig


# ------------------------------------------------------ F5c n-type / intrinsic junction
def fig_n_intrinsic_junction(ND=1e16):
    """Abrupt n-type/intrinsic step junction: intrinsic material at x<0,
    ND-doped n-type at x>0. Shows the length scale blow up from nm to µm
    on the intrinsic side, where there are no fixed dopant ions to
    balance the exposed donor charge -- only piled-up majority electrons,
    screened over that side's own (much larger, ni-set) Debye length."""
    notes_style()
    sol = pnjunction.solve_step_junction_poisson(0.0, ND)
    xu = sol.x * 1e4  # um
    fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.2), sharex=True, gridspec_kw=dict(hspace=0.14))
    a = axs[0]
    span = 3.0
    m = (xu > -span) & (xu < span)
    a.semilogy(xu[m], sol.n[m], color=C["blue"], label="electrons n(x)")
    a.semilogy(xu[m], sol.p[m], color=C["orange"], label="holes p(x)")
    a.axhline(1e10, color="#a0aec0", ls=":", lw=1, label="n_i = 1e10 cm⁻³")
    a.set_ylim(1e3, 3e16); a.yaxis.set_major_formatter(LOG_FORMATTER); a.yaxis.set_minor_formatter(NullFormatter())
    a.set_ylabel("cm⁻³"); a.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.02, 0.5))
    a.set_title("n-type / intrinsic step junction, ND = 1e16 cm⁻³, 300 K (numerical Poisson)", fontsize=7.6)
    box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0)
    a.text(-2.8, 3e5, "intrinsic side: µm-scale spread\n(piled-up electrons only, no ions)",
           fontsize=6.4, color=C["gray"], bbox=box)
    a.text(0.35, 3e5, "n-side: nm-scale edge\n(exposed donor ions)",
           fontsize=6.4, color=C["gray"], bbox=box)
    a = axs[1]
    a.plot(xu[m], sol.psi[m], color=C["dark"])
    a.set_ylabel("Potential (V)"); a.set_xlabel("x (µm)   [step junction at 0; intrinsic left, n-type right]")
    return fig


# ---------------------------------------------------------------- F6 Deal-Grove
def fig_deal_grove(initial_oxide_um=0.023, n_points=90):
    """Deal-Grove growth, wet and dry, (111) and (100), 1000 C, with linear/parabolic asymptotes."""
    notes_style()
    t = np.logspace(-2, 2, n_points)
    Ad, Bd = oxidation.rate_constants("dry", "111"); Aw, Bw = oxidation.rate_constants("wet", "111")
    Ad1, _ = oxidation.rate_constants("dry", "100"); Aw1, _ = oxidation.rate_constants("wet", "100")
    tau111 = oxidation.tau_from_initial_oxide(initial_oxide_um, Ad, Bd)
    tau100 = oxidation.tau_from_initial_oxide(initial_oxide_um, Ad1, Bd)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.loglog(t, oxidation.thickness(t, Aw, Bw), color=C["blue"], label="wet (95 °C H₂O bubbler), (111)")
    ax.loglog(t, oxidation.thickness(t, Aw1, Bw), color=C["blue"], ls="--", label="wet, (100)")
    ax.loglog(t, oxidation.thickness(t, Ad, Bd, tau111), color=C["orange"], label="dry O₂, (111)")
    ax.loglog(t, oxidation.thickness(t, Ad1, Bd, tau100), color=C["orange"], ls="--", label="dry O₂, (100)")
    ax.axhspan(0.001, 0.03, color="#edf2f7")
    ax.text(0.011, 0.0016, "< 30 nm: Deal–Grove unreliable\n(thin-oxide regime)", fontsize=7, color=C["gray"], va="bottom")
    tt = np.array([0.01, 0.3]); ax.loglog(tt, oxidation.linear_rate(Aw, Bw) * tt, color="#a0aec0", lw=0.8, ls=":")
    ax.text(0.011, 0.024, "slope 1: linear, B/A", fontsize=6.6, color=C["gray"], rotation=32)
    tt = np.array([3, 100]); ax.loglog(tt, np.sqrt(Bw * tt), color="#a0aec0", lw=0.8, ls=":")
    ax.text(7, 0.55, "slope ½: parabolic, √(Bt)", fontsize=6.6, color=C["gray"], rotation=17)
    ax.xaxis.set_major_formatter(PLAIN_FORMATTER); ax.yaxis.set_major_formatter(PLAIN_FORMATTER)
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Oxidation time (h)"); ax.set_ylabel("Oxide thickness (µm)"); ax.set_ylim(1e-3, 4); ax.set_xlim(1e-2, 1e2)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("Deal–Grove growth at 1000 °C, 1 atm (x² + Ax = B(t+τ))\n"
                 "Constants: Deal & Grove 1965 [(111) data]; (100): A × 1.68", fontsize=7.6)
    return fig


FIGURES = {
    "silicon_diamond_cubic_unit_cell": fig_diamond,
    "fermi_level_carrier_distribution_n_type_si": fig_fermi,
    "dopant_ionization_vs_temperature": fig_ionization,
    "silicon_effective_mass_curvature": fig_ek,
    "pn_junction_equilibrium_numerical_poisson": fig_pn,
    "pn_junction_debye_length_vs_depletion_width": fig_debye_width_scaling,
    "n_type_intrinsic_step_junction_numerical_poisson": fig_n_intrinsic_junction,
    "deal_grove_oxide_growth_1000C": fig_deal_grove,
}


def build_all(outdir, minify=True, png=False):
    """Generate every notes figure into outdir; returns {name: path}."""
    return {name: save_figure(fn(), os.path.join(outdir, name), png=png, minify=minify)
            for name, fn in FIGURES.items()}
