"""
Figure generators for the biased-diode and MOS-capacitor Semiconductor Notes
pages written 2026-09-29: "PN Junction Diode Current", "PN Diode Charge
Storage", and the MOS Capacitor flat-band/threshold update. Same style and
size rules as figures.py (plain Unicode text, no mathtext; minified SVGs
6-16 KB). Physics comes from diode.py, electrostatics.py and mosfet.py.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from .figures import C, LOG_FORMATTER, notes_style, save_figure

# ============================================================ biased diode (2026-09-29)
_DIODE = dict(NA=1e16, ND=1e20, Dn=30.0, Dp=2.0, tau_n=1e-6, tau_p=1e-7)


def fig_diode_forward_iv(area=1e-4, Rs=10.0, tau_dep=1e-6, n_points=140):
    """Forward I-V of a one-sided N+P diode: diffusion (n=1) and SCR
    recombination (n=2) components plus series resistance, with the local
    ideality factor underneath. Analytical model (diode.forward_iv)."""
    from . import diode
    notes_style()
    p = _DIODE
    V = np.linspace(0.02, 1.0, n_points)
    r = diode.forward_iv(V, area, Rs, p["NA"], p["ND"], p["Dn"], p["Dp"], p["tau_n"], p["tau_p"], tau_dep)
    r0 = diode.forward_iv(V, area, 0.0, p["NA"], p["ND"], p["Dn"], p["Dp"], p["tau_n"], p["tau_p"], tau_dep)
    n = diode.local_ideality(V, r["I"])
    fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.4), sharex=True, gridspec_kw=dict(hspace=0.1, height_ratios=[2.2, 1]))
    a = axs[0]
    a.semilogy(V, r0["I_diff"], color=C["blue"], ls="--", lw=1.1, label="diffusion (quasi-neutral), n = 1")
    a.semilogy(V, r0["I_scr"], color=C["orange"], ls="--", lw=1.1, label="SCR recombination, n = 2")
    a.semilogy(V, r0["I"], color=C["gray"], ls=":", lw=1.1, label="sum, no series R")
    a.semilogy(V, r["I"], color=C["dark"], label="terminal current, Rs = %g Ω" % Rs)
    a.set_ylim(1e-12, 1e-1); a.yaxis.set_major_formatter(LOG_FORMATTER); a.yaxis.set_minor_formatter(NullFormatter())
    a.set_yticks([1e-12, 1e-9, 1e-6, 1e-3]); a.set_ylabel("Current (A)")
    a.legend(frameon=False, fontsize=6.8, loc="lower right")
    a.set_title("N+P diode, NA = 1e16 cm⁻³, τn = τdep = 1 µs, area 100×100 µm², 300 K (analytical model)", fontsize=7.4)
    a.text(0.03, 3e-8, "slope 120 mV/dec\n(SCR dominates)", fontsize=6.6, color=C["orange"])
    a.text(0.50, 1e-3, "slope 60 mV/dec", fontsize=6.6, color=C["blue"])
    a.text(0.78, 2e-6, "IR drop\nbends curve", fontsize=6.6, color=C["dark"])
    a = axs[1]
    a.plot(V, n, color=C["dark"])
    a.axhline(1, color="#cbd5e0", lw=0.6); a.axhline(2, color="#cbd5e0", lw=0.6)
    a.set_ylim(0.8, 4.0); a.set_ylabel("local ideality n")
    a.set_xlabel("Terminal voltage (V)")
    return fig


def fig_diode_leakage_arrhenius(VR=3.0, tau_g=1e-6, n_points=70):
    """Reverse leakage components vs temperature: diffusion (∝ ni², Ea ≈ Eg)
    and depletion-region generation (∝ ni, Ea ≈ Eg/2). Analytical model."""
    from . import diode
    notes_style()
    p = _DIODE
    T = np.linspace(250, 500, n_points)
    Jd, Jg = diode.reverse_leakage_components(T, VR, p["NA"], p["ND"], p["Dn"], p["Dp"], p["tau_n"], p["tau_p"], tau_g)
    Ed, Eg = diode.arrhenius_activation_energy(T, Jd), diode.arrhenius_activation_energy(T, Jg)
    x = 1000.0 / T
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    ax.semilogy(x, Jd, color=C["blue"], label="diffusion (neutral region), fitted Ea = %.2f eV" % Ed)
    ax.semilogy(x, Jg, color=C["orange"], label="generation (depletion region), fitted Ea = %.2f eV" % Eg)
    ax.semilogy(x, Jd + Jg, color=C["dark"], ls=":", lw=1.2, label="total")
    i = np.argmin(np.abs(np.log(Jd / Jg)))
    ax.axvline(x[i], color=C["gray"], ls="--", lw=0.8)
    ax.text(x[i] + 0.05, 1e-12, "crossover ≈ %.0f °C" % (T[i] - 273.15), fontsize=6.8, color=C["gray"])
    ax.yaxis.set_major_formatter(LOG_FORMATTER); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("1000 / T (1/K)"); ax.set_ylabel("Leakage current density (A/cm²)")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    sec = ax.secondary_xaxis("top", functions=(lambda v: 1000.0 / v - 273.15, lambda c: 1000.0 / (c + 273.15)))
    sec.set_xticks([0, 50, 100, 150, 200]); sec.set_xlabel("°C", fontsize=7)
    ax.set_title("N+P junction at %g V reverse, NA = 1e16, τn = τg = 1 µs (analytical; D, τ held fixed)" % VR, fontsize=7.2)
    return fig


def fig_diode_charge_storage(area=1e-4, n_points=40):
    """(a) Injected excess-electron profiles in the P side at three forward
    biases (linear scale: stored charge is the area). (b) Diffusion vs
    junction capacitance versus forward current. Analytical model."""
    from . import diode
    from .constants import q, eps_si
    notes_style()
    p = _DIODE
    L = np.sqrt(p["Dn"] * p["tau_n"])
    n_p0 = diode.ni_T(300.0) ** 2 / p["NA"]
    x = np.linspace(0, 4 * L, n_points)
    fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.8), gridspec_kw=dict(hspace=0.42))
    a = axs[0]
    cols = [C["blue"], C["purple"], C["red"]]
    for V, col in zip((0.60, 0.62, 0.64), cols):
        prof = diode.excess_minority_profile(x, V, n_p0, L)
        a.plot(x * 1e4, prof / 1e14, color=col, label="V = %.2f V" % V)
        a.fill_between(x * 1e4, 0, prof / 1e14, color=col, alpha=0.08)
    a.axvline(L * 1e4, color=C["gray"], ls="--", lw=0.8)
    a.text(L * 1e4 + 3, 2.9, "Ln = %.0f µm" % (L * 1e4), fontsize=6.8, color=C["gray"])
    a.set_xlabel("distance into P side from depletion edge (µm)")
    a.set_ylabel("excess electrons (1e14 cm⁻³)")
    a.legend(frameon=False, fontsize=7, loc="upper right")
    a.set_title("(a) Stored charge = shaded area; +20 mV multiplies every point by e^(20/25.9) = 2.2", fontsize=7.4)
    a = axs[1]
    J0 = diode.diffusion_saturation_current_density(p["NA"], p["ND"], p["Dn"], p["Dp"], p["tau_n"], p["tau_p"])
    I = np.logspace(-9, -2, 60)
    Vj = thermal_voltage_local() * np.log1p(I / (area * J0))
    Cd = diode.diffusion_capacitance(I, p["tau_n"])
    Cj = area * eps_si / diode.depletion_width(p["NA"], p["ND"], Vj)
    a.loglog(I, Cd * 1e12, color=C["red"], label="diffusion C = τ I / Vt")
    a.loglog(I, Cj * 1e12, color=C["blue"], label="junction (depletion) C = εA / W")
    from matplotlib.ticker import NullLocator
    a.xaxis.set_major_formatter(LOG_FORMATTER); a.xaxis.set_minor_locator(NullLocator())
    a.yaxis.set_major_formatter(LOG_FORMATTER); a.yaxis.set_minor_locator(NullLocator())
    a.set_xlabel("forward current (A)"); a.set_ylabel("capacitance (pF)")
    a.legend(frameon=False, fontsize=7, loc="upper left")
    a.set_title("(b) N+P, NA = 1e16, τn = 1 µs, 100×100 µm² (charge-control model, 300 K)", fontsize=7.4)
    return fig


def thermal_voltage_local(T=300.0):
    from .constants import thermal_voltage
    return thermal_voltage(T)


def fig_diode_reverse_recovery(taus=(1e-6, 1e-7), R=2000.0, V_F=20.7, V_R=20.0):
    """Reverse-recovery transient of a one-sided N+P long diode switched from
    ~10 mA forward to ~10 mA reverse, for two minority-carrier lifetimes
    (finite-difference diffusion model, diode.simulate_reverse_recovery)."""
    from . import diode
    notes_style()
    p = _DIODE
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    cols = [C["blue"], C["red"]]
    for tau, col in zip(taus, cols):
        r = diode.simulate_reverse_recovery(tau, p["Dn"], p["NA"], 1e-4, R, V_F, V_R, t_end_over_tau=1.25e-6 / tau)
        st = max(1, len(r["t"]) // 400)
        t = np.concatenate([[-0.15e-6, 0.0], r["t"][1::st]])
        I = np.concatenate([[r["I_F"], r["I_F"]], r["I"][1::st]])
        lab = "τn = %g µs (t_s = %.0f ns)" % (tau * 1e6, r["t_s"] * 1e9)
        ax.plot(t * 1e6, I * 1e3, color=col, label=lab, drawstyle="default")
        ax.axvline(r["t_s"] * 1e6, color=col, ls=":", lw=0.8)
    ax.axhline(0, color="#cbd5e0", lw=0.6)
    ax.set_xlim(-0.15, 1.2); ax.set_xlabel("time after switching (µs)"); ax.set_ylabel("diode current (mA)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.text(0.24, -10.9, "← storage phase: current set by the external circuit", fontsize=6.6, color=C["gray"])
    ax.text(0.55, -4.0, "decay phase: junction\nnow reverse-biased", fontsize=6.6, color=C["gray"])
    ax.set_title("N+P long diode, NA = 1e16, IF ≈ IR ≈ 10 mA; 1-D diffusion model, junction C neglected", fontsize=7.2)
    return fig


# ============================================================ MOS capacitor (2026-09-29)
def fig_moscap_band_and_wdep(N_A=1e17, tox_nm=5.0, n_points=61):
    """(a) Band diagram of an n+ poly / SiO2 / p-Si capacitor at V_G = 0, with
    the flat-band bands for comparison (oxide drawn 3x thicker than scale).
    (b) Depletion width and inversion charge versus gate voltage from the
    full (Kingston-Neustadter) charge relation, V_FB = -(Eg/2 + phi_F)."""
    from . import electrostatics as es, mosfet
    from .constants import q, eps_si
    notes_style()
    Cox = mosfet.cox_from_tox(tox_nm * 1e-7)
    phiF = mosfet.phi_F(N_A)
    Eg = 1.12
    VFB = -(Eg / 2 + phiF)
    VT = mosfet.vt_uniform(N_A, Cox, VFB)
    fig, axs = plt.subplots(2, 1, figsize=(5.2, 6.4), gridspec_kw=dict(hspace=0.42))
    # ---- (a) band diagram at VG = 0
    a = axs[0]
    psis = es.solve_psi_s(np.array([0.0 - VFB]), N_A, Cox)[0]
    W = np.sqrt(2 * eps_si * psis / (q * N_A))
    Ecb = Eg / 2 + phiF
    xs = np.linspace(0, 160e-7, 45)
    psi = np.where(xs < W, psis * (1 - xs / W) ** 2, 0.0)
    xs_nm = xs * 1e7
    a.plot(xs_nm, Ecb - psi, color=C["blue"], label="Ec")
    a.plot(xs_nm, Ecb - Eg - psi, color=C["orange"], label="Ev")
    a.plot(xs_nm, phiF - psi, color=C["gray"], ls="-.", lw=0.9, label="Ei")
    a.plot(xs_nm, 0 * xs_nm, color=C["dark"], ls="--", lw=1.0, label="EF")
    a.plot(xs_nm, 0 * xs_nm + Ecb, color=C["blue"], ls=":", lw=0.8)
    a.plot(xs_nm, 0 * xs_nm + Ecb - Eg, color=C["orange"], ls=":", lw=0.8)
    s = 3.0                                  # oxide exaggeration factor
    xo = -tox_nm * s
    Ecs = Ecb - psis
    Vox = 0.0 - VFB - psis
    a.plot([xo, 0], [Ecs + 3.1 - Vox, Ecs + 3.1], color=C["green"], lw=1.4)
    a.plot([xo, xo], [Ecs + 3.1 - Vox, 0.0], color=C["green"], lw=0.8)
    a.plot([0, 0], [Ecs + 3.1, Ecs], color=C["green"], lw=0.8)
    a.text(xo + 0.5, Ecs + 3.1 + 0.1, "SiO₂ Ec (tilted by Vox)", fontsize=6.4, color=C["green"])
    xg = np.array([xo - 30, xo])
    a.plot(xg, [0.0, 0.0], color=C["blue"]); a.plot(xg, [-Eg, -Eg], color=C["orange"])
    a.text(xo - 29, 0.15, "n+ poly gate\n(flat, Ec ≈ EF)", fontsize=6.4, color=C["dark"])
    a.text(60, Ecb + 0.12, "flat-band position (dotted)", fontsize=6.4, color=C["gray"])
    a.annotate("", xy=(3, Ecb - psis), xytext=(3, Ecb), arrowprops=dict(arrowstyle="<->", color=C["red"], lw=0.8))
    a.text(5, Ecb - psis / 2, "qψs = %.2f eV" % psis, fontsize=6.6, color=C["red"])
    a.axvline(W * 1e7, color=C["gray"], ls="--", lw=0.6)
    a.text(W * 1e7 + 2, -1.35, "W = %.0f nm" % (W * 1e7), fontsize=6.4, color=C["gray"])
    a.set_xlim(xo - 30, 160); a.set_ylim(-1.5, 4.5)
    a.set_xlabel("depth into silicon (nm); oxide drawn %gx thicker than scale" % s)
    a.set_ylabel("Energy (eV), EF = 0")
    a.legend(frameon=False, fontsize=6.4, loc="upper right", ncol=4)
    a.set_title("(a) VG = 0: n+ poly / %g nm SiO₂ / p-Si, NA = %.0e cm⁻³ (VFB = %.2f V)" % (tox_nm, N_A, VFB), fontsize=7.4)
    # ---- (b) W and Qinv vs VG
    a = axs[1]
    VG = np.linspace(VFB - 0.6, VT + 1.4, n_points)
    ps = es.solve_psi_s(VG - VFB, N_A, Cox)
    Qt = es.semiconductor_charge_p(ps, N_A)
    Qm = es.semiconductor_charge_p(ps, N_A, majority_only=True)
    Wd = np.where(ps > 0, np.sqrt(2 * eps_si * np.clip(ps, 0, None) / (q * N_A)), 0.0)
    Qinv = np.where(ps > 0, -(Qt - Qm), 0.0)
    a.plot(VG, Wd * 1e7, color=C["blue"], label="depletion width W (nm)")
    Wmax = es.max_depletion_width(N_A) * 1e7
    a.axhline(Wmax, color=C["blue"], ls=":", lw=0.8)
    a.text(VFB - 0.55, Wmax + 3, "Wdmax = √(4εφF/qNA) = %.0f nm" % Wmax, fontsize=6.4, color=C["blue"])
    a.set_ylabel("W (nm)", color=C["blue"]); a.set_ylim(0, 1.45 * Wmax)
    b = a.twinx()
    b.spines["right"].set_visible(True)
    b.plot(VG, Qinv * 1e7, color=C["red"], label="inversion charge")
    b.set_ylabel("|Qinv| (1e-7 C/cm²)", color=C["red"])
    for v, lab in ((VFB, "VFB"), (VT, "VT")):
        a.axvline(v, color=C["gray"], ls="--", lw=0.7)
        a.text(v + 0.02, 0.05 * Wmax, lab, fontsize=6.8, color=C["gray"])
    a.set_xlabel("gate voltage VG (V)")
    a.set_title("(b) Past VT the extra gate charge goes into the inversion layer; W saturates", fontsize=7.4)
    return fig


FIGURES = {
    "diode_forward_iv_ideality": fig_diode_forward_iv,
    "diode_reverse_leakage_arrhenius": fig_diode_leakage_arrhenius,
    "diode_stored_charge_and_diffusion_capacitance": fig_diode_charge_storage,
    "diode_reverse_recovery_transient": fig_diode_reverse_recovery,
    "moscap_band_diagram_and_depletion_width": fig_moscap_band_and_wdep,
}


def build_all(outdir, minify=True, png=False):
    """Generate every figure in this module into outdir; returns {name: path}."""
    return {name: save_figure(fn(), os.path.join(outdir, name), png=png, minify=minify)
            for name, fn in FIGURES.items()}
