"""Figures for "Geometry Dependence of MOS C-V (Area-Perimeter Scaling)".

Fig 1  Illustrative (synthetic) model: ideal low-frequency C-V per unit area
       plus a perimeter term made of a bias-independent fringe/overlap
       component and a narrow edge strip whose local threshold is lower
       (STI-corner early inversion). (a) C/(Cox*A) for a 1x1 um and a
       100x100 um capacitor. (b) Area/perimeter extraction: C/A vs P/A at
       three biases; intercept = C_a, slope = C_p(V).
Fig 2  Inverted channel of a capacitor with source/drain contacts treated as
       a distributed RC line: apparent C/Cox(WL) vs frequency for several L.

Not experimental data. Parameters: p-Si N_A = 1e17 cm^-3, t_ox = 5 nm,
V_FB = 0, edge strip 50 nm wide with dV_T = -0.3 V, fringe/overlap
0.1 fF/um, mu_eff = 300 cm^2/Vs, V_G - V_T = 0.5 V.
"""
import numpy as np
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib.pyplot as plt

from semiconductor_lib.constants import eps_ox
from semiconductor_lib.electrostatics import (solve_psi_s, differential_Cs,
                                              series_C, threshold_voltage)
from semiconductor_lib.geometry_cv import (area_perimeter_extract,
                                           channel_rc_capacitance, channel_rc_tau)

NA, tox = 1e17, 5e-7                    # cm^-3, cm
Cox = eps_ox / tox                      # F/cm^2
VT = threshold_voltage(NA, Cox)


def lf_frac(Vg):
    psi = solve_psi_s(Vg, NA, Cox)
    return series_C(Cox, differential_Cs(psi, NA)) / Cox


Vg = np.linspace(-2.0, 3.0, 121)
c_main = lf_frac(Vg)
c_edge = lf_frac(Vg + 0.3)              # edge strip: V_T lowered by 0.3 V
w_e = 0.05                              # um
cf_over_cox = 0.1 / (Cox * 1e-8 * 1e15)  # (fF/um)/(fF/um^2) -> um

def c_per_area(PA):
    return c_main + PA * (cf_over_cox + w_e * (c_edge - c_main))

fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
for side, col in [(100.0, COLORS["blue"]), (1.0, COLORS["red"])]:
    ax[0].plot(Vg, c_per_area(4.0 / side), color=col, lw=1.6,
               label=f"{side:g} x {side:g} µm (P/A = {4/side:g} µm⁻¹)")
ax[0].axvline(VT, color=COLORS["gray"], ls=":", lw=1)
ax[0].text(VT + 0.05, 1.07, "V_T (area)", color=COLORS["gray"], fontsize=8)
ax[0].set_xlabel("Gate voltage V_G (V)")
ax[0].set_ylabel("C / (C_ox · A)")
ax[0].set_ylim(0, 1.12)
ax[0].legend(fontsize=7.5, loc=(0.02, 0.04))
ax[0].set_title("(a) Same stack, different size", fontsize=10)

sides = np.array([1, 2, 5, 10, 20, 50, 100.0])
A, P = sides ** 2, 4 * sides
cox_ff = Cox * 1e-8 * 1e15              # fF/um^2
C_all = np.array([c_per_area(p / a) * cox_ff * a for a, p in zip(A, P)])  # fF
Ca, Cp = area_perimeter_extract(A, P, C_all)
ia = np.argmin(abs(Vg + 1.5)); idp = np.argmin(abs(Vg - 0.85)); iinv = np.argmin(abs(Vg - 2.5))
xs = np.linspace(0, 4.2, 10)
for i, col, lab, mk, ls in [(ia, COLORS["blue"], "accumulation (−1.5 V)", "o", "-"),
                    (idp, COLORS["orange"], "depletion (0.85 V)", "o", "-"),
                    (iinv, COLORS["green"], "inversion (2.5 V)", "x", "--")]:
    ax[1].plot(P / A, C_all[:, i] / A, mk, color=col, ms=5)
    ax[1].plot(xs, Ca[i] + Cp[i] * xs, color=col, lw=1.2, ls=ls,
               label=f"{lab}: C_p = {Cp[i]*1e3:.0f} aF/µm")
ax[1].set_xlabel("P / A (µm⁻¹)")
ax[1].set_ylabel("C / A (fF/µm²)")
ax[1].legend(fontsize=7.5, loc="center right")
ax[1].set_title("(b) Area–perimeter extraction", fontsize=10)
fig.tight_layout()
fig.savefig("geom_cv_fig1.svg")

# --- Fig 2: channel RC roll-off ----------------------------------------
Cox_si = Cox * 1e4                      # F/m^2
mu, Vov = 0.03, 0.5
f = np.logspace(3, 10, 120)
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for L, col in [(1e-6, COLORS["green"]), (10e-6, COLORS["blue"]),
               (30e-6, COLORS["purple"]), (100e-6, COLORS["red"])]:
    c = channel_rc_capacitance(f, L, 1e-5, Cox_si, mu, Vov) / (Cox_si * 1e-5 * L)
    fc = 1 / (2 * np.pi * channel_rc_tau(L, mu, Vov))
    ax.semilogx(f, c, color=col, lw=1.6, label=f"L = {L*1e6:g} µm (1/2πτ = " + (f"{fc/1e9:.3g} GHz)" if fc > 1e9 else f"{fc/1e6:.3g} MHz)"))
for fm in [1e5, 1e6]:
    ax.axvline(fm, color=COLORS["gray"], ls=":", lw=1)
ax.text(1.1e6, 0.05, "1 MHz", color=COLORS["gray"], fontsize=8)
ax.text(1.1e5, 0.05, "100 kHz", color=COLORS["gray"], fontsize=8)
ax.set_xlabel("Measurement frequency (Hz)")
ax.set_ylabel("C_inv / (C_ox W L)")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, loc="center left")
fig.tight_layout()
fig.savefig("geom_cv_fig2.svg")

for n in ["geom_cv_fig1", "geom_cv_fig2"]:
    full_minify_pipeline(f"{n}.svg", f"{n}_min.svg")

# printed checks
print("VT", VT, "Cox fF/um2", cox_ff, "cf/cox um", cf_over_cox)
print("1x1 accumulation excess", c_per_area(4.0)[0] - 1)
print("Cp aF/um acc,dep,inv", Cp[ia]*1e3, Cp[idp]*1e3, Cp[iinv]*1e3, "Ca/Cox", Ca[[ia, idp, iinv]] / cox_ff)
for L in [100e-6]:
    c = channel_rc_capacitance(np.array([1e5, 1e6]), L, 1e-5, Cox_si, mu, Vov) / (Cox_si * 1e-5 * L)
    print("L=100um C/CoxWL at 100k,1M", c)
