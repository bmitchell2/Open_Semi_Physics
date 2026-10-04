"""Figures for the Velocity Saturation, Inversion-Layer Mobility, Optical
Lithography, EUV and Plasma Physics notes. All model calculations from
semiconductor_lib; BSIM points from ngspice_vsat_crosscheck.py.
Usage: FIG_OUT=<dir> python high_field_litho_plasma_figures.py"""
import os, numpy as np
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib.pyplot as plt
from semiconductor_lib import high_field as hf, lithography as li, plasma as pl
from semiconductor_lib.mosfet import cox_from_tox
from semiconductor_lib.constants import eps_si, q
OUT = os.environ.get("FIG_OUT", ".")
C = COLORS


def save(fig, name):
    raw = os.path.join(OUT, name + "_raw.svg"); fin = os.path.join(OUT, name + ".svg")
    fig.savefig(raw, bbox_inches="tight"); plt.close(fig)
    full_minify_pipeline(raw, fin); os.remove(raw)
    print(name, os.path.getsize(fin))


# ---------------- F1 velocity saturation
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
E = np.logspace(2, 6, 80)
for car, col in (("electron", C["blue"]), ("hole", C["red"])):
    vm, Ec, beta = hf.canali_params(car)
    a.loglog(E, hf.canali_velocity(E, car), color=col, lw=2, label=f"{car}s")
    a.loglog(E, vm / Ec * E, color=col, ls=":", lw=1)
    a.axhline(vm, color=col, ls="--", lw=0.8)
a.set_ylim(1e5, 3e7); a.set_xlabel("Electric field E (V/cm)"); a.set_ylabel("Drift velocity (cm/s)")
a.set_title("(a) Bulk Si, 300 K (Canali model)", fontsize=10)
a.text(2e2, 1.1e7, "dotted: v = μ₀E; dashed: v_m", fontsize=8)
a.legend(fontsize=8, loc="lower right")
mu, vsat, Vov, Cox = 148.0, 1.38e7, 1.35, cox_from_tox(4e-7)
L = np.logspace(-5.5, -2.5, 80)     # cm
norm = hf.idsat_vsat(Vov, mu, vsat, Cox, 1, L) / hf.idsat_long_channel(Vov, mu, Cox, 1, L)
a2 = 2 * vsat / mu * L
b.semilogx(L * 1e4, norm, color=C["blue"], lw=2, label="n = 1 model (E_sat = v_sat/μ)")
b.semilogx(L * 1e4, 1 / (1 + Vov / a2), color=C["purple"], lw=2, ls="--", label="Hu/BSIM form (E_sat = 2v_sat/μ)")
b.axhline(1, color=C["gray"], lw=1, ls=":", label="square law")
Lb = np.array([0.17, 0.27, 0.42, 0.92, 1.92]); rb = np.array([0.63, 0.72, 0.80, 0.90, 0.95])
b.plot(Lb, rb, "o", color=C["red"], label="ngspice BSIM3 (PTM 180 nm)")
b.set_xlabel("Effective channel length L_eff (µm)"); b.set_ylabel("I_Dsat / I_Dsat,square-law")
b.set_title("(b) Saturation current vs length, V_ov ≈ 1.35 V", fontsize=10)
b.set_ylim(0, 1.1); b.legend(fontsize=7, loc="lower right")
fig.tight_layout(); save(fig, "velocity_saturation")

# ---------------- F2 universal mobility
fig, ax = plt.subplots(figsize=(6.2, 4.2))
Ee = np.logspace(4.7, 6.4, 120)
ax.loglog(Ee / 1e6, hf.universal_mobility(Ee), color=C["blue"], lw=2.5, label="electrons, universal fit")
ax.loglog(Ee / 1e6, hf.universal_mobility(Ee, "hole"), color=C["red"], lw=2.5, label="holes, universal fit")
sel = (Ee > 3e5) & (Ee < 2e6)
from scipy.optimize import least_squares
res = least_squares(lambda p: np.log(hf.matthiessen_inversion(Ee[sel], *np.exp(p))[0]) - np.log(hf.universal_mobility(Ee[sel])), [np.log(400), np.log(800)])
Aph, Asr = np.exp(res.x)
tot, ph, sr = hf.matthiessen_inversion(Ee, Aph, Asr)
ax.loglog(Ee / 1e6, ph, color=C["green"], ls=":", lw=1.3, label=r"phonon $\propto E^{-0.3}$")
ax.loglog(Ee / 1e6, sr, color=C["orange"], ls=":", lw=1.3, label=r"surface roughness $\propto E^{-2}$")
# Coulomb deviation for heavy channel doping (illustrative)
for NA, ls in ((3e17, "-."), (1e18, "--")):
    phiF = 0.0259 * np.log(NA / 1e10)
    Ndep = np.sqrt(4 * eps_si * phiF * NA / q)            # cm^-2 at threshold
    Ninv = 2 * (Ee * eps_si / q - Ndep)
    ok = Ninv > 2e11
    muC = 2500.0 * (Ninv[ok] / 1e12) / (NA / 1e17)
    t, _, _ = hf.matthiessen_inversion(Ee[ok], Aph, Asr, muC)
    ax.loglog(Ee[ok] / 1e6, t, color=C["purple"], ls=ls, lw=1.5, label=f"+ Coulomb, N_A = {NA:.0e} cm⁻³")
ax.set_xlabel("Effective field E_eff (MV/cm)"); ax.set_ylabel("Effective mobility μ_eff (cm²/V·s)")
ax.set_ylim(30, 1500); ax.legend(fontsize=7, loc="lower left")
ax.set_title("Inversion-layer mobility vs effective field (model)", fontsize=10)
fig.tight_layout(); save(fig, "universal_mobility")
print("Aph, Asr =", round(Aph), round(Asr))

# ---------------- F3 diffraction imaging
lam, NA = 193.0, 1.35
cases = [("p = 200 nm, on-axis", 200, 0.0, C["blue"]),
         ("p = 100 nm, on-axis", 100, 0.0, C["red"]),
         ("p = 100 nm, dipole σ ≈ 0.71", 100, lam / 200, C["green"])]
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw=dict(width_ratios=[1, 1.4]))
for i, (lab, p, s, col) in enumerate(cases):
    y = -i
    a.plot([-NA, NA], [y, y], color=C["gray"], lw=6, alpha=0.25, solid_capstyle="butt")
    for m in range(-3, 4):
        x = s + m * lam / p
        inside = abs(x) <= NA
        if abs(x) < 2.6:
            a.plot(x, y, "o", ms=8, color=col if inside else "white", mec=col)
            a.text(x, y + 0.22, str(m), ha="center", fontsize=7)
    a.text(-2.6, y - 0.38, lab, fontsize=7, color=col)
a.axvline(-NA, color=C["gray"], lw=0.8, ls="--"); a.axvline(NA, color=C["gray"], lw=0.8, ls="--")
a.set_xlim(-2.7, 2.7); a.set_ylim(-2.7, 0.6); a.set_yticks([])
a.set_xlabel("Order direction sin θ × n  (pupil edge = ±NA)")
a.set_title("(a) Orders captured (filled) by NA = 1.35", fontsize=10)
x = np.linspace(-150, 150, 301)
for lab, p, s, col in cases:
    if s == 0:
        I = li.aerial_image_grating(x, p, lam, NA)
    else:
        I = 0.5 * (li.aerial_image_grating(x, p, lam, NA, s) + li.aerial_image_grating(x, p, lam, NA, -s))
    b.plot(x, I, color=col, lw=2, label=lab)
b.set_xlabel("Position at wafer (nm)"); b.set_ylabel("Aerial-image intensity (clear field = 1)")
b.set_title("(b) Coherent aerial image, 1:1 lines/spaces", fontsize=10)
b.legend(fontsize=7, loc="upper right"); b.set_ylim(0, 1.6)
fig.tight_layout(); save(fig, "litho_diffraction_imaging")

# ---------------- F4 EUV shot noise
fig, ax = plt.subplots(figsize=(6, 3.8))
D = np.linspace(5, 100, 100)
for lamx, col, lab in ((193, C["blue"], "ArF 193 nm"), (13.5, C["red"], "EUV 13.5 nm")):
    N = li.photons_per_area(D, lamx, 100)
    ax.plot(D, 100 * li.shot_noise(N), color=col, lw=2, label=f"{lab}, incident")
    ax.plot(D, 100 * li.shot_noise(0.2 * N), color=col, lw=1.3, ls="--", label=f"{lab}, 20 % absorbed")
ax.set_xlabel("Dose (mJ/cm²)"); ax.set_ylabel("Photon shot noise in 10 × 10 nm (%)")
ax.set_ylim(0, 12); ax.legend(fontsize=8)
ax.set_title("Poisson photon noise per (10 nm)² pixel (model)", fontsize=10)
fig.tight_layout(); save(fig, "euv_shot_noise")

# ---------------- F5 sheath vs mean free path
fig, ax = plt.subplots(figsize=(6, 3.9))
V = np.logspace(1, 3.3, 100)
for ne, col in ((1e10, C["blue"]), (1e11, C["purple"]), (1e12, C["red"])):
    ax.loglog(V, 10 * pl.child_sheath_thickness(V, 3.0, ne), color=col, lw=2, label=f"sheath, n_e = {ne:.0e} cm⁻³")
for p, ls in ((5, ":"), (30, "--"), (100, "-.")):
    ax.axhline(10 * pl.ion_mean_free_path(p), color=C["gray"], ls=ls, lw=1.2)
    ax.text(12, 10 * pl.ion_mean_free_path(p) * 1.12, f"ion mean free path, {p} mTorr Ar", fontsize=7, color=C["gray"])
ax.set_xlabel("Sheath voltage (V)"); ax.set_ylabel("Length (mm)")
ax.set_title("Child-law sheath thickness vs ion mean free path (T_e = 3 eV)", fontsize=10)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); save(fig, "sheath_vs_mfp")
