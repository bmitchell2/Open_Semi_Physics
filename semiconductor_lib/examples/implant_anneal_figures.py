"""Figures for Semiconductor Notes pages: TED, EOR defects, implanted nitrogen.
All curves are 1-D continuum models from semiconductor_lib.implant_anneal
(illustrative, not measured data)."""
import numpy as np
import matplotlib.pyplot as plt
from semiconductor_lib import implant_anneal as ia
from semiconductor_lib.plotting import apply_style, full_minify_pipeline

apply_style()

# ---- 1. TED boron profiles + dissolution-time Arrhenius -----------------
x = np.linspace(0, 500, 1001)
c0 = ia.gaussian_profile(x, 1e14, 30, 10)
T, t = 800.0, 1800.0
eq = ia.diffuse_ted(x, c0, T, t)
ted = ia.diffuse_ted(x, c0, T, t, S0=500, tau=300)
tedc = ia.diffuse_ted(x, c0, T, t, S0=500, tau=300, c_cluster=1e19)
def xj(c, lvl=1e17): return x[np.where(c > lvl)[0][-1]]
D = ia.boron_intrinsic_diffusivity(T)
print("sqrt(Dt) eq nm", np.sqrt(D*t)*1e7, " with TED", np.sqrt(ia.ted_budget(D,t,500,300))*1e7)
print("xj@1e17: asimp %.1f eq %.1f ted %.1f tedc %.1f" % (xj(c0), xj(eq), xj(ted), xj(tedc)))
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
a.semilogy(x, c0, "k--", lw=1, label="as implanted")
a.semilogy(x, eq, label="equilibrium D*, 800 °C 30 min")
a.semilogy(x, ted, label="TED, no clustering")
a.semilogy(x, tedc, label="TED, peak >1e19 immobile")
a.set_ylim(1e15, 1e20); a.set_xlim(0, 200)
a.set_xlabel("Depth (nm)"); a.set_ylabel("Boron (cm$^{-3}$)")
a.legend(fontsize=6.5, loc="upper right"); a.set_title("(a) Tail moves, peak stays", fontsize=9)
Ts = np.linspace(650, 1100, 50)
b.semilogy(Ts, ia.relative_dissolution_time(Ts, 800), color="C3")
b.axhline(1, color="0.6", lw=0.6)
b.set_xlabel("Anneal temperature (°C)"); b.set_ylabel("TED duration / duration at 800 °C")
b.set_title("(b) Damage-dissolution time, Ea = 3.8 eV", fontsize=9)
fig.tight_layout(); fig.savefig("ted_raw.svg")
full_minify_pipeline("ted_raw.svg", "ted_boron_profiles.svg")
print("tau ratios", ia.relative_dissolution_time([700, 900, 1000, 1050], 800))

# ---- 2. EOR placement schematic ----------------------------------------
xe = np.linspace(0, 80, 801)
dmg = ia.gaussian_profile(xe, 1e15, 15, 9) * 70   # displaced Si atoms
thr = 1.15e22
zac = ia.eor_depth(xe, dmg, thr)
print("a/c depth nm", zac)
fig, a = plt.subplots(figsize=(4.6, 3.0))
a.semilogy(xe, dmg, color="C0", label="displaced-atom density (Ge PAI)")
a.axhline(thr, color="k", ls="--", lw=0.8, label="amorphization threshold")
a.axvspan(0, zac, color="C0", alpha=0.08)
a.axvspan(zac, zac + 8, color="C3", alpha=0.25, label="EOR band (interstitial excess)")
bprof = ia.gaussian_profile(xe, 1e15, 8, 5)
a.semilogy(xe, bprof, color="C2", label="boron (shallow implant)")
a.text(zac / 2, 3e22, "amorphous\n(regrows by SPE)", ha="center", fontsize=7)
a.set_ylim(1e18, 2e23); a.set_xlim(0, 70)
a.set_xlabel("Depth (nm)"); a.set_ylabel("cm$^{-3}$")
a.legend(fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig("eor_raw.svg")
full_minify_pipeline("eor_raw.svg", "eor_placement.svg")

# ---- 3. Implanted nitrogen trap-limited release ------------------------
xn = np.linspace(0, 250, 251)
n0 = ia.gaussian_profile(xn, 1e14, 55, 20)
mins = [12, 30, 120]
prof, q = ia.nitrogen_trap_limited(xn, n0, 2e-4, 1e-11, [m * 60 for m in mins])
tt = np.linspace(1, 240, 60) * 60
_, qf = ia.nitrogen_trap_limited(xn, n0, 2e-4, 1e-11, tt)
_, qs = ia.nitrogen_trap_limited(xn, n0, 2e-4, 1e-11, tt, q_max=4e13)
print("interface fraction at 12/30/120 min", q / 1e14)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
a.plot(xn, n0, "k--", lw=1, label="as implanted")
for m, p in zip(mins, prof):
    a.plot(xn, p, label=f"{m} min")
a.set_xlabel("Depth below Si/SiO$_2$ interface (nm)"); a.set_ylabel("N in Si (cm$^{-3}$)")
a.set_xlim(0, 150); a.legend(fontsize=7)
a.set_title("(a) Peak falls, profile does not broaden", fontsize=9)
b.plot(tt / 60, 100 * qf / 1e14, label="unlimited interface capacity")
b.plot(tt / 60, 100 * qs / 1e14, label="interface saturates at 4e13 cm$^{-2}$")
b.set_xlabel("Anneal time (min)"); b.set_ylabel("Dose trapped at interface (%)")
b.set_ylim(0, 100); b.legend(fontsize=7, loc="lower right")
b.set_title("(b) Interface uptake", fontsize=9)
fig.tight_layout(); fig.savefig("n_raw.svg")
full_minify_pipeline("n_raw.svg", "nitrogen_trap_limited.svg")
