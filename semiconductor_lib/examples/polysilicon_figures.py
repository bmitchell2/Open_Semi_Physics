"""Figures for the polysilicon notes: SPC kinetics, Seto resistivity and TCR vs doping. Usage: python polysilicon_figures.py [outdir]"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
from semiconductor_lib import polysilicon as P

apply_style()
c = list(COLORS.values())
import os, sys
out = (sys.argv[1] if len(sys.argv) > 1 else ".") + os.sep

# 1. SPC grain size and crystallization time vs anneal temperature
T = np.linspace(550, 1050, 60)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for Ean, col in ((3.9, c[0]), (4.9, c[1])):
    ax[0].plot(T, P.spc_grain_size_ratio(T, Ea_nuc=Ean), color=col, label=f"E$_N$ = {Ean} eV")
    ax[1].semilogy(T, 10.0 * P.spc_time_ratio(T, Ea_nuc=Ean) * 3600, color=col, label=f"E$_N$ = {Ean} eV")
ax[0].axhline(1, color="0.6", lw=0.8, ls=":")
ax[0].set_xlabel("Crystallization temperature (°C)"); ax[0].set_ylabel("Grain size / grain size at 600 °C")
ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("Final grain size", fontsize=9)
ax[1].axhline(1, color="0.6", lw=0.8, ls=":"); ax[1].text(560, 1.6, "1 s (spike RTA)", fontsize=7, color="0.4")
ax[1].set_xlabel("Crystallization temperature (°C)"); ax[1].set_ylabel("Time to crystallize (s)")
ax[1].set_title("Time (normalized to 10 h at 600 °C)", fontsize=9); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(out + "spc_raw.svg"); plt.close(fig)

# 2. Resistivity vs doping for several grain sizes (Seto model)
N = np.logspace(16.5, 20.3, 70)
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for L, col in ((20e-7, c[0]), (50e-7, c[1]), (200e-7, c[2])):
    ax.loglog(N, [P.poly_resistivity(n, L) for n in N], color=col, label=f"poly, L = {L*1e7:.0f} nm")
ax.loglog(N, P.single_crystal_resistivity(N), color="k", ls="--", lw=1, label="single crystal")
ax.set_xlabel("Boron doping (cm$^{-3}$)"); ax.set_ylabel("Resistivity (Ω·cm)")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(out + "rho_raw.svg"); plt.close(fig)

# 3. TCR vs doping
N = np.logspace(17.5, 20.3, 60)
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for L, col in ((20e-7, c[0]), (50e-7, c[1]), (200e-7, c[2])):
    tcr = [P.poly_tcr_ppm(n, L) for n in N]
    ax.semilogx(N, tcr, color=col, label=f"L = {L*1e7:.0f} nm")
    Nz = P.zero_tcr_doping(L); ax.plot([Nz], [0], "o", color=col, ms=4)
ax.axhline(0, color="0.5", lw=0.8)
ax.set_yscale("symlog", linthresh=1000, linscale=0.6)
ax.set_yticks([-1e4,-1e3,0,1e3]); ax.set_yticklabels(["−10⁴","−10³","0","+10³"]); ax.set_ylim(-8e4,4e3)
ax.set_xlabel("Boron doping (cm$^{-3}$)"); ax.set_ylabel("TCR at 300 K (ppm/K)")
ax.legend(frameon=False, fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(out + "tcr_raw.svg"); plt.close(fig)

for name in ("spc", "rho", "tcr"):
    a, b = full_minify_pipeline(out + name + "_raw.svg", out + name + ".svg")
    print(name, a, "->", b)
