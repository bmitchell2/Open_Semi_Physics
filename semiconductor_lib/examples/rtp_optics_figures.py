"""Figures for 'Optical Absorption in Silicon' and 'Rapid Thermal Processing' pages.
Uses semiconductor_lib.optics; regenerate with: python rtp_optics_figures.py"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from semiconductor_lib import optics as o
from semiconductor_lib.plotting import apply_style, full_minify_pipeline
apply_style()

# Fig 1: lamp vs wafer spectra, with Si edge and pyrometer bands
lam = np.linspace(0.3, 6.0, 300)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
L = o.planck_radiance(lam, 3000.0)
ax.plot(lam, L / L.max(), color="k", lw=2, label="Filament, 3000 K (×1/peak)")
ref = o.planck_radiance(lam, 1473.15).max()
for TC, c in [(800, "#1f77b4"), (1000, "#2ca02c"), (1200, "#d62728")]:
    ax.plot(lam, o.planck_radiance(lam, TC + 273.15) / ref, color=c, label=f"Wafer, {TC} °C (×1/peak at 1200 °C)")
ax.axvline(1.1, color="gray", ls="--", lw=1)
ax.text(1.13, 0.93, "Si edge, 300 K\n(≈1.1 µm)", fontsize=8, color="gray")
ax.axvspan(2.65, 2.75, color="orange", alpha=0.3)
ax.axvspan(4.3, 4.7, color="purple", alpha=0.2)
ax.text(2.8, 0.55, "2.7 µm\n(OH-quartz\nblocks lamp)", fontsize=8)
ax.text(4.75, 0.55, "4.3–4.7 µm", fontsize=8)
ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Normalized spectral radiance")
ax.set_xlim(0.3, 6); ax.set_ylim(0, 1.05); ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.82))
ax.set_title("Ideal blackbody spectra: lamp filament vs wafer", fontsize=10)
fig.tight_layout(); fig.savefig("raw1.svg"); plt.close(fig)

# Fig 2: alpha vs wavelength, 300 K data + high-T model
fig, ax = plt.subplots(figsize=(6.4, 4.0))
l300 = np.linspace(0.30, 1.20, 120)
ax.semilogy(l300, o.si_alpha_300k(l300), "k", lw=2, label="300 K (Franta 2017 data)")
lhot = np.linspace(1.1, 2.6, 150)
for TC, c in [(700, "#1f77b4"), (900, "#2ca02c"), (1100, "#d62728")]:
    ax.semilogy(lhot, o.timans_alpha(lhot, TC), color=c, label=f"{TC} °C (Timans model)")
    ax.semilogy(lhot, o.timans_alpha_fc(lhot, TC), color=c, ls=":", lw=1)
for lm, mk in [(1.31, "o"), (1.54, "s"), (2.3, "^")]:
    _, (t0, t1) = o.TIMANS_QUARTIC[lm]
    for TC, c in [(700, "#1f77b4"), (900, "#2ca02c"), (1100, "#d62728")]:
        if t0 <= TC <= t1:
            ax.semilogy(lm, o.timans_measured_fit(lm, TC), mk, color=c, ms=6, mfc="none")
ax.set_xlabel("Wavelength (µm)"); ax.set_ylabel("Absorption coefficient α (cm⁻¹)")
ax.set_ylim(1e-2, 1e7); ax.set_xlim(0.3, 2.6)
ax.text(1.65, 2.5, "dotted: free-carrier term only\nmarkers: fits to measured data", fontsize=8)
sec = ax.secondary_yaxis("right", functions=(lambda a: 1e4 / np.clip(a, 1e-12, None), lambda d: 1e4 / np.clip(d, 1e-12, None)))
sec.set_ylabel("1/e depth (µm)")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Silicon absorption: room temperature vs RTP temperatures", fontsize=10)
fig.tight_layout(); fig.savefig("raw2.svg"); plt.close(fig)

# Fig 3: depth vs temperature
fig, ax = plt.subplots(figsize=(6.0, 3.8))
T = np.linspace(700, 1200, 60)
for lm, c in [(1.1, "#9467bd"), (1.31, "#1f77b4"), (1.54, "#2ca02c"), (2.3, "#d62728")]:
    ax.semilogy(T, o.absorption_depth_um(o.timans_alpha(lm, T)), color=c, label=f"{lm} µm")
ax.axhline(775, color="gray", ls="--", lw=1); ax.text(705, 830, "300 mm wafer thickness (775 µm)", fontsize=8, color="gray")
ax.set_xlabel("Wafer temperature (°C)"); ax.set_ylabel("1/e absorption depth (µm)")
ax.set_ylim(1, 2000); ax.legend(fontsize=8, title="Wavelength", title_fontsize=8)
ax.set_title("Lightly doped Si: absorption depth vs temperature (Timans model)", fontsize=10)
fig.tight_layout(); fig.savefig("raw3.svg"); plt.close(fig)

for i in (1, 2, 3):
    full_minify_pipeline(f"raw{i}.svg", f"fig{i}.svg")
