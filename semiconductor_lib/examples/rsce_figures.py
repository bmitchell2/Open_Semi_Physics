"""
Figure for Semiconductor Notes, Reverse Short-Channel Effect page:
  rsce_vs_pocket_dose.svg - low-VDS constant-current V_T versus gate length
                            for several pocket doses, and its gate-length
                            (CD) sensitivity dVT/dL.
Quasi-2-D numerical model (semiconductor_lib.band_diagrams), compared with the
lateral-average-doping (BSIM4 LPE0-type) estimate. Model results only.

Verified results (tox = 2 nm, n+ poly, N_ch = 1e18, 8 nm pocket sigma):
  long-channel V_T 0.263 V (matches vt_uniform at the same criterion, <1 mV);
  V_T(40 nm) = 0.076 / 0.239 / 0.356 / 0.471 V for pocket peaks
  0 / 2.5e18 / 5e18 / 8e18; dVT/dL at 40 nm ~ +8..11 / +1.2..1.9 /
  -2.2..-2.5 / -5..-6 mV/nm; LPE0 = 100 nm; the lateral-average estimate
  gives 0.40 V at 100 nm versus 0.30 V from the current-weighted model.
"""
import sys
import numpy as np
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
from semiconductor_lib.band_diagrams import (
    threshold_voltage_q2d_current, gaussian_pocket_profile, pocket_lpe0,
    phi_bulk, EG_SI)
from semiconductor_lib.mosfet import vt_uniform
from semiconductor_lib.constants import eps_ox

out = sys.argv[1] if len(sys.argv) > 1 else "."
tox, Nch, sig, ncrit, VDS = 2e-7, 1e18, 8e-7, 2e18, 0.05
Lnm = np.array([30, 33, 36, 40, 45, 50, 60, 70, 85, 100, 130, 170, 250,
                400, 700, 1200, 2000])
peaks = [0.0, 2.5e18, 5e18, 8e18]
cols = [COLORS['red'], COLORS['green'], COLORS['blue'], COLORS['purple']]
res = {}
for pk in peaks:
    res[pk] = np.array([threshold_voltage_q2d_current(
        L * 1e-7, VDS, gaussian_pocket_profile(L * 1e-7, Nch, pk, sig), tox,
        ncrit, npts=3001 if L > 500 else (1601 if L > 200 else 801))
        for L in Lnm])

# lateral-average (BSIM4 LPE0-type) estimate for the 5e18 pocket, no SCE,
# n+ poly gate, same surface-density criterion (n_s = ncrit)
Cox = eps_ox / tox
LPE0 = pocket_lpe0(Nch, 5e18, sig)
Lf = np.geomspace(30, 2000, 50)
Nav = Nch * (1 + LPE0 / (Lf * 1e-7))
vt_av = np.array([vt_uniform(N, Cox, V_FB=-(EG_SI / 2) + phi_bulk(N),
                             n_phit=np.log(ncrit / N)) for N in Nav])

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for pk, c in zip(peaks, cols):
    lab = "no pocket" if pk == 0 else f"pocket peak {pk / 1e18:g}×10¹⁸ cm⁻³"
    axs[0].plot(Lnm, res[pk], '-o', ms=2.5, color=c, label=lab)
    m = Lnm <= 250
    Lm = np.sqrt(Lnm[m][1:] * Lnm[m][:-1])
    axs[1].plot(Lm, np.diff(res[pk][m]) / np.diff(Lnm[m]) * 1e3, '-o',
                ms=2.5, color=c, label=lab)
axs[0].plot(Lf, vt_av, ':', color=COLORS['blue'], lw=1.3,
            label="lateral-average estimate, 5×10¹⁸ pocket")
axs[0].set_ylim(-0.1, 0.75)
axs[1].axhline(0, color='0.6', lw=0.8)
for a, ticks in [(axs[0], [30, 100, 300, 1000, 2000]),
                 (axs[1], [30, 50, 100, 200])]:
    a.set_xscale('log')
    a.xaxis.set_major_locator(FixedLocator(ticks))
    a.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    a.xaxis.set_minor_locator(NullLocator())
    a.set_xlabel('Gate length L (nm)')
    a.tick_params(labelsize=8)
axs[0].set_ylabel('Constant-current VT at VDS = 0.05 V (V)')
axs[0].set_title('Reverse short-channel effect versus pocket dose', fontsize=10)
axs[1].set_ylabel('dVT/dL (mV/nm)')
axs[1].set_title('Gate-length (CD) sensitivity of VT', fontsize=10)
axs[0].legend(fontsize=6.5, frameon=False)
axs[1].legend(fontsize=6.5, frameon=False)
fig.tight_layout()
fig.savefig(f"{out}/_rsce.svg")
full_minify_pipeline(f"{out}/_rsce.svg", f"{out}/rsce_vs_pocket_dose.svg")

print(f"LPE0 = {LPE0 * 1e7:.0f} nm")
for pk in peaks:
    print(f"pk={pk:.1e}: " + " ".join(f"{L}:{v:.3f}" for L, v in zip(Lnm, res[pk])))
    d = np.diff(res[pk]) / np.diff(Lnm) * 1e3
    print("   dVT/dL near 40 nm (36-40, 40-45):", np.round(d[2:4], 2), "mV/nm")
for L in (40, 100, 1000):
    i = np.argmin(abs(Lf - L)); print(f"avg model L~{Lf[i]:.0f}: {vt_av[i]:.3f}")
