"""
Figure for Semiconductor Notes, Reverse Short-Channel Effect page:
  rsce_vt_vs_inverse_L.svg - low-VDS constant-current V_T versus 1/L, with and
  without pockets. On a 1/L axis an edge-localized (pocket) contribution is
  diluted as 1/L and appears as a straight line through the long-channel
  intercept at 1/L = 0, while short-channel roll-off (exponential in L/l)
  appears as a sharp knee at large 1/L.
Quasi-2-D numerical model (semiconductor_lib.band_diagrams), same device as
rsce_figures.py. Model results only, not measured or TCAD data.

Verified results (tox = 2 nm, n+ poly, N_ch = 1e18, 8 nm pocket sigma):
  all curves meet at the long-channel intercept 0.262-0.267 V (1/L = 0.2/um);
  values at 1/L = 24-26/um bracket the 40 nm values of rsce_figures.py
  (0.076 / 0.239 / 0.356 / 0.471 V). Initial slope dVT/d(1/L), 1/L = 0.2-2:
  -2.1 / -0.3 / +3.9 / +17 mV*um for pocket peaks 0 / 2.5e18 / 5e18 / 8e18;
  the lateral-average (LPE0 = 100 nm) estimate for 5e18 gives +16 mV*um,
  about 4x the current-weighted model. No-pocket V_T crosses 0 V near
  1/L = 30/um (L ~ 33 nm). Roll-up peaks (grid 2/um): 0.386 V at
  1/L = 38/um (L ~ 26 nm) for 5e18 and 0.590 V at 46-48/um (L ~ 21 nm) for
  8e18; 2.5e18 never rises and crosses 0 V near 56/um (18 nm). The axis runs
  to 1/L = 66/um (15 nm); the region L < 2l (l = 14.5 nm for the 1e18
  channel) is shaded because the quasi-2-D model is only qualitative there.
  The no-pocket device has no threshold below ~16 nm (NaN, not plotted).
"""
import sys
import numpy as np
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "path.simplify": False})  # keep every model point
from semiconductor_lib.band_diagrams import (
    char_length, threshold_voltage_q2d_current, gaussian_pocket_profile, pocket_lpe0,
    phi_bulk, EG_SI)
from semiconductor_lib.mosfet import vt_uniform
from semiconductor_lib.constants import eps_ox

out = sys.argv[1] if len(sys.argv) > 1 else "."
tox, Nch, sig, ncrit, VDS = 2e-7, 1e18, 8e-7, 2e18, 0.05
invL = np.concatenate([[0.2, 0.5, 1, 2, 3, 4, 5, 6, 8],
                       np.arange(10, 67, 2)]).astype(float)   # 1/um
Lnm = 1000.0 / invL
peaks = [0.0, 2.5e18, 5e18, 8e18]
cols = [COLORS['red'], COLORS['green'], COLORS['blue'], COLORS['purple']]


def vt_or_nan(L, pk):
    # NaN where the gate has lost control (no threshold in the search range)
    try:
        return threshold_voltage_q2d_current(
            L * 1e-7, VDS, gaussian_pocket_profile(L * 1e-7, Nch, pk, sig),
            tox, ncrit, npts=3001 if L > 500 else (1601 if L > 200 else 801))
    except ValueError:
        return np.nan


res = {pk: np.array([vt_or_nan(L, pk) for L in Lnm]) for pk in peaks}

Cox = eps_ox / tox
LPE0 = pocket_lpe0(Nch, 5e18, sig)
xf = np.linspace(0.05, 66, 80)
Nav = Nch * (1 + LPE0 / (1e-4 / xf))
vt_av = np.array([vt_uniform(N, Cox, V_FB=-(EG_SI / 2) + phi_bulk(N),
                             n_phit=np.log(ncrit / N)) for N in Nav])

fig, ax = plt.subplots(figsize=(6.4, 4.0))
# below L = 2l (channel-centre scale length) the quasi-2-D model is qualitative
x2l = 1e-4 / (2 * char_length(Nch, tox))
ax.axvspan(x2l, 67, color='0.92', zorder=0, lw=0)
ax.text(x2l + 1, -0.085, 'L < 2l: model qualitative', fontsize=6.5,
        color='0.35')
for pk, c in zip(peaks, cols):
    lab = "no pocket" if pk == 0 else f"pocket peak {pk / 1e18:g}×10¹⁸ cm⁻³"
    ax.plot(invL, res[pk], '-o', ms=2.5, color=c, label=lab)
    if pk > 0:
        i = int(np.nanargmax(res[pk]))
        if 0 < i < len(invL) - 1:
            ax.annotate(f'peak {res[pk][i]:.2f} V\nL ≈ {Lnm[i]:.0f} nm',
                        (invL[i], res[pk][i]), xytext=(0, 9),
                        textcoords='offset points', ha='center',
                        fontsize=6, color=c)
ax.plot(xf, vt_av, ':', color=COLORS['blue'], lw=1.3,
        label="lateral-average (LPE0) estimate, 5×10¹⁸ pocket")
ax.set_xlim(0, 67); ax.set_ylim(-0.1, 0.75)
ax.set_xlabel('1/L (µm⁻¹)')
ax.set_ylabel('Constant-current VT at VDS = 0.05 V (V)')
ax.set_title('VT versus 1/L: roll-off, RSCE and the pocket', fontsize=10)
top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
tl = [200, 100, 50, 40, 30, 25, 20, 17, 15]
top.set_xticks([1000 / t for t in tl]); top.set_xticklabels([str(t) for t in tl])
top.set_xlabel('L (nm)')
ax.legend(fontsize=6.5, frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig(f"{out}/_inv.svg")
full_minify_pipeline(f"{out}/_inv.svg", f"{out}/rsce_vt_vs_inverse_L.svg")

print(f"LPE0 = {LPE0 * 1e7:.0f} nm")
for pk in peaks:
    v = res[pk]
    s = (v[3] - v[0]) / (invL[3] - invL[0]) * 1e3   # mV per (1/um) = mV*um
    print(f"pk={pk:.1e}: VT(1/L=0.2)={v[0]:.3f}  slope(0.2-2)={s:.1f} mV*um  "
          f"peak VT={np.nanmax(v):.3f} at 1/L={invL[np.nanargmax(v)]}  "
          + " ".join(f"{x:g}:{y:.3f}" for x, y in zip(invL, v)))
i = np.argmin(abs(xf - 1)); print("avg slope near 0:", (vt_av[i]-vt_av[0])/(xf[i]-xf[0])*1e3)
