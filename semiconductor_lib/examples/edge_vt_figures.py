"""
Figure for Semiconductor Notes page on source/drain nitrogen implant and V_T:
  edge_vs_global_vt_shift.svg - V_T shift versus channel length for an
  edge-localized change (local V_T of 0.1 um segments at each junction raised
  or lowered by 100 mV) compared with a length-independent (global) shift
  from 1e11 q/cm^2 of positive fixed charge on a 5 nm oxide (nFET).
Series-segment model (semiconductor_lib.edge_vt). Model results only.

Verified results: at L = 20 um, +100 mV edges give +5.7 mV (subthreshold
constant-current) and +1.0 mV (strong-inversion length average); -100 mV edges
give -0.32 mV; the global term is -23 mV at every length.
"""
import sys
import numpy as np
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib.pyplot as plt
from semiconductor_lib.edge_vt import (edge_vt_shift_subthreshold,
                                       edge_vt_shift_strong_inversion)
from semiconductor_lib.constants import q, eps_ox

out = sys.argv[1] if len(sys.argv) > 1 else "."
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
L = np.logspace(np.log10(0.3), 2, 60)          # um
w, d = 0.1, 0.1
cox = eps_ox / 5e-7
glob = -1e11 * q / cox * 1e3
up = 1e3 * edge_vt_shift_subthreshold(L, w, d)
upsi = 1e3 * edge_vt_shift_strong_inversion(L, w, d)
dn = 1e3 * edge_vt_shift_subthreshold(L, w, -d)

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.semilogx(L, up, color=COLORS["blue"], label="edge +100 mV, subthreshold")
ax.semilogx(L, upsi, color=COLORS["blue"], ls="--", label="edge +100 mV, strong inversion")
ax.semilogx(L, dn, color=COLORS["red"], label="edge -100 mV, subthreshold")
ax.axhline(glob, color=COLORS["green"], label=r"global: $10^{11}$ q/cm$^2$ fixed charge, 5 nm")
ax.axhline(0, color="0.6", lw=0.5)
ax.axvline(20, color="0.4", ls=":", lw=0.8)
ax.text(21, 45, "L = 20 um", fontsize=7, color="0.3")
ax.set_xlabel("Channel length L (um)")
ax.set_ylabel(r"$\Delta V_T$ (mV)")
ax.set_ylim(-35, 90)
ax.legend(fontsize=6.5, frameon=False, loc="upper right")
ax.set_title("Edge-localized vs global V_T shift (0.1 um edge per side)", fontsize=8)
fig.tight_layout()
raw = f"{out}/edge_raw.svg"
fig.savefig(raw)
full_minify_pipeline(raw, f"{out}/edge_vs_global_vt_shift.svg")
print("glob", round(glob, 2), "at20", round(float(edge_vt_shift_subthreshold(20, w, d))*1e3, 2),
      round(float(edge_vt_shift_strong_inversion(20, w, d))*1e3, 2),
      round(float(edge_vt_shift_subthreshold(20, w, -d))*1e3, 2))
