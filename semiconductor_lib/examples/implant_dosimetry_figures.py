"""Figure for Semiconductor Notes 'Implant Tool Characterization & Matching'.

(a) Residual dose error from resist-outgassing charge exchange vs end-station
    pressure, for two implanters with different true K-factors, and the error
    left when tool B runs tool A's fitted K.  K_A = 2827 /Torr is the measured
    example in US 6,657,209; K_B = 4000 /Torr is illustrative.
(b) Placement change produced by a 0.5 deg tilt error vs nominal tilt:
    lateral reach of the range (R_p = 30 nm) compared with the shadow edge of
    a 100 nm gate and a 300 nm resist edge.
Analytical models (semiconductor_lib.implant_dosimetry), not measured data.
"""
import numpy as np
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib.pyplot as plt
from semiconductor_lib import implant_dosimetry as d

KA, KB = 2827.0, 4000.0
P = np.linspace(0, 4e-5, 41)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1))
ax1.plot(P * 1e6, [100 * d.residual_dose_error(KA, 0, p) for p in P], color=COLORS["blue"], label="Tool A, no comp.")
ax1.plot(P * 1e6, [100 * d.residual_dose_error(KB, 0, p) for p in P], color=COLORS["red"], label="Tool B, no comp.")
ax1.plot(P * 1e6, [100 * d.residual_dose_error(KB, KA, p) for p in P], color=COLORS["red"], ls="--", label="Tool B with Tool A's K")
ax1.axhline(0, color=COLORS["gray"], lw=0.8, ls=":", label="Each tool, own K")
ax1.set_xlabel("End-station pressure during implant (µTorr)")
ax1.set_ylabel("Delivered dose error (%)")
ax1.set_title("(a) Charge-exchange dose error", fontsize=10)
ax1.legend(fontsize=7.5, frameon=False)

th = np.linspace(5, 45, 41)
ax2.plot(th, [d.lateral_reach_sensitivity(30, t, 0.5) for t in th], color=COLORS["green"], label="Lateral reach, $R_p$ = 30 nm")
ax2.plot(th, [d.shadow_sensitivity(100, t, 0.5) for t in th], color=COLORS["purple"], label="Shadow edge, 100 nm gate")
ax2.plot(th, [d.shadow_sensitivity(300, t, 0.5) for t in th], color=COLORS["orange"], label="Shadow edge, 300 nm resist")
ax2.set_xlabel("Nominal tilt (deg)")
ax2.set_ylabel("Shift per 0.5° tilt error (nm)")
ax2.set_title("(b) Geometric angle sensitivity", fontsize=10)
ax2.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig("implant_dose_angle_sensitivity_raw.svg")
full_minify_pipeline("implant_dose_angle_sensitivity_raw.svg", "implant_dose_angle_sensitivity.svg")
