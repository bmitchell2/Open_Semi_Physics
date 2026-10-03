"""Figures for the RF oscillator, RF transistor FOM, and spiral-inductor
topic pages in Semiconductor Notes.  All curves are analytical models
(semiconductor_lib.rf), not measured data."""
import os, sys
import numpy as np
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib
matplotlib.rcParams["axes.formatter.use_mathtext"] = False
matplotlib.rcParams["xtick.minor.visible"] = False
import matplotlib.pyplot as plt
from semiconductor_lib import rf

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
c = COLORS


def save(fig, name):
    for a in fig.axes:
        a.minorticks_off()
    raw = os.path.join(OUT, name.replace(".svg", "_raw.svg"))
    fig.savefig(raw, bbox_inches="tight"); plt.close(fig)
    full_minify_pipeline(raw, os.path.join(OUT, name)); os.remove(raw)


# 1. Leeson phase noise vs tank Q -------------------------------------------
f0, P, F, fc = 5e9, 1e-3, 3.0, 300e3
df = np.logspace(3, 9, 120)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for Q, col in ((5, c["red"]), (10, c["purple"]), (20, c["blue"])):
    ax.semilogx(df, rf.leeson_phase_noise(df, f0, Q, P, F, fc), color=col,
                label=f"Q = {Q}  (f0/2Q = {f0/2/Q/1e6:.0f} MHz)")
ax.axvline(fc, color=c["gray"], ls=":", lw=1)
ax.text(fc * 1.15, -55, "1/f³ corner\n(300 kHz)", fontsize=8, color=c["gray"])
ax.text(3e3, -88, "-30 dB/dec\n(upconverted 1/f)", fontsize=8)
ax.text(2.5e6, -118, "-20 dB/dec\n(white noise)", fontsize=8)
ax.text(1.5e7, -176, "flat floor 2FkT/P", fontsize=8)
ax.set_xlabel("Offset frequency Δf (Hz)")
ax.set_ylabel("SSB phase noise L(Δf) (dBc/Hz)")
ax.set_title("Leeson model: f0 = 5 GHz, Psig = 1 mW, F = 3", fontsize=10)
ax.set_ylim(-180, -40); ax.grid(True, which="major", alpha=0.3)
ax.legend(fontsize=8, loc="upper right")
save(fig, "fig_leeson_phase_noise.svg")

# 2. PLL noise shaping vs loop bandwidth ------------------------------------
N, L_pfd = 125, -150.0          # 40 MHz reference -> 5 GHz output
df = np.logspace(3, 8, 120)
L_vco = rf.leeson_phase_noise(df, f0, 15, 1e-3, F, fc)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.semilogx(df, np.full_like(df, L_pfd + rf.reference_noise_multiplication_db(N)),
            color=c["gray"], ls="--", label="reference × N² (−108 dBc/Hz)")
ax.semilogx(df, L_vco, color=c["gray"], ls=":", label="free-running VCO (Q = 15)")
for fbw, col in ((1e5, c["red"]), (1e6, c["blue"]), (1e7, c["orange"])):
    ax.semilogx(df, rf.pll_output_phase_noise_db(df, L_pfd, L_vco, N, fbw),
                color=col, label=f"PLL output, loop BW {fbw/1e6:g} MHz")
ax.set_xlabel("Offset frequency Δf (Hz)")
ax.set_ylabel("SSB phase noise L(Δf) (dBc/Hz)")
ax.set_title("PLL noise shaping (first-order loop model, no peaking)", fontsize=10)
ax.set_ylim(-150, -50); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
save(fig, "fig_pll_noise_shaping.svg")

# 3. fT / fmax vs gate finger width -----------------------------------------
W = 32.0
gm, cgs, cgd, gds = 1.5e-3 * W, 0.8e-15 * W, 0.35e-15 * W, 0.15e-3 * W
fT = rf.ft_mosfet(gm, cgs, cgd)
wf = np.logspace(np.log10(0.25), np.log10(8), 40)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for dc, col, lab in ((False, c["red"], "gate contacted one end"),
                     (True, c["blue"], "gate contacted both ends")):
    Rg = rf.gate_resistance(8.0, wf * 1e-6, 40e-9, W / wf, dc, 30.0)
    ax.semilogx(wf, rf.fmax_mosfet(fT, Rg, gds, cgd, Ri=1 / (5 * gm), Rs=150 / W) / 1e9,
                color=col, label=f"fmax, {lab}")
ax.axhline(fT / 1e9, color=c["gray"], ls="--", label=f"fT = {fT/1e9:.0f} GHz (independent of fingering)")
ax.set_xlabel("Gate finger width Wf (µm)   [total W = 32 µm]")
ax.set_ylabel("Frequency (GHz)")
ax.set_title("40 nm-class NMOS, illustrative parameters", fontsize=10)
ax.set_xticks([0.25, 0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"]); ax.minorticks_off()
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
save(fig, "fig_fmax_finger_width.svg")

# 4. Spiral inductor Q vs frequency and substrate ---------------------------
f = np.logspace(8.3, 10.6, 160)
Rs = rf.series_resistance_skin(1.0, 3e-6, 1.72e-8, f)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for (R, C), col, lab in ((rf.substrate_branch(0.1, 4000), c["red"], "bulk Si, 10 Ω·cm"),
                          ((5.0, 1e-18), c["purple"], "10 Ω·cm + patterned ground shield"),
                          (rf.substrate_branch(10.0, 4000), c["blue"], "high-resistivity Si / SOI, 1 kΩ·cm")):
    ax.semilogx(f / 1e9, rf.inductor_q_one_port(f, 2e-9, Rs, 100e-15, R, C), color=col, label=lab)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Q = Im(Z)/Re(Z)")
ax.set_title("2 nH spiral, 3 µm Cu top metal, single-pi model", fontsize=10)
ax.set_ylim(-5, 25); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
save(fig, "fig_inductor_q_substrate.svg")
