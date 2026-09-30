"""
Figures for Semiconductor Notes:
  pn_band_diagrams_bias_doping.svg  - p-n junction band diagrams, 4 doping
                                      cases x (equilibrium, forward, reverse)
  mosfet_lateral_band_dibl.svg      - quasi-2-D source-to-drain E_c at V_G=0,
                                      long / short / short+pocket nFET
  mosfet_vt_barrier_vs_L.svg        - V_T roll-off, RSCE, DIBL and off-state
                                      barrier height versus gate length
Analytical / numerical models only (semiconductor_lib.band_diagrams).
"""
import sys
import numpy as np
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib.pyplot as plt
from semiconductor_lib.band_diagrams import (
    pn_band_diagram, lateral_surface_potential, threshold_voltage_q2d,
    gaussian_pocket_profile, EG_SI)

out = sys.argv[1] if len(sys.argv) > 1 else "."
C = COLORS

# ---------------------------------------------------------------- p-n grid
cases = [("Symmetric  $N_A=N_D=10^{17}$", 1e17, 1e17),
         ("Degenerate P$^+$N$^+$  $N_A=N_D=5\\times10^{19}$", 5e19, 5e19),
         ("One-sided N$^+$P  $N_A=10^{16}$, $N_D=10^{20}$", 1e16, 1e20),
         ("One-sided P$^+$N  $N_A=10^{20}$, $N_D=10^{16}$", 1e20, 1e16)]
biases = [(0.0, "Equilibrium, $V_A=0$"), (0.5, "Forward, $V_A=+0.5$ V"),
          (-1.0, "Reverse, $V_A=-1.0$ V")]
fig, axs = plt.subplots(4, 3, figsize=(10, 11), sharey='row')
for i, (title, NA, ND) in enumerate(cases):
    ref = pn_band_diagram(NA, ND, -1.0)
    span = 1.5 * max(ref['xn'], ref['xp'])
    lo = -span if ref['xp'] > 0.02 * span else -0.25 * span
    hi = span if ref['xn'] > 0.02 * span else 0.25 * span
    x = np.linspace(lo, hi, 61)
    for j, (VA, lab) in enumerate(biases):
        ax = axs[i, j]
        d = pn_band_diagram(NA, ND, VA, x=x)
        xn = x * 1e7
        ax.axvspan(-d['xp'] * 1e7, d['xn'] * 1e7, color='0.92', lw=0)
        ax.plot(xn, d['Ec'], color=C['blue'], lw=1.6, label='$E_c$')
        ax.plot(xn, d['Ev'], color=C['purple'], lw=1.6, label='$E_v$')
        def flat(yv, **kw):
            m = ~np.isnan(yv)
            ax.plot([xn[m][0], xn[m][-1]], [yv[m][0], yv[m][0]], '--', **kw)
        if VA == 0:
            flat(np.zeros_like(xn), color=C['green'], lw=1.2, label='$E_F$')
        else:
            flat(d['EFn'], color=C['blue'], lw=1.1, label='$E_{Fn}$')
            flat(d['EFp'], color=C['red'], lw=1.1, label='$E_{Fp}$')
        ax.text(0.03, 0.04, 'P', transform=ax.transAxes, fontsize=10,
                color=C['purple'], fontweight='bold')
        ax.text(0.93, 0.04, 'N', transform=ax.transAxes, fontsize=10,
                color=C['blue'], fontweight='bold')
        if i == 0:
            ax.set_title(lab, fontsize=10)
        if j == 0:
            ax.set_ylabel('Energy (eV)')
            ax.text(0.0, 1.13 if i == 0 else 1.03, title,
                    transform=ax.transAxes, fontsize=9.5, fontweight='bold')
        if i == 3:
            ax.set_xlabel('x (nm)')
        ax.tick_params(labelsize=8)
        ax.set_xlim(xn[0], xn[-1])
axs[0, 0].legend(fontsize=7, loc='center left', frameon=False)
axs[0, 1].legend(fontsize=7, loc='center left', frameon=False)
fig.tight_layout(h_pad=2.2)
fig.savefig(f"{out}/_pn.svg")
full_minify_pipeline(f"{out}/_pn.svg", f"{out}/pn_band_diagrams_bias_doping.svg")

# ------------------------------------------------------- MOSFET lateral E_c
tox = 2e-7
Lsd = 20e-7            # drawn width of the n+ source/drain regions in the plot
uni = lambda y: 2e18 * np.ones_like(y)
devs = [("Long channel, $L$ = 1 µm, uniform $N_A=2\\times10^{18}$", 1e-4,
         lambda L: uni),
        ("Short, $L$ = 40 nm, no pocket", 40e-7, lambda L: uni),
        ("Short, $L$ = 40 nm, with pockets", 40e-7,
         lambda L: gaussian_pocket_profile(L, 1e18, 5e18, 8e-7))]
fig = plt.figure(figsize=(11.5, 3.9))
gs = fig.add_gridspec(1, 4, width_ratios=[0.55, 0.55, 1, 1], wspace=0.08)
a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1], sharey=a0)
a2 = fig.add_subplot(gs[2], sharey=a0); a3 = fig.add_subplot(gs[3], sharey=a0)
panels = [((a0, a1), devs[0]), ((a2,), devs[1]), ((a3,), devs[2])]
for axes, (title, L, prof) in panels:
    for VDS, col in [(0.05, C['blue']), (1.1, C['red'])]:
        s = lateral_surface_potential(L, 0.0, VDS, prof(L), tox,
                                      npts=4001 if L > 1e-5 else 801)
        k = max(1, len(s['y']) // 120)
        idx = np.r_[np.arange(0, len(s['y']) - 1, k), len(s['y']) - 1]
        if L > 1e-5:   # keep resolution near the junctions of the long device
            yy = s['y']
            idx = np.where((yy < 90e-7) | (yy > L - 90e-7))[0][::4]
            idx = np.r_[idx, len(yy) - 1]
        y = s['y'][idx] * 1e7
        Ec = s['Ec'][idx]
        ys = np.r_[-Lsd * 1e7, y, y[-1] + Lsd * 1e7]
        Ecs = np.r_[Ec[0], Ec, Ec[-1]]
        for ax in axes:
            ax.plot(ys, Ecs, color=col, lw=1.8,
                    label=f"$E_c$, $V_{{DS}}$ = {VDS:g} V "
                          f"(barrier {s['barrier']:.2f} eV)")
            ax.plot([y[-1], ys[-1]], [-VDS, -VDS], '--', color=col, lw=1.1)
    for ax in axes:
        ax.plot([ys[0], 0], [0, 0], '--', color=C['green'], lw=1.1)
        ax.axvspan(-Lsd * 1e7, 0, color='0.92', lw=0)
        ax.axvspan(y[-1], y[-1] + Lsd * 1e7, color='0.92', lw=0)
        ax.tick_params(labelsize=8)
    if len(axes) == 2:
        axes[0].set_xlim(-Lsd * 1e7, 80)
        axes[1].set_xlim(y[-1] - 80, y[-1] + Lsd * 1e7)
        axes[1].tick_params(labelleft=False)
        axes[0].set_xticks([0, 40])
        axes[1].set_xticks([960, 1000])
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['left'].set_visible(False)
        axes[0].set_title(title, fontsize=9, loc='left')
        axes[0].set_xlabel('y from source (nm)')
        axes[1].set_xlabel('... to drain (nm)')
        axes[0].legend(fontsize=6.5, loc='lower left', frameon=False)
        axes[0].text(0.03, 0.93, 'n$^+$ S', transform=axes[0].transAxes,
                     fontsize=8)
        axes[1].text(0.6, 0.93, 'n$^+$ D', transform=axes[1].transAxes,
                     fontsize=8)
    else:
        ax = axes[0]
        ax.set_xlim(ys[0], ys[-1])
        ax.set_xticks([0, 20, 40])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('y, source → drain (nm)')
        ax.tick_params(labelleft=False)
        ax.legend(fontsize=7, loc='lower left', frameon=False)
a0.set_ylabel('Energy along surface (eV)\n(0 = source $E_F$)')
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.9)
fig.savefig(f"{out}/_lat.svg")
full_minify_pipeline(f"{out}/_lat.svg", f"{out}/mosfet_lateral_band_dibl.svg")

# --------------------------------------------- V_T and barrier versus L
Ls = np.array([30, 35, 40, 50, 60, 80, 100, 150, 250, 500]) * 1e-7
ncrit = 2e18
res = {}
for name, pf in [("no pocket", lambda L: uni),
                 ("pocket", lambda L: gaussian_pocket_profile(L, 1e18, 5e18,
                                                              8e-7))]:
    r = []
    for L in Ls:
        n = 1601 if L > 2e-5 else 801
        vl = threshold_voltage_q2d(L, 0.05, pf(L), tox, ncrit, npts=n)
        vs = threshold_voltage_q2d(L, 1.1, pf(L), tox, ncrit, npts=n)
        b = lateral_surface_potential(L, 0.0, 1.1, pf(L), tox, npts=n)
        r.append((vl, vs, b['barrier']))
    res[name] = np.array(r)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
Lnm = Ls * 1e7
for name, col in [("no pocket", C['red']), ("pocket", C['blue'])]:
    r = res[name]
    axs[0].plot(Lnm, r[:, 0], '-o', ms=3, color=col,
                label=f"{name}, $V_{{DS}}$ = 0.05 V")
    axs[0].plot(Lnm, r[:, 1], '--s', ms=3, color=col,
                label=f"{name}, $V_{{DS}}$ = 1.1 V")
    axs[1].plot(Lnm, r[:, 2], '-o', ms=3, color=col, label=name)
axs[0].set_xscale('log'); axs[1].set_xscale('log')
axs[0].set_xlabel('Gate length L (nm)'); axs[1].set_xlabel('Gate length L (nm)')
axs[0].set_ylabel('$V_T$ (V)')
axs[0].set_title('Threshold roll-off, RSCE and DIBL', fontsize=10)
axs[1].set_ylabel('Source barrier at $V_G$ = 0 (eV)')
axs[1].set_title('Off-state barrier at $V_{DS}$ = 1.1 V', fontsize=10)
axs[1].axhspan(0, 4 * 0.02585, color='0.9', lw=0)
axs[1].text(150, 0.04, 'barrier < 4kT:\npunch-through', fontsize=7.5)
axs[0].legend(fontsize=7, frameon=False)
axs[1].legend(fontsize=7, frameon=False)
for a in axs:
    a.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(f"{out}/_vt.svg")
full_minify_pipeline(f"{out}/_vt.svg", f"{out}/mosfet_vt_barrier_vs_L.svg")

for name in res:
    print(name)
    for L, (vl, vs, b) in zip(Lnm, res[name]):
        print(f"  L={L:5.0f} nm  VTlin={vl:.3f}  VTsat={vs:.3f}  "
              f"DIBL={(vl - vs) / 1.05 * 1e3:6.0f} mV/V  barrier={b:.3f} eV")
