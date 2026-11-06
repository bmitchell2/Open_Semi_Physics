"""Figures for the channeling/tilt, alignment-mark and field-isolation
pages of Semiconductor Notes. Run from the semiconductor_lib/ folder;
writes SVGs to /tmp, then minify with plotting.minify_svg."""
import sys; sys.path.insert(0, '.')
from semiconductor_lib.plotting import apply_style, COLORS, minify_svg
apply_style()
import numpy as np, matplotlib.pyplot as plt
from semiconductor_lib import implant_geometry as ig, alignment_marks as am, field_isolation as fi
C = COLORS

# 1. channeling critical angle and quad-mode shadowing
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
E = np.logspace(2.7, 5.3, 40)
for Z, name, col in ((5, 'B', C['blue']), (15, 'P', C['green']), (33, 'As', C['red'])):
    ax[0].semilogx(E / 1e3, ig.critical_angle_deg(Z, E, '100'), color=col, label=f'{name} along ⟨100⟩')
ax[0].semilogx(E / 1e3, ig.critical_angle_deg(5, E, '110'), color=C['blue'], ls=':', label='B along ⟨110⟩')
ax[0].axhline(7, color=C['gray'], ls='--', lw=1); ax[0].text(0.6, 7.3, 'typical 7° tilt', color=C['gray'], fontsize=8)
ax[0].set_xlabel('Ion energy (keV)'); ax[0].set_ylabel('Critical angle (°)')
ax[0].set_title('(a) Lindhard axial critical angle in Si', fontsize=10); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 14)
gap, h, tilt = 200.0, 150.0, 7.0
x = np.linspace(0, gap, 801)
ax[1].plot(x, ig.surface_dose_in_gap(x, gap, h, tilt, (0.0,)), color=C['gray'], lw=1.2, label='single pass (twist toward +x)')
ax[1].plot(x, ig.surface_dose_in_gap(x, gap, h, tilt), color=C['blue'], lw=2, label='quad, twists 0/90/180/270°')
ax[1].plot(x, ig.surface_dose_in_gap(x, gap, h, tilt, (45., 135., 225., 315.)), color=C['red'], ls='--', label='quad, twists 45/135/225/315°')
ax[1].set_xlabel('Position across a 200 nm gap between 150 nm walls (nm)')
ax[1].set_ylabel('Fraction of nominal dose at surface'); ax[1].set_ylim(-0.05, 1.12)
ax[1].set_title('(b) Shadowing at 7° tilt', fontsize=10); ax[1].legend(fontsize=7.5, loc='lower center')
fig.tight_layout(); fig.savefig('/tmp/f1.svg'); plt.close(fig)

# 2. alignment mark first-order signal vs depth
fig, ax = plt.subplots(figsize=(6.4, 3.6))
d = np.linspace(0, 450, 451)
for lam, col in ((532, C['green']), (633, C['red']), (780, C['purple']), (850, C['gray'])):
    ax.plot(d, am.first_order_efficiency(d, lam), color=col, label=f'{lam} nm')
ax.axvspan(100, 150, color=C['orange'], alpha=0.12); ax.text(103, 0.43, 'common\ndepth window', fontsize=8, color=C['orange'])
ax.set_xlabel('Mark step depth in air (nm)'); ax.set_ylabel('±1st-order efficiency (each order)')
ax.set_ylim(0, 0.5); ax.legend(fontsize=8, title='wavelength', title_fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig('/tmp/f2.svg'); plt.close(fig)

# 3. field transistor threshold vs doping under the trench
fig, ax = plt.subplots(figsize=(6.4, 3.8))
N = np.logspace(15.5, 18, 60)
for Qf, col, lab in ((0, C['blue'], 'Q_f = 0'), (1e11, C['green'], 'Q_f = 1×10¹¹ cm⁻²'), (1e12, C['red'], 'Q_f = 1×10¹² cm⁻² (e.g. after high total dose)')):
    ax.semilogx(N, fi.field_threshold(N, 300e-7, Qf), color=col, label=lab)
ax.semilogx(N, fi.field_threshold(N, 400e-7), color=C['blue'], ls=':', label='Q_f = 0, 400 nm oxide')
for v, txt, xt in ((1.1, 'core supply', 3.2e17), (5, '5 V I/O', 3.2e17), (40, '40 V routing', 4e15)):
    ax.axhline(v, color=C['gray'], lw=0.8, ls='--'); ax.text(xt, v + 0.8, txt, fontsize=7.5, color=C['gray'])
ax.set_xlabel('Acceptor doping at the trench surface, N_A (cm⁻³)'); ax.set_ylabel('Field transistor V_T (V)')
ax.set_ylim(-5, 50); ax.legend(fontsize=7.5, loc='upper left')
fig.tight_layout(); fig.savefig('/tmp/f3.svg'); plt.close(fig)

if __name__ == '__main__':
    for i in (1, 2, 3):
        print(minify_svg(f'/tmp/f{i}.svg', f'/tmp/f{i}m.svg'))
