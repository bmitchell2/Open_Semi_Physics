"""Figure for the Layout-Dependent Effects (LOD and WPE) page of Semiconductor
Notes. Model geometry terms are exact BSIM4 forms; process coefficients are
illustrative. Run from semiconductor_lib/; writes /tmp/lde.svg."""
import sys; sys.path.insert(0, '.')
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import numpy as np, matplotlib.pyplot as plt
from semiconductor_lib import layout_effects as le
C = COLORS
KU_N, KU_P = -0.011, 0.0165          # um, illustrative: -10 % / +15 % at SA=SB=0.2 um, L=40 nm
fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.6))
sa = np.logspace(-1, 1, 25)
ax[0].semilogx(sa, 100 * (le.lod_mobility_ratio(sa, sa, 0.04, KU_N) - 1), color=C['blue'], label='NMOS')
ax[0].semilogx(sa, 100 * (le.lod_mobility_ratio(sa, sa, 0.04, KU_P) - 1), color=C['red'], label='PMOS')
ax[0].axhline(0, color=C['gray'], lw=0.8)
ax[0].set_xlabel('SA = SB, gate edge to STI edge (µm)'); ax[0].set_ylabel('Δµ/µ vs stress-free (%)')
ax[0].set_title('(a) LOD mobility, L = 40 nm', fontsize=10); ax[0].legend(fontsize=8)
t = le.finger_stress_terms(0.3, 0.3, 0.5, 8, 0.3)
ax[1].bar(np.arange(1, 9), t / t.min(), color=[C['orange'] if i in (0, 7) else C['blue'] for i in range(8)])
ax[1].set_xlabel('Finger index (NF = 8)'); ax[1].set_ylabel('1/(SAᵢ+L/2)+1/(SBᵢ+L/2), relative')
ax[1].set_title('(b) Stress term per finger', fontsize=10); ax[1].set_xticks(range(1, 9))
sc = np.logspace(np.log10(0.2), 1, 25)
k = 0.0225                             # V, illustrative KVTH0WE
ax[2].loglog(sc, 1e3 * le.wpe_delta_vth(sc, 1.0, k), color=C['green'], label='KVTH0WE·SCA, d = 1 µm')
ax[2].loglog(sc, 1e3 * k / sc**2, color=C['gray'], ls='--', lw=1, label=r'point device, $\propto 1/SC^2$')
ax[2].set_xlabel('SC, well edge to device edge (µm)'); ax[2].set_ylabel(r'$\Delta V_T$ (mV)')
ax[2].set_title(r'(c) Well proximity $\Delta V_T$', fontsize=10); ax[2].legend(fontsize=8); ax[2].set_ylim(0.1, 300)
fig.tight_layout(); fig.savefig('/tmp/lde_raw.svg'); plt.close(fig)
full_minify_pipeline('/tmp/lde_raw.svg', '/tmp/lde.svg')
