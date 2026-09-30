"""
Figure generators for the GIDL / pocket-leakage and FinFET Semiconductor
Notes pages (2026-09-29). Same style and size rules as figures.py.
Physics comes from leakage.py and multigate.py.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from .figures import C, LOG_FORMATTER, notes_style, save_figure


def fig_pocket_btbt(ND=1e20, VR=1.2, n_points=60):
    """Peak field and relative BTBT current density at the extension/pocket
    junction versus pocket (lighter-side) doping. Analytical model."""
    from . import leakage
    notes_style()
    Np = np.logspace(np.log10(3e17), 19, n_points)
    E = leakage.peak_field(Np, ND, VR)
    fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.0), sharex=True,
                            gridspec_kw=dict(hspace=0.12, height_ratios=[1, 1.5]))
    a = axs[0]
    a.semilogx(Np, E / 1e6, color=C["dark"])
    a.set_ylabel("Peak field (MV/cm)")
    a.set_title("Abrupt N+ extension (1e20 cm⁻³) / p-type pocket, reverse bias 1.2 V "
                "(depletion approximation)", fontsize=7.4)
    b = axs[1]
    ref = np.argmin(abs(Np - 1e18))
    for B, col, lab in [(3.6e7, C["blue"], "B = 3.6e7 V/cm (direct, m* = 0.2 m0)"),
                        (2.0e7, C["red"], "B = 2.0e7 V/cm (typical calibrated Si)")]:
        J = leakage.btbt_current_density(E, VR, B_override=B)
        b.loglog(Np, J / J[ref], color=col, label=lab)
    b.axhline(1.0, color=C["gray"], lw=0.6, ls=":")
    b.set_ylim(1e-6, 1e8)
    b.yaxis.set_major_formatter(LOG_FORMATTER); b.yaxis.set_minor_formatter(NullFormatter())
    b.set_yticks([1e-6, 1e-3, 1e0, 1e3, 1e6])
    b.set_ylabel("BTBT current density\n(relative to 1e18 cm⁻³)")
    b.set_xlabel("Pocket (lighter-side) doping (cm⁻³)")
    b.xaxis.set_major_formatter(LOG_FORMATTER)
    b.legend(frameon=False, fontsize=6.8, loc="lower right")
    return fig


def fig_natural_length(t_ox_nm=1.0, n_points=50):
    """Minimum gate length 5*lambda versus body thickness (fin width, or bulk
    depletion depth) for 1, 2, 3 and 4 equivalent gates. First-order model."""
    from . import multigate
    notes_style()
    t = np.linspace(4, 25, n_points)
    fig, a = plt.subplots(figsize=(5.2, 3.6))
    styles = [(1, C["dark"], "single gate (planar: t = depletion depth)"),
              (2, C["blue"], "double gate (FinFET sidewalls)"),
              (3, C["green"], "tri-gate"),
              (4, C["red"], "gate-all-around")]
    for n, col, lab in styles:
        Lmin = multigate.min_gate_length(t * 1e-7, t_ox_nm * 1e-7, n, ratio=5.0) * 1e7
        a.plot(t, Lmin, color=col, label=lab)
    a.set_xlabel("Body thickness t (nm): fin width, film thickness, or depletion depth")
    a.set_ylabel("5λ (nm): approx. shortest gate length")
    a.set_title("Natural length λ = √(εsi t tox / N εox), EOT = %g nm (first-order model)" % t_ox_nm,
                fontsize=7.4)
    a.set_xlim(4, 25); a.set_ylim(0, 60)
    a.grid(alpha=0.25, lw=0.5)
    a.legend(frameon=False, fontsize=6.8, loc="upper left")
    return fig


FIGURES = {
    "pocket_junction_field_and_btbt": fig_pocket_btbt,
    "multigate_natural_length_min_gate_length": fig_natural_length,
}


def build_all(outdir, minify=True, png=False):
    """Generate every figure in this module into outdir; returns {name: path}."""
    return {name: save_figure(fn(), os.path.join(outdir, name), png=png, minify=minify)
            for name, fn in FIGURES.items()}
