"""
Figure generators for the Semiconductor Notes pages "Forming Gas Anneal
(Hydrogen Sinter)" and "Metal-Semiconductor Contacts" (2026-10-03).
Same style/size rules as figures.py. Physics from hydrogen_passivation.py
and contacts.py.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from .figures import C, LOG_FORMATTER, notes_style, save_figure
from . import hydrogen_passivation as hp
from . import contacts as ct


def fig_fga_kinetics(n_points=80):
    """(a) Brower time constants vs temperature; (b) fraction of passivated
    interface sites lost in a hydrogen-free anneal. Analytical model."""
    notes_style()
    Tc = np.linspace(250, 800, n_points)
    T = hp.c_to_k(Tc)
    fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.4), sharex=True,
                            gridspec_kw=dict(hspace=0.12))
    a = axs[0]
    a.semilogy(Tc, 1 / hp.k_dissociation(T), color=C["red"],
               label="Si-H dissociation, 1/k_d (Ed = 2.56 eV)")
    for H2, ls in [(1e17, "-"), (1e18, "--")]:
        a.semilogy(Tc, 1 / (hp.k_passivation(T) * H2), color=C["blue"], ls=ls,
                   label="passivation, 1/(k_f[H2]), [H2] = %s cm-3 (Ef = 1.66 eV)" % ("1e17" if H2 == 1e17 else "1e18"))
    a.axvspan(400, 450, color=C["gray"], alpha=0.15, lw=0)
    a.axhline(1800, color=C["gray"], lw=0.8, ls=":")
    a.text(255, 2600, "30 min", fontsize=7, color=C["gray"])
    a.text(404, 2e8, "typical\nFGA", fontsize=7, color=C["gray"])
    a.set_ylim(1e-8, 1e12)
    a.yaxis.set_major_formatter(LOG_FORMATTER)
    a.set_ylabel("Time constant (s)")
    a.legend(fontsize=6.4, frameon=False, loc="lower left")
    a.set_title("Brower kinetics for interface dangling bonds (Pb), single activation energies",
                fontsize=7.4)
    b = axs[1]
    for t, ls, lab in [(1800, "-", "30 min"), (60, "--", "60 s (RTA)")]:
        b.plot(Tc, 100 * hp.depassivated_in_inert(t, T), color=C["dark"], ls=ls,
               label="%s in N2 / vacuum (no H2)" % lab)
    b.axvspan(400, 450, color=C["gray"], alpha=0.15, lw=0)
    b.set_ylabel("Passivated sites lost (%)")
    b.set_xlabel("Anneal temperature (°C)")
    b.set_ylim(-2, 102)
    b.legend(fontsize=7, frameon=False, loc="upper left")
    return fig


def fig_contact_resistivity(n_points=70):
    """Specific contact resistivity vs surface doping (Hu field-emission
    model) for three barrier heights, with the thermionic limit. Model."""
    notes_style()
    N = np.logspace(np.log10(2e19), np.log10(6e20), n_points)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for phi, col in [(0.65, C["red"]), (0.45, C["orange"]), (0.30, C["blue"])]:
        ax.loglog(N, ct.tunnelling_rho_c(phi, N), color=col,
                  label="φB = %.2f V (tunnelling)" % phi)
    ax.axhline(7.07e-9, color=C["gray"], ls="--", lw=0.9)
    ax.text(2.1e19, 9.5e-9, "1 kΩ for a 30 nm diameter contact (7e-9 Ω·cm²)", fontsize=7, color=C["gray"])
    ax.axhline(ct.thermionic_rho_c(0.65), color=C["red"], ls=":", lw=0.9)
    ax.text(2.1e19, ct.thermionic_rho_c(0.65) * 0.25,
            "thermionic only, φB = 0.65 V: %.0f Ω·cm²" % ct.thermionic_rho_c(0.65),
            fontsize=7, color=C["red"])
    ax.set_xlabel("Active doping at the silicide interface (cm⁻³)")
    ax.set_ylabel("Specific contact resistivity (Ω·cm²)")
    ax.xaxis.set_major_formatter(LOG_FORMATTER)
    ax.yaxis.set_major_formatter(LOG_FORMATTER)
    ax.set_ylim(1e-10, 1e3)
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.set_title("Field-emission model (Hu Eq. 4.21.8, m* = 0.26 m0, 300 K)", fontsize=7.4)
    return fig


def build_all(outdir, minify=True, png=False):
    os.makedirs(outdir, exist_ok=True)
    out = {}
    out["fga_kinetics"] = save_figure(fig_fga_kinetics(), os.path.join(outdir, "fga_kinetics"),
                                      png=png, minify=minify)
    out["contact_rho"] = save_figure(fig_contact_resistivity(), os.path.join(outdir, "contact_rho"),
                                     png=png, minify=minify)
    return out
