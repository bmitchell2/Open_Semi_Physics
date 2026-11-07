"""Surface-potential regimes of a p-type MOS surface (N_A = 1e17 cm^-3).

Top: surface hole and electron densities versus band bending psi_s
(Boltzmann statistics). Bottom: magnitude of the total induced
semiconductor charge from the full Kingston-Neustadter relation, compared
with the depletion-approximation charge sqrt(2 q eps_si N_A psi_s).
Regime boundaries at psi_s = 0, phi_F and 2 phi_F. Analytical model.
"""
import numpy as np
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib.pyplot as plt
from semiconductor_lib.constants import q, eps_si
from semiconductor_lib.electrostatics import (
    surface_carrier_densities_p, fermi_potential_p, semiconductor_charge_p)

NA = 1e17
phiF = fermi_potential_p(NA)
psi = np.linspace(-0.15, 1.0, 140)
p_s, n_s = surface_carrier_densities_p(psi, NA)
Qs = np.abs(semiconductor_charge_p(psi, NA))
psi_d = psi[psi > 0]
Qdep = np.sqrt(2 * q * eps_si * NA * psi_d)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 5.6), sharex=True)
regions = [(-0.15, 0, "accumulation", "#eaeef2"), (0, phiF, "depletion", "#ffffff"),
           (phiF, 2 * phiF, "weak\ninversion", "#fff8c5"), (2 * phiF, 1.0, "strong\ninversion", "#dafbe1")]
for ax in (a1, a2):
    for x0, x1, _, c in regions:
        ax.axvspan(x0, x1, color=c, lw=0)
    for x in (0, phiF, 2 * phiF):
        ax.axvline(x, color=COLORS["gray"], lw=0.7, ls=":")
for x0, x1, lab, _ in regions:
    a1.text((x0 + x1) / 2, 3e20, lab, ha="center", va="top", fontsize=8)

a1.semilogy(psi, p_s, color=COLORS["red"], label="holes $p_s$")
a1.semilogy(psi, n_s, color=COLORS["blue"], label="electrons $n_s$")
a1.axhline(NA, color=COLORS["gray"], lw=0.8, ls="--")
a1.text(-0.13, NA * 2, "$N_A$", fontsize=8)
a1.axhline(1e10, color=COLORS["gray"], lw=0.8, ls="--")
a1.text(-0.13, 2e10, "$n_i$", fontsize=8)
a1.set_ylim(1e1, 1e21)
a1.set_ylabel("surface density (cm$^{-3}$)")
a1.legend(loc="lower left", fontsize=8, frameon=False)

a2.semilogy(psi, Qs, color=COLORS["purple"], label="total $|Q_s|$ (full solution)")
a2.semilogy(psi_d, Qdep, color=COLORS["green"], ls="--", label="depletion charge only")
a2.set_ylim(1e-9, 3e-5)
a2.set_ylabel("$|Q_s|$ (C/cm$^2$)")
a2.set_xlabel(r"band bending $\psi_s$ (V)   [$\phi_F$ = %.2f V, $2\phi_F$ = %.2f V]" % (phiF, 2 * phiF))
a2.legend(loc="upper left", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("surface_regimes_raw.svg")
full_minify_pipeline("surface_regimes_raw.svg", "surface_regimes.svg")
