"""
Drift-diffusion transport relations for silicon. Backs the "Carrier
Transport: Drift and Diffusion" note. Units: mobility cm^2/V s, diffusion
coefficient cm^2/s, fields V/cm, densities cm^-3, lengths cm unless noted.
Low-doping 300 K lattice-limited mobilities are used as defaults (Ioffe).
"""
import numpy as np

from .bands import M0
from .constants import q, thermal_voltage

MU_N_LATTICE = 1400.0         # electrons, lightly doped, 300 K
MU_P_LATTICE = 450.0          # holes
M_COND_ELECTRON = 0.26        # conductivity masses (units of m0)
M_COND_HOLE = 0.39            # Hu


def einstein_diffusion(mu, T=300.0):
    """D = mu kT/q (non-degenerate), cm^2/s. 1400 -> 36.2, 450 -> 11.6 at 300 K."""
    return np.asarray(mu) * thermal_voltage(T)


def conductivity(n, p, mu_n=MU_N_LATTICE, mu_p=MU_P_LATTICE):
    """sigma = q (n mu_n + p mu_p), S/cm."""
    return q * (np.asarray(n) * mu_n + np.asarray(p) * mu_p)


def resistivity(n, p, mu_n=MU_N_LATTICE, mu_p=MU_P_LATTICE):
    """rho = 1/sigma, ohm cm."""
    return 1.0 / conductivity(n, p, mu_n, mu_p)


def scattering_time(mu, m_cond):
    """tau = mu m_c / q in seconds, from mobility (cm^2/V s) and conductivity mass (units of m0)."""
    return np.asarray(mu) * 1e-4 * m_cond * M0 / q


def drift_velocity(mu, E):
    """v = mu E (cm/s); valid at low field only (velocity saturates at high field)."""
    return np.asarray(mu) * np.asarray(E)


def diffusion_length_um(D, tau):
    """L = sqrt(D tau) in micrometres, for D in cm^2/s and lifetime tau in seconds."""
    return np.sqrt(np.asarray(D) * np.asarray(tau)) * 1e4


def electron_current_density(n, mu_n, E, D_n, dn_dx):
    """J_n = q n mu_n E + q D_n dn/dx  (A/cm^2)."""
    return q * n * mu_n * E + q * D_n * dn_dx


def hole_current_density(p, mu_p, E, D_p, dp_dx):
    """J_p = q p mu_p E - q D_p dp/dx  (A/cm^2)."""
    return q * p * mu_p * E - q * D_p * dp_dx
