"""
Metal(silicide)-semiconductor contacts: Schottky barrier, thermionic
emission, and tunnelling ohmic contacts.

Backs the Semiconductor Notes page "Metal-Semiconductor Contacts (Schottky
Barriers and Ohmic Contacts)".

Relations (C. Hu, Modern Semiconductor Devices for ICs, Ch. 4, Secs 4.16-4.21;
Sze & Ng, Physics of Semiconductor Devices, Ch. 3):
  Fermi-level pinning (empirical fit to metals on n-Si, Hu Eq. 4.16.3):
      phi_Bn = 0.7 + 0.2 (psi_M - 4.75)  [V];  phi_Bn + phi_Bp ~ Eg
  Thermionic emission:  J = A* T^2 exp(-phi_B/V_T) (exp(V/V_T) - 1),
      A* ~ 100 A/cm^2/K^2 for Si electrons (Hu uses K = 100).
  Characteristic tunnelling energy (Padovani-Stratton):
      E00 = (q hbar / 2) sqrt(N / (m_t eps_s))  [eV]
      kT >> E00: thermionic; kT ~ E00: thermionic-field; E00 >> kT: field
      emission (tunnelling).
  Field-emission specific contact resistivity (Hu Eqs 4.21.1-4.21.8):
      depletion width  W = sqrt(2 eps_s phi_B / (q N)),
      tunnelling prob. P = exp(-H phi_B / sqrt(N)),
      H = (4 pi / h) sqrt(eps_s m_t)   [SI; phi_B in V, q cancels]
      (exponent = 2 (W/2) sqrt(8 pi^2 m q phi_B)/h with W from above)
      rho_c = 2 exp(H phi_B/sqrt(N)) / (q v_thx H sqrt(N)),
      v_thx = sqrt(2 kT / (pi m))  (one-direction mean speed).
  Hu notes the prefactor and H are usually fitted to data; the exponential
  dependence on phi_B/sqrt(N) is the robust physics.

Units: cm, V, cm^-3, Ohm cm^2 unless noted.
"""
import numpy as np

from .constants import q, eps_si, k_B

HBAR = 1.054571817e-34   # J s
H_PLANCK = 6.62607015e-34
M0 = 9.1093837015e-31    # kg
A_STAR_SI_N = 112.0      # A/cm^2/K^2 (Sze & Ng, n-Si); Hu rounds to 100


def pinned_barrier_n(psi_m):
    """Hu's empirical pinned barrier for metals on n-Si (V)."""
    return 0.7 + 0.2 * (np.asarray(psi_m, float) - 4.75)


def depletion_width_cm(phi_B, N):
    return np.sqrt(2 * eps_si * np.asarray(phi_B, float) / (q * np.asarray(N, float)))


def e00_eV(N, m_rel=0.26):
    """Padovani-Stratton characteristic tunnelling energy E00 (eV)."""
    eps = eps_si * 1e2            # F/cm -> F/m
    N_m3 = np.asarray(N, float) * 1e6
    m = m_rel * M0
    return (HBAR / 2) * np.sqrt(N_m3 / (m * eps))   # J/C*... -> in V (= eV)


def thermionic_J(phi_B, V, T=300.0, A_star=A_STAR_SI_N):
    Vt = k_B * T / q
    return A_star * T**2 * np.exp(-phi_B / Vt) * (np.exp(np.asarray(V, float) / Vt) - 1)


def thermionic_rho_c(phi_B, T=300.0, A_star=A_STAR_SI_N):
    """Zero-bias specific contact resistivity for pure thermionic emission."""
    Vt = k_B * T / q
    return (Vt / (A_star * T**2)) * np.exp(np.asarray(phi_B, float) / Vt)


def tunnelling_rho_c(phi_B, N, m_rel=0.26, T=300.0):
    """Hu's field-emission specific contact resistivity (Ohm cm^2)."""
    eps = eps_si * 1e2
    m = m_rel * M0
    H = (4 * np.pi / H_PLANCK) * np.sqrt(eps * m)            # m^-3/2 per V
    N_m3 = np.asarray(N, float) * 1e6
    vthx = np.sqrt(2 * k_B * T / (np.pi * m))                 # m/s
    rho_m2 = 2 * np.exp(H * phi_B / np.sqrt(N_m3)) / (q * vthx * H * np.sqrt(N_m3))
    return rho_m2 * 1e4                                       # Ohm cm^2


def tunnelling_probability(phi_B, N, m_rel=0.26):
    eps = eps_si * 1e2
    H = (4 * np.pi / H_PLANCK) * np.sqrt(eps * m_rel * M0)
    return np.exp(-H * phi_B / np.sqrt(np.asarray(N, float) * 1e6))


def contact_resistance_ohm(rho_c, diameter_cm):
    """Vertical contact resistance of a circular contact of given diameter
    (no current crowding): R = rho_c / area."""
    return rho_c / (np.pi * (diameter_cm / 2) ** 2)
