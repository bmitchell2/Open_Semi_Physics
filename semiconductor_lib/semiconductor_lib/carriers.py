"""
Carrier statistics from effective masses and the Fermi level: Fermi-Dirac
occupation, 3-D density of states, effective density of states, and the
numerical-integration cross-check of the Boltzmann closed forms. Backs the
"Fermi Level and Carrier Statistics" and "Degenerate Semiconductors" notes.
Units: eV, cm^-3, K.
"""
import numpy as np
from scipy.integrate import quad

from .bands import M0
from .constants import k_B, q, ni300
from . import dopants

H_PLANCK = 6.62607015e-34     # J s
KB_EV = k_B / q


def fermi_dirac(E_minus_EF_eV, T=300.0):
    """Occupation probability 1/(1+exp((E-EF)/kT))."""
    return 1.0 / (1.0 + np.exp(np.asarray(E_minus_EF_eV) / (KB_EV * T)))


def dos_3d(E_eV, m_rel):
    """
    Parabolic-band density of states (states per cm^3 per eV, both spins) at
    energy E_eV measured INTO the band from its edge; zero for E <= 0.
    """
    E = np.clip(np.asarray(E_eV, float), 0, None)
    return 4 * np.pi * (2 * m_rel * M0) ** 1.5 / H_PLANCK ** 3 * np.sqrt(E * q) * q * 1e-6


def effective_dos(m_rel, T=300.0):
    """Effective density of states 2 (2 pi m kT / h^2)^(3/2), cm^-3.
    m=1.08 gives 2.8e19 (Hu); m=1.18 gives 3.2e19 (Ioffe) at 300 K."""
    return 2 * (2 * np.pi * m_rel * M0 * k_B * T / H_PLANCK ** 2) ** 1.5 * 1e-6


def boltzmann_density(edge_minus_EF_eV, N_eff, T=300.0):
    """n = Nc exp(-(Ec-EF)/kT)  (or p = Nv exp(-(EF-Ev)/kT)), cm^-3."""
    return N_eff * np.exp(-np.asarray(edge_minus_EF_eV) / (KB_EV * T))


def density_numerical(Ec_minus_EF, m_rel, T=300.0, E_max=1.0):
    """Electron density by integrating dos_3d x fermi_dirac numerically (cm^-3);
    agrees with boltzmann_density to ~0.02% when EF is >> kT below Ec."""
    return quad(lambda E: dos_3d(E, m_rel) * fermi_dirac(E + Ec_minus_EF, T), 0, E_max)[0]


def ec_minus_ef(N_D, Nc, T=300.0):
    """Ec - EF = kT ln(Nc/ND) for complete ionization (eV)."""
    return KB_EV * T * np.log(Nc / N_D)


def fermi_potential(N, T=300.0, ni=ni300):
    """phi_F = (kT/q) ln(N/ni) in volts (0.357 V for N=1e16 at 300 K)."""
    return KB_EV * T * np.log(N / ni)


def intrinsic_level_offset_eV(T=300.0):
    """Ei - midgap = -(kT/2) ln(Nc/Nv): about -7 meV in silicon at 300 K."""
    return -(KB_EV * T / 2) * np.log(dopants.Nc(T) / dopants.Nv(T))


def density_numerical_wide(Ec_minus_EF, m_rel, T=300.0, kT_span=25.0):
    """
    Like density_numerical(), but sets the integration ceiling wide enough
    (kT_span kT above EF, or above Ec if that is higher) to stay accurate
    when EF sits near or inside the band (degenerate case), where the
    Fermi tail extends well past a fixed 1 eV ceiling. Used to cross-check
    the Boltzmann form against the exact Fermi-Dirac integral all the way
    from non-degenerate into degenerate doping; see the "Degenerate
    Semiconductors" note.
    """
    E_max = max(1.0, -Ec_minus_EF + kT_span * KB_EV * T)
    return density_numerical(Ec_minus_EF, m_rel, T, E_max=E_max)


def boltzmann_validity_ratio(Ec_minus_EF, m_rel, T=300.0):
    """
    Ratio of the Boltzmann closed form to the exact numerically-integrated
    Fermi-Dirac density, n_Boltzmann / n_exact, as a function of Ec-EF
    (eV; negative once EF crosses into the conduction band). Close to 1
    while non-degenerate (Ec-EF >> kT) and grows above 1 as EF approaches
    or enters the band, because the Boltzmann form keeps growing
    exponentially while Pauli exclusion caps the true occupation.
    """
    Ec_minus_EF = np.atleast_1d(np.asarray(Ec_minus_EF, dtype=float))
    n_exact = np.array([density_numerical_wide(x, m_rel, T) for x in Ec_minus_EF])
    n_boltz = boltzmann_density(Ec_minus_EF, effective_dos(m_rel, T), T)
    return n_boltz / n_exact


def ionized_donor_fraction_fixed_EF(Ec_minus_EF, Ed=0.045, T=300.0, g=2):
    """
    Ionized fraction of donors when EF is fixed from n = N_D (Hu, Example 1-6
    style estimate): 1 / (1 + g exp((Ed - (Ec-EF))/kT)). The self-consistent
    value comes from dopants.solve_donor.
    """
    return 1.0 / (1.0 + g * np.exp((Ed - np.asarray(Ec_minus_EF)) / (KB_EV * T)))
