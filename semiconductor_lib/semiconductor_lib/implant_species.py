"""
Ion stopping and implant-species selection helpers.

- ZBL universal nuclear stopping and Lindhard-Scharff (LSS) electronic
  stopping in silicon, for comparing where each mechanism dominates.
- Energy partition for molecular ions (BF2+, As2+, B18H22+): on impact
  each atom carries kinetic energy in proportion to its mass.
- Space-charge (perveance) scaling of the transportable beam current,
  used to compare monomer and molecular beams at equal per-atom energy.

Units: energy keV (unless noted), stopping eV/nm in Si.
Validity: LSS electronic stopping (S_e proportional to sqrt(E)) holds below
roughly 25 keV/amu * Z1^(4/3) (ion velocity below the Thomas-Fermi
electron velocity); above that S_e peaks and falls (Bethe regime).
"""
import numpy as np
from .implant_bca import lss_k

N_SI_CM3 = 4.994e22
Z_SI, M_SI = 14, 28.086


def zbl_reduced_energy(Z1, M1, E_keV, Z2=Z_SI, M2=M_SI):
    """ZBL reduced energy epsilon (dimensionless)."""
    return 32.53 * M2 * E_keV / (Z1 * Z2 * (M1 + M2) * (Z1 ** 0.23 + Z2 ** 0.23))


def zbl_nuclear_stopping(Z1, M1, E_keV, Z2=Z_SI, M2=M_SI, N_cm3=N_SI_CM3):
    """ZBL universal nuclear stopping, eV/nm, in a target of density N_cm3."""
    eps = zbl_reduced_energy(Z1, M1, E_keV, Z2, M2)
    sn = np.where(eps <= 30,
                  np.log(1 + 1.1383 * eps) / (2 * (eps + 0.01321 * eps ** 0.21226 + 0.19593 * eps ** 0.5)),
                  np.log(eps) / (2 * eps))
    Sn_eVcm2 = 8.462e-15 * Z1 * Z2 * M1 * sn / ((M1 + M2) * (Z1 ** 0.23 + Z2 ** 0.23))
    return Sn_eVcm2 * N_cm3 * 1e-7


def lss_electronic_stopping(Z1, M1, E_keV, Z2=Z_SI, N_cm3=N_SI_CM3, ke_scale=1.0):
    """LSS electronic stopping S_e = k sqrt(E), eV/nm (ke_scale=1 is textbook LSS)."""
    k = lss_k(Z1, M1, Z2) * ke_scale          # eV*A^2/sqrt(eV)
    S_eVA2 = k * np.sqrt(np.asarray(E_keV, float) * 1e3)
    return S_eVA2 * 1e-16 * N_cm3 * 1e-7


def crossover_energy_keV(Z1, M1, Z2=Z_SI, M2=M_SI, ke_scale=1.0):
    """Energy (keV) above which electronic stopping exceeds nuclear stopping."""
    from scipy.optimize import brentq
    f = lambda lg: np.log(lss_electronic_stopping(Z1, M1, 10 ** lg, Z2, ke_scale=ke_scale)
                          / zbl_nuclear_stopping(Z1, M1, 10 ** lg, Z2, M2))
    return 10 ** brentq(f, -3, 5)


def molecular_atom_energy(E_beam_keV, m_atom, m_molecule):
    """Kinetic energy (keV) carried by one atom of mass m_atom when a molecular
    ion of mass m_molecule and total energy E_beam_keV breaks up on impact."""
    return E_beam_keV * m_atom / m_molecule


def relative_atom_current(n_atoms, m_atom, m_ion):
    """Space-charge-limited dopant-atom current of a molecular beam relative to
    the monomer beam, both delivering the same per-atom energy.

    Transportable current at fixed perveance scales as I ~ (q/M)^0.5 V^1.5;
    the molecule must run at V = E_atom * M/m, so the atom current ratio is
    n * (M/m). BF2+: 49/11 = 4.45; As2+: 2 * 2 = 4."""
    return n_atoms * m_ion / m_atom
