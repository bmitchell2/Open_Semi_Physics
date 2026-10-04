"""
Low-temperature plasma and sheath relations for plasma etch. Backs the
"Plasma Physics for Etch" and "Plasma Etch Sources" notes.

Standard results from Lieberman and Lichtenberg, Principles of Plasma
Discharges and Materials Processing, 2nd ed. (Wiley, 2005), ch. 2, 6, 11:
* Debye length  lambda_D = sqrt(eps0 Te / (e n_e))
* Bohm velocity u_B = sqrt(e Te / M)
* Floating wall: plasma-to-wall potential (Te/2)[1 + ln(M / (2 pi m_e))]
  (presheath Te/2 plus collisionless sheath (Te/2) ln(M/(2 pi m_e)))
* Child-law sheath thickness  s = (sqrt(2)/3) lambda_D (2 V0 / Te)^(3/4)
* Capacitive area ratio V_a / V_b = (A_b / A_a)^q, q = 4 in the ideal
  collisionless Child-law limit; measured q is typically 1 to 2.5.

Units: Te in eV, densities in cm^-3, lengths in cm, pressure in mTorr.
"""
import numpy as np

EPS0_SI = 8.8541878128e-12
E_SI = 1.602176634e-19
ME_SI = 9.1093837015e-31
AMU_SI = 1.66053906660e-27
KB_SI = 1.380649e-23


def debye_length(Te_eV, ne_cm3):
    """Electron Debye length (cm)."""
    ne = np.asarray(ne_cm3, dtype=float) * 1e6
    return np.sqrt(EPS0_SI * Te_eV / (E_SI * ne)) * 100.0


def bohm_velocity(Te_eV, M_amu):
    """Bohm (ion sound) velocity at the sheath edge (cm/s)."""
    return np.sqrt(E_SI * Te_eV / (M_amu * AMU_SI)) * 100.0


def floating_potential_drop(Te_eV, M_amu):
    """Plasma potential minus floating-wall potential (V)."""
    ratio = M_amu * AMU_SI / (2.0 * np.pi * ME_SI)
    return 0.5 * Te_eV * (1.0 + np.log(ratio))


def child_sheath_thickness(V0, Te_eV, ne_cm3):
    """Collisionless Child-law sheath thickness (cm) for sheath voltage V0."""
    lam = debye_length(Te_eV, ne_cm3)
    return np.sqrt(2.0) / 3.0 * lam * (2.0 * np.asarray(V0, dtype=float) / Te_eV) ** 0.75


def gas_density(p_mTorr, T_K=300.0):
    """Neutral gas density (cm^-3)."""
    p_Pa = np.asarray(p_mTorr, dtype=float) * 1e-3 * 133.322
    return p_Pa / (KB_SI * T_K) * 1e-6


def ion_mean_free_path(p_mTorr, sigma_cm2=5e-15, T_K=300.0):
    """Ion-neutral mean free path (cm). The default cross-section is a
    representative argon charge-exchange value (~5e-15 cm^2)."""
    return 1.0 / (gas_density(p_mTorr, T_K) * sigma_cm2)


def ion_angular_spread_deg(Ti_eV, V_sheath):
    """Characteristic ion arrival half-angle (deg) after collisionless
    acceleration through V_sheath, arctan(sqrt(Ti / V_sheath))."""
    return np.degrees(np.arctan(np.sqrt(Ti_eV / np.asarray(V_sheath, dtype=float))))


def area_ratio_voltage(A_small, A_large, q=4.0):
    """Ratio of sheath voltage at the small electrode to that at the large
    electrode, (A_large / A_small)^q."""
    return (A_large / A_small) ** q


def mean_ion_energy_collisionless(V_dc, V_plasma=0.0):
    """Mean ion energy (eV, singly charged) arriving at an RF-biased wafer
    in the low-frequency, collisionless limit: plasma potential minus the
    DC self-bias, V_p - V_dc (V_dc is negative)."""
    return V_plasma - V_dc
