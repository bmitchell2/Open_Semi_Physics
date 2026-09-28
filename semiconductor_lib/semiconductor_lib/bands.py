"""
Effective-mass and band-curvature helpers for silicon. Backs the "Effective
Mass" and "Band Structure" notes.

Masses are in units of the free-electron mass m0. k in nm^-1, energy in eV.
The isotropic parabolic hole model is an approximation: real hole bands are
warped (Dresselhaus-Kip-Kittel), so the heavy/light masses are averages.
"""
import numpy as np

from .constants import q

HBAR = 1.054571817e-34        # J s
M0 = 9.1093837015e-31         # kg
HBAR2_2M0 = HBAR ** 2 / (2 * M0) / q * 1e18   # eV nm^2 (0.0381)

ML_SI = 0.98                  # longitudinal electron mass (Ioffe; DKK 0.97, TU Wien 0.916)
MT_SI = 0.19                  # transverse electron mass
M_HEAVY_HOLE = 0.49
M_LIGHT_HOLE = 0.16
M_SPLIT_OFF = 0.24
SPLIT_OFF_EV = 0.044          # split-off band lies this far below the valence-band top


def conductivity_mass(ml=ML_SI, mt=MT_SI):
    """Electron conductivity mass for six equivalent ellipsoidal valleys, 3/(1/ml + 2/mt)."""
    return 3.0 / (1.0 / ml + 2.0 / mt)


def dos_mass(ml=ML_SI, mt=MT_SI, valleys=6):
    """Electron density-of-states mass, valleys^(2/3) (ml mt^2)^(1/3)."""
    return valleys ** (2.0 / 3.0) * (ml * mt * mt) ** (1.0 / 3.0)


def hole_dos_mass_two_band(mhh=M_HEAVY_HOLE, mlh=M_LIGHT_HOLE):
    """Two-band hole DOS mass (neglects warping and the split-off band; ~0.55, below Ioffe's 0.81)."""
    return (mhh ** 1.5 + mlh ** 1.5) ** (2.0 / 3.0)


def hole_conductivity_mass_two_band(mhh=M_HEAVY_HOLE, mlh=M_LIGHT_HOLE):
    """Two-band hole conductivity mass (~0.37; Hu lists 0.39)."""
    return (mhh ** 1.5 + mlh ** 1.5) / (mhh ** 0.5 + mlh ** 0.5)


def parabolic_band(k_nm, m_rel):
    """E(k) = hbar^2 k^2 / (2 m*) in eV for k in nm^-1."""
    return HBAR2_2M0 * np.asarray(k_nm) ** 2 / m_rel


def curvature_mass(E_eV, k_nm):
    """
    Effective mass (units of m0) from the numerical curvature at the centre of
    a sampled E(k): m*/m0 = (hbar^2/m0) / (d2E/dk2). Use a fine k grid
    symmetric about the band extremum.
    """
    d2 = np.gradient(np.gradient(E_eV, k_nm), k_nm)[len(k_nm) // 2]
    return 2 * HBAR2_2M0 / d2


def tight_binding_chain(k, t=1.0, a=1.0):
    """1-D tight-binding band E = -2 t cos(k a); curvature at k=0 gives m* = hbar^2/(2 t a^2)."""
    return -2.0 * t * np.cos(np.asarray(k) * a)


def cyclotron_field_T(f_hz, m_rel):
    """Resonance field B = 2 pi f m*/q (tesla) for cyclotron frequency f."""
    return 2 * np.pi * f_hz * m_rel * M0 / q
