"""Parallel-plate dielectric relations: bound (polarization) charge, field
screening, and the fixed-charge versus fixed-voltage pictures.

All quantities per unit area in the library cm convention (cm, V, C/cm^2, F/cm^2, J/cm^2;
eps0 from constants is in F/cm).
Linear, isotropic, lossless dielectric assumed (P = eps0*(k-1)*E).
"""
from .constants import eps0, q

def bound_charge_fraction(k):
    """Fraction of the free plate charge cancelled by the bound surface charge:
    sigma_b / sigma_f = 1 - 1/k."""
    return 1.0 - 1.0 / k

def field_fixed_charge(sigma_f, k):
    """Field inside the dielectric for free surface charge sigma_f (C/cm^2):
    E = sigma_f / (k*eps0). The k-fold reduction from vacuum is the screening."""
    return sigma_f / (k * eps0)

def voltage_fixed_charge(sigma_f, k, t):
    """Voltage across thickness t for fixed free charge: V = sigma_f*t/(k*eps0)."""
    return field_fixed_charge(sigma_f, k) * t

def charge_fixed_voltage(V, k, t):
    """Free plate charge per area held by a source at voltage V: sigma = k*eps0*V/t."""
    return k * eps0 * V / t

def field_fixed_voltage(V, t):
    """With the voltage held fixed, the field is V/t independent of k."""
    return V / t

def polarization(E, k):
    """Polarization (dipole moment per volume, C/cm^2) = bound surface charge density:
    P = eps0*(k-1)*E."""
    return eps0 * (k - 1.0) * E

def stored_energy_per_area(V, k, t):
    """Energy per area 0.5*C*V^2 = 0.5*k*eps0*V^2/t (J/cm^2)."""
    return 0.5 * k * eps0 * V**2 / t

def electrons_per_cm2(sigma):
    """Convert a surface charge (C/cm^2) to electrons per cm^2."""
    return sigma / q
