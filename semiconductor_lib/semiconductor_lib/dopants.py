"""
Silicon carrier statistics and shallow-dopant ionization.

Temperature models follow the Ioffe NSM Archive fits. The ionization solver
is self-consistent (charge neutrality + Fermi-Dirac occupancy of an
isolated dopant level) and is NON-DEGENERATE with a fixed ionization
energy: it is unreliable above roughly 1e18 cm^-3, where dopant levels
broaden into an impurity band and dopants are effectively fully ionized
(Hu, Sec. 1.9). Backs the "Fermi Level and Carrier Statistics" and
"Shallow Donors and Acceptors" notes.
"""
from collections import namedtuple

import numpy as np
from scipy.optimize import brentq

from .constants import k_B, q

KB_EV = k_B / q   # eV/K

Ionization = namedtuple("Ionization", "fraction n p E_F Ec_minus_EF")


def band_gap(T):
    """Eg(T) in eV, empirical fit 1.17 - 4.73e-4 T^2/(T+636) (Ioffe)."""
    return 1.17 - 4.73e-4 * T ** 2 / (T + 636.0)


def Nc(T):
    """Conduction-band effective density of states, cm^-3 (3.2e19 at 300 K)."""
    return 6.2e15 * T ** 1.5


def Nv(T):
    """Valence-band effective density of states, cm^-3 (1.8e19 at 300 K)."""
    return 3.5e15 * T ** 1.5


def intrinsic_concentration(T):
    """n_i(T) = sqrt(Nc Nv) exp(-Eg/2kT), cm^-3 (~9e9 at 300 K with these fits)."""
    return np.sqrt(Nc(T) * Nv(T)) * np.exp(-band_gap(T) / (2 * KB_EV * T))


def hydrogenic_binding_energy_meV(m_rel, eps_r):
    """Hydrogen-like donor/acceptor binding energy 13.6057 eV * m*/eps_r^2 (meV)."""
    return 13.6057 * m_rel / eps_r ** 2 * 1e3


def hydrogenic_radius_A(m_rel, eps_r):
    """Effective Bohr radius eps_r/m* * 0.529177 A."""
    return 0.529177 * eps_r / m_rel


def solve_donor(ND, T, Ed=0.045, g=2, NA=0.0):
    """
    Ionized fraction of donors. Energies in eV relative to Ev=0.
    Ed is Ec - E_donor (0.045 eV for phosphorus). Returns Ionization.
    """
    kT, ncT, nvT, eg = KB_EV * T, Nc(T), Nv(T), band_gap(T)

    def charge(EF):
        n = ncT * np.exp((EF - eg) / kT)
        p = nvT * np.exp(-EF / kT)
        ndp = ND / (1 + g * np.exp((EF - (eg - Ed)) / kT))
        return p + ndp - n - NA

    EF = brentq(charge, -0.2, eg + 0.2, xtol=1e-12)
    n = ncT * np.exp((EF - eg) / kT)
    p = nvT * np.exp(-EF / kT)
    ndp = ND / (1 + g * np.exp((EF - (eg - Ed)) / kT))
    return Ionization(ndp / ND, n, p, EF, eg - EF)


def solve_acceptor(NA, T, Ea=0.045, g=4, ND=0.0):
    """Ionized fraction of acceptors (Ea = E_acceptor - Ev; g=4 for silicon)."""
    kT, ncT, nvT, eg = KB_EV * T, Nc(T), Nv(T), band_gap(T)

    def charge(EF):
        n = ncT * np.exp((EF - eg) / kT)
        p = nvT * np.exp(-EF / kT)
        nam = NA / (1 + g * np.exp((Ea - EF) / kT))
        return p + ND - n - nam

    EF = brentq(charge, -0.2, eg + 0.2, xtol=1e-12)
    n = ncT * np.exp((EF - eg) / kT)
    p = nvT * np.exp(-EF / kT)
    nam = NA / (1 + g * np.exp((Ea - EF) / kT))
    return Ionization(nam / NA, n, p, EF, eg - EF)
