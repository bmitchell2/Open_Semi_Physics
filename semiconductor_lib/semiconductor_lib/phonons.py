"""
Lattice-vibration (phonon) helpers. Backs the "Phonons" note.

diatomic_chain_dispersion() is the exact classical-mechanics result for a
1-D chain with two different masses per unit cell and a single
nearest-neighbour spring constant (e.g. Kittel, "Introduction to Solid
State Physics"; Ashcroft & Mermin, "Solid State Physics", ch. 22). It is
used here as a generic, illustrative two-branch (acoustic/optical) model
in arbitrary units, NOT a fit to silicon's measured phonon dispersion
(which needs real interatomic force constants from inelastic neutron
scattering and is not attempted here).
"""
import numpy as np

HBAR_EV_S = 6.582119569e-16   # reduced Planck constant, eV s
KB_EV = 8.617333262e-5        # Boltzmann constant, eV/K


def diatomic_chain_dispersion(k_a, C=1.0, m1=1.0, m2=2.0):
    """
    Angular frequency branches (in units of sqrt(C/m1)) of a 1-D diatomic
    chain with spring constant C and masses m1, m2, versus k_a = k*a where
    a is the unit-cell repeat distance (2x the atomic spacing). Returns
    (omega_acoustic, omega_optical), both >= 0, same shape as k_a.

    omega^2 = (C/(m1 m2)) [ (m1+m2) -+ sqrt((m1+m2)^2 - 4 m1 m2 sin^2(k_a/2)) ]
    (- sign: acoustic branch; + sign: optical branch.)
    """
    k_a = np.asarray(k_a, dtype=float)
    s2 = np.sin(k_a / 2.0) ** 2
    disc = np.sqrt(np.clip((m1 + m2) ** 2 - 4.0 * m1 * m2 * s2, 0.0, None))
    omega_acoustic = np.sqrt(np.clip((C / (m1 * m2)) * ((m1 + m2) - disc), 0.0, None))
    omega_optical = np.sqrt(np.clip((C / (m1 * m2)) * ((m1 + m2) + disc), 0.0, None))
    return omega_acoustic, omega_optical


def bose_einstein(E_phonon_eV, T):
    """Phonon occupation number 1/(exp(E/kT)-1); E in eV, T in K. Diverges as E,T->0 together."""
    E = np.asarray(E_phonon_eV, dtype=float)
    x = E / (KB_EV * np.asarray(T, dtype=float))
    return 1.0 / (np.expm1(x))
