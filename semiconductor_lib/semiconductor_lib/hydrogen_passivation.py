"""
Hydrogen passivation and depassivation of Si/SiO2 interface dangling bonds
(P_b centres) -- the chemistry behind the forming-gas (sinter) anneal.

Backs the Semiconductor Notes page "Forming Gas Anneal (Hydrogen Sinter)".

Kinetics (Brower and co-workers, EPR on (111) P_b centres):
  Passivation   H2 + P_b  -> HP_b + H   d[P_b]/dt = -k_f [H2] [P_b]
      k_f = k_f0 exp(-E_f/kT),  k_f0 = 1.94e-6 cm^3/s,  E_f = 1.66 eV
      (K. L. Brower, Phys. Rev. B 38, 9657, 1988)
  Dissociation  HP_b      -> P_b + H    d[HP_b]/dt = -k_d [HP_b]
      k_d = k_d0 exp(-E_d/kT),  k_d0 = 1.2e12 1/s,  E_d = 2.56 eV
      (K. L. Brower, Phys. Rev. B 42, 3444, 1990)
Both are first order in the defect, so with [H2] held fixed the unpassivated
fraction relaxes exponentially to the steady state
      f_ss = k_d / (k_d + k_f [H2]),   tau = 1 / (k_d + k_f [H2]).
Single activation energies are used; Stesmans showed the real (100)
interface has a spread of activation energies (Gaussian, ~0.1-0.2 eV wide),
which smears the onset of depassivation to somewhat lower temperatures.
[H2] is the molecular hydrogen concentration dissolved in the oxide at the
interface (cm^-3); it is not well known and is treated as a parameter.

Units: K, eV, s, cm^-3.
"""
import numpy as np

K_B_EV = 8.617333262e-5  # eV/K

KF0 = 1.94e-6   # cm^3/s
EF = 1.66       # eV
KD0 = 1.2e12    # 1/s
ED = 2.56       # eV


def c_to_k(T_c):
    return np.asarray(T_c, dtype=float) + 273.15


def k_passivation(T_K, kf0=KF0, Ef=EF):
    """Brower forward (passivation) rate constant k_f (cm^3/s)."""
    return kf0 * np.exp(-Ef / (K_B_EV * np.asarray(T_K, float)))


def k_dissociation(T_K, kd0=KD0, Ed=ED):
    """Brower dissociation rate constant k_d (1/s) of Si-H at the interface."""
    return kd0 * np.exp(-Ed / (K_B_EV * np.asarray(T_K, float)))


def steady_state_unpassivated(T_K, H2_cm3):
    """Steady-state fraction of interface dangling bonds left unpassivated."""
    kd = k_dissociation(T_K)
    kfH = k_passivation(T_K) * H2_cm3
    return kd / (kd + kfH)


def relaxation_time(T_K, H2_cm3):
    """Time constant (s) for approach to the steady state at fixed [H2]."""
    return 1.0 / (k_dissociation(T_K) + k_passivation(T_K) * H2_cm3)


def unpassivated_after(t_s, T_K, H2_cm3, f0=1.0):
    """Unpassivated fraction after time t_s at temperature T_K and [H2],
    starting from unpassivated fraction f0 (1 = fully depassivated)."""
    fss = steady_state_unpassivated(T_K, H2_cm3)
    tau = relaxation_time(T_K, H2_cm3)
    return fss + (f0 - fss) * np.exp(-np.asarray(t_s, float) / tau)


def depassivated_in_inert(t_s, T_K):
    """Fraction of initially passivated sites lost after time t_s in a
    hydrogen-free ambient (N2 or vacuum): 1 - exp(-k_d t)."""
    return 1.0 - np.exp(-k_dissociation(T_K) * np.asarray(t_s, float))
