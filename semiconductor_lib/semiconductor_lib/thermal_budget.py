"""
Thermal budget bookkeeping and fast-diffusing metal transport in silicon.

dt_sum              : sum of D_i*t_i over a sequence of isothermal steps.
equivalent_time     : the same budget expressed as time at a reference T.
profile_dt          : D*t integrated over an arbitrary T(t) profile (ramps).
metal_diffusivity   : interstitial Fe and Cu diffusivity (Arrhenius fits).
fe_solubility       : Fe solid solubility in Si.

Dopant constants: Plummer, Deal & Griffin, Silicon VLSI Technology (2000),
Table 7-5 intrinsic diffusivity (boron D0 = 0.76 cm^2/s, Ea = 3.46 eV).
Metal constants: Istratov & Weber reviews (Appl. Phys. A 1998/1999).
"""
import numpy as np

K_EV = 8.617333262e-5   # eV/K

DOPANTS = {   # intrinsic diffusivity D0 (cm^2/s), Ea (eV)
    "B": (0.76, 3.46),
    "P": (3.85, 3.66),
    "As": (0.066, 3.44),
}

METALS = {    # interstitial diffusivity D0 (cm^2/s), Ea (eV)
    "Fe": (1.0e-3, 0.67),    # Istratov, Hieslmair, Weber 1999
    "Cu": (3.0e-4, 0.18),    # Istratov, Weber 1998 (intrinsic Si)
}


def arrhenius(D0, Ea, T_C):
    return D0 * np.exp(-Ea / (K_EV * (np.asarray(T_C, float) + 273.15)))


def dt_sum(steps, species="B"):
    """steps = [(T_C, t_seconds), ...]; returns (total D*t in cm^2,
    list of per-step D*t)."""
    D0, Ea = DOPANTS[species] if isinstance(species, str) else species
    parts = [arrhenius(D0, Ea, T) * t for T, t in steps]
    return float(np.sum(parts)), parts


def equivalent_time(steps, T_ref_C, species="B"):
    """Time (s) at T_ref that gives the same D*t as the step list."""
    D0, Ea = DOPANTS[species] if isinstance(species, str) else species
    total, _ = dt_sum(steps, (D0, Ea))
    return total / arrhenius(D0, Ea, T_ref_C)


def profile_dt(t, T_C, species="B"):
    """D*t (cm^2) integrated over a temperature profile T_C(t) sampled at
    times t (s), trapezoidal rule. Captures ramp contributions."""
    D0, Ea = DOPANTS[species] if isinstance(species, str) else species
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(arrhenius(D0, Ea, T_C), t))


def diffusion_length(D, t):
    return np.sqrt(np.asarray(D) * np.asarray(t))


def metal_diffusivity(metal, T_C):
    D0, Ea = METALS[metal]
    return arrhenius(D0, Ea, T_C)


def fe_solubility(T_C):
    """Fe solid solubility in Si, cm^-3 (Istratov et al. 1999 fit,
    S = 8.4e25 exp(-2.86 eV/kT), valid ~800-1200 C)."""
    return 8.4e25 * np.exp(-2.86 / (K_EV * (np.asarray(T_C, float) + 273.15)))
