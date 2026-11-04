"""
Electrochemical corrosion kinetics for CMP and post-CMP cleaning.

Mixed-potential (Wagner-Traud / Evans-diagram) treatment of a corroding
metal: an anodic metal-dissolution reaction and a cathodic oxidant-
reduction reaction (H2O2 or dissolved O2) each follow Butler-Volmer
kinetics written with decadic Tafel slopes; the cathodic branch can be
capped by a mass-transport limiting current (Koutecky-Levich combination).
The corrosion potential is where net current is zero.  For a galvanic
couple the zero-current condition is written on total current, so the
area ratio of the two metals enters directly.

Also provided: the Stern-Geary relation that converts a measured
polarization resistance into corrosion current density (ASTM G59/G102),
and Faraday's-law conversion of current density into a penetration rate.

Units: potentials in V (any consistent reference), current densities in
A/cm^2, Tafel slopes in V/decade, areas in cm^2, density in g/cm^3,
molar mass in g/mol.  Rates returned in nm/min.

Kinetic parameters used in examples are illustrative, not fitted to any
particular slurry; real values must come from potentiodynamic data in
the actual chemistry, ideally under abrasion.
"""
import numpy as np
from scipy.optimize import brentq

F_FARADAY = 96485.33212  # C/mol


# ------------------------------------------------------------ kinetics
def anodic_current(E, E_eq, i0, beta_a):
    """Anodic (oxidation) partial current density, A/cm^2 (positive).
    Tafel form i0*10^((E-E_eq)/beta_a); the back reaction is neglected,
    valid once E is a few tens of mV above E_eq."""
    return i0 * 10.0 ** ((np.asarray(E, float) - E_eq) / beta_a)


def cathodic_current(E, E_eq, i0, beta_c, i_lim=None):
    """Cathodic (reduction) partial current density, A/cm^2, returned as a
    positive magnitude.  Activation-controlled Tafel branch
    i0*10^(-(E-E_eq)/beta_c), combined with a mass-transport limit i_lim
    as 1/i = 1/i_act + 1/i_lim when i_lim is given."""
    i_act = i0 * 10.0 ** (-(np.asarray(E, float) - E_eq) / beta_c)
    if i_lim is None:
        return i_act
    return 1.0 / (1.0 / i_act + 1.0 / i_lim)


def _net(E, m):
    """Net current density (anodic minus cathodic) on one metal dict m with
    keys Ea, i0a, ba (metal dissolution) and Ec, i0c, bc, ilim (oxidant
    reduction on that metal's surface)."""
    return (anodic_current(E, m["Ea"], m["i0a"], m["ba"])
            - cathodic_current(E, m["Ec"], m["i0c"], m["bc"], m.get("ilim")))


def corrosion_point(metal, E_lo=-1.5, E_hi=1.5):
    """Corrosion potential (V) and corrosion current density (A/cm^2) of a
    freely corroding metal: the root of net current = 0."""
    E = brentq(lambda e: _net(e, metal), E_lo, E_hi, xtol=1e-9)
    return E, float(anodic_current(E, metal["Ea"], metal["i0a"], metal["ba"]))


def galvanic_couple(m1, A1, m2, A2, E_lo=-1.5, E_hi=1.5):
    """Mixed potential of two short-circuited metals of areas A1, A2 (cm^2).

    Zero total current: A1*net1(E) + A2*net2(E) = 0.  Returns
    (E_mix, i_a1, i_a2): the couple potential and each metal's anodic
    (dissolution) current density at that potential.  The metal with the
    lower free corrosion potential is driven anodic; the smaller its area
    relative to its partner, the larger its dissolution current density."""
    f = lambda e: A1 * _net(e, m1) + A2 * _net(e, m2)
    E = brentq(f, E_lo, E_hi, xtol=1e-9)
    return (E,
            float(anodic_current(E, m1["Ea"], m1["i0a"], m1["ba"])),
            float(anodic_current(E, m2["Ea"], m2["i0a"], m2["ba"])))


# ---------------------------------------------------- measurement helpers
def stern_geary_B(beta_a, beta_c):
    """Stern-Geary constant B (V) from decadic Tafel slopes (V/decade):
    B = beta_a*beta_c / (2.303*(beta_a+beta_c)).  If the cathodic branch
    is fully diffusion-limited (beta_c -> infinity), B = beta_a/2.303."""
    if np.isinf(beta_c):
        return beta_a / np.log(10.0)
    return beta_a * beta_c / (np.log(10.0) * (beta_a + beta_c))


def icorr_from_rp(Rp_ohm_cm2, beta_a, beta_c):
    """Corrosion current density (A/cm^2) from polarization resistance
    (ohm cm^2, slope dE/di near E_corr, ASTM G59) via Stern-Geary."""
    return stern_geary_B(beta_a, beta_c) / Rp_ohm_cm2


def penetration_rate_nm_min(i_A_cm2, molar_mass=63.546, n=2, rho=8.96):
    """Uniform penetration rate (nm/min) for anodic current density i
    (A/cm^2) by Faraday's law: v = i*M/(n*F*rho).  Defaults are copper
    dissolving as Cu(2+) (M = 63.546 g/mol, rho = 8.96 g/cm^3)."""
    v_cm_s = np.asarray(i_A_cm2, float) * molar_mass / (n * F_FARADAY * rho)
    return v_cm_s * 1e7 * 60.0
