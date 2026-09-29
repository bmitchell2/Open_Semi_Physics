"""
Mobility-limiting scattering mechanisms: phonon (lattice) scattering and
ionized-impurity (Coulomb) scattering, combined by Matthiessen's rule.
Backs the "Ionized Impurity Scattering" and "Phonons" notes.

caughey_thomas() implements the verified Si low-field mobility fit of
Caughey, D.M. and Thomas, R.E., "Carrier Mobilities in Silicon Empirically
Related to Doping and Field," Proc. IEEE 55, 2192 (1967), with the
temperature-dependent parameter set as documented by the Tidy3D device
simulator (docs.flexcompute.com, CaugheyThomasMobility, retrieved 2026-09).
This is an empirical fit to the TOTAL (lattice + impurity) mobility; it
does not separate the two mechanisms on its own. lattice_mobility() is a
second, independently documented empirical power law for the phonon-only
component (nextnano "constant mobility model" / Palankovski parameter
database, mu_max=1417 cm^2/Vs, exponent 2.5, electrons, 300 K reference).
Combining the two via Matthiessen's rule at a single reference temperature
(matthiessen_impurity_component()) gives a reasonable 300 K impurity-only
estimate; see the module docstring warning in the note itself about
extrapolating this decomposition away from 300 K.
"""
import numpy as np

# Caughey-Thomas parameters for silicon at 300 K, with their own
# temperature-scaling exponents (Tidy3D documentation of the 1967 fit).
CT_ELECTRONS = dict(mu_min=52.2, mu_max=1471.0, ref_N=9.68e16, exp_N=0.68,
                     exp_1=-0.57, exp_2=-2.33, exp_3=2.4, exp_4=-0.146)
CT_HOLES = dict(mu_min=44.9, mu_max=470.5, ref_N=2.23e17, exp_N=0.719,
                 exp_1=-0.57, exp_2=-2.33, exp_3=2.4, exp_4=-0.146)

# Lattice(phonon)-only mobility at 300 K and its power-law exponent
# (nextnano "constant mobility model" database; Palankovski parameter set).
MU_LATTICE_300_ELECTRONS = 1417.0
MU_LATTICE_EXPONENT_ELECTRONS = 2.5
MU_LATTICE_300_HOLES = 470.5
MU_LATTICE_EXPONENT_HOLES = 2.2


def caughey_thomas(N, T, params):
    """
    Empirical low-field mobility (cm^2/V s), Caughey & Thomas (1967) with
    the temperature-dependent parameter set above. N is total ionized
    doping (cm^-3), T in K. Valid near and somewhat above room temperature;
    not validated for cryogenic T (freeze-out and other mechanisms take
    over below roughly 100-150 K and are not in this fit).
    """
    t = np.asarray(T, dtype=float) / 300.0
    N = np.asarray(N, dtype=float)
    mu_max_T = params["mu_max"] * t ** params["exp_2"]
    mu_min_T = params["mu_min"] * t ** params["exp_1"]
    ref_N_T = params["ref_N"] * t ** params["exp_3"]
    exp_N_T = params["exp_N"] * t ** params["exp_4"]
    return (mu_max_T - mu_min_T) / (1.0 + (N / ref_N_T) ** exp_N_T) + mu_min_T


def lattice_mobility(T, mu300=MU_LATTICE_300_ELECTRONS, exponent=MU_LATTICE_EXPONENT_ELECTRONS):
    """Phonon(lattice)-only mobility power law, mu300 (T/300)^-exponent, cm^2/V s."""
    return mu300 * (np.asarray(T, dtype=float) / 300.0) ** (-exponent)


def matthiessen_combine(*mobilities):
    """Combine independent scattering-limited mobilities: 1/mu = sum(1/mu_i)."""
    inv = sum(1.0 / np.asarray(m, dtype=float) for m in mobilities)
    return 1.0 / inv


def matthiessen_impurity_component(mu_total_300, mu_lattice_300=MU_LATTICE_300_ELECTRONS):
    """
    Back out the implied 300 K impurity-limited mobility from a total
    mobility (e.g. from caughey_thomas()) and a separately documented
    lattice-only mobility, via Matthiessen's rule: 1/mu_I = 1/mu_tot -
    1/mu_L. Returns NaN if mu_tot >= mu_L (the two source fits are not
    perfectly self-consistent at very light doping; only meaningful once
    mu_tot is measurably below mu_lattice_300).
    """
    mu_total_300 = np.asarray(mu_total_300, dtype=float)
    inv_I = 1.0 / mu_total_300 - 1.0 / mu_lattice_300
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(inv_I > 0, 1.0 / np.where(inv_I > 0, inv_I, np.nan), np.nan)


def impurity_mobility_idealized(T, mu_I_300, exponent=1.5):
    """
    Idealized (weak-screening, non-degenerate) temperature scaling of
    ionized-impurity-limited mobility, mu_I_300 (T/300)^exponent, cm^2/V s.
    This is the textbook Conwell-Weisskopf/Brooks-Herring asymptotic power
    law (screening term varies only logarithmically with T); it is an
    approximation used here for illustrating the competition with phonon
    scattering, not a validated quantitative model at cryogenic T (it
    omits dopant freeze-out and the T-dependence of screening itself).
    """
    return mu_I_300 * (np.asarray(T, dtype=float) / 300.0) ** exponent
