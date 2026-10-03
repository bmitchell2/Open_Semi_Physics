"""
Thermal radiation and optical absorption of silicon for radiative (RTP) heating.

Contents
--------
* Planck spectral radiance, Wien peak, and the fraction of blackbody power in a
  wavelength band (ideal blackbody, emissivity = 1).
* Room-temperature (300 K) absorption coefficient of float-zone silicon from the
  Franta et al. (2017) dispersion model, tabulated in the refractiveindex.info
  database (public domain, CC0). Sampled values are embedded below.
* Semi-empirical high-temperature absorption model of Timans (US 7,976,216 B2,
  Mattson Technology, 2011): band-edge (phonon-assisted) term plus intrinsic
  free-carrier term. Fitted to measured data at 1.152-2.3 um, 700-1100 C, for
  lightly doped silicon.

  IMPORTANT: the band-edge term is [135 (1.24/lam) + 0.07 T - 168]^2 only when
  the bracket is positive (photon energy above the temperature-dependent
  threshold). Squaring a negative bracket gives a spurious large absorption
  below the edge (e.g. ~2200 cm^-1 instead of ~70 cm^-1 at 2.3 um, 700 C).
  `timans_alpha_bg` clamps the bracket at zero. With the clamp, the model agrees
  with the patent's quartic fits to measured data within ~10-15 % inside the
  fitted ranges (checked in tests).

Units: wavelength in um, temperature in C for the Timans model (K for Planck),
absorption coefficient in cm^-1, depth in um.
"""
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

H = 6.62607015e-34       # J s
C0 = 2.99792458e8        # m/s
KB = 1.380649e-23        # J/K
WIEN_B_UM_K = 2897.771955  # um K


def planck_radiance(lam_um, T_K):
    """Blackbody spectral radiance, W m^-2 sr^-1 um^-1."""
    lam = np.asarray(lam_um, dtype=float) * 1e-6
    x = H * C0 / (lam * KB * T_K)
    return 2 * H * C0 ** 2 / lam ** 5 / np.expm1(x) * 1e-6


def wien_peak_um(T_K):
    """Wavelength of peak spectral radiance (per unit wavelength), um."""
    return WIEN_B_UM_K / np.asarray(T_K, dtype=float)


def band_fraction(lam1_um, lam2_um, T_K, n=20000):
    """Fraction of total blackbody power emitted between lam1 and lam2 (um)."""
    lam = np.linspace(lam1_um, lam2_um, n)
    band = _trapz(planck_radiance(lam, T_K), lam)
    sigma = 5.670374419e-8
    total = sigma * T_K ** 4 / np.pi           # W m^-2 sr^-1
    return band / total


def alpha_from_k(k, lam_um):
    """Absorption coefficient (cm^-1) from extinction coefficient k."""
    return 4 * np.pi * np.asarray(k) / (np.asarray(lam_um) * 1e-4)


def absorption_depth_um(alpha_cm):
    """1/e intensity absorption depth, um."""
    return 1e4 / np.asarray(alpha_cm, dtype=float)


# Franta et al. 2017 (refractiveindex.info, CC0), 300 K float-zone Si.
# (wavelength um, alpha cm^-1) computed from tabulated k via alpha = 4 pi k / lam.
SI_ALPHA_300K_FRANTA = np.array([
    (0.30, 1.727e6), (0.35, 1.054e6), (0.40, 1.010e5), (0.45, 2.867e4),
    (0.50, 1.342e4), (0.55, 7.680e3), (0.60, 4.364e3), (0.65, 2.593e3),
    (0.70, 1.695e3), (0.75, 1.232e3), (0.80, 9.555e2), (0.85, 6.748e2),
    (0.90, 3.748e2), (0.95, 1.719e2), (1.00, 6.113e1), (1.05, 1.671e1),
    (1.10, 3.880e0), (1.15, 7.53e-1), (1.20, 1.368e-2),
])


def si_alpha_300k(lam_um):
    """Log-interpolated 300 K Si absorption coefficient (cm^-1), 0.30-1.20 um."""
    lam_t, a_t = SI_ALPHA_300K_FRANTA.T
    lam = np.asarray(lam_um, dtype=float)
    if np.any(lam < lam_t[0]) or np.any(lam > lam_t[-1]):
        raise ValueError("si_alpha_300k valid for 0.30-1.20 um")
    return np.exp(np.interp(lam, lam_t, np.log(a_t)))


def timans_alpha_bg(lam_um, T_C):
    """Band-edge (phonon-assisted interband) absorption, cm^-1 (Timans model)."""
    b = 135.0 * (1.24 / np.asarray(lam_um, dtype=float)) + 0.07 * np.asarray(T_C, dtype=float) - 168.0
    return np.where(b > 0, b, 0.0) ** 2


def timans_alpha_fc(lam_um, T_C):
    """Intrinsic free-carrier absorption, cm^-1 (Timans model, ~lam^1.5)."""
    T_K = np.asarray(T_C, dtype=float) + 273.0
    return 8.2869e-6 * np.asarray(lam_um, dtype=float) ** 1.5 * T_K ** 3.1867 * np.exp(-7000.0 / T_K)


def timans_alpha(lam_um, T_C):
    """Total absorption coefficient, cm^-1. Fitted 1.1-2.5 um, 700-1100 C;
    the author suggests ~600-1200 C and 1.1-20 um as a physically based
    extrapolation range. Lightly doped Si only (no dopant free carriers)."""
    return timans_alpha_bg(lam_um, T_C) + timans_alpha_fc(lam_um, T_C)


# Quartic fits to the measured data (patent Table A), alpha = sum A_i T^i.
TIMANS_QUARTIC = {
    1.31: ([3.632433e4, -1.680142e2, 2.914564e-1, -2.270963e-4, 6.866667e-8], (725, 975)),
    1.54: ([4.002451e4, -1.718140e2, 2.789008e-1, -2.049313e-4, 5.848485e-8], (800, 1100)),
    2.30: ([-2.152712e4, 1.011942e2, -1.742175e-1, 1.282249e-4, -3.272494e-8], (700, 1100)),
}


def timans_measured_fit(lam_um, T_C):
    """Patent quartic fit to measured alpha (cm^-1) at 1.31, 1.54 or 2.3 um."""
    coeffs, _ = TIMANS_QUARTIC[lam_um]
    T = np.asarray(T_C, dtype=float)
    return sum(c * T ** i for i, c in enumerate(coeffs))
