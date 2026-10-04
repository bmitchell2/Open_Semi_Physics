"""
Optical and EUV lithography relations. Backs the "Optical Lithography",
"Photoresist Processing", "Photoresist as an Implant Mask" and EUV notes.

Contents
--------
* Rayleigh resolution and depth of focus; the k1 = 0.25 two-beam limit.
* Coherent aerial image of a binary 1:1 line/space grating from a sum of
  the diffraction orders the pupil accepts, for on-axis or tilted (dipole)
  illumination. Scalar, thin-mask (Kirchhoff) model: it shows which orders
  form the image, not resist-level CD.
* Photon counting and Poisson shot noise for a given dose and wavelength.
* Thin-film interference periods (swing curve, interferometric endpoint).
* Gaussian implant-mask transmission and the resist thickness needed.
* Tanaka capillary-collapse stress for resist lines.

Units: lengths in nm unless stated, dose in mJ/cm^2.
References: Mack, Fundamental Principles of Optical Lithography (2007);
Levinson, Principles of Lithography, 4th ed.; Tanaka, Morigami and Atoda,
Jpn. J. Appl. Phys. 32, 6059 (1993).
"""
import numpy as np
from scipy.special import erfc, erfcinv

HC_EV_NM = 1239.841984     # h c in eV nm
E_CHARGE = 1.602176634e-19


def photon_energy_eV(lam_nm):
    """Photon energy (eV) at vacuum wavelength lam_nm."""
    return HC_EV_NM / np.asarray(lam_nm, dtype=float)


def rayleigh_resolution(k1, lam_nm, NA):
    """Printable half-pitch R = k1 lambda / NA (nm)."""
    return k1 * lam_nm / NA


def depth_of_focus(k2, lam_nm, NA):
    """Depth of focus DOF = k2 lambda / NA^2 (nm)."""
    return k2 * lam_nm / NA ** 2


def min_half_pitch(lam_nm, NA):
    """Absolute single-exposure half-pitch limit, lambda/(4 NA) (k1 = 0.25):
    zero and first orders at opposite edges of the pupil."""
    return lam_nm / (4.0 * NA)


def min_pitch(lam_nm, NA, sigma=0.0):
    """Smallest pitch that passes two orders, lambda / (NA (1 + sigma)).
    sigma = 0 is coherent on-axis illumination (pitch lambda/NA);
    sigma -> 1 approaches the lambda/(2 NA) two-beam limit."""
    return lam_nm / (NA * (1.0 + sigma))


def grating_orders(pitch_nm, lam_nm, NA, sin_illum=0.0, m_max=5):
    """Diffraction orders m whose direction sine (sin_illum + m lam/p) falls
    inside the pupil (|.| <= NA)."""
    m = np.arange(-m_max, m_max + 1)
    s = sin_illum + m * lam_nm / pitch_nm
    return m[np.abs(s) <= NA + 1e-12]


def aerial_image_grating(x_nm, pitch_nm, lam_nm, NA, sin_illum=0.0,
                         duty=0.5):
    """Coherent aerial-image intensity of a binary grating (clear fraction
    `duty`) for one illumination direction. Normalized so a clear field
    gives 1. For symmetric dipole illumination average the +/- sin_illum
    results (incoherent sum)."""
    x = np.asarray(x_nm, dtype=float)
    amp = np.zeros_like(x, dtype=complex)
    for m in grating_orders(pitch_nm, lam_nm, NA, sin_illum):
        c = duty * np.sinc(m * duty)          # Fourier coefficient of the mask
        amp += c * np.exp(2j * np.pi * m * x / pitch_nm)
    return np.abs(amp) ** 2


def image_contrast(I):
    """Michelson contrast (Imax - Imin)/(Imax + Imin)."""
    I = np.asarray(I)
    return (I.max() - I.min()) / (I.max() + I.min() + 1e-30)


def photons_per_area(dose_mJ_cm2, lam_nm, area_nm2):
    """Mean number of incident photons in `area_nm2` at the given dose."""
    E_ph = photon_energy_eV(lam_nm) * E_CHARGE           # J
    dose_J_nm2 = dose_mJ_cm2 * 1e-3 / 1e14              # J per nm^2
    return dose_J_nm2 * area_nm2 / E_ph


def shot_noise(N):
    """Relative Poisson fluctuation 1/sqrt(N)."""
    return 1.0 / np.sqrt(np.asarray(N, dtype=float))


def interference_period(lam_nm, n):
    """Thickness change per interference fringe, lambda/(2n) (nm): the
    standing-wave / swing-curve period in resist, and the film loss per
    fringe in laser-interferometric etch endpoint."""
    return lam_nm / (2.0 * n)


def quarter_wave_arc(n_resist, lam_nm):
    """Ideal top antireflective coating: index sqrt(n_resist), thickness
    lambda/(4 n_arc). Returns (n_arc, t_arc_nm)."""
    n_arc = np.sqrt(n_resist)
    return n_arc, lam_nm / (4.0 * n_arc)


def mask_transmission(T, Rp, dRp):
    """Fraction of a Gaussian implant profile (range Rp, straggle dRp) that
    penetrates a mask of thickness T (same length units)."""
    return 0.5 * erfc((T - Rp) / (np.sqrt(2.0) * dRp))


def mask_thickness_for(fraction, Rp, dRp):
    """Mask thickness that transmits only `fraction` of the implant."""
    return Rp + np.sqrt(2.0) * dRp * erfcinv(2.0 * fraction)


def tanaka_collapse_stress(gamma_N_m, theta_deg, space, height, width):
    """Maximum bending stress (Pa) at the base of a resist line pulled by
    the rinse meniscus (Tanaka et al. 1993):
        sigma = (6 gamma cos(theta) / S) (H / W)^2
    gamma: surface tension (N/m); space S, height H, width W in metres."""
    return 6.0 * gamma_N_m * np.cos(np.radians(theta_deg)) / space * (height / width) ** 2


def proximity_resolution(lam_nm, gap_nm):
    """Order-of-magnitude Fresnel-limited proximity-print resolution,
    sqrt(lambda g) (nm)."""
    return np.sqrt(lam_nm * gap_nm)
