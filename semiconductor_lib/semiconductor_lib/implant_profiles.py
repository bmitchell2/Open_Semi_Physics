"""
Implanted-dopant depth profiles: moments, Pearson IV profiles and the
analytic LSS projected-range estimate.

- profile_moments: Rp (mean), dRp (std), skewness gamma, kurtosis beta of a
  depth profile (the four quantities tabulated for analytic implant models).
- pearson4_profile: dose-normalized Pearson IV profile with given moments,
  built by integrating Pearson's differential equation
  d ln f/ds = (s - a)/(b0 + a s + b2 s^2), s = x - Rp. Reduces to a Gaussian
  for gamma = 0, beta = 3. Valid Pearson IV region: beta > (48 + 39 g^2 +
  6 (g^2+4)^1.5)/(32 - g^2): 3.0 at g=0, ~3.3 at |g|=0.4, ~4.7 at |g|=0.94.
- lss_projected_range: analytic LSS chain (ZBL reduced energy, ZBL nuclear
  stopping, LSS electronic stopping k*sqrt(eps)), path length R, Lindhard
  projection Rp = R/(1 + M2/(3 M1)) and dRp ~ (2/3) Rp sqrt(M1 M2)/(M1+M2).
  Verified against Gibbons-type tables: overestimates Rp by ~10-40% (worst
  for B and at low energy); dRp estimate is fair for P/As/Sb, high for B.

Units: depth in nm, energy in keV, dose in cm^-2, concentration in cm^-3.
"""
import numpy as np
from scipy.integrate import quad

_trapz = getattr(np, "trapezoid", None) or np.trapz

A0_NM = 0.0529177
N_SI_NM3 = 49.94          # Si atoms per nm^3
Z_SI, M_SI = 14, 28.086


def profile_moments(x, c):
    """Rp, dRp, skewness, kurtosis (non-excess) of a profile c(x)."""
    x = np.asarray(x, float)
    w = np.clip(np.asarray(c, float), 0, None)
    w = w / w.sum()
    m = np.sum(x * w)
    s = np.sqrt(np.sum((x - m) ** 2 * w))
    g = np.sum((x - m) ** 3 * w) / s ** 3
    b = np.sum((x - m) ** 4 * w) / s ** 4
    return m, s, g, b


def pearson4_beta_min(gamma):
    """Lower kurtosis bound for a Pearson IV (Type IV) distribution."""
    g2 = gamma ** 2
    return (48 + 39 * g2 + 6 * (g2 + 4) ** 1.5) / (32 - g2)


def pearson4_profile(x_nm, dose_cm2, Rp_nm, dRp_nm, gamma, beta):
    """Dose-normalized Pearson IV concentration (cm^-3) on grid x_nm.
    The grid should extend several dRp beyond the profile on both sides."""
    x = np.asarray(x_nm, float)
    if abs(gamma) < 1e-9 and abs(beta - 3) < 1e-9:      # Gaussian limit
        f = np.exp(-0.5 * ((x - Rp_nm) / dRp_nm) ** 2)
        return dose_cm2 * f / (_trapz(f, x) * 1e-7)
    if beta <= pearson4_beta_min(gamma):
        raise ValueError("beta below Pearson IV bound for this gamma")
    s = x - Rp_nm
    A = 10 * beta - 12 * gamma ** 2 - 18
    a = -gamma * dRp_nm * (beta + 3) / A
    b0 = -dRp_nm ** 2 * (4 * beta - 3 * gamma ** 2) / A
    b2 = -(2 * beta - 3 * gamma ** 2 - 6) / A
    q = 4 * b0 * b2 - a ** 2                       # > 0 for Type IV
    sq = np.sqrt(q)
    # closed-form integral of (s - a)/(b0 + a s + b2 s^2)
    lnf = (np.log(np.abs(b2 * s ** 2 + a * s + b0)) / (2 * b2)
           - (a + a / (2 * b2)) * 2 / sq * np.arctan((2 * b2 * s + a) / sq))
    f = np.exp(lnf - lnf.max())
    area_cm = _trapz(f, x) * 1e-7
    return dose_cm2 * f / area_cm


def gaussian_peak(dose_cm2, dRp_nm):
    """Peak concentration (cm^-3) of a Gaussian implant: Q/(sqrt(2 pi) dRp)."""
    return dose_cm2 / (np.sqrt(2 * np.pi) * dRp_nm * 1e-7)


def _zbl_sn(eps):
    if eps <= 30:
        return np.log(1 + 1.1383 * eps) / (2 * (eps + 0.01321 * eps ** 0.21226
                                                + 0.19593 * eps ** 0.5))
    return np.log(eps) / (2 * eps)


def lss_projected_range(Z1, M1, E_keV, Z2=Z_SI, M2=M_SI, N_nm3=N_SI_NM3):
    """Analytic LSS estimate. Returns (R, Rp, dRp) in nm."""
    a = 0.8854 * A0_NM / (Z1 ** 0.23 + Z2 ** 0.23)
    eps = 32.53 * M2 * E_keV / (Z1 * Z2 * (M1 + M2) * (Z1 ** 0.23 + Z2 ** 0.23))
    k = (0.0793 * Z1 ** (2 / 3) * Z2 ** 0.5 * (M1 + M2) ** 1.5
         / ((Z1 ** (2 / 3) + Z2 ** (2 / 3)) ** 0.75 * M1 ** 1.5 * M2 ** 0.5))
    rho = quad(lambda e: 1 / (_zbl_sn(e) + k * np.sqrt(e)), 0, eps, limit=200)[0]
    R = rho / (N_nm3 * 4 * np.pi * a ** 2 * M1 * M2 / (M1 + M2) ** 2)
    Rp = R / (1 + M2 / (3 * M1))
    dRp = 2 / 3 * Rp * np.sqrt(M1 * M2) / (M1 + M2)
    return R, Rp, dRp
