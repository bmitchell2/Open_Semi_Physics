"""
Polarization-induced 2DEG in Ga-face AlxGa1-xN/GaN heterostructures.

Backs the Semiconductor Notes page "AlGaN/GaN HEMT (Polarization-Induced
2DEG and Normally-Off p-GaN Gate)".

Model (Ambacher et al., J. Appl. Phys. 87, 334 (2000), Eqs. and Table I):
  P_SP(x)  = -0.052 x - 0.029                               C/m^2
  a(x)     = -0.077 x + 3.189                               Angstrom
  e31(x)   = -0.11 x - 0.49 ; e33(x) = 0.73 x + 0.73        C/m^2
  c13(x)   = 5 x + 103      ; c33(x) = -32 x + 405          GPa
  P_PE(x)  = 2 (a0 - a)/a * (e31 - e33 c13/c33)             C/m^2
  sigma(x) = |P_SP(x) + P_PE(x) - P_SP(0)|                  C/m^2
  eps_r(x) = -0.5 x + 9.5
  e*phi_b  = 1.3 x + 0.84 eV  (Schottky/surface barrier)
  Eg(x)    = 6.13 x + 3.42 (1-x) - 1.0 x (1-x) eV ; dEc = 0.7 (Eg(x)-Eg(0))
  n_s = sigma/q - eps0 eps_r/(d q^2) * (q phi_b + E_F - dEc)
  E_F = E0 + pi hbar^2 n_s / m*, E0 = (9 pi hbar q^2 n_s / (8 eps0 eps_r sqrt(8 m*)))^(2/3)
with m* = 0.22 m0. Solved self-consistently by bisection.

Validity: pseudomorphic (coherently strained) Ga-face barrier, undoped,
no gate dielectric, surface pinned at phi_b. Real surfaces, passivation,
and partial relaxation at high x shift n_s; values are model results,
not measurements. Units: SI internally; n_s returned in cm^-2.
"""
import numpy as np

q = 1.602176634e-19
eps0 = 8.8541878128e-12          # F/m (SI here)
hbar = 1.054571817e-34
m0 = 9.1093837015e-31


def p_sp(x):
    """Spontaneous polarization of AlxGa1-xN, C/m^2."""
    return -0.052 * x - 0.029


def p_pe(x):
    """Piezoelectric polarization of AlxGa1-xN strained to GaN, C/m^2."""
    a0, a = 3.189, -0.077 * x + 3.189
    e31, e33 = -0.11 * x - 0.49, 0.73 * x + 0.73
    c13, c33 = 5 * x + 103, -32 * x + 405
    return 2 * (a0 - a) / a * (e31 - e33 * c13 / c33)


def sigma_pol(x):
    """Net bound polarization sheet charge at the interface, C/m^2 (positive)."""
    return np.abs(p_sp(x) + p_pe(x) - p_sp(0.0))


def _params(x):
    eps_r = -0.5 * x + 9.5
    phi_b = 1.3 * x + 0.84
    Eg = lambda y: 6.13 * y + 3.42 * (1 - y) - 1.0 * y * (1 - y)
    dEc = 0.7 * (Eg(x) - Eg(0.0))
    return eps_r, phi_b, dEc


def fermi_level(ns_m2, eps_r, m_eff=0.22):
    """E_F above conduction-band edge at interface (eV) for sheet density ns (m^-2)."""
    m = m_eff * m0
    E0 = (9 * np.pi * hbar * q**2 * ns_m2 / (8 * eps0 * eps_r * np.sqrt(8 * m)))**(2 / 3)
    return (E0 + np.pi * hbar**2 * ns_m2 / m) / q


def sheet_density(x, d_nm, m_eff=0.22):
    """Self-consistent 2DEG density (cm^-2) for Al fraction x and barrier thickness d (nm).
    Returns 0 below the critical thickness."""
    eps_r, phi_b, dEc = _params(x)
    d = d_nm * 1e-9
    sig = sigma_pol(x)

    def f(ns):
        return ns - (sig / q - eps0 * eps_r / (d * q) * (phi_b + fermi_level(ns, eps_r, m_eff) - dEc))

    if f(0.0) >= 0:   # no positive root: barrier too thin
        return 0.0
    lo, hi = 0.0, sig / q
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi) * 1e-4


def critical_thickness(x):
    """Barrier thickness (nm) below which no 2DEG forms (n_s = 0), from
    d_cr = eps0 eps_r (phi_b - dEc) / (q sigma)."""
    eps_r, phi_b, dEc = _params(x)
    return eps0 * eps_r * (phi_b - dEc) / sigma_pol(x) * 1e9
