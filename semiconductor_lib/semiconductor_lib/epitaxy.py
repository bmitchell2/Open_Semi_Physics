"""
epitaxy.py -- strained-layer heteroepitaxy helpers (SiGe on Si(001)).

Functions
---------
sige_lattice_constant(x, bowing=True)
    Relaxed Si(1-x)Ge(x) lattice constant (nm). Dismukes et al. (1964)
    quadratic fit a = 0.54310 + 0.01992 x + 0.002733 x^2 nm, or linear Vegard.
misfit(x, ...)
    Fractional lattice mismatch f = (a_SiGe - a_Si) / a_Si.
matthews_blakeslee_hc(x, nu=0.28)
    Equilibrium (force-balance) critical thickness for 60-degree misfit
    dislocations on (001):  h = b(1 - nu cos^2 alpha) / (8 pi f (1+nu) cos lambda)
    * [ln(h/b) + 1], with cos^2 alpha = 1/4 and cos lambda = 1/2, i.e.
    h = b (1 - nu/4) / (4 pi f (1+nu)) * [ln(h/b) + 1].
people_bean_hc(x, nu=0.28)
    People-Bean (1985) energy-balance estimate, usually read as the practical
    metastable limit for low-temperature growth:
    h = (1-nu)/(1+nu) * b^2 / (16 pi sqrt(2) a) * ln(h/b) / f^2.
facet_angle_deg(hkl1, hkl2)
    Angle between two cubic crystal planes (deg); (111) vs (001) = 54.74 deg.

Verification (tests/test_epitaxy.py): Ge/Si mismatch 4.17 %; MB h_c(x=0.2)
about 12 nm and h_c(x=1) about 1 nm (literature 10-20 nm and ~1 nm);
People-Bean h_c(x=0.2) about 200 nm; both decrease monotonically with x and
PB > MB; facet angle reproduces arccos(1/sqrt(3)).
"""
import numpy as np

A_SI_NM = 0.54310
A_GE_NM = 0.56579


def sige_lattice_constant(x, bowing=True):
    x = np.asarray(x, dtype=float)
    if bowing:
        return 0.54310 + 0.01992 * x + 0.002733 * x ** 2
    return A_SI_NM + (A_GE_NM - A_SI_NM) * x


def misfit(x, bowing=True):
    return (sige_lattice_constant(x, bowing) - A_SI_NM) / A_SI_NM


def _solve_fixed_point(rhs, h0, iters=200):
    h = h0
    for _ in range(iters):
        h_new = rhs(h)
        if abs(h_new - h) < 1e-9 * max(h, 1e-12):
            return h_new
        h = h_new
    return h


def matthews_blakeslee_hc(x, nu=0.28, b_nm=None):
    """Equilibrium critical thickness (nm) of Si(1-x)Ge(x) on Si(001)."""
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(xs)
    for i, xi in enumerate(xs):
        f = misfit(xi)
        b = b_nm if b_nm is not None else sige_lattice_constant(xi) / np.sqrt(2)
        pref = b * (1 - nu / 4) / (4 * np.pi * f * (1 + nu))
        out[i] = _solve_fixed_point(lambda h: pref * (np.log(h / b) + 1), 10 * b)
    return out if np.ndim(x) else float(out[0])


def people_bean_hc(x, nu=0.28, b_nm=0.40, a_nm=A_SI_NM):
    """People-Bean energy-balance critical thickness (nm) on Si(001)."""
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(xs)
    for i, xi in enumerate(xs):
        f = misfit(xi)
        pref = (1 - nu) / (1 + nu) * b_nm ** 2 / (16 * np.pi * np.sqrt(2) * a_nm) / f ** 2
        out[i] = _solve_fixed_point(lambda h: pref * np.log(h / b_nm), 100 * b_nm)
    return out if np.ndim(x) else float(out[0])


def facet_angle_deg(hkl1=(1, 1, 1), hkl2=(0, 0, 1)):
    n1, n2 = np.asarray(hkl1, float), np.asarray(hkl2, float)
    c = abs(n1 @ n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))
