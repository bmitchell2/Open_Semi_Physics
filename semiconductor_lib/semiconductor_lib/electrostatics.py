"""
Verified 1-D ideal MOS capacitor electrostatics.

Reference formulation: Kingston & Neustadter (1955) / Garrett & Brattain
(1955), as presented in Sze & Ng, "Physics of Semiconductor Devices,"
3rd ed., Ch. 6, and Nicollian & Brews, "MOS Physics and Technology."

Physics checks this module has been validated against (see
tests/test_electrostatics.py):
  - Numerical threshold voltage vs. the analytic ideal-device formula
    (matched to < 0.5% in the original derivation).
  - Correct capacitance ordering C_LF >= C_HF >= C_deep-depletion in
    inversion.
  - Cmin (deep-depletion / HF minimum) vs. the analytic max-depletion-
    width formula.

Convention (p-type substrate, e.g. NMOS body):
  psi_s > 0  -> bands bend down at surface -> holes repelled -> depletion/inversion
  psi_s < 0  -> bands bend up  at surface -> holes attracted -> accumulation
  Vg = psi_s - Qs(psi_s)/Cox   (ideal device, V_FB = 0, no oxide charge)

For an n-type substrate (e.g. the accumulation-mode varactor body), the
same functional form applies with N_A -> N_D and the OPPOSITE sign
convention: psi_s > 0 -> electron accumulation; psi_s < 0 -> depletion.
This is implemented as a separate explicit function
(semiconductor_charge_n) rather than a sign flag, to avoid convention
bugs -- a real bug of exactly this kind was caught and fixed during
verification (see CHANGELOG note in the docstring of
semiconductor_charge_n).
"""
import numpy as np
from scipy.optimize import brentq

from .constants import q, k_B, eps_si, eps_ox, ni300, thermal_voltage


def _G(u, r):
    """Dimensionless charge-squared function G(u,r); u = psi_s/Vt."""
    uc = np.clip(u, -300.0, 300.0)   # avoid float overflow far outside physical range
    return (np.exp(-uc) + u - 1.0) + r * (np.exp(uc) - u - 1.0)


def _G_majority_only(u):
    """Majority-carrier-only version (r -> 0): used for HF / deep-depletion
    differential response, where minority carriers cannot follow the AC or
    DC-transient signal."""
    uc = np.clip(u, -300.0, 300.0)
    return np.exp(-uc) + u - 1.0


def semiconductor_charge_p(psi_s, N_A, T=300.0, majority_only=False):
    """Qs (C/cm^2) induced in a p-type semiconductor for surface potential
    psi_s (V). majority_only=True drops the minority-carrier (electron)
    term -- used to build the frozen-inversion-charge (HF / deep-depletion)
    branch."""
    Vt = thermal_voltage(T)
    p_po = N_A
    n_po = ni300 ** 2 / N_A
    r = 0.0 if majority_only else n_po / p_po
    u = psi_s / Vt
    G = np.maximum(_G(u, r), 0.0)
    A = np.sqrt(2.0 * eps_si * k_B * T * N_A)
    return -np.sign(psi_s) * A * np.sqrt(G)


def semiconductor_charge_n(psi_s, N_D, T=300.0, majority_only=False):
    """Qs (C/cm^2) induced in an n-type semiconductor. Accumulation of
    electrons occurs at psi_s > 0 (opposite sign convention to p-type).
    Gauss's law requires Qs and psi_s to always carry OPPOSITE sign,
    exactly as for the p-type case -- this is not type-dependent."""
    Vt = thermal_voltage(T)
    n_no = N_D
    p_no = ni300 ** 2 / N_D
    r = 0.0 if majority_only else p_no / n_no
    u = -psi_s / Vt          # sign flip so that the SAME G(u,r) form applies
    G = np.maximum(_G(u, r), 0.0)
    A = np.sqrt(2.0 * eps_si * k_B * T * N_D)
    return -np.sign(psi_s) * A * np.sqrt(G)


def solve_psi_s(Vg_array, N, Cox, T=300.0, charge_fn=semiconductor_charge_p,
                 majority_only=False, psi_bracket=(-1.6, 1.6)):
    """Solve Vg = psi_s - Qs(psi_s)/Cox for psi_s at each Vg (ideal device, VFB=0)."""
    psi_out = np.zeros_like(Vg_array, dtype=float)
    lo, hi = psi_bracket

    def f(psi_s, Vg):
        return psi_s - charge_fn(psi_s, N, T, majority_only) / Cox - Vg

    for i, Vg in enumerate(Vg_array):
        a, b = lo, hi
        fa, fb = f(a, Vg), f(b, Vg)
        tries = 0
        while fa * fb > 0 and tries < 20:
            a -= 0.5
            b += 0.5
            fa, fb = f(a, Vg), f(b, Vg)
            tries += 1
        psi_out[i] = brentq(f, a, b, args=(Vg,), xtol=1e-13, rtol=1e-13)
    return psi_out


def differential_Cs(psi_s, N, T=300.0, charge_fn=semiconductor_charge_p,
                     majority_only=False, dpsi=1e-6):
    """Numerical dQs/dpsi_s -> semiconductor capacitance Cs = -dQs/dpsi_s (F/cm^2)."""
    Qp = charge_fn(psi_s + dpsi, N, T, majority_only)
    Qm = charge_fn(psi_s - dpsi, N, T, majority_only)
    return -(Qp - Qm) / (2 * dpsi)


def series_C(Cox, Cs):
    """Oxide and semiconductor capacitance in series."""
    return Cox * Cs / (Cox + Cs)


def threshold_voltage(N_A, Cox, T=300.0):
    """Analytic ideal-device (V_FB=0) threshold voltage for a p-type substrate."""
    Vt = thermal_voltage(T)
    phi_F = Vt * np.log(N_A / ni300)
    return 2 * phi_F + np.sqrt(2 * eps_si * q * N_A * (2 * phi_F)) / Cox


def max_depletion_width(N_A, T=300.0):
    """Analytic maximum depletion width at the onset of strong inversion."""
    Vt = thermal_voltage(T)
    phi_F = Vt * np.log(N_A / ni300)
    return np.sqrt(4 * eps_si * phi_F / (q * N_A))
