"""
Generation-recombination lifetime models: SRH doping dependence, Zerbst
transient analysis, and DLTS (Arrhenius trap-energy extraction).

Physics checks these functions have been validated against (see
tests/test_lifetime.py):
  - SRH doping-dependence log-log slope approaches -1 above N_ref
    (matches literature Si/Ge lifetime studies).
  - Zerbst relaxation ODE recovers input tau_g and s0 to < 0.1% when
    fit against its own simulated transient.
  - DLTS Arrhenius fit recovers input E_a and sigma to < 0.01% when
    fit against its own simulated transient.

These are illustrative/phenomenological models for teaching and
figure-generation, calibrated to reproduce literature-reported
qualitative behavior -- not fits to specific measured process data.
"""
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit

from .constants import q, k_B, ni300


def srh_lifetime(N, tau0=1e-3, N_ref=7e15):
    """Phenomenological SRH generation lifetime vs. doping N (cm^-3).
    tau_g = tau0 / (1 + N/N_ref). Constant (defect-limited) at low doping,
    falls off close to 1/N above N_ref (literature onset ~7e15 cm^-3 for Si)."""
    return tau0 / (1.0 + N / N_ref)


def zerbst_transient(t, N_A, Cox, W_eq, W_dd0, tau_g, s0):
    """Simulate the Zerbst C-t relaxation after a deep-depletion pulse.
    Returns C(t) (F/cm^2). Generation drives the excess depletion width
    (W - W_eq) back to zero; net generation vanishes at equilibrium so the
    transient asymptotes smoothly rather than overshooting."""
    from .constants import eps_si

    def dWdt(dW, _t):
        return -(ni300 / N_A) * (dW / tau_g + s0) if dW > 0 else 0.0

    dW0 = W_dd0 - W_eq
    dW_t = odeint(dWdt, dW0, t, hmax=max(t) / 500).flatten()
    dW_t = np.clip(dW_t, 0, None)
    W_t = dW_t + W_eq
    return 1.0 / (1.0 / Cox + W_t / eps_si)


def zerbst_extract(t, C_t, N_A, Cox, W_eq=None):
    """Fit a simulated (or measured) Zerbst C-t transient to recover
    tau_g and s0. Uses the full quadratic governing relation, not the
    small-signal linear approximation, since the linear read is only
    an approximation to the true curve near equilibrium.

    x_eq (the equilibrium value of Cox/C - 1) is taken from the data's
    own late-time asymptote, not recomputed from W_eq via a separate
    formula path -- a real, previously-caught bug: an independently
    computed x_eq differs from the data's true asymptote by enough
    floating-point/ODE-settling mismatch to bias tau_g and s0 by tens
    of percent, since the fit is a difference (x - x_eq) inside a term
    divided by tau_g. W_eq is accepted for API compatibility but unused
    when the data itself is available."""
    from .constants import eps_si

    y_C = Cox / C_t
    x = y_C - 1.0
    x_eq = np.median(x[int(0.9 * len(t)):])

    dy2_dt = np.gradient(y_C ** 2, t)
    zerbst_y = -dy2_dt

    def model(x, tau_g, s0):
        inner = (x - x_eq) / tau_g + Cox * s0 / eps_si
        return np.where(x > x_eq, 2 * (1 + x) * (ni300 / N_A) * inner, 0.0)

    n = len(t)
    lo, hi = int(0.02 * n), int(0.95 * n)
    popt, _ = curve_fit(model, x[lo:hi], zerbst_y[lo:hi],
                         p0=[1e-3, 1.0], bounds=([1e-6, 1e-4], [1.0, 1e4]))
    return popt  # (tau_g, s0)


def dlts_emission_rate(T, Ea, sigma, gamma=2.5e21):
    """SRH/DLTS emission-rate law: e_n(T) = gamma * sigma * T^2 * exp(-Ea/kT).
    gamma is the standard v_th * Nc prefactor combination for silicon."""
    return gamma * sigma * T ** 2 * np.exp(-Ea * q / (k_B * T))


def dlts_arrhenius_extract(T_scan, en_scan):
    """Fit ln(e_n/T^2) vs 1/T to recover trap activation energy Ea (eV)
    and apparent capture cross-section sigma (cm^2)."""
    x = 1.0 / T_scan
    y = np.log(en_scan / T_scan ** 2)

    def line(invT, Ea, ln_prefactor):
        return ln_prefactor - Ea * q / k_B * invT

    popt, _ = curve_fit(line, x, y, p0=[0.3, 45])
    Ea_fit, lnA_fit = popt
    sigma_fit = np.exp(lnA_fit) / 2.5e21
    return Ea_fit, sigma_fit
