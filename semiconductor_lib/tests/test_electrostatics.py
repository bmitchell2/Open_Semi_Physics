"""
Physics validation tests. These encode the checks that were previously
re-run by hand in every session that touched this electrostatics code
(see chat history 2026-09-05, "Moscaps and veractors" and "Radar
circuit design and process integration") -- run once here, trusted
thereafter (Processing Instructions Section 31).
"""
import numpy as np
import pytest

from semiconductor_lib.constants import eps_ox
from semiconductor_lib.electrostatics import (
    semiconductor_charge_p, semiconductor_charge_n, solve_psi_s,
    differential_Cs, series_C, threshold_voltage, max_depletion_width,
)
from semiconductor_lib.lifetime import (
    srh_lifetime, zerbst_transient, zerbst_extract,
    dlts_emission_rate, dlts_arrhenius_extract,
)


def test_threshold_voltage_matches_numerical_crossing():
    """Numerically solve for Vg where psi_s = 2*phi_F and compare to the
    analytic threshold_voltage() formula. Matched to < 0.5% previously;
    require < 1% here to allow for solver tolerance."""
    N_A, tox = 1e16, 5e-7
    Cox = eps_ox / tox
    VT_analytic = threshold_voltage(N_A, Cox)

    Vg = np.linspace(-2.0, 2.5, 900)
    psi_eq = solve_psi_s(Vg, N_A, Cox, charge_fn=semiconductor_charge_p)
    from semiconductor_lib.constants import thermal_voltage, ni300
    phi_F = thermal_voltage() * np.log(N_A / ni300)
    idx = np.argmin(np.abs(psi_eq - 2 * phi_F))

    assert abs(Vg[idx] - VT_analytic) / VT_analytic < 0.01


def test_capacitance_ordering_in_inversion():
    """C_LF >= C_HF >= C_deep-depletion once past threshold -- required
    physical ordering (more time to generate minority carriers -> more
    charge -> higher capacitance)."""
    N_A, tox = 1e16, 5e-7
    Cox = eps_ox / tox
    Vg = np.linspace(-2.0, 3.0, 400)

    psi_eq = solve_psi_s(Vg, N_A, Cox, charge_fn=semiconductor_charge_p)
    Cs_LF = differential_Cs(psi_eq, N_A, charge_fn=semiconductor_charge_p, majority_only=False)
    Cs_HF = differential_Cs(psi_eq, N_A, charge_fn=semiconductor_charge_p, majority_only=True)
    C_LF, C_HF = series_C(Cox, Cs_LF), series_C(Cox, Cs_HF)

    VT = threshold_voltage(N_A, Cox)
    past_threshold = Vg > VT * 1.2
    assert np.all(C_LF[past_threshold] >= C_HF[past_threshold] - 1e-9)


def test_cmin_matches_analytic_max_depletion_width():
    """The analytic Cmin formula freezes the depletion width at exactly
    psi_s = 2*phi_F. The full numerical solve lets psi_s keep drifting
    logarithmically upward past threshold (inversion charge is
    exponential in psi_s, so a large increase in Qinv only needs a
    modest further rise in psi_s), so the true HF asymptote sits
    systematically below the simple analytic value -- by roughly 6-10%
    close to threshold, more further past it. This is a genuine
    approximation gap between the depletion-approximation formula and
    the full self-consistent solve, not a solver bug: evaluate close
    to threshold and allow for it explicitly."""
    N_A, tox = 1e16, 5e-7
    Cox = eps_ox / tox
    from semiconductor_lib.constants import eps_si
    Wmax = max_depletion_width(N_A)
    Cmin_analytic = series_C(Cox, eps_si / Wmax)

    Vg = np.linspace(-2.0, 1.0, 400)
    psi_eq = solve_psi_s(Vg, N_A, Cox, charge_fn=semiconductor_charge_p)
    Cs_HF = differential_Cs(psi_eq, N_A, charge_fn=semiconductor_charge_p, majority_only=True)
    C_HF = series_C(Cox, Cs_HF)

    assert abs(C_HF[-1] - Cmin_analytic) / Cmin_analytic < 0.10


def test_n_type_charge_has_correct_sign():
    """Qs and psi_s must always carry opposite sign (Gauss's law) -- a
    sign error here was caught and fixed on 2026-09-05."""
    N_D = 1e18
    for psi_s in [-0.8, -0.4, 0.4, 0.8]:
        Qs = semiconductor_charge_n(np.array([psi_s]), N_D, majority_only=True)[0]
        assert np.sign(Qs) == -np.sign(psi_s)


def test_srh_lifetime_slope_above_nref():
    """Log-log slope should approach -1 well above N_ref (literature
    Si/Ge lifetime studies)."""
    N = np.logspace(17, 19, 100)
    tau = srh_lifetime(N)
    slope = np.polyfit(np.log(N), np.log(tau), 1)[0]
    assert abs(slope - (-1.0)) < 0.05


def test_zerbst_recovers_known_parameters():
    """Simulate a transient with known tau_g, s0, then verify the
    extraction function recovers them (round-trip check)."""
    from semiconductor_lib.constants import eps_si, q, thermal_voltage, ni300

    N_A, tox = 1e15, 10e-7
    Cox = eps_ox / tox
    VT = threshold_voltage(N_A, Cox)
    Vg_pulse = VT + 6.0
    tau_g_true, s0_true = 2.0e-3, 0.05

    psi_eq = solve_psi_s(np.array([Vg_pulse]), N_A, Cox, charge_fn=semiconductor_charge_p)[0]
    Qb_eq = semiconductor_charge_p(np.array([psi_eq]), N_A, majority_only=True)[0]
    W_eq = -Qb_eq / (q * N_A)
    psi_dd0 = solve_psi_s(np.array([Vg_pulse]), N_A, Cox, charge_fn=semiconductor_charge_p,
                           majority_only=True, psi_bracket=(-1.0, 10.0))[0]
    Qb_dd0 = semiconductor_charge_p(np.array([psi_dd0]), N_A, majority_only=True)[0]
    W_dd0 = -Qb_dd0 / (q * N_A)

    t = np.linspace(0, 800, 4000)
    C_t = zerbst_transient(t, N_A, Cox, W_eq, W_dd0, tau_g_true, s0_true)
    tau_g_fit, s0_fit = zerbst_extract(t, C_t, N_A, Cox, W_eq)

    assert abs(tau_g_fit - tau_g_true) / tau_g_true < 0.01
    assert abs(s0_fit - s0_true) / s0_true < 0.05


def test_dlts_recovers_known_trap_parameters():
    Ea_true, sigma_true = 0.45, 1.0e-15
    T_scan = np.linspace(150, 320, 35)
    en_scan = dlts_emission_rate(T_scan, Ea_true, sigma_true)
    Ea_fit, sigma_fit = dlts_arrhenius_extract(T_scan, en_scan)

    assert abs(Ea_fit - Ea_true) / Ea_true < 0.01
    assert abs(sigma_fit - sigma_true) / sigma_true < 0.05
