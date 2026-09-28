"""
Physics validation for pnjunction.debye_length,
depletion_width_over_debye_length, and solve_step_junction_poisson.
"""
import numpy as np
import pytest

from semiconductor_lib import pnjunction
from semiconductor_lib.constants import q, eps_si, ni300, thermal_voltage


def test_debye_length_matches_hand_formula():
    N = 1e16
    Vt = thermal_voltage(300.0)
    expected = np.sqrt(eps_si * Vt / (q * N))
    assert pnjunction.debye_length(N) == pytest.approx(expected, rel=1e-12)


def test_debye_length_decreases_with_doping():
    Ns = [1e14, 1e15, 1e16, 1e17, 1e18, 1e19]
    LDs = [pnjunction.debye_length(N) for N in Ns]
    assert all(a > b for a, b in zip(LDs, LDs[1:]))
    # ~ sqrt(10) shrink per decade of doping
    for a, b in zip(LDs, LDs[1:]):
        assert 2.5 < a / b < 4.0


def test_width_over_debye_length_symmetric_matches_closed_form():
    # Symmetric case: W/L_D = 2*sqrt(Vbi/Vt) exactly, from
    # W = sqrt(4 eps Vbi / (q N)) and L_D = sqrt(eps Vt / (q N)).
    for N in (1e15, 1e16, 1e17, 1e18):
        Vbi = pnjunction.built_in_potential(N, N)
        Vt = thermal_voltage(300.0)
        expected = 2.0 * np.sqrt(Vbi / Vt)
        got = pnjunction.depletion_width_over_debye_length(N, N)
        assert got == pytest.approx(expected, rel=1e-9)


def test_width_over_debye_length_is_order_ten_at_moderate_doping():
    # Physically plausible magnitude check (Section 23 "Magnitude").
    ratio = pnjunction.depletion_width_over_debye_length(1e16, 1e16)
    assert 8.0 < ratio < 13.0


def test_width_over_debye_length_grows_with_doping():
    # Vbi grows only logarithmically with doping, but L_D shrinks as
    # 1/sqrt(N), so the ratio should slowly grow with doping.
    r_lo = pnjunction.depletion_width_over_debye_length(1e15, 1e15)
    r_hi = pnjunction.depletion_width_over_debye_length(1e18, 1e18)
    assert r_hi > r_lo


def test_step_junction_matches_solve_pn_poisson_for_symmetric_pn():
    NA = ND = 1e16
    sol_new = pnjunction.solve_step_junction_poisson(-NA, ND, L=2.5e-4)
    sol_old = pnjunction.solve_pn_poisson(NA, ND, L=2.5e-4)
    # Built-in potential (psi at the two ends) should agree to high precision
    Vbi_new = sol_new.psi[-1] - sol_new.psi[0]
    Vbi_old = sol_old.psi[-1] - sol_old.psi[0]
    assert Vbi_new == pytest.approx(Vbi_old, rel=1e-6)
    assert Vbi_new == pytest.approx(pnjunction.built_in_potential(NA, ND), rel=1e-6)
    # Depletion widths from the integrated charge should agree to a few %
    xp_new, xn_new = pnjunction.numerical_widths(sol_new, NA, ND)
    xp_old, xn_old = pnjunction.numerical_widths(sol_old, NA, ND)
    assert xp_new == pytest.approx(xp_old, rel=0.03)
    assert xn_new == pytest.approx(xn_old, rel=0.03)


def test_step_junction_n_type_intrinsic_matches_analytic_vbi():
    # One side undoped (intrinsic): Vbi -> Vt*ln(ND/ni) for ND >> ni,
    # via Vbi = Vt*asinh(ND/2ni) (exact) since asinh(0)=0 on the
    # intrinsic side.
    ND = 1e16
    Vt = thermal_voltage(300.0)
    sol = pnjunction.solve_step_junction_poisson(0.0, ND)
    Vbi_numeric = sol.psi[-1] - sol.psi[0]
    Vbi_exact_asinh = Vt * np.arcsinh(ND / (2 * ni300))
    Vbi_approx_log = Vt * np.log(ND / ni300)
    assert Vbi_numeric == pytest.approx(Vbi_exact_asinh, rel=1e-6)
    # The textbook log approximation should be close but not exact
    assert Vbi_numeric == pytest.approx(Vbi_approx_log, rel=1e-3)


def test_step_junction_n_type_intrinsic_is_charge_neutral_overall():
    ND = 1e16
    sol = pnjunction.solve_step_junction_poisson(0.0, ND)
    dx = sol.x[1] - sol.x[0]
    total_charge = np.sum(sol.rho) * dx
    # Should integrate to ~0 (charge neutrality of the whole structure);
    # compare against a characteristic charge scale of the problem.
    scale = q * ND * dx * 10
    assert abs(total_charge) < scale


def test_step_junction_n_type_intrinsic_spreads_over_microns_not_nm():
    # This is the key qualitative result discussed in the September 2026
    # session: replacing the p-type side with intrinsic material inflates
    # the length scale on that side from nm to order-micron, because the
    # relevant Debye length there is set by ni, not by a doping level.
    ND = 1e16
    sol = pnjunction.solve_step_junction_poisson(0.0, ND)
    ld_intrinsic = pnjunction.debye_length(ni300)
    ld_doped = pnjunction.debye_length(ND)
    assert ld_intrinsic / ld_doped == pytest.approx(np.sqrt(ND / ni300), rel=1e-9)
    assert ld_intrinsic > 100 * ld_doped
    # ~40.9 um at 300 K for ni=1e10 (order-of-magnitude check; the
    # micron-scale spread on the intrinsic side is the qualitative point)
    assert 20e-4 < ld_intrinsic < 60e-4
