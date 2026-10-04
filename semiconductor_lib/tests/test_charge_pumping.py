import numpy as np
from semiconductor_lib import charge_pumping as cp, mosfet
from semiconductor_lib.constants import q, thermal_voltage

ARGS = dict(V_FB=-1.0, V_T=0.5, dV_A=2.0)


def test_emission_levels_straddle_midgap_and_window_positive():
    Ee, Eh = cp.emission_levels(100e-9, 100e-9, **ARGS)
    assert Ee > 0 and Eh < 0
    dE = cp.energy_window(100e-9, 100e-9, **ARGS)
    assert np.isclose(dE, Ee - Eh) and 0.2 < dE < 1.0


def test_window_shrinks_logarithmically_with_slower_edges():
    t = np.array([1e-8, 1e-7, 1e-6])
    dE = cp.energy_window(t, t, **ARGS)
    # d(dE)/d ln(t) = -2 kT per e-fold of both edges
    slope = np.diff(dE) / np.diff(np.log(t))
    assert np.allclose(slope, -2 * thermal_voltage(300), rtol=1e-9)


def test_icp_round_trip_and_linear_in_f():
    A, dE = 1e-7, 0.5
    i1 = cp.icp(1e10, 1e6, A, dE)
    assert np.isclose(cp.dit_from_icp(i1, 1e6, A, dE), 1e10)
    assert np.isclose(cp.icp(1e10, 2e6, A, dE) / i1, 2.0)


def test_single_trap_charge_matches_groeseneken_example():
    # SISC 2008 tutorial: one trap at 3 MHz gives ~0.48 pA
    assert np.isclose(q * 3e6, 0.48e-12, rtol=0.01)


def test_elliot_curve_regions():
    vb = np.linspace(-3, 1, 801)
    i = cp.elliot_curve(vb, 2.0, -1.0, 0.5, 1.0, 0.02, 0.02)
    assert i[np.argmin(abs(vb + 2.5))] < 1e-6          # top below V_T
    assert i[np.argmin(abs(vb + 1.25))] > 0.99          # plateau
    assert i[np.argmin(abs(vb - 0.0))] < 1e-6           # base above V_FB
    # edges at V_T - dV_A and V_FB
    assert np.isclose(cp.elliot_curve(-1.5, 2.0, -1.0, 0.5, 1.0, 0.02, 0.02), 0.5, atol=0.01)
    assert np.isclose(cp.elliot_curve(-1.0, 2.0, -1.0, 0.5, 1.0, 0.02, 0.02), 0.5, atol=0.01)
    # amplitude smaller than |V_T - V_FB|: no plateau
    assert cp.elliot_curve(vb, 1.0, -1.0, 0.5, 1.0, 0.02, 0.02).max() < 1e-3


def test_edge_stretch_matches_groeseneken_table():
    # tox = 20 nm: D_it 1e11 -> ~100 mV transition spread (SISC 2008)
    cox = mosfet.cox_from_tox(20e-7)
    s = cp.edge_stretch(1e11, cox, dpsi=1.0)
    assert 0.07 < s < 0.13


def test_swing_dit_term():
    cox = mosfet.cox_from_tox(2e-7)
    s0 = cp.swing_with_dit(cox, 0.2 * cox, 0.0)
    assert np.isclose(s0, np.log(10) * thermal_voltage(300) * 1.2)
    assert cp.swing_with_dit(cox, 0.2 * cox, 1e12) > s0
