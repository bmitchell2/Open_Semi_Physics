"""Physics checks for hydrogen_passivation, contacts, gate_power_punchthrough."""
import numpy as np
import pytest

from semiconductor_lib import hydrogen_passivation as hp
from semiconductor_lib import contacts as ct
from semiconductor_lib import gate_power_punchthrough as g
from semiconductor_lib.constants import eps_ox


def test_brower_onsets():
    # Brower: dissociation in vacuum becomes significant above ~550 C
    assert hp.depassivated_in_inert(1800, hp.c_to_k(450)) < 0.01
    assert 0.2 < hp.depassivated_in_inert(1800, hp.c_to_k(550)) < 0.6
    assert hp.depassivated_in_inert(1800, hp.c_to_k(650)) > 0.99


def test_passivation_monotonic_and_steady_state():
    T = hp.c_to_k(np.linspace(300, 800, 50))
    assert np.all(np.diff(hp.k_dissociation(T)) > 0)
    assert np.all(np.diff(hp.k_passivation(T)) > 0)
    # Ed > Ef, so the unpassivated steady state rises with temperature
    f = hp.steady_state_unpassivated(T, 1e17)
    assert np.all(np.diff(f) > 0)
    # more H2 -> more passivation
    assert hp.steady_state_unpassivated(T[10], 1e18) < hp.steady_state_unpassivated(T[10], 1e17)


def test_relaxation_limits():
    T = hp.c_to_k(400)
    assert hp.unpassivated_after(0, T, 1e17) == pytest.approx(1.0)
    assert hp.unpassivated_after(1e6, T, 1e17) == pytest.approx(
        hp.steady_state_unpassivated(T, 1e17), rel=1e-6)


def test_e00_sze_value():
    # Sze: E00 = 1.85e-11 sqrt(N/(m_rel eps_r)) eV -> 33.5 meV at 1e19, m=0.26
    assert ct.e00_eV(1e19) == pytest.approx(0.0335, rel=0.02)
    assert ct.e00_eV(4e19) == pytest.approx(2 * ct.e00_eV(1e19), rel=1e-9)


def test_tunnelling_trends_and_magnitude():
    N = np.logspace(19.5, 20.7, 20)
    r = ct.tunnelling_rho_c(0.65, N)
    assert np.all(np.diff(r) < 0)
    assert ct.tunnelling_rho_c(0.4, 1e20) < ct.tunnelling_rho_c(0.65, 1e20)
    # within an order of magnitude of Hu Fig 4-46 / measured ~1e-7 Ohm cm2 at ~1e20
    assert 1e-8 < ct.tunnelling_rho_c(0.6, 1e20) < 3e-6
    # tunnelling contact far below the thermionic-only value
    assert ct.tunnelling_rho_c(0.65, 1e20) < 1e-6 * ct.thermionic_rho_c(0.65)


def test_pinning_and_contact_resistance():
    assert ct.pinned_barrier_n(4.75) == pytest.approx(0.7)
    assert ct.contact_resistance_ohm(7.07e-9, 30e-7) == pytest.approx(1000, rel=0.01)


def test_poly_depletion_hu_example():
    # Hu Example 5-3: Vox = 1 V, 2 nm, Npoly = 8e19 -> W ~1.3 nm, phi ~0.11 V
    W = g.poly_depletion_width_cm(eps_ox * 1.0 / 2e-7, 8e19)
    assert W * 1e7 == pytest.approx(1.3, rel=0.06)
    assert g.poly_depletion_potential(W, 8e19) == pytest.approx(0.11, rel=0.06)


def test_poly_depletion_solver_trends():
    a = g.solve_poly_depletion(1.0, 1.2e-7, 1e20)
    b = g.solve_poly_depletion(1.0, 1.2e-7, 2e19)
    assert a["Q_inv"] < a["Q_inv_no_polydep"]
    assert b["W_dpoly_cm"] > a["W_dpoly_cm"]
    assert b["Q_inv"] < a["Q_inv"]
    assert a["T_oxe_cm"] > 1.2e-7


def test_punchthrough_scaling():
    assert g.punchthrough_voltage_1d(100e-7, 1e18) > g.punchthrough_voltage_1d(80e-7, 1e18)
    assert g.punchthrough_voltage_1d(100e-7, 2e18) > g.punchthrough_voltage_1d(100e-7, 1e18)


def test_power():
    assert g.dynamic_power(0.1, 1e-9, 1.0, 2e9) == pytest.approx(0.2)
    assert g.dynamic_power(1, 1e-15, 2.0, 1e9) == pytest.approx(4 * g.dynamic_power(1, 1e-15, 1.0, 1e9))
