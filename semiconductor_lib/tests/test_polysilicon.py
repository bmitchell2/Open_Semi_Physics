"""Physics checks for semiconductor_lib.polysilicon (Seto grain-boundary model, SPC, matching)."""
import numpy as np
import pytest
from semiconductor_lib import polysilicon as P


def test_critical_doping_seto():
    # Qt = 3.34e12 cm^-2, 20 nm grains -> 1.67e18 cm^-3
    assert P.critical_doping() == pytest.approx(1.67e18, rel=0.01)


def test_hall_mobility_minimum_matches_seto_measurement():
    # Seto (1975) measured the Hall mobility minimum near 2e18 cm^-3
    Ns = np.logspace(16.5, 20, 71)
    mu = [P.hall_mobility(N) for N in Ns]
    Nmin = Ns[int(np.argmin(mu))]
    assert 1.0e18 < Nmin < 3.0e18


def test_heavy_doping_approaches_single_crystal():
    N = 1e20
    assert P.poly_resistivity(N) / P.single_crystal_resistivity(N) < 1.3
    assert P.poly_tcr_ppm(N) > 0


def test_light_doping_negative_tcr_and_activation():
    assert P.poly_tcr_ppm(1e17) < -10000
    Ea = P.activation_energy_eV(1e16)
    assert 0.3 < Ea < 0.6            # single-level model gives E_t - E_V ~ 0.37 eV


def test_resistivity_falls_with_grain_size_at_moderate_doping():
    r = [P.poly_resistivity(5e18, L) for L in (20e-7, 50e-7, 200e-7)]
    assert r[0] > r[1] > r[2]


def test_zero_tcr_doping_moves_lower_for_larger_grains():
    Nz = [P.zero_tcr_doping(L) for L in (20e-7, 50e-7, 200e-7)]
    assert Nz[0] > Nz[1] > Nz[2]
    assert 1e18 < Nz[2] < Nz[0] < 3e19


def test_regime_switch_at_full_depletion():
    assert P.seto_state(1e18)["regime"] == "full"
    assert P.seto_state(5e18)["regime"] == "partial"


def test_spc_cooler_gives_larger_grains_and_longer_time():
    d = P.spc_grain_size_ratio([550, 600, 700])
    t = P.spc_time_ratio([550, 600, 700])
    assert d[1] == pytest.approx(1.0) and d[0] > 1 > d[2]
    assert t[0] > 1 > t[2]
    # a higher nucleation barrier strengthens the grain-size dependence
    assert P.spc_grain_size_ratio(700, Ea_nuc=4.9) < P.spc_grain_size_ratio(700, Ea_nuc=3.9)


def test_mismatch_grain_counting_matches_monte_carlo():
    s_el = 0.1
    mc = P.monte_carlo_grid_sigma(20, 20, s_el, trials=4000)
    an = P.resistor_mismatch_sigma(20.0, 20.0, 1.0, s_el)
    assert mc == pytest.approx(an, rel=0.1)
