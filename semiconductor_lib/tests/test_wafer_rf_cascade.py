import numpy as np
import pytest
from semiconductor_lib import wafer
from semiconductor_lib import rf_system as rf
from semiconductor_lib.transport import resistivity
from semiconductor_lib.scattering import caughey_thomas, CT_HOLES
from semiconductor_lib.thermal_budget import dt_sum


def test_resistivity_to_doping_chart_values_and_roundtrip():
    N10 = wafer.resistivity_to_doping(10.0, "p")
    assert 1.2e15 < N10 < 1.5e15          # standard chart ~1.3e15
    Np = wafer.resistivity_to_doping(0.01, "p")
    assert 7e18 < Np < 1.1e19              # heavily doped p+ substrate
    mu = caughey_thomas(N10, 300.0, CT_HOLES)
    assert abs(resistivity(0.0, N10, mu_p=mu) - 10.0) < 1e-6


def test_updiffusion_profile_limits_midpoint_and_scaling():
    Ns, Ne, Dt = 1e19, 1.3e15, 1e-10
    assert wafer.substrate_updiffusion_profile(0.0, Ns, Ne, Dt) == pytest.approx(0.5 * (Ns + Ne))
    assert wafer.substrate_updiffusion_profile(20.0, Ns, Ne, Dt) == pytest.approx(Ne, rel=1e-6)
    assert wafer.substrate_updiffusion_profile(-20.0, Ns, Ne, Dt) == pytest.approx(Ns, rel=1e-6)
    prof = wafer.substrate_updiffusion_profile(np.linspace(0, 3, 50), Ns, Ne, Dt)
    assert np.all(np.diff(prof) <= 0)       # monotonic decrease into the epi
    r1 = wafer.updiffusion_rise(Ns, Ne, Dt, 2 * Ne)
    r4 = wafer.updiffusion_rise(Ns, Ne, 4 * Dt, 2 * Ne)
    assert r4 / r1 == pytest.approx(2.0, rel=1e-9)   # rise ~ sqrt(Dt)
    assert wafer.usable_epi_thickness(4.0, Ns, Ne, Dt) == pytest.approx(4.0 - r1)


def test_updiffusion_rise_magnitude_for_a_1100C_hour():
    Dt, _ = dt_sum([(1100.0, 3600.0)], "B")
    rise = wafer.updiffusion_rise(1e19, 1.3e15, Dt, 2.6e15)
    assert 0.5 < rise < 2.0                 # order of a micron


def test_self_resonance_and_apparent_inductance():
    f = rf.self_resonant_frequency(2e-9, 50e-15)
    assert f == pytest.approx(15.92e9, rel=1e-3)
    assert rf.apparent_inductance(2e-9, f / 2, f) == pytest.approx(2e-9 * 4 / 3)
    assert rf.apparent_inductance(2e-9, 2 * f, f) < 0   # capacitive above SRF


def test_db_conventions_agree_for_one_signal():
    assert rf.db_power(0.1) == pytest.approx(-10.0)
    assert rf.db_amplitude(0.1) == pytest.approx(-20.0)
    assert rf.db_power(0.1 ** 2) == pytest.approx(rf.db_amplitude(0.1))


def test_friis_cascade():
    assert rf.friis_noise_figure_db([(20, 2.0)]) == pytest.approx(2.0)
    # matched loss in front of an LNA adds its dB directly to the NF
    assert rf.friis_noise_figure_db([(-1.5, 1.5), (15, 2.0), (10, 12)]) == pytest.approx(
        1.5 + rf.friis_noise_figure_db([(15, 2.0), (10, 12)]), abs=1e-9)
    # high first-stage gain suppresses later stages
    lo = rf.friis_noise_figure_db([(25, 2.0), (10, 12)])
    hi = rf.friis_noise_figure_db([(5, 2.0), (10, 12)])
    assert lo < 2.2 < hi


def test_range_correlation_limits():
    tau = 2 * 0.1 / 2.998e8
    small = rf.range_correlation_factor_db(1e6, 0.1)
    assert small == pytest.approx(20 * np.log10(2 * np.pi * 1e6 * tau), abs=0.01)
    assert small < -40                       # near reflection strongly suppressed
    R_half = 2.998e8 / (4 * 1e6)            # df * tau = 1/2
    assert rf.range_correlation_factor_db(1e6, R_half) == pytest.approx(6.02, abs=0.01)
