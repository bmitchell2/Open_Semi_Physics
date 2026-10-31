"""Physics checks for semiconductor_lib.geometry_cv."""
import numpy as np
import pytest

from semiconductor_lib.geometry_cv import (
    area_perimeter_extract, channel_rc_admittance, channel_rc_capacitance,
    channel_rc_tau, parallel_mode_capacitance,
)

COX = 6.9e-3      # F/m^2 (5 nm SiO2)
MU = 0.03         # m^2/V/s (300 cm^2/V/s)


def test_area_perimeter_recovers_synthetic_components():
    side = np.array([1, 2, 5, 10, 50, 100]) * 1e-6
    A, P = side ** 2, 4 * side
    Ca_true = np.array([6.9e-3, 2.0e-3, 6.0e-3])
    Cp_true = np.array([1.0e-10, 3.0e-10, 1.5e-10])
    C = np.outer(A, Ca_true) + np.outer(P, Cp_true)
    Ca, Cp = area_perimeter_extract(A, P, C)
    assert np.allclose(Ca, Ca_true, rtol=1e-9)
    assert np.allclose(Cp, Cp_true, rtol=1e-6)


def test_area_perimeter_needs_range_of_PA():
    with pytest.raises(ValueError):
        area_perimeter_extract([1e-12, 1e-12], [4e-6, 4e-6], [1e-14, 1e-14])


@pytest.mark.parametrize("feed", ["both", "one"])
def test_channel_low_frequency_limit_is_full_oxide_capacitance(feed):
    L, W = 20e-6, 10e-6
    C = channel_rc_capacitance(1.0, L, W, COX, MU, 0.5, feed)
    assert C == pytest.approx(COX * W * L, rel=1e-6)


def test_channel_high_frequency_capacitance_vanishes():
    L, W = 100e-6, 10e-6
    C = channel_rc_capacitance(1e11, L, W, COX, MU, 0.5)
    assert C < 0.01 * COX * W * L


@pytest.mark.parametrize("feed,k", [("both", 12.0), ("one", 3.0)])
def test_low_frequency_series_resistance_is_R_over_k(feed, k):
    L, W, Vov = 50e-6, 10e-6, 0.5
    Y = channel_rc_admittance(10.0, L, W, COX, MU, Vov, feed)
    R_ch = L / (W * MU * COX * Vov)
    assert np.real(1.0 / Y) == pytest.approx(R_ch / k, rel=1e-4)
    tau = channel_rc_tau(L, MU, Vov, feed)
    assert tau == pytest.approx((R_ch / k) * COX * W * L, rel=1e-12)


def test_capacitance_rolloff_monotonic_and_longer_channel_rolls_off_first():
    f = np.logspace(3, 10, 200)
    c_short = channel_rc_capacitance(f, 10e-6, 1e-5, COX, MU, 0.5) / (COX * 1e-5 * 10e-6)
    c_long = channel_rc_capacitance(f, 100e-6, 1e-5, COX, MU, 0.5) / (COX * 1e-5 * 100e-6)
    assert np.all(np.diff(c_short) <= 1e-12)
    assert np.all(c_long <= c_short + 1e-12)


def test_parallel_mode_series_resistance():
    C, Rs, f = 69e-12, 1e3, 1e6
    x = 2 * np.pi * f * Rs * C
    assert parallel_mode_capacitance(C, Rs, f) == pytest.approx(C / (1 + x ** 2), rel=1e-12)
    assert parallel_mode_capacitance(C, 0.0, f) == pytest.approx(C, rel=1e-12)
