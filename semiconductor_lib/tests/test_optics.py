import numpy as np
import pytest
_trapz = getattr(np, "trapezoid", None) or np.trapz
from semiconductor_lib import optics as o


def test_wien_and_stefan_boltzmann():
    assert o.wien_peak_um(3000) == pytest.approx(0.966, rel=1e-3)
    lam = np.linspace(0.05, 200, 400000)
    total = _trapz(o.planck_radiance(lam, 1273.15), lam) * np.pi
    assert total == pytest.approx(5.670374419e-8 * 1273.15 ** 4, rel=2e-3)
    peak = lam[np.argmax(o.planck_radiance(lam, 1273.15))]
    assert peak == pytest.approx(o.wien_peak_um(1273.15), rel=1e-3)


def test_band_fraction_monotonic_and_bounded():
    f = o.band_fraction(0.2, 1.1, 3000)
    assert 0.2 < f < 0.4           # ~34 % of a 3000 K blackbody lies below 1.1 um
    assert o.band_fraction(0.2, 1.1, 3200) > f


def test_room_temperature_absorption_matches_green_2008():
    # Green (2008) 300 K: ~1.1e4 @500 nm, ~850 @800 nm, ~64 @1000 nm, ~3.5 @1100 nm
    for lam, ref in [(0.5, 1.11e4), (0.8, 850), (1.0, 64), (1.1, 3.5)]:
        assert o.si_alpha_300k(lam) == pytest.approx(ref, rel=0.25)
    lam = np.linspace(0.40, 1.20, 50)
    assert np.all(np.diff(o.si_alpha_300k(lam)) < 0)


def test_timans_clamp_and_agreement_with_measured_fits():
    # without the clamp the 2.3 um value at 700 C would be ~2200 cm^-1
    assert o.timans_alpha_bg(2.3, 700) == 0.0
    for lam, (_, (t0, t1)) in o.TIMANS_QUARTIC.items():
        for T in np.linspace(t0, t1, 6):
            assert o.timans_alpha(lam, T) == pytest.approx(o.timans_measured_fit(lam, T), rel=0.15)


def test_timans_monotonic_in_temperature_and_wavelength_edge():
    T = np.linspace(700, 1200, 51)
    for lam in (1.1, 1.31, 1.54, 2.3, 4.0):
        assert np.all(np.diff(o.timans_alpha(lam, T)) > 0)
    # absorption edge: alpha falls with wavelength across 1.1-1.5 um at fixed T
    lam = np.linspace(1.1, 1.5, 20)
    assert np.all(np.diff(o.timans_alpha(lam, 900)) < 0)


def test_depth_units():
    assert o.absorption_depth_um(1e4) == pytest.approx(1.0)
    assert o.alpha_from_k(1e-3, 1.0) == pytest.approx(4 * np.pi * 10)
