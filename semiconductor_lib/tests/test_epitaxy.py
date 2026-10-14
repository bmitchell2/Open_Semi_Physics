import numpy as np
from semiconductor_lib import epitaxy as ep


def test_ge_si_mismatch():
    assert abs(ep.misfit(1.0) - 0.0417) < 0.001
    assert abs(ep.misfit(1.0, bowing=False) - 0.0418) < 0.001


def test_mb_magnitudes():
    h20 = ep.matthews_blakeslee_hc(0.2)
    assert 9 < h20 < 16          # literature ~10-20 nm for 20 % Ge
    assert 0.5 < ep.matthews_blakeslee_hc(1.0) < 2.0  # pure Ge on Si ~1 nm


def test_people_bean_magnitude_and_ordering():
    pb = ep.people_bean_hc(0.2)
    assert 150 < pb < 300
    x = np.linspace(0.1, 0.5, 9)
    assert np.all(ep.people_bean_hc(x) > ep.matthews_blakeslee_hc(x))


def test_monotonic_decrease():
    x = np.linspace(0.05, 1.0, 40)
    assert np.all(np.diff(ep.matthews_blakeslee_hc(x)) < 0)
    assert np.all(np.diff(ep.people_bean_hc(x)) < 0)


def test_mb_self_consistent():
    x = 0.3
    h = ep.matthews_blakeslee_hc(x)
    b = ep.sige_lattice_constant(x) / np.sqrt(2)
    rhs = b * (1 - 0.28 / 4) / (4 * np.pi * ep.misfit(x) * 1.28) * (np.log(h / b) + 1)
    assert abs(rhs - h) / h < 1e-6


def test_facet_angle():
    assert abs(ep.facet_angle_deg((1, 1, 1), (0, 0, 1)) - np.degrees(np.arccos(1 / np.sqrt(3)))) < 1e-9
    assert abs(ep.facet_angle_deg((1, 1, 1), (0, 0, 1)) - 54.7356) < 1e-3
