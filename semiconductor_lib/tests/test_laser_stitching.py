"""Physics and signal-processing checks for semiconductor_lib.laser_stitching."""
import numpy as np
import pytest
from semiconductor_lib import laser_stitching as ls

P = 3.8


def alt(d):
    return lambda n: 1 + d * (1 - 2 * (n % 2))


def test_4pp_sensitivity_integrates_to_ln2_over_pi():
    # uniform-sheet limit: integral of J_AD . J_BC = ln2/pi -> normalized 1
    xs = np.arange(-12, 12, 0.025) + 0.009
    t = np.sinh(np.linspace(-11.3, 11.3, 6001)) * 0.01
    X, Y = np.meshgrid(xs, t, indexing="ij")
    k = np.trapezoid(ls.fpp_sensitivity_2d(X, Y, 1.0), t, axis=1)
    assert np.trapezoid(k, xs) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("orient,fd", [
    # finite-difference resistor-network values (examples/fpp_fd_crosscheck.py)
    ("perpendicular", [0.637, 0.483, 0.659]),
    ("parallel", [0.622, 0.252, -0.077])])
def test_4pp_transfer_function_matches_finite_difference(orient, fd):
    xk, k = ls.fpp_kernel_1d(1.0, orient)
    H = ls.transfer_function(xk, k, [1 / 7.6, 1 / 3.8, 1 / 1.9])
    assert np.allclose(H, fd, atol=0.02)
    assert ls.transfer_function(xk, k, [0.0])[0] == pytest.approx(1.0, abs=1e-6)


def test_micro_4pp_is_transparent_at_stitch_frequencies():
    xk, k = ls.fpp_kernel_1d(0.01, "perpendicular")
    H = ls.transfer_function(xk, k, [1 / 3.8, 2 / 3.8, 4 / 3.8])
    assert np.all(H > 0.99)


def test_identical_passes_are_p_periodic_with_no_odd_harmonics():
    x = np.arange(100, 100 + 20 * P, 0.01)
    rs = ls.simulate_rs(x)
    assert np.allclose(ls.simulate_rs(x + P), rs, rtol=1e-9)
    ev, od = ls.odd_even_power(x, rs, P)
    assert od < 1e-3 * ev


def test_odd_harmonic_is_linear_in_power_mismatch():
    x = np.arange(100, 100 + 20 * P, 0.01)
    o1 = ls.odd_even_power(x, ls.simulate_rs(x, power=alt(0.002)), P)[1]
    o2 = ls.odd_even_power(x, ls.simulate_rs(x, power=alt(0.004)), P)[1]
    assert o2 / o1 == pytest.approx(2.0, rel=0.03)


def test_pure_peak_temperature_response_hides_sharp_stitch():
    # gamma = 0 (max-T), sharp edges, exact 50% overlap: every point sees a
    # full pass, so the stitch vanishes
    x = np.arange(100, 100 + 10 * P, 0.01)
    rs = ls.simulate_rs(x, gamma=0.0, sigma=0.05)
    assert np.ptp(rs) / rs.mean() < 1e-4


def test_hotter_pass_lowers_rs():
    assert ls.rs_temperature_sensitivity() < 0


def test_fold_and_harmonic_fit_recover_a_sine():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 300, 20000))          # non-uniform sampling
    y = 2.0 * np.cos(2 * np.pi * x / P + 0.4)
    _, m, _ = ls.fold(x, y, P, 60)
    assert np.ptp(m) == pytest.approx(4.0, rel=0.01)
    A = ls.harmonic_amplitudes(x, y, P, 2)
    assert abs(A[0]) == pytest.approx(2.0, rel=1e-3)
    assert np.angle(A[0]) == pytest.approx(0.4, abs=1e-3)


def test_spectrum_scaling_gives_zero_to_peak_amplitude():
    dx = 0.1
    x = np.arange(0, 39 * 7.6, dx)
    f, A = ls.amplitude_spectrum(1.5 * np.sin(2 * np.pi * x / P), dx)
    i = np.argmin(abs(f - 1 / P))
    assert abs(A[i]) == pytest.approx(1.5, rel=1e-3)


def test_detrend_window_must_be_two_pitches_for_odd_even_term():
    dx = 0.01
    x = np.arange(0, 40 * 2 * P, dx)
    y = np.cos(2 * np.pi * x / (2 * P))
    r2, v2 = ls.detrend_moving(x, y, 2 * P)
    r1, v1 = ls.detrend_moving(x, y, P)
    assert np.ptp(r2[v2]) / 2 == pytest.approx(1.0, rel=0.01)
    assert np.ptp(r1[v1]) / 2 == pytest.approx(1 - 2 / np.pi, rel=0.02)
