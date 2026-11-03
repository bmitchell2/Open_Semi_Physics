"""Physics checks: pad-embedded and rolling scratch-source tracks, and
shear-driven agglomeration helpers."""
import numpy as np
import pytest

from semiconductor_lib import cmp
from semiconductor_lib import cmp_scratch as cs

D = 0.18


def test_matched_speeds_pad_point_is_circle_radius_d():
    w = cmp.rpm_to_rad_s(93)
    t = np.linspace(0, 2 * np.pi / w, 2001)
    for p0 in [(0.10, 0.0), (0.18, 0.0), (0.25, 0.05)]:
        x, y = cs.pad_point_track_wafer(p0, w, w, D, t)
        r = np.hypot(x - p0[0], y - p0[1])
        assert np.allclose(r, D, rtol=1e-12)
        k = cs.track_curvature(x, y)[5:-5]
        assert np.allclose(np.abs(k), 1 / D, rtol=1e-3)


def test_pad_point_matches_relative_velocity():
    wp, wc = cmp.rpm_to_rad_s(93), cmp.rpm_to_rad_s(87)
    p0 = (D + 0.05, 0.02)
    t = np.array([0.0, 1e-6])
    x, y = cs.pad_point_track_wafer(p0, wp, wc, D, t)
    speed = np.hypot(x[1] - x[0], y[1] - y[0]) / 1e-6
    r = np.hypot(p0[0] - D, p0[1])
    phi = np.arctan2(p0[1], p0[0] - D)
    assert speed == pytest.approx(cmp.relative_velocity(r, phi, wp, wc, D),
                                  rel=1e-4)


def test_rolling_particle_matched_is_circle_half_radius():
    w = cmp.rpm_to_rad_s(93)
    t, x, y = cs.rolling_particle_track_wafer((0.0, 0.0), w, w, D,
                                               2 * np.pi / w, n=4000)
    cx, cy = x.mean(), y.mean()
    r = np.hypot(x - cx, y - cy)
    assert np.allclose(r, D / 2, rtol=2e-3)


def test_rolling_equals_half_pad_speed_mismatched():
    wp, wc = cmp.rpm_to_rad_s(93), cmp.rpm_to_rad_s(80)
    t, x, y = cs.rolling_particle_track_wafer((0.03, -0.02), wp, wc, D,
                                               1e-3, n=11)
    v_roll = np.hypot(x[1] - x[0], y[1] - y[0]) / (t[1] - t[0])
    r = np.hypot(0.03, -0.02)
    phi = np.arctan2(-0.02, 0.03)
    assert v_roll == pytest.approx(0.5 * cmp.relative_velocity(r, phi, wp, wc, D),
                                   rel=1e-3)


def test_ortho_peri_ratio_value_and_cubic_scaling():
    r50 = cs.ortho_to_peri_collision_ratio(50e-9, 1e4)
    assert r50 == pytest.approx(1.08, rel=0.02)
    assert cs.ortho_to_peri_collision_ratio(500e-9, 1e4) == pytest.approx(1000 * r50)
