"""Physics validation for the cmp module and resist-focus helpers."""
import numpy as np
import pytest

from semiconductor_lib import cmp
from semiconductor_lib import lithography as li


def test_preston_linear():
    assert cmp.preston_rate(2.0, 1.0, 3.0) == pytest.approx(2 * cmp.preston_rate(1.0, 1.0, 3.0))


def test_threshold_rate_zero_below_threshold():
    r = cmp.threshold_rate([1.0, 2.0, 3.0], 1.0, 1.0, 2.0)
    assert r[0] == 0 and r[1] == 0 and r[2] == pytest.approx(1.0)


def test_matched_speeds_give_uniform_velocity():
    w = cmp.rpm_to_rad_s(90)
    d = 0.18
    r = np.linspace(0, 0.15, 6)
    v = cmp.mean_relative_velocity(r, w, w, d)
    assert np.allclose(v, w * d, rtol=1e-9)


def test_mismatch_raises_edge_velocity_either_sign():
    d = 0.18
    for wc_rpm in (80, 100):
        v = cmp.mean_relative_velocity([0.0, 0.15], cmp.rpm_to_rad_s(90),
                                       cmp.rpm_to_rad_s(wc_rpm), d)
        assert v[1] > v[0]


def test_bending_length_scaling():
    L1 = cmp.pad_bending_length(300e6, 1.27e-3, 5e6, 1.27e-3)
    L2 = cmp.pad_bending_length(16 * 300e6, 1.27e-3, 5e6, 1.27e-3)
    assert L2 / L1 == pytest.approx(2.0, rel=1e-9)   # L ~ E^(1/4)
    assert 1e-3 < L1 < 1e-2                          # millimetres


def test_force_balance_and_continuity():
    P, rho, hc = 20e3, 0.3, 200.0
    for h in (0, 50, 150, 199.999, 300):
        Pu, Pd = cmp.contact_pressures(h, P, rho, hc)
        assert rho * Pu + (1 - rho) * Pd == pytest.approx(P, rel=1e-6)
    Pu_a, Pd_a = cmp.contact_pressures(hc - 1e-6, P, rho, hc)
    assert Pu_a == pytest.approx(P / rho, rel=1e-6) and Pd_a == pytest.approx(0, abs=1e-3)


def test_preston_step_decays_exponentially_below_contact_height():
    P, rho, hc, h0 = 20e3, 0.5, 200.0, 150.0
    K = 1e-2  # nm/s per Pa (v folded in)
    rate = lambda p: K * p
    t, up, dn, step = cmp.step_height_evolution(rate, P, rho, h0, hc, 1.0, dt=1e-4)
    tau = rho * hc / (K * P)
    assert step[-1] == pytest.approx(h0 * np.exp(-t[-1] / tau), rel=2e-3)


def test_threshold_slurry_self_stops_when_planar():
    P, rho, hc, h0 = 20e3, 0.5, 200.0, 400.0
    rate = lambda p: float(cmp.threshold_rate(p, 1.0, 0.05, 1.2 * P))
    t, up, dn, step = cmp.step_height_evolution(rate, P, rho, h0, hc, 600.0, dt=0.05)
    # stall step where P_up falls to P_th: h* = hc (P_th/P - 1) rho/(1-rho)
    h_star = hc * (1.2 - 1) * rho / (1 - rho)
    assert step[-1] == pytest.approx(h_star, rel=0.02)
    # removal stalls: no change over the last 10 % of the run
    i = int(0.9 * len(t))
    assert up[-1] - up[i] < 1e-6


def test_mid_resist_focus():
    assert li.mid_resist_focus_offset(1000, 1.7) == pytest.approx(294.1, rel=1e-3)
    assert li.scanner_defocus_for_depth(100, 1.7, 1.44) == pytest.approx(84.7, rel=1e-3)


def test_stall_step_helper_matches_integration():
    assert cmp.threshold_stall_step(20e3, 24e3, 0.5, 200.0) == pytest.approx(40.0)
    assert cmp.threshold_stall_step(20e3, 18e3, 0.5, 200.0) == 0.0
