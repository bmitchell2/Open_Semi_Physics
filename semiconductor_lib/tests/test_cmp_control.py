"""Physics validation for cmp_control (endpoint, cleaning, conditioning, R2R)."""
import numpy as np
import pytest

from semiconductor_lib import cmp_control as c

SI_670 = 3.82 + 0.016j


def test_reflectance_periodic_in_fringe_period():
    p = c.fringe_period_nm(670.0, 1.46)
    assert p == pytest.approx(229.45, rel=1e-3)          # ~230 nm per fringe for oxide
    d = np.array([50.0, 50.0 + p, 50.0 + 2 * p])
    R = c.thin_film_reflectance(d, 670.0, 1.46, SI_670)
    assert np.allclose(R, R[0], rtol=1e-4)


def test_reflectance_zero_film_is_bare_substrate():
    R = c.thin_film_reflectance(0.0, 670.0, 1.46, SI_670, n_amb=1.333)
    r = (1.333 - SI_670) / (1.333 + SI_670)
    assert R == pytest.approx(abs(r) ** 2, rel=1e-9)


def test_reflectance_bounded():
    R = c.thin_film_reflectance(np.linspace(0, 1000, 401), 550.0, 1.46, SI_670)
    assert np.all((R >= 0) & (R <= 1))


def test_copper_skin_depth_textbook():
    # ~66 um at 1 MHz, scales as f^-1/2
    d1 = c.skin_depth_m(1.72e-8, 1e6)
    assert d1 == pytest.approx(66e-6, rel=0.02)
    assert c.skin_depth_m(1.72e-8, 4e6) == pytest.approx(d1 / 2, rel=1e-9)


def test_copper_tcr_thickness_error_sign():
    err = c.apparent_thickness_error(500.0, 10.0)
    assert -20 < err < -17          # hotter film reads thinner by ~3.8 %


def test_stokes_layer_megasonic():
    assert c.stokes_layer_m(1e6) * 1e6 == pytest.approx(0.564, rel=0.01)
    assert c.stokes_layer_m(4e4) / c.stokes_layer_m(1e6) == pytest.approx(5.0, rel=1e-9)


def test_debye_length_water():
    # 0.304 nm / sqrt(I [M]) at 25 C
    assert c.debye_length_electrolyte_m(1e-3) * 1e9 == pytest.approx(9.6, rel=0.02)
    assert c.debye_length_electrolyte_m(0.1) * 1e9 == pytest.approx(0.96, rel=0.02)


def test_smoluchowski_sign_and_magnitude():
    z = c.zeta_smoluchowski_V(-3e-8)
    assert z < 0 and z * 1e3 == pytest.approx(-38.4, rel=0.02)


def test_conditioner_gapfree_speed():
    assert c.max_gapfree_sweep_speed(0.10, 90) == pytest.approx(0.15)


def test_ewma_lags_ramp_by_slope_over_lambda():
    n, R0, s = 300, 5.0, -0.004
    rate = R0 + s * np.arange(n)
    inc = np.full(n, 800.0)
    for lam in (0.1, 0.3, 0.6):
        _, used, _ = c.r2r_simulate(rate, inc, 300.0, R0, lam)
        assert used[-1] - rate[-1] == pytest.approx(-s / lam, rel=1e-3)


def test_double_ewma_removes_ramp_bias():
    n, R0, s = 300, 5.0, -0.004
    rate = R0 + s * np.arange(n)
    inc = np.full(n, 800.0)
    out, _, _ = c.r2r_simulate(rate, inc, 300.0, R0, 0.3, lam2=0.1)
    assert abs(out[-1] - 300.0) < 1e-6


def test_r2r_on_target_without_drift_or_noise():
    n = 20
    rate = np.full(n, 4.0)
    inc = np.linspace(780, 820, n)
    out, _, _ = c.r2r_simulate(rate, inc, 300.0, 4.0, 0.3)
    assert np.allclose(out, 300.0)


# ---------------------------------------------------------------- DLVO
R50, A_SIO2 = 50e-9, 0.8e-20


def test_vdw_force_matches_hamaker_sphere_plate():
    D, h = 0.4e-9, 1e-14
    F = -(c.vdw_sphere_plate_J(D + h, R50, A_SIO2)
          - c.vdw_sphere_plate_J(D - h, R50, A_SIO2)) / (2 * h)
    assert F == pytest.approx(-A_SIO2 * R50 / (6 * D ** 2), rel=1e-6)  # attractive
    assert abs(F) == pytest.approx(0.42e-9, rel=0.05)                   # ~0.4 nN


def test_edl_sign_and_debye_decay():
    lam = c.debye_length_electrolyte_m(1e-3)
    V1 = c.edl_sphere_plate_J(5e-9, R50, -0.04, -0.04, 1e-3)
    V2 = c.edl_sphere_plate_J(5e-9 + lam, R50, -0.04, -0.04, 1e-3)
    assert V1 > 0                                   # like charges repel
    assert V2 / V1 == pytest.approx(np.exp(-1), rel=1e-9)
    assert c.edl_sphere_plate_J(5e-9, R50, -0.04, 0.04, 1e-3) < 0  # opposite attract


def test_barrier_collapses_with_ionic_strength():
    b = [c.dlvo_barrier_kT(R50, A_SIO2, -0.04, -0.04, I)
         for I in (1e-4, 1e-3, 1e-2, 1e-1)]
    assert all(x > y for x, y in zip(b, b[1:]))     # monotone decrease
    assert b[1] > 20                                # dilute clean: large barrier
    assert b[3] < 0.6 * b[1]                        # salt shrinks it at fixed zeta
    # weakly charged particle at slurry-like ionic strength: no barrier at all
    assert c.dlvo_barrier_kT(R50, A_SIO2, -0.015, -0.015, 0.1) == 0.0


def test_barrier_vanishes_at_iep():
    assert c.dlvo_barrier_kT(R50, A_SIO2, 0.0, -0.04, 1e-3) == 0.0
