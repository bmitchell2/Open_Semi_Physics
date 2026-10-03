import math
import pytest

from semiconductor_lib import implant_dosimetry as d


def test_gas_density_loschmidt_scale():
    # 1 Torr at 300 K ~ 3.22e16 cm^-3
    assert d.gas_density_cm3(1.0) == pytest.approx(3.22e16, rel=0.01)


def test_no_gas_no_neutralization():
    assert d.surviving_ion_fraction(1e-15, 0.0, 100) == 1.0
    assert d.decel_contamination_fraction(1e-15, 0.0, 50) == 0.0
    assert d.residual_dose_error(3000, 0, 0.0) == 0.0


def test_compensation_with_true_k_is_exact():
    assert d.residual_dose_error(2827, 2827, 2e-5) == pytest.approx(0.0, abs=1e-15)


def test_overdose_sign_and_magnitude():
    # Patent example K = 2827 /Torr: 2e-5 Torr outgassing gives ~5.8 % overdose
    err = d.residual_dose_error(2827, 0, 2e-5)
    assert err > 0
    assert err == pytest.approx(math.exp(2827 * 2e-5) - 1, rel=1e-12)
    assert 0.05 < err < 0.065


def test_fit_recovers_k():
    K, I0 = 2827.0, 6.7
    P = [0, 5e-6, 1e-5, 2e-5, 4e-5]
    I = [I0 * math.exp(-K * p) for p in P]
    Kf, I0f = d.fit_k_factor(P, I)
    assert Kf == pytest.approx(K, rel=1e-9)
    assert I0f == pytest.approx(I0, rel=1e-9)


def test_k_from_cross_section_plausible():
    # sigma ~ 1e-15 cm^2 over ~1 m gives K of order 1e3 /Torr
    K = d.k_factor_from_cross_section(1e-15, 100)
    assert 1e3 < K < 1e4


def test_contamination_monotonic_in_pressure():
    f1 = d.decel_contamination_fraction(1e-15, 1e-6, 50)
    f2 = d.decel_contamination_fraction(1e-15, 5e-6, 50)
    assert 0 < f1 < f2 < 0.01


def test_depth_ratio_limits():
    assert d.contaminant_depth_ratio(1.0) == 1.0
    assert d.contaminant_depth_ratio(4.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        d.contaminant_depth_ratio(0.5)


def test_tilt_geometry():
    lat, vert = d.tilted_projection(30.0, 0.0)
    assert lat == 0.0 and vert == 30.0
    lat, vert = d.tilted_projection(30.0, 30.0)
    assert lat == pytest.approx(15.0) and vert == pytest.approx(30 * math.cos(math.pi / 6))
    # shadowing amplifies angle error by h/(R_p cos^3) relative to lateral reach
    s_lat = d.lateral_reach_sensitivity(30.0, 30.0, 0.5)
    s_sh = d.shadow_sensitivity(300.0, 30.0, 0.5)
    assert s_sh / s_lat == pytest.approx(300 / (30 * math.cos(math.pi / 6) ** 3), rel=1e-9)
    assert d.shadow_length(100.0, 45.0) == pytest.approx(100.0)
