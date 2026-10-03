import numpy as np
import pytest
from semiconductor_lib import scaling as s

K = np.sqrt(2)


def test_dennard_identities():
    r = s.scaling_ratios(K)
    assert r["power_density"] == pytest.approx(1.0)
    assert r["delay"] == pytest.approx(1 / K)
    assert r["energy_per_switch"] == pytest.approx(K ** -3)
    assert r["field"] == pytest.approx(1.0)
    assert r["power_per_device"] == pytest.approx(K ** -2)


def test_dennard_independent_of_current_model():
    a = s.scaling_ratios(K, K, "long")
    b = s.scaling_ratios(K, K, "velsat")
    for k in ("current", "delay", "power_density"):
        assert a[k] == pytest.approx(b[k])


def test_constant_voltage_long_channel_kappa_cubed():
    r = s.scaling_ratios(2.0, 1.0, "long")
    assert r["power_density"] == pytest.approx(8.0)
    assert r["field"] == pytest.approx(2.0)


def test_fixed_voltage_fixed_frequency_density_grows_kappa():
    r = s.scaling_ratios(K, 1.0)
    assert r["power_density_fixed_f"] == pytest.approx(K)


def test_generation_curves_monotonic():
    _, d = s.power_density_vs_generation(6, regime="dennard")
    assert np.allclose(d, 1.0)
    _, v = s.power_density_vs_generation(6, regime="fixedV_velsat")
    assert np.all(np.diff(v) > 0) and v[-1] == pytest.approx(K ** 12)


def test_off_current_decade_per_swing():
    assert s.off_current_ratio(0.060, 0.060) == pytest.approx(10.0)
    assert s.off_current_ratio(0.0) == pytest.approx(1.0)


def test_moore_doubling():
    assert s.moore_count(1967, 64, 1965, 2.0) == pytest.approx(128)


def test_bad_inputs():
    with pytest.raises(ValueError):
        s.scaling_ratios(-1)
    with pytest.raises(ValueError):
        s.scaling_ratios(K, K, "foo")
