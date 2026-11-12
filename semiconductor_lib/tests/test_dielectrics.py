import math
import pytest
from semiconductor_lib import dielectrics as d
from semiconductor_lib.constants import eps0, eps_ox

K_OX = eps_ox / eps0


def test_bound_fraction_limits():
    assert d.bound_charge_fraction(1.0) == 0.0            # vacuum: nothing cancelled
    assert d.bound_charge_fraction(K_OX) == pytest.approx(0.744, abs=1e-3)
    assert d.bound_charge_fraction(20) == pytest.approx(0.95)
    assert d.bound_charge_fraction(1e9) == pytest.approx(1.0)  # conductor limit


def test_gauss_net_charge_sets_field():
    # field from the net (free - bound) charge equals the screened field
    s = 1e-6
    for k in (3.9, 11.7, 25):
        net = s * (1 - d.bound_charge_fraction(k))
        assert d.field_fixed_charge(s, k) == pytest.approx(net / eps0)


def test_polarization_equals_bound_charge():
    s, k = 2e-6, 20
    E = d.field_fixed_charge(s, k)
    assert d.polarization(E, k) == pytest.approx(s * d.bound_charge_fraction(k))


def test_fixed_charge_vs_fixed_voltage():
    V, t = 1.2, 2e-7
    s_ox = d.charge_fixed_voltage(V, K_OX, t)
    # fixed voltage: field independent of k, charge scales with k
    assert d.field_fixed_voltage(V, t) == pytest.approx(6e6)
    assert d.charge_fixed_voltage(V, 20, t) / s_ox == pytest.approx(20 / K_OX)
    # fixed charge: inserting k=20 drops the voltage by K_OX/20
    assert d.voltage_fixed_charge(s_ox, 20, t) == pytest.approx(V * K_OX / 20)
    # 6 MV/cm across SiO2 holds ~1.3e13 electrons/cm^2 (Gauss: eps_ox*E/q)
    assert d.electrons_per_cm2(s_ox) == pytest.approx(1.29e13, rel=0.01)


def test_energy_is_half_QV():
    V, k, t = 1.2, 20, 5e-7
    Q = d.charge_fixed_voltage(V, k, t)
    assert d.stored_energy_per_area(V, k, t) == pytest.approx(0.5 * Q * V)
