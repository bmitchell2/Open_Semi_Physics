import numpy as np
from semiconductor_lib import gate_tunneling as gt


def test_fn_B_matches_literature_range():
    # Lenzlinger-Snow-type B for Si/SiO2 is about 2.4e8 V/cm (240 MV/cm)
    B = gt.fn_B()
    assert 220 < B < 270


def test_trapezoid_and_triangle_join_continuously():
    t = 5.0
    a = gt.wkb_ln_transmission(t, gt.PHI_B_SIO2_E - 1e-6)
    b = gt.wkb_ln_transmission(t, gt.PHI_B_SIO2_E + 1e-6)
    assert abs(a - b) / abs(a) < 1e-4


def test_rectangular_limit():
    t = 2.0
    small = gt.wkb_ln_transmission(t, 1e-4)
    zero = gt.wkb_ln_transmission(t, 0.0)
    assert abs(small - zero) / abs(zero) < 1e-3


def test_transmission_monotonic():
    V = np.linspace(0.2, 6, 50)
    lnT = gt.wkb_ln_transmission(3.0, V)
    assert np.all(np.diff(lnT) > 0)          # more voltage, more transmission
    t = np.linspace(1.0, 4.0, 30)
    lnT = gt.wkb_ln_transmission(t, 1.0)
    assert np.all(np.diff(lnT) < 0)          # thicker, less transmission


def test_about_one_decade_per_two_angstrom():
    d = gt.decades_per_angstrom(V_ox=1.0, t_ox_nm=1.8)
    assert 0.35 < d < 0.65


def test_fn_current_magnitude():
    # ~1e-2 to 1 A/cm^2 near 10-11 MV/cm is the classic FN regime
    J = gt.fn_current_density(10.0)
    assert 1e-4 < J < 10
    # FN plot ln(J/E^2) vs 1/E is a straight line with slope -B
    E = np.array([8.0, 12.0])
    y = np.log(gt.fn_current_density(E) / E ** 2)
    slope = (y[1] - y[0]) / (1 / E[1] - 1 / E[0])
    assert abs(-slope - gt.fn_B()) / gt.fn_B() < 1e-6
