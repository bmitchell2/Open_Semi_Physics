import numpy as np
import pytest
from semiconductor_lib import corrosion as cr
from semiconductor_lib import mos_capacitance as mc

CU = dict(Ea=0.0, i0a=1e-6, ba=0.06, Ec=0.6, i0c=1e-9, bc=0.12, ilim=1e-3)
CO = dict(Ea=-0.25, i0a=1e-6, ba=0.06, Ec=0.6, i0c=1e-9, bc=0.12, ilim=1e-3)


# ------------------------------------------------------------- corrosion
def test_corrosion_point_is_zero_net_current_between_equilibria():
    E, i = cr.corrosion_point(CU)
    assert CU["Ea"] < E < CU["Ec"]
    assert abs(cr._net(E, CU)) < 1e-12
    assert i > 0


def test_cathodic_branch_saturates_at_limiting_current():
    i = cr.cathodic_current(-2.0, 0.6, 1e-9, 0.12, i_lim=1e-3)
    assert i == pytest.approx(1e-3, rel=1e-6)


def test_stern_geary_standard_value():
    # 120 mV/decade on both branches gives the textbook B of about 26 mV
    assert cr.stern_geary_B(0.12, 0.12) == pytest.approx(0.02605, rel=1e-3)
    assert cr.stern_geary_B(0.06, np.inf) == pytest.approx(0.06 / np.log(10))


def test_faraday_copper_matches_astm_g102_scale():
    # 1 uA/cm^2 of Cu -> Cu2+ is about 0.0116 mm/yr (ASTM G102 table value)
    nm_per_yr = cr.penetration_rate_nm_min(1e-6) * 60 * 24 * 365.25
    assert nm_per_yr / 1e6 == pytest.approx(0.0116, rel=0.01)


def test_galvanic_mixed_potential_and_area_ratio():
    E_cu, _ = cr.corrosion_point(CU)
    E_co, i_co_free = cr.corrosion_point(CO)
    rates = []
    for ratio in (0.001, 1, 10, 100):
        E, i_cu, i_co = cr.galvanic_couple(CU, ratio, CO, 1.0)
        assert E_co < E < E_cu                      # couple sits between
        rates.append(i_co)
    assert rates[0] == pytest.approx(i_co_free, rel=0.02)   # tiny cathode
    assert np.all(np.diff(rates) > 0)               # small anode suffers
    _, i_cu_c, _ = cr.galvanic_couple(CU, 10, CO, 1.0)
    assert i_cu_c < cr.corrosion_point(CU)[1]       # cathode protected


# ------------------------------------------------------- MOS capacitance
def test_cox_and_overlap():
    assert mc.cox_fF_um2(2.0) == pytest.approx(17.27, rel=1e-3)
    assert mc.overlap_cap(2.0, 8.0) == pytest.approx(0.1381, rel=1e-3)


def test_outer_fringe_limits_and_bsim_default():
    assert mc.outer_fringe_cap(2.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    a, b = mc.outer_fringe_cap(2.0, [50, 100])
    assert b > a and b / a < 1.5                    # logarithmic growth
    # BSIM default CF uses a 0.4 um gate: 2*eps_ox/pi*ln(1+4e-7/TOXE)
    eps_ox = 3.9 * 8.8541878128e-12
    cf_bsim = 2 * eps_ox / np.pi * np.log(1 + 4e-7 / 2e-9) * 1e15 / 1e6
    assert mc.outer_fringe_cap(2.0, 400.0) == pytest.approx(cf_bsim, rel=1e-9)


def test_outer_fringe_numerical_crosscheck():
    for H in (20.0, 90.0, 200.0):
        an = mc.outer_fringe_cap(2.0, H)
        side = mc.outer_fringe_fd(2.0, H)
        assert 1.0 < side / an < 1.3                # analytic slightly low
    plane = mc.outer_fringe_fd(2.0, 90.0, quantity="plane")
    assert plane > mc.outer_fringe_fd(2.0, 90.0)    # gate-top flux added


def test_gate_contact_and_meyer():
    c1 = mc.gate_contact_cap(90, 30, 5, 1.0)
    assert c1 == pytest.approx(0.1328, rel=1e-3)
    assert mc.gate_contact_cap(180, 60, 5, 1.0) == pytest.approx(c1)
    assert mc.gate_contact_cap(90, 30, 5, 0.4) == pytest.approx(0.4 * c1)
    assert sum(mc.meyer_partition("linear")) == pytest.approx(1.0)
    assert mc.meyer_partition("saturation") == (pytest.approx(2 / 3), 0.0)
