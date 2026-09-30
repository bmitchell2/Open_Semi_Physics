import numpy as np
import pytest
from semiconductor_lib import short_channel as sc
from semiconductor_lib.mosfet import cox_from_tox

NA = 2e18
COX = cox_from_tox(2.0e-7)


def test_yau_small_xj_limit_sqrt():
    Wd = sc.depletion_width_at_threshold(NA)
    L, xj = 40e-7, 1e-9 * Wd / 1e-9 * 1e-3   # xj = Wd/1000
    exact = sc.yau_charge_sharing_dvt(NA, COX, L, xj)
    approx = (1.602176634e-19 * NA * Wd / COX) * np.sqrt(2 * Wd * xj) / L
    assert exact == pytest.approx(approx, rel=0.03)


def test_yau_large_xj_limit_saturates():
    Wd = sc.depletion_width_at_threshold(NA)
    L = 40e-7
    exact = sc.yau_charge_sharing_dvt(NA, COX, L, 1000 * Wd)
    approx = (1.602176634e-19 * NA * Wd ** 2 / COX) / L
    assert exact == pytest.approx(approx, rel=0.01)


def test_yau_monotonic():
    L = np.linspace(20e-7, 200e-7, 50)
    d = sc.yau_charge_sharing_dvt(NA, COX, L, 20e-7)
    assert np.all(np.diff(d) < 0)
    xj = np.linspace(2e-7, 80e-7, 50)
    d2 = sc.yau_charge_sharing_dvt(NA, COX, 40e-7, xj)
    assert np.all(np.diff(d2) > 0)


def test_brews_cube_root():
    a = sc.brews_lmin_um(0.1, 100, 0.5)
    b = sc.brews_lmin_um(0.8, 100, 0.5)
    assert b / a == pytest.approx(2.0, rel=1e-9)


def test_young_lambda_hand_value():
    # t_si = 7 nm, EOT = 1.3 nm: sqrt(3 * 7 * 1.3) nm = 5.225 nm
    lam = sc.scale_length_soi(7e-7, 1.3e-7)
    assert lam * 1e7 == pytest.approx(np.sqrt(3 * 7 * 1.3), rel=1e-3)


def test_double_gate_ratio():
    r = sc.scale_length_double_gate(7e-7, 1.3e-7) / sc.scale_length_soi(7e-7, 1.3e-7)
    assert r == pytest.approx(1 / np.sqrt(2), rel=1e-9)


def test_liu_trends():
    l = sc.scale_length_bulk(2e-7, sc.depletion_width_at_threshold(NA))
    assert sc.liu_dvt(40e-7, l, 1.1) > sc.liu_dvt(40e-7, l, 0.05)
    assert sc.liu_dvt(30e-7, l, 1.1) > sc.liu_dvt(60e-7, l, 1.1)


def test_dibl_length_ratio_cube_root():
    assert sc.dibl_length_ratio(1, 1, 1, 1, 1, 8) == pytest.approx(2.0)
