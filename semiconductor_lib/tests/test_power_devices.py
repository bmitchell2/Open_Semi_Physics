"""Physics checks for semiconductor_lib.power_devices."""
import numpy as np
import pytest
from semiconductor_lib import power_devices as pd


def test_unipolar_limit_self_consistent():
    # Ron,sp from the closed form equals W/(q mu N) built from W and N.
    for name, m in pd.MATERIALS.items():
        BV = 650.0
        W = pd.drift_width(BV, m["Ec"])
        N = pd.drift_doping(BV, m["eps_r"], m["Ec"])
        direct = W / (pd.q * m["mu"] * N)
        assert pd.material_ron_sp(BV, name) == pytest.approx(direct, rel=1e-12)
        # depletion of N over W reaches Ec at the junction (Gauss's law)
        assert pd.q * N * W / (m["eps_r"] * pd.eps0) == pytest.approx(m["Ec"], rel=1e-12)


def test_scaling_exponents():
    BV = np.array([100.0, 1000.0])
    r = pd.material_ron_sp(BV, "GaN")
    assert np.log10(r[1] / r[0]) == pytest.approx(2.0, rel=1e-9)   # constant Ec -> BV^2
    s = pd.si_ron_sp_baliga(BV)
    assert np.log10(s[1] / s[0]) == pytest.approx(2.5, rel=1e-9)   # doping-dependent Ec


def test_baliga_si_fit_consistent_with_parallel_plane():
    # 5.93e-9 BV^2.5 should match W/(q mu N) from the power-law BV(N), W(N) fits
    N = 1e15
    BV, W = pd.si_parallel_plane(N)
    direct = W / (pd.q * 1360.0 * N)
    assert pd.si_ron_sp_baliga(BV) == pytest.approx(direct, rel=0.05)
    assert 250 < BV < 350                     # ~300 V at 1e15 cm^-3 (textbook)


def test_magnitudes_650V():
    si = pd.si_ron_sp_baliga(650.0)
    assert 0.04 < si < 0.09                   # tens of mOhm.cm^2
    gan = pd.material_ron_sp(650.0, "GaN")
    sic = pd.material_ron_sp(650.0, "4H-SiC")
    assert gan < sic < si
    assert si / gan > 300                     # wide-bandgap advantage ~ (Ec ratio)^3


def test_lateral_and_resurf():
    r = pd.lateral_ron_sp(650.0, 1e13, 1800.0, 1e6)
    assert 1e-4 < r < 3e-4                    # ~0.15 mOhm.cm^2 drift-only
    qsi = pd.depletable_sheet_charge(11.7, 0.3e6)
    assert 1.5e12 < qsi < 2.5e12              # RESURF dose scale for Si
    qgan = pd.depletable_sheet_charge(9.5, 3.3e6)
    assert 1.5e13 < qgan < 2.0e13             # exceeds typical 2DEG density


def test_charge_pumps():
    assert pd.inverter_vout(5.0, 0.0, 1e6, 1e-6) == pytest.approx(-5.0)
    # 10 mA, 1 MHz, 1 uF -> 1/(fC) = 1 ohm -> 10 mV droop
    assert pd.inverter_vout(5.0, 10e-3, 1e6, 1e-6) == pytest.approx(-4.99)
    assert pd.dickson_vout(5.0, 3, 0.0, 1e6, 1e-9) == pytest.approx(20.0)
    assert pd.dickson_vout(5.0, 3, 1e-4, 1e6, 1e-9, V_d=0.6) < 4 * 4.4


def test_figure_functions(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from semiconductor_lib import circuits
    p1 = pd.plot_ron_vs_bv(str(tmp_path / "r.svg"))
    circuits.inverting_charge_pump(str(tmp_path / "c.svg"))
    assert (tmp_path / "r.svg").stat().st_size > 5000
    assert (tmp_path / "c.svg").stat().st_size > 2000
