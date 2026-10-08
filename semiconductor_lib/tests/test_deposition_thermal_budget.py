"""
Physics checks for the deposition and thermal_budget modules (session
2026-10-07: PVD/CVD/ALD, oxidation, thermal budget and gettering pages).
"""
import numpy as np
import pytest

from semiconductor_lib import deposition as dp, thermal_budget as tb


# ------------------------------------------------------------ thermal budget
def test_boron_dt_hotter_short_step_dominates():
    tot, (d800, d900) = tb.dt_sum([(800, 3600), (900, 900)])
    assert d900 > 5 * d800                    # 15 min at 900 C beats 1 h at 800 C
    assert np.sqrt(tot) * 1e7 == pytest.approx(10.4, rel=0.02)   # nm


def test_equivalent_time_is_consistent():
    steps = [(800, 3600), (900, 900)]
    t_eq = tb.equivalent_time(steps, 900)
    assert t_eq / 60 == pytest.approx(17.5, rel=0.02)
    tot, _ = tb.dt_sum(steps)
    assert tb.arrhenius(*tb.DOPANTS["B"], 900) * t_eq == pytest.approx(tot)


def test_profile_dt_reduces_to_isothermal():
    t = np.linspace(0, 600, 601)
    T = np.full_like(t, 1000.0)
    iso, _ = tb.dt_sum([(1000, 600)])
    assert tb.profile_dt(t, T) == pytest.approx(iso, rel=1e-9)


def test_fe_crosses_wafer_in_about_an_hour_at_1000C():
    D = tb.metal_diffusivity("Fe", 1000)
    assert 1e-6 < D < 5e-6
    assert tb.diffusion_length(D, 3600) * 1e4 == pytest.approx(895, rel=0.05)
    assert tb.metal_diffusivity("Cu", 1000) > 10 * D


def test_fe_solubility_magnitude_and_trend():
    assert 1e14 < tb.fe_solubility(1000) < 1e15
    assert tb.fe_solubility(700) < 1e-2 * tb.fe_solubility(1000)


# ---------------------------------------------------------------- CVD rate
def test_grove_limits():
    T = np.array([600.0, 1400.0])
    G = dp.grove_growth_rate(T, ks0=1e9, Ea_eV=1.6, hg=1.0)
    kT = dp.K_B * T / dp.Q_E
    ks = 1e9 * np.exp(-1.6 / kT)
    assert G[0] == pytest.approx(ks[0], rel=1e-3)   # reaction limited
    assert G[1] == pytest.approx(1.0, rel=0.05)     # transport limited
    assert np.all(np.diff(dp.grove_growth_rate(np.linspace(600, 1400, 50),
                                                1e7, 1.6, 1.0)) > 0)


# --------------------------------------------------------- conformality
def test_conformality_limits_and_trend():
    assert dp.via_bottom_to_top(0.0, 0.1) == pytest.approx(1.0)
    r = dp.via_bottom_to_top(np.array([2, 5, 10, 20]), 0.01)
    assert np.all(np.diff(r) < 0)
    # lower sticking -> better coverage at fixed AR
    assert dp.via_bottom_to_top(10, 1e-3) > dp.via_bottom_to_top(10, 1e-2)
    # small-phi expansion 1 - phi^2/2
    phi = dp.thiele_modulus(1, 1e-4)
    assert dp.via_bottom_to_top(1, 1e-4) == pytest.approx(1 - phi ** 2 / 2, rel=1e-6)


# -------------------------------------------------------------- Berg model
def test_berg_limits():
    P = np.logspace(-6, 0, 300)
    r = dp.berg_reactive_sputter(P, S=0.1)
    assert r["theta_t"][0] < 0.01 and r["theta_t"][-1] > 0.95   # metallic -> poisoned
    assert r["rate"][0] > 5 * r["rate"][-1]                       # rate collapses
    assert np.all((r["theta_c"] >= 0) & (r["theta_c"] <= 1))


def test_berg_hysteresis_removed_by_pumping():
    P = np.logspace(-5, 0, 400)
    assert dp.berg_has_hysteresis(P, S=0.1)
    assert not dp.berg_has_hysteresis(P, S=10.0)


# ---------------------------------------------------------------- Stoney
def test_stoney_magnitude():
    # 775 um Si(100) biaxial modulus E/(1-nu) ~ 180 GPa, 100 nm film,
    # flat -> R = 1 km gives ~180 MPa
    s = dp.stoney_stress(E_s=130e9, nu_s=0.28, t_s=775e-6, t_f=100e-9,
                         R_before=np.inf, R_after=1000.0)
    assert s / 1e6 == pytest.approx(180, rel=0.05)
