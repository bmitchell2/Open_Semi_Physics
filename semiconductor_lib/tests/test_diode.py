"""Physics validation for semiconductor_lib.diode (biased p-n diode)."""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from semiconductor_lib import diode as d
from semiconductor_lib.constants import q, ni300, thermal_voltage

# One-sided N+P diode used on the Semiconductor Notes pages
NA, ND = 1e16, 1e20
DN, DP = 30.0, 2.0
TN, TP = 1e-6, 1e-7
AREA = 1e-4


def test_ni_T_anchored_and_activated():
    assert d.ni_T(300.0) == pytest.approx(ni300, rel=1e-12)
    # ni rises roughly 35x from 300 K to 350 K (Eg/2 activation plus T^1.5)
    assert 25 < d.ni_T(350.0) / d.ni_T(300.0) < 50


def test_injection_goes_into_lighter_side():
    """Hu Sec. 4.8: carriers are injected mainly into the lighter-doped side."""
    assert d.injection_ratio(NA, ND, DN, DP, TN, TP) > 1e3
    assert d.injection_ratio(1e16, 1e16, DN, DN, TN, TN) == pytest.approx(1.0)


def test_ideality_two_at_low_bias_one_at_mid_bias():
    V = np.linspace(0.05, 0.75, 71)
    r = d.forward_iv(V, AREA, 0.0, NA, ND, DN, DP, TN, TP, 1e-6)
    n = d.local_ideality(V, r["I"])
    assert n[np.argmin(abs(V - 0.2))] == pytest.approx(2.0, abs=0.1)
    assert n[np.argmin(abs(V - 0.62))] == pytest.approx(1.0, abs=0.05)


def test_series_resistance_raises_apparent_ideality():
    V = np.linspace(0.5, 1.0, 51)
    r = d.forward_iv(V, AREA, 10.0, NA, ND, DN, DP, TN, TP, 1e-6)
    n = d.local_ideality(V, r["I"])
    assert n[-2] > 3.0
    assert np.all(r["Vj"] <= V + 1e-12)


def test_forward_voltage_tempco_matches_closed_form_and_is_about_minus_2mV():
    J = 10.0
    V = d.forward_voltage_at_current(J, NA, ND, DN, DP, TN, TP)
    num = d.forward_voltage_tempco(J, NA, ND, DN, DP, TN, TP)
    ana = d.forward_voltage_tempco_analytic(V)
    assert num == pytest.approx(ana, rel=0.03)
    assert -2.5e-3 < num < -1.5e-3


def test_leakage_activation_energies_and_crossover():
    """Diffusion leakage ~ ni^2 (Ea ~ Eg), generation ~ ni (Ea ~ Eg/2);
    generation dominates at 300 K for tau ~ 1 us, diffusion at high T."""
    T = np.linspace(250, 500, 60)
    Jd, Jg = d.reverse_leakage_components(T, 3.0, NA, ND, DN, DP, TN, TP, 1e-6)
    Ed = d.arrhenius_activation_energy(T, Jd)
    Eg = d.arrhenius_activation_energy(T, Jg)
    assert 1.15 < Ed < 1.40
    assert 0.55 < Eg < 0.70
    assert Ed / Eg == pytest.approx(2.0, rel=0.05)
    assert Jg[np.argmin(abs(T - 300))] > 100 * Jd[np.argmin(abs(T - 300))]
    assert Jd[-1] > Jg[-1]


def test_breakdown_sze_gibbons_and_critical_field():
    assert d.breakdown_voltage_sze(1e16, Eg=1.1) == pytest.approx(60.0)
    Ec = d.critical_field_from_vb(1e17, d.breakdown_voltage_sze(1e17))
    assert 4e5 < Ec < 7e5            # ~5e5 V/cm at 1e17 cm^-3 (Hu Sec. 4.5.3)
    assert d.breakdown_voltage_one_sided(1e17, Ec) == pytest.approx(d.breakdown_voltage_sze(1e17))


def test_miller_multiplication():
    assert d.miller_multiplication(0.0, 10.0) == pytest.approx(1.0)
    assert d.miller_multiplication(9.0, 10.0, 4.0) == pytest.approx(1 / (1 - 0.9 ** 4))


def test_stored_charge_integral_equals_I_tau():
    """Charge control (Hu Eq. 4.10.2): area under the injected-electron
    profile equals I_n * tau_n."""
    I = 1e-3
    L = np.sqrt(DN * TN)
    J0n = q * ni300 ** 2 * DN / (L * NA)
    Vj = thermal_voltage() * np.log1p(I / (AREA * J0n))
    x = np.linspace(0, 25 * L, 200001)
    Q = q * AREA * trapezoid(d.excess_minority_profile(x, Vj, ni300 ** 2 / NA, L), x)
    assert Q == pytest.approx(d.stored_charge(I, TN), rel=5e-3)
    assert d.diffusion_capacitance(I, TN) == pytest.approx(TN * I / thermal_voltage())


def test_kingston_limits():
    assert d.storage_time_kingston(1.0, 1.0, 1.0) == pytest.approx(0.2275, abs=1e-3)
    # charge control overestimates
    assert d.storage_time_charge_control(1.0, 1.0, 1.0) > 3 * d.storage_time_kingston(1.0, 1.0, 1.0)


@pytest.mark.parametrize("tau", [1e-6, 1e-7])
def test_reverse_recovery_simulation_matches_kingston(tau):
    r = d.simulate_reverse_recovery(tau, DN, NA, AREA, 2000.0, 20.7, 20.0, t_end_over_tau=0.6)
    storage = (r["t"] > 0) & (r["t"] < 0.9 * r["t_s"])
    I_R = -np.mean(r["I"][storage])          # reverse current actually drawn during storage
    ts_k = d.storage_time_kingston(tau, r["I_F"], I_R)
    assert r["t_s"] == pytest.approx(ts_k, rel=0.01)
    # storage time scales with lifetime (lifetime killing)
    assert 0.2 < r["t_s"] / tau < 0.24
    # current decays after the storage phase
    assert abs(r["I"][-1]) < 0.3 * I_R
