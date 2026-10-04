"""Physics validation for high_field, lithography, and plasma modules."""
import numpy as np
import pytest

from semiconductor_lib import high_field as hf
from semiconductor_lib import lithography as li
from semiconductor_lib import plasma as pl
from semiconductor_lib.mosfet import cox_from_tox


# ---------------------------------------------------------------- high_field
def test_canali_bulk_limits():
    vm, Ec, beta = hf.canali_params("electron", 300)
    assert 0.9e7 < vm < 1.1e7          # ~1e7 cm/s saturation velocity
    mu0 = vm / Ec                       # low-field mobility of the fit
    assert 1300 < mu0 < 1600
    v_low = hf.canali_velocity(10.0, "electron")
    assert v_low == pytest.approx(mu0 * 10.0, rel=1e-3)   # ohmic limit
    assert hf.canali_velocity(1e6, "electron") == pytest.approx(vm, rel=0.02)
    # holes are slower and saturate at higher field
    assert hf.canali_velocity(1e4, "hole") < hf.canali_velocity(1e4, "electron")
    # v_sat falls with temperature
    assert hf.canali_params("electron", 400)[0] < vm


def test_velocity_monotonic():
    E = np.logspace(1, 6, 200)
    v = hf.canali_velocity(E)
    assert np.all(np.diff(v) > 0)


def test_vdsat_limits():
    mu, vsat, Vov = 250.0, 8e6, 0.5
    # long channel -> V_ov
    assert hf.vdsat_vsat(Vov, mu, vsat, 100e-4) == pytest.approx(Vov, rel=0.01)
    # short channel -> sqrt(2 Esat L Vov)
    L = 5e-7
    a = hf.e_sat(mu, vsat) * L
    assert hf.vdsat_vsat(Vov, mu, vsat, L) == pytest.approx(np.sqrt(2 * a * Vov), rel=0.15)
    assert hf.vdsat_vsat(Vov, mu, vsat, L) < Vov


def test_idsat_long_and_short_limits():
    mu, vsat, Vov, W = 250.0, 8e6, 0.6, 1e-4
    Cox = cox_from_tox(2e-7)
    L_long = 50e-4
    assert hf.idsat_vsat(Vov, mu, vsat, Cox, W, L_long) == pytest.approx(
        hf.idsat_long_channel(Vov, mu, Cox, W, L_long), rel=0.01)
    # short channel: L-independent, -> W vsat Cox Vov
    i1 = hf.idsat_vsat(Vov, mu, vsat, Cox, W, 2e-7)
    i2 = hf.idsat_vsat(Vov, mu, vsat, Cox, W, 1e-7)
    assert i2 / i1 < 1.25                         # halving L adds little
    assert i2 < W * vsat * Cox * Vov


def test_id_vsat_continuous_at_vdsat():
    mu, vsat, Vt, Cox, W, L = 250.0, 8e6, 0.4, cox_from_tox(2e-7), 1e-4, 4e-6
    vdsat = hf.vdsat_vsat(0.6, mu, vsat, L)
    i_at = hf.id_vsat(1.0, vdsat, Vt, mu, vsat, Cox, W, L)
    i_beyond = hf.id_vsat(1.0, vdsat + 0.5, Vt, mu, vsat, Cox, W, L)
    assert i_at == pytest.approx(i_beyond, rel=1e-9)
    assert i_at == pytest.approx(hf.idsat_vsat(0.6, mu, vsat, Cox, W, L), rel=1e-9)


def test_universal_mobility():
    assert hf.universal_mobility(1e4) == pytest.approx(670, rel=0.01)
    assert hf.universal_mobility(0.67e6) == pytest.approx(335, rel=1e-6)
    E = np.logspace(5, 6.3, 50)
    assert np.all(np.diff(hf.universal_mobility(E)) < 0)
    assert np.all(hf.universal_mobility(E, "hole") < hf.universal_mobility(E))


def test_effective_field_eta():
    Qd, Qi = 5e-7, 1e-6
    En = hf.effective_field(Qd, Qi, "electron")
    Ep = hf.effective_field(Qd, Qi, "hole")
    assert En > Ep                       # eta_n = 1/2 > eta_p = 1/3
    # bias form: tox = 2 nm, Vgs = 1.0, Vt = 0.4 -> ~1.17 MV/cm
    assert hf.effective_field_nmos(1.0, 0.4, 2e-7) == pytest.approx(1.17e6, rel=0.02)


def test_matthiessen_below_each_component():
    E = np.logspace(5.3, 6.3, 20)
    tot, ph, sr = hf.matthiessen_inversion(E, 330.0, 1500.0)
    assert np.all(tot < ph) and np.all(tot < sr)


# --------------------------------------------------------------- lithography
def test_photon_energies():
    assert li.photon_energy_eV(193) == pytest.approx(6.42, abs=0.01)
    assert li.photon_energy_eV(13.5) == pytest.approx(91.8, abs=0.1)


def test_two_beam_limit():
    assert li.min_half_pitch(193, 1.35) == pytest.approx(35.7, abs=0.1)
    assert li.rayleigh_resolution(0.25, 193, 1.35) == pytest.approx(li.min_half_pitch(193, 1.35))
    assert li.min_pitch(193, 1.35, sigma=1.0) == pytest.approx(2 * li.min_half_pitch(193, 1.35))


def test_aerial_image_order_capture():
    x = np.linspace(0, 200, 401)
    # pitch 200 nm, 193i: on-axis passes 0, +-1 -> modulated image
    assert set(li.grating_orders(200, 193, 1.35)) == {-1, 0, 1}
    assert li.image_contrast(li.aerial_image_grating(x, 200, 193, 1.35)) > 0.9
    # pitch 100 nm: on-axis only the zero order -> flat image
    x2 = np.linspace(0, 100, 201)
    flat = li.aerial_image_grating(x2, 100, 193, 1.35)
    assert li.image_contrast(flat) < 1e-9
    # tilting the illumination by lambda/(2p) admits orders 0 and -1
    s = 193 / (2 * 100)
    assert set(li.grating_orders(100, 193, 1.35, s)) == {-1, 0}
    assert li.image_contrast(li.aerial_image_grating(x2, 100, 193, 1.35, s)) > 0.9


def test_clear_field_normalization():
    x = np.linspace(0, 100, 51)
    assert np.allclose(li.aerial_image_grating(x, 100, 193, 1.35, duty=1.0), 1.0)


def test_shot_noise_euv_vs_arf():
    nE = li.photons_per_area(30, 13.5, 100)     # 10 x 10 nm at 30 mJ/cm2
    nA = li.photons_per_area(30, 193, 100)
    assert nA / nE == pytest.approx(193 / 13.5, rel=1e-6)
    assert 1800 < nE < 2300               # ~20 photons/nm^2 incident
    assert li.shot_noise(100) == pytest.approx(0.1)


def test_interference_period():
    assert li.interference_period(632.8, 1.46) == pytest.approx(216.7, abs=0.2)


def test_mask_thickness_roundtrip():
    T = li.mask_thickness_for(1e-5, 300.0, 60.0)
    assert li.mask_transmission(T, 300.0, 60.0) == pytest.approx(1e-5, rel=1e-6)
    assert T == pytest.approx(300 + 4.265 * 60, rel=1e-3)


def test_tanaka_scaling():
    s1 = li.tanaka_collapse_stress(0.072, 0, 40e-9, 80e-9, 40e-9)
    s2 = li.tanaka_collapse_stress(0.072, 0, 40e-9, 160e-9, 40e-9)
    assert s2 / s1 == pytest.approx(4.0)       # (H/W)^2


# -------------------------------------------------------------------- plasma
def test_debye_length_value():
    # 743 sqrt(Te/ne) cm rule of thumb
    assert pl.debye_length(3.0, 1e10) == pytest.approx(743 * np.sqrt(3e-10), rel=0.01)


def test_floating_potential_argon():
    dV = pl.floating_potential_drop(3.0, 39.95)
    assert dV / 3.0 == pytest.approx(5.18, abs=0.05)


def test_child_law_scaling():
    s1 = pl.child_sheath_thickness(100, 3, 1e11)
    s2 = pl.child_sheath_thickness(1600, 3, 1e11)
    assert s2 / s1 == pytest.approx(16 ** 0.75, rel=1e-9)
    # denser plasma -> thinner sheath
    assert pl.child_sheath_thickness(300, 3, 1e12) < pl.child_sheath_thickness(300, 3, 1e10)


def test_mean_free_path_and_angle():
    assert pl.gas_density(10) == pytest.approx(3.22e14, rel=0.01)
    assert pl.ion_mean_free_path(10) == pytest.approx(0.62, rel=0.02)
    assert pl.ion_angular_spread_deg(0.05, 300) < 1.0


def test_area_ratio():
    assert pl.area_ratio_voltage(1.0, 2.0, q=4) == pytest.approx(16.0)
    assert pl.area_ratio_voltage(1.0, 1.0) == pytest.approx(1.0)
