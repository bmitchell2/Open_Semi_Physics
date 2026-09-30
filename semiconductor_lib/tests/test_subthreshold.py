import numpy as np
from semiconductor_lib import subthreshold as st, mosfet
from semiconductor_lib.constants import thermal_voltage

NA, COX = 3e17, mosfet.cox_from_tox(3e-7)
VFB, BETA = -1.0, 300 * mosfet.cox_from_tox(3e-7) * 10


def test_phi_F_unchanged_at_300K_and_falls_with_T():
    assert np.isclose(mosfet.phi_F(NA, 300.0), thermal_voltage(300) * np.log(NA / 1e10))
    assert mosfet.phi_F(NA, 400.0) < mosfet.phi_F(NA, 300.0)


def test_slope_is_ln10_m_phit():
    vt, m = st.device_params(NA, COX, VFB)
    vg = np.linspace(vt - 0.4, vt - 0.2, 50)
    i = st.id_subthreshold(vg, 1.0, vt, m, BETA)
    assert np.isclose(st.swing_from_curve(vg, i), mosfet.subthreshold_swing(m), rtol=1e-6)


def test_matches_charge_sheet_in_weak_inversion():
    # m is frozen at its 2 phi_F value, while the true local m rises deeper in
    # weak inversion, so agreement is best near V_T: within 15 % for the top
    # ~3 decades (V_GS - V_T > -0.2 V), and the charge-sheet swing is within
    # 7 % of ln10 m phi_t over the full range.
    vt, m = st.device_params(NA, COX, VFB)
    for vg in np.linspace(vt - 0.2, vt - 0.03, 6):
        a = st.id_subthreshold(vg, 0.5, vt, m, BETA)
        b = mosfet.id_charge_sheet(vg, 0.5, NA, COX, BETA, VFB)
        assert abs(a / b - 1) < 0.15
    vg = np.linspace(vt - 0.4, vt - 0.05, 36)
    i = np.array([mosfet.id_charge_sheet(v, 0.5, NA, COX, BETA, VFB) for v in vg])
    s = np.diff(vg) / np.diff(np.log10(i))
    assert np.all(np.abs(s / mosfet.subthreshold_swing(m) - 1) < 0.07)


def test_vds_saturation():
    vt, m = st.device_params(NA, COX, VFB)
    pt = thermal_voltage(300)
    r = st.id_subthreshold(0.0, 3 * pt, vt, m, BETA) / st.id_subthreshold(0.0, 1.0, vt, m, BETA)
    assert np.isclose(r, 1 - np.exp(-3), rtol=1e-3)


def test_vt_tempco_plausible():
    d = st.vt_tempco(NA, COX, VFB)
    assert -2e-3 < d < -0.5e-3
