"""Physics validation for semiconductor_lib.mosfet."""
import numpy as np
import pytest
from scipy.optimize import brentq

from semiconductor_lib import mosfet as M
from semiconductor_lib.constants import q, eps_si

TOX = 3e-7
COX = M.cox_from_tox(TOX)
NA = 3e17
VFB = -1.0
BETA = 300 * COX * 10


def _vt_m(n_phit=6):
    return (M.vt_uniform(NA, COX, VFB, n_phit=n_phit),
            M.bulk_charge_factor_m(NA, COX, n_phit=n_phit))


@pytest.mark.parametrize("ov,vds", [(1.0, 0.05), (1.0, 2.0), (1.5, 0.05), (0.5, 0.05)])
def test_charge_sheet_matches_square_law_in_strong_inversion(ov, vds):
    """With phi_0 = 2 phi_F + 6 kT/q the two models agree within 5 %."""
    vt, m = _vt_m()
    a = M.id_square_law(vt + ov, vds, vt, BETA, m)
    b = M.id_charge_sheet(vt + ov, vds, NA, COX, BETA, VFB)
    assert abs(b / a - 1) < 0.05


def test_charge_sheet_subthreshold_swing():
    vt, m = _vt_m(0)
    i1 = M.id_charge_sheet(vt - 0.35, 0.05, NA, COX, BETA, VFB)
    i2 = M.id_charge_sheet(vt - 0.25, 0.05, NA, COX, BETA, VFB)
    S = 0.10 / np.log10(i2 / i1)
    assert abs(S / M.subthreshold_swing(m) - 1) < 0.10
    assert 0.0595 < M.subthreshold_swing(1.0) < 0.0600   # 59.5-60 mV/dec at 300 K


def test_charge_sheet_saturates_monotonically():
    vt, _ = _vt_m()
    ids = [M.id_charge_sheet(vt + 1, v, NA, COX, BETA, VFB) for v in np.linspace(0.05, 3, 30)]
    d = np.diff(ids)
    assert np.all(d >= -1e-15) and d[-1] < 1e-3 * d[0]


def test_square_law_continuous_at_vdsat_and_gds_zero():
    vt, m = 0.4, 1.2
    vdsat = 0.5 / m
    lo = M.id_square_law(0.9, vdsat - 1e-6, vt, 1e-3, m)
    hi = M.id_square_law(0.9, vdsat + 1e-6, vt, 1e-3, m)
    assert abs(hi - lo) / hi < 1e-5


def test_two_layer_matches_numeric_poisson():
    N1, N2, xs, psi = 1e16, 5e18, 30e-7, 0.6
    W, Q = M.depletion_two_layer(psi, N1, N2, xs)
    x = np.linspace(0, W, 200001)
    rho = q * np.where(x < xs, N1, N2)
    E = np.cumsum(rho[::-1])[::-1] * (x[1] - x[0]) / eps_si
    assert abs(np.trapezoid(E, x) / psi - 1) < 1e-3


def test_two_layer_reduces_to_uniform():
    for vsb in (0.0, 1.0):
        a = M.vt_two_layer(4e17, 4e17, 20e-7, COX, VFB, vsb)
        b = M.vt_uniform(4e17, COX, VFB, vsb)
        assert abs(a - b) < 1e-9


def test_retrograde_same_wdep_lower_vt_same_initial_body_slope():
    """At equal zero-bias depletion depth, a low-high profile gives lower V_T
    but the same small-bias body-effect slope C_dep/C_ox."""
    N1, N2, xs = 1e16, 5e18, 30e-7
    C = M.cox_from_tox(2e-7)
    W0, _ = M.depletion_two_layer(2 * M.phi_F(N1), N1, N2, xs)
    Nu = brentq(lambda N: np.sqrt(4 * eps_si * M.phi_F(N) / (q * N)) - W0, 1e15, 1e20)
    assert M.vt_two_layer(N1, N2, xs, C) < M.vt_uniform(Nu, C) - 0.3
    h = 1e-3
    s_r = (M.vt_two_layer(N1, N2, xs, C, V_SB=h) - M.vt_two_layer(N1, N2, xs, C)) / h
    s_u = (M.vt_uniform(Nu, C, V_SB=h) - M.vt_uniform(Nu, C)) / h
    assert abs(s_r / s_u - 1) < 0.05


def test_rdf_integral_matches_closed_form_and_quarter_power():
    C = M.cox_from_tox(2e-7)
    L = W = 100e-7
    for N in (2e17, 2e18):
        Wd = np.sqrt(4 * eps_si * M.phi_F(N) / (q * N))
        x = np.linspace(0, Wd, 4001)
        a = M.sigma_vt_rdf(x, np.full_like(x, N), Wd, C, L, W)
        b = M.sigma_vt_rdf_uniform(N, C, L, W)
        assert abs(a / b - 1) < 1e-3
    r = M.sigma_vt_rdf_uniform(2e18, C, L, W) / M.sigma_vt_rdf_uniform(2e17, C, L, W)
    assert 1.6 < r < 2.0   # ~N^(1/4) (=1.78) with a weak phi_F correction


def test_clm_lengths():
    assert M.clm_delta_L_quasi2d(0.2, 0.2, 2e-7, 30e-7) == pytest.approx(0.0, abs=1e-12)
    a = M.clm_delta_L_quasi2d(0.6, 0.2, 2e-7, 30e-7)
    b = M.clm_delta_L_quasi2d(1.2, 0.2, 2e-7, 30e-7)
    assert 0 < a < b < 100e-7
    assert M.clm_delta_L_1d(1.0, 0.2, 3e18) < M.clm_delta_L_1d(1.0, 0.2, 3e17)
