"""Physics checks for semiconductor_lib.band_diagrams."""
import numpy as np
import pytest

from semiconductor_lib.constants import thermal_voltage, eps_ox
from semiconductor_lib.band_diagrams import (
    joyce_dixon_eta, built_in_potential_fd, pn_band_diagram, EG_SI, NC_SI,
    NV_SI, lateral_surface_potential, lateral_phi_uniform_analytic,
    long_channel_phi_s, char_length, threshold_voltage_q2d,
    gaussian_pocket_profile, phi_bulk)
from semiconductor_lib.mosfet import vt_uniform

Vt = thermal_voltage(300.0)


def test_joyce_dixon_nondegenerate_limit():
    assert joyce_dixon_eta(1e15, NC_SI) == pytest.approx(np.log(1e15 / NC_SI),
                                                         abs=1e-4)


def test_joyce_dixon_degenerate_positive():
    # N = 2 Nc is degenerate: E_F above E_c
    assert joyce_dixon_eta(2 * NC_SI, NC_SI) > 0


def test_vbi_matches_boltzmann_when_nondegenerate():
    ni_eff = np.sqrt(NC_SI * NV_SI) * np.exp(-EG_SI / (2 * Vt))
    NA, ND = 1e16, 1e17
    assert built_in_potential_fd(NA, ND) == pytest.approx(
        Vt * np.log(NA * ND / ni_eff**2), abs=2e-3)


@pytest.mark.parametrize("NA,ND,VA", [(1e17, 1e17, 0.0), (1e16, 1e20, 0.5),
                                      (1e20, 1e16, -1.0), (5e19, 5e19, -1.0)])
def test_pn_band_diagram_consistency(NA, ND, VA):
    d = pn_band_diagram(NA, ND, VA)
    # charge neutrality
    assert NA * d['xp'] == pytest.approx(ND * d['xn'], rel=1e-10)
    # total band bending = Vbi - VA (p-side bands higher for VA < Vbi)
    assert d['Ec'][0] - d['Ec'][-1] == pytest.approx(d['Vbi'] - VA, abs=1e-9)
    # quasi-Fermi split equals applied bias
    assert np.nanmax(d['EFn']) - np.nanmax(d['EFp']) == pytest.approx(VA)
    # band edges continuous (no jumps larger than the local slope allows)
    assert np.max(np.abs(np.diff(d['Ec']))) < 0.05 * (d['Vbi'] - VA) + 1e-3


def test_lateral_fd_matches_analytic_uniform():
    tox, N, L, VG, VDS = 2e-7, 2e18, 40e-7, 0.0, 1.1
    s = lateral_surface_potential(L, VG, VDS, lambda y: N * np.ones_like(y),
                                  tox, npts=1601)
    phiL = long_channel_phi_s(VG, N, tox)
    lam = char_length(N, tox)
    phi0 = s['phi_s'][0]
    ana = lateral_phi_uniform_analytic(s['y'], L, phiL, lam, phi0, VDS)
    assert np.max(np.abs(s['phi_s'] - ana)) < 5e-4


def test_long_channel_vt_matches_classical_formula():
    tox, N = 2e-7, 2e18
    Cox = eps_ox / tox
    VFB = -(EG_SI / 2 - phi_bulk(N))     # n+ poly on p-body
    vt_classic = vt_uniform(N, Cox, V_FB=VFB)
    vt_q2d = threshold_voltage_q2d(1e-4, 0.05, lambda y: N * np.ones_like(y),
                                   tox, n_crit=N, npts=2001)
    assert vt_q2d == pytest.approx(vt_classic, abs=3e-3)


def test_dibl_trends_and_pocket_benefit():
    tox, ncrit = 2e-7, 2e18
    uni = lambda y: 2e18 * np.ones_like(y)
    dibl = []
    for L in (100e-7, 60e-7, 40e-7):
        dibl.append(threshold_voltage_q2d(L, 0.05, uni, tox, ncrit)
                    - threshold_voltage_q2d(L, 1.1, uni, tox, ncrit))
    assert dibl[0] < dibl[1] < dibl[2]          # DIBL grows as L shrinks
    L = 40e-7
    pk = gaussian_pocket_profile(L, 1e18, 5e18, 8e-7)
    dibl_pk = (threshold_voltage_q2d(L, 0.05, pk, tox, ncrit)
               - threshold_voltage_q2d(L, 1.1, pk, tox, ncrit))
    assert dibl_pk < dibl[2]                    # pockets suppress DIBL


def test_reverse_short_channel_effect_with_pockets():
    tox, ncrit = 2e-7, 2e18
    vt = [threshold_voltage_q2d(L, 0.05,
                                gaussian_pocket_profile(L, 1e18, 5e18, 8e-7),
                                tox, ncrit)
          for L in (200e-7, 60e-7, 40e-7)]
    assert vt[0] < vt[1] < vt[2]                # V_T rises as L shrinks
