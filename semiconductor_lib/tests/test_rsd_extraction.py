import numpy as np
from semiconductor_lib.rsd_extraction import (channel_resistivity, r_total_linear,
                                              fit_lines, extract_rsd_dl)

W, MU, COX = 1e-4, 250.0, 1.57e-6
LM = np.array([40, 50, 70, 100, 150, 250]) * 1e-7
VGT = np.array([0.4, 0.6, 0.8, 1.0])


def _arr(R_sd, dL, rsd_of_v=None):
    rows = []
    for v in VGT:
        r = R_sd if rsd_of_v is None else rsd_of_v(v)
        rows.append(r_total_linear(LM, v, W, MU, COX, r, dL))
    return np.array(rows)


def test_exact_recovery_overlap():
    R_sd, dL, *_ , res = extract_rsd_dl(LM, _arr(250.0, 10e-7))
    assert abs(R_sd - 250.0) < 1e-6 and abs(dL - 10e-7) < 1e-12 and res < 1e-6


def test_underlap_gives_negative_dl():
    _, dL, *_ = extract_rsd_dl(LM, _arr(250.0, -6e-7))
    assert dL < 0 and abs(dL + 6e-7) < 1e-12


def test_slope_scales_inverse_overdrive():
    rho, _ = fit_lines(LM, _arr(250.0, 10e-7))
    assert np.allclose(rho * VGT, rho[0] * VGT[0])
    assert np.allclose(rho, channel_resistivity(VGT, W, MU, COX))


def test_lines_cross_at_dl_rsd():
    R = _arr(250.0, 10e-7)
    vals = [r_total_linear(10e-7, v, W, MU, COX, 250.0, 10e-7) for v in VGT]
    assert np.allclose(vals, 250.0)
    assert R.shape == (len(VGT), len(LM))


def test_bias_dependent_rsd_flagged_by_residual():
    gap = lambda v: 250.0 + 300.0 * (0.4 / v) ** 2
    *_, res = extract_rsd_dl(LM, _arr(250.0, -6e-7, gap))
    assert res > 5.0   # clearly non-collinear (ohm)
