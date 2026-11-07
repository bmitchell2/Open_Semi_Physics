import numpy as np
from scipy.integrate import quad
from semiconductor_lib import layout_effects as le


def test_inv_sa_single_and_multifinger():
    assert np.isclose(le.inv_sa(0.3, 0.1), 1 / 0.35)
    # NF = 3: average of the three finger terms (BSIM4 definition)
    exp = np.mean([1 / (0.3 + 0.05 + i * (0.2 + 0.1)) for i in range(3)])
    assert np.isclose(le.inv_sa(0.3, 0.1, nf=3, sd=0.2), exp)
    # per-finger terms are symmetric and largest at the outer fingers
    t = le.finger_stress_terms(0.3, 0.3, 0.1, 6, 0.2)
    assert np.allclose(t, t[::-1]) and t[0] > t[2]


def test_lod_mobility_limits_and_sign():
    sa = np.array([0.1, 0.3, 1.0, 10.0, 1e6])
    r_n = le.lod_mobility_ratio(sa, sa, 0.04, ku0=-0.01)
    r_p = le.lod_mobility_ratio(sa, sa, 0.04, ku0=+0.01)
    assert np.all(np.diff(r_n) > 0) and np.all(np.diff(r_p) < 0)
    assert np.isclose(r_n[-1], 1.0, atol=1e-6) and np.isclose(r_p[-1], 1.0, atol=1e-6)
    # equals 1 at the reference geometry
    assert np.isclose(le.lod_mobility_ratio(0.5, 0.5, 0.04, -0.01, 0.5, 0.5), 1.0)
    assert np.isclose(le.lod_delta_vth(0.5, 0.7, 0.04, 0.003, 0.5, 0.7), 0.0)


def test_wpe_integrals_match_quadrature():
    d, scref = 0.8, 1.0
    for sc in (0.2, 0.7, 3.0):
        a = quad(lambda s: scref**2 / s**2, sc, sc + d)[0] / d
        b = quad(lambda s: s / scref * np.exp(-10 * s / scref), sc, sc + d)[0] / d
        c = quad(lambda s: s / scref * np.exp(-20 * s / scref), sc, sc + d)[0] / d
        assert np.isclose(le.wpe_sca(sc, d, scref), a, rtol=1e-9)
        assert np.isclose(le.wpe_scb(sc, d, scref), b, rtol=1e-9)
        assert np.isclose(le.wpe_scc(sc, d, scref), c, rtol=1e-9)


def test_wpe_point_device_limit_and_monotonic():
    sc = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
    assert np.allclose(le.wpe_sca(sc, 1e-6), 1 / sc**2, rtol=1e-4)
    dv = le.wpe_delta_vth(sc, 1.0, 0.02, web=1.0, wec=1.0)
    assert np.all(np.diff(dv) < 0) and np.all(dv > 0)
