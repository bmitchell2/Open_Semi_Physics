import numpy as np
import pytest
from semiconductor_lib.implant_profiles import (profile_moments, pearson4_profile,
    pearson4_beta_min, gaussian_peak, lss_projected_range)

X = np.linspace(-300, 600, 9001)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def test_gaussian_limit_moments_and_peak():
    c = pearson4_profile(X, 1e15, 100, 40, 0.0, 3.0)
    Rp, dRp, g, b = profile_moments(X, c)
    assert Rp == pytest.approx(100, abs=0.1) and dRp == pytest.approx(40, rel=1e-3)
    assert abs(g) < 1e-3 and b == pytest.approx(3, abs=1e-2)
    assert c.max() == pytest.approx(gaussian_peak(1e15, 40), rel=1e-3)


@pytest.mark.parametrize("g,b", [(-0.6, 3.8), (0.5, 3.6), (-0.3, 3.5)])
def test_pearson4_reproduces_moments_and_dose(g, b):
    c = pearson4_profile(X, 1e15, 100, 40, g, b)
    Rp, dRp, gg, bb = profile_moments(X, c)
    assert Rp == pytest.approx(100, abs=0.5) and dRp == pytest.approx(40, rel=0.01)
    assert gg == pytest.approx(g, abs=0.02) and bb == pytest.approx(b, abs=0.05)
    assert _trapz(c, X * 1e-7) == pytest.approx(1e15, rel=1e-6)


def test_skew_sign_sets_peak_side_of_Rp():
    neg = pearson4_profile(X, 1e15, 100, 40, -0.6, 3.8)   # B-like: tail to surface
    pos = pearson4_profile(X, 1e15, 100, 40, 0.5, 3.6)    # heavy-ion-like
    assert X[neg.argmax()] > 100 and X[pos.argmax()] < 100


def test_beta_bound_enforced():
    assert pearson4_beta_min(0.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        pearson4_profile(X, 1e15, 100, 40, -0.9, 3.5)


def test_lss_trends_and_accuracy_vs_tables():
    B = lss_projected_range(5, 11.009, 100)
    As = lss_projected_range(33, 74.92, 100)
    assert B[1] < B[0] and As[1] < As[0]                  # projection shortens
    assert As[1] / As[0] > B[1] / B[0]                    # heavy ions travel straighter
    # Gibbons-type tabulated Rp at 100 keV (nm): B 307, P 123, As 58, Sb 43.
    for (Z, M), ref in {(5, 11.009): 307, (15, 30.974): 123,
                        (33, 74.92): 58, (51, 121.76): 43}.items():
        Rp = lss_projected_range(Z, M, 100)[1]
        assert 1.0 < Rp / ref < 1.45                       # known 10-40 % overestimate
    Es = [10, 20, 50, 100, 200]
    rps = [lss_projected_range(15, 30.974, E)[1] for E in Es]
    assert all(np.diff(rps) > 0)
