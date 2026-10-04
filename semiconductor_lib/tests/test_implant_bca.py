"""Physics checks for the amorphous-target BCA Monte Carlo (implant_bca)."""
import numpy as np
import pytest
from semiconductor_lib import implant_bca as bca

B = (5, 11.009)
P = (15, 30.97)
AS = (33, 74.92)


@pytest.mark.parametrize("eps", [0.01, 0.1, 1.0, 10.0])
def test_tabulated_angles_reproduce_zbl_nuclear_stopping(eps):
    b = np.linspace(1e-4, 40, 40000)
    th = bca.theta_cm(np.full(b.size, eps), b)
    sn = 2 * eps * np.trapezoid(np.sin(th / 2) ** 2 * b, b)
    zbl = np.log(1 + 1.1383 * eps) / (2 * (eps + 0.01321 * eps ** 0.21226 + 0.19593 * eps ** 0.5))
    assert abs(sn / zbl - 1) < 0.05


@pytest.mark.parametrize("ion,E,ref,tol", [(B, 100e3, 300, 0.10), (P, 100e3, 120, 0.10), (AS, 100e3, 58, 0.15)])
def test_ranges_in_si_match_reference(ion, E, ref, tol):
    r = bca.screen_oxide_split(*ion, E, 0, n=1500)
    assert abs(r["Rp_nm"] / ref - 1) < tol


def test_oxide_shifts_profile_by_about_its_thickness():
    r0 = bca.screen_oxide_split(*AS, 5000, 0, n=4000)
    r3 = bca.screen_oxide_split(*AS, 5000, 30, n=4000)
    shift_A = (r0["Rp_nm"] - r3["Rp_nm"]) * 10
    assert 0.8 < shift_A / 30 < 1.2


def test_dose_in_si_falls_monotonically_with_oxide_for_sub_kev_boron():
    si = [bca.screen_oxide_split(*B, 500, t, n=6000)["silicon"] for t in (0, 10, 20, 30)]
    assert all(a > b for a, b in zip(si, si[1:]))
    assert si[-1] < 0.6  # roughly half the 500 eV dose stays in a 3 nm oxide


def test_fractions_sum_to_one():
    r = bca.screen_oxide_split(*B, 1000, 20, n=2000)
    assert abs(r["back"] + r["oxide"] + r["silicon"] - 1) < 1e-12
