import numpy as np
from semiconductor_lib import leakage, multigate
from semiconductor_lib.constants import q, eps_si


def test_neff_limits():
    assert np.isclose(leakage.n_eff(1e16, 1e20), 1e16, rtol=1e-3)
    assert np.isclose(leakage.n_eff(1e18, 1e18), 5e17)


def test_peak_field_matches_depletion_approx():
    # symmetric check against E = 2 V / W
    NA, ND, VR = 1e16, 5e16, 2.5
    Vbi = leakage.built_in_potential(NA, ND)
    W = np.sqrt(2 * eps_si / q * (1/NA + 1/ND) * (Vbi + VR))
    assert np.isclose(leakage.peak_field(NA, ND, VR), 2 * (Vbi + VR) / W, rtol=1e-9)
    # value in the PN Junction Electrostatics worked example: 91.6 kV/cm
    assert abs(leakage.peak_field(NA, ND, VR) - 91.6e3) / 91.6e3 < 0.01


def test_btbt_constant_and_monotonic():
    B = leakage.btbt_B(0.2, 1.12)
    assert 3.5e7 < B < 3.7e7
    E = np.logspace(5.5, 6.5, 20)
    J = leakage.btbt_current_density(E, 1.0)
    assert np.all(np.diff(J) > 0)
    # ~1 A/cm^2 near 2 MV/cm for m* = 0.2 m0 (Taur & Ning magnitude)
    J2 = leakage.btbt_current_density(2e6, 1.0)
    assert 0.1 < J2 < 10


def test_btbt_rises_with_pocket_doping():
    Np = np.array([1e18, 3e18, 1e19])
    E = leakage.peak_field(Np, 1e20, 1.2)
    J = leakage.btbt_current_density(E, 1.2, B_override=2.0e7)
    assert np.all(np.diff(J) > 0)
    assert J[-1] / J[0] > 1e3


def test_gidl_surface_field():
    # 2 nm oxide, V_DG = 2.4 V: (2.4-1.2)/(3 x 2e-7) = 2 MV/cm
    Es = leakage.gidl_surface_field(2.4, 2e-7, eps_ratio=3.0)
    assert np.isclose(Es, 2.0e6)
    assert leakage.gidl_surface_field(1.0, 2e-7) == 0.0


def test_natural_length():
    # DG, t_si = 10 nm, t_ox = 1 nm: sqrt(3*10*1/2) = 3.87 nm
    lam = multigate.natural_length(10e-7, 1e-7, 2) / 1e-7
    assert abs(lam - np.sqrt(3.0 * 10 / 2)) < 0.02
    # more gates -> shorter lambda; thinner body -> shorter lambda
    assert multigate.natural_length(10e-7, 1e-7, 3) < multigate.natural_length(10e-7, 1e-7, 2)
    assert multigate.natural_length(5e-7, 1e-7, 2) < multigate.natural_length(10e-7, 1e-7, 2)
    assert np.isclose(multigate.min_gate_length(10e-7, 1e-7, 2),
                      5 * multigate.natural_length(10e-7, 1e-7, 2))
