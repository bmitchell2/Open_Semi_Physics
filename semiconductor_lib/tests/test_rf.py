import numpy as np
from semiconductor_lib import rf


def test_tank_and_startup():
    L, C = 1e-9, 1.0132e-12
    f0 = rf.tank_resonance(L, C)
    assert abs(f0 - 5e9) / 5e9 < 1e-3
    Rp = rf.tank_parallel_resistance(15, 5e9, 1e-9)
    assert abs(Rp - 471.2) < 0.5
    assert abs(rf.cross_coupled_min_gm(Rp) - 4.244e-3) < 1e-5


def test_leeson_slopes_and_floor():
    f0, Q, P = 5e9, 10, 1e-3
    # -20 dB/dec between flicker corner and f0/2Q: use region 1e6..1e7 well above fc and below f0/2Q=250 MHz
    a, b = rf.leeson_phase_noise([1e6, 1e7], f0, Q, P, f_flicker=1e3)
    assert abs((b - a) + 20) < 0.2
    # -30 dB/dec well below the flicker corner
    a, b = rf.leeson_phase_noise([1e2, 1e3], f0, Q, P, f_flicker=1e6)
    assert abs((b - a) + 30) < 0.3
    # far-out floor approaches 10log(2FkT/P)
    far = rf.leeson_phase_noise(1e12, f0, Q, P, F=2)
    floor = 10 * np.log10(2 * 2 * rf.K_B * 300 / P)
    assert abs(far - floor) < 0.05
    # doubling Q improves the 1/f^2 region by 6 dB
    l1 = rf.leeson_phase_noise(1e6, f0, 10, P)
    l2 = rf.leeson_phase_noise(1e6, f0, 20, P)
    assert abs((l1 - l2) - 6.02) < 0.05


def test_fom_and_pll():
    assert abs(rf.oscillator_fom(-120, 5e9, 1e6, 1e-3) - (-193.98)) < 0.01
    assert abs(rf.reference_noise_multiplication_db(125) - 41.94) < 0.01


def test_ft_fmax():
    W = 32.0  # um
    gm, cgs, cgd, gds = 1.5e-3 * W, 0.8e-15 * W, 0.35e-15 * W, 0.15e-3 * W
    fT = rf.ft_mosfet(gm, cgs, cgd)
    assert 150e9 < fT < 300e9
    Rg1 = rf.gate_resistance(8, 1e-6, 40e-9, 32, double_contact=False)
    Rg2 = rf.gate_resistance(8, 1e-6, 40e-9, 32, double_contact=True)
    assert abs(Rg1 / Rg2 - 4) < 1e-9
    f1 = rf.fmax_mosfet(fT, Rg1, gds, cgd)
    f2 = rf.fmax_mosfet(fT, Rg2, gds, cgd)
    assert f2 > f1                     # lower Rg -> higher fmax
    assert rf.fmax_mosfet(fT, 1e-9, gds, cgd) > 10 * fT  # Rg->0 limit diverges
    # HBT example: fT 300 GHz, rb 20 ohm, Cbc 5 fF -> ~345 GHz
    assert abs(rf.fmax_hbt(300e9, 20, 5e-15) / 1e9 - 345.5) < 1


def test_passives():
    d = rf.skin_depth(1.72e-8, 1e9)
    assert abs(d * 1e6 - 2.09) < 0.02          # Cu at 1 GHz ~2.1 um
    L3 = rf.spiral_inductance_wheeler(3, 200e-6, 140e-6)
    L6 = rf.spiral_inductance_wheeler(6, 200e-6, 140e-6)
    assert 1e-9 < L3 < 5e-9
    assert abs(L6 / L3 - 4) < 1e-9            # n^2 scaling at fixed geometry
    # skin effect only increases resistance, approaching R_dc at low f
    r_lo = rf.series_resistance_skin(2.0, 3e-6, 1.72e-8, 1e6)
    r_hi = rf.series_resistance_skin(2.0, 3e-6, 1.72e-8, 10e9)
    assert abs(r_lo - 2.0) < 0.05 and r_hi > 2.5
    # dielectric relaxation: R*C = rho*eps
    R, C = rf.substrate_branch(0.1, 4000)
    assert abs(R * C - 0.1 * rf.EPS_SI) < 1e-20


def test_inductor_q_substrate_trend():
    f = np.logspace(8, 11, 400)
    Rs = rf.series_resistance_skin(2.0, 3e-6, 1.72e-8, f)
    q = {}
    for rho in (0.1, 10.0):   # 10 and 1000 ohm-cm in ohm-m
        R, C = rf.substrate_branch(rho, 4000)
        q[rho] = rf.inductor_q_one_port(f, 2e-9, Rs, 100e-15, R, C)
    assert q[10.0].max() > q[0.1].max()       # high-res substrate helps
    # Q crosses zero at self-resonance (inductive -> capacitive)
    assert (q[0.1] < 0).any()


def test_pll_shaping_limits():
    lo = rf.pll_output_phase_noise_db(1e2, -150, -80, 100, 1e5)
    hi = rf.pll_output_phase_noise_db(1e9, -150, -140, 100, 1e5)
    assert abs(lo - (-150 + 40)) < 0.1     # in-band: reference * N^2
    assert abs(hi - (-140)) < 0.1          # out-of-band: free-running VCO
