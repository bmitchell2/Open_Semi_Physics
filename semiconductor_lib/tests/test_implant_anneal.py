import numpy as np
from semiconductor_lib import implant_anneal as ia


def test_boron_diffusivity_values():
    # 1000 C, 10 s -> sqrt(Dt) ~ 3.9 nm (matches USJ page worked example)
    D = ia.boron_intrinsic_diffusivity(1000.0)
    assert abs(np.sqrt(D * 10) * 1e7 - 3.9) < 0.2
    # monotonic in T
    assert ia.boron_intrinsic_diffusivity(900) < ia.boron_intrinsic_diffusivity(1000)


def test_gaussian_dose():
    x = np.linspace(0, 200, 2001)
    c = ia.gaussian_profile(x, 1e14, 60, 15)
    assert abs(ia.integrated_dose(x, c) / 1e14 - 1) < 1e-3


def test_ted_budget_limits():
    D = 1e-16
    assert ia.ted_budget(D, 100, 0, 10) == D * 100
    # long anneal: extra budget saturates at D*S0*tau
    assert abs(ia.ted_budget(D, 1e6, 1e3, 10) - (D * 1e6 + D * 1e3 * 10)) / (D * 1e6 + 1e4 * D) < 1e-6


def test_relative_dissolution_time():
    assert ia.relative_dissolution_time(800, 800) == 1.0
    assert ia.relative_dissolution_time(900, 800) < 1.0
    # 3.8 eV: 100 C step near 800 C shortens time by ~40x
    r = ia.relative_dissolution_time(900, 800)
    assert 0.01 < r < 0.05


def test_diffuse_conserves_dose_and_ted_spreads_more():
    x = np.linspace(0, 400, 801)
    c0 = ia.gaussian_profile(x, 1e14, 30, 10)
    eq = ia.diffuse_ted(x, c0, 800, 1800, S0=0)
    ted = ia.diffuse_ted(x, c0, 800, 1800, S0=5000, tau=300)
    d0 = ia.integrated_dose(x, c0)
    for c in (eq, ted):
        assert abs(ia.integrated_dose(x, c) / d0 - 1) < 5e-3
    def depth(c, level=1e17):
        return x[np.where(c > level)[0][-1]]
    assert depth(ted) > depth(eq) + 20


def test_cluster_pins_peak():
    x = np.linspace(0, 400, 801)
    c0 = ia.gaussian_profile(x, 1e14, 30, 10)
    ted = ia.diffuse_ted(x, c0, 800, 1800, S0=5000, tau=300, c_cluster=1e19)
    i = np.argmax(c0)
    assert ted[i] >= 1e19 * 0.99  # peak held at or above the cluster level
    assert abs(ia.integrated_dose(x, ted) / ia.integrated_dose(x, c0) - 1) < 5e-3


def test_nitrogen_conservation_and_interface_growth():
    x = np.linspace(0, 300, 301)
    n0 = ia.gaussian_profile(x, 5e13, 60, 20)
    times = [60, 600, 3600]
    prof, q = ia.nitrogen_trap_limited(x, n0, 1e-3, 1e-12, times)
    for p, qi in zip(prof, q):
        assert abs((ia.integrated_dose(x, p) + qi) / 5e13 - 1) < 0.02
    assert np.all(np.diff(q) > 0)
    # peak drops with time
    assert prof[-1].max() < prof[0].max()


def test_nitrogen_interface_saturation():
    x = np.linspace(0, 300, 301)
    n0 = ia.gaussian_profile(x, 5e13, 60, 20)
    _, q = ia.nitrogen_trap_limited(x, n0, 1e-2, 1e-11, [3600], q_max=2e13)
    assert q[-1] <= 2e13 * 1.0001


def test_eor_depth():
    x = np.linspace(0, 100, 1001)
    d = ia.gaussian_profile(x, 1e15, 20, 8) * 100  # displaced atoms (~100 per ion)
    z = ia.eor_depth(x, d, 1.15e22)
    assert z is not None and z > 20
    assert ia.eor_depth(x, d * 1e-3, 1.15e22) is None
