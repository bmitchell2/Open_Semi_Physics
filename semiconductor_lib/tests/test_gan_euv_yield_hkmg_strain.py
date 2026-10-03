"""Physics checks for modules added with the GaN HEMT, HKMG, strain,
EUV lithography, chiplet-yield and 3D NAND pages."""
import numpy as np
import pytest

from semiconductor_lib import gan_hemt, euv_multilayer, yield_models, hkmg, strain, wordline_rc
from semiconductor_lib.constants import q


def test_gan_polarization_charge_matches_ambacher():
    # Ambacher 2000: sigma/q ~ 1.6-1.7e13 cm^-2 at x = 0.3, spontaneous part > half
    s = gan_hemt.sigma_pol(0.3) / q * 1e-4
    assert 1.5e13 < s < 1.8e13
    share = abs(gan_hemt.p_sp(0.3) - gan_hemt.p_sp(0)) / gan_hemt.sigma_pol(0.3)
    assert 0.5 < share < 0.65


def test_gan_sheet_density_trends():
    # increases with d and x, saturates below sigma/q, zero below critical thickness
    n10, n30 = gan_hemt.sheet_density(0.3, 10), gan_hemt.sheet_density(0.3, 30)
    assert 0 < n10 < n30 < gan_hemt.sigma_pol(0.3) / q * 1e-4
    assert gan_hemt.sheet_density(0.25, 20) < gan_hemt.sheet_density(0.34, 20)
    dcr = gan_hemt.critical_thickness(0.3)
    assert gan_hemt.sheet_density(0.3, 0.9 * dcr) == 0.0
    assert gan_hemt.sheet_density(0.3, 1.5 * dcr) > 0.0
    # thick-barrier value in the measured 1e13 range
    assert 1.2e13 < n30 < 1.6e13


def test_euv_multilayer_theoretical_peak():
    R = euv_multilayer.multilayer_reflectivity(np.linspace(13.2, 13.8, 121), 60, 6.9)
    assert 0.72 < R.max() < 0.76          # ideal Mo/Si limit ~74-75 %
    # monotonic rise then saturation with bilayer count
    r = [euv_multilayer.multilayer_reflectivity(13.5, n, 6.9) for n in (10, 20, 40, 80)]
    assert r[0] < r[1] < r[2] <= r[3] + 1e-3
    assert r[3] - r[2] < 0.03
    assert abs(euv_multilayer.chain_throughput(0.7, 10) - 0.7**10) < 1e-12


def test_yield_models_limits():
    assert yield_models.yield_poisson(0, 0.1) == pytest.approx(1.0)
    A = np.array([0.5, 2, 8])
    p, m, nb = (yield_models.yield_poisson(A, 0.1), yield_models.yield_murphy(A, 0.1),
                yield_models.yield_negbin(A, 0.1, 3))
    assert np.all(p <= m + 1e-12) and np.all(m <= nb + 1e-12)   # clustering helps large dies
    assert yield_models.yield_negbin(8, 0.1, 1e6) == pytest.approx(np.exp(-0.8), rel=1e-4)
    mono = yield_models.good_silicon_area_per_product(8, 1, 0.1)
    four = yield_models.good_silicon_area_per_product(8, 4, 0.1)
    assert four < mono
    # zero defects: chiplets only cost overhead
    assert yield_models.good_silicon_area_per_product(8, 4, 0.0, d2d_overhead=0.1) == pytest.approx(8.8)


def test_hkmg_eot_and_vt():
    assert hkmg.eot(0.6e-7, 2e-7, 20) == pytest.approx(0.99e-7)
    # reproduces the MOS Capacitor page worked example (n+ poly, 5 nm, 1e17): +0.10 V
    assert hkmg.vt_nmos(4.05, 1e17, 5e-7) == pytest.approx(0.10, abs=0.01)
    # NMOS/PMOS symmetry about midgap
    assert hkmg.vt_nmos(4.61, 3e17, 1e-7) == pytest.approx(-hkmg.vt_pmos(4.61, 3e17, 1e-7), abs=1e-9)
    # dVt/dphi_m = 1
    assert hkmg.vt_nmos(4.5, 3e17, 1e-7) - hkmg.vt_nmos(4.4, 3e17, 1e-7) == pytest.approx(0.1)


def test_strain_signs():
    assert strain.mobility_change("n", sigma_L_Pa=+1e8) > 0      # tensile helps electrons
    assert strain.mobility_change("p", sigma_L_Pa=-1e8) > 0      # compressive helps holes
    assert strain.mobility_change("p", sigma_L_Pa=+1e8) < 0
    assert strain.mobility_change("p", sigma_L_Pa=-1e9) == pytest.approx(0.718)


def test_wordline_rc_scaling():
    R1 = wordline_rc.wordline_resistance(1e-7, 1e-3, 4e-6, 30e-9)
    R2 = wordline_rc.wordline_resistance(1e-7, 2e-3, 4e-6, 30e-9)
    t1 = wordline_rc.distributed_rc_delay(R1, 1e-12)
    t2 = wordline_rc.distributed_rc_delay(R2, 2e-12)
    assert t2 / t1 == pytest.approx(4.0)       # L^2
