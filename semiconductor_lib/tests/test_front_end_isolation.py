"""Physics checks for implant_geometry, alignment_marks, field_isolation
and the <100> strain coefficients."""
import numpy as np
from semiconductor_lib import implant_geometry as ig
from semiconductor_lib import alignment_marks as am
from semiconductor_lib import field_isolation as fi
from semiconductor_lib import strain


def test_critical_angle_vs_7_degree_tilt():
    # B and P from 10 keV up: axial <100> channels are narrower than a 7 deg tilt
    for Z in (5, 15):
        for E in (1e4, 3e4, 1e5):
            psi = float(ig.critical_angle_deg(Z, E, "100"))
            assert 1.0 < psi < 7.0, (Z, E, psi)
    # ultra-low energy: the estimated channel acceptance exceeds 7 deg,
    # which is why sub-keV implants rely on amorphization, not tilt
    assert float(ig.critical_angle_deg(5, 1e3, "100")) > 7.0


def test_critical_angle_trends():
    # falls with energy, rises with ion Z, wider along <110> than <100>
    assert ig.critical_angle_deg(5, 1e5) < ig.critical_angle_deg(5, 1e4)
    assert ig.critical_angle_deg(33, 1e4) > ig.critical_angle_deg(5, 1e4)
    assert ig.critical_angle_deg(5, 1e4, "110") > ig.critical_angle_deg(5, 1e4, "100")


def test_shadow_length_7deg_150nm():
    assert abs(ig.shadow_length(150.0, 7.0) - 18.42) < 0.05
    assert ig.shadow_length(150.0, 7.0, 180.0) == 0.0


def test_quad_mode_edge_and_centre_dose():
    x = np.array([1.0, 100.0, 199.0])
    d = ig.surface_dose_in_gap(x, 200.0, 150.0, 7.0)
    assert np.allclose(d, [0.75, 1.0, 0.75])
    single = ig.surface_dose_in_gap(x, 200.0, 150.0, 7.0, twists_deg=(0.0,))
    assert np.allclose(single, [0.0, 1.0, 1.0])     # one-sided shadow


def test_quad_total_dose_conserved_when_open():
    x = np.linspace(0, 1000, 2001)
    d = ig.surface_dose_in_gap(x, 1000.0, 0.0, 7.0)
    assert np.allclose(d, 1.0)


def test_inside_corner_half_dose():
    assert ig.inside_corner_dose(150.0, 7.0) == 0.5


def test_min_open_space_halo():
    assert abs(ig.min_open_space(150.0, 30.0) - 86.6) < 0.1


def test_quarter_wave_and_null():
    d = am.quarter_wave_depth(633.0)
    assert abs(d - 158.25) < 1e-9
    assert abs(am.first_order_efficiency(d, 633.0) - (2 / np.pi) ** 2) < 1e-12
    assert am.first_order_efficiency(2 * d, 633.0) < 1e-20


def test_epi_edge_motion_symmetric_and_shift():
    assert am.epi_edge_motion(5.0, -0.2, 0.2) == (0.0, 2.0)   # widens, no shift
    c, w = am.epi_edge_motion(5.0, 0.6, 0.6)
    assert abs(c - 3.0) < 1e-12 and abs(w) < 1e-12           # pure shift


def test_field_threshold_magnitude_and_charge():
    vt = fi.field_threshold(1e17, 300e-7)
    assert 10.0 < vt < 20.0
    dv = vt - fi.field_threshold(1e17, 300e-7, Qf_cm2=1e12)
    Cox = fi.eps_ox / 300e-7
    assert abs(dv - fi.q * 1e12 / Cox) < 1e-9


def test_field_threshold_rises_with_doping_and_thickness():
    assert fi.field_threshold(3e17, 300e-7) > fi.field_threshold(1e17, 300e-7)
    assert fi.field_threshold(1e17, 400e-7) > fi.field_threshold(1e17, 300e-7)


def test_depletion_width():
    w = fi.depletion_width_cm(1e17, 5.0)
    assert 0.25e-4 < w < 0.30e-4


def test_strain_100_direction():
    s = -300e6
    gain_110 = strain.mobility_change("p", s)
    gain_100 = strain.mobility_change("p", s, direction="100")
    assert gain_110 > 0.2 and 0 < gain_100 < 0.03
    assert strain.mobility_change("n", 300e6, direction="100") > strain.mobility_change("n", 300e6)
