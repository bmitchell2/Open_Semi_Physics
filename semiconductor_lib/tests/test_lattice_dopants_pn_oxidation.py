"""
Physics validation tests for the lattice, dopants, pnjunction and oxidation
modules. Each test encodes a check that was verified by hand while building
the corresponding Semiconductor Notes pages (session 2026-09-28).
"""
import numpy as np
import pytest

from semiconductor_lib import lattice, dopants, pnjunction, oxidation
from semiconductor_lib.constants import q, eps_si


# ------------------------------------------------------------------ lattice
def test_neighbor_shells_A_and_B_sites():
    for site in [(0, 0, 0), (.25, .25, .25)]:
        s = lattice.neighbor_shells(site)
        assert s[0] == (2.352, 4)
        assert s[1] == (3.84, 12)
        assert s[2] == (4.503, 12)


def test_body_centre_is_vacant_tetrahedral_site():
    assert not lattice.is_atom_site((.5, .5, .5))
    s = lattice.neighbor_shells((.5, .5, .5), n_shells=2)
    assert s[0] == (2.352, 4)          # same distance as a bond
    assert s[1] == (2.716, 6)


def test_bulk_numbers():
    assert lattice.bond_length() == pytest.approx(2.3517, abs=1e-3)
    assert lattice.atomic_density() == pytest.approx(4.994e22, rel=1e-3)
    assert lattice.mass_density() == pytest.approx(2.329, abs=2e-3)
    assert lattice.packing_fraction() == pytest.approx(0.340, abs=1e-3)


def test_planar_density_counted_matches_analytic():
    for hkl in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]:
        counted = lattice.planar_density_counted(hkl)
        analytic = lattice.planar_density_analytic(hkl)
        # every atomic plane holds one plane's worth of atoms; finite-disc counting noise is about 2%
        for _, dens in counted:
            assert dens == pytest.approx(analytic, rel=0.03)


def test_planar_density_values_and_111_bilayer_spacing():
    assert lattice.planar_density_analytic((1, 0, 0)) == pytest.approx(6.78e14, rel=2e-3)
    assert lattice.planar_density_analytic((1, 1, 0)) == pytest.approx(9.59e14, rel=2e-3)
    assert lattice.planar_density_analytic((1, 1, 1)) == pytest.approx(7.83e14, rel=2e-3)
    offs = np.array([o for o, _ in lattice.planar_density_counted((1, 1, 1))])
    gaps = np.round(np.diff(offs), 2)
    assert set(gaps) == {0.78, 2.35}                 # bilayer: 0.784 A within, 2.352 A between
    assert list(gaps[:4]) in ([2.35, 0.78, 2.35, 0.78], [0.78, 2.35, 0.78, 2.35])


def test_average_over_cut_position_equals_bulk_density():
    """sigma / layer spacing is the bulk density for every orientation."""
    a_cm = lattice.A_SI * 1e-8
    n_bulk = lattice.atomic_density()
    assert lattice.planar_density_analytic((1, 0, 0)) / (a_cm / 4) == pytest.approx(n_bulk, rel=1e-3)
    assert lattice.planar_density_analytic((1, 1, 0)) / (a_cm / (2 * np.sqrt(2))) == pytest.approx(n_bulk, rel=1e-3)
    # (111): a bilayer (2 planes) per repeat length a/sqrt(3)
    assert 2 * lattice.planar_density_analytic((1, 1, 1)) / (a_cm / np.sqrt(3)) == pytest.approx(n_bulk, rel=1e-3)


# ------------------------------------------------------------------ dopants
def test_band_gap_values():
    assert dopants.band_gap(300) == pytest.approx(1.1245, abs=2e-4)
    assert dopants.band_gap(77) == pytest.approx(1.166, abs=2e-3)
    assert dopants.band_gap(400) == pytest.approx(1.097, abs=2e-3)


def test_dos_and_ni_at_300K():
    assert dopants.Nc(300) == pytest.approx(3.2e19, rel=0.02)
    assert dopants.Nv(300) == pytest.approx(1.8e19, rel=0.02)
    assert 8e9 < dopants.intrinsic_concentration(300) < 1.1e10


def test_fermi_level_position_n_type():
    """Ec-EF = kT ln(Nc/ND) when fully ionized: 0.209 eV at 1e16 cm^-3."""
    r = dopants.solve_donor(1e16, 300)
    assert r.Ec_minus_EF == pytest.approx(0.209, abs=0.005)
    assert r.n == pytest.approx(1e16, rel=0.01)


def test_phosphorus_ionization_300K():
    assert dopants.solve_donor(1e16, 300).fraction == pytest.approx(0.9965, abs=0.002)
    assert dopants.solve_donor(1e17, 300).fraction == pytest.approx(0.967, abs=0.005)


def test_freeze_out_phosphorus_1e16():
    assert dopants.solve_donor(1e16, 77).fraction == pytest.approx(0.38, abs=0.03)
    assert dopants.solve_donor(1e16, 50).fraction < 0.10


def test_boron_less_ionized_than_phosphorus():
    """Same energy, degeneracy 4 vs 2: acceptors are less completely ionized at 1e17."""
    assert dopants.solve_acceptor(1e17, 300).fraction == pytest.approx(0.899, abs=0.01)
    assert dopants.solve_acceptor(1e17, 300).fraction < dopants.solve_donor(1e17, 300).fraction


def test_mass_action_law_in_solution():
    r = dopants.solve_donor(1e16, 300)
    assert r.n * r.p == pytest.approx(dopants.intrinsic_concentration(300) ** 2, rel=1e-6)


def test_hydrogenic_estimates():
    assert dopants.hydrogenic_binding_energy_meV(0.26, 11.7) == pytest.approx(25.8, abs=0.2)
    assert dopants.hydrogenic_radius_A(0.26, 11.7) == pytest.approx(23.8, abs=0.2)


# ------------------------------------------------------------------ p-n junction
@pytest.fixture(scope="module")
def junction():
    NA, ND = 1e16, 5e16
    out = {}
    for VA in (0.0, 0.5, -2.5):
        out[VA] = (pnjunction.solve_pn_poisson(NA, ND, VA),
                   pnjunction.depletion_approximation(NA, ND, VA))
    return NA, ND, out


def test_depletion_approximation_reference_numbers(junction):
    NA, ND, out = junction
    d = out[0.0][1]
    assert d["Vbi"] == pytest.approx(0.756, abs=2e-3)
    assert d["W"] * 1e4 == pytest.approx(0.342, abs=2e-3)
    assert d["xp"] * 1e4 == pytest.approx(0.285, abs=2e-3)
    assert d["Emax"] == pytest.approx(4.41e4, rel=0.01)
    assert d["C_per_area"] * 1e9 == pytest.approx(30.3, abs=0.3)


def test_solver_converges_and_is_neutral(junction):
    NA, ND, out = junction
    for VA, (sol, _) in out.items():
        assert sol.residual < 1e-6
        xp, xn = pnjunction.numerical_widths(sol, NA, ND)
        assert abs(NA * xp - ND * xn) / (NA * xp) < 1e-3


def test_potential_drop_equals_Vbi_minus_VA(junction):
    NA, ND, out = junction
    for VA, (sol, d) in out.items():
        assert sol.psi[-1] - sol.psi[0] == pytest.approx(d["Vbi"] - VA, abs=1e-6)


def test_mass_action_holds_at_metallurgical_junction(junction):
    """Depletion region is empty because of the barrier, not recombination: np = ni^2 at x=0."""
    _, _, out = junction
    sol = out[0.0][0]
    i0 = np.argmin(np.abs(sol.x))
    assert sol.n[i0] * sol.p[i0] / 1e20 == pytest.approx(1.0, abs=1e-3)


def test_numerical_width_is_below_depletion_approximation(junction):
    """Numerical xp / approximate xp - 1: about -3.4% at 0 V, -7.4% at +0.5 V, -0.8% at -2.5 V.
    (Equivalently the approximation exceeds the numerical width by 3.5%, 8.0%, 0.8%.)"""
    NA, ND, out = junction
    for VA, expected in [(0.0, -0.034), (0.5, -0.074), (-2.5, -0.008)]:
        sol, d = out[VA]
        xp_num, _ = pnjunction.numerical_widths(sol, NA, ND)
        assert xp_num / d["xp"] - 1 == pytest.approx(expected, abs=0.003)


# ------------------------------------------------------------------ oxidation
def test_tau_matches_deal_grove_paper():
    A, B = oxidation.rate_constants("dry", "111")
    assert oxidation.tau_from_initial_oxide(0.023, A, B) == pytest.approx(0.37, abs=0.005)


def test_inverse_consistency():
    A, B = oxidation.rate_constants("wet", "100")
    t = np.array([0.1, 1.0, 5.0])
    assert oxidation.time_to_grow(oxidation.thickness(t, A, B), A, B) == pytest.approx(t)


def test_short_and_long_time_limits():
    A, B = oxidation.rate_constants("wet", "111")
    assert oxidation.thickness(1e-3, A, B) == pytest.approx(oxidation.linear_rate(A, B) * 1e-3, rel=0.01)
    assert oxidation.thickness(1e3, A, B) == pytest.approx(np.sqrt(B * 1e3), rel=0.07)


def test_orientation_only_changes_linear_constant():
    A111, B111 = oxidation.rate_constants("dry", "111")
    A100, B100 = oxidation.rate_constants("dry", "100")
    assert B111 == B100
    assert oxidation.linear_rate(A111, B111) / oxidation.linear_rate(A100, B100) == pytest.approx(1.68)


def test_worked_examples_from_note():
    A, B = oxidation.rate_constants("wet", "100")
    assert float(oxidation.time_to_grow(0.5, A, B)) == pytest.approx(1.53, abs=0.01)
    A, B = oxidation.rate_constants("wet", "111")
    assert float(oxidation.time_to_grow(0.5, A, B)) == pytest.approx(1.26, abs=0.01)
    assert float(oxidation.thickness(1.0, A, B)) == pytest.approx(0.435, abs=0.002)


def test_silicon_consumption_ratio():
    assert float(oxidation.silicon_consumed(1.0)) == pytest.approx(0.45, abs=0.002)
