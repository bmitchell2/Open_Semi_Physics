"""
Physics validation for scattering.py, phonons.py, and the degeneracy-ratio
addition to carriers.py. Encodes numbers verified while writing the
"Ionized Impurity Scattering," "Phonons," and "Degenerate Semiconductors"
notes (session 2026-09-29).
"""
import numpy as np
import pytest

from semiconductor_lib import carriers, scattering, phonons


# ------------------------------------------------------------- scattering
def test_caughey_thomas_lightly_doped_matches_lattice_mobility():
    # As N -> 0 at 300 K, Caughey-Thomas must reduce to mu_max (~1471,
    # close to the independently documented 1400-1417 cm^2/Vs lattice value).
    mu = scattering.caughey_thomas(1e10, 300.0, scattering.CT_ELECTRONS)
    assert mu == pytest.approx(1471.0, rel=1e-3)


def test_caughey_thomas_heavily_doped_drops_below_100():
    # N=1e20 cm^-3 is well past degenerate doping; total mobility should
    # be far below the lattice-limited value (order 60-70 cm^2/Vs).
    mu = scattering.caughey_thomas(1e20, 300.0, scattering.CT_ELECTRONS)
    assert 40.0 < mu < 90.0


def test_caughey_thomas_monotonic_in_doping():
    Ns = np.array([1e15, 1e16, 1e17, 1e18, 1e19, 1e20])
    mus = scattering.caughey_thomas(Ns, 300.0, scattering.CT_ELECTRONS)
    assert np.all(np.diff(mus) < 0)


def test_matthiessen_combine_reduces_mobility():
    combined = scattering.matthiessen_combine(1000.0, 500.0)
    # 1/combined = 1/1000 + 1/500 -> combined = 333.33
    assert combined == pytest.approx(1000.0 * 500.0 / 1500.0, rel=1e-6)
    assert combined < min(1000.0, 500.0)


def test_matthiessen_impurity_component_positive_when_doped():
    mu_total_300 = scattering.caughey_thomas(1e18, 300.0, scattering.CT_ELECTRONS)
    mu_I = scattering.matthiessen_impurity_component(mu_total_300)
    assert mu_I > 0
    # Recombining lattice and this implied impurity term must reproduce
    # the original total mobility (self-consistency of the decomposition).
    recombined = scattering.matthiessen_combine(scattering.MU_LATTICE_300_ELECTRONS, mu_I)
    assert recombined == pytest.approx(mu_total_300, rel=1e-6)


def test_lattice_mobility_increases_as_temperature_falls():
    assert scattering.lattice_mobility(100.0) > scattering.lattice_mobility(300.0)


def test_impurity_mobility_idealized_decreases_as_temperature_falls():
    mu_I_300 = 500.0
    assert scattering.impurity_mobility_idealized(100.0, mu_I_300) < mu_I_300
    assert scattering.impurity_mobility_idealized(300.0, mu_I_300) == pytest.approx(mu_I_300)


# --------------------------------------------------------------- phonons
def test_diatomic_chain_acoustic_branch_vanishes_at_zone_center():
    omega_ac, omega_opt = phonons.diatomic_chain_dispersion(0.0, C=1.0, m1=1.0, m2=2.0)
    assert omega_ac == pytest.approx(0.0, abs=1e-9)
    assert omega_opt > 0.0


def test_diatomic_chain_optical_above_acoustic_everywhere():
    k = np.linspace(0.0, np.pi, 50)
    omega_ac, omega_opt = phonons.diatomic_chain_dispersion(k, C=1.0, m1=1.0, m2=2.0)
    assert np.all(omega_opt >= omega_ac)


def test_diatomic_chain_gap_at_zone_boundary():
    # At k*a = pi, the two branches split into sqrt(2C/m1) and sqrt(2C/m2),
    # producing a finite gap for unequal masses (the "optical gap").
    omega_ac, omega_opt = phonons.diatomic_chain_dispersion(np.pi, C=1.0, m1=1.0, m2=2.0)
    assert omega_opt - omega_ac > 0.1


def test_diatomic_chain_equal_masses_closes_gap():
    # With m1 == m2 the two-atom basis is artificial; branches must meet
    # at the zone boundary (monatomic-chain limit, no optical gap).
    omega_ac, omega_opt = phonons.diatomic_chain_dispersion(np.pi, C=1.0, m1=1.0, m2=1.0)
    assert omega_ac == pytest.approx(omega_opt, rel=1e-9)


def test_bose_einstein_large_at_low_energy_high_T():
    assert phonons.bose_einstein(0.001, 300.0) > phonons.bose_einstein(0.1, 300.0)


# ------------------------------------------------------- degeneracy ratio
def test_boltzmann_ratio_near_unity_far_from_band():
    ratio = carriers.boltzmann_validity_ratio([0.30], 1.18, 300.0)
    assert ratio[0] == pytest.approx(1.0, abs=0.01)


def test_boltzmann_ratio_at_band_edge_is_known_degeneracy_factor():
    # At EF = Ec exactly, n_Boltzmann/n_exact = 1/[(2/sqrt(pi)) F_{1/2}(0)]
    # ~= 1.307, a standard tabulated Fermi-Dirac degeneracy factor.
    ratio = carriers.boltzmann_validity_ratio([0.0], 1.18, 300.0)
    assert ratio[0] == pytest.approx(1.307, abs=0.01)


def test_boltzmann_ratio_grows_once_degenerate():
    ratio = carriers.boltzmann_validity_ratio([0.10, -0.10], 1.18, 300.0)
    assert ratio[1] > ratio[0] > 1.0
