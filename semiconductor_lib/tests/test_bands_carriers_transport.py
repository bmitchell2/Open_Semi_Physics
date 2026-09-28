"""
Physics validation for bands.py, carriers.py and transport.py. Each test
encodes a number verified while writing the Effective Mass, Fermi Level and
Drift/Diffusion notes (session 2026-09-28).
"""
import numpy as np
import pytest

from semiconductor_lib import bands, carriers, transport, dopants


# ------------------------------------------------------------------ bands
def test_hbar2_over_2m0():
    assert bands.HBAR2_2M0 == pytest.approx(0.0381, abs=2e-4)   # eV nm^2


def test_electron_masses():
    assert bands.conductivity_mass() == pytest.approx(0.26, abs=0.002)
    assert bands.dos_mass() == pytest.approx(1.08, abs=0.005)


def test_two_band_hole_masses_below_experimental_values():
    """Two-band estimates (0.55 DOS, 0.37 cond.) neglect warping; Ioffe/Hu list 0.81 and 0.39."""
    assert bands.hole_dos_mass_two_band() == pytest.approx(0.55, abs=0.01)
    assert bands.hole_conductivity_mass_two_band() == pytest.approx(0.37, abs=0.005)
    assert bands.hole_dos_mass_two_band() < 0.81


def test_electrons_lighter_for_transport_but_heavier_for_dos():
    """The 'electrons light, holes heavy' statement flips between the two averages."""
    assert bands.conductivity_mass() < 0.39
    assert bands.dos_mass() > 0.81


def test_curvature_recovers_mass():
    k = np.linspace(-0.01, 0.01, 401)
    assert bands.curvature_mass(bands.parabolic_band(k, 0.19), k) == pytest.approx(0.19, rel=1e-6)


def test_tight_binding_curvature():
    """E = -2t cos(ka): d2E/dk2 = 2 t a^2 at k=0, so m* = hbar^2/(2 t a^2)."""
    k = np.linspace(-0.05, 0.05, 1001)
    d2 = np.gradient(np.gradient(bands.tight_binding_chain(k, 1.0, 1.0), k), k)[500]
    assert d2 == pytest.approx(2.0, abs=1e-4)


def test_cyclotron_resonance_fields_at_24GHz():
    """Heavy/light holes: 0.42 T and 0.137 T (UCSD 152B: 4210 G and 1370 G)."""
    assert bands.cyclotron_field_T(24e9, 0.49) == pytest.approx(0.421, abs=0.002)
    assert bands.cyclotron_field_T(24e9, 0.16) == pytest.approx(0.137, abs=0.001)


# ------------------------------------------------------------------ carriers
def test_effective_dos_from_masses():
    assert carriers.effective_dos(1.08) == pytest.approx(2.8e19, rel=0.02)     # Hu
    assert carriers.effective_dos(1.18) == pytest.approx(3.2e19, rel=0.02)     # Ioffe


def test_numerical_integral_matches_boltzmann_closed_form():
    """Integrating D(E) f(E) reproduces Nc exp(-(Ec-EF)/kT) to better than 0.05%."""
    num = carriers.density_numerical(0.209, 1.08)
    closed = carriers.boltzmann_density(0.209, carriers.effective_dos(1.08))
    assert num == pytest.approx(closed, rel=5e-4)
    assert num == pytest.approx(8.68e15, rel=5e-3)


def test_fermi_dirac_half_at_EF_and_boltzmann_tail():
    assert carriers.fermi_dirac(0.0) == pytest.approx(0.5)
    assert carriers.fermi_dirac(0.209) == pytest.approx(3.1e-4, rel=0.03)


def test_hu_example_fermi_level_position():
    """Hu Example 1-3: Nc = 2.8e19, n = 1e17 -> Ec - EF = 0.146 eV."""
    assert carriers.ec_minus_ef(1e17, 2.8e19) == pytest.approx(0.146, abs=0.002)


def test_hu_example_incomplete_ionization():
    """Hu Example 1-6 style: EF fixed at Ec-146 meV, Ed = 45 meV -> 3.9% non-ionized."""
    assert 1 - carriers.ionized_donor_fraction_fixed_EF(0.146) == pytest.approx(0.039, abs=0.002)


def test_self_consistent_beats_fixed_EF_estimate():
    """Solving EF self-consistently gives fewer non-ionized donors than fixing EF from n = ND."""
    fixed = 1 - carriers.ionized_donor_fraction_fixed_EF(carriers.ec_minus_ef(1e17, dopants.Nc(300)))
    selfc = 1 - dopants.solve_donor(1e17, 300).fraction
    assert selfc < fixed


def test_fermi_potential_and_intrinsic_level():
    assert carriers.fermi_potential(1e16) == pytest.approx(0.357, abs=0.002)
    assert carriers.intrinsic_level_offset_eV() == pytest.approx(-0.0074, abs=3e-4)


# ------------------------------------------------------------------ transport
def test_einstein_relation_matches_ioffe_diffusion_coefficients():
    assert transport.einstein_diffusion(1400) == pytest.approx(36.2, abs=0.2)   # Ioffe <= 36
    assert transport.einstein_diffusion(450) == pytest.approx(11.6, abs=0.2)    # Ioffe <= 12


def test_intrinsic_resistivity_near_ioffe_value():
    rho = transport.resistivity(1e10, 1e10)
    assert rho == pytest.approx(3.4e5, rel=0.02)
    assert rho == pytest.approx(3.2e5, rel=0.08)          # Ioffe 3.2e5


def test_doped_resistivity():
    assert transport.resistivity(1e15, 0.0) == pytest.approx(4.46, abs=0.02)
    assert transport.resistivity(0.0, 1e15) == pytest.approx(13.9, abs=0.1)


def test_scattering_times():
    assert transport.scattering_time(1400, transport.M_COND_ELECTRON) * 1e12 == pytest.approx(0.21, abs=0.005)
    assert transport.scattering_time(450, transport.M_COND_HOLE) * 1e12 == pytest.approx(0.10, abs=0.005)


def test_mobility_ratio_exceeds_mass_ratio():
    """Electron/hole mobility ratio (3.1) is only partly the mass ratio (1.5); the rest is scattering time."""
    mu_ratio = transport.MU_N_LATTICE / transport.MU_P_LATTICE
    m_ratio = transport.M_COND_HOLE / transport.M_COND_ELECTRON
    tau_ratio = transport.scattering_time(1400, 0.26) / transport.scattering_time(450, 0.39)
    assert mu_ratio == pytest.approx(3.11, abs=0.01) and m_ratio == pytest.approx(1.5, abs=0.01)
    assert tau_ratio == pytest.approx(2.07, abs=0.05)


def test_diffusion_lengths():
    L = transport.diffusion_length_um(36, np.array([1e-9, 1e-6, 1e-5]))
    assert L == pytest.approx([1.9, 60.0, 190.0], rel=0.01)


def test_drift_velocity_low_field():
    assert transport.drift_velocity(1400, 1e3) == pytest.approx(1.4e6)


def test_equilibrium_current_zero_for_boltzmann_profile():
    """n = n0 exp(q psi/kT) with E = -dpsi/dx makes drift cancel diffusion (Einstein relation)."""
    from semiconductor_lib.constants import thermal_voltage
    Vt = thermal_voltage(300)
    x = np.linspace(0, 1e-5, 20001)
    psi = 0.05 * (x / x[-1])                      # 50 mV ramp
    n = 1e15 * np.exp(psi / Vt)
    E = -np.gradient(psi, x)
    J = transport.electron_current_density(n, 1400, E, transport.einstein_diffusion(1400), np.gradient(n, x))
    scale = 1.602e-19 * n * 1400 * np.abs(E)
    assert np.max(np.abs(J[5:-5]) / scale[5:-5]) < 1e-3
