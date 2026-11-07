import numpy as np
from semiconductor_lib.electrostatics import (
    surface_carrier_densities_p, fermi_potential_p, oxide_field_and_drop,
    semiconductor_charge_p, threshold_voltage)
from semiconductor_lib.constants import ni300, eps_ox


def test_regime_boundaries():
    NA = 1e17
    phiF = fermi_potential_p(NA)
    p0, n0 = surface_carrier_densities_p(0.0, NA)
    assert np.isclose(p0, NA) and np.isclose(n0, ni300**2 / NA)
    p1, n1 = surface_carrier_densities_p(phiF, NA)
    assert np.isclose(p1, ni300, rtol=1e-9) and np.isclose(n1, ni300, rtol=1e-9)
    p2, n2 = surface_carrier_densities_p(2 * phiF, NA)
    assert np.isclose(n2, NA, rtol=1e-9) and np.isclose(p2, ni300**2 / NA, rtol=1e-9)
    # mass action holds everywhere
    psi = np.linspace(-0.2, 1.0, 50)
    p, n = surface_carrier_densities_p(psi, NA)
    assert np.allclose(p * n, ni300**2, rtol=1e-9)


def test_oxide_field_independent_of_thickness():
    NA = 1e17
    Q1, E1, V1 = oxide_field_and_drop(NA, 2e-7)
    Q2, E2, V2 = oxide_field_and_drop(NA, 300e-7)
    assert np.isclose(E1, E2) and np.isclose(Q1, Q2)
    assert np.isclose(V2 / V1, 150.0)
    # consistent with threshold formula (ideal VFB = 0)
    phiF = fermi_potential_p(NA)
    assert np.isclose(threshold_voltage(NA, eps_ox / 2e-7), 2 * phiF + V1)
    # magnitudes: ~0.1 V at 2 nm, ~14-15 V at 300 nm, field ~5e5 V/cm
    assert 0.08 < V1 < 0.13 and 13 < V2 < 16 and 4e5 < E1 < 6e5


def test_depletion_charge_matches_full_solution_at_threshold():
    NA = 1e17
    phiF = fermi_potential_p(NA)
    Q, _, _ = oxide_field_and_drop(NA, 2e-7)
    Qs_full = abs(semiconductor_charge_p(2 * phiF, NA))
    # full solution includes inversion electrons and a kT/q correction: within ~5%
    assert abs(Qs_full - Q) / Q < 0.05
