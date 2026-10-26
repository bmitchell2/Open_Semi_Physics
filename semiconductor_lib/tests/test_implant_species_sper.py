"""Physics checks for implant_species and sper."""
import numpy as np
from semiconductor_lib import implant_species as isp, sper

B, P, AS = (5, 10.81), (15, 30.97), (33, 74.92)


def test_crossover_energy_rises_with_ion_mass():
    eb, ep, ea = (isp.crossover_energy_keV(*i) for i in (B, P, AS))
    assert eb < ep < ea
    assert 5 < eb < 30          # textbook boron crossover is ~10-20 keV
    assert 300 < ea < 3000      # arsenic: hundreds of keV to ~1 MeV


def test_nuclear_stopping_has_a_maximum_and_electronic_rises():
    E = np.logspace(-1, 4, 400)
    sn = isp.zbl_nuclear_stopping(*AS, E)
    k = np.argmax(sn)
    assert 0 < k < len(E) - 1
    se = isp.lss_electronic_stopping(*AS, E)
    assert np.all(np.diff(se) > 0)


def test_molecular_energy_partition_and_current_gain():
    assert abs(isp.molecular_atom_energy(5.0, 10.81, 48.81) - 1.107) < 0.01
    assert abs(isp.relative_atom_current(1, 10.81, 48.81) - 48.81 / 10.81) < 1e-9
    assert abs(isp.relative_atom_current(2, 74.92, 149.84) - 4.0) < 1e-9


def test_sper_magnitudes_and_orientation():
    # ~0.1-1 nm/s near 550 C; microseconds for 20 nm near 1300 C
    v550 = sper.sper_velocity_cm_s(550) * 1e7
    assert 0.05 < v550 < 1.0
    assert sper.regrowth_time_s(20, 1300) < 1e-4
    assert sper.regrowth_time_s(20, 600, "111") > 10 * sper.regrowth_time_s(20, 600, "100")
