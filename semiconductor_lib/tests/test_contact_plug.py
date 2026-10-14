"""Physics checks for the lined contact-plug resistance model."""
import numpy as np
from semiconductor_lib import deposition as dp

RW, RTIN = 5.3e-8, 150e-8   # ohm m


def test_no_liner_reduces_to_rho_L_over_A():
    d, L = 50e-9, 100e-9
    R = dp.lined_plug_resistance(d, L, 0.0, RW, RTIN, bottom_liner=False)
    assert np.isclose(R, RW * L / (np.pi * (d / 2) ** 2))


def test_liner_penalty_grows_as_diameter_shrinks():
    d = np.array([80e-9, 40e-9, 20e-9])
    lined = dp.lined_plug_resistance(d, 50e-9, 3e-9, RW, RTIN)
    bare = dp.lined_plug_resistance(d, 50e-9, 0.0, RW, RTIN, bottom_liner=False)
    ratio = lined / bare
    assert np.all(np.diff(ratio) > 0) and ratio[0] > 1


def test_liner_filling_hole_conducts_through_liner_only():
    d, L, t = 6e-9, 50e-9, 3e-9
    R = dp.lined_plug_resistance(d, L, t, RW, RTIN)
    A = np.pi * (d / 2) ** 2
    assert np.isfinite(R) and np.isclose(R, RTIN * (L + t) / A)


def test_size_effect_limits():
    assert np.isclose(dp.wire_size_effect(RW, 19.1e-9, 1.0), RW, rtol=1e-6)
    assert np.isclose(dp.wire_size_effect(RW, 19.1e-9, 1e-6, p=1.0), RW)
    assert dp.wire_size_effect(RW, 19.1e-9, 20e-9) > 1.5 * RW
