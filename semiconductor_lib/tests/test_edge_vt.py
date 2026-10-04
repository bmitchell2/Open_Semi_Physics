import numpy as np
from semiconductor_lib.edge_vt import (edge_vt_shift_subthreshold,
                                       edge_vt_shift_strong_inversion)
from semiconductor_lib.constants import k_B, q


def test_zero_edge_change_gives_zero():
    assert abs(edge_vt_shift_subthreshold(20.0, 0.1, 0.0)) < 1e-12


def test_long_channel_limit_vanishes():
    assert abs(edge_vt_shift_subthreshold(1e6, 0.1, 0.1)) < 1e-4
    assert abs(edge_vt_shift_strong_inversion(1e6, 0.1, 0.1)) < 1e-6


def test_monotonic_in_length_and_sign():
    L = np.logspace(0, 2, 20)
    up = edge_vt_shift_subthreshold(L, 0.1, 0.1)
    dn = edge_vt_shift_subthreshold(L, 0.1, -0.1)
    assert np.all(np.diff(up) < 0) and np.all(up > 0)
    assert np.all(np.diff(dn) > 0) and np.all(dn < 0)


def test_lowered_edge_bounded_by_removed_length():
    nphit = 1.3 * k_B * 300 / q
    bound = nphit * np.log(1 - 0.2 / 2.0)
    v = edge_vt_shift_subthreshold(2.0, 0.1, -1.0)
    assert bound - 1e-6 <= v < 0
    assert abs(v - bound) < 1e-3


def test_raised_edge_exceeds_linear_average_in_subthreshold():
    assert edge_vt_shift_subthreshold(20.0, 0.1, 0.1) > \
        edge_vt_shift_strong_inversion(20.0, 0.1, 0.1)
