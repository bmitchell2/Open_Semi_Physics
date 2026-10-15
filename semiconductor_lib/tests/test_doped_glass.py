"""Checks for semiconductor_lib.doped_glass."""
from semiconductor_lib.doped_glass import bpsg_network_composition


def test_bpsg_network_composition():
    pure = bpsg_network_composition(0.0, 0.0)
    assert pure["x_Si"] == 1.0 and pure["connectivity"] == 4.0
    r = bpsg_network_composition(4.0, 4.0)
    # 4 wt% B is about one fifth of the network formers; 4 wt% P about 7%
    assert 0.18 < r["x_B"] < 0.23 and 0.06 < r["x_P"] < 0.09
    assert 3.6 < r["connectivity"] < 3.8
    assert bpsg_network_composition(5.0, 5.0)["connectivity"] < r["connectivity"]
