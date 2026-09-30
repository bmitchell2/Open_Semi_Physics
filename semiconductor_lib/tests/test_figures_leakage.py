"""Smoke and size tests for the GIDL / FinFET figure generators."""
import os
import xml.dom.minidom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from semiconductor_lib import figures_leakage as F

EXPECTED_AXES = {"pocket_junction_field_and_btbt": 2,
                 "multigate_natural_length_min_gate_length": 1}


@pytest.mark.parametrize("name", list(EXPECTED_AXES))
def test_figure_builds(name):
    fig = F.FIGURES[name]()
    assert len(fig.axes) == EXPECTED_AXES[name]
    plt.close(fig)


def test_build_all_small_valid_svgs(tmp_path):
    pytest.importorskip("scour")
    out = F.build_all(str(tmp_path))
    for path in out.values():
        assert os.path.getsize(path) < 60_000
        xml.dom.minidom.parse(path)
