"""Smoke and size tests for the 2026-09-29 diode / MOS figure generators."""
import os
import xml.dom.minidom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from semiconductor_lib import figures_devices as F

EXPECTED_AXES = {
    "diode_forward_iv_ideality": 2,
    "diode_reverse_leakage_arrhenius": 1,          # secondary °C axis is a child axes
    "diode_stored_charge_and_diffusion_capacitance": 2,
    "diode_reverse_recovery_transient": 1,
    "moscap_band_diagram_and_depletion_width": 3,  # band diagram, W, twin Qinv
}


@pytest.mark.parametrize("name", list(EXPECTED_AXES))
def test_figure_builds_with_expected_panels(name):
    fig = F.FIGURES[name]()
    assert len(fig.axes) == EXPECTED_AXES[name]
    plt.close(fig)


def test_build_all_produces_small_valid_svgs(tmp_path):
    pytest.importorskip("scour")
    out = F.build_all(str(tmp_path))
    assert set(out) == set(EXPECTED_AXES)
    for name, path in out.items():
        assert os.path.getsize(path) < 20_000, name
        xml.dom.minidom.parse(path)
