"""Smoke and size tests for the notes figure generators."""
import os
import xml.dom.minidom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from semiconductor_lib import figures

EXPECTED_AXES = {
    "silicon_diamond_cubic_unit_cell": 1,
    "fermi_level_carrier_distribution_n_type_si": 4,
    "dopant_ionization_vs_temperature": 2,
    "silicon_effective_mass_curvature": 2,
    "pn_junction_equilibrium_numerical_poisson": 3,
    "pn_junction_debye_length_vs_depletion_width": 2,
    "n_type_intrinsic_step_junction_numerical_poisson": 2,
    "deal_grove_oxide_growth_1000C": 1,
}


@pytest.mark.parametrize("name", list(EXPECTED_AXES))
def test_figure_builds_with_expected_panels(name):
    fig = figures.FIGURES[name]()
    assert len(fig.axes) == EXPECTED_AXES[name]
    plt.close(fig)


def test_build_all_produces_small_valid_svgs(tmp_path):
    """Notion inline attachments are pasted as text: keep every figure under 30 KB.
    (matplotlib mathtext would roughly double the size; see figures.py docstring.)"""
    pytest.importorskip("scour")
    out = figures.build_all(str(tmp_path))
    assert set(out) == set(EXPECTED_AXES)
    for name, path in out.items():
        assert os.path.getsize(path) < 30_000, name
        xml.dom.minidom.parse(path)          # well-formed XML


def test_no_mathtext_glyph_outlines(tmp_path):
    """Regression: '$...$' labels or 10^n log ticks turn into glyph paths and bloat the SVG."""
    fig = figures.fig_deal_grove()
    raw = figures.save_figure(fig, str(tmp_path / "dg"), minify=False)
    s = open(raw).read()
    assert "10^" not in s and "mathtext" not in s
    assert "<text" in s                       # labels stay as text, not outlines
