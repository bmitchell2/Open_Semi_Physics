"""The SVG minification pipeline must not move plotted data.

Regression test: scour with --set-precision=2 rounded relative path
coordinates to two significant figures, so long multi-point lines drifted
visibly away from their own markers (found on the RSCE V_T-versus-1/L figure).
"""
import re
import pytest
import numpy as np

pytest.importorskip("scour")
from semiconductor_lib.plotting import apply_style, full_minify_pipeline


def _abs_points(d):
    """Absolute vertices of an SVG path made of M/L/H/V/Z commands."""
    toks = re.findall(r'[MLHVZmlhvz]|-?\d*\.?\d+(?:e-?\d+)?', d)
    pts, cmd, x, y, i = [], None, 0.0, 0.0, 0
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            continue
        if cmd in 'MLml':
            dx, dy = float(toks[i]), float(toks[i + 1]); i += 2
            x, y = (x + dx, y + dy) if cmd.islower() else (dx, dy)
            if cmd == 'm': cmd = 'l'
            if cmd == 'M': cmd = 'L'
        elif cmd in 'Hh':
            v = float(toks[i]); i += 1
            x = x + v if cmd == 'h' else v
        elif cmd in 'Vv':
            v = float(toks[i]); i += 1
            y = y + v if cmd == 'v' else v
        pts.append((x, y))
    return np.array(pts)


def _longest_path(svg):
    ds = re.findall(r'\sd="([^"]+)"', svg)
    ds = [d for d in ds if not re.search(r'[CcQqSsAa]', d)]
    return max((_abs_points(d) for d in ds), key=len)


def test_minify_preserves_line_vertices(tmp_path):
    apply_style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(0, 33, 40)
    y = 0.26 + 0.008 * x + 0.004 * np.sin(x)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, y, '-')
    raw, out = tmp_path / "raw.svg", tmp_path / "min.svg"
    with matplotlib.rc_context({"path.simplify": False}):
        fig.savefig(raw)
    plt.close(fig)
    full_minify_pipeline(str(raw), str(out))
    a, b = _longest_path(raw.read_text()), _longest_path(out.read_text())
    assert len(a) == len(b) >= 40
    # every vertex within 0.05 pt of the original
    assert np.max(np.abs(a - b)) < 0.05
