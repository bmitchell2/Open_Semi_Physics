"""
Shared matplotlib style preset and SVG minification pipeline.

Keeps generated figures compact enough for Notion's inline attachment
path (notion-create-attachment `content` field, ~200KB limit) without a
separate manual minification pass each session. See Processing
Instructions, Section 31, for the full policy this implements.
"""
import re
import subprocess
import matplotlib


def apply_style():
    """Apply the standard rcParams used across Semiconductor Notes figures.
    Call this once at the top of any plotting script, before importing
    pyplot."""
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['path.simplify'] = True
    matplotlib.rcParams['path.simplify_threshold'] = 1.0


# Standard color palette used across figures, for visual consistency.
COLORS = {
    "blue": "#1f6feb",
    "red": "#d1242f",
    "purple": "#8250df",
    "green": "#1a7f37",
    "orange": "#9a6700",
    "gray": "#57606a",
}


def minify_svg(path_in, path_out):
    """Strip metadata/DOCTYPE, round numeric precision, and collapse
    whitespace in an SVG file. Typically used after `scour` (see
    minify_with_scour) for a final pass; can also be used standalone.
    Returns (size_before, size_after) in bytes."""
    import os
    with open(path_in, 'r') as f:
        s = f.read()
    s = re.sub(r'<metadata>.*?</metadata>', '', s, flags=re.DOTALL)
    s = re.sub(r'<!DOCTYPE[^>]*>\s*', '', s)
    s = re.sub(r'-?\d+\.\d{3,}', lambda m: f"{float(m.group(0)):.2f}", s)
    s = re.sub(r'>\s+<', '><', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n+', '', s)
    with open(path_out, 'w') as f:
        f.write(s)
    return os.path.getsize(path_in), os.path.getsize(path_out)


def minify_with_scour(path_in, path_out):
    """Run `scour` (pip install scour) for structural SVG minification
    (id stripping, precision reduction, metadata removal). Run
    minify_svg() on the result for a final whitespace pass."""
    subprocess.run([
        "python3", "-m", "scour.scour",
        "-i", path_in, "-o", path_out,
        "--enable-id-stripping", "--shorten-ids",
        "--remove-metadata", "--strip-xml-prolog",
        "--set-precision=2", "--create-groups",
    ], check=True, capture_output=True)


def full_minify_pipeline(path_in, path_out):
    """Run scour, then the whitespace/precision pass, in one call.
    Typical result: a matplotlib SVG with the style preset applied
    lands in the 8-50KB range after this pipeline."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        minify_with_scour(path_in, tmp_path)
        return minify_svg(tmp_path, path_out)
    finally:
        os.unlink(tmp_path)
