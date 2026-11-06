"""
Alignment-mark optics and epitaxial pattern shift.

Backs the Semiconductor Notes page "Alignment Mark Design: Zero Layer,
Mark Depth, Segmentation and Epitaxial Pattern Shift".

First-order signal of a reflective binary (two-level) phase grating in
the scalar thin-grating approximation:
    eta_1 = (2 sin(pi D) / pi)^2 * sin^2(phi / 2),  phi = 4 pi n d / lambda
D is the line duty cycle, d the step depth, n the index of the medium
filling the step, lambda the vacuum wavelength. Maximum at the
quarter-wave depth d = lambda / (4 n) (158 nm for 633 nm in air); zero
at the half-wave depth. Ignores films over the mark, sidewall angle and
polarization; it is the bare-step limit used to choose a starting depth.

Epitaxial pattern shift: if the two edges of a feature move laterally by
r_left*t and r_right*t for epi thickness t (r = lateral shift per unit
thickness, positive in the +x direction), the mark centre moves by
t*(r_left + r_right)/2 and its width changes by t*(r_right - r_left).
"""
import numpy as np


def first_order_efficiency(depth_nm, wavelength_nm, n=1.0, duty=0.5):
    phi = 4.0 * np.pi * n * np.asarray(depth_nm, float) / wavelength_nm
    return (2.0 * np.sin(np.pi * duty) / np.pi) ** 2 * np.sin(phi / 2.0) ** 2


def quarter_wave_depth(wavelength_nm, n=1.0):
    return wavelength_nm / (4.0 * n)


def epi_edge_motion(t_epi, r_left, r_right):
    """Return (centre shift, width change) for epi thickness t_epi."""
    return t_epi * 0.5 * (r_left + r_right), t_epi * (r_right - r_left)
