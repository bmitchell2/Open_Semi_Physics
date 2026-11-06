"""
Implant beam geometry: axial channeling critical angles, and shadowing
of tilted implants by gate stacks and resist walls, including the
residual dose deficit of quad-mode (four-twist) implants.

Backs the Semiconductor Notes page "Ion Channeling and Implant
Tilt/Twist (Critical Angle, Shadowing and Quad-Mode Implants)".

Channeling (Lindhard 1965; Gemmell, Rev. Mod. Phys. 46, 129 (1974)):
    psi_1 = sqrt(2 Z1 Z2 e^2 / (E d))                 high-energy limit
    psi_2 = sqrt(C a / (sqrt(2) d) * psi_1),  C^2~3    used when psi_1 > a/d
e^2 = 14.3996 eV*Angstrom, E ion energy (eV), d atom spacing along the
axis (Angstrom), a the screening length (Angstrom). Estimates are good to
a factor of about 1.5. Verified: B and P along <100> from 10 keV up have
critical angles below 7 degrees (B: 4.5 deg at 10 keV, 2.5 deg at
100 keV); at 1 keV boron the estimate (8.0 deg) exceeds 7 degrees, which
is why sub-keV implants rely on amorphization rather than tilt.

Shadowing: a wall of height h shadows a strip h*tan(theta)*cos(phi) wide
on the side facing away from the beam, where theta is the tilt from the
wafer normal and phi the angle between the in-plane beam direction and
the wall normal. Walls are treated as opaque and infinitely long; the
ion's lateral travel inside silicon is ignored (surface arrival only).
Verified: quad mode leaves 3/4 dose along a wall and 1/2 in an inside
corner; 150 nm wall at 7 deg shadows 18.4 nm.
"""
import numpy as np

E2 = 14.3996          # e^2/(4 pi eps0), eV*Angstrom
A0 = 0.529177         # Bohr radius, Angstrom
A_SI = 5.431          # silicon lattice constant, Angstrom

# atom spacing along the major axes of diamond-cubic Si (Angstrom)
AXIS_SPACING_SI = {"100": A_SI, "110": A_SI / np.sqrt(2), "111": A_SI * np.sqrt(3) / 4}


def screening_length(Z1, Z2):
    """ZBL universal screening length (Angstrom)."""
    return 0.8853 * A0 / (Z1 ** 0.23 + Z2 ** 0.23)


def critical_angle_deg(Z1, E_eV, axis="100", Z2=14, C=np.sqrt(3.0)):
    """Lindhard critical angle (degrees) for axial channeling in Si.
    Returns psi_2 when psi_1 exceeds a/d (the low-energy regime that
    covers essentially all dopant implants), otherwise psi_1."""
    d = AXIS_SPACING_SI[axis]
    a = screening_length(Z1, Z2)
    psi1 = np.sqrt(2.0 * Z1 * Z2 * E2 / (np.asarray(E_eV, float) * d))
    psi2 = np.sqrt(C * a / (np.sqrt(2.0) * d) * psi1)
    psi = np.where(psi1 > a / d, psi2, psi1)
    return np.degrees(psi)


def shadow_length(h, tilt_deg, phi_deg=0.0):
    """Width of the shadow cast by a wall of height h (same units),
    for tilt theta and in-plane angle phi between beam and wall normal.
    Zero when the beam's in-plane component points away from the
    shadowed side (cos(phi) <= 0)."""
    c = np.cos(np.radians(phi_deg))
    return h * np.tan(np.radians(tilt_deg)) * np.maximum(c, 0.0)


def surface_dose_in_gap(x, gap, h, tilt_deg, twists_deg=(0.0, 90.0, 180.0, 270.0)):
    """Fraction of the nominal dose arriving at the silicon surface at
    positions x (0..gap) between two parallel walls of height h at x=0
    and x=gap. Each entry of twists_deg is one pass carrying an equal
    share of the dose; twist is the in-plane beam direction measured
    from +x (the normal of the left wall, pointing into the gap)."""
    x = np.asarray(x, float)
    dose = np.zeros_like(x)
    t = np.tan(np.radians(tilt_deg))
    for tw in twists_deg:
        cx = np.cos(np.radians(tw))
        lit = np.ones_like(x, dtype=bool)
        if cx > 1e-12:             # moving +x: left wall shadows its right side
            lit &= x >= h * t * cx
        elif cx < -1e-12:          # moving -x: right wall shadows its left side
            lit &= x <= gap - h * t * (-cx)
        dose += lit
    return dose / len(twists_deg)


def inside_corner_dose(h, tilt_deg, twists_deg=(0.0, 90.0, 180.0, 270.0), eps=1e-6):
    """Dose fraction at the foot of an inside (concave) corner formed by
    a wall normal to +x at x=0 and a wall normal to +y at y=0."""
    t = np.tan(np.radians(tilt_deg))
    lit = 0
    for tw in twists_deg:
        cx, cy = np.cos(np.radians(tw)), np.sin(np.radians(tw))
        blocked = (cx > 1e-12 and eps < h * t * cx) or (cy > 1e-12 and eps < h * t * cy)
        lit += 0 if blocked else 1
    return lit / len(twists_deg)


def min_open_space(h, tilt_deg):
    """Smallest opening (same units as h) in which a pass aimed at the
    far wall still reaches the foot of that wall: s = h*tan(theta).
    Below this the pass that should place a pocket under a gate edge
    is blocked by the neighbouring gate or resist."""
    return h * np.tan(np.radians(tilt_deg))
