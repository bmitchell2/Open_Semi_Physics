"""
Diamond-cubic silicon geometry, verified by enumerating atom positions
(not by formula alone). Backs the "Silicon Crystal Structure" and "Crystal
Planes, Miller Indices, and Wafer Orientation" notes.

Units: lengths in Angstrom unless the name says cm; densities in cm^-3
(volume) or cm^-2 (planar).
"""
import numpy as np

A_SI = 5.431            # lattice constant at 300 K, Angstrom (Ioffe; Hu gives 5.43)
M_SI = 28.0855          # atomic weight, g/mol
N_AVOGADRO = 6.02214076e23

_FCC = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
_BASIS = np.array([[0, 0, 0], [.25, .25, .25]])


def bond_length(a=A_SI):
    """Nearest-neighbour distance, sqrt(3)/4 * a (Angstrom)."""
    return np.sqrt(3) * a / 4


def atomic_density(a=A_SI):
    """Atoms per cm^3: 8 atoms per cubic cell."""
    return 8 / (a * 1e-8) ** 3


def mass_density(a=A_SI, M=M_SI):
    """g/cm^3."""
    return 8 * M / (N_AVOGADRO * (a * 1e-8) ** 3)


def packing_fraction():
    """Touching-sphere packing fraction, pi*sqrt(3)/16 = 0.340."""
    return np.pi * np.sqrt(3) / 16


def atoms(n_cells=3, a=A_SI):
    """Atom coordinates (Angstrom) in the block of cubic cells [-n, n]^3."""
    r = np.arange(-n_cells, n_cells + 1)
    g = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    return np.concatenate([(g + f + b) * a for f in _FCC for b in _BASIS])


def neighbor_shells(origin_frac, n_shells=3, a=A_SI, n_cells=3):
    """
    Distance shells around a point given in fractional cell coordinates.
    Returns [(distance_A, count), ...] for the first n_shells shells.
    A-site (0,0,0) and B-site (1/4,1/4,1/4) both give 4 @ 2.352, 12 @ 3.840,
    12 @ 4.503; the body centre (1/2,1/2,1/2) is a vacant tetrahedral site
    with 4 @ 2.352 and 6 @ 2.716.
    """
    pts = atoms(n_cells, a)
    d = np.linalg.norm(pts - np.asarray(origin_frac, float) * a, axis=1)
    d = np.round(d[d > 1e-6], 3)
    u, c = np.unique(d, return_counts=True)
    return list(zip(u[:n_shells].tolist(), c[:n_shells].tolist()))


def is_atom_site(frac, a=A_SI, n_cells=3):
    """True if an atom sits at the fractional position."""
    pts = atoms(n_cells, a)
    return bool(np.any(np.linalg.norm(pts - np.asarray(frac, float) * a, axis=1) < 1e-6))


def planar_density_analytic(hkl, a=A_SI):
    """Atoms/cm^2 in ONE atomic plane of (100), (110) or (111)."""
    a_cm = a * 1e-8
    table = {(1, 0, 0): 2 / a_cm ** 2,
             (1, 1, 0): 2 * np.sqrt(2) / a_cm ** 2,
             (1, 1, 1): 4 / (np.sqrt(3) * a_cm ** 2)}
    return table[tuple(hkl)]


def planar_density_counted(hkl, a=A_SI, n_cells=9, radius=38.0, thickness=0.02):
    """
    Slab-count atoms per unit area on each atomic plane along the normal
    (planes with offsets 0 to 6.5 Angstrom). Returns [(offset_A, cm^-2), ...].
    For (111) the gaps between successive planes alternate 0.784 A (within a
    bilayer) and 2.352 A (between bilayers). Finite-disc counting has about
    +/-2% statistical noise; use planar_density_analytic for exact values.
    """
    n = np.array(hkl, float)
    n /= np.linalg.norm(n)
    pts = atoms(n_cells, a)
    proj = pts @ n
    out = []
    for off in np.unique(np.round(proj, 3)):
        if -0.01 < off < 6.5:
            p = pts[np.abs(proj - off) < thickness]
            inplane = p - np.outer(p @ n, n)
            cnt = np.sum(np.linalg.norm(inplane, axis=1) < radius)
            out.append((float(off), cnt / (np.pi * radius ** 2) * 1e16))
    return out
