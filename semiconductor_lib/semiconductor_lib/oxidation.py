"""
Deal-Grove thermal oxidation: x^2 + A x = B (t + tau).

Constants are Deal & Grove (1965) at 1000 C, 1 atm, for (111) silicon;
(100) is obtained from (B/A)_111 = 1.68 (B/A)_100 with B unchanged, i.e.
A_100 = 1.68 A_111. Valid for oxide thicker than about 30 nm; below that
the thin-oxide growth-rate enhancement (Massoud, Plummer, Irene 1985)
must be added and gate oxides should be calibrated on the tool.
Units: A in um, B in um^2/h, time in hours, thickness in um.
"""
import numpy as np

from .lattice import atomic_density

DEAL_GROVE_1000C = {
    "dry": {"A": 0.165, "B": 0.0117},   # dry O2, (111)
    "wet": {"A": 0.226, "B": 0.287},    # H2O from a 95 C bubbler, (111)
}
BA_RATIO_111_OVER_100 = 1.68
N_OX = 2.25e22   # SiO2 units per cm^3 (Deal & Grove oxidant density for O2)


def rate_constants(ambient="dry", orientation="111", table=DEAL_GROVE_1000C):
    """Return (A, B) for the ambient and orientation ('111' or '100')."""
    A, B = table[ambient]["A"], table[ambient]["B"]
    if str(orientation) == "100":
        A *= BA_RATIO_111_OVER_100
    elif str(orientation) != "111":
        raise ValueError("orientation must be '111' or '100'")
    return A, B


def tau_from_initial_oxide(x_i, A, B):
    """Time shift for an initial oxide x_i (um): (x_i^2 + A x_i)/B, hours."""
    return (x_i ** 2 + A * x_i) / B


def thickness(t, A, B, tau=0.0):
    """Oxide thickness (um) after t hours."""
    return (A / 2) * (np.sqrt(1 + (np.asarray(t) + tau) / (A ** 2 / (4 * B))) - 1)


def time_to_grow(x, A, B, tau=0.0):
    """Hours to reach thickness x (um)."""
    return (np.asarray(x) ** 2 + A * np.asarray(x)) / B - tau


def linear_rate(A, B):
    """B/A, um/h (thin-oxide, surface-reaction-limited regime)."""
    return B / A


def silicon_consumed(x_ox, N_ox=N_OX, N_si=None):
    """Silicon thickness consumed by an oxide of thickness x_ox (about 0.45 x_ox)."""
    N_si = atomic_density() if N_si is None else N_si
    return np.asarray(x_ox) * N_ox / N_si
