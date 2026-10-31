"""
Geometry dependence of MOS capacitor C-V: area/perimeter decomposition,
distributed channel (inversion-layer) RC, and series-resistance distortion of
parallel-mode capacitance readings.

Backs the Semiconductor Notes page
"Geometry Dependence of MOS C-V (Area-Perimeter Scaling)".

1. Area/perimeter decomposition (empirical superposition, per bias point)
       C(V) = C_a(V) * A + C_p(V) * P
   Dividing by A gives C/A = C_a + C_p * (P/A): a straight line in P/A whose
   intercept is the per-area term and whose slope is the per-length term.
   `area_perimeter_extract` does the least-squares fit at every bias point.

2. Inversion-layer channel as a distributed RC line (capacitor with
   source/drain contacts). Per unit length along the channel:
       r = 1 / (W mu Cox Vov)   (ohm/m),   c = Cox W   (F/m)
   An open-ended line of length l has Y = sqrt(j w c / r) tanh(gamma l),
   gamma = sqrt(j w r c). Fed from both ends (source and drain) the gate sees
   two such lines of length L/2 in parallel. Low-frequency limit: total
   capacitance Cox W L in series with R_ch/12 (both ends) or R_ch/3 (one end),
   giving
       tau = L^2 / (k mu Vov),   k = 12 (both ends) or 3 (one end).
   Cox cancels: the corner depends only on length, mobility and overdrive.

3. Series resistance seen by a parallel-mode (Cp-G) LCR measurement:
       Y = 1 / (Rs + 1/(G + j w C)),   C_measured = Im(Y)/w
   which for G = 0 reduces to C / (1 + (w Rs C)^2).

Units: SI throughout (m, F, ohm, s, m^2/V/s).
"""
import numpy as np


def area_perimeter_extract(A, P, C):
    """Least-squares split of measured capacitance into area and perimeter terms.

    A, P : 1-D arrays of the n structures' areas (m^2) and perimeters (m).
    C    : array of shape (n,) or (n, m): capacitance of each structure at m bias
           points (F).
    Returns (C_a, C_p): per-area (F/m^2) and per-length (F/m) components, each
    of shape (m,) (or scalars for 1-D input). Needs at least two structures
    with different P/A.
    """
    A = np.asarray(A, dtype=float)
    P = np.asarray(P, dtype=float)
    C = np.asarray(C, dtype=float)
    if np.ptp(P / A) == 0:
        raise ValueError("structures must span a range of P/A")
    M = np.column_stack([A, P])
    sol, *_ = np.linalg.lstsq(M, C, rcond=None)
    return sol[0], sol[1]


def channel_rc_admittance(f, L, W, Cox, mu, Vov, feed="both"):
    """Small-signal gate admittance (S) of an inverted channel treated as a
    uniform distributed RC line, with the source/drain as AC ground.

    f : frequency (Hz, scalar or array). L, W : channel length and width (m).
    Cox : F/m^2. mu : channel mobility (m^2/V/s). Vov = V_G - V_T > 0 (V).
    feed : 'both' (source and drain) or 'one' (single-sided contact).
    """
    f = np.asarray(f, dtype=float)
    w = 2 * np.pi * f
    r = 1.0 / (W * mu * Cox * Vov)
    c = Cox * W

    def open_line(length):
        gamma = np.sqrt(1j * w * r * c)
        z0_inv = np.sqrt(1j * w * c / r)
        return z0_inv * np.tanh(gamma * length)

    if feed == "both":
        return 2.0 * open_line(L / 2.0)
    if feed == "one":
        return open_line(L)
    raise ValueError("feed must be 'both' or 'one'")


def channel_rc_capacitance(f, L, W, Cox, mu, Vov, feed="both"):
    """Apparent parallel-mode capacitance Im(Y)/(2 pi f) of the inverted channel (F)."""
    Y = channel_rc_admittance(f, L, W, Cox, mu, Vov, feed)
    return np.imag(Y) / (2 * np.pi * np.asarray(f, dtype=float))


def channel_rc_tau(L, mu, Vov, feed="both"):
    """Low-frequency series-RC time constant of the inverted channel (s)."""
    k = {"both": 12.0, "one": 3.0}[feed]
    return L ** 2 / (k * mu * Vov)


def parallel_mode_capacitance(C, Rs, f, G=0.0):
    """Capacitance reported by a parallel-mode (Cp-G) meter when the true
    capacitance C (F) with parallel conductance G (S) sits behind a series
    resistance Rs (ohm), at frequency f (Hz)."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    Y = 1.0 / (Rs + 1.0 / (G + 1j * w * C))
    return np.imag(Y) / w
