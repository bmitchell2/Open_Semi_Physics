"""
System-level RF relations used alongside rf.py.

self_resonant_frequency     : f_SRF = 1/(2 pi sqrt(L C_p)) for an inductor
                              with lumped parallel parasitic capacitance.
apparent_inductance         : L / (1 - (f/f_SRF)^2) seen below self-resonance.
db_power, db_amplitude      : 10 log10 and 20 log10 conventions.
friis_noise_figure_db       : cascade noise figure (Friis, Proc. IRE 1944).
range_correlation_factor_db : suppression of correlated LO phase noise in a
                              coherent radar beat signal (Budge & Burt 1993).

Verified in tests/test_wafer_rf_cascade.py.
"""
import numpy as np


def self_resonant_frequency(L, C_p):
    """Self-resonant frequency (Hz) of an inductor L (H) with lumped
    parallel parasitic capacitance C_p (F): f = 1 / (2 pi sqrt(L C_p))."""
    return 1.0 / (2.0 * np.pi * np.sqrt(L * C_p))


def apparent_inductance(L, f, f_srf):
    """Inductance seen at the terminals of a lossless L || C_p below
    self-resonance: L / (1 - (f/f_srf)^2). Diverges at f_srf; the part is
    capacitive above it (returned value negative)."""
    x = np.asarray(f, dtype=float) / f_srf
    return L / (1.0 - x ** 2)


def db_power(ratio):
    """Power ratio to decibels: 10 log10(P2/P1)."""
    return 10.0 * np.log10(ratio)


def db_amplitude(ratio):
    """Voltage or current ratio to decibels: 20 log10(V2/V1), valid when
    both are measured across the same impedance."""
    return 20.0 * np.log10(ratio)


def friis_noise_figure_db(stages):
    """Cascade noise figure (dB) by the Friis formula.
    stages = [(gain_dB, NF_dB), ...] from the antenna inward. A matched
    passive loss of L dB at the reference temperature is (-L, L).
    F_tot = F1 + (F2 - 1)/G1 + (F3 - 1)/(G1 G2) + ..."""
    F_tot, G_run = 0.0, 1.0
    for i, (g_db, nf_db) in enumerate(stages):
        F = 10 ** (nf_db / 10.0)
        F_tot += F if i == 0 else (F - 1.0) / G_run
        G_run *= 10 ** (g_db / 10.0)
    return 10.0 * np.log10(F_tot)


def range_correlation_factor_db(df, R, c=2.998e8):
    """Suppression (dB) of LO phase noise at offset df (Hz) in the beat
    signal of a coherent (FMCW/CW) radar for a reflector at range R (m).
    The transmit and LO phase noise are the same process delayed by
    tau = 2R/c, so the mixer output carries phi(t) - phi(t - tau), whose
    PSD is the LO PSD times 4 sin^2(pi df tau) (Budge & Burt, 1993).
    Small delays: ~ 20 log10(2 pi df tau), strong suppression for close
    targets; the factor reaches +6 dB at df*tau = 1/2 (no benefit)."""
    tau = 2.0 * np.asarray(R, dtype=float) / c
    return 10.0 * np.log10(4.0 * np.sin(np.pi * np.asarray(df, dtype=float) * tau) ** 2)
