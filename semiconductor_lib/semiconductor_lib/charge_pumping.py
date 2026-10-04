"""
Charge pumping (CP) interface-trap characterization.

Implements the two-level charge-pumping model of Groeseneken et al.
(IEEE TED 31(1), 42, 1984): SRH emission levels set by the pulse edges,
the swept energy window, the charge-pumping current, inversion to D_it,
and an analytical base-level ("Elliot") curve for illustration.

Units: cm, s, V, eV, A, cm^-2 eV^-1.  Pulse convention (n-channel):
the base level V_base sits toward accumulation, the top level
V_base + dV_A toward inversion.

Sign convention for the emission levels (energies relative to E_i):
    E_em,e - E_i = -kT ln(sigma_n v_th n_i t_em,e)   (> 0, upper half)
    E_em,h - E_i = +kT ln(sigma_p v_th n_i t_em,h)   (< 0, lower half)
    t_em,e = t_f |V_FB - V_T| / |dV_A|,  t_em,h = t_r |V_FB - V_T| / |dV_A|
so the energy window is
    dE = E_em,e - E_em,h = -2 kT ln( v_th n_i sqrt(sigma_n sigma_p)
                                     sqrt(t_r t_f) |V_FB - V_T| / |dV_A| ).
The argument of the log is << 1 for realistic edges, so dE > 0.
"""
import numpy as np
from math import erf, sqrt
from .constants import q, thermal_voltage

V_TH = 1.0e7   # carrier thermal velocity, cm/s (300 K, Si)


def emission_times(t_r, t_f, V_FB, V_T, dV_A):
    """Time the surface spends between V_FB and V_T on each edge (s).
    Returns (t_em_e, t_em_h): falling edge (electron emission),
    rising edge (hole emission)."""
    frac = abs(V_FB - V_T) / abs(dV_A)
    return t_f * frac, t_r * frac


def emission_levels(t_r, t_f, V_FB, V_T, dV_A, sigma_n=1e-16, sigma_p=1e-16,
                    n_i=1e10, v_th=V_TH, T=300.0):
    """(E_em,e - E_i, E_em,h - E_i) in eV.  Only traps between the two
    levels recombine; outside them the captured carrier is re-emitted
    to its own band before the opposite carrier arrives."""
    kT = thermal_voltage(T)
    te, th = emission_times(t_r, t_f, V_FB, V_T, dV_A)
    Ee = -kT * np.log(sigma_n * v_th * n_i * te)
    Eh = +kT * np.log(sigma_p * v_th * n_i * th)
    return Ee, Eh


def energy_window(t_r, t_f, V_FB, V_T, dV_A, sigma_n=1e-16, sigma_p=1e-16,
                  n_i=1e10, v_th=V_TH, T=300.0):
    """Swept energy window dE = E_em,e - E_em,h (eV).  Clipped at zero:
    if the edges are so slow that both carriers are re-emitted, no
    trap pumps."""
    Ee, Eh = emission_levels(t_r, t_f, V_FB, V_T, dV_A, sigma_n, sigma_p,
                             n_i, v_th, T)
    return np.maximum(Ee - Eh, 0.0)


def icp(D_it, f, A_G, dE):
    """Charge-pumping current (A): I_cp = q f A_G D_it dE.
    D_it: mean interface-trap density over dE (cm^-2 eV^-1);
    f: pulse frequency (Hz); A_G: gate area (cm^2); dE: eV."""
    return q * f * A_G * D_it * dE


def dit_from_icp(I_cp, f, A_G, dE):
    """Mean D_it (cm^-2 eV^-1) from a measured plateau current."""
    return I_cp / (q * f * A_G * dE)


def nit_from_icp(I_cp, f, A_G):
    """Traps pumped per unit area per cycle, N_it = I_cp/(q f A_G)
    (cm^-2).  Used directly for stress-induced Delta N_it."""
    return I_cp / (q * f * A_G)


def edge_stretch(D_it, Cox, dpsi=0.5):
    """Extra gate voltage (V) needed to move the surface potential by
    dpsi (V) when interface traps must be charged: q D_it dpsi / C_ox.
    Sets the minimum width of a base-level transition edge."""
    return q * D_it * dpsi / Cox


def _ncdf(x):
    return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(x) / sqrt(2.0)))


def elliot_curve(V_base, dV_A, V_FB, V_T, I_max, sigma_FB=0.05, sigma_T=0.05):
    """Analytical base-level (Elliot) curve at constant amplitude dV_A.

    Current flows only when the pulse reaches both states each cycle:
      top    V_base + dV_A > V_T   (rising edge at V_base = V_T - dV_A)
      bottom V_base       < V_FB   (falling edge at V_base = V_FB)
    The local V_T and V_FB are treated as Gaussian-distributed along the
    channel (edges near S/D, D_it stretch-out), which sets the edge widths.
    Illustration only: not a self-consistent device solution.
    """
    V_base = np.asarray(V_base, float)
    p_inv = _ncdf((V_base + dV_A - V_T) / sigma_T)
    p_acc = _ncdf((V_FB - V_base) / sigma_FB)
    return I_max * p_inv * p_acc


def swing_with_dit(Cox, Cdep, D_it, T=300.0):
    """Subthreshold swing (V/dec) including interface-trap capacitance
    C_it = q D_it:  S = ln10 (kT/q)(1 + (C_dep + C_it)/C_ox)."""
    return np.log(10) * thermal_voltage(T) * (1 + (Cdep + q * D_it) / Cox)
