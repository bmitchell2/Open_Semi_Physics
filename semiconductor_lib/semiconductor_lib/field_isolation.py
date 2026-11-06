"""
Field (parasitic) transistor threshold over isolation oxide, and
junction depletion widths for diffusion-to-diffusion punch-through.

Backs the Semiconductor Notes page "Field Isolation Capability: Field
Transistor Threshold and Diffusion-to-Diffusion Punch-Through".

Long-channel threshold of the parasitic MOSFET whose gate dielectric is
the isolation oxide (uniform doping, depletion approximation):
    V_T = V_FB + 2 phi_F + sqrt(2 q eps_s N_A (2 phi_F)) / C_ox
    V_FB = phi_ms - q Q_f / C_ox,   C_ox = eps_ox / t_ox
N_A in cm^-3, t_ox in cm, Q_f fixed oxide charge in charges/cm^2
(positive for net positive charge), phi_ms in V. 2-D fringing through
the trench sidewalls and corners lowers the real value; treat the
result as an upper bound for the trench bottom.
Verified: 1e17 cm^-3 under 300 nm oxide gives 14.3 V; 1e12 cm^-2 of
positive oxide charge lowers it by q*Qf/Cox = 13.9 V, to about 0.4 V.
"""
import numpy as np
from .constants import q, eps_si, eps_ox, ni300, thermal_voltage


def fermi_potential(N_A, T=300.0):
    return thermal_voltage(T) * np.log(np.asarray(N_A, float) / ni300)


def field_threshold(N_A, t_ox_cm, Qf_cm2=0.0, phi_ms=None, T=300.0):
    phiF = fermi_potential(N_A, T)
    Cox = eps_ox / t_ox_cm
    if phi_ms is None:                 # n+ poly over p-type Si
        phi_ms = -(0.56 + phiF)
    Vfb = phi_ms - q * Qf_cm2 / Cox
    Qd = np.sqrt(2.0 * q * eps_si * N_A * 2.0 * phiF)
    return Vfb + 2.0 * phiF + Qd / Cox


def depletion_width_cm(N, V_reverse, V_bi=0.9):
    """One-sided abrupt junction depletion width into the lightly doped
    side (cm)."""
    return np.sqrt(2.0 * eps_si * (V_bi + V_reverse) / (q * N))
