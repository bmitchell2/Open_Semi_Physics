"""
High-k / metal-gate stack relations: equivalent oxide thickness and the
threshold voltage set by the gate's effective work function.

Backs the Semiconductor Notes page "High-k / Metal Gate (HKMG) Stack".

EOT of a series stack (interfacial SiO2 + high-k):
    EOT = t_IL + t_hk * (3.9 / k_hk)
Long-channel threshold voltage with flat-band set by work functions
(no oxide charge, classical, no quantization):
  NMOS (p-body, N_A):  V_FB = phi_m - (chi + Eg/2 + phi_F)
                       V_T  = V_FB + 2 phi_F + sqrt(2 eps_si q N_A 2phi_F)/C_ox
  PMOS (n-body, N_D):  V_FB = phi_m - (chi + Eg/2 - phi_F)
                       V_T  = V_FB - 2 phi_F - sqrt(2 eps_si q N_D 2phi_F)/C_ox
phi_m is the effective work function (eV numerically = V) of the gate on
the dielectric, which includes interface dipoles; chi = 4.05 eV, Eg = 1.12 eV.
Units: cm, V, cm^-3, F/cm^2.
"""
import numpy as np

from .constants import q, eps_si, eps_ox, ni300, thermal_voltage

CHI_SI = 4.05
EG_SI = 1.12


def eot(t_il_cm, t_hk_cm, k_hk):
    """Equivalent SiO2 thickness of an interfacial-layer + high-k stack (cm)."""
    return t_il_cm + t_hk_cm * 3.9 / k_hk


def physical_thickness_for_eot(eot_cm, k):
    """Physical thickness of a single dielectric of constant k giving the EOT."""
    return eot_cm * k / 3.9


def vt_nmos(phi_m, N_A, eot_cm, T=300.0):
    Vt = thermal_voltage(T)
    phiF = Vt * np.log(N_A / ni300)
    Cox = eps_ox / eot_cm
    Vfb = phi_m - (CHI_SI + EG_SI / 2 + phiF)
    return Vfb + 2 * phiF + np.sqrt(2 * eps_si * q * N_A * 2 * phiF) / Cox


def vt_pmos(phi_m, N_D, eot_cm, T=300.0):
    Vt = thermal_voltage(T)
    phiF = Vt * np.log(N_D / ni300)
    Cox = eps_ox / eot_cm
    Vfb = phi_m - (CHI_SI + EG_SI / 2 - phiF)
    return Vfb - 2 * phiF - np.sqrt(2 * eps_si * q * N_D * 2 * phiF) / Cox
