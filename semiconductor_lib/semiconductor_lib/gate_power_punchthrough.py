"""
Small device/circuit relations used by several Semiconductor Notes pages:
polysilicon gate depletion in inversion, a first-order (1-D) punchthrough
voltage, and CMOS switching / static power.

Poly depletion (Hu Sec. 5.8, Eqs 5.8.1-5.8.3, 5.9.2):
    W_dpoly = eps_ox E_ox / (q N_poly) = |Q_gate| / (q N_poly)
    phi_poly = q N_poly W_dpoly^2 / (2 eps_si)
    T_oxe = T_ox + W_dpoly/3 + T_inv/3   (3 ~ eps_si/eps_ox)
  The gate depletes when the gate charge has the sign that removes the
  poly's majority carriers: N+ poly with positive gate charge (nFET in
  depletion/inversion), P+ poly with negative gate charge (pFET).

1-D punchthrough estimate: source and drain depletion regions in a body of
uniform doping N touch when
    sqrt(2 eps (V_bi)/(qN)) + sqrt(2 eps (V_bi + V_D)/(qN)) = L_met.
This ignores 2-D field sharing and the gate, so it overestimates V_pt; it is
a scaling statement (V_pt ~ q N L^2 / (2 eps)), not a design value.

CMOS power:
    P_dyn = alpha C_L V_DD^2 f      (charging/discharging load capacitance)
    P_static = I_leak V_DD
    Ratioed (NMOS-load) gate with output low: P = I_load V_DD continuously.

Units: cm, V, cm^-3, F, Hz, W.
"""
import numpy as np

from .constants import q, eps_si, eps_ox


def poly_depletion_width_cm(Q_gate_C_cm2, N_poly):
    return np.abs(Q_gate_C_cm2) / (q * N_poly)


def poly_depletion_potential(W_cm, N_poly):
    return q * N_poly * W_cm**2 / (2 * eps_si)


def toxe_cm(t_ox_cm, W_dpoly_cm, T_inv_cm=0.0):
    k = eps_si / eps_ox
    return t_ox_cm + W_dpoly_cm / k + T_inv_cm / k


def solve_poly_depletion(Vov, t_ox_cm, N_poly, Q_dep_C_cm2=0.0, T_inv_cm=0.0,
                         n_iter=200):
    """Self-consistent gate charge and poly depletion for gate overdrive
    Vov = V_G - V_T (V) in strong inversion. The overdrive is shared between
    the oxide (incl. inversion-layer centroid) and the poly depletion drop:
        Vov = Q_inv / C_ox' + phi_poly(Q_gate),  Q_gate = Q_inv + Q_dep,
    with C_ox' = eps_ox/(t_ox + T_inv/3). Returns dict with Q_inv, W_dpoly,
    phi_poly, T_oxe and the inversion charge without poly depletion."""
    k = eps_si / eps_ox
    Cox_eff = eps_ox / (t_ox_cm + T_inv_cm / k)
    Qinv = Cox_eff * Vov
    for _ in range(n_iter):
        W = poly_depletion_width_cm(Qinv + Q_dep_C_cm2, N_poly)
        phi = poly_depletion_potential(W, N_poly)
        Qnew = Cox_eff * max(Vov - phi, 0.0)
        if abs(Qnew - Qinv) < 1e-15:
            break
        Qinv = 0.5 * (Qinv + Qnew)
    W = poly_depletion_width_cm(Qinv + Q_dep_C_cm2, N_poly)
    return dict(Q_inv=Qinv, Q_inv_no_polydep=Cox_eff * Vov, W_dpoly_cm=W,
                phi_poly=poly_depletion_potential(W, N_poly),
                T_oxe_cm=toxe_cm(t_ox_cm, W, T_inv_cm))


def punchthrough_voltage_1d(L_cm, N, V_bi=1.0):
    """Drain bias at which 1-D source and drain depletion widths sum to L.
    Returns NaN-free value; negative means touching at zero bias."""
    a = 2 * eps_si / (q * N)
    Ws = np.sqrt(a * V_bi)
    Wd_needed = L_cm - Ws
    if Wd_needed <= 0:
        return -V_bi
    return Wd_needed**2 / a - V_bi


def dynamic_power(alpha, C_F, Vdd, f_Hz):
    return alpha * C_F * Vdd**2 * f_Hz


def static_power(I_leak_A, Vdd):
    return I_leak_A * Vdd
