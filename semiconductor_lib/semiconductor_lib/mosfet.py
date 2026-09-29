"""
Long-channel MOSFET DC models, body effect, channel-length modulation, and
random-dopant-fluctuation (RDF) threshold-voltage mismatch.

Units follow the rest of the package: cm, V, cm^-3, F/cm^2, A.
n-channel device on a p-type body; V_SB >= 0 is reverse body bias.

Models
------
- Charge-control (square-law) model with bulk-charge factor m.
- Brews charge-sheet model (Solid-State Electronics 21, 345, 1978):
  surface potential at source and drain ends, drift + diffusion, continuous
  from weak to strong inversion.
- Body effect: uniform doping (gamma formulation) and an abrupt two-layer
  (retrograde / low-high) vertical profile solved exactly in the depletion
  approximation.
- Channel-length modulation: 1-D depletion estimate and quasi-2-D
  (Ko/Hu, BSIM-style) estimate of the velocity-saturated region length.
- RDF: 1-D charge-sheet sensitivity integral (Stolk et al., IEEE TED 45,
  1960, 1998). For uniform doping it reduces to
  sigma_VT = (q/Cox) * sqrt(N_A * W_dep / (3 * L * W)).

Verified (tests/test_mosfet.py): square law with phi_0 = 2 phi_F + 6 kT/q
matches the charge-sheet model within 5 % in strong inversion; charge-sheet
subthreshold swing within 10 % of ln(10) m kT/q; two-layer solver matches
direct Poisson integration to 0.1 %.
"""
import numpy as np
from scipy.optimize import brentq

from .constants import q, eps_si, eps_ox, ni300, thermal_voltage


# --------------------------------------------------------------------------
# Basic electrostatic quantities
# --------------------------------------------------------------------------
def phi_F(N_A, T=300.0):
    """Bulk Fermi potential (V) of a p-type body."""
    return thermal_voltage(T) * np.log(N_A / ni300)


def cox_from_tox(tox_cm, eps=eps_ox):
    """Oxide capacitance per area (F/cm^2)."""
    return eps / tox_cm


def body_factor_gamma(N_A, Cox):
    """Body-effect coefficient gamma = sqrt(2 q eps_si N_A)/Cox, V^0.5."""
    return np.sqrt(2.0 * q * eps_si * N_A) / Cox


def bulk_charge_factor_m(N_A, Cox, V_SB=0.0, T=300.0, n_phit=0.0):
    """m = 1 + gamma / (2 sqrt(phi_0 + V_SB)) = 1 + C_dep/C_ox,
    phi_0 = 2 phi_F + n_phit * kT/q (see vt_uniform)."""
    phi0 = 2 * phi_F(N_A, T) + n_phit * thermal_voltage(T)
    return 1.0 + body_factor_gamma(N_A, Cox) / (2.0 * np.sqrt(phi0 + V_SB))


def vt_uniform(N_A, Cox, V_FB=0.0, V_SB=0.0, T=300.0, n_phit=0.0):
    """Long-channel threshold voltage (V, gate-to-source) for uniform doping.
    Strong-inversion surface potential phi_0 = 2 phi_F + n_phit * kT/q.
    n_phit = 0 is the textbook criterion; n_phit ~ 6 (Tsividis) matches the
    linearly extrapolated V_T of the charge-sheet model to within ~0.01 V."""
    phi0 = 2 * phi_F(N_A, T) + n_phit * thermal_voltage(T)
    g = body_factor_gamma(N_A, Cox)
    return V_FB + phi0 + g * np.sqrt(phi0 + V_SB)


def depletion_two_layer(psi, N1, N2, x_s):
    """Depletion width (cm) and depletion charge (C/cm^2) for band bending psi
    in an abrupt two-layer profile: N1 for 0<x<x_s, N2 for x>x_s.
    Depletion approximation, exact for the step profile."""
    psi1 = q * N1 * x_s ** 2 / (2 * eps_si)          # band bending when W = x_s
    if psi <= psi1:
        W = np.sqrt(2 * eps_si * psi / (q * N1))
        return W, q * N1 * W
    # psi = psi1 + q N2 d x_s/eps + q N2 d^2/(2 eps),  d = W - x_s
    # (second term: field of the heavy-layer charge crossing the light layer)
    a = q * N2 / (2 * eps_si)
    b = q * N2 * x_s / eps_si
    c = psi1 - psi
    d = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return x_s + d, q * (N1 * x_s + N2 * d)


def vt_two_layer(N1, N2, x_s, Cox, V_FB=0.0, V_SB=0.0, T=300.0):
    """Threshold voltage for a low-high (retrograde) two-layer profile.
    Inversion defined by surface doping N1 (psi_s = 2 phi_F(N1) + V_SB)."""
    pF = phi_F(N1, T)
    _, Qd = depletion_two_layer(2 * pF + V_SB, N1, N2, x_s)
    return V_FB + 2 * pF + Qd / Cox


# --------------------------------------------------------------------------
# Drain-current models
# --------------------------------------------------------------------------
def id_square_law(V_GS, V_DS, V_T, beta, m=1.0, lam=0.0):
    """Charge-control (square-law) drain current (A). beta = mu Cox W/L (A/V^2).
    Zero below V_T (the model has no subthreshold conduction)."""
    Vov = np.maximum(np.asarray(V_GS, float) - V_T, 0.0)
    Vdsat = Vov / m
    V_DS = np.asarray(V_DS, float)
    lin = beta * (Vov * V_DS - 0.5 * m * V_DS ** 2)
    sat = 0.5 * beta * Vov ** 2 / m
    return np.where(V_DS < Vdsat, lin, sat) * (1 + lam * V_DS)


def surface_potential(V_GB, V_ch, N_A, Cox, V_FB=0.0, T=300.0):
    """Surface potential (V) from the implicit charge-sheet relation
    V_GB - V_FB = psi + gamma sqrt(psi + phi_t exp((psi - 2 phi_F - V_ch)/phi_t)),
    where V_ch is the local channel (quasi-Fermi) potential referred to body."""
    pt = thermal_voltage(T)
    pF = phi_F(N_A, T)
    g = body_factor_gamma(N_A, Cox)
    Vg = V_GB - V_FB

    def f(psi):
        arg = np.clip((psi - 2 * pF - V_ch) / pt, -200, 200)
        return psi + g * np.sqrt(max(psi + pt * np.exp(arg), 0.0)) - Vg

    return brentq(f, 1e-9, max(Vg, 1e-6) + 1e-3, xtol=1e-12)


def id_charge_sheet(V_GS, V_DS, N_A, Cox, beta, V_FB=0.0, V_SB=0.0, T=300.0):
    """Brews charge-sheet drain current (A); drift + diffusion, valid from weak
    through strong inversion. beta = mu Cox W/L."""
    pt = thermal_voltage(T)
    g = body_factor_gamma(N_A, Cox)
    V_GB = V_GS + V_SB
    ps0 = surface_potential(V_GB, V_SB, N_A, Cox, V_FB, T)
    psL = surface_potential(V_GB, V_SB + V_DS, N_A, Cox, V_FB, T)
    Vg = V_GB - V_FB
    drift = (Vg * (psL - ps0) - 0.5 * (psL ** 2 - ps0 ** 2)
             - (2.0 / 3.0) * g * (psL ** 1.5 - ps0 ** 1.5))
    diff = pt * (psL - ps0) + pt * g * (np.sqrt(psL) - np.sqrt(ps0))
    return beta * (drift + diff)


def subthreshold_swing(m, T=300.0):
    """Ideal subthreshold swing (V/decade) = ln(10) m kT/q."""
    return np.log(10) * m * thermal_voltage(T)


# --------------------------------------------------------------------------
# Channel-length modulation
# --------------------------------------------------------------------------
def clm_delta_L_1d(V_DS, V_Dsat, N_A):
    """1-D estimate of the pinched-off length (cm): one-sided junction
    depletion driven by V_DS - V_Dsat. Overestimates the doping sensitivity
    because it ignores the gate's 2-D field; use for trends only."""
    dV = np.maximum(np.asarray(V_DS, float) - V_Dsat, 0.0)
    return np.sqrt(2 * eps_si * dV / (q * N_A))


def clm_delta_L_quasi2d(V_DS, V_Dsat, tox, xj, E_sat=4e4):
    """Quasi-2-D (Ko/Hu) length of the velocity-saturated region (cm):
    dL = l ln[((V_DS-V_Dsat)/l + E_m)/E_sat],  E_m = sqrt(((V_DS-V_Dsat)/l)^2 + E_sat^2),
    l = sqrt(3 tox xj) (characteristic length for SiO2/Si). E_sat in V/cm."""
    l = np.sqrt(3.0 * tox * xj)
    dV = np.maximum(np.asarray(V_DS, float) - V_Dsat, 0.0)
    Em = np.sqrt((dV / l) ** 2 + E_sat ** 2)
    return l * np.log((dV / l + Em) / E_sat)


# --------------------------------------------------------------------------
# Random dopant fluctuation
# --------------------------------------------------------------------------
def sigma_vt_rdf(x, N, W_dep, Cox, L, W):
    """RDF threshold sigma (V) for a vertical profile N(x) (arrays, cm, cm^-3):
    sigma^2 = (q/Cox)^2 / (L W) * integral_0^Wdep N(x) (1 - x/Wdep)^2 dx.
    The (1 - x/Wdep)^2 weight: a dopant at the surface shifts V_T fully, one at
    the depletion edge not at all (1-D charge-sheet sensitivity). Optimistic
    for deep heavy layers; atomistic simulation shows larger spreads."""
    x = np.asarray(x, float)
    N = np.asarray(N, float)
    mask = x <= W_dep
    wgt = (1 - x[mask] / W_dep) ** 2
    integral = np.trapezoid(N[mask] * wgt, x[mask])
    return (q / Cox) * np.sqrt(integral / (L * W))


def sigma_vt_rdf_uniform(N_A, Cox, L, W, T=300.0):
    """Closed form for uniform doping: (q/Cox) sqrt(N_A W_dep / (3 L W))."""
    W_dep = np.sqrt(4 * eps_si * phi_F(N_A, T) / (q * N_A))
    return (q / Cox) * np.sqrt(N_A * W_dep / (3 * L * W))
