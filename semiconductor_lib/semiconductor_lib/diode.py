"""
Biased p-n diode: forward I-V with space-charge-region (SCR) current and
series resistance, reverse leakage components versus temperature, junction
breakdown estimates, charge storage / diffusion capacitance, and the
forward-to-reverse switching (reverse-recovery) transient.

Backs the Semiconductor Notes pages "PN Junction Diode Current (Forward
Bias, Reverse Leakage, and Breakdown)" and "PN Diode Charge Storage
(Diffusion Capacitance and Reverse Recovery)". Complements pnjunction.py,
which covers equilibrium electrostatics.

Models and assumptions
----------------------
- Long-base ideal diode (Shockley): J0 = q ni^2 (Dn/(Ln NA) + Dp/(Lp ND)).
  Low-level injection only; high-level injection is NOT modelled.
- SCR recombination/generation (Sah-Noyce-Shockley, Hu Eq. 4.9.9):
  J_scr = q ni W/tau_dep * (exp(V/2Vt) - 1), with W from the depletion
  approximation at the junction voltage. Under reverse bias this is the
  generation current q ni W / tau_dep.
- ni(T) uses the dopants.py Ioffe fits, rescaled so ni(300 K) = ni300, so
  numbers stay consistent with the rest of the package. D and tau are held
  temperature-independent unless the caller varies them (stated on the
  pages that use these functions).
- Charge control: Q = I tau, C_diff = dQ/dV = tau I / (n Vt) (Hu 4.10-4.11).
  The small-signal admittance of a long-base diode gives half this value at
  low frequency; the charge-control value is the quasi-static one.
- Reverse recovery: 1-D time-dependent minority-carrier diffusion in the
  lightly doped side of a one-sided (N+P) long diode, backward-Euler finite
  differences, Shockley boundary condition at the junction edge, series
  resistor to the source. Junction (depletion) capacitance and hole
  injection into the N+ side are neglected, so the model reproduces the
  constant-current storage phase and the diffusion-limited decay phase of
  Kingston (Proc. IRE 42, 829, 1954).

Verified (tests/test_diode.py): ideality ~2 at low bias and ~1 at mid bias;
dV/dT at fixed current matches the closed-form tempco to <3 % and is about
-2 mV/K; integrated excess-carrier profile equals I*tau to <0.5 %;
finite-difference storage time matches Kingston's erf result to <1 % when
the actual storage-phase reverse current is used;
Sze-Gibbons breakdown voltage gives 60 V at 1e16 cm^-3.

Units: cm, s, V, A, cm^-3, F. Positive V = forward bias.
"""
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import erfinv

from .constants import q, eps_si, ni300, thermal_voltage
from . import dopants, pnjunction


# --------------------------------------------------------------------------
# Temperature-dependent intrinsic density
# --------------------------------------------------------------------------
def ni_T(T):
    """Intrinsic density ni(T), cm^-3, from the dopants.py Ioffe fit,
    rescaled so ni(300 K) = ni300 (1e10 cm^-3)."""
    return ni300 * dopants.intrinsic_concentration(T) / dopants.intrinsic_concentration(300.0)


# --------------------------------------------------------------------------
# DC current components
# --------------------------------------------------------------------------
def diffusion_saturation_current_density(NA, ND, Dn, Dp, tau_n, tau_p, T=300.0):
    """Long-base Shockley saturation current density J0 (A/cm^2)."""
    ni = ni_T(T)
    Ln, Lp = np.sqrt(Dn * tau_n), np.sqrt(Dp * tau_p)
    return q * ni ** 2 * (Dn / (Ln * NA) + Dp / (Lp * ND))


def injection_ratio(NA, ND, Dn, Dp, tau_n, tau_p):
    """Ratio of electron current injected into P to hole current injected
    into N, J_n/J_p = (Dn/(Ln NA)) / (Dp/(Lp ND)). Large when ND >> NA:
    injection goes mainly into the lighter-doped side."""
    Ln, Lp = np.sqrt(Dn * tau_n), np.sqrt(Dp * tau_p)
    return (Dn / (Ln * NA)) / (Dp / (Lp * ND))


def depletion_width(NA, ND, Vj, T=300.0):
    """Depletion width (cm) at junction voltage Vj, clipped so that
    Vbi - Vj >= 2 Vt (the depletion approximation fails near Vbi)."""
    ni = ni_T(T)
    Vbi = pnjunction.built_in_potential(NA, ND, T, ni)
    V = np.minimum(Vj, Vbi - 2 * thermal_voltage(T))
    return np.sqrt(2 * eps_si * (Vbi - V) / q * (1 / NA + 1 / ND))


def junction_current_density(Vj, NA, ND, Dn, Dp, tau_n, tau_p, tau_dep, T=300.0):
    """Returns (J_total, J_diff, J_scr) in A/cm^2 at junction voltage Vj."""
    Vt = thermal_voltage(T)
    Vj = np.asarray(Vj, float)
    J0 = diffusion_saturation_current_density(NA, ND, Dn, Dp, tau_n, tau_p, T)
    Jd = J0 * np.expm1(Vj / Vt)
    W = depletion_width(NA, ND, Vj, T)
    Js = q * ni_T(T) * W / tau_dep * np.expm1(Vj / (2 * Vt))
    return Jd + Js, Jd, Js


def forward_iv(V_terminal, area, Rs, NA, ND, Dn, Dp, tau_n, tau_p, tau_dep, T=300.0):
    """Terminal I-V including series resistance: V = Vj + I Rs.
    Returns dict with I (A), Vj, I_diff, I_scr (A)."""
    V_terminal = np.atleast_1d(np.asarray(V_terminal, float))
    I = np.zeros_like(V_terminal)
    Vj = np.zeros_like(V_terminal)

    def f(vj, v):
        return area * junction_current_density(vj, NA, ND, Dn, Dp, tau_n, tau_p, tau_dep, T)[0] * Rs + vj - v

    for i, v in enumerate(V_terminal):
        if Rs == 0:
            Vj[i] = v
        else:
            Vj[i] = brentq(f, min(v, 0.0) - 1e-9, max(v, 0.0) + 1e-9, args=(v,), xtol=1e-14)
    Jt, Jd, Js = junction_current_density(Vj, NA, ND, Dn, Dp, tau_n, tau_p, tau_dep, T)
    return dict(I=area * Jt, Vj=Vj, I_diff=area * Jd, I_scr=area * Js)


def local_ideality(V, I, T=300.0):
    """Local ideality factor n = (1/Vt) dV/d(ln I) (numerical)."""
    return np.gradient(np.asarray(V, float), np.log(np.asarray(I, float))) / thermal_voltage(T)


def forward_voltage_at_current(J, NA, ND, Dn, Dp, tau_n, tau_p, T=300.0):
    """Ideal (diffusion-only) junction voltage (V) that carries current density J."""
    J0 = diffusion_saturation_current_density(NA, ND, Dn, Dp, tau_n, tau_p, T)
    return thermal_voltage(T) * np.log1p(J / J0)


def forward_voltage_tempco(J, NA, ND, Dn, Dp, tau_n, tau_p, T=300.0, dT=0.5):
    """Numerical dV/dT (V/K) at fixed current density, D and tau held fixed."""
    Vp = forward_voltage_at_current(J, NA, ND, Dn, Dp, tau_n, tau_p, T + dT)
    Vm = forward_voltage_at_current(J, NA, ND, Dn, Dp, tau_n, tau_p, T - dT)
    return (Vp - Vm) / (2 * dT)


def forward_voltage_tempco_analytic(V, T=300.0):
    """Closed form dV/dT = (V - Eg0_lin - 3 kT/q)/T for J0 ∝ ni^2 ∝ T^3
    exp(-Eg/kT), where Eg0_lin = Eg(T) - T dEg/dT is the linearly
    extrapolated 0-K gap (≈1.2 eV for Si near 300 K)."""
    h = 0.5
    dEg = (dopants.band_gap(T + h) - dopants.band_gap(T - h)) / (2 * h)
    Eg0_lin = dopants.band_gap(T) - T * dEg
    return (V - Eg0_lin - 3 * thermal_voltage(T)) / T


# --------------------------------------------------------------------------
# Reverse leakage and breakdown
# --------------------------------------------------------------------------
def reverse_leakage_components(T, VR, NA, ND, Dn, Dp, tau_n, tau_p, tau_g):
    """Reverse-bias leakage (A/cm^2) at reverse voltage VR > 0:
    returns (J_diff, J_gen). J_diff ∝ ni^2 (activation ≈ Eg),
    J_gen = q ni W / tau_g ∝ ni (activation ≈ Eg/2)."""
    Jd = diffusion_saturation_current_density(NA, ND, Dn, Dp, tau_n, tau_p, T)
    W = depletion_width(NA, ND, -VR, T)
    Jg = q * ni_T(T) * W / tau_g
    return Jd, Jg


def arrhenius_activation_energy(T, J):
    """Least-squares Arrhenius activation energy (eV) from J(T):
    slope of ln J vs 1/(kT)."""
    kT = thermal_voltage(np.asarray(T, float))
    slope = np.polyfit(1.0 / kT, np.log(np.asarray(J, float)), 1)[0]
    return -slope


def breakdown_voltage_one_sided(N, E_crit):
    """Avalanche/tunnelling breakdown of a one-sided abrupt junction,
    V_B = eps E_crit^2 / (2 q N) (V_bi neglected). N = lighter doping."""
    return eps_si * E_crit ** 2 / (2 * q * N)


def breakdown_voltage_sze(N, Eg=1.12):
    """Empirical avalanche breakdown of a one-sided abrupt junction
    (Sze and Gibbons 1966): V_B = 60 (Eg/1.1)^1.5 (N/1e16)^-0.75 V."""
    return 60.0 * (Eg / 1.1) ** 1.5 * (N / 1e16) ** -0.75


def critical_field_from_vb(N, VB):
    """Peak field (V/cm) at breakdown implied by V_B for a one-sided junction."""
    return np.sqrt(2 * q * N * VB / eps_si)


def miller_multiplication(V, VB, n=4.0):
    """Empirical avalanche multiplication M = 1 / (1 - (V/VB)^n), V < VB."""
    return 1.0 / (1.0 - (np.asarray(V, float) / VB) ** n)


# --------------------------------------------------------------------------
# Charge storage and diffusion capacitance
# --------------------------------------------------------------------------
def excess_minority_profile(x, V, n_p0, L, T=300.0):
    """Excess minority density (cm^-3) at distance x (cm) from the edge of
    the quasi-neutral region of a long base: n_p0 (e^{V/Vt}-1) e^{-x/L}."""
    return n_p0 * np.expm1(V / thermal_voltage(T)) * np.exp(-np.asarray(x, float) / L)


def stored_charge(I, tau):
    """Charge-control stored minority charge Q = I tau (C)."""
    return I * tau


def diffusion_capacitance(I, tau, T=300.0, n=1.0):
    """Charge-control diffusion capacitance C = tau I / (n Vt) (F)."""
    return tau * I / (n * thermal_voltage(T))


def short_base_transit_time(W_B, D):
    """Stored-charge time constant of a short base (W_B << L): W_B^2 / (2 D) (s)."""
    return W_B ** 2 / (2 * D)


def storage_time_kingston(tau, I_F, I_R):
    """Kingston (1954) storage time of a long planar diode switched from
    forward current I_F to constant reverse current I_R:
    erf(sqrt(t_s/tau)) = I_F / (I_F + I_R)."""
    return tau * erfinv(I_F / (I_F + I_R)) ** 2


def storage_time_charge_control(tau, I_F, I_R):
    """Lumped charge-control estimate t_s = tau ln(1 + I_F/I_R) (time for
    the total stored charge, not the junction-edge density, to reach zero;
    overestimates the Kingston value)."""
    return tau * np.log1p(I_F / I_R)


def simulate_reverse_recovery(tau, D, NA, area, R, V_F, V_R, t_end_over_tau=1.5,
                              n_grid=1200, L_domain=6.0, dt_over_tau=5e-4, T=300.0):
    """
    Finite-difference switching transient of a one-sided N+P long diode.

    The source steps from +V_F to -V_R at t = 0 through series resistor R.
    Solves d(dn)/dt = D d2(dn)/dx2 - dn/tau on the P side, x in [0, L_domain*L],
    with dn(0) = n_p0 (exp(v/Vt) - 1) and dn(end) = 0; diode current
    I = q A D (-d dn/dx)|0 must equal (V_s - v)/R at every step.

    Returns dict: t (s), I (A), v (junction V), I_F, t_s (first time v <= 0).
    """
    Vt = thermal_voltage(T)
    ni = ni_T(T)
    n_p0 = ni ** 2 / NA
    L = np.sqrt(D * tau)
    xL = L_domain * L
    h = xL / n_grid
    m = n_grid - 1                    # interior unknowns x_1 .. x_{n-1}

    def solve_system(inv_dt, rhs_old):
        # (inv_dt + 1/tau + 2D/h^2) u_i - D/h^2 (u_{i-1} + u_{i+1}) = inv_dt*old_i
        a = D / h ** 2
        diag = np.full(m, inv_dt + 1.0 / tau + 2 * a)
        ab = np.zeros((3, m))
        ab[0, 1:] = -a
        ab[1, :] = diag
        ab[2, :-1] = -a
        b0 = inv_dt * rhs_old
        e = np.zeros(m)
        e[0] = a                        # coefficient multiplying dn(0)
        ua = solve_banded((1, 1), ab, b0)
        ub = solve_banded((1, 1), ab, e)
        return ua, ub                   # u = ua + dn0 * ub

    def flux_affine(ua, ub):
        # I = qAD * (3 dn0 - 4 u1 + u2)/(2h) = alpha + beta*dn0
        alpha = q * area * D * (-4 * ua[0] + ua[1]) / (2 * h)
        beta = q * area * D * (3 - 4 * ub[0] + ub[1]) / (2 * h)
        return alpha, beta

    def solve_v(Vs, alpha, beta):
        g = lambda v: (Vs - v) / R - (alpha + beta * n_p0 * np.expm1(v / Vt))
        return brentq(g, -abs(Vs) - 5.0, 1.2, xtol=1e-13)

    # steady forward state (inv_dt = 0)
    ua, ub = solve_system(0.0, np.zeros(m))
    alpha, beta = flux_affine(ua, ub)
    v = solve_v(V_F, alpha, beta)
    dn0 = n_p0 * np.expm1(v / Vt)
    u = ua + dn0 * ub
    I_F = (V_F - v) / R

    dt = dt_over_tau * tau
    n_steps = int(round(t_end_over_tau / dt_over_tau))
    t_out, I_out, v_out = [0.0], [I_F], [v]
    t_s = None
    for k in range(1, n_steps + 1):
        ua, ub = solve_system(1.0 / dt, u)
        alpha, beta = flux_affine(ua, ub)
        v = solve_v(-V_R, alpha, beta)
        dn0 = n_p0 * np.expm1(v / Vt)
        u = ua + dn0 * ub
        I = (-V_R - v) / R
        t = k * dt
        if t_s is None and v <= 0.0:
            # interpolate the zero crossing of v
            t_s = t - dt * v / (v - v_out[-1])
        t_out.append(t)
        I_out.append(I)
        v_out.append(v)
    return dict(t=np.array(t_out), I=np.array(I_out), v=np.array(v_out), I_F=I_F, t_s=t_s,
                L=L, n_p0=n_p0)
