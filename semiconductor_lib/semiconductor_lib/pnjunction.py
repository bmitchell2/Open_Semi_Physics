"""
Abrupt silicon p-n junction electrostatics: closed-form depletion
approximation and a numerical nonlinear-Poisson solver (Boltzmann
carriers, damped Newton iteration). Backs the "PN Junction Electrostatics"
note.

Bias model: the applied voltage VA is imposed as a separation of the
electron and hole quasi-Fermi levels with negligible current (quasi-static
depletion regime); it is not a drift-diffusion solution and is not valid
near VA = Vbi. Positive VA = forward bias on the p-side.
Units: cm, V, cm^-3.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .constants import q, eps_si, ni300, thermal_voltage


def built_in_potential(NA, ND, T=300.0, ni=ni300):
    """Vbi = (kT/q) ln(NA ND / ni^2), volts."""
    return thermal_voltage(T) * np.log(NA * ND / ni ** 2)


def debye_length(N, T=300.0, ni=ni300, eps=eps_si):
    """
    Extrinsic Debye (screening) length L_D = sqrt(eps*Vt/(q*N)), cm.

    N is the local majority-carrier concentration (cm^-3), e.g. NA or ND
    in the neutral bulk on one side of a junction. This is the 1/e decay
    length of a small potential perturbation in that bulk; it sets how
    sharply the carrier profile rounds off at each edge of the depletion
    approximation, not the total depletion width (see
    depletion_width_over_debye_length).
    """
    return np.sqrt(eps * thermal_voltage(T) / (q * N))


def depletion_width_over_debye_length(NA, ND, T=300.0, ni=ni300, eps=eps_si):
    """
    Ratio W / L_D of the total equilibrium (zero-bias) depletion width to
    the *smaller* of the two sides' Debye lengths (the more heavily doped
    side, which has the shorter L_D and the sharper edge).

    Closed form for the symmetric case NA=ND=N collapses to
    W/L_D = 2*sqrt(Vbi/Vt); this function returns the general asymmetric
    result from the depletion approximation and debye_length directly, so
    the two stay consistent by construction. The ratio is what actually
    explains why a several-hundred-nm depletion width can coexist with a
    Debye length of only a few tens of nm: the edge rounds off over about
    one L_D, but the field needs several L_D worth of integrated charge
    (Poisson's equation) to support the full built-in potential, which is
    typically 25-35 kT/q.
    """
    Vbi = built_in_potential(NA, ND, T, ni)
    W = depletion_approximation(NA, ND, 0.0, T, ni, eps)["W"]
    LD = min(debye_length(NA, T, ni, eps), debye_length(ND, T, ni, eps))
    return W / LD


def depletion_approximation(NA, ND, VA=0.0, T=300.0, ni=ni300, eps=eps_si):
    """
    Depletion approximation. Returns dict with Vbi, W, xn, xp (cm),
    Emax (V/cm) and C_per_area (F/cm^2).
    """
    Vbi = built_in_potential(NA, ND, T, ni)
    W = np.sqrt(2 * eps * (Vbi - VA) / q * (1 / NA + 1 / ND))
    return dict(Vbi=Vbi, W=W, xn=W * NA / (NA + ND), xp=W * ND / (NA + ND),
                Emax=2 * (Vbi - VA) / W, C_per_area=eps / W)


@dataclass
class PNSolution:
    x: np.ndarray        # cm, metallurgical junction at 0, p-side x<0
    psi: np.ndarray      # V
    E: np.ndarray        # V/cm
    n: np.ndarray        # cm^-3
    p: np.ndarray        # cm^-3
    rho: np.ndarray      # C/cm^3
    iterations: int
    residual: float      # max |Poisson residual| relative to q*max(NA,ND)/eps/Vt


def solve_pn_poisson(NA, ND, VA=0.0, T=300.0, ni=ni300, eps=eps_si,
                     L=2.5e-4, N=20001, tol=1e-10, max_iter=200):
    """Nonlinear Poisson solution on [-L, L] with ohmic-contact boundary values."""
    Vt = thermal_voltage(T)
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    Nd = np.where(x < 0, -NA, ND)
    c = q / (eps * Vt)
    uL = (VA - Vt * np.log(NA / ni)) / Vt
    uR = (Vt * np.log(ND / ni)) / Vt
    u = np.where(x < 0, uL, uR).astype(float)

    def carriers(u):
        return (ni * np.exp(np.clip(VA / Vt - u, -700, 700)),
                ni * np.exp(np.clip(u, -700, 700)))

    for it in range(max_iter):
        p, n = carriers(u)
        F = np.zeros(N)
        F[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2 + c * (p - n + Nd)[1:-1]
        J = diags([np.ones(N - 1) / dx ** 2, -2 / dx ** 2 - c * (p + n),
                   np.ones(N - 1) / dx ** 2], [-1, 0, 1], format="lil")
        J[0, :] = 0; J[0, 0] = 1; F[0] = u[0] - uL
        J[-1, :] = 0; J[-1, -1] = 1; F[-1] = u[-1] - uR
        du = spsolve(J.tocsr(), -F)
        step = np.max(np.abs(du))
        u = u + min(1.0, 2.0 / step) * du
        if step < tol:
            break
    p, n = carriers(u)
    psi = u * Vt
    rho = q * (p - n + Nd)
    return PNSolution(x, psi, -np.gradient(psi, x), n, p, rho, it,
                      np.max(np.abs(F[1:-1])) / (c * max(NA, ND)))


def solve_step_junction_poisson(N_left, N_right, VA=0.0, T=300.0, ni=ni300,
                                eps=eps_si, L=None, N=20001, tol=1e-10,
                                max_iter=200):
    """
    Nonlinear-Poisson equilibrium solution for an abrupt step junction with
    arbitrary SIGNED net doping on each side: N_left, N_right in cm^-3,
    positive for net donors (n-type), negative for net acceptors (p-type),
    and 0 for undoped (intrinsic) material. This generalises
    solve_pn_poisson (which assumes p-type left / n-type right, both
    nonzero) to also cover one-sided and n-type/intrinsic or
    p-type/intrinsic junctions, using
        n - p = Nnet,  n*p = ni^2  =>  n = ni*e^u, p = ni*e^-u,
        u = asinh(Nnet / (2*ni))
    for the charge-neutral bulk boundary condition, which is well defined
    (u=0) even when one side is exactly intrinsic -- unlike a boundary
    condition written as Vt*ln(N/ni), which diverges there.

    If L is None, the half-domain length is chosen automatically as
    40 times the larger of the two sides' Debye lengths (using ni as the
    floor carrier concentration on an intrinsic side), which is enough for
    the bulk boundary conditions to sit in genuinely field-free material.

    Returns a PNSolution. For NA=-N_left, ND=N_right both large compared
    with ni, this matches solve_pn_poisson to numerical precision (see
    tests/test_pnjunction_step_junction.py); for one side equal to 0 it
    is the n-type/intrinsic (or p-type/intrinsic) case, where the
    depletion-approximation formulas above do not apply because there are
    no fixed ions on the intrinsic side, only piled-up majority carriers,
    and the relevant length scale is that side's own (much larger) Debye
    length.
    """
    Vt = thermal_voltage(T)
    if L is None:
        Nfloor = max(abs(N_left), ni)
        Nfloor_r = max(abs(N_right), ni)
        L = 40.0 * max(debye_length(Nfloor, T, ni, eps),
                       debye_length(Nfloor_r, T, ni, eps))
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    Nd = np.where(x < 0, N_left, N_right)
    c = q / (eps * Vt)
    uL = np.arcsinh(N_left / (2 * ni))
    uR = np.arcsinh(N_right / (2 * ni))
    u = np.where(x < 0, uL, uR).astype(float)

    def carriers(u):
        return (ni * np.exp(np.clip(VA / Vt - u, -700, 700)),
                ni * np.exp(np.clip(u, -700, 700)))

    it = 0
    for it in range(max_iter):
        p, n = carriers(u)
        F = np.zeros(N)
        F[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2 + c * (p - n + Nd)[1:-1]
        J = diags([np.ones(N - 1) / dx ** 2, -2 / dx ** 2 - c * (p + n),
                   np.ones(N - 1) / dx ** 2], [-1, 0, 1], format="lil")
        J[0, :] = 0; J[0, 0] = 1; F[0] = u[0] - uL
        J[-1, :] = 0; J[-1, -1] = 1; F[-1] = u[-1] - uR
        du = spsolve(J.tocsr(), -F)
        step = np.max(np.abs(du))
        u = u + min(1.0, 2.0 / step) * du
        if step < tol:
            break
    p, n = carriers(u)
    psi = u * Vt
    rho = q * (p - n + Nd)
    return PNSolution(x, psi, -np.gradient(psi, x), n, p, rho, it,
                      np.max(np.abs(F[1:-1])) / (c * max(abs(N_left), abs(N_right), ni)))


def numerical_widths(sol, NA, ND):
    """
    Depletion widths (xp, xn in cm) from the integrated space charge on each
    side of the metallurgical junction. Uses the rectangle rule so the
    integral matches the solver's finite-difference Gauss law (the doping
    step sits on a grid node; a trapezoid split there drops about one grid
    cell of charge and breaks charge neutrality at the 1e-3 level).
    """
    dx = sol.x[1] - sol.x[0]
    i0 = int(np.argmin(np.abs(sol.x)))
    xp = abs(sol.rho[:i0].sum()) * dx / (q * NA)
    xn = sol.rho[i0:].sum() * dx / (q * ND)
    return xp, xn
