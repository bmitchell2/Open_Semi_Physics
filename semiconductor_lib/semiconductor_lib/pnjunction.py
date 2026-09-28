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
