"""Series-resistance and channel-length extraction from linear-region MOSFET arrays.

Implements the channel-resistance (Terada-Muta / Chern) method:

    R_tot(L_m, V_gt) = R_sd + rho_ch(V_gt) * (L_m - dL),
    rho_ch = 1 / (W * mu_eff * C_ox * V_gt)      (linear region, small V_DS)

For each gate overdrive V_gt a straight-line fit of R_tot against mask length
L_m gives slope rho_ch and intercept b = R_sd - rho_ch * dL. Regressing b
against rho_ch over all overdrives gives intercept R_sd and slope -dL, which is
the least-squares form of "where the lines cross".

Convention: L_eff = L_m - dL. Gate overlap gives dL > 0; an extension that
stops short of the gate edge (underlap) gives dL < 0.

Assumptions: R_sd and dL independent of L_m and V_gt; mu_eff independent of
L_m at fixed overdrive. Use constant overdrive (V_GS - V_T(L)), not constant
V_GS, so threshold roll-off/roll-up does not tilt the lines.
"""
import numpy as np


def channel_resistivity(V_gt, W, mu_eff, C_ox):
    """Channel resistance per unit length (ohm/cm) in the linear region.

    V_gt: gate overdrive (V); W: width (cm); mu_eff: cm^2/V/s; C_ox: F/cm^2.
    """
    return 1.0 / (W * mu_eff * C_ox * np.asarray(V_gt, dtype=float))


def r_total_linear(L_m, V_gt, W, mu_eff, C_ox, R_sd, dL):
    """Total linear-region resistance (ohm) for mask lengths L_m (cm)."""
    L_m = np.asarray(L_m, dtype=float)
    return R_sd + channel_resistivity(V_gt, W, mu_eff, C_ox) * (L_m - dL)


def fit_lines(L_m, R_tot):
    """Fit R_tot = b + rho*L_m for each overdrive.

    L_m: (n_L,) mask lengths; R_tot: (n_V, n_L) array, one row per overdrive.
    Returns (rho, b) arrays of length n_V.
    """
    L_m = np.asarray(L_m, dtype=float)
    R_tot = np.atleast_2d(np.asarray(R_tot, dtype=float))
    rho, b = np.empty(len(R_tot)), np.empty(len(R_tot))
    for i, row in enumerate(R_tot):
        rho[i], b[i] = np.polyfit(L_m, row, 1)
    return rho, b


def extract_rsd_dl(L_m, R_tot):
    """Return (R_sd, dL, rho, b, residual_rms) by regressing b against rho.

    residual_rms (ohm) is the scatter of the (rho, b) points about the fitted
    line; a large value relative to R_sd flags bias-dependent R_sd or dL.
    """
    rho, b = fit_lines(L_m, R_tot)
    slope, R_sd = np.polyfit(rho, b, 1)
    resid = b - (R_sd + slope * rho)
    return R_sd, -slope, rho, b, float(np.sqrt(np.mean(resid ** 2)))
