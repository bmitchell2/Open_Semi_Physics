"""
Short-channel electrostatics: threshold-voltage roll-off, the role of the
source/drain junction depth x_j, and electrostatic scale lengths for bulk,
fully depleted SOI (UTB) and double-gate devices.

Units follow the rest of the package: cm, V, cm^-3, F/cm^2.
n-channel device on a p-type body.

Models
------
- Yau charge-sharing model (Solid-State Electron. 17, 1059, 1974), in the
  form given by Taur & Ning, "Fundamentals of Modern VLSI Devices":
      dVT = (q N_A W_d / Cox) * (x_j / L) * (sqrt(1 + 2 W_d / x_j) - 1)
  dVT is the threshold REDUCTION (positive number). Linear in 1/L, so it
  overestimates roll-off at very short L and ignores V_DS (no DIBL). Its
  value here is the explicit x_j dependence:
      x_j << W_d :  dVT ~ (q N_A W_d / Cox) * sqrt(2 W_d x_j) / L   (~ sqrt x_j)
      x_j >> W_d :  dVT ~ (q N_A W_d^2 / Cox) / L                    (saturates)
- Brews et al. empirical long-channel limit (IEEE EDL 1, 2, 1980):
      L_min[um] = 0.4 * (x_j[um] * t_ox[A] * (W_S + W_D)[um]^2)^(1/3)
  Calibrated on devices with t_ox 10-100 nm and L of about 1-10 um; use
  for trends (cube-root dependence on x_j), not for nanoscale targets.
- Hu's semi-quantitative DIBL length, l_d proportional to
  (T_oxe W_dep X_j)^(1/3) (Hu, Modern Semiconductor Devices for ICs,
  Eq. 7.3.4). Only the proportionality is defined; dibl_length_ratio
  returns ratios between two designs.
- Quasi-2D scale lengths:
      bulk (Liu et al., IEEE TED 40, 86, 1993, eta = 1):
          l = sqrt(eps_si t_ox W_d / eps_ox)
      single-gate fully depleted SOI (Young, IEEE TED 36, 399, 1989):
          lambda = sqrt(eps_si t_si t_ox / eps_ox)
      symmetric double gate: lambda = sqrt(eps_si t_si t_ox / (2 eps_ox))
  and the Liu roll-off/DIBL expression
      dVT = [2 (V_bi - phi_s) + V_DS] * [exp(-L / 2l) + 2 exp(-L / l)].

Verified (tests/test_short_channel.py): Yau limits in both x_j regimes,
monotonic increase of dVT with x_j and decrease with L, Brews cube-root
scaling, Young lambda reproduces a hand value, double-gate lambda is
1/sqrt(2) of single gate, Liu DIBL increases with V_DS and decays with L.
"""
import numpy as np

from .constants import q, eps_si, eps_ox, ni300, thermal_voltage


def depletion_width_at_threshold(N_A, T=300.0):
    """Maximum (threshold) depletion width, cm, uniform p-type body."""
    phi_F = thermal_voltage(T) * np.log(N_A / ni300)
    return np.sqrt(4 * eps_si * phi_F / (q * N_A))


def yau_charge_sharing_dvt(N_A, Cox, L, xj, W_d=None, T=300.0):
    """Threshold reduction (V, positive) from Yau charge sharing.

    N_A cm^-3, Cox F/cm^2, L cm (metallurgical channel length), xj cm.
    W_d defaults to the threshold depletion width."""
    if W_d is None:
        W_d = depletion_width_at_threshold(N_A, T)
    return (q * N_A * W_d / Cox) * (xj / L) * (np.sqrt(1 + 2 * W_d / xj) - 1)


def brews_lmin_um(xj_um, tox_A, Ws_plus_Wd_um):
    """Brews et al. (1980) empirical minimum long-channel length, um."""
    return 0.4 * (xj_um * tox_A * Ws_plus_Wd_um ** 2) ** (1.0 / 3.0)


def dibl_length_ratio(toxe1, wdep1, xj1, toxe2, wdep2, xj2):
    """Ratio l_d2 / l_d1 from Hu's l_d ~ (T_oxe W_dep X_j)^(1/3)."""
    return ((toxe2 * wdep2 * xj2) / (toxe1 * wdep1 * xj1)) ** (1.0 / 3.0)


def scale_length_bulk(tox, W_d, eps_ins=eps_ox):
    """Liu et al. (1993) bulk characteristic length (eta = 1), cm."""
    return np.sqrt(eps_si * tox * W_d / eps_ins)


def scale_length_soi(t_si, tox, eps_ins=eps_ox):
    """Young (1989) single-gate fully depleted SOI natural length, cm."""
    return np.sqrt(eps_si * t_si * tox / eps_ins)


def scale_length_double_gate(t_si, tox, eps_ins=eps_ox):
    """Symmetric double-gate natural length, cm."""
    return np.sqrt(eps_si * t_si * tox / (2 * eps_ins))


def liu_dvt(L, l, V_DS, V_bi=0.9, phi_s=0.8):
    """Liu et al. (1993) short-channel threshold reduction, V (positive).

    phi_s is the surface potential at threshold (about 2 phi_F)."""
    return (2 * (V_bi - phi_s) + V_DS) * (np.exp(-L / (2 * l)) + 2 * np.exp(-L / l))
