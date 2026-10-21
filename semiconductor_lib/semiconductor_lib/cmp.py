"""
Chemical-mechanical polishing (CMP): removal-rate laws, carrier/platen
kinematics, pad bending length, and a one-dimensional up/down-area
step-height model for pattern-density-dependent planarization.

Models (all illustrative, engineering-level):

* Preston law  R = Kp * P * v  (empirical, Preston 1927).
* Threshold ("hockey-stick") law  R = Kc * v * max(P - P_th, 0), a
  phenomenological description of additive-passivated ceria slurries
  whose rate is near zero below a critical local pressure.
* Relative pad-wafer velocity for a rotating carrier (angular speed
  w_c, centre at distance d from the platen axis) on a rotating platen
  (w_p): v_rel = w_p x D + (w_p - w_c) x r.  Equal speeds give a
  spatially uniform |v_rel| = w_p d.
* Bending (decay) length of a stiff top pad on a compliant subpad,
  modelled as a thin plate on a Winkler foundation:
  L = (4 D / k)^(1/4),  D = E h^3 / (12 (1 - nu^2)),  k = E_sub / h_sub.
* Up/down step-height model with a contact height h_c:  above h_c only
  the up areas (pattern density rho) carry load, P_up = P/rho; below h_c
  the pad also touches the down areas and the pressure difference
  scales linearly with step height,
      P_up - P_dn = P h / (rho h_c),   rho P_up + (1 - rho) P_dn = P.
  Up and down surfaces are integrated in time with any rate law.
  Compare Stine et al., IEEE TSM 11, 129 (1998) and Ouma et al.,
  IEEE TSM 15, 232 (2002) for the effective-density formulation.

Units: pressure in Pa, velocity in m/s, rates in nm/s, lengths in nm
unless stated otherwise.
"""
import numpy as np

PSI = 6894.757  # Pa per psi


# ------------------------------------------------------------ rate laws
def preston_rate(P, v, Kp):
    """Preston removal rate R = Kp*P*v.  Kp in nm/(Pa*m) gives nm/s."""
    return Kp * np.asarray(P, float) * v


def threshold_rate(P, v, Kc, P_th):
    """Threshold (non-Prestonian) rate R = Kc*v*max(P - P_th, 0)."""
    P = np.asarray(P, float)
    return Kc * v * np.clip(P - P_th, 0.0, None)


# ------------------------------------------------------------ kinematics
def relative_velocity(r, phi, w_p, w_c, d):
    """Magnitude of pad-wafer relative velocity (m/s) at wafer radius r
    (m) and angle phi (rad), platen speed w_p and carrier speed w_c
    (rad/s, same sense), carrier centre at distance d (m) from the
    platen axis."""
    r = np.asarray(r, float)
    # platen velocity of the wafer point minus the wafer point's velocity
    # = w_p z x (D + r) - w_c z x r = w_p z x D + (w_p - w_c) z x r
    Dx, Dy = d, 0.0
    rx, ry = r * np.cos(phi), r * np.sin(phi)
    vx = -w_p * Dy - (w_p - w_c) * ry
    vy = w_p * Dx + (w_p - w_c) * rx
    return np.hypot(vx, vy)


def mean_relative_velocity(r, w_p, w_c, d, n_phi=720):
    """Relative speed at wafer radius r averaged over one wafer turn."""
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    r = np.atleast_1d(np.asarray(r, float))
    out = np.array([relative_velocity(ri, phi, w_p, w_c, d).mean()
                    for ri in r])
    return out if out.size > 1 else out[0]


def rpm_to_rad_s(rpm):
    return np.asarray(rpm, float) * 2 * np.pi / 60.0


# ------------------------------------------------------------ pad mechanics
def pad_bending_length(E_top, h_top, E_sub, h_sub, nu=0.4):
    """Decay length (m) of a deflection in a stiff top pad (modulus
    E_top, thickness h_top) on a Winkler subpad (E_sub, h_sub).
    Moduli in Pa, thicknesses in m."""
    D = E_top * h_top ** 3 / (12.0 * (1.0 - nu ** 2))
    k = E_sub / h_sub
    return (4.0 * D / k) ** 0.25


# ------------------------------------------------------------ step height
def contact_pressures(h, P, rho, h_c):
    """Up- and down-area pressures for step height h (nm), nominal
    pressure P, up-area density rho (0<rho<=1) and contact height h_c."""
    h = max(h, 0.0)
    if rho >= 1.0 or h <= 0.0:
        return P, P
    if h >= h_c:
        return P / rho, 0.0
    P_dn = P * (1.0 - h / h_c)
    P_up = P_dn + P * h / (rho * h_c)
    return P_up, P_dn


def step_height_evolution(rate_fn, P, rho, h0, h_c, t_end, dt=0.05):
    """Integrate up- and down-area removal for a single density region.

    rate_fn(P_local) -> removal rate (nm/s).  Returns arrays
    (t, removed_up, removed_down, step) in s and nm.  Down-area
    removal begins only when the pad touches it (P_dn > 0)."""
    n = int(np.ceil(t_end / dt)) + 1
    t = np.linspace(0, t_end, n)
    up = np.zeros(n)
    dn = np.zeros(n)
    for i in range(1, n):
        step = h0 - (up[i - 1] - dn[i - 1])
        P_up, P_dn = contact_pressures(step, P, rho, h_c)
        up[i] = up[i - 1] + float(rate_fn(P_up)) * dt
        dn[i] = dn[i - 1] + float(rate_fn(P_dn)) * dt
        # once planar, both surfaces move together
        if h0 - (up[i] - dn[i]) < 0:
            dn[i] = up[i] - h0
    step = h0 - (up - dn)
    return t, up, dn, np.clip(step, 0, None)


def threshold_stall_step(P, P_th, rho, h_c):
    """Residual step (nm) at which a threshold slurry stops planarizing:
    the up-area pressure P[1 + h(1/rho - 1)/h_c] falls to P_th.
    Zero if P_th <= P (the blanket surface still polishes)."""
    if P_th <= P:
        return 0.0
    return h_c * (P_th / P - 1.0) * rho / (1.0 - rho)
