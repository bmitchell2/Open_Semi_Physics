"""Post-implant anneal models: transient-enhanced diffusion (TED), trap-limited
release of implanted nitrogen toward a Si/SiO2 interface sink, and
end-of-range (EOR) band placement from a damage profile.

All models are 1-D, illustrative, and continuum. Units: depth in nm,
concentration in cm^-3, time in s, temperature in degrees C unless noted.

Physics summary
---------------
* Boron intrinsic diffusivity: D = 0.76 exp(-3.46 eV / kT) cm^2/s (Fair; also
  used on the USJ page of the knowledge base).
* TED: D_eff(t) = D* (C_I/C_I*) with an interstitial supersaturation that
  decays as the damage dissolves; modelled as S(t) = 1 + S0 exp(-t/tau).
  Integrated diffusion budget  D* t + D* S0 tau (1 - exp(-t/tau)).
  Optional clustering: material above c_cluster is held immobile, a crude
  stand-in for boron-interstitial clusters that pin the peak.
* Defect-dissolution time: tau(T) proportional to exp(Ea/kT), Ea ~ 3.8 eV for
  {311} dissolution (Stolk et al., JAP 81, 6031, 1997).
* Implanted nitrogen (after Adam et al., JAP 87, 2282, 2000): an immobile
  population converts to a fast mobile species at rate k; the mobile species is
  captured at the Si/SiO2 interface (x = 0) up to an optional capacity q_max
  (cm^-2), after which the interface reflects (Dokumaci et al., MRS 669, 2001).
"""
import numpy as np

K_B_EV = 8.617333262e-5  # eV/K


def boron_intrinsic_diffusivity(T_C, D0=0.76, Ea=3.46):
    """Intrinsic boron diffusivity in Si (cm^2/s)."""
    T = np.asarray(T_C, dtype=float) + 273.15
    return D0 * np.exp(-Ea / (K_B_EV * T))


def gaussian_profile(x_nm, dose_cm2, Rp_nm, dRp_nm):
    """Gaussian implant profile (cm^-3) for dose in cm^-2, lengths in nm."""
    x = np.asarray(x_nm, dtype=float)
    dRp_cm = dRp_nm * 1e-7
    return dose_cm2 / (np.sqrt(2 * np.pi) * dRp_cm) * np.exp(-0.5 * ((x - Rp_nm) / dRp_nm) ** 2)


def integrated_dose(x_nm, c_cm3):
    """Areal dose (cm^-2) of a profile on a uniform grid in nm."""
    return float(np.trapezoid(c_cm3, np.asarray(x_nm) * 1e-7))


def ted_budget(D_star, t, S0, tau):
    """Integrated D*t (cm^2) including a decaying supersaturation S0 exp(-t/tau)."""
    return D_star * t + D_star * S0 * tau * (1.0 - np.exp(-t / tau))


def relative_dissolution_time(T_C, Tref_C, Ea=3.8):
    """tau(T)/tau(Tref) for an Arrhenius dissolution process with activation Ea (eV)."""
    T = np.asarray(T_C, dtype=float) + 273.15
    Tr = Tref_C + 273.15
    return np.exp(Ea / K_B_EV * (1.0 / T - 1.0 / Tr))


def _implicit_step(c, D_face, dx, dt):
    """One backward-Euler step of dc/dt = d/dx(D dc/dx), zero-flux ends.
    D_face has length n-1 (diffusivity at cell faces). Returns new c."""
    n = c.size
    r = D_face * dt / dx ** 2
    lower = np.zeros(n); upper = np.zeros(n); diag = np.ones(n)
    diag[:-1] += r; diag[1:] += r
    upper[:-1] = -r; lower[1:] = -r
    # Thomas algorithm
    cp = np.zeros(n); dp = np.zeros(n)
    cp[0] = upper[0] / diag[0]; dp[0] = c[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / m if i < n - 1 else 0.0
        dp[i] = (c[i] - lower[i] * dp[i - 1]) / m
    out = np.zeros(n); out[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        out[i] = dp[i] - cp[i] * out[i + 1]
    return out


def diffuse_ted(x_nm, c0, T_C, t_total, S0=0.0, tau=1.0, c_cluster=None, nsteps=400):
    """Diffuse a profile under constant temperature with TED enhancement.

    D_eff(t) = D*(T) [1 + S0 exp(-t/tau)]; if c_cluster is given, the fraction of
    each cell above c_cluster is immobile (kept out of the diffusing pool).
    Zero-flux surface (inert, capped surface). Time steps are graded so the
    fast early transient is resolved. Returns the final total profile (cm^-3)."""
    x = np.asarray(x_nm, dtype=float)
    dx = (x[1] - x[0]) * 1e-7
    D_star = float(boron_intrinsic_diffusivity(T_C))
    c = np.array(c0, dtype=float)
    immobile = np.zeros_like(c)
    if c_cluster is not None:
        immobile = np.clip(c - c_cluster, 0, None)
        c = c - immobile
    edges = t_total * (np.geomspace(1e-4, 1.0, nsteps))
    edges = np.concatenate([[0.0], edges])
    for t0, t1 in zip(edges[:-1], edges[1:]):
        tm = 0.5 * (t0 + t1)
        D = D_star * (1.0 + S0 * np.exp(-tm / tau))
        c = _implicit_step(c, np.full(c.size - 1, D), dx, t1 - t0)
    return c + immobile


def nitrogen_trap_limited(x_nm, n0, k_release, D_mobile, times, q_max=None, nsub=200):
    """Trap-limited redistribution of implanted N toward an interface sink at x=0.

    Immobile N converts to mobile N at rate k_release (1/s); mobile N diffuses
    with D_mobile (cm^2/s) and is captured at x=0 (perfect sink) until the
    trapped areal density reaches q_max (cm^-2), after which the boundary is
    zero-flux. Back boundary is zero-flux. Returns (profiles_total, q_interface)
    evaluated at each requested time; profiles exclude the interface sheet."""
    x = np.asarray(x_nm, dtype=float)
    dx = (x[1] - x[0]) * 1e-7
    imm = np.array(n0, dtype=float)
    mob = np.zeros_like(imm)
    q = 0.0
    out_p, out_q = [], []
    t_prev = 0.0
    for t_target in times:
        span = t_target - t_prev
        if span > 0:
            dt = span / nsub
            for _ in range(nsub):
                rel = imm * (1.0 - np.exp(-k_release * dt))
                imm -= rel; mob += rel
                sink_open = q_max is None or q < q_max
                n = mob.size
                r = D_mobile * dt / dx ** 2
                diag = np.ones(n) + 2 * r; upper = np.full(n, -r); lower = np.full(n, -r)
                diag[-1] = 1 + r  # zero flux at back
                if sink_open:
                    pass  # Dirichlet c=0 ghost at x=-dx: diag[0] keeps 1+2r
                else:
                    diag[0] = 1 + r
                before = mob.sum()
                cp = np.zeros(n); dp = np.zeros(n)
                cp[0] = upper[0] / diag[0]; dp[0] = mob[0] / diag[0]
                for i in range(1, n):
                    m = diag[i] - lower[i] * cp[i - 1]
                    cp[i] = upper[i] / m if i < n - 1 else 0.0
                    dp[i] = (mob[i] - lower[i] * dp[i - 1]) / m
                new = np.zeros(n); new[-1] = dp[-1]
                for i in range(n - 2, -1, -1):
                    new[i] = dp[i] - cp[i] * new[i + 1]
                captured = (before - new.sum()) * dx
                if q_max is not None and q + captured > q_max:
                    excess = (q + captured - q_max) / dx
                    new[0] += excess
                    captured = q_max - q
                q += captured
                mob = new
        out_p.append(imm + mob)
        out_q.append(q)
        t_prev = t_target
    return out_p, np.array(out_q)


def eor_depth(x_nm, damage_cm3, threshold_cm3):
    """Depth (nm) of the amorphous/crystalline interface: deepest point where the
    displaced-atom density exceeds the amorphization threshold. Returns None if
    the profile never reaches threshold (non-amorphizing implant)."""
    x = np.asarray(x_nm, dtype=float)
    d = np.asarray(damage_cm3, dtype=float)
    idx = np.where(d >= threshold_cm3)[0]
    if idx.size == 0:
        return None
    i = idx[-1]
    if i == x.size - 1:
        return float(x[-1])
    # linear interpolation to the crossing
    x0, x1, d0, d1 = x[i], x[i + 1], d[i], d[i + 1]
    return float(x0 + (threshold_cm3 - d0) * (x1 - x0) / (d1 - d0))
