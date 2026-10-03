"""
Energy-band diagrams along a device, for teaching figures and quick checks.

Two models:

1. ``pn_band_diagram`` - abrupt p-n junction under bias in the depletion
   approximation, with quasi-Fermi levels. Band-edge offsets in the neutral
   regions use the Joyce-Dixon approximation to the inverse Fermi-Dirac
   integral, so degenerate (N > Nc or Nv) sides are handled; band-gap
   narrowing is ignored. The built-in potential is taken from band alignment,
   V_bi = E_g - (E_c - E_F)_n - (E_F - E_v)_p, which reduces to
   (kT/q) ln(N_A N_D / n_i^2) for non-degenerate doping.
   Convention: p-side at x < 0, n-side at x > 0, the n-side neutral region is
   the energy reference (its Fermi level is 0 eV) and V_A is applied to the
   p-side (V_A > 0 is forward bias). Energies are in eV.

2. ``lateral_surface_potential`` - quasi-two-dimensional (Young 1989;
   Liu et al. 1993) surface potential along the channel of a bulk n-MOSFET
   below threshold, for a laterally non-uniform body doping (pockets):

       d^2 phi_s/dy^2 = (phi_s - phi_L(y)) / lambda(y)^2

   phi_L(y) is the local long-channel surface potential set by the gate,
   lambda = sqrt(eps_si t_ox W_dep / eps_ox) the characteristic length, and
   the boundary conditions are the n+ source and drain potentials. Solved by
   finite differences; for uniform doping it reproduces the closed-form sinh
   solution (tested). Potentials are referenced to the intrinsic level with
   the source/body Fermi level at 0 V, so the electron energy of the
   conduction band along the surface is E_c(y) = E_g/2 - q phi_s(y).
   Valid in depletion/weak inversion only (inversion charge is not included)
   and for the surface path only (sub-surface punch-through is not modelled).
"""
import numpy as np
from scipy.optimize import brentq

from .constants import q, eps_si, eps_ox, ni300, thermal_voltage

EG_SI = 1.12      # eV, 300 K
NC_SI = 2.8e19    # cm^-3, 300 K (Sze & Ng)
NV_SI = 1.04e19   # cm^-3, 300 K (Sze & Ng)


# --------------------------------------------------------------------------
# Fermi-level position in a neutral region
# --------------------------------------------------------------------------
def joyce_dixon_eta(N, Nc):
    """Reduced Fermi energy eta = (E_F - E_c)/kT for electron density N
    (Joyce & Dixon, Appl. Phys. Lett. 31, 354, 1977). Accurate to <1e-3 for
    N/Nc up to ~8. Reduces to ln(N/Nc) for N << Nc."""
    r = np.asarray(N, dtype=float) / Nc
    a1, a2, a3, a4 = 3.53553e-1, -4.95009e-3, 1.48386e-4, -4.42563e-6
    return np.log(r) + a1 * r + a2 * r**2 + a3 * r**3 + a4 * r**4


def band_offsets(N, carrier, T=300.0, Nc=NC_SI, Nv=NV_SI):
    """Distance (eV) from the Fermi level to the majority band edge in a
    neutral region: E_c - E_F for 'n', E_F - E_v for 'p'. Negative means
    degenerate (E_F inside the band)."""
    Vt = thermal_voltage(T)
    Neff = Nc if carrier == 'n' else Nv
    return -Vt * joyce_dixon_eta(N, Neff)


def built_in_potential_fd(NA, ND, T=300.0, Eg=EG_SI):
    """Built-in potential from band alignment with Joyce-Dixon offsets."""
    return Eg - band_offsets(ND, 'n', T) - band_offsets(NA, 'p', T)


# --------------------------------------------------------------------------
# p-n junction band diagram under bias (depletion approximation)
# --------------------------------------------------------------------------
def pn_band_diagram(NA, ND, VA=0.0, x=None, T=300.0, Eg=EG_SI, eps=eps_si):
    """Band diagram of an abrupt junction. Returns dict with x (cm), Ec, Ev,
    EFn, EFp (eV, NaN outside the region where each is meaningful), plus
    Vbi, xn, xp, W (cm).

    EFn is drawn flat across the n-side and the depletion region, EFp flat
    across the p-side and the depletion region (the standard low-injection
    picture); beyond the depletion edges the minority quasi-Fermi level
    rejoins the majority one over a diffusion length, which is not modelled.
    """
    Vbi = built_in_potential_fd(NA, ND, T, Eg)
    if VA >= Vbi:
        raise ValueError("depletion approximation invalid for V_A >= V_bi")
    W = np.sqrt(2 * eps * (Vbi - VA) / q * (1 / NA + 1 / ND))
    xn = W * NA / (NA + ND)
    xp = W * ND / (NA + ND)
    if x is None:
        span = 1.6 * max(xn, xp)
        x = np.linspace(-span, span, 801)
    x = np.asarray(x, dtype=float)

    # Electrostatic potential psi (V), 0 in the n-side bulk,
    # -(Vbi - VA) in the p-side bulk; parabolic in each depleted part.
    psi = np.empty_like(x)
    Vtot = Vbi - VA
    psi[x >= xn] = 0.0
    psi[x <= -xp] = -Vtot
    m_n = (x > 0) & (x < xn)
    m_p = (x <= 0) & (x > -xp)
    psi[m_n] = -q * ND / (2 * eps) * (xn - x[m_n])**2
    psi[m_p] = -Vtot + q * NA / (2 * eps) * (x[m_p] + xp)**2

    Ec_n = band_offsets(ND, 'n', T)          # Ec - EF in n bulk
    Ec = Ec_n - psi                          # electron energy = -q psi
    Ev = Ec - Eg
    EFn = np.where(x >= -xp, 0.0, np.nan)
    EFp = np.where(x <= xn, -VA, np.nan)
    return dict(x=x, Ec=Ec, Ev=Ev, EFn=EFn, EFp=EFp, psi=psi,
                Vbi=Vbi, W=W, xn=xn, xp=xp)


# --------------------------------------------------------------------------
# Quasi-2-D lateral surface potential of a bulk n-MOSFET
# --------------------------------------------------------------------------
def phi_bulk(N, T=300.0, ni=ni300):
    """Potential of neutral p-type silicon relative to the intrinsic level
    (Fermi level at 0): -phi_F = -(kT/q) ln(N/ni)."""
    return -thermal_voltage(T) * np.log(N / ni)


def long_channel_phi_s(VG, N, tox, phi_gate0=EG_SI / 2, T=300.0):
    """Long-channel surface potential (relative to the intrinsic level, Fermi
    level 0) in depletion for a p-body of doping N under gate voltage VG,
    depletion approximation. phi_gate0 is the gate work-function potential in
    the same reference (+Eg/2 for n+ poly). Clipped at 2 phi_F of band
    bending plus 10 kT/q so that a threshold criterion slightly beyond
    2 phi_F can still be located; inversion charge is not included, so the
    result is only meaningful up to about threshold."""
    Cox = eps_ox / tox
    gamma = np.sqrt(2 * q * eps_si * N) / Cox
    phib = phi_bulk(N, T)
    drive = VG + phi_gate0 - phib              # V_G - V_FB
    drive = np.maximum(drive, 0.0)
    sq = (-gamma + np.sqrt(gamma**2 + 4 * drive)) / 2
    psi_s = np.minimum(sq**2, -2 * phib + 10 * thermal_voltage(T))
    return phib + psi_s


def char_length(N, tox, T=300.0):
    """lambda = sqrt(eps_si t_ox W_dm / eps_ox), W_dm at 2 phi_F (cm)."""
    phiF = -phi_bulk(N, T)
    Wdm = np.sqrt(2 * eps_si * 2 * phiF / (q * N))
    return np.sqrt(eps_si * tox * Wdm / eps_ox)


def lateral_surface_potential(L, VG, VDS, N_of_y, tox, N_sd=1e20, npts=801,
                              T=300.0):
    """Solve the quasi-2-D equation for phi_s(y), 0 <= y <= L (cm).

    N_of_y : callable giving the local body doping (cm^-3) at position y (cm)
    Returns dict with y, phi_s, phi_L, lambda_, Ec (eV, = Eg/2 - phi_s) and
    the source-referenced barrier height (eV) = max(Ec) - Ec(source end).
    """
    y = np.linspace(0.0, L, npts)
    N = np.asarray(N_of_y(y), dtype=float) * np.ones_like(y)
    phiL = long_channel_phi_s(VG, N, tox, T=T)
    lam = char_length(N, tox, T)
    phi0 = thermal_voltage(T) * np.log(N_sd / ni300)
    h = y[1] - y[0]
    n = npts - 2
    main = -2.0 / h**2 - 1.0 / lam[1:-1]**2
    off = np.ones(n - 1) / h**2
    A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    b = -phiL[1:-1] / lam[1:-1]**2
    b[0] -= phi0 / h**2
    b[-1] -= (phi0 + VDS) / h**2
    phi = np.empty(npts)
    phi[0], phi[-1] = phi0, phi0 + VDS
    phi[1:-1] = np.linalg.solve(A, b)
    Ec = EG_SI / 2 - phi
    return dict(y=y, phi_s=phi, phi_L=phiL, lambda_=lam, Ec=Ec,
                barrier=Ec.max() - Ec[0])


def lateral_phi_uniform_analytic(y, L, phiL, lam, phi0, VDS):
    """Closed-form quasi-2-D solution for uniform doping (Liu et al. 1993)."""
    s = np.sinh(L / lam)
    return (phiL + (phi0 - phiL) * np.sinh((L - y) / lam) / s
            + (phi0 + VDS - phiL) * np.sinh(y / lam) / s)


def threshold_voltage_q2d(L, VDS, N_of_y, tox, n_crit, T=300.0, **kw):
    """V_T by a constant surface-electron-density criterion: the gate voltage
    at which the minimum surface electron density along the channel,
    n_i exp(phi_s,min / V_t), reaches n_crit (cm^-3)."""
    phi_crit = thermal_voltage(T) * np.log(n_crit / ni300)

    def f(VG):
        s = lateral_surface_potential(L, VG, VDS, N_of_y, tox, T=T, **kw)
        return s['phi_s'].min() - phi_crit
    return brentq(f, -2.0, 3.0, xtol=1e-5)


def gaussian_pocket_profile(L, N_ch, N_pk, sigma):
    """Body doping with Gaussian pockets of peak N_pk (added to N_ch) centred
    at the source (y=0) and drain (y=L) edges, lateral sigma (cm)."""
    def N(y):
        y = np.asarray(y, dtype=float)
        return (N_ch + N_pk * np.exp(-y**2 / (2 * sigma**2))
                + N_pk * np.exp(-(L - y)**2 / (2 * sigma**2)))
    return N


def threshold_voltage_q2d_current(L, VDS, N_of_y, tox, n_crit, T=300.0, **kw):
    """V_T by a constant-current (length-weighted) criterion.

    At low V_DS the subthreshold channel acts as resistances in series, so the
    drain current per square is proportional to the harmonic-mean surface
    electron density  n_eff = L / integral(dy / n(y)),  n = n_i exp(phi_s/V_t).
    V_T is the gate voltage at which n_eff = n_crit. Unlike the minimum-density
    criterion of threshold_voltage_q2d, this weights each region by its length,
    so short high-barrier pockets matter in proportion to how much of the
    channel they occupy. This mirrors a measured constant-current V_T
    (I_D = I_crit * W/L) and gives the gradual reverse short-channel rise of a
    pocketed device; as L -> infinity it returns the uniform channel-centre
    V_T. Valid at low V_DS only (series-resistance picture)."""
    Vt = thermal_voltage(T)
    ln_crit = np.log(n_crit / ni300)

    def f(VG):
        s = lateral_surface_potential(L, VG, VDS, N_of_y, tox, T=T, **kw)
        x = -s['phi_s'] / Vt
        m = x.max()
        e = np.exp(x - m)
        integ = np.sum(0.5 * (e[1:] + e[:-1]) * np.diff(s['y']))  # trapezoid
        # ln(n_eff / n_i) = ln L - ln integral(exp(-phi/Vt) dy)
        return (np.log(L) - m - np.log(integ)) - ln_crit
    return brentq(f, -2.0, 3.0, xtol=1e-5)


def pocket_lpe0(N_ch, N_pk, sigma):
    """Lateral-average-doping length LPE0 (cm) of two Gaussian pockets
    (gaussian_pocket_profile): the channel-average doping is
    N_ch * (1 + LPE0 / L) with LPE0 = N_pk * sigma * sqrt(2 pi) / N_ch, i.e.
    the integrated lateral pocket dose of both pockets divided by N_ch.
    This is the physical meaning of the BSIM4 LPE0 parameter; the fitted
    BSIM4 value is usually smaller because the averaging picture overstates
    the pocket's effect (see threshold_voltage_q2d_current)."""
    return N_pk * sigma * np.sqrt(2 * np.pi) / N_ch
