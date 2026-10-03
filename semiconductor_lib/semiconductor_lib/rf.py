"""
RF device and circuit models: oscillator phase noise, transistor RF figures
of merit (fT, fmax), multi-finger gate resistance, and on-chip spiral-inductor
models (inductance, skin effect, single-pi one-port Q).

All functions use SI units unless a docstring says otherwise.

References
----------
- D. B. Leeson, "A simple model of feedback oscillator noise spectrum,"
  Proc. IEEE 54(2), 329-330 (1966).
- B. Razavi, RF Microelectronics, 2nd ed. (2012), ch. 8 (oscillators) and
  ch. 2 (fT/fmax).
- S. S. Mohan, M. Hershenson, S. P. Boyd, T. H. Lee, "Simple accurate
  expressions for planar spiral inductances," IEEE JSSC 34(10), 1419-1424
  (1999).
- C. P. Yue, S. S. Wong, "Physical modeling of spiral inductors on silicon,"
  IEEE Trans. Electron Devices 47(3), 560-568 (2000).
"""
import numpy as np

K_B = 1.380649e-23   # J/K
MU0 = 4e-7 * np.pi   # H/m
EPS0 = 8.8541878128e-12
EPS_SI = 11.7 * EPS0


# --------------------------------------------------------------------------
# Oscillators
# --------------------------------------------------------------------------
def tank_resonance(L, C):
    """Resonant frequency f0 = 1/(2 pi sqrt(LC)) [Hz]."""
    return 1.0 / (2 * np.pi * np.sqrt(L * C))


def tank_parallel_resistance(Q, f0, L):
    """Equivalent parallel loss resistance of an LC tank, Rp = Q*w0*L [ohm]."""
    return Q * 2 * np.pi * f0 * L


def cross_coupled_min_gm(Rp):
    """Minimum per-device transconductance for start-up of a cross-coupled
    pair across a differential tank of parallel resistance Rp: the pair
    presents -2/gm, so oscillation needs gm >= 2/Rp [S]."""
    return 2.0 / Rp


def leeson_phase_noise(df, f0, Q, P_sig, F=2.0, f_flicker=0.0, T=300.0):
    """Single-sideband phase noise L(df) [dBc/Hz] from Leeson's model:

        L = 10 log10[ (2 F k T / P_sig) (1 + (f0/(2 Q df))^2) (1 + f_c/|df|) ]

    df : offset frequency (Hz), f0 : carrier (Hz), Q : loaded tank Q,
    P_sig : signal power in the tank (W), F : empirical device noise factor,
    f_flicker : 1/f^3 corner frequency (Hz).
    Gives -30 dB/dec below f_flicker, -20 dB/dec above it out to the
    half-bandwidth f0/(2Q), and a flat floor beyond.  Semi-empirical: F and
    f_flicker are fitted, not predicted."""
    df = np.asarray(df, dtype=float)
    floor = 2 * F * K_B * T / P_sig
    shaping = 1 + (f0 / (2 * Q * df)) ** 2
    flick = 1 + f_flicker / np.abs(df)
    return 10 * np.log10(floor * shaping * flick)


def oscillator_fom(L_dbc, f0, df, P_dc):
    """Standard oscillator figure of merit [dBc/Hz] (more negative = better):
    FOM = L(df) - 20 log10(f0/df) + 10 log10(P_dc / 1 mW)."""
    return L_dbc - 20 * np.log10(f0 / df) + 10 * np.log10(P_dc / 1e-3)


def reference_noise_multiplication_db(N):
    """In-band PLL output noise rises by 20 log10(N) above the reference
    (or PFD-referred) phase noise, N = f_out / f_ref [dB]."""
    return 20 * np.log10(N)


# --------------------------------------------------------------------------
# Transistor RF figures of merit
# --------------------------------------------------------------------------
def ft_mosfet(gm, Cgs, Cgd):
    """Unity current-gain frequency fT = gm / (2 pi (Cgs + Cgd)) [Hz]."""
    return gm / (2 * np.pi * (Cgs + Cgd))


def gate_resistance(Rsh, W_finger, L_gate, n_fingers, double_contact=True,
                    R_contact_per_finger=0.0):
    """Effective small-signal gate resistance of a multi-finger MOSFET [ohm].

    Distributed RC along each finger reduces the end-to-end finger
    resistance Rsh*W/L by a factor 3 (contact at one end) or 12 (contacts at
    both ends).  Fingers are in parallel."""
    factor = 12.0 if double_contact else 3.0
    r_finger = Rsh * W_finger / (factor * L_gate) + R_contact_per_finger
    return r_finger / n_fingers


def fmax_mosfet(fT, Rg, gds, Cgd, Ri=0.0, Rs=0.0):
    """Maximum oscillation frequency (unilateral power gain = 1) [Hz]:

        fmax = (fT/2) / sqrt( gds (Rg + Ri + Rs) + 2 pi fT Rg Cgd )
    """
    return (fT / 2.0) / np.sqrt(gds * (Rg + Ri + Rs) + 2 * np.pi * fT * Rg * Cgd)


def fmax_hbt(fT, rb, Cbc):
    """Bipolar fmax = sqrt( fT / (8 pi rb Cbc) ) [Hz]."""
    return np.sqrt(fT / (8 * np.pi * rb * Cbc))


# --------------------------------------------------------------------------
# On-chip passives
# --------------------------------------------------------------------------
def skin_depth(rho, f, mu_r=1.0):
    """Skin depth delta = sqrt(rho / (pi f mu)) [m]."""
    return np.sqrt(rho / (np.pi * f * MU0 * mu_r))


def spiral_inductance_wheeler(n, d_out, d_in, shape="square"):
    """Modified Wheeler expression (Mohan et al. 1999) for a planar spiral
    inductance [H].  n turns, outer/inner diameters in metres.
    Quoted accuracy about 2-3 % against field solvers for typical layouts."""
    coeff = {"square": (2.34, 2.75), "hexagonal": (2.33, 3.82),
             "octagonal": (2.25, 3.55)}
    K1, K2 = coeff[shape]
    d_avg = 0.5 * (d_out + d_in)
    rho = (d_out - d_in) / (d_out + d_in)
    return K1 * MU0 * n ** 2 * d_avg / (1 + K2 * rho)


def series_resistance_skin(R_dc, t_metal, rho_metal, f):
    """Series resistance with skin effect for a strip of thickness t
    (Yue & Wong form): R = R_dc * t / (delta (1 - exp(-t/delta)))."""
    d = skin_depth(rho_metal, f)
    return R_dc * t_metal / (d * (1 - np.exp(-t_metal / d)))


def substrate_branch(rho_sub, R_geom_per_ohm_m, eps_sub=EPS_SI):
    """Return (R_sub, C_sub) for a substrate path whose resistance is
    R_sub = rho_sub * R_geom (R_geom in 1/m).  C_sub follows from the
    dielectric relaxation constraint R_sub * C_sub = rho_sub * eps_sub."""
    R_sub = rho_sub * R_geom_per_ohm_m
    C_sub = rho_sub * eps_sub / R_sub
    return R_sub, C_sub


def inductor_q_one_port(f, L, R_s, C_ox, R_sub, C_sub, C_p=0.0):
    """Q of a single-pi spiral-inductor model with port 2 grounded:
    series branch (R_s + jwL) || C_p, in parallel with the port-1 shunt
    branch C_ox in series with (R_sub || C_sub).  Q = Im(Z)/Re(Z).
    Does not include magnetically induced substrate eddy-current loss."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    Zs = R_s + 1j * w * L
    if C_p > 0:
        Zs = 1 / (1 / Zs + 1j * w * C_p)
    Zsub = 1 / (1 / R_sub + 1j * w * C_sub)
    Zsh = 1 / (1j * w * C_ox) + Zsub
    Z = 1 / (1 / Zs + 1 / Zsh)
    return Z.imag / Z.real


def pll_output_phase_noise_db(df, L_ref_db, L_vco_db, N, f_bw):
    """Idealized PLL output phase noise [dBc/Hz] with a first-order loop
    shape of bandwidth f_bw: the reference (PFD-referred) noise, raised by
    20 log10 N, is low-pass filtered; the free-running VCO noise is high-pass
    filtered.  Real type-II loops add peaking near f_bw; this omits it."""
    df = np.asarray(df, dtype=float)
    x2 = (df / f_bw) ** 2
    ref = 10 ** ((L_ref_db + 20 * np.log10(N)) / 10) / (1 + x2)
    vco = 10 ** (L_vco_db / 10) * x2 / (1 + x2)
    return 10 * np.log10(ref + vco)
