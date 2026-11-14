"""
Electron tunneling through a gate dielectric: WKB transmission for the
trapezoidal (direct-tunneling) and triangular (Fowler-Nordheim) barrier,
and the Fowler-Nordheim current density.

Backs the Semiconductor Notes page "Gate Dielectric Tunneling (Direct,
Fowler-Nordheim and Trap-Assisted)".

Models and assumptions
----------------------
- Rigid-band, uniform-field barrier. At the injecting interface the barrier
  is phi_B (the conduction-band offset, about 3.1-3.2 eV for electrons from
  Si into thermal SiO2). A voltage V_ox across thickness t_ox tilts the top
  linearly, so the far-side height is phi_B - V_ox (eV, with V_ox in volts).
- WKB exponent for an electron at the injecting band edge:
      ln T = -2 * integral sqrt(2 m_ox q (phi(x))) / hbar dx
  Trapezoid (V_ox < phi_B):
      ln T = -(4 sqrt(2 m_ox q) / (3 hbar)) * t_ox/V_ox * [phi_B^1.5 - (phi_B - V_ox)^1.5]
  Triangle (V_ox >= phi_B, Fowler-Nordheim): the electron tunnels only to the
  classical turning point x = phi_B / E, giving ln T = -B/E with
      B = 4 sqrt(2 m_ox q) phi_B^1.5 / (3 hbar)  (V/m).
  The two expressions join continuously at V_ox = phi_B.
- Fowler-Nordheim current density (Lenzlinger & Snow 1969):
      J = A E^2 exp(-B/E),  A = q^3 m0 / (8 pi h m_ox phi_B q)
  Image-force lowering, temperature and the 2-D inversion subband structure
  are neglected, so absolute FN current is good to a factor of a few;
  Lenzlinger & Snow measured about 5-10x below theory, attributed to trapping.
- Direct-tunneling transmission here is a single-energy (band-edge) WKB
  factor, not a current. A real direct-tunneling current integrates over the
  supply of electrons (inversion charge, subbands) and is best taken from a
  calibrated model (Lee & Hu 2001) or measured data. The robust result is the
  exponential thickness sensitivity: about one decade per 2 A of SiO2.

Defaults: m_ox = 0.42 m0 (fitted electron tunneling mass in SiO2, between
the ~0.4 of Lenzlinger & Snow and the 0.4-0.5 used in direct-tunneling fits),
phi_B = 3.15 eV.

Units: SI internally; public functions take t_ox in nm, field in MV/cm,
voltages in V, barrier heights in eV, and return J in A/cm^2.
"""
import numpy as np

q = 1.602176634e-19
hbar = 1.054571817e-34
h = 2 * np.pi * hbar
m0 = 9.1093837015e-31

PHI_B_SIO2_E = 3.15     # eV, electron barrier Si -> SiO2 (IPE, ~3.1-3.25)
PHI_B_SIO2_H = 4.6      # eV, hole barrier (valence-band offset, ~4.4-4.6)
M_OX_E = 0.42           # electron tunneling mass in SiO2 / m0


def _coef(m_rel):
    """4 sqrt(2 m q) / (3 hbar), units 1/(m * V^0.5)."""
    return 4 * np.sqrt(2 * m_rel * m0 * q) / (3 * hbar)


def wkb_ln_transmission(t_ox_nm, V_ox, phi_B=PHI_B_SIO2_E, m_rel=M_OX_E):
    """ln of WKB transmission at the injecting band edge.

    Handles both the trapezoid (V_ox < phi_B, direct tunneling) and the
    triangle (V_ox >= phi_B, Fowler-Nordheim). V_ox > 0 means the barrier
    tilts down toward the collecting side. V_ox -> 0 gives the rectangular
    barrier limit -2 kappa t_ox.
    """
    t = np.asarray(t_ox_nm, dtype=float) * 1e-9
    V = np.asarray(V_ox, dtype=float)
    C = _coef(m_rel)
    t, V = np.broadcast_arrays(t, V)
    out = np.empty(np.broadcast(t, V).shape)
    small = V < 1e-6
    tri = V >= phi_B
    trap = ~small & ~tri
    kappa = np.sqrt(2 * m_rel * m0 * q * phi_B) / hbar
    out[small] = -2 * kappa * t[small]
    out[trap] = -C * t[trap] / V[trap] * (phi_B ** 1.5 - (phi_B - V[trap]) ** 1.5)
    E = V[tri] / t[tri]            # V/m
    out[tri] = -C * phi_B ** 1.5 / E
    return out


def fn_B(phi_B=PHI_B_SIO2_E, m_rel=M_OX_E):
    """Fowler-Nordheim slope constant B in MV/cm."""
    return _coef(m_rel) * phi_B ** 1.5 / 1e8      # V/m -> MV/cm


def fn_A(phi_B=PHI_B_SIO2_E, m_rel=M_OX_E):
    """Fowler-Nordheim prefactor A in A/V^2 (used with E in V/m, giving A/m^2)."""
    return q ** 3 * m0 / (8 * np.pi * h * m_rel * m0 * phi_B * q)   # A/V^2


def fn_current_density(E_MVcm, phi_B=PHI_B_SIO2_E, m_rel=M_OX_E):
    """Fowler-Nordheim current density, A/cm^2, for oxide field E in MV/cm."""
    E = np.asarray(E_MVcm, dtype=float) * 1e8       # V/m
    J = fn_A(phi_B, m_rel) * E ** 2 * np.exp(-fn_B(phi_B, m_rel) * 1e8 / E)  # A/m^2
    return J * 1e-4


def decades_per_angstrom(V_ox=1.0, t_ox_nm=1.8, phi_B=PHI_B_SIO2_E, m_rel=M_OX_E, dt_nm=0.01):
    """Thickness sensitivity of direct-tunneling transmission at fixed V_ox (decades per A)."""
    a = wkb_ln_transmission(t_ox_nm, V_ox, phi_B, m_rel)
    b = wkb_ln_transmission(t_ox_nm + dt_nm, V_ox, phi_B, m_rel)
    return float((a - b) / np.log(10) / (dt_nm * 10))


def fn_onset_voltage(phi_B=PHI_B_SIO2_E):
    """Oxide voltage at which the trapezoid becomes a triangle (V)."""
    return phi_B
