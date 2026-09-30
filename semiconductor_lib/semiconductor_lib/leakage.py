"""
Field-driven junction leakage: band-to-band tunnelling (BTBT) at an abrupt
junction and the classical gate-induced drain leakage (GIDL) surface field.

Backs the Semiconductor Notes pages "Gate-Induced Drain Leakage (GIDL)",
"Pocket / Halo Implantation" and "PN Junction Diode Current".

Models and assumptions
----------------------
- Abrupt junction, depletion approximation. Peak field
  E_max = sqrt(2 q N_eff (V_bi + V_R) / eps_si), with
  N_eff = N_A N_D / (N_A + N_D) (reduced doping). For a one-sided junction
  N_eff is the lighter doping (Taur & Ning Sec. 2.3; Yuan et al.,
  IEEE TDMR 8, 501, 2008, Eq. 2).
- BTBT current density: uniform-field direct-tunnelling (Kane) form used by
  Taur & Ning and by Yuan et al. 2008, Eq. (1):
      J = sqrt(2 m*/Eg) q^3 E V_R / (4 pi^3 hbar^2) * exp(-B/E),
      B = 4 sqrt(2 m*) Eg^(3/2) / (3 q hbar).
  With m* = 0.2 m0 and Eg = 1.12 eV, B = 3.6e7 V/cm. Silicon is indirect and
  real BTBT is phonon-assisted; calibrated TCAD (Hurkx-type) models use
  smaller effective B (about 1.9e7 to 2.3e7 V/cm). The absolute current is
  therefore uncertain by orders of magnitude; the exponential sensitivity to
  field (and so to doping) is the robust result. Use `B_override` to test it.
- Classical GIDL (gate-drain overlap, Chan et al., IEDM 1987):
  silicon surface field E_s ~ (V_DG - 1.2 V)/(3 t_ox), where 1.2 V is the
  empirical band bending (~Eg/q plus a small offset) needed before BTBT can
  start and 3 is eps_si/eps_ox. Oxide field is 3 E_s.

Units: cm, V, cm^-3, A/cm^2.
"""
import numpy as np

from .constants import q, eps_si, eps_ox, ni300, thermal_voltage

hbar = 1.054571817e-34      # J s
m0 = 9.1093837015e-31       # kg


def n_eff(NA, ND):
    """Reduced doping N_A N_D/(N_A+N_D), cm^-3. Dominated by the lighter side."""
    NA = np.asarray(NA, dtype=float)
    ND = np.asarray(ND, dtype=float)
    return NA * ND / (NA + ND)


def built_in_potential(NA, ND, T=300.0, ni=ni300):
    """Non-degenerate built-in potential, V (Boltzmann; approximate above ~1e19)."""
    return thermal_voltage(T) * np.log(np.asarray(NA) * np.asarray(ND) / ni**2)


def peak_field(NA, ND, VR, Vbi=None, T=300.0):
    """Peak field at an abrupt junction under reverse bias VR (>=0), V/cm.
    If Vbi is None it is computed from the doping."""
    if Vbi is None:
        Vbi = built_in_potential(NA, ND, T)
    return np.sqrt(2.0 * q * n_eff(NA, ND) * (Vbi + VR) / eps_si)


def btbt_B(m_eff=0.2, Eg=1.12):
    """Exponential field constant B of the Kane form, V/cm."""
    ms = m_eff * m0
    Egj = Eg * q
    return 4.0 * np.sqrt(2.0 * ms) * Egj**1.5 / (3.0 * q * hbar) / 100.0


def btbt_current_density(E, VR, m_eff=0.2, Eg=1.12, B_override=None):
    """Uniform-field BTBT current density, A/cm^2 (see module docstring).
    E in V/cm, VR in V. B_override (V/cm) replaces only the exponential
    constant, to test sensitivity to the tunnelling model."""
    E = np.asarray(E, dtype=float)
    ms = m_eff * m0
    Egj = Eg * q
    E_si = E * 100.0
    pre = np.sqrt(2.0 * ms / Egj) * q**3 * E_si * VR / (4.0 * np.pi**3 * hbar**2)
    B = btbt_B(m_eff, Eg) if B_override is None else B_override
    return pre * np.exp(-B / E) / 1.0e4


def gidl_surface_field(V_DG, t_ox, offset=1.2, eps_ratio=None):
    """Classical overlap-GIDL silicon surface field, V/cm (Chan et al. 1987).
    V_DG drain-to-gate voltage (V), t_ox oxide thickness (cm)."""
    if eps_ratio is None:
        eps_ratio = eps_si / eps_ox
    return np.maximum(np.asarray(V_DG, dtype=float) - offset, 0.0) / (eps_ratio * t_ox)
