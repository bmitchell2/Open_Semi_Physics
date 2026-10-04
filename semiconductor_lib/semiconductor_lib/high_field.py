"""
High-field carrier transport and inversion-layer mobility for MOSFETs.
Backs the "Velocity Saturation" and "Inversion-Layer Mobility and the
Universal Mobility Curve" notes.

Sources
-------
* Bulk-silicon velocity-field curve: Canali et al., IEEE Trans. Electron
  Devices 22, 1045 (1975), eq. 2a, with the temperature exponents of
  Jacoboni et al., Solid-State Electron. 20, 77 (1977), Table 5:
      v(E) = v_m (E/E_c) / [1 + (E/E_c)^beta]^(1/beta)
  electrons: v_m = 1.43e9 T^-0.87 cm/s, E_c = 1.01 T^1.55 V/cm,
             beta = 2.57e-2 T^0.66
  holes:     v_m = 1.62e8 T^-0.52 cm/s, E_c = 1.24 T^1.68 V/cm,
             beta = 0.46 T^0.17
  (parameter set as documented by Allpix Squared, retrieved 2026-10).
* Inversion-layer velocity saturation and the MOSFET current model:
  J. del Alamo, MIT 6.720J Lecture 30 (2007). Inversion-layer v_sat is
  8e6 cm/s (electrons) and 6e6 cm/s (holes). With the n = 1 form
  v = mu E / (1 + E/E_sat), E_sat = v_sat/mu,
      I_D = (W mu C_ox / L) (V_ov - V_DS/2) V_DS / (1 + V_DS/(E_sat L))
      V_Dsat = E_sat L [sqrt(1 + 2 V_ov/(E_sat L)) - 1]
      I_Dsat = W v_sat C_ox (V_ov - V_Dsat)
* Universal mobility fit (same source): mu_eff = mu0/[1 + (E_eff/E0)^nu],
  electrons mu0 = 670 cm^2/Vs, E0 = 0.67 MV/cm, nu = 1.6;
  holes mu0 = 160, E0 = 0.7 MV/cm, nu = 1.
* Effective field: E_eff = (Q_dep + eta Q_inv)/eps_si with eta = 1/2 for
  electrons on (100) and 1/3 for holes (Takagi et al., IEEE TED 41, 2357
  and 2363 (1994)).

Units: cm, V, s, F/cm^2 throughout.
"""
import numpy as np

from .constants import eps_si, eps_ox, q

# ---------------------------------------------------------------------------
# Bulk velocity-field curve (Canali / Jacoboni)
# ---------------------------------------------------------------------------
_CANALI = {
    "electron": dict(vm=(1.43e9, -0.87), Ec=(1.01, 1.55), beta=(2.57e-2, 0.66)),
    "hole": dict(vm=(1.62e8, -0.52), Ec=(1.24, 1.68), beta=(0.46, 0.17)),
}


def canali_params(carrier="electron", T=300.0):
    """Return (v_m [cm/s], E_c [V/cm], beta) for bulk Si at temperature T."""
    p = _CANALI[carrier]
    vm = p["vm"][0] * T ** p["vm"][1]
    Ec = p["Ec"][0] * T ** p["Ec"][1]
    beta = p["beta"][0] * T ** p["beta"][1]
    return vm, Ec, beta


def canali_velocity(E, carrier="electron", T=300.0):
    """Bulk drift velocity (cm/s) versus field E (V/cm), Canali model."""
    vm, Ec, beta = canali_params(carrier, T)
    x = np.abs(np.asarray(E, dtype=float)) / Ec
    return vm * x / (1.0 + x ** beta) ** (1.0 / beta)


# ---------------------------------------------------------------------------
# Inversion-layer velocity saturation and drain current
# ---------------------------------------------------------------------------
VSAT_INV = {"electron": 8.0e6, "hole": 6.0e6}   # cm/s, del Alamo 6.720


def v_field_simple(E, mu, vsat):
    """n = 1 velocity-field model v = mu E / (1 + E/E_sat), E_sat = vsat/mu.
    Saturates to vsat as E -> infinity; used for the analytic I-V model."""
    E = np.asarray(E, dtype=float)
    return mu * E / (1.0 + E / (vsat / mu))


def e_sat(mu, vsat):
    """Critical field E_sat = v_sat / mu (V/cm)."""
    return vsat / mu


def vdsat_vsat(V_ov, mu, vsat, L):
    """Saturation drain voltage with velocity saturation (V).
    Tends to V_ov for E_sat L >> V_ov and to sqrt(2 E_sat L V_ov) for
    E_sat L << V_ov."""
    a = e_sat(mu, vsat) * L
    return a * (np.sqrt(1.0 + 2.0 * V_ov / a) - 1.0)


def id_vsat(V_GS, V_DS, V_T, mu, vsat, Cox, W, L):
    """Drain current (A) with velocity saturation (n = 1 model, m = 1).
    V_DS is clipped at V_Dsat, beyond which I_D = W vsat Cox (V_ov - V_Dsat).
    Channel-length modulation and series resistance are not included."""
    V_ov = np.maximum(np.asarray(V_GS, dtype=float) - V_T, 0.0)
    a = e_sat(mu, vsat) * L
    vdsat = np.where(V_ov > 0, a * (np.sqrt(1.0 + 2.0 * V_ov / a) - 1.0), 0.0)
    vds = np.minimum(np.asarray(V_DS, dtype=float), vdsat)
    return W * mu * Cox / L * (V_ov - vds / 2.0) * vds / (1.0 + vds / a)


def idsat_vsat(V_ov, mu, vsat, Cox, W, L):
    """Saturation current (A) with velocity saturation."""
    return W * vsat * Cox * (V_ov - vdsat_vsat(V_ov, mu, vsat, L))


def idsat_long_channel(V_ov, mu, Cox, W, L):
    """Square-law saturation current (A), m = 1."""
    return W * mu * Cox * V_ov ** 2 / (2.0 * L)


# ---------------------------------------------------------------------------
# Inversion-layer (effective) mobility
# ---------------------------------------------------------------------------
UNIVERSAL = {
    "electron": dict(mu0=670.0, E0=0.67e6, nu=1.6),   # cm^2/Vs, V/cm
    "hole": dict(mu0=160.0, E0=0.70e6, nu=1.0),
}
ETA = {"electron": 0.5, "hole": 1.0 / 3.0}   # Takagi 1994, (100) surface


def effective_field(Q_dep, Q_inv, carrier="electron"):
    """E_eff (V/cm) from depletion and inversion charge magnitudes (C/cm^2)."""
    return (np.abs(Q_dep) + ETA[carrier] * np.abs(Q_inv)) / eps_si


def effective_field_nmos(V_GS, V_T, tox):
    """Bias-based E_eff estimate for an n+ poly nMOSFET (del Alamo 6.720):
    E_eff ~ (eps_ox/eps_si) (V_GS + V_T) / (2 t_ox). tox in cm."""
    return (eps_ox / eps_si) * (V_GS + V_T) / (2.0 * tox)


def universal_mobility(E_eff, carrier="electron"):
    """Empirical universal-curve mobility (cm^2/Vs) versus E_eff (V/cm).
    Captures phonon and surface-roughness limited behavior; does not
    include Coulomb scattering, so it overestimates mobility at low E_eff
    in heavily doped channels."""
    p = UNIVERSAL[carrier]
    return p["mu0"] / (1.0 + (np.asarray(E_eff, dtype=float) / p["E0"]) ** p["nu"])


def matthiessen_inversion(E_eff, mu_ph_coef, mu_sr_coef, mu_c=None,
                          ph_exp=-0.3, sr_exp=-2.0):
    """Illustrative Matthiessen decomposition of inversion-layer mobility.
    mu_ph = mu_ph_coef (E/1MV/cm)^ph_exp, mu_sr = mu_sr_coef (E/1MV/cm)^sr_exp
    (exponents from Takagi 1994 for electrons on (100)); mu_c optional
    Coulomb term (array or scalar). Returns (mu_total, mu_ph, mu_sr)."""
    x = np.asarray(E_eff, dtype=float) / 1e6
    mu_ph = mu_ph_coef * x ** ph_exp
    mu_sr = mu_sr_coef * x ** sr_exp
    inv = 1.0 / mu_ph + 1.0 / mu_sr
    if mu_c is not None:
        inv = inv + 1.0 / np.asarray(mu_c, dtype=float)
    return 1.0 / inv, mu_ph, mu_sr
