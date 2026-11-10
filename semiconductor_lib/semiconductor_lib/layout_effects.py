"""Layout-dependent effects (LDE) in planar bulk MOSFETs.

Two instance-geometry models used in BSIM4 and other compact models:

* STI stress / length of diffusion (LOD), BSIM4.3 onward
  (Su et al., CICC 2003). The geometric term is
      Inv_sa = (1/NF) * sum_i 1 / (SA + 0.5*L + i*(SD + L)),  i = 0..NF-1
  and likewise Inv_sb. Mobility scales as (1 + rho)/(1 + rho_ref) with
  rho = KU0/Kstress_u0 * (Inv_sa + Inv_sb); VTH0 shifts by
  KVTH0/Kstress_vth0 * (Inv_sa + Inv_sb - Inv_saref - Inv_sbref).

* Well proximity effect (WPE), BSIM4.5 onward (CMC model; Hook et al.,
  IEEE TED 50, 1946, 2003). Scattered-dopant basis functions are averaged
  over the gate area to give SCA, SCB, SCC; then
      VTH0 += KVTH0WE * (SCA + WEB*SCB + WEC*SCC).
  The closed forms here are for one straight well edge at distance SC from
  the device, with the device extending a distance d away from the edge
  (the integrals used in layout-extraction rule decks).

Lengths may be in any consistent unit (the defaults assume micrometres).
Process coefficients (KU0, KVTH0, KVTH0WE, WEB, WEC) are technology fit
parameters; values used in examples are illustrative only.
"""
import numpy as np


# ---------------------------------------------------------------- LOD
def inv_sa(sa, l, nf=1, sd=0.0):
    """BSIM4 Inv_sa (1/length): finger-averaged 1/(SA_i + L/2).

    sa: gate edge to STI edge on the A side for the first finger; l: drawn
    gate length; nf: number of fingers; sd: diffusion length between fingers.
    Inv_sb is the same function with SB (the sum is symmetric).
    """
    i = np.arange(int(nf))
    sa = np.asarray(sa, dtype=float)[..., None]
    return np.mean(1.0 / (sa + 0.5 * l + i * (sd + l)), axis=-1)


def finger_stress_terms(sa, sb, l, nf, sd):
    """Per-finger (1/(SA_i+L/2) + 1/(SB_i+L/2)), outer fingers first/last."""
    i = np.arange(int(nf))
    sa_i = sa + i * (sd + l)
    sb_i = sb + (nf - 1 - i) * (sd + l)
    return 1.0 / (sa_i + 0.5 * l) + 1.0 / (sb_i + 0.5 * l)


def lod_mobility_ratio(sa, sb, l, ku0, saref=None, sbref=None, nf=1, sd=0.0,
                       kstress=1.0):
    """mu_eff(SA, SB) / mu_eff(SAREF, SBREF) from the BSIM4 stress model.

    ku0 has units of length (same unit as sa). With saref = sbref = None the
    reference is a stress-free device (Inv_saref = Inv_sbref = 0).
    """
    rho = ku0 / kstress * (inv_sa(sa, l, nf, sd) + inv_sa(sb, l, nf, sd))
    if saref is None:
        rho_ref = 0.0
    else:
        rho_ref = ku0 / kstress * (1.0 / (saref + 0.5 * l) + 1.0 / (sbref + 0.5 * l))
    return (1.0 + rho) / (1.0 + rho_ref)


def lod_delta_vth(sa, sb, l, kvth0, saref=None, sbref=None, nf=1, sd=0.0,
                  kstress=1.0):
    """VTH0 shift (V) from the BSIM4 stress model; kvth0 in V*length."""
    term = inv_sa(sa, l, nf, sd) + inv_sa(sb, l, nf, sd)
    if saref is not None:
        term = term - (1.0 / (saref + 0.5 * l) + 1.0 / (sbref + 0.5 * l))
    return kvth0 / kstress * term


# ---------------------------------------------------------------- WPE
def wpe_sca(sc, d, scref=1.0):
    """SCA for one straight well edge: (1/d) * integral_sc^{sc+d} scref^2/s^2 ds."""
    sc = np.asarray(sc, dtype=float)
    return scref ** 2 / d * (1.0 / sc - 1.0 / (sc + d))


def _wpe_exp_term(sc, d, scref, k):
    a = k / scref
    g = lambda s: (scref / k * s + scref ** 2 / k ** 2) * np.exp(-a * s)
    return (g(sc) - g(sc + d)) / (d * scref)


def wpe_scb(sc, d, scref=1.0):
    """SCB for one straight edge: average of (s/scref) exp(-10 s/scref)."""
    return _wpe_exp_term(np.asarray(sc, dtype=float), d, scref, 10.0)


def wpe_scc(sc, d, scref=1.0):
    """SCC for one straight edge: average of (s/scref) exp(-20 s/scref)."""
    return _wpe_exp_term(np.asarray(sc, dtype=float), d, scref, 20.0)


def wpe_delta_vth(sc, d, kvth0we, web=0.0, wec=0.0, scref=1.0):
    """VTH0 shift (V) = KVTH0WE * (SCA + WEB*SCB + WEC*SCC)."""
    return kvth0we * (wpe_sca(sc, d, scref) + web * wpe_scb(sc, d, scref)
                      + wec * wpe_scc(sc, d, scref))


# ------------------------------------------------- STI thermal mismatch
# Origin of the LOD stress: the trench oxide relaxes by viscous flow above
# roughly 950-975 C (EerNisse, APL 30, 290, 1977; APL 35, 8, 1979), so the
# oxide/silicon pair is stress-free at that "freeze-in" temperature. On
# cool-down silicon contracts more than SiO2, leaving the oxide plug
# oversized for its trench and the active area in compression.

_trapz = getattr(np, "trapezoid", None) or np.trapz
ALPHA_SIO2 = 0.5e-6  # 1/K, fused/thermal silica, roughly constant 300-1300 K


def alpha_si(T_K):
    """Linear thermal expansion coefficient of silicon (1/K).

    Okada & Tokumaru, J. Appl. Phys. 56, 314 (1984), empirical fit valid
    120-1500 K:
        alpha = (3.725*(1 - exp(-5.88e-3*(T - 124))) + 5.548e-4*T) * 1e-6
    Gives 2.57e-6 /K at 300 K and about 4.4e-6 /K at 1250 K.
    """
    T = np.asarray(T_K, dtype=float)
    return (3.725 * (1.0 - np.exp(-5.88e-3 * (T - 124.0)))
            + 5.548e-4 * T) * 1e-6


def sti_thermal_mismatch_strain(T_freeze_C, T_room_C=25.0,
                                alpha_ox=ALPHA_SIO2, n=400):
    """Free thermal-mismatch strain between Si and SiO2 on cool-down.

    eps = integral_{T_room}^{T_freeze} (alpha_Si(T) - alpha_ox) dT

    This is the unconstrained misfit (how much too large the oxide plug is
    relative to its trench, as a fraction), not the stress in the channel,
    which is a geometry-dependent fraction of E*eps. Positive means the
    silicon ends up in compression next to the oxide.
    """
    T = np.linspace(T_room_C + 273.15, T_freeze_C + 273.15, n)
    return float(_trapz(alpha_si(T) - alpha_ox, T))
