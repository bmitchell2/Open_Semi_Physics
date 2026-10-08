"""
Thin-film deposition models used on the Semiconductor Notes Deposition pages.

Contents
--------
grove_growth_rate      : Grove two-resistance CVD rate (surface reaction in
                         series with gas-phase mass transfer).
via_bottom_to_top      : Knudsen-diffusion / first-order-loss (Thiele) estimate
                         of reactant depletion down a cylindrical via or a
                         long trench, for CVD conformality.
berg_reactive_sputter  : Steady-state Berg model of reactive sputtering
                         (target poisoning and the flow hysteresis loop).
stoney_stress          : Film stress from substrate curvature (Stoney).
molecular_flux_per_Pa  : Hertz-Knudsen impingement flux.

Units are SI unless a docstring says otherwise.
"""
import numpy as np

K_B = 1.380649e-23        # J/K
Q_E = 1.602176634e-19     # C
AMU = 1.66053907e-27      # kg
SCCM = 4.4615e17          # molecules/s per sccm (0 C, 1 atm reference)


# ---------------------------------------------------------------- CVD rate
def grove_growth_rate(T, ks0, Ea_eV, hg, Cg=1.0, N_film=1.0):
    """Grove model: G = (Cg/N) * ks*hg/(ks+hg), ks = ks0*exp(-Ea/kT).

    T in K. ks0 and hg share units (velocity); G is in the units of
    hg*Cg/N. hg may be a scalar or an array matching T. The slower of the
    two 'conductances' ks (surface reaction) and hg (gas transport)
    controls the rate: reaction-limited when ks << hg, transport-limited
    when ks >> hg.
    """
    kT = K_B * np.asarray(T, float) / Q_E
    ks = ks0 * np.exp(-Ea_eV / kT)
    return (Cg / N_film) * ks * hg / (ks + hg)


# ------------------------------------------------------- feature conformality
def thiele_modulus(aspect_ratio, sticking, geometry="via"):
    """Thiele modulus for first-order reactant loss in a high-aspect-ratio
    feature in the free-molecular (Knudsen) regime.

    via (cylinder, diameter d):  phi = AR*sqrt(3*s)
        from D_K = d*vbar/3 and wall loss s*n*vbar/4 per unit area.
    trench (long slit, width w): phi = AR*sqrt(2*s) using the slit
        Knudsen diffusivity approximated as D_K ~ w*vbar/2.
    Valid for s << 1 (many wall collisions before reaction).
    """
    AR = np.asarray(aspect_ratio, float)
    s = np.asarray(sticking, float)
    if geometry == "via":
        return AR * np.sqrt(3.0 * s)
    if geometry == "trench":
        return AR * np.sqrt(2.0 * s)
    raise ValueError("geometry must be 'via' or 'trench'")


def via_bottom_to_top(aspect_ratio, sticking, geometry="via"):
    """Ratio of reactant concentration (and hence local CVD rate) at the
    feature bottom to that at the opening: 1/cosh(phi). Neglects the
    bottom face, film growth changing the geometry, and re-emission
    beyond what the net sticking coefficient already describes."""
    return 1.0 / np.cosh(thiele_modulus(aspect_ratio, sticking, geometry))


# ------------------------------------------------------ reactive sputtering
def molecular_flux_per_Pa(mass_amu, T=300.0):
    """Hertz-Knudsen impingement flux per unit pressure, molecules/(m^2 s Pa)."""
    return 1.0 / np.sqrt(2 * np.pi * mass_amu * AMU * K_B * T)


def berg_reactive_sputter(P, J=100.0, A_t=0.01, A_c=0.5, Y_m=0.5, Y_c=0.05,
                          alpha=1.0, S=0.1, mass_amu=28.0, T=300.0, n_atoms=2):
    """Steady-state Berg model of reactive sputtering.

    P       : reactive-gas partial pressure(s), Pa
    J       : target ion current density, A/m^2
    A_t,A_c : target and collector (wafer + walls) areas, m^2
    Y_m,Y_c : sputter yields of metal and compound (atoms/ion)
    alpha   : sticking coefficient of the reactive gas
    S       : pumping speed, m^3/s
    n_atoms : reactive atoms per molecule (2 for N2 or O2)

    Balances (per unit area, compound formed per reactive *atom*):
      target    : (J/q) Y_c th_t = n*alpha*F (1-th_t)
      collector : (J/q) Y_c th_t A_t (1-th_c) + n*alpha*F (1-th_c) A_c
                  = (J/q) Y_m (1-th_t) A_t th_c
    with F = P * molecular_flux_per_Pa. Returns a dict with theta_t,
    theta_c, total supply flow Q (sccm) and sputtered-atom rate R
    (atoms/s) leaving the target. Plotting Q against P gives the S-shaped
    curve whose fold produces the hysteresis loop when Q is the control.
    """
    P = np.asarray(P, float)
    F = P * molecular_flux_per_Pa(mass_amu, T)
    ions = J / Q_E
    th_t = n_atoms * alpha * F / (ions * Y_c + n_atoms * alpha * F)
    dep_c = ions * Y_c * th_t * A_t          # compound arriving at collector
    dep_m = ions * Y_m * (1 - th_t) * A_t    # metal arriving at collector
    react_c = n_atoms * alpha * F * A_c      # compound formed by gas on collector
    th_c = (dep_c + react_c) / (dep_c + react_c + dep_m)
    q_target = alpha * F * (1 - th_t) * A_t          # molecules/s consumed
    q_coll = alpha * F * (1 - th_c) * A_c
    q_pump = S * P / (K_B * T)
    Q = (q_target + q_coll + q_pump) / SCCM
    R = ions * A_t * (Y_m * (1 - th_t) + Y_c * th_t)
    return {"theta_t": th_t, "theta_c": th_c, "Q_sccm": Q, "rate": R,
            "Q_target": q_target / SCCM, "Q_collector": q_coll / SCCM,
            "Q_pump": q_pump / SCCM}


def berg_has_hysteresis(P, **kw):
    """True if Q(P) is non-monotonic, i.e. a flow-controlled process has a
    hysteresis loop (two stable pressures for one flow)."""
    Q = berg_reactive_sputter(P, **kw)["Q_sccm"]
    return bool(np.any(np.diff(Q) < 0))


# ------------------------------------------------------------------ stress
def stoney_stress(E_s, nu_s, t_s, t_f, R_before, R_after):
    """Stoney film stress (Pa) from substrate curvature change.

    sigma = E_s t_s^2 / (6 (1-nu_s) t_f) * (1/R_after - 1/R_before)
    E_s in Pa, thicknesses and radii in m. Valid for t_f << t_s, uniform
    thin film, small deflection. Sign: positive = tensile when the
    curvature convention makes a tensile film concave on the film side
    (R > 0 for film-side-concave)."""
    return E_s * t_s ** 2 / (6.0 * (1.0 - nu_s) * t_f) * (1.0 / R_after - 1.0 / R_before)
