"""
Polycrystalline-silicon physics: grain-boundary trapping and conduction
(Seto model), temperature coefficient of resistance, solid-phase
crystallization (SPC) grain-size/time scaling, and a grain-counting
estimate of resistor mismatch. Backs the Semiconductor Notes pages
"Polysilicon Resistors: Grain-Boundary Conduction, TCR and Matching" and
"Polysilicon and Amorphous Silicon LPCVD".

Model and sources
-----------------
Grain-boundary trapping follows J. Y. W. Seto, "The electrical properties
of polycrystalline silicon films," J. Appl. Phys. 46, 5247 (1975),
doi:10.1063/1.321593: 1-D grains of size L, a single trap level at the
boundary with areal density Qt, depletion approximation on both sides of
each boundary, thermionic emission over the boundary barrier. Seto's fit
for boron-doped LPCVD poly: Qt = 3.34e12 cm^-2, trap 0.37 eV above E_V
(about 0.19 eV below the intrinsic level), grain size about 20 nm; Hall
mobility minimum near 2e18 cm^-3. These are the defaults (p-type).

Two regimes, solved self-consistently with trap occupancy:
  * partially depleted (Qt*f < N*L): W = Qt f / 2N,  V_B = q (Qt f)^2 / (8 eps N)
  * fully depleted     (Qt*f >= N*L): f = N L / Qt,  V_B = q N L^2 / (8 eps),
    and the Fermi level is set by the trap occupancy.
Resistivity = grain-boundary thermionic term + grain-interior term:
  rho_GB = sqrt(2 pi m* kT) exp(q V_B/kT) / (L q^2 p_c)    (Seto, linearized)
  rho_g  = 1 / (q p_c mu_CT(N, T))                         (Caughey-Thomas)
p_c is the majority-carrier density in the grain centre.

Limitations: single trap level, uniform cubic grains, full dopant
activation in the grain (no dopant segregation to the boundary; see
gate_power_punchthrough for the segregation/active-fraction picture),
non-degenerate statistics. Lightly doped films: the single-level model
gives an activation energy E_t - E_V (~0.37 eV with Seto's level); real
films approach E_g/2 because traps are distributed in energy. Use for
trends and orders of magnitude, not for sign-off of a resistor TCR.

SPC kinetics: Avrami (JMAK) scaling for random nucleation (rate I) and
isotropic growth (velocity v) in a film whose grains span its thickness
(2-D growth): grain size d ~ (v/I)^(1/3), crystallization time
t ~ (I v^2)^(-1/3). Activation energies: growth ~3.1-3.2 eV, nucleation
~3.9-5 eV (R. B. Iverson and R. Reif, J. Appl. Phys. 62, 1675 (1987);
Kung, Iverson and Reif, MRS Proc. on Si-implanted LPCVD a-Si: 3.9 eV
unimplanted, 4.9 eV after Si+ implant, growth 3.2 eV unchanged).
"""
import numpy as np
from scipy.optimize import brentq

from .constants import q, k_B, eps_si
from .scattering import caughey_thomas, CT_HOLES

M0 = 9.1093837015e-31          # kg
KB_EV = k_B / q                # eV/K

# Seto (1975) boron-doped LPCVD polysilicon fit
SETO_QT = 3.34e12              # cm^-2
SETO_DET_EV = 0.19             # trap depth below intrinsic level, eV (0.37 eV above E_V)
SETO_GRAIN_CM = 20e-7          # 20 nm
EG_300 = 1.12                  # eV


def ni_T(T, ni300=1.0e10, Eg=EG_300):
    """Intrinsic density vs temperature, ni300 (T/300)^1.5 exp(-Eg/2k (1/T-1/300)), cm^-3."""
    T = float(T)
    return ni300 * (T / 300.0) ** 1.5 * np.exp(-Eg / (2 * KB_EV) * (1.0 / T - 1.0 / 300.0))


def critical_doping(Qt=SETO_QT, L_cm=SETO_GRAIN_CM):
    """Doping at which a grain of size L is just fully depleted by Qt traps, N* = Qt/L (cm^-3)."""
    return Qt / L_cm


def seto_state(N, L_cm=SETO_GRAIN_CM, Qt=SETO_QT, dEt_eV=SETO_DET_EV, T=300.0):
    """
    Self-consistent grain-boundary state for majority-carrier (hole) doping N.

    Returns dict(V_B [V], p_center [cm^-3], f [trap fraction charged],
    W [cm, depletion width each side], regime 'partial' or 'full').
    """
    kT = KB_EV * T
    ni = ni_T(T)
    # partially depleted trial: p_c = N, E_F - E_t at centre from statistics
    EF_minus_Et_c = dEt_eV - kT * np.log(N / ni)

    def VB_of_f(f):
        return q * (Qt * f) ** 2 / (8 * eps_si * N)

    def g(f):
        x = (EF_minus_Et_c + VB_of_f(f)) / kT
        return f - 1.0 / (1.0 + np.exp(np.clip(x, -700, 700)))

    f = brentq(g, 0.0, 1.0, xtol=1e-14)
    W = Qt * f / (2 * N)
    if W <= L_cm / 2:
        return dict(V_B=VB_of_f(f), p_center=N, f=f, W=W, regime="partial")

    # fully depleted grain: charge balance fixes occupancy, occupancy fixes E_F
    f = min(N * L_cm / Qt, 1.0)
    V_B = q * N * L_cm ** 2 / (8 * eps_si)
    # f = 1/(1+exp((E_F - E_t,GB)/kT)),  E_t,GB = E_t,c - qV_B
    EF_minus_Et_c = kT * np.log(1.0 / f - 1.0) - V_B if f < 1 else -V_B
    # p_c = ni exp((E_i - E_F)/kT), E_i - E_t,c = dEt
    p_c = ni * np.exp((dEt_eV - EF_minus_Et_c) / kT)
    return dict(V_B=V_B, p_center=min(p_c, N), f=f, W=L_cm / 2, regime="full")


def poly_resistivity(N, L_cm=SETO_GRAIN_CM, Qt=SETO_QT, dEt_eV=SETO_DET_EV,
                     T=300.0, m_rel=0.5, return_parts=False):
    """Polysilicon resistivity (ohm cm): grain-boundary thermionic + grain interior."""
    s = seto_state(N, L_cm, Qt, dEt_eV, T)
    p_m3 = s["p_center"] * 1e6
    L_m = L_cm * 1e-2
    rho_gb = (np.sqrt(2 * np.pi * m_rel * M0 * k_B * T) * np.exp(s["V_B"] / (KB_EV * T))
              / (L_m * q ** 2 * p_m3)) * 100.0       # ohm m -> ohm cm
    mu = caughey_thomas(N, T, CT_HOLES)
    rho_g = 1.0 / (q * s["p_center"] * mu)
    rho = rho_gb + rho_g
    if return_parts:
        return rho, rho_gb, rho_g, s
    return rho


def single_crystal_resistivity(N, T=300.0):
    """p-type single-crystal Si resistivity (ohm cm), full ionization, Caughey-Thomas mobility."""
    return 1.0 / (q * N * caughey_thomas(N, T, CT_HOLES))


def poly_tcr_ppm(N, L_cm=SETO_GRAIN_CM, T=300.0, dT=1.0, **kw):
    """Temperature coefficient of resistance (1/R dR/dT) in ppm/K, central difference."""
    r_hi = poly_resistivity(N, L_cm, T=T + dT, **kw)
    r_lo = poly_resistivity(N, L_cm, T=T - dT, **kw)
    r0 = poly_resistivity(N, L_cm, T=T, **kw)
    return (r_hi - r_lo) / (2 * dT) / r0 * 1e6


def zero_tcr_doping(L_cm, N_lo=1e17, N_hi=1e21, **kw):
    """Doping (cm^-3) where the model TCR crosses zero for grain size L."""
    h = lambda logN: poly_tcr_ppm(10 ** logN, L_cm, **kw)
    return 10 ** brentq(h, np.log10(N_lo), np.log10(N_hi), xtol=1e-6)


def activation_energy_eV(N, L_cm=SETO_GRAIN_CM, T1=280.0, T2=320.0, **kw):
    """Arrhenius activation energy of resistivity between T1 and T2 (eV)."""
    r1 = poly_resistivity(N, L_cm, T=T1, **kw)
    r2 = poly_resistivity(N, L_cm, T=T2, **kw)
    return KB_EV * np.log(r1 / r2) / (1.0 / T1 - 1.0 / T2)


# ---------------------------------------------------------------- SPC kinetics
def spc_grain_size_ratio(T_C, T_ref_C=600.0, Ea_nuc=3.9, Ea_growth=3.2):
    """Final SPC grain size relative to T_ref, 2-D Avrami: d ~ (v/I)^(1/3)."""
    T = np.asarray(T_C, float) + 273.15
    Tr = T_ref_C + 273.15
    return np.exp((Ea_nuc - Ea_growth) / 3.0 / KB_EV * (1.0 / T - 1.0 / Tr))


def spc_time_ratio(T_C, T_ref_C=600.0, Ea_nuc=3.9, Ea_growth=3.2):
    """SPC completion time relative to T_ref, 2-D Avrami: t ~ (I v^2)^(-1/3)."""
    T = np.asarray(T_C, float) + 273.15
    Tr = T_ref_C + 273.15
    return np.exp((Ea_nuc + 2 * Ea_growth) / 3.0 / KB_EV * (1.0 / T - 1.0 / Tr))


# ------------------------------------------------------------- matching
def grain_element_sigma(gb_fraction, sigma_VB_V, T=300.0):
    """
    Relative resistance spread of one grain element, to first order:
    the grain-boundary part varies as exp(qV_B/kT), so
    sigma_r/r ~ gb_fraction * sigma_VB / (kT/q).
    gb_fraction = rho_GB/rho (0..1); sigma_VB = barrier-height spread (V).
    """
    return gb_fraction * sigma_VB_V / (KB_EV * T)


def resistor_mismatch_sigma(W_um, L_um, grain_um, sigma_element):
    """
    Relative spread of a resistor built from (W L / grain^2) independent
    grain elements: sigma_R/R = sigma_element * grain / sqrt(W L).
    Pair mismatch sigma(dR/R) is sqrt(2) times this. Returns sigma_R/R.
    Equivalent matching coefficient A_R = sqrt(2) sigma_element grain (um).
    """
    return sigma_element * grain_um / np.sqrt(W_um * L_um)


def monte_carlo_grid_sigma(nx, ny, sigma_element, trials=4000, seed=0):
    """
    Monte Carlo check of grain-counting: nx grains along current, ny across,
    each grain a resistor with lognormal spread sigma_element; rows are
    parallel strips of series grains (no lateral current). Returns relative
    std of total resistance.
    """
    rng = np.random.default_rng(seed)
    r = np.exp(rng.normal(0.0, sigma_element, size=(trials, ny, nx)))
    strips = r.sum(axis=2)                     # series along length
    total = 1.0 / (1.0 / strips).sum(axis=1)   # strips in parallel
    return total.std() / total.mean()


def hall_mobility(N, L_cm=SETO_GRAIN_CM, T=300.0, **kw):
    """Effective (Hall-like) mobility 1/(q p_c rho), cm^2/V s. Minimum sits near N* = Qt/L."""
    rho, _, _, s = poly_resistivity(N, L_cm, T=T, return_parts=True, **kw)
    return 1.0 / (q * s["p_center"] * rho)
