"""
CMP process control, in-situ sensing and post-CMP cleaning physics.

Engineering-level helpers used on the Semiconductor Notes CMP pages:

* Normal-incidence reflectance of a single transparent film on a
  substrate under an ambient (Airy summation).  Used for optical
  endpoint: reflectance oscillates with remaining film thickness with
  period lambda / (2 n_film).
* Electromagnetic skin depth delta = sqrt(2 rho / (omega mu0)) and the
  copper resistivity temperature coefficient, for eddy-current endpoint.
* Oscillatory (Stokes) boundary-layer thickness delta = sqrt(2 nu/omega)
  for megasonic cleaning.
* Electrolyte Debye length (symmetric 1:1 electrolyte) and the
  Smoluchowski zeta potential from electrophoretic mobility.
* Pad-conditioner coverage: largest radial sweep speed that leaves no
  unconditioned ring between successive platen revolutions.
* Run-to-run (R2R) polish-time control: feedforward of incoming
  thickness plus EWMA (single) or double-EWMA (drift-tracking) feedback
  on the removal-rate estimate.

All models are textbook or phenomenological; see docstrings.
"""
import numpy as np

EPS0 = 8.8541878128e-12   # F/m
MU0 = 4e-7 * np.pi        # H/m
K_B = 1.380649e-23        # J/K
Q = 1.602176634e-19       # C
N_A = 6.02214076e23       # 1/mol

RHO_CU_20C = 1.72e-8      # ohm m, bulk copper at 20 C
TCR_CU = 3.9e-3           # 1/K, bulk copper near room temperature


# ------------------------------------------------------- optical endpoint
def thin_film_reflectance(d_nm, lam_nm, n_film, n_sub, n_amb=1.333):
    """Normal-incidence power reflectance of one film (thickness d_nm,
    complex or real index n_film) on a semi-infinite substrate (n_sub,
    may be complex n + ik) under an ambient of index n_amb (water by
    default, as under the slurry film on a CMP pad window)."""
    d = np.asarray(d_nm, dtype=float)
    lam = np.asarray(lam_nm, dtype=float)
    r01 = (n_amb - n_film) / (n_amb + n_film)
    r12 = (n_film - n_sub) / (n_film + n_sub)
    phase = np.exp(-2j * 2 * np.pi * n_film * d / lam)
    r = (r01 + r12 * phase) / (1 + r01 * r12 * phase)
    return np.abs(r) ** 2


def fringe_period_nm(lam_nm, n_film):
    """Film thickness removed per full interference fringe (nm)."""
    return lam_nm / (2.0 * n_film)


# ------------------------------------------------------- eddy-current endpoint
def skin_depth_m(rho_ohm_m, f_hz, mu_r=1.0):
    """Electromagnetic skin depth (m) of a conductor."""
    return np.sqrt(2.0 * rho_ohm_m / (2 * np.pi * f_hz * MU0 * mu_r))


def copper_resistivity(T_C):
    """Bulk copper resistivity (ohm m), linear TCR about 20 C."""
    return RHO_CU_20C * (1.0 + TCR_CU * (np.asarray(T_C) - 20.0))


def apparent_thickness_error(true_thk_nm, dT_K, tcr=TCR_CU):
    """Thickness error (nm) if a sheet-conductance eddy signal calibrated
    at one temperature is read dT_K hotter: G = t/rho, so t_app = t/(1+a dT)."""
    return true_thk_nm / (1.0 + tcr * dT_K) - true_thk_nm


# ------------------------------------------------------- cleaning physics
def stokes_layer_m(f_hz, nu_m2_s=1.0e-6):
    """Oscillatory (Stokes) boundary-layer thickness sqrt(2 nu / omega) (m)."""
    return np.sqrt(2.0 * nu_m2_s / (2 * np.pi * f_hz))


def debye_length_electrolyte_m(I_molar, T_K=298.15, eps_r=78.5):
    """Debye length (m) of a 1:1 aqueous electrolyte of ionic strength
    I_molar (mol/L)."""
    I = np.asarray(I_molar, dtype=float) * 1000.0 * N_A   # ions per m^3 (per species)
    return np.sqrt(eps_r * EPS0 * K_B * T_K / (2.0 * Q ** 2 * I))


def zeta_smoluchowski_V(mobility_m2_Vs, eta_Pa_s=0.89e-3, eps_r=78.5):
    """Zeta potential (V) from electrophoretic mobility, Smoluchowski limit
    (thin double layer: particle radius >> Debye length)."""
    return np.asarray(mobility_m2_Vs) * eta_Pa_s / (eps_r * EPS0)


# ------------------------------------------------------- pad conditioning
def max_gapfree_sweep_speed(disk_diameter_m, platen_rpm):
    """Largest radial conditioner speed (m/s) that advances less than one
    disk diameter per platen revolution."""
    return disk_diameter_m * platen_rpm / 60.0


# ------------------------------------------------------- run-to-run control
def r2r_simulate(true_rate, incoming, target, rate0, lam=0.3, lam2=None,
                 meas_noise=0.0, rng=None):
    """Simulate feedforward + EWMA feedback polish-time control.

    true_rate : array of actual removal rates per wafer (nm/s)
    incoming  : array of incoming thickness (nm), assumed measured exactly
    target    : post-polish target thickness (nm)
    rate0     : initial rate estimate (nm/s)
    lam       : EWMA weight on the newest rate measurement
    lam2      : if given, double-EWMA (predictor-corrector) weight on the
                rate slope, which removes steady lag under a linear drift
    meas_noise: std dev of post-polish thickness metrology (nm)

    Returns (output_thickness, rate_estimate_used, polish_time)."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(true_rate)
    a, b = float(rate0), 0.0
    out = np.empty(n); used = np.empty(n); times = np.empty(n)
    for k in range(n):
        r_hat = a + b
        t = (incoming[k] - target) / r_hat
        thk = incoming[k] - true_rate[k] * t
        thk_meas = thk + meas_noise * rng.standard_normal()
        r_meas = (incoming[k] - thk_meas) / t
        if lam2 is None:
            a = lam * r_meas + (1 - lam) * r_hat
        else:
            a_new = lam * r_meas + (1 - lam) * r_hat
            b = lam2 * (r_meas - a) + (1 - lam2) * b
            a = a_new
        out[k], used[k], times[k] = thk, r_hat, t
    return out, used, times
