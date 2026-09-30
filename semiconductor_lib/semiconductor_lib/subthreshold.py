"""
Subthreshold (weak-inversion) conduction and MOSFET off-state current.

Units: cm, V, cm^-3, F/cm^2, A.  n-channel device on a p-type body.

Model
-----
Weak-inversion drain current is diffusion of electrons injected over the
source-to-channel potential barrier. With the surface potential in weak
inversion following the gate with slope 1/m (m = 1 + C_dep/C_ox), the
inversion charge at the source end is exponential in V_GS and the current is
(Taur & Ning, Fundamentals of Modern VLSI Devices, 2nd ed., Sec. 3.1.3.2):

    I_D = beta (m - 1) phi_t^2 exp[(V_GS - V_T + eta V_DS) / (m phi_t)]
          * (1 - exp(-V_DS / phi_t))

beta = mu C_ox W/L, phi_t = kT/q, eta = DIBL coefficient (V/V, 0 for a
long channel). V_T here is the 2 phi_F (textbook) threshold, which is the
reference point at which this weak-inversion form is anchored.

Verified (tests/test_subthreshold.py):
- slope of log10(I_D) vs V_GS equals 1/(ln10 m phi_t) exactly;
- the closed form agrees with the Brews charge-sheet model
  (mosfet.id_charge_sheet) to within 15 % for the top ~3 decades below V_T
  at 300 K. Deeper in weak inversion it underestimates (by ~35 % at 6
  decades) because m is frozen at its 2 phi_F value while the true local
  C_dep/C_ox grows as psi_s falls; the charge-sheet swing is 3-6 % larger;
- current saturates in V_DS after a few phi_t (95 % at 3 phi_t);
- V_T temperature coefficient for N_A = 3e17, t_ox = 3 nm is about
  -1.5 mV/K, in the -0.5 to -2 mV/K range reported for bulk MOSFETs.
"""
import numpy as np

from .constants import thermal_voltage
from . import mosfet


def id_subthreshold(V_GS, V_DS, V_T, m, beta, T=300.0, eta=0.0):
    """Weak-inversion drain current (A), valid for V_GS well below V_T."""
    pt = thermal_voltage(T)
    V_GS = np.asarray(V_GS, float)
    return (beta * (m - 1.0) * pt ** 2
            * np.exp((V_GS - V_T + eta * V_DS) / (m * pt))
            * (1.0 - np.exp(-np.asarray(V_DS, float) / pt)))


def device_params(N_A, Cox, V_FB=0.0, T=300.0, V_SB=0.0):
    """V_T (2 phi_F criterion) and m at temperature T for a uniformly doped
    body. Returns (V_T, m)."""
    vt = mosfet.vt_uniform(N_A, Cox, V_FB, V_SB, T, n_phit=0.0)
    m = mosfet.bulk_charge_factor_m(N_A, Cox, V_SB, T, n_phit=0.0)
    return vt, m


def ioff(N_A, Cox, beta300, V_DD, V_FB=0.0, T=300.0, eta=0.0,
         mobility_exponent=-1.5):
    """Off-state subthreshold current (A) at V_GS = 0, V_DS = V_DD.
    Mobility is scaled as (T/300)^mobility_exponent (phonon-limited)."""
    vt, m = device_params(N_A, Cox, V_FB, T)
    beta = beta300 * (T / 300.0) ** mobility_exponent
    return float(id_subthreshold(0.0, V_DD, vt, m, beta, T, eta))


def vt_tempco(N_A, Cox, V_FB=0.0, T=300.0, dT=1.0):
    """dV_T/dT (V/K) from the uniform-doping V_T with ni(T); V_FB held fixed
    (true for a midgap-referenced metal gate; an n+ poly gate adds a smaller
    term of the same sign)."""
    return (device_params(N_A, Cox, V_FB, T + dT)[0]
            - device_params(N_A, Cox, V_FB, T - dT)[0]) / (2 * dT)


def swing_from_curve(V_GS, I_D):
    """Minimum subthreshold swing (V/decade) of a sampled I_D(V_GS) curve."""
    s = np.diff(np.asarray(V_GS)) / np.diff(np.log10(np.asarray(I_D)))
    s = s[np.isfinite(s) & (s > 0)]
    return float(np.min(s))
