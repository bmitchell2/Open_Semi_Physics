"""
Beamline implanter dosimetry and angle-sensitivity models.

Covers three mechanisms that make tilted, low-energy implants (pocket/halo,
extension) drift or mismatch between implanters:

1. Charge-exchange neutralization and pressure compensation
   I_dose = I_meas * exp(K P)  (exponential single-reaction model; the form
   used in Axcelis' pressure-compensation patent US 6,657,209).
2. Energy contamination from neutrals formed upstream of a deceleration lens.
3. Tilt geometry: projection of the range onto lateral/vertical axes and
   the shadow cast by a mask or gate of height h.

Units: pressure in Torr, cross-sections in cm^2, lengths in cm unless noted,
angles in degrees at the public interface.
"""
import math

from .constants import k_B

TORR_TO_PA = 133.322


def gas_density_cm3(P_torr, T=300.0):
    """Ideal-gas number density (cm^-3) at pressure P_torr and temperature T."""
    n_m3 = P_torr * TORR_TO_PA / (k_B * T)
    return n_m3 * 1e-6


def surviving_ion_fraction(sigma_cm2, P_torr, L_cm, T=300.0):
    """Fraction of ions that keep their charge over path length L_cm through
    gas at P_torr, for a single dominant charge-exchange cross-section.
    Beer-Lambert attenuation: exp(-sigma n L)."""
    return math.exp(-sigma_cm2 * gas_density_cm3(P_torr, T) * L_cm)


def k_factor_from_cross_section(sigma_cm2, L_cm, T=300.0):
    """Pressure-compensation K-factor (1/Torr) implied by sigma*L: K = sigma*L*n/P.
    Real K values also absorb gauge position and pressure gradients, so this
    is an order-of-magnitude check, not a substitute for a fitted K."""
    return sigma_cm2 * L_cm * gas_density_cm3(1.0, T)


def true_to_measured_dose_ratio(K_true, P_torr):
    """Delivered dopant flux / Faraday-measured flux = exp(K_true P).
    Neutralized ions still implant but are not counted."""
    return math.exp(K_true * P_torr)


def residual_dose_error(K_true, K_used, P_torr):
    """Fractional over-dose remaining when the dose controller compensates with
    K_used but the tool's true factor is K_true: exp((K_true-K_used) P) - 1.
    K_used = 0 means no compensation."""
    return math.exp((K_true - K_used) * P_torr) - 1.0


def fit_k_factor(pressures_torr, measured_currents):
    """Least-squares K from ln(I_meas) = ln(I0) - K P, for a constant true
    particle flux. Returns (K, I0)."""
    n = len(pressures_torr)
    if n < 2:
        raise ValueError("need at least two points")
    ys = [math.log(i) for i in measured_currents]
    mx = sum(pressures_torr) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in pressures_torr)
    sxy = sum((x - mx) * (y - my) for x, y in zip(pressures_torr, ys))
    slope = sxy / sxx
    return -slope, math.exp(my - slope * mx)


def decel_contamination_fraction(sigma_cm2, P_torr, L_pre_cm, T=300.0):
    """Fraction of the beam neutralized in the line-of-sight drift region
    upstream of a deceleration lens. Those neutrals reach the wafer at the
    transport (pre-decel) energy. Small-fraction exact form 1 - exp(-sigma n L)."""
    return 1.0 - surviving_ion_fraction(sigma_cm2, P_torr, L_pre_cm, T)


def contaminant_depth_ratio(decel_ratio, exponent=0.5):
    """Approximate range ratio of contaminant (transport energy) to main beam
    (final energy), R ~ E^exponent. Exponent ~0.5 for low-energy B in Si where
    electronic stopping dominates; it approaches 1 when nuclear stopping
    dominates. Order-of-magnitude only."""
    if decel_ratio < 1:
        raise ValueError("decel ratio must be >= 1")
    return decel_ratio ** exponent


def tilted_projection(R_p, tilt_deg):
    """(lateral, vertical) projection of a straight-line range R_p at tilt."""
    t = math.radians(tilt_deg)
    return R_p * math.sin(t), R_p * math.cos(t)


def lateral_reach_sensitivity(R_p, tilt_deg, d_tilt_deg):
    """Change in lateral reach for a tilt error d_tilt_deg: R_p cos(theta) dtheta."""
    return R_p * math.cos(math.radians(tilt_deg)) * math.radians(d_tilt_deg)


def shadow_length(h, tilt_deg):
    """Shadow cast by an edge of height h at tilt (projected in the tilt plane)."""
    return h * math.tan(math.radians(tilt_deg))


def shadow_sensitivity(h, tilt_deg, d_tilt_deg):
    """Change in shadow length for a tilt error: h sec^2(theta) dtheta."""
    t = math.radians(tilt_deg)
    return h / math.cos(t) ** 2 * math.radians(d_tilt_deg)
