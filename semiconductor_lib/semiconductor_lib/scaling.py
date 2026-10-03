"""
MOSFET scaling rules: Dennard constant-field scaling, generalized
(Baccarani) scaling, post-Dennard fixed-voltage scaling, the
subthreshold-leakage cost of lowering V_T, and Moore's-law counts.

All results are dimensionless RATIOS (new / old) for one scaling step.

Notation
--------
kappa : linear dimension scale factor (>1). L, W, t_ox, x_j -> /kappa.
        One "full node" in classic usage is kappa = sqrt(2) (0.7x linear,
        0.5x area).
lam   : voltage scale factor (>=1). V_DD, V_T -> /lam.
        lam = kappa  -> Dennard constant-field scaling (field unchanged).
        lam = 1      -> constant-voltage scaling.
        1 < lam < kappa -> generalized scaling; field rises by kappa/lam.

Current models (drive current per device)
-----------------------------------------
'long'   : square law, I ~ (W/L) Cox V^2           -> I ratio = kappa/lam^2
'velsat' : velocity saturated, I ~ W Cox V v_sat    -> I ratio = 1/lam
Under Dennard scaling (lam = kappa) both give I ratio = 1/kappa.

Derived ratios
--------------
C (gate)            = 1/kappa          (area/kappa^2, thickness/kappa)
delay  CV/I
energy CV^2         = 1/(kappa lam^2)
power/device CV^2 f   with f = 1/delay
power density       = power/device * kappa^2   (device density ~ kappa^2)
power density at fixed frequency = C V^2 * kappa^2 = kappa/lam^2

Checks (see tests/test_scaling.py): Dennard -> power density exactly 1,
delay 1/kappa, energy 1/kappa^3; constant-voltage long-channel ->
power density kappa^3 (textbook result, e.g. Taur & Ning Table 4.1).

Reference: R. H. Dennard et al., IEEE JSSC SC-9, 256 (1974);
G. Baccarani, M. R. Wordeman, R. H. Dennard, IEEE T-ED 31, 452 (1984).
"""
import numpy as np


def scaling_ratios(kappa, lam=None, current_model="long"):
    """Return a dict of new/old ratios for one scaling step.
    lam defaults to kappa (Dennard constant-field scaling)."""
    if lam is None:
        lam = kappa
    if kappa <= 0 or lam <= 0:
        raise ValueError("kappa and lam must be positive")
    C = 1.0 / kappa
    V = 1.0 / lam
    if current_model == "long":
        I = kappa / lam ** 2
    elif current_model == "velsat":
        I = 1.0 / lam
    else:
        raise ValueError("current_model must be 'long' or 'velsat'")
    delay = C * V / I
    f = 1.0 / delay
    energy = C * V ** 2
    p_dev = energy * f
    density = kappa ** 2
    return {
        "dimension": 1.0 / kappa,
        "voltage": V,
        "field": kappa / lam,
        "doping": kappa ** 2 / lam,          # Baccarani: eps*kappa, eps = kappa/lam
        "capacitance": C,
        "current": I,
        "delay": delay,
        "frequency": f,
        "energy_per_switch": energy,
        "power_per_device": p_dev,
        "device_density": density,
        "power_density": p_dev * density,
        "power_density_fixed_f": energy * density,
    }


def power_density_vs_generation(n_gen, kappa=np.sqrt(2), regime="dennard"):
    """Cumulative power-density ratio after 0..n_gen generations.
    regime: 'dennard' (lam=kappa), 'fixedV_fixedf' (lam=1, f held),
    'fixedV_velsat' (lam=1, f = 1/delay, velocity-saturated current)."""
    g = np.arange(n_gen + 1)
    if regime == "dennard":
        r = scaling_ratios(kappa)["power_density"]
    elif regime == "fixedV_fixedf":
        r = scaling_ratios(kappa, 1.0)["power_density_fixed_f"]
    elif regime == "fixedV_velsat":
        r = scaling_ratios(kappa, 1.0, "velsat")["power_density"]
    else:
        raise ValueError(regime)
    return g, r ** g


def off_current_ratio(delta_VT, S=0.060):
    """Factor by which I_off rises when V_T is LOWERED by delta_VT (V),
    for subthreshold swing S (V/decade). I_off ~ 10^(-V_T/S)."""
    return 10.0 ** (np.asarray(delta_VT) / S)


def moore_count(t, N0, t0, T_double):
    """Component count N(t) = N0 * 2^((t - t0)/T_double)."""
    return N0 * 2.0 ** ((np.asarray(t) - t0) / T_double)
