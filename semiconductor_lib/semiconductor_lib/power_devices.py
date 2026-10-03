"""
Drift-region limits for unipolar power devices, lateral RESURF charge,
and ideal switched-capacitor charge pumps.

Backs the Semiconductor Notes pages
"Lateral Power Transistors (Drift Region, RESURF, Field Plates, and the
On-Resistance-Breakdown Tradeoff)" and "Charge Pumps (Switched-Capacitor
Voltage Inversion and Multiplication)".

Units: cm, V, cm^2/V.s, F/cm, Ohm.cm^2 unless stated.

Models
------
1. Constant-critical-field (punch-through-free, triangular field) drift
   region, the textbook unipolar limit (Baliga, J. Appl. Phys. 53, 1759
   (1982)):
       W  = 2 BV / Ec,   N = eps Ec^2 / (2 q BV)
       Ron,sp = W / (q mu N) = 4 BV^2 / (eps mu Ec^3)
   Ron scales as BV^2 when Ec is treated as constant.
2. Silicon with doping-dependent Ec (Baliga's power-law impact-ionization
   fit, parallel-plane junction):
       BV = 5.34e13 N^-3/4 ;  W = 2.67e10 N^-7/8   (N in cm^-3, W in cm)
       Ron,sp = 5.93e-9 BV^2.5   (Ohm.cm^2, mu_n = 1360 cm^2/V.s)
   The extra half power appears because Ec itself falls slowly with the
   lighter doping that higher BV requires.
3. Lateral 2DEG drift region at average field E_avg:
       L_D = BV / E_avg ;  Ron,sp = L_D^2 / (q n_s mu) = BV^2/(q n_s mu E_avg^2)
   (area per unit width taken as L_D; contacts, channel and pitch
   overhead are excluded, so real devices sit well above this).
4. RESURF / depletable-charge bound: the largest sheet charge that a
   field can fully deplete before reaching Ec is Q/q = eps Ec / q.
5. Ideal charge pumps (slow-switching limit, ideal switches):
       inverter: V_out = -V_in + I_L / (f C_fly)
       Dickson N-stage: V_out = (N+1)(V_in - V_d) - N I_L / (f C)
"""
import numpy as np

q = 1.602176634e-19          # C
eps0 = 8.8541878128e-14      # F/cm

# Material set used on the power-device page (300 K). Ec and bulk mu from
# EPC, "GaN Transistors for Efficient Power Conversion", Ch.1, Table 1.1;
# 4H-SiC mu is the bulk electron mobility along c commonly used in the
# Baliga limit; eps_r are standard values.
MATERIALS = {
    "Si":     {"eps_r": 11.7, "Ec": 0.3e6, "mu": 1350.0},
    "4H-SiC": {"eps_r": 9.7,  "Ec": 2.5e6, "mu": 900.0},
    "GaN":    {"eps_r": 9.5,  "Ec": 3.3e6, "mu": 990.0},
}


def drift_width(BV, Ec):
    """Minimum drift (depletion) width for a triangular field, W = 2BV/Ec (cm)."""
    return 2.0 * np.asarray(BV, float) / Ec


def drift_doping(BV, eps_r, Ec):
    """Optimum drift doping N = eps Ec^2 / (2 q BV) (cm^-3)."""
    return eps_r * eps0 * Ec**2 / (2.0 * q * np.asarray(BV, float))


def unipolar_ron_sp(BV, eps_r, mu, Ec):
    """Ideal specific on-resistance of a vertical unipolar drift region,
    Ron,sp = 4 BV^2 / (eps mu Ec^3), in Ohm.cm^2 (constant Ec)."""
    BV = np.asarray(BV, float)
    return 4.0 * BV**2 / (eps_r * eps0 * mu * Ec**3)


def material_ron_sp(BV, name):
    m = MATERIALS[name]
    return unipolar_ron_sp(BV, m["eps_r"], m["mu"], m["Ec"])


def baliga_figure_of_merit(eps_r, mu, Ec):
    """BFOM = eps mu Ec^3 (the denominator of the unipolar limit)."""
    return eps_r * eps0 * mu * Ec**3


def si_ron_sp_baliga(BV):
    """Silicon unipolar limit with doping-dependent Ec: 5.93e-9 BV^2.5 Ohm.cm^2."""
    return 5.93e-9 * np.asarray(BV, float)**2.5


def si_parallel_plane(N):
    """Silicon parallel-plane avalanche breakdown and depletion width at
    breakdown (Baliga power-law fit). Returns (BV in V, W in cm)."""
    N = np.asarray(N, float)
    return 5.34e13 * N**-0.75, 2.67e10 * N**-0.875


def lateral_ron_sp(BV, ns, mu, E_avg):
    """Drift-only specific on-resistance of a lateral 2DEG device,
    BV^2 / (q ns mu E_avg^2), Ohm.cm^2. ns in cm^-2, E_avg in V/cm."""
    BV = np.asarray(BV, float)
    return BV**2 / (q * ns * mu * E_avg**2)


def depletable_sheet_charge(eps_r, Ec):
    """Largest sheet density (cm^-2) a field can deplete before reaching
    Ec: eps Ec / q. Sets the scale of the RESURF drift dose."""
    return eps_r * eps0 * Ec / q


def inverter_vout(V_in, I_load, f_sw, C_fly):
    """Ideal inverting charge pump output, slow-switching limit:
    V_out = -V_in + I_load/(f C_fly). Output impedance 1/(f C_fly)."""
    return -V_in + I_load / (f_sw * C_fly)


def dickson_vout(V_in, n_stages, I_load, f_sw, C, V_d=0.0):
    """Dickson charge pump output: (N+1)(V_in - V_d) - N I_load/(f C)."""
    return (n_stages + 1) * (V_in - V_d) - n_stages * I_load / (f_sw * C)


def plot_ron_vs_bv(filename, ns=1e13, mu_2deg=1800.0, E_avg=1e6):
    """Log-log drift Ron,sp vs BV for Si (Baliga fit), 4H-SiC and GaN
    (constant-Ec limit) plus a lateral 2DEG drift region. Call
    plotting.apply_style() first. Visually verified 2026-10-03."""
    import matplotlib.pyplot as plt
    from .plotting import COLORS
    BV = np.logspace(1, 4, 40)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.loglog(BV, si_ron_sp_baliga(BV)*1e3, color=COLORS["blue"],
              label="Si (Baliga, $\\propto BV^{2.5}$)")
    ax.loglog(BV, material_ron_sp(BV, "4H-SiC")*1e3, color=COLORS["purple"],
              label="4H-SiC ($E_c$ 2.5 MV/cm)")
    ax.loglog(BV, material_ron_sp(BV, "GaN")*1e3, color=COLORS["green"],
              label="GaN bulk ($E_c$ 3.3 MV/cm)")
    ax.loglog(BV, lateral_ron_sp(BV, ns, mu_2deg, E_avg)*1e3, "o", ms=3.5,
              mfc="none", markevery=3, color=COLORS["red"],
              label="GaN lateral 2DEG ($n_s$=%.0e, $\\mu$=%d, $\\bar E$=%.1f MV/cm)"
                    % (ns, mu_2deg, E_avg/1e6))
    ax.axvline(650, color=COLORS["gray"], lw=0.8, ls=":")
    ax.text(680, 2e-4, "650 V", color=COLORS["gray"], fontsize=8)
    ax.set_xlabel("Breakdown voltage BV (V)")
    ax.set_ylabel("Drift specific on-resistance (m$\\Omega\\cdot$cm$^2$)")
    ax.set_xlim(10, 1e4); ax.set_ylim(1e-5, 1e4)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    fig.tight_layout(); fig.savefig(filename); plt.close(fig)
    return filename
