"""
Cross-check: velocity saturation of I_Dsat versus channel length, ngspice +
PTM 180 nm BSIM3 card (180nm_bulk.pm) against semiconductor_lib.high_field.

Method: NMOS, W = 10 um, V_GS = 1.8 V. V_T extracted per length by constant
current (100 nA x W/L at V_DS = 0.05 V) to remove V_T roll-off/RSCE.
Effective length L_eff = L - 2*Lint = L - 0.08 um (card Lint = 40 nm).
Mobility fitted to the L = 10 um device (square law, vertical-field
degradation absorbed): mu = 148 cm^2/Vs. v_sat = 1.38e7 cm/s from the card.
Output-conductance (CLM + DIBL) gain removed by extrapolating the
saturation-region I_D(V_DS) line (1.2-1.8 V) back to V_Dsat.

Verified (2026-10-04, ngspice-42):
  L_drawn  L_eff  sim/square-law  sim/(Esat=2vsat/mu)  sim/(Esat=vsat/mu)
  0.25     0.17   0.63            0.89                 1.09
  0.35     0.27   0.72            0.91                 1.07
  0.50     0.42   0.80            0.94                 1.06
  1.00     0.92   0.90            0.98                 1.04
  2.00     1.92   0.95            0.99                 1.03
  CLM + DIBL add 6.8 % (L = 0.25 um) to 0.9 % (2 um) above the knee.
Conclusion: the simulator loses 37 % against the square law at
L_eff = 0.17 um; the two analytic velocity-saturation forms bracket it
(n = 1 form high by <= 9 %, Hu/BSIM form low by <= 11 %). Using drawn L
instead of L_eff, or I_D at V_DS = V_DD instead of at the knee, hides the
penalty almost entirely, a measurement trap worth remembering.
"""
import os, subprocess
import numpy as np
from semiconductor_lib.mosfet import cox_from_tox

HERE = os.path.dirname(os.path.abspath(__file__))
CARD = os.path.join(HERE, "180nm_bulk.pm")
W_UM, VGS, LINT2, VSAT = 10.0, 1.8, 0.08, 1.38e7


def drain_current(L_um, vgs, vds):
    net = f"""vsat
.include {CARD}
M1 d g 0 0 NMOS L={L_um}u W={W_UM}u
Vd d 0 {vds}
Vg g 0 {vgs}
.control
op
print -i(Vd)
.endc
.end
"""
    path = os.path.join(HERE, "_vsat_tmp.cir")
    with open(path, "w") as f:
        f.write(net)
    out = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True).stdout
    os.remove(path)
    for line in out.splitlines():
        if "-i(vd)" in line.lower() and "=" in line:
            return float(line.split("=")[1])
    raise RuntimeError(out[-500:])


def vt_const_current(L_um):
    vg = np.arange(0.0, 1.2, 0.005)
    I = np.array([drain_current(L_um, v, 0.05) for v in vg])
    return np.interp(1e-7 * W_UM / L_um, I, vg)


def main():
    Cox = cox_from_tox(4e-7)
    W = W_UM * 1e-4
    Lr = 10.0
    mu = drain_current(Lr, VGS, 1.8) * 2 * (Lr - LINT2) * 1e-4 / (W * Cox * (VGS - vt_const_current(Lr)) ** 2)
    print(f"mu_fit = {mu:.1f} cm^2/Vs")
    for L in [0.25, 0.35, 0.5, 1.0, 2.0]:
        Le = (L - LINT2) * 1e-4
        Vov = VGS - vt_const_current(L)
        sq = W * Cox * mu * Vov ** 2 / (2 * Le)
        a2 = 2 * VSAT / mu * Le
        hu = sq / (1 + Vov / a2)
        a1 = VSAT / mu * Le
        n1 = W * VSAT * Cox * (Vov - a1 * (np.sqrt(1 + 2 * Vov / a1) - 1))
        vds = np.array([1.2, 1.5, 1.8])
        Ids = np.array([drain_current(L, VGS, v) for v in vds])
        slope = np.polyfit(vds, Ids, 1)[0]
        knee = Ids[-1] - slope * (1.8 - a2 * Vov / (a2 + Vov))
        print(f"L={L:4.2f}  sim/square={knee/sq:.2f}  sim/Hu={knee/hu:.2f}  sim/n1={knee/n1:.2f}")


if __name__ == "__main__":
    main()
