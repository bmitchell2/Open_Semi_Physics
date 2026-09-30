"""
Cross-check: subthreshold swing, DIBL, and off-current temperature scaling
from a real BSIM3 model card (ngspice + PTM 180 nm, 180nm_bulk.pm) against
semiconductor_lib.subthreshold.

Setup: NMOS, W = 10 um, L = 1.8 um and 0.18 um, V_DS = 0.05 V and 1.8 V,
T = 27 C and 125 C, V_GS swept -0.3 to 1.0 V. gmin is set to 1e-18 S:
with the default gmin (1e-12 S) ngspice adds a ~3.6 pA drain-to-bulk floor
that masks the long-channel off-current. V_T is extracted by constant
current, 100 nA x W/L.

Verified (2026-09-29, ngspice-42):
  swing, L = 1.8 um:    74.1 mV/dec (27 C), 99.5 mV/dec (125 C); ratio 1.34
                        vs T ratio 398/300 = 1.33
  analytic (N_A = 5.95e17, t_ox = 4 nm, V_T matched to 0.367 V):
                        75.5 and 101.7 mV/dec  -> agreement within 3 %
  DIBL:                 7 mV/V (L = 1.8 um), 77 mV/V (L = 0.18 um)
  I_off at V_DS = 1.8 V, 27 C: 2.3 pA/um (L = 1.8 um), 0.94 nA/um (0.18 um)
  I_off(125 C)/I_off(27 C): 175x (1.8 um), 61x (0.18 um)
  dV_T/dT:              -1.2 mV/K (card, kt1 = -0.37 fitted);
                        ideal uniform-doping model gives -1.5 mV/K and so
                        overestimates the I_off rise (730x). The I_off ratio
                        is exponentially sensitive to dV_T/dT; use measured
                        or model-card tempco for projections.
"""
import os
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_CARD = os.path.join(HERE, "180nm_bulk.pm")


def sweep(L, vds, temp, W=10e-6, step=0.01):
    """Return (V_GS, |I_D|) arrays from an ngspice DC sweep."""
    net = f"""subthreshold sweep
.include {MODEL_CARD}
.options temp={temp} tnom=27 gmin=1e-18
M1 d g 0 0 NMOS L={L} W={W}
Vd d 0 {vds}
Vg g 0 0
.control
dc Vg -0.3 1.0 {step}
wrdata _sub_out.txt -i(Vd)
.endc
.end
"""
    with open("_sub.cir", "w") as f:
        f.write(net)
    subprocess.run(["ngspice", "-b", "_sub.cir"], capture_output=True, check=True)
    d = np.loadtxt("_sub_out.txt")
    os.remove("_sub_out.txt"); os.remove("_sub.cir")
    return d[:, 0], np.abs(d[:, 1])


def vt_constant_current(vg, i, W, L, i_crit=100e-9):
    return float(np.interp(np.log(i_crit * W / L), np.log(i), vg))


def main():
    res = {}
    for L in (1.8e-6, 0.18e-6):
        for T in (27, 125):
            for vds in (0.05, 1.8):
                vg, i = sweep(L, vds, T)
                ss = np.diff(vg) / np.diff(np.log10(i))
                m = (vg[1:] > -0.1) & (vg[1:] < 0.25)
                res[(L, T, vds)] = dict(ss=np.min(ss[m]), ioff=np.interp(0, vg, i),
                                        vt=vt_constant_current(vg, i, 10e-6, L))
    for L in (1.8e-6, 0.18e-6):
        r = res
        print(f"L = {L*1e6:.2f} um: S = {r[(L,27,1.8)]['ss']*1e3:.1f} / "
              f"{r[(L,125,1.8)]['ss']*1e3:.1f} mV/dec (27/125 C), "
              f"DIBL = {(r[(L,27,0.05)]['vt']-r[(L,27,1.8)]['vt'])/1.75*1e3:.0f} mV/V, "
              f"Ioff = {r[(L,27,1.8)]['ioff']/10:.3g} A/um, "
              f"Ioff ratio 125/27 = {r[(L,125,1.8)]['ioff']/r[(L,27,1.8)]['ioff']:.0f}")


if __name__ == "__main__":
    main()
