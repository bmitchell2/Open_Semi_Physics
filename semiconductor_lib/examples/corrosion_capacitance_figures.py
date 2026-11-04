"""Figures for the copper CMP corrosion page (Evans diagram, galvanic
area-ratio effect) and the MOSFET gate capacitance page (outer fringe vs
gate height with numerical cross-check; contact share of extrinsic Cgd).
Illustrative parameters; analytical/numerical models, not measured data."""
import numpy as np
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib.pyplot as plt
from semiconductor_lib import corrosion as cr
from semiconductor_lib import mos_capacitance as mc

C = COLORS
# ------------------------------------------------------------ figure 1
CU = dict(Ea=0.0, i0a=1e-6, ba=0.06, Ec=0.6, i0c=1e-9, bc=0.12, ilim=1e-3)
CO = dict(Ea=-0.25, i0a=1e-6, ba=0.06, Ec=0.6, i0c=1e-9, bc=0.12, ilim=1e-3)
M_CO = (58.933, 2, 8.90)

fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2))
E = np.linspace(-0.35, 0.45, 160)
a.semilogx(cr.anodic_current(E, CU["Ea"], CU["i0a"], CU["ba"]), E, c=C["orange"], label="Cu → Cu²⁺ (anodic)")
a.semilogx(cr.anodic_current(E, CO["Ea"], CO["i0a"], CO["ba"]), E, c=C["purple"], label="Co → Co²⁺ (anodic)")
a.semilogx(cr.cathodic_current(E, CU["Ec"], CU["i0c"], CU["bc"], CU["ilim"]), E, c=C["blue"], label="H₂O₂ reduction (cathodic)")
for m, col, nm in ((CU, C["orange"], "Cu"), (CO, C["purple"], "Co")):
    Ec, ic = cr.corrosion_point(m)
    a.plot(ic, Ec, "o", c=col, ms=5)
    a.annotate(f"{nm}: $E_{{corr}}$, $i_{{corr}}$", (ic, Ec), xytext=(4, -11 if nm == "Co" else 4),
               textcoords="offset points", fontsize=7, color=col)
a.set_xlim(1e-7, 1e-2); a.set_ylim(-0.35, 0.45)
a.set_xlabel("|current density| (A/cm²)"); a.set_ylabel("potential (V, arbitrary reference)")
a.set_title("(a) Evans diagram", fontsize=9); a.legend(fontsize=6.5, loc="upper left", frameon=False)

ratio = np.logspace(-2, 3, 40)
rate = [cr.penetration_rate_nm_min(cr.galvanic_couple(CU, r, CO, 1.0)[2], *M_CO) for r in ratio]
free = cr.penetration_rate_nm_min(cr.corrosion_point(CO)[1], *M_CO)
b.loglog(ratio, rate, c=C["purple"], label="Co coupled to Cu")
b.axhline(free, ls="--", c=C["gray"], lw=1, label="Co alone")
b.axvline(33, ls=":", c=C["red"], lw=1)
b.text(40, 13, "3 nm liner beside\n100 nm line", fontsize=6.5, color=C["red"])
b.set_xlabel("area ratio $A_{Cu}/A_{Co}$"); b.set_ylabel("Co dissolution (nm/min)")
b.set_title("(b) small anode, large cathode", fontsize=9); b.legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("/home/claude/evans_galvanic_raw.svg"); plt.close(fig)
full_minify_pipeline("/home/claude/evans_galvanic_raw.svg", "/home/claude/cu_evans_galvanic.svg")

# ------------------------------------------------------------ figure 2
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2))
H = np.linspace(5, 200, 60)
a.plot(H, mc.outer_fringe_cap(2.0, H, 3.9), c=C["blue"], label="outer fringe, oxide spacer (k = 3.9)")
a.plot(H, mc.outer_fringe_cap(2.0, H, 7.0), c=C["purple"], label="outer fringe, nitride spacer (k = 7)")
Hn = [20, 50, 90, 140, 200]
a.plot(Hn, [mc.outer_fringe_fd(2.0, h) for h in Hn], "o", c=C["blue"], ms=4, label="2-D Laplace, sidewall flux (k = 3.9)")
for Lov, ls in ((5, ":"), (10, "--")):
    a.axhline(mc.overlap_cap(2.0, Lov), ls=ls, c=C["gray"], lw=1)
    a.text(203, mc.overlap_cap(2.0, Lov), f"overlap,\n$L_{{ov}}$={Lov} nm", fontsize=6, va="center", color=C["gray"])
a.set_xlim(0, 250); a.set_ylim(0, 0.2)
a.set_xlabel("gate height H (nm)"); a.set_ylabel("C per side (fF/µm)")
a.set_title("(a) outer fringe vs overlap ($t_{ox}$ = 2 nm)", fontsize=9); a.legend(fontsize=6, frameon=False, loc="lower right")

hs = np.linspace(0.5, 6, 60)
base = mc.overlap_cap(2.0, 8.0) + mc.outer_fringe_cap(2.0, 90.0, 5.5)
for cov, col, lab in ((0.4, C["orange"], "round plugs (40 % coverage)"), (1.0, C["red"], "trench contact (full width)")):
    cc = mc.gate_contact_cap(hs, 1.0, 5.0, cov)
    b.plot(hs, 100 * cc / (cc + base), c=col, label=lab)
cc40 = mc.gate_contact_cap(90, 30, 5.0, 0.4)
b.plot(3, 100 * cc40 / (cc40 + base), "o", c=C["orange"], ms=5)
b.annotate("40 nm planar\nestimate", (3, 100 * cc40 / (cc40 + base)), xytext=(6, -16), textcoords="offset points", fontsize=6.5)
b.set_xlabel("gate height / gate-to-contact spacing, H/s"); b.set_ylabel("contact share of extrinsic $C_{gd}$ (%)")
b.set_ylim(0, 75); b.set_title("(b) gate-to-contact share", fontsize=9); b.legend(fontsize=6.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("/home/claude/cgd_raw.svg"); plt.close(fig)
full_minify_pipeline("/home/claude/cgd_raw.svg", "/home/claude/mosfet_cgd_components.svg")

if __name__ == "__main__":
    print("base overlap+fringe fF/um:", round(base, 4), " contact(40nm):", round(cc40, 4),
          " share:", round(100 * cc40 / (cc40 + base), 1), "%")
    for r in (3, 4.5, 6):
        for cov in (0.4, 1.0):
            cc = mc.gate_contact_cap(r, 1.0, 5.0, cov)
            print(f"H/s={r} cov={cov}: share {100*cc/(cc+base):.1f}%")
