"""Screen-oxide dose retention and range shift (amorphous BCA model).

Generates screen_oxide_dose_retention.svg for the Semiconductor Notes page
'Implant Screen Oxide'. Model: semiconductor_lib.implant_bca (amorphous
target, no channeling), 7 deg tilt, SiO2 density 2.20 g/cm^3."""
import numpy as np
import matplotlib.pyplot as plt
from semiconductor_lib import implant_bca as bca
from semiconductor_lib.plotting import apply_style, full_minify_pipeline

apply_style()
cases = [("B 0.5 keV", 5, 11.009, 500), ("B 1 keV", 5, 11.009, 1000), ("B 2 keV", 5, 11.009, 2000),
         ("As 2 keV", 33, 74.92, 2000), ("As 5 keV", 33, 74.92, 5000)]
t = np.arange(0, 45, 5)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
for lab, Z, M, E in cases:
    r = [bca.screen_oxide_split(Z, M, E, tt, n=8000) for tt in t]
    si = np.array([x["silicon"] for x in r]); rp = np.array([x["Rp_nm"] for x in r])
    a1.plot(t, 100 * si / si[0], marker="o", ms=3, label=lab)
    a2.plot(t, rp - rp[0], marker="o", ms=3, label=lab)
for ax in (a1, a2):
    ax.axvspan(10, 30, color="0.9", zorder=0)
    ax.set_xlabel("Screen oxide thickness (Å)")
a1.set_ylabel("Dose reaching Si (% of bare-Si value)")
a1.set_ylim(40, 102); a1.legend(fontsize=7, loc="lower left")
a2.plot(t, -t / 10, "k--", lw=0.8, label="shift = t_ox")
a2.set_ylabel("Change in mean depth in Si (nm)")
a2.legend(fontsize=7, loc="lower left")
a1.set_title("(a) Dose retained in Si", fontsize=9); a2.set_title("(b) Profile shift toward surface", fontsize=9)
fig.tight_layout()
fig.savefig("screen_oxide_raw.svg")
full_minify_pipeline("screen_oxide_raw.svg", "screen_oxide_dose_retention.svg")
