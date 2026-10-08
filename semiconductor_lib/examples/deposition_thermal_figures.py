"""Figures for the Semiconductor Notes CVD, reactive-sputtering and gettering
pages (2026-10-07). Verified: Berg fold flows ~1.95 sccm (metallic to
poisoned) and ~0.93 sccm (return) at S = 0.1 m^3/s; Fe solubility equals
1e12 cm^-3 near 760 C and 1e10 cm^-3 near 635 C."""
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import numpy as np, matplotlib.pyplot as plt
from semiconductor_lib import deposition as dp, thermal_budget as tb


def save(fig, name):
    fig.savefig(name+"_raw.svg", bbox_inches="tight"); plt.close(fig)
    print(name, full_minify_pipeline(name+"_raw.svg", name+".svg"))

# 1 Grove model, Arrhenius
T = np.linspace(500, 1200, 120)+273.15
fig, ax = plt.subplots(figsize=(6,3.8))
for hg, lab, c in [(1.0, "higher pressure: low $h_g$", COLORS["red"]),
                   (50.0, "low pressure: $h_g$ \u00d750", COLORS["blue"])]:
    G = dp.grove_growth_rate(T, ks0=3e8, Ea_eV=1.6, hg=hg)
    ax.semilogy(1000/T, G, color=c, label=lab)
    ax.axhline(hg, color=c, ls=":", lw=0.8)
ks = 3e8*np.exp(-1.6/(8.617e-5*T))
ax.semilogy(1000/T, ks, color=COLORS["gray"], ls="--", lw=0.8, label="$k_s$ alone (slope $-E_a/k$)")
ax.set_ylim(1e-3, 200); ax.set_xlabel("1000/T (K$^{-1}$)"); ax.set_ylabel("growth rate (normalized)")
ax.text(1.18, 0.003, "reaction-limited", fontsize=8); ax.text(0.70, 1.6, "transport-limited", fontsize=8, color=COLORS["red"])
ax2 = ax.secondary_xaxis("top", functions=(lambda x: 1000/np.maximum(x,1e-3)-273.15, lambda t: 1000/(t+273.15)))
ax2.set_xlabel("T (\u00b0C)"); ax2.set_xticks([500,600,700,800,900,1000,1200]); ax.legend(fontsize=7, loc="lower left"); ax.set_title("Grove model, $E_a$ = 1.6 eV (illustrative)", fontsize=9)
save(fig, "cvd_grove_arrhenius")

# 2 conformality
AR = np.linspace(0, 30, 200)
fig, ax = plt.subplots(figsize=(6,3.6))
for s, c in [(1e-1, COLORS["red"]), (1e-2, COLORS["orange"]), (1e-3, COLORS["green"]), (1e-4, COLORS["blue"])]:
    ax.plot(AR, dp.via_bottom_to_top(AR, s), color=c, label=f"s = {s:.0e}")
ax.axhline(1.0, color=COLORS["purple"], ls="--", lw=1, label="ALD (saturated, ideal)")
ax.set_xlabel("via aspect ratio L/d"); ax.set_ylabel("bottom/top reactant flux")
ax.set_ylim(0, 1.08); ax.legend(fontsize=7, loc="upper right"); ax.set_title("Knudsen-regime depletion in a via: 1/cosh(\u221a(3s)\u00b7AR)", fontsize=9)
save(fig, "cvd_via_conformality")

# 3 Berg hysteresis
P = np.logspace(-5, 0, 800)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3))
for S, c, lab in [(0.1, COLORS["red"], "S = 0.1 m\u00b3/s"), (10.0, COLORS["blue"], "S = 10 m\u00b3/s")]:
    r = dp.berg_reactive_sputter(P, S=S)
    a1.loglog(r["Q_sccm"], P, color=c, label=lab)
    a2.semilogx(r["Q_sccm"], r["rate"]/r["rate"][0], color=c, label=lab)
a1.set_xlabel("N$_2$ supply flow (sccm)"); a1.set_ylabel("N$_2$ partial pressure (Pa)")
a2.set_xlabel("N$_2$ supply flow (sccm)"); a2.set_ylabel("sputter rate / metallic rate")
for a in (a1, a2): a.legend(fontsize=7); a.set_xlim(0.3, 200)
r = dp.berg_reactive_sputter(P, S=0.1); Q = r["Q_sccm"]; i = np.argmax(np.diff(Q) < 0); j = i + np.argmax(np.diff(Q[i:]) > 0)
a1.annotate("", xy=(Q[i]*1.02, P[j+5]*3), xytext=(Q[i]*1.02, P[i]), arrowprops=dict(arrowstyle="->", color=COLORS["red"]))
a1.annotate("", xy=(Q[j]*0.98, P[i]*0.5), xytext=(Q[j]*0.98, P[j]), arrowprops=dict(arrowstyle="->", color=COLORS["red"]))
fig.suptitle("Berg model, Ti target in N$_2$ (illustrative parameters)", fontsize=9); fig.tight_layout()
save(fig, "reactive_sputter_hysteresis")
print("fold flows sccm", Q[i], Q[j])

# 4 Fe solubility / supersaturation
Tc = np.linspace(500, 1200, 200)
fig, ax = plt.subplots(figsize=(6,3.6))
ax.semilogy(Tc, tb.fe_solubility(Tc), color=COLORS["blue"], label="Fe solubility in Si (Istratov 1999 fit)")
for N, c in [(1e12, COLORS["red"]), (1e10, COLORS["orange"])]:
    ax.axhline(N, color=c, ls="--", lw=1, label=f"Fe contamination {N:.0e} cm$^{{-3}}$")
    Tx = Tc[np.argmin(abs(tb.fe_solubility(Tc)-N))]; ax.axvline(Tx, color=c, ls=":", lw=0.8)
    print("crossing", N, Tx)
ax.fill_between(Tc, 1e8, np.minimum(tb.fe_solubility(Tc),1e12), where=tb.fe_solubility(Tc)<1e12, color=COLORS["red"], alpha=0.08)
ax.text(560, 3e10, "supersaturated\n(at 10$^{12}$ cm$^{-3}$)", fontsize=8, color=COLORS["red"])
ax.set_xlabel("temperature (\u00b0C)"); ax.set_ylabel("concentration (cm$^{-3}$)"); ax.set_ylim(1e8, 1e16); ax.legend(fontsize=7, loc="upper left")
save(fig, "fe_solubility_supersaturation")
