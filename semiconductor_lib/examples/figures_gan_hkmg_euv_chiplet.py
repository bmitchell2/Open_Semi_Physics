"""Figures for the GaN HEMT, HKMG, EUV lithography and chiplet pages of Semiconductor Notes.
Run from a writable directory; writes *_raw.svg and minified *.svg."""
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib.pyplot as plt, numpy as np
from semiconductor_lib import gan_hemt as g, hkmg as h, euv_multilayer as e, yield_models as y
from semiconductor_lib.constants import q
C=[COLORS[k] for k in ("blue","red","purple","green")]
def save(fig,name):
    fig.tight_layout(); fig.savefig(f"{name}_raw.svg"); plt.close(fig)
    print(name, full_minify_pipeline(f"{name}_raw.svg", f"{name}.svg"))
# 1 GaN
fig,ax=plt.subplots(figsize=(5.6,3.6))
d=np.linspace(0.5,40,80)
for x,c in zip([0.15,0.25,0.30,0.35],C):
    ax.plot(d,[g.sheet_density(x,di)/1e13 for di in d],color=c,label=f"x = {x:.2f}")
    ax.axhline(g.sigma_pol(x)/q*1e-4/1e13,color=c,ls=":",lw=0.8)
ax.set_xlabel("AlGaN barrier thickness d (nm)"); ax.set_ylabel("2DEG density $n_s$ ($10^{13}$ cm$^{-2}$)")
ax.set_xlim(0,40); ax.set_ylim(0,2.1); ax.legend(frameon=False,fontsize=8,title="Al fraction",title_fontsize=8)
ax.set_title("Ambacher model, Ga-face, undoped; dotted = $\\sigma_{pol}/q$",fontsize=9)
save(fig,"gan_2deg_vs_barrier")
# 2 HKMG
fig,ax=plt.subplots(figsize=(5.6,3.6))
wf=np.linspace(3.9,5.3,57)
for N,ls in [(1e17,"--"),(5e17,"-")]:
    ax.plot(wf,h.vt_nmos(wf,N,1e-7),color=C[0],ls=ls,label=f"NMOS, NA={N:.0e} cm⁻³")
    ax.plot(wf,h.vt_pmos(wf,N,1e-7),color=C[1],ls=ls,label=f"PMOS, ND={N:.0e} cm⁻³")
for v,lab in [(4.05,"Si Ec"),(4.61,"midgap"),(5.17,"Si Ev")]:
    ax.axvline(v,color="gray",lw=0.7,ls=":"); ax.text(v+0.02,0.95,lab,fontsize=7,color="gray")
ax.axhline(0,color="k",lw=0.6)
ax.set_xlabel("Effective gate work function φm (eV)"); ax.set_ylabel("Long-channel VT (V)")
ax.set_ylim(-1.1,1.1); ax.legend(frameon=False,fontsize=7,loc="lower right")
ax.set_title("EOT 1 nm, classical, no oxide charge",fontsize=9)
save(fig,"hkmg_vt_vs_workfunction")
# 3 EUV
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.0,3.3))
lam=np.linspace(12.9,14.1,241)
for N,c in zip([10,20,40,60],C):
    a1.plot(lam,100*e.multilayer_reflectivity(lam,N,6.9),color=c,label=f"{N} bilayers")
a1.set_xlabel("Wavelength (nm)"); a1.set_ylabel("Reflectivity (%)"); a1.legend(frameon=False,fontsize=7)
a1.set_title("Ideal Mo/Si, period 6.9 nm",fontsize=9)
n=np.arange(1,13)
for R,c in zip([0.74,0.70,0.65],C):
    a2.semilogy(n,100*R**n,"o-",ms=3,color=c,label=f"R = {R:.2f}")
a2.set_xlabel("Number of mirrors"); a2.set_ylabel("Transmitted fraction (%)"); a2.legend(frameon=False,fontsize=7)
a2.set_title("Optical throughput R^N",fontsize=9)
save(fig,"euv_multilayer_reflectivity")
# 4 yield
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.0,3.3))
A=np.linspace(0.05,9,100); D0=0.1
a1.plot(A,100*y.yield_poisson(A,D0),color=C[0],label="Poisson")
a1.plot(A,100*y.yield_murphy(A,D0),color=C[1],label="Murphy")
a1.plot(A,100*y.yield_negbin(A,D0,3),color=C[2],label="Neg. binomial, α=3")
a1.axvline(8.58,color="gray",ls=":"); a1.text(8.1,60,"reticle\nlimit",fontsize=7,color="gray",ha="right")
a1.set_xlabel("Die area (cm²)"); a1.set_ylabel("Die yield (%)"); a1.legend(frameon=False,fontsize=7)
a1.set_title("D0 = 0.1 cm⁻²",fontsize=9)
ns=[1,2,4,8,16]
for ov,pk,c,lab in [(0,1,C[0],"no overhead"),(0.10,0.97,C[1],"10% D2D area, 97% assembly yield")]:
    v=[y.good_silicon_area_per_product(8,k,D0,d2d_overhead=ov if k>1 else 0,pkg_yield=pk if k>1 else 1) for k in ns]
    a2.plot(ns,v,"o-",ms=3,color=c,label=lab)
a2.axhline(8,color="gray",ls=":"); a2.set_xscale("log",base=2); a2.set_xticks(ns); a2.set_xticklabels(ns)
a2.set_xlabel("Number of chiplets (8 cm² total logic)"); a2.set_ylabel("Wafer area per good product (cm²)")
a2.legend(frameon=False,fontsize=7); a2.set_title("Neg. binomial, α=3",fontsize=9)
save(fig,"chiplet_yield")
