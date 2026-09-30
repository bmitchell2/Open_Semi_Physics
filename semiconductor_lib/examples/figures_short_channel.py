"""Generates the figures on the Semiconductor Notes pages "Short-Channel Effects
(Scale Length, V_T Roll-off, DIBL and Junction Depth)" and "Silicon-on-Insulator
(SOI, PD-SOI and FD-SOI)". Run from an empty directory; writes minified SVGs.
All curves are analytical (Yau charge sharing); cross-sections are schematic."""
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from semiconductor_lib import short_channel as sc
from semiconductor_lib.mosfet import cox_from_tox
q=1.602176634e-19
NA=2e18; tox=2.0e-7; Cox=cox_from_tox(tox); Wd=sc.depletion_width_at_threshold(NA)

# ---- Figure 1: Yau charge-sharing roll-off ----
fig,ax=plt.subplots(1,2,figsize=(8.6,3.4))
L=np.linspace(25e-7,300e-7,40)
for xj,c in [(10,COLORS['blue']),(20,COLORS['purple']),(40,COLORS['red'])]:
    ax[0].plot(L*1e7,1e3*sc.yau_charge_sharing_dvt(NA,Cox,L,xj*1e-7),color=c,label=f'xj = {xj} nm')
ax[0].set_xlabel('Channel length L (nm)'); ax[0].set_ylabel('VT reduction ΔVT (mV)')
ax[0].set_title('(a) Roll-off vs L', fontsize=10); ax[0].legend(frameon=False); ax[0].set_xlim(25,300); ax[0].set_ylim(0,300)
r=np.logspace(-2,2,40); xj=r*Wd
d=1e3*sc.yau_charge_sharing_dvt(NA,Cox,40e-7,xj)
ax[1].loglog(r,d,color='k',label='Yau model')
pre=q*NA*Wd/Cox/40e-7
ax[1].loglog(r[r<1],1e3*pre*np.sqrt(2*Wd*xj[r<1]),'--',color=COLORS['blue'],label='shallow: ∝ √xj')
ax[1].loglog(r[r>1],1e3*pre*Wd*np.ones((r>1).sum()),'--',color=COLORS['red'],label='deep: saturates')
ax[1].set_xlabel('xj / Wd'); ax[1].set_ylabel('ΔVT at L = 40 nm (mV)')
ax[1].set_title('(b) Dependence on junction depth', fontsize=10); ax[1].legend(frameon=False,fontsize=8)
fig.text(0.5,-0.02,'Analytical charge-sharing model (Yau): NA = 2e18 cm⁻³, EOT = 2 nm, Wd = 25 nm. Illustrative, not measured; linear in 1/L so it overstates roll-off at short L.',ha='center',fontsize=7.5)
fig.tight_layout(); fig.savefig('raw1.svg',bbox_inches='tight'); plt.close(fig)

# ---- Figure 2: cross-section, shallow vs deep junction ----
def xsec(ax,xj,title):
    Lc=40; Wdn=25; tox_d=3
    ax.add_patch(Rectangle((-20,-70),Lc+40,70,color='#f2f2f2',zorder=0))
    ax.add_patch(Rectangle((0,0),Lc,tox_d,color='#c9d1d9')); ax.add_patch(Rectangle((0,tox_d),Lc,10,color='#57606a'))
    ax.text(Lc/2,tox_d+5,'Gate',ha='center',va='center',color='w',fontsize=8)
    for x0,lab in [(-20,'S'),(Lc,'D')]:
        ax.add_patch(Rectangle((x0,-xj),20,xj,color='#8fb3f0')); ax.text(x0+10,-xj/2,lab+' (n+)',ha='center',va='center',fontsize=8)
    ax.plot([-20,Lc+20],[-Wdn,-Wdn],':',color='k',lw=0.8); ax.text(Lc+19,-Wdn-4,'depletion edge Wd',ha='right',fontsize=7)
    # shared charge wedges (charge controlled by S/D, Yau trapezoid picture)
    for side in [0,1]:
        xe=0 if side==0 else Lc; sgn=1 if side==0 else -1
        r=xj*(np.sqrt(1+2*Wdn/xj)-1)
        ax.add_patch(Polygon([[xe,0],[xe+sgn*r,-Wdn],[xe,-Wdn]],color=COLORS['orange'],alpha=0.45,lw=0))
    ax.add_patch(Rectangle((0,-Wdn),Lc,Wdn,fill=False,ec=COLORS['green'],lw=1))
    ax.add_patch(FancyArrowPatch((Lc,-min(xj,Wdn)*0.7),(Lc*0.35,-min(xj,Wdn)*0.7),arrowstyle='->',mutation_scale=9,color=COLORS['red'],lw=1.2))
    ax.text(Lc*0.62,-min(xj,Wdn)*0.7-4,'drain field',fontsize=7,color=COLORS['red'],ha='center')
    ax.set_xlim(-20,Lc+20); ax.set_ylim(-60,16); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title,fontsize=9)
fig,ax=plt.subplots(1,2,figsize=(8.2,3.6))
xsec(ax[0],10,'Shallow junction, xj = 10 nm'); xsec(ax[1],40,'Deep junction, xj = 40 nm')
fig.text(0.5,-0.04,'Tan wedges: depletion charge imaged by S/D instead of the gate (charge-sharing wedges, width from the Yau model).\nThe drain sidewall facing the channel grows with xj, so drain-to-barrier coupling grows too. Schematic, 1:1 scale, L = 40 nm.',ha='center',fontsize=7.5,linespacing=1.6)
fig.savefig('raw2.svg',bbox_inches='tight'); plt.close(fig)

# ---- Figure 3: bulk vs FDSOI cross-sections ----
fig,ax=plt.subplots(1,2,figsize=(8.2,3.2))
def gate(a,Lc=30):
    a.add_patch(Rectangle((0,0),Lc,2,color='#c9d1d9')); a.add_patch(Rectangle((0,2),Lc,10,color='#57606a')); a.text(Lc/2,7,'Gate',color='w',ha='center',va='center',fontsize=8)
a=ax[0]; a.add_patch(Rectangle((-20,-60),70,60,color='#f2f2f2')); gate(a)
for x0 in [-20,30]: a.add_patch(Rectangle((x0,-15),20,15,color='#8fb3f0'))
a.plot([-20,50],[-22,-22],':',color='k',lw=.8); a.text(49,-27,'depletion edge',ha='right',fontsize=7)
a.annotate('',xy=(5,-18),xytext=(30,-18),arrowprops=dict(arrowstyle='->',color=COLORS['red'],lw=1.2)); a.text(17,-24,'sub-surface\nleakage path',fontsize=7,color=COLORS['red'],ha='center',va='top')
a.set_title('Bulk planar',fontsize=9)
a=ax[1]; a.add_patch(Rectangle((-20,-60),70,33,color='#f2f2f2')); a.add_patch(Rectangle((-20,-27),70,20,color='#d0e8d0')); a.text(15,-17,'BOX (25 nm)',ha='center',fontsize=8)
a.add_patch(Rectangle((0,-7),30,7,color='#fff3c4')); gate(a)
for x0 in [-20,30]: a.add_patch(Rectangle((x0,-7),20,7,color='#8fb3f0')); a.add_patch(Rectangle((x0,0),20,6,color='#b8cff5'))
a.text(15,-3.5,'Si 7 nm',ha='center',va='center',fontsize=7); a.text(-10,3,'raised S/D',ha='center',fontsize=6); a.text(15,-45,'substrate / back gate',ha='center',fontsize=8)
a.set_title('UTBB FDSOI (28 nm-class)',fontsize=9)
for a in ax: a.set_xlim(-20,50); a.set_ylim(-60,14); a.set_aspect('equal'); a.axis('off')
fig.text(0.5,0.0,'Schematic, approximately to scale. In FDSOI the film thickness replaces both xj and Wd: no silicon lies far from the gate.',ha='center',fontsize=7.5)
fig.savefig('raw3.svg',bbox_inches='tight'); plt.close(fig)
import os
for i,n in [(1,'sce_vt_rolloff_xj.svg'),(2,'sce_xsec_shallow_vs_deep.svg'),(3,'bulk_vs_fdsoi_xsec.svg')]:
    full_minify_pipeline(f'raw{i}.svg',n); print(n, os.path.getsize(n))
