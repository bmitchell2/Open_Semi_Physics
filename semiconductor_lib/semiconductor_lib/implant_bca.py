"""
Minimal vectorized binary-collision-approximation (BCA, TRIM-style) Monte
Carlo for ion implantation into AMORPHOUS multilayer targets, e.g. a thin
screen oxide on silicon.

Physics: ZBL universal screened potential with exact (tabulated) CM
scattering angles; fixed free-flight path N^-1/3 with one collision per
step (impact parameter up to p_max = (pi N^(2/3))^-1/2); continuous
Lindhard-Scharff electronic stopping with Bragg's rule for compounds,
multiplied by ke_scale (default 1.5, calibrated so that 100 keV B, P and
As ranges in Si land within about 5-10% of tabulated values: B ~300 nm,
P ~120 nm, As ~58 nm). No crystal structure, so NO channeling; no
sputtering, dose-dependent surface oxidation or damage accumulation.

Units: energy eV, length Angstrom, density atoms/A^3.
"""
import numpy as np
A0=0.529177; E2=14.3998
def zbl_phi(x):
    return (0.18175*np.exp(-3.1998*x)+0.50986*np.exp(-0.94229*x)
            +0.28022*np.exp(-0.4029*x)+0.028171*np.exp(-0.20162*x))
def zbl_dphi(x):
    return (-3.1998*0.18175*np.exp(-3.1998*x)-0.94229*0.50986*np.exp(-0.94229*x)
            -0.4029*0.28022*np.exp(-0.4029*x)-0.20162*0.028171*np.exp(-0.20162*x))
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

_TABLE = {}
_GLX, _GLW = np.polynomial.legendre.leggauss(64)
_GS = (_GLX + 1) / 2
_GW = _GLW / 2


def theta_exact(eps, b):
    """Exact CM scattering angle (rad) for the ZBL universal potential.

    eps: reduced energy; b: reduced impact parameter p/a. Evaluates the
    classical scattering integral with the substitution u = 1 - s^2 that
    removes the inverse-square-root singularity at the turning point."""
    if b == 0:
        return np.pi
    g = lambda r: 1 - zbl_phi(r) / r / eps - (b / r) ** 2
    R = brentq(g, 1e-8, max(b, 1e-3) + 60, xtol=1e-14)
    u = 1 - _GS ** 2
    gv = np.maximum(g(R / u), 1e-300)
    return np.pi - 2 * np.sum(_GW * (b / R) / np.sqrt(gv) * 2 * _GS)


def _table():
    if "interp" not in _TABLE:
        le = np.linspace(-5, 4, 181)
        bb = np.concatenate([[0.0], np.logspace(-3, np.log10(40), 220)])
        T = np.array([[theta_exact(10 ** l, b) for b in bb] for l in le])
        _TABLE.update(le=le, b=bb, interp=RegularGridInterpolator((le, bb), T))
    return _TABLE


def theta_cm(eps, b):
    """Tabulated exact ZBL CM scattering angle (vectorized)."""
    t = _table()
    le = np.clip(np.log10(eps), t["le"][0], t["le"][-1])
    bb = np.clip(b, 0, t["b"][-1])
    th = t["interp"](np.column_stack([le, bb]))
    return np.where(np.asarray(b) >= t["b"][-1], 0.0, np.clip(th, 0, np.pi))


def lss_k(Z1,M1,Z2):
    """Lindhard-Scharff electronic stopping coefficient, eV*A^2/sqrt(eV)."""
    return 1.212*Z1**(7/6)*Z2/((Z1**(2/3)+Z2**(2/3))**1.5*np.sqrt(M1))
SI=dict(N=0.04994,species=[(14,28.086,1.0)])
def sio2(density=2.20):
    Nmol=density/60.08*6.02214e23*1e-24  # molecules/A^3
    return dict(N=3*Nmol,species=[(14,28.086,1/3),(8,15.999,2/3)])
def run(Z1,M1,E0,layers,n=4000,tilt_deg=0.0,Ecut=5.0,ke_scale=1.5,seed=1):
    """layers: list of (thickness_A, material); last layer semi-infinite (thickness None).
    Returns final depths (A) of stopped ions (negative=backscattered -> nan)."""
    rng=np.random.default_rng(seed)
    bounds=np.cumsum([l[0] for l in layers[:-1]]) if len(layers)>1 else np.array([])
    E=np.full(n,float(E0)); z=np.zeros(n); 
    th=np.deg2rad(tilt_deg); cz=np.full(n,np.cos(th)); cx=np.full(n,np.sin(th)); cy=np.zeros(n)
    alive=np.ones(n,bool); final=np.full(n,np.nan)
    mats=[l[1] for l in layers]

    def matidx(zz):
        return np.searchsorted(bounds,zz,side='right') if len(bounds) else np.zeros_like(zz,dtype=int)
    for step in range(200000):
        idx=np.nonzero(alive)[0]
        if idx.size==0: break
        m=matidx(z[idx])
        Nloc=np.array([mats[k]['N'] for k in range(len(mats))])[m]
        L=Nloc**(-1/3)
        # move
        z[idx]+=L*cz[idx]
        # electronic loss over path (Bragg rule)
        Se=np.zeros(idx.size)
        for k,mat in enumerate(mats):
            sel=m==k
            if not sel.any(): continue
            ke=sum(f*lss_k(Z1,M1,Z2) for Z2,M2,f in mat['species'])*ke_scale
            Se[sel]=ke*np.sqrt(E[idx][sel])
        E[idx]-=Nloc*L*Se
        # choose target atom
        Z2=np.empty(idx.size); M2=np.empty(idx.size)
        for k,mat in enumerate(mats):
            sel=np.nonzero(m==k)[0]
            if sel.size==0: continue
            fr=np.array([s[2] for s in mat['species']]); pick=rng.choice(len(fr),size=sel.size,p=fr)
            Z2[sel]=np.array([s[0] for s in mat['species']])[pick]; M2[sel]=np.array([s[1] for s in mat['species']])[pick]
        a=0.8854*A0/(Z1**0.23+Z2**0.23)
        Ec=np.maximum(E[idx],1e-3)
        eps=a*M2*Ec/(Z1*Z2*E2*(M1+M2))
        pmax=1/np.sqrt(np.pi*Nloc**(2/3))
        p=pmax*np.sqrt(rng.random(idx.size))
        theta=theta_cm(eps,p/a)
        gm=4*M1*M2/(M1+M2)**2
        T=gm*Ec*np.sin(theta/2)**2
        E[idx]=Ec-T
        psi=np.arctan2(np.sin(theta),np.cos(theta)+M1/M2)
        phi=2*np.pi*rng.random(idx.size)
        # rotate direction
        ux,uy,uz=cx[idx],cy[idx],cz[idx]
        sp,cp=np.sin(psi),np.cos(psi)
        sf,cf=np.sin(phi),np.cos(phi)
        den=np.sqrt(np.maximum(1-uz**2,1e-12))
        big=den>1e-6
        nx=np.where(big,sp*(ux*uz*cf-uy*sf)/den+ux*cp,sp*cf)
        ny=np.where(big,sp*(uy*uz*cf+ux*sf)/den+uy*cp,sp*sf)
        nz=np.where(big,-sp*cf*den+uz*cp,np.sign(uz)*cp)
        nrm=np.sqrt(nx*nx+ny*ny+nz*nz)
        cx[idx],cy[idx],cz[idx]=nx/nrm,ny/nrm,nz/nrm
        # termination
        back=z[idx]<0
        stop=(E[idx]<Ecut)&~back
        final[idx[stop]]=z[idx[stop]]
        alive[idx[back|stop]]=False
    return final


def screen_oxide_split(Z1, M1, E0, t_ox_A, n=20000, tilt_deg=7.0, density=2.20,
                       ke_scale=1.5, seed=7):
    """Implant through a screen oxide of thickness t_ox_A (Angstrom) on Si.

    Returns dict with fractions backscattered / stopped in oxide / in Si, and
    the mean depth and straggle (nm) of the ions in Si, measured from the
    oxide/Si interface."""
    layers = [(None, SI)] if t_ox_A == 0 else [(t_ox_A, sio2(density)), (None, SI)]
    d = run(Z1, M1, E0, layers, n=n, tilt_deg=tilt_deg, ke_scale=ke_scale, seed=seed)
    back = np.isnan(d).mean()
    ox = ((d < t_ox_A) & ~np.isnan(d)).mean()
    si = d[d >= t_ox_A] - t_ox_A
    return dict(back=back, oxide=ox, silicon=1 - back - ox,
                Rp_nm=si.mean() / 10, dRp_nm=si.std() / 10)
