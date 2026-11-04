"""
MOSFET gate capacitance components per unit gate width (planar device).

Cgd (or Cgs) = intrinsic channel part + overlap + outer fringe
               + inner fringe (only without inversion) + gate-to-contact.

* Intrinsic: long-channel Meyer partition of W*L*Cox (linear: 1/2 to each
  of source and drain; saturation: 2/3 to source, ~0 to drain; off: 0).
* Overlap: parallel plate through the gate dielectric over the
  extension, Cox*L_ov.
* Outer fringe (gate sidewall -> drain surface through the spacer):
  conformal-mapping result of Shrivastava and Fitzpatrick (IEEE TED 29,
  1870, 1982), C_of = (2*eps/pi)*ln(1 + H/t_ox).  BSIM3/4 use the same
  form for the default CF with a fixed 0.4 um gate height.
* Gate-to-contact: parallel-plate estimate eps*H/s times the fraction of
  the width covered by contacts (round plugs < 1, trench contacts ~ 1).
  This ignores fringing around the plugs and is therefore a lower bound.

outer_fringe_fd() is an independent 2-D finite-difference Laplace solve
of the gate-corner geometry, used to cross-check the analytic formula.

Inputs in nm; outputs in fF per um of gate width.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS0_F_PER_NM = 8.8541878128e-21          # F/nm
_TO_FF_PER_UM = 1e15 * 1e3                 # F/nm  -> fF/um


def cox_fF_um2(t_ox_nm, k=3.9):
    """Gate oxide capacitance per area, fF/um^2."""
    return k * EPS0_F_PER_NM / t_ox_nm * 1e15 * 1e6


def overlap_cap(t_ox_nm, L_ov_nm, k=3.9):
    """Overlap capacitance per width, fF/um: Cox * L_ov."""
    return cox_fF_um2(t_ox_nm, k) * L_ov_nm * 1e-3


def outer_fringe_cap(t_ox_nm, H_nm, k_spacer=3.9):
    """Outer fringe per width, fF/um: (2 eps/pi) ln(1 + H/t_ox)."""
    return (2.0 * k_spacer * EPS0_F_PER_NM / np.pi
            * np.log1p(np.asarray(H_nm, float) / t_ox_nm) * _TO_FF_PER_UM)


def gate_contact_cap(H_nm, s_nm, k=5.0, coverage=1.0):
    """Gate-sidewall-to-contact capacitance per width, fF/um (parallel
    plate, k*eps0*H/s, scaled by contact coverage of the width)."""
    return (k * EPS0_F_PER_NM * np.asarray(H_nm, float)
            / np.asarray(s_nm, float) * coverage * _TO_FF_PER_UM)


def meyer_partition(region):
    """Fractions (f_gs, f_gd) of W*L*Cox assigned to Cgs and Cgd in the
    long-channel Meyer model: 'linear' (Vds -> 0), 'saturation', 'off'."""
    return {"linear": (0.5, 0.5), "saturation": (2.0 / 3.0, 0.0),
            "off": (0.0, 0.0)}[region]


# ---------------------------------------------------- numerical cross-check
def _axis(lo, hi, h_min, ratio=1.08, h_max=None, anchors=()):
    """Monotonic grid between lo and hi, fine near anchors, geometric away."""
    pts = {lo, hi, *anchors}
    for a in [lo, *anchors]:
        x, h = a, h_min
        while x < hi:
            x += h
            pts.add(min(x, hi))
            h = min(h * ratio, h_max or 1e9)
    for a in [hi, *anchors]:
        x, h = a, h_min
        while x > lo:
            x -= h
            pts.add(max(x, lo))
            h = min(h * ratio, h_max or 1e9)
    g = np.array(sorted(pts))
    keep = np.concatenate([[True], np.diff(g) > 1e-9 * (hi - lo)])
    return g[keep]


def outer_fringe_fd(t_ox_nm, H_nm, k=3.9, extent_nm=3000.0, h_min=None,
                    quantity="sidewall"):
    """Outer fringe per width (fF/um) from a 2-D finite-volume Laplace solve.

    Uniform dielectric k.  Grounded silicon surface at y = 0; gate is a
    perfect conductor at 1 V filling x <= 0, t_ox <= y <= t_ox + H, and
    extending extent_nm to the left; all outer boundaries are zero-flux.

    quantity = "sidewall": flux leaving the gate sidewall (x = 0) into
        x > 0.  This is the quantity the analytic Shrivastava-Fitzpatrick
        formula approximates; it exceeds the analytic value by 14 % (H = 200 nm)
        to 27 % (H = 20 nm) at t_ox = 2 nm (the analytic form is lower because it
        omits the field concentrated at the bottom corner).
    quantity = "plane": total charge on the silicon surface at x > 0.  In
        this empty domain the gate top's flux has nowhere else to go and
        also lands there, giving 1.6-2.6x the sidewall value.  In a real
        device contacts and metal above intercept that flux, so it appears
        as gate-to-contact/metal capacitance instead.
    """
    h_min = h_min or t_ox_nm / 8.0
    tol = 1e-9
    xs = _axis(-extent_nm, extent_nm, h_min, anchors=(0.0,))
    ys = _axis(0.0, extent_nm, h_min, anchors=(t_ox_nm, t_ox_nm + H_nm))
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    gate = (X <= tol) & (Y >= t_ox_nm - tol) & (Y <= t_ox_nm + H_nm + tol)
    fixed = gate | np.isclose(Y, 0.0)
    V = np.where(gate, 1.0, 0.0)
    free = ~fixed
    idx = -np.ones((nx, ny), int)
    idx[free] = np.arange(free.sum())
    wx = np.zeros(nx); d = np.diff(xs); wx[:-1] += d / 2; wx[1:] += d / 2
    wy = np.zeros(ny); d = np.diff(ys); wy[:-1] += d / 2; wy[1:] += d / 2
    rows, cols, vals = [], [], []
    b = np.zeros(free.sum())
    for i, j in zip(*np.nonzero(free)):
        n = idx[i, j]; diag = 0.0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if not (0 <= ii < nx and 0 <= jj < ny):
                continue                       # zero-flux outer boundary
            g = (wy[j] / abs(xs[ii] - xs[i]) if di
                 else wx[i] / abs(ys[jj] - ys[j]))
            diag += g
            if fixed[ii, jj]:
                b[n] += g * V[ii, jj]
            else:
                rows.append(n); cols.append(idx[ii, jj]); vals.append(-g)
        rows.append(n); cols.append(n); vals.append(diag)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(free.sum(),) * 2)
    V[free] = spla.spsolve(A, b)
    scale = k * EPS0_F_PER_NM * _TO_FF_PER_UM
    if quantity == "sidewall":
        j0 = int(np.argmin(np.abs(xs)))
        js = (ys >= t_ox_nm - tol) & (ys <= t_ox_nm + H_nm + tol)
        flux = ((V[j0, js] - V[j0 + 1, js]) / (xs[j0 + 1] - xs[j0]) * wy[js]).sum()
    elif quantity == "plane":
        jx = xs > tol
        flux = ((V[jx, 1] - V[jx, 0]) / (ys[1] - ys[0]) * wx[jx]).sum()
    else:
        raise ValueError("quantity must be 'sidewall' or 'plane'")
    return float(flux * scale)
