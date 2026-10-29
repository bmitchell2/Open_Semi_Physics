"""Independent cross-check of the 4PP transfer function H(f) in
semiconductor_lib.laser_stitching with a finite-difference resistor network.

Sheet Rs(x) = 1 + eps cos(2 pi x / Lambda), grid h = 0.05 mm, 30 x 30 mm
domain, pin pitch 1 mm, probe centred on a cosine crest. H = (V/V0 - 1)/eps.

Verified 2026-10-28: uniform sheet V0 pi/ln2 = 1.008 (finite-domain and
grid error 0.8%).
  Lambda (mm)    7.6     3.8     1.9
  perpendicular  0.637   0.483   0.659   (kernel: 0.635 0.488 0.669)
  parallel       0.622   0.252  -0.077   (kernel: 0.631 0.261 -0.072)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def fd_4pp(Lam, eps, orient, s=1.0, h=0.05, L=30.0):
    n = int(L / h) + 1
    xs = np.linspace(-L / 2, L / 2, n)
    X, _ = np.meshgrid(xs, xs, indexing="ij")
    Rs = 1 + eps * np.cos(2 * np.pi * X / Lam)
    idx = np.arange(n * n).reshape(n, n)
    gx = 1 / (0.5 * (Rs[1:, :] + Rs[:-1, :]))
    gy = 1 / (0.5 * (Rs[:, 1:] + Rs[:, :-1]))
    A = np.concatenate([idx[1:, :].ravel(), idx[:, 1:].ravel()])
    B = np.concatenate([idx[:-1, :].ravel(), idx[:, :-1].ravel()])
    G = np.concatenate([gx.ravel(), gy.ravel()])
    M = sp.coo_matrix((np.concatenate([G, G, -G, -G]),
                       (np.concatenate([A, B, A, B]), np.concatenate([A, B, B, A]))),
                      shape=(n * n, n * n)).tolil()
    M[0, :] = 0
    M[0, 0] = 1
    M = M.tocsr()
    k = [int(round((p + L / 2) / h)) for p in np.array([-1.5, -0.5, 0.5, 1.5]) * s]
    c = int(round(L / 2 / h))
    nodes = [idx[ki, c] for ki in k] if orient == "perpendicular" else [idx[c, ki] for ki in k]
    I = np.zeros(n * n)
    I[nodes[0]], I[nodes[3]] = 1, -1
    V = spl.spsolve(M, I)
    return V[nodes[1]] - V[nodes[2]]


if __name__ == "__main__":
    for o in ["perpendicular", "parallel"]:
        V0 = fd_4pp(3.8, 0, o)
        print(o, "V0*pi/ln2 =", round(V0 * np.pi / np.log(2), 3))
        for Lam in [7.6, 3.8, 1.9]:
            print("  Lambda", Lam, "H =", round((fd_4pp(Lam, 0.02, o) / V0 - 1) / 0.02, 3))
