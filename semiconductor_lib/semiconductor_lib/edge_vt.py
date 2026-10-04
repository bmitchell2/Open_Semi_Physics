"""
Threshold-voltage shift of a long MOSFET caused by short edge segments
(regions next to source and drain whose local V_T differs from the channel
centre), for example from implant-damage-driven boron redistribution.

Series-segment model. The channel is split into a centre of local threshold
V_T0 and n_sides edge segments of width w and local threshold V_T0 + dVT_edge.
Each segment is assumed long compared with the 2-D scale length so its local
V_T is well defined, and V_DS is small.

Subthreshold, constant-current criterion: the segments conduct like series
resistors with conductance proportional to exp((V_G - V_T,i)/(n kT/q)), so

    dVT_eff = n (kT/q) ln[ (L - n_sides w + n_sides w exp(dVT_edge/(n kT/q))) / L ]

A raised edge (dVT_edge > 0) is weighted exponentially; a lowered edge
(dVT_edge < 0) can at most remove its own length from the channel, giving
dVT_eff -> n (kT/q) ln(1 - n_sides w / L).

Strong inversion (linear region): segment resistance ~ w_i / (V_G - V_T,i);
to first order in dVT_edge / (V_G - V_T0) the extracted shift is the
length-weighted average  dVT_eff = (n_sides w / L) dVT_edge.
"""
import numpy as np
from .constants import k_B, q


def edge_vt_shift_subthreshold(L, w, dvt_edge, n=1.3, T=300.0, n_sides=2):
    """Constant-current (subthreshold) V_T shift of a channel of length L (any
    length unit, same as w) with n_sides edge segments of width w whose local
    threshold differs by dvt_edge (V). n is the subthreshold ideality factor."""
    L = np.asarray(L, dtype=float)
    if np.any(n_sides * w >= L):
        raise ValueError("edge segments must be shorter than the channel")
    nphit = n * k_B * T / q
    frac = n_sides * w / L
    return nphit * np.log(1.0 - frac + frac * np.exp(dvt_edge / nphit))


def edge_vt_shift_strong_inversion(L, w, dvt_edge, n_sides=2):
    """First-order linear-region (strong inversion) shift: length-weighted
    average of the edge threshold change."""
    L = np.asarray(L, dtype=float)
    return n_sides * w / L * dvt_edge
