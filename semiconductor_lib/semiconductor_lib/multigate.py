"""
Short-channel electrostatic scale length ("natural length") for planar and
multigate MOSFETs.

Backs the Semiconductor Notes page "FinFET (Multigate MOSFET Electrostatics)".

Model
-----
lambda = sqrt( eps_si * t_si * t_ox / (N * eps_ox) )
(Colinge, Solid-State Electron. 48, 897 (2004); Ferain, Colinge & Colinge,
Nature 479, 310 (2011)). N is the "equivalent number of gates": 1 for a
single gate, 2 for a double gate (FinFET sidewalls), about 3 for tri-gate,
about 4 for gate-all-around. For a planar bulk device t_si is replaced by
the maximum depletion depth (Yan, Ourmazd & Lee, IEEE TED 39, 1704, 1992).
Short-channel effects are small when L_g exceeds roughly 5-10 lambda.
This is a first-order parabolic-potential result; it omits quantum
confinement, fringing through spacers, and the (1 + eps_ox t_si/4 eps_si t_ox)
correction of the more exact double-gate forms.

Units: cm.
"""
import numpy as np

from .constants import eps_si, eps_ox


def natural_length(t_si, t_ox, n_gates=1.0, eps_s=eps_si, eps_i=eps_ox):
    """Scale length lambda (cm). t_si: body thickness / fin width or bulk
    depletion depth (cm); t_ox: (equivalent) oxide thickness (cm)."""
    return np.sqrt(eps_s * np.asarray(t_si, float) * np.asarray(t_ox, float)
                   / (n_gates * eps_i))


def min_gate_length(t_si, t_ox, n_gates=1.0, ratio=5.0):
    """Gate length (cm) at which L/lambda equals `ratio` (default 5)."""
    return ratio * natural_length(t_si, t_ox, n_gates)
