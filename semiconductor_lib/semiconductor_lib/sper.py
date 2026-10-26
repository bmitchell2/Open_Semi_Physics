"""
Solid-phase epitaxial regrowth (SPER) of ion-amorphized silicon.

Intrinsic (100) velocity v = v0 exp(-Ea/kT) with v0 = 4.64e8 cm/s and
Ea = 2.70 eV (Roth et al., thick a-Si layers; reproduced at 2.68 eV in
later buried-layer work). Orientation factors are approximate ratios to
(100) from Csepregi et al., J. Appl. Phys. 49, 3906 (1978): (110) about
2.5x slower, (111) about 20x slower. Dopant (B, P, As) enhancement and
F/O/C retardation are not included.
Valid roughly 450-1300 C; at the highest temperatures random nucleation
in the amorphous layer can compete with epitaxy, especially with As or F.
"""
import numpy as np

K_B_EV = 8.617333262e-5
ORIENTATION_FACTOR = {"100": 1.0, "110": 0.4, "111": 0.05}


def sper_velocity_cm_s(T_C, orientation="100", v0=4.64e8, Ea=2.70):
    """Intrinsic SPER interface velocity (cm/s)."""
    T = np.asarray(T_C, float) + 273.15
    return ORIENTATION_FACTOR[orientation] * v0 * np.exp(-Ea / (K_B_EV * T))


def regrowth_time_s(thickness_nm, T_C, orientation="100", **kw):
    """Time (s) to regrow an amorphous layer of given thickness at constant T."""
    return thickness_nm * 1e-7 / sper_velocity_cm_s(T_C, orientation, **kw)
