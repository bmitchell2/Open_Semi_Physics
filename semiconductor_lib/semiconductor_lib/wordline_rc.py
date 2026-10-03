"""
Distributed-RC delay of a long resistive word line (e.g., a 3D NAND
gate-replacement word line), and resistivity scaling with layer thinning.

Backs the Semiconductor Notes page "3D NAND Flash Process Integration".

Uniform distributed RC line driven at one end, open at the other:
    t_50% ~= 0.38 * R_total * C_total          (Elmore-based, Sakurai 1983)
    R_total = rho * L / (w * t),   C_total = c' * L
so delay grows as L^2: doubling word-line length quadruples delay.
Units: SI.
"""


def wordline_resistance(rho_ohm_m, length_m, width_m, thickness_m):
    return rho_ohm_m * length_m / (width_m * thickness_m)


def distributed_rc_delay(R_total, C_total):
    """50 % step-response delay of a distributed RC line (s)."""
    return 0.38 * R_total * C_total
