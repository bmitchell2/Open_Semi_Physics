"""
Low-stress mobility change of silicon MOSFET channels from the bulk
piezoresistance coefficients.

Backs the Semiconductor Notes page "Strain-Enhanced Mobility in Silicon".

Linear piezoresistance (valid for |stress| up to roughly a few hundred MPa;
at higher stress the response saturates and inversion-layer quantization
changes the coefficients):
    d(mu)/mu ~= -(pi_L * sigma_L + pi_T * sigma_T)
sigma > 0 is tensile, sigma < 0 compressive (Pa). Coefficients for a
<110> channel on a (001) wafer, bulk Si at 300 K, in 1e-11 /Pa
(Smith, Phys. Rev. 94, 42 (1954), as tabulated by Thompson et al.,
IEEE TED 51, 1790 (2004)):
    electrons: pi_L = -31.6, pi_T = -17.6
    holes:     pi_L = +71.8, pi_T = -66.3
"""
PI_110 = {
    "n": {"L": -31.6e-11, "T": -17.6e-11},
    "p": {"L": 71.8e-11, "T": -66.3e-11},
}


def mobility_change(carrier, sigma_L_Pa=0.0, sigma_T_Pa=0.0):
    """Fractional mobility change d(mu)/mu for a <110>/(001) channel."""
    c = PI_110[carrier]
    return -(c["L"] * sigma_L_Pa + c["T"] * sigma_T_Pa)
