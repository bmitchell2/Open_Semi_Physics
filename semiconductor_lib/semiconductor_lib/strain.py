"""
Low-stress mobility change of silicon MOSFET channels from the bulk
piezoresistance coefficients.

Backs the Semiconductor Notes page "Strain-Enhanced Mobility in Silicon".

Linear piezoresistance (valid for |stress| up to roughly a few hundred MPa;
at higher stress the response saturates and inversion-layer quantization
changes the coefficients):
    d(mu)/mu ~= -(pi_L * sigma_L + pi_T * sigma_T)
sigma > 0 is tensile, sigma < 0 compressive (Pa). Bulk Si at 300 K, in
1e-11 /Pa (Smith, Phys. Rev. 94, 42 (1954)).

<110> channel on a (001) wafer (as tabulated by Thompson et al.,
IEEE TED 51, 1790 (2004)):
    electrons: pi_L = -31.6, pi_T = -17.6
    holes:     pi_L = +71.8, pi_T = -66.3
<100> channel on a (001) wafer (45-degree rotated notch), where
pi_L = pi_11 and pi_T = pi_12:
    electrons: pi_L = -102.2, pi_T = +53.4
    holes:     pi_L = +6.6,   pi_T = -1.1
"""
PI_110 = {
    "n": {"L": -31.6e-11, "T": -17.6e-11},
    "p": {"L": 71.8e-11, "T": -66.3e-11},
}

PI_100 = {
    "n": {"L": -102.2e-11, "T": 53.4e-11},
    "p": {"L": 6.6e-11, "T": -1.1e-11},
}

_TABLES = {"110": PI_110, "100": PI_100}


def mobility_change(carrier, sigma_L_Pa=0.0, sigma_T_Pa=0.0, direction="110"):
    """Fractional mobility change d(mu)/mu for a channel along
    <direction> ("110" default, or "100") on a (001) wafer."""
    c = _TABLES[direction][carrier]
    return -(c["L"] * sigma_L_Pa + c["T"] * sigma_T_Pa)
