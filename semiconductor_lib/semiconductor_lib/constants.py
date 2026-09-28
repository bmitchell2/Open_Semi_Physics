"""
Physical constants and standard material parameters used across the
semiconductor_lib package. Values match the conventions used throughout
the Semiconductor Notes knowledge base (Sze & Ng, "Physics of
Semiconductor Devices," 3rd ed.).

All quantities are in cgs-practical semiconductor units unless noted:
  lengths     : cm
  potentials  : V
  doping      : cm^-3
  capacitance : F/cm^2
"""

q = 1.602176634e-19        # elementary charge, C
k_B = 1.380649e-23         # Boltzmann constant, J/K
eps0 = 8.8541878128e-14    # vacuum permittivity, F/cm

eps_si = 11.7 * eps0       # silicon relative permittivity
eps_ox = 3.9 * eps0        # SiO2 relative permittivity
eps_hfo2 = 25.0 * eps0     # HfO2 (representative high-k value)
eps_si3n4 = 9.0 * eps0     # Si3N4

ni300 = 1.0e10             # intrinsic carrier concentration, cm^-3, at 300 K


def thermal_voltage(T=300.0):
    """kT/q in volts, at temperature T (Kelvin)."""
    return k_B * T / q
