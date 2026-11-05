"""
Starting-wafer helpers: substrate resistivity and p/p+ epitaxial-wafer
boron up-diffusion.

resistivity_to_doping        : invert rho = 1/(q N mu(N)) for the majority
                               carrier, using the Caughey-Thomas mobility fit
                               already in semiconductor_lib.scattering.
substrate_updiffusion_profile: boron profile in the epi layer of a p/p+ wafer
                               after a thermal budget D*t (constant-D erfc
                               solution to a step at the epi/substrate
                               interface, both sides semi-infinite).
updiffusion_rise             : how far into the epi the profile reaches a
                               chosen threshold concentration.
usable_epi_thickness         : epi thickness left below a threshold
                               (as-grown thickness minus the rise).

Assumptions and limits (see the Notion page "Starting Wafers for CMOS"):
- Constant (intrinsic) boron diffusivity. At p+ substrate levels near
  1e19 cm^-3 boron is close to intrinsic at 1000-1100 C (n_i ~ 1e19), so the
  error is modest, but concentration-enhanced diffusion makes the real rise
  somewhat larger at lower temperatures. Use as an estimate, not a TCAD
  replacement.
- Ignores autodoping during epi growth (boron evaporated from the substrate
  and re-incorporated), which mainly softens the as-grown transition.
- Ignores oxidation-enhanced diffusion and the surface boundary condition,
  so it is valid while the rise stays well below the epi thickness.

Verified against: midpoint/limit properties of the erfc solution, the
sqrt(Dt) scaling of the rise, and the standard 300 K resistivity chart
(10 ohm-cm p-type ~ 1.3e15 cm^-3; 0.01 ohm-cm p-type ~ 8-9e18 cm^-3).
"""
import numpy as np
from scipy.special import erfc, erfcinv
from scipy.optimize import brentq

from .constants import q
from .scattering import caughey_thomas, CT_ELECTRONS, CT_HOLES


def resistivity_to_doping(rho_ohm_cm, carrier="p", T=300.0):
    """Net majority doping (cm^-3) that gives resistivity rho (ohm-cm),
    with the Caughey-Thomas mobility mu(N, T). Full ionization assumed."""
    params = CT_HOLES if carrier == "p" else CT_ELECTRONS

    def f(logN):
        N = 10.0 ** logN
        return 1.0 / (q * N * caughey_thomas(N, T, params)) - rho_ohm_cm

    return 10.0 ** brentq(f, 11.0, 21.5)


def substrate_updiffusion_profile(x_um, N_sub, N_epi, Dt_cm2):
    """Boron concentration (cm^-3) at depth x_um (um) above the original
    epi/substrate interface (positive into the epi), after budget Dt (cm^2).
    Dt = 0 returns the abrupt as-grown step."""
    x_cm = np.asarray(x_um, dtype=float) * 1e-4
    if Dt_cm2 <= 0:
        return np.where(x_cm > 0, N_epi, N_sub).astype(float)
    return N_epi + 0.5 * (N_sub - N_epi) * erfc(x_cm / (2.0 * np.sqrt(Dt_cm2)))


def updiffusion_rise(N_sub, N_epi, Dt_cm2, N_threshold):
    """Distance (um) into the epi at which the up-diffused boron falls to
    N_threshold (must lie between N_epi and N_sub/2 + N_epi/2)."""
    if not (N_epi < N_threshold < 0.5 * (N_sub + N_epi)):
        raise ValueError("threshold must lie between N_epi and the midpoint")
    arg = 2.0 * (N_threshold - N_epi) / (N_sub - N_epi)
    return 2.0 * np.sqrt(Dt_cm2) * erfcinv(arg) * 1e4


def usable_epi_thickness(t_epi_um, N_sub, N_epi, Dt_cm2, threshold_factor=2.0):
    """Epi thickness (um) whose boron stays below threshold_factor x N_epi.
    The default (twice the background) is a common definition of where the
    lightly doped layer 'ends'; pick the factor that matters for the device
    (for example, the doping that would shift a junction's breakdown)."""
    return t_epi_um - updiffusion_rise(N_sub, N_epi, Dt_cm2,
                                       threshold_factor * N_epi)
