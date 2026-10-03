"""
Reflectivity of periodic Mo/Si EUV multilayer (distributed Bragg) mirrors.

Backs the Semiconductor Notes page "Extreme Ultraviolet (EUV) Lithography".

Model: characteristic-matrix (transfer-matrix) method for a stack of
homogeneous absorbing layers with ideal, abrupt, smooth interfaces,
s-polarisation, at angle theta from normal (default near normal).
Complex refractive index n~ = n + i k, values at 13.5 nm from the
CXRO (Henke) tables: Mo n=0.9238, k=0.00644; Si n=0.99888, k=0.00183.
Ideal stacks give a peak of about 74 % at 13.5 nm for ~40-60 bilayers,
the known theoretical limit (Montcalm et al., UCRL-JC-128289: ~75 %);
measured mirrors reach ~67-70 % because of interface intermixing and
roughness, which this model omits.

Also: optical throughput of a chain of N mirrors, T = R^N.
"""
import numpy as np

N_MO = 0.9238 + 0.00644j
N_SI = 0.99888 + 0.00183j


def multilayer_reflectivity(wavelength_nm, n_bilayers, period_nm=6.9, gamma=0.4,
                            n_top=N_SI, n_bot=N_MO, n_sub=N_SI, theta_deg=0.0):
    """Intensity reflectivity of [top/bottom] x n_bilayers on substrate.
    gamma = thickness fraction of the absorber (Mo) in each period.
    Top layer of each period is Si (spacer), bottom is Mo (absorber)."""
    lam = np.atleast_1d(np.asarray(wavelength_nm, float))
    th = np.deg2rad(theta_deg)
    d_mo, d_si = gamma * period_nm, (1 - gamma) * period_nm
    layers = [(n_top, d_si), (n_bot, d_mo)] * int(n_bilayers)
    R = np.empty_like(lam)
    for i, L in enumerate(lam):
        k0 = 2 * np.pi / L
        # s-pol admittance  eta = n cos(theta_j), cos from Snell with ambient n=1
        def cos_t(n):
            return np.sqrt(1 - (np.sin(th) / n)**2 + 0j)
        M = np.eye(2, dtype=complex)
        for n, d in layers:
            c = cos_t(n)
            delta = k0 * n * c * d
            eta = n * c
            Mj = np.array([[np.cos(delta), -1j * np.sin(delta) / eta],
                           [-1j * eta * np.sin(delta), np.cos(delta)]])
            M = M @ Mj
        eta0, etas = np.cos(th), n_sub * cos_t(n_sub)
        B, C = M @ np.array([1.0, etas])
        r = (eta0 * B - C) / (eta0 * B + C)
        R[i] = abs(r)**2
    return R if R.size > 1 else R[0]


def chain_throughput(R, n_mirrors):
    """Fraction of light surviving n_mirrors reflections at reflectivity R."""
    return R ** n_mirrors


def bragg_period(wavelength_nm, theta_deg=0.0, order=1):
    """First-order Bragg period, uncorrected for refraction: d = m lambda / (2 cos theta)."""
    return order * wavelength_nm / (2 * np.cos(np.deg2rad(theta_deg)))
