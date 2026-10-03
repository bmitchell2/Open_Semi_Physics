"""
Random-defect die-yield models and the monolithic-versus-chiplet tradeoff.

Backs the Semiconductor Notes page "Chiplets and 2.5D Advanced Packaging".

Models (A = die area in cm^2, D0 = random killer-defect density in cm^-2):
  Poisson            Y = exp(-A D0)
  Murphy             Y = ((1 - exp(-A D0)) / (A D0))^2
  Negative binomial  Y = (1 + A D0 / alpha)^(-alpha)   (Stapper; alpha = clustering)
Poisson is the most pessimistic for large dies; clustering (small alpha)
raises the yield of large dies because defects pile up on a few dies.

Silicon cost per good product, ignoring edge loss and packaging:
  monolithic : A / Y(A)
  n chiplets : n * (A/n) / Y(A/n)       (known-good-die tested before assembly)
Plus an optional area overhead per chiplet for die-to-die PHYs and a
packaging/assembly yield Y_pkg applied to the whole assembly.
"""
import numpy as np


def yield_poisson(A, D0):
    return np.exp(-np.asarray(A) * D0)


def yield_murphy(A, D0):
    x = np.asarray(A, float) * D0
    with np.errstate(invalid="ignore", divide="ignore"):
        y = ((1 - np.exp(-x)) / x)**2
    return np.where(x == 0, 1.0, y)


def yield_negbin(A, D0, alpha=3.0):
    return (1 + np.asarray(A) * D0 / alpha)**(-alpha)


def good_silicon_area_per_product(A_total, n_chiplets, D0, model=yield_negbin,
                                  d2d_overhead=0.0, pkg_yield=1.0, **kw):
    """Wafer area (cm^2) consumed per good product.
    d2d_overhead: fractional area added to each chiplet for die-to-die I/O.
    pkg_yield: assembly yield of the multi-die package (1.0 for monolithic)."""
    if n_chiplets == 1:
        return A_total / model(A_total, D0, **kw)
    a = A_total / n_chiplets * (1 + d2d_overhead)
    return n_chiplets * a / model(a, D0, **kw) / pkg_yield
