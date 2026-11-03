"""
CMP scratch-source kinematics and shear-driven slurry agglomeration.

* Track, in wafer coordinates, of a scratch source fixed in the pad (exact
  rigid-body result) and of a free particle rolling between pad and wafer
  (idealized no-slip).  With matched head and platen speeds a
  pad-embedded source scratches circular arcs of radius d (the
  carrier-to-platen offset) and a rolling particle traces radius d/2.
* Ratio of shear-induced (orthokinetic) to Brownian (perikinetic)
  collision frequency, 4 eta G a^3 / kT (Smoluchowski), which also sets
  the scale of hydrodynamic work available to push pairs over a DLVO
  barrier.

Lengths in m, angular speeds in rad/s.  Uses cmp.rpm_to_rad_s for rpm
conversion.
"""
import numpy as np
from .constants import k_B as K_B


# ------------------------------------------------- scratch-source tracks
def _rot(theta, x, y):
    c, s = np.cos(theta), np.sin(theta)
    return c * x - s * y, s * x + c * y


def pad_point_track_wafer(p0, w_p, w_c, d, t):
    """Track, in wafer-fixed coordinates (m), of a point fixed on the pad.

    Platen axis at the origin, carrier centre at (d, 0) (m), platen and
    carrier rotating in the same sense at w_p and w_c (rad/s).  p0 = (x, y)
    is the pad point's lab position at t = 0 (m).  Exact rigid-body result

        w(t) = R[(w_p - w_c) t] p0 - R[-w_c t] (d, 0),

    the sum of two circular motions.  For w_p = w_c the track is a circle
    of radius d centred at p0: a pad-embedded particle scratches arcs whose
    radius equals the carrier-to-platen offset, wherever it sits on the
    pad.  Returns (x, y) arrays.
    """
    t = np.asarray(t, float)
    ax, ay = _rot((w_p - w_c) * t, p0[0], p0[1])
    bx, by = _rot(-w_c * t, d, 0.0)
    return ax - bx, ay - by


def rolling_particle_track_wafer(x0, w_p, w_c, d, t_end, n=4000):
    """Track (wafer coordinates, m) of a free particle that rolls without
    slip between pad and wafer.  Its centre moves at the mean of the two
    surface velocities, i.e. half the pad velocity seen in the wafer
    frame.  Idealized (no slip, no sticking).  Fixed-step RK4 from wafer
    position x0 = (x, y).  For w_p = w_c the track is a circle of radius
    d/2.  Returns (t, x, y)."""
    def f(tt, x, y):
        # lab position, pad and wafer velocities, relative velocity
        qx, qy = _rot(w_c * tt, x, y)
        qx += d
        vrx = -w_p * qy + w_c * qy
        vry = w_p * qx - w_c * (qx - d)
        ux, uy = _rot(-w_c * tt, vrx, vry)
        return 0.5 * ux, 0.5 * uy

    t = np.linspace(0.0, t_end, n)
    h = t[1] - t[0]
    xs = np.empty(n)
    ys = np.empty(n)
    xs[0], ys[0] = x0
    for i in range(n - 1):
        x, y, tt = xs[i], ys[i], t[i]
        k1 = f(tt, x, y)
        k2 = f(tt + h / 2, x + h / 2 * k1[0], y + h / 2 * k1[1])
        k3 = f(tt + h / 2, x + h / 2 * k2[0], y + h / 2 * k2[1])
        k4 = f(tt + h, x + h * k3[0], y + h * k3[1])
        xs[i + 1] = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        ys[i + 1] = y + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return t, xs, ys


def track_curvature(x, y):
    """Signed curvature (1/m) of a sampled planar track."""
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)


# ------------------------------------------------ shear-driven agglomeration
def ortho_to_peri_collision_ratio(a_m, shear_rate_s, eta_Pa_s=0.89e-3,
                                  T_K=298.15):
    """Ratio of shear-induced (orthokinetic) to Brownian (perikinetic)
    collision frequency for equal spheres of radius a_m in simple shear
    (Smoluchowski rate constants, no hydrodynamic or colloidal
    interactions):  beta_ortho/beta_peri = 4 eta G a^3 / (k_B T).
    Above 1, flow rather than diffusion brings particles together.  The
    same group, 4 eta G a^3, is also the order of the hydrodynamic work
    (force ~ eta G a^2 over distance ~ a) available to push a colliding
    pair together, in units of k_B T, so when it approaches the DLVO
    barrier height (tens of k_B T) shear can drive pairs over the barrier
    that Brownian collisions cannot cross."""
    a = np.asarray(a_m, float)
    return 4.0 * eta_Pa_s * shear_rate_s * a ** 3 / (K_B * T_K)
