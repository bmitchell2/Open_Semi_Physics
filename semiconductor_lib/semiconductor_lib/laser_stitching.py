"""Laser-anneal stitching: scan simulation, sheet-resistance response,
spectral (FFT) and phase-folding analysis, and four-point-probe spatial
transfer function.

Coordinates: x is the step (stitch) axis in mm. A line beam whose long
axis (width W) lies along x is scanned along y, then stepped by pitch p in
x. With p = W/2 every point is crossed by two passes.

Models (illustrative, 1-D across the stitch axis)
-------------------------------------------------
* Long-axis beam profile: flat top of width W with erf edges (rms edge
  width sigma), optional linear tilt and sinusoidal ripple on the top.
* Peak temperature of pass n at x:  T_n(x) = T_base + dT * P_n * prof(x - c_n)
  (peak rise proportional to absorbed power, a linear-heating assumption).
* Activation from an Arrhenius budget per pass, b_n = exp(-Ea/k (1/T_n -
  1/T_peak)), combined as B = b_max + gamma * (sum of the other b_n):
      f(x) = 1 - exp(-x1 * B),   Rs(x) = Rs_full / f(x)
  gamma (second-pass efficacy) interpolates between a pure peak-temperature
  (max) response, gamma = 0, and a fully cumulative budget, gamma = 1. A
  re-heat that deactivates would correspond to gamma < 0. x1 sets how
  saturated a single full pass is (x1 = 1.6 gives f = 0.80).
* Four-point probe sensitivity (reciprocity): S = J_AD . J_BC, where J_AD is
  the current density for unit current A->D and J_BC the same for B->C, in
  an infinite uniform sheet. For uniform Rs, integral of S over the plane
  equals ln2/pi, which recovers V/I = Rs ln2/pi.  The 1-D kernel across the
  stripes is S integrated along the stripe direction; its Fourier transform
  is the probe transfer function H(f).
"""
import numpy as np

K_B_EV = 8.617333262e-5


# ------------------------------------------------------------ beam and scan
def beam_profile(u, W=7.6, sigma=0.35, tilt=0.0, ripple=0.0, ripple_period=1.5):
    """Normalized long-axis intensity at offset u (mm) from beam centre.
    tilt: fractional intensity change from centre to the +W/2 edge.
    ripple: fractional cosine ripple amplitude on the flat top."""
    from scipy.special import erf
    u = np.asarray(u, float)
    s = np.sqrt(2.0) * max(sigma, 1e-6)
    top = 0.5 * (erf((u + W / 2) / s) - erf((u - W / 2) / s))
    mod = 1.0 + tilt * u / (W / 2) + ripple * np.cos(2 * np.pi * u / ripple_period)
    return top * mod


def pass_centres(x_min, x_max, pitch=3.8, step_errors=None, x0=0.0, W=7.6):
    """Centres of all passes that touch [x_min, x_max]. step_errors is a
    callable n -> lateral offset (mm) added to pass n."""
    n_lo = int(np.floor((x_min - W - x0) / pitch)) - 1
    n_hi = int(np.ceil((x_max + W - x0) / pitch)) + 1
    n = np.arange(n_lo, n_hi + 1)
    c = x0 + n * pitch
    if step_errors is not None:
        c = c + np.array([step_errors(k) for k in n])
    return n, c


def simulate_rs(x, pitch=3.8, W=7.6, sigma=0.35, tilt=0.0, ripple=0.0,
                ripple_period=1.5, power=None, step_errors=None, x0=0.0,
                T_base=400.0, T_peak=1300.0, Ea=2.0, x1=1.6, gamma=0.15,
                Rs_full=500.0, return_T=False):
    """True sheet resistance (ohm/sq) along x (mm).
    power: callable n -> relative power of pass n (default 1).
    Defaults give a partially saturated, temperature-sensitive monitor
    (about -0.33 %/K at the peak)."""
    x = np.asarray(x, float)
    n, c = pass_centres(x.min(), x.max(), pitch, step_errors, x0, W)
    Tpk = T_peak + 273.15
    bs = []
    Tmax = np.full_like(x, T_base)
    for k, ck in zip(n, c):
        P = 1.0 if power is None else power(k)
        prof = beam_profile(x - ck, W, sigma, tilt, ripple, ripple_period)
        if not np.any(prof > 1e-4):
            continue
        T = T_base + (T_peak - T_base) * P * prof + 273.15
        bs.append(np.exp(-Ea / K_B_EV * (1.0 / T - 1.0 / Tpk)))
        Tmax = np.maximum(Tmax, T - 273.15)
    bs = np.array(bs)
    bmax = bs.max(axis=0)
    B = bmax + gamma * (bs.sum(axis=0) - bmax)
    f = 1.0 - np.exp(-x1 * B)
    Rs = Rs_full / f
    return (Rs, Tmax) if return_T else Rs


def rs_temperature_sensitivity(T_peak=1300.0, Ea=2.0, x1=1.6, gamma=0.15,
                               n_pass=2):
    """d ln Rs / d T_peak (1/K) when all n_pass passes shift together."""
    Tk = T_peak + 273.15
    x = x1 * (1.0 + gamma * (n_pass - 1))
    dlnf_dlnB = x * np.exp(-x) / (1.0 - np.exp(-x))
    return -(Ea / (K_B_EV * Tk**2)) * dlnf_dlnB


# ----------------------------------------------------------------- spectra
def amplitude_spectrum(y, dx, window="none", detrend=True):
    """One-sided amplitude spectrum scaled to zero-to-peak sine amplitude.
    Returns f (1/mm), complex coefficients A (amplitude * e^{i phase})."""
    y = np.asarray(y, float)
    if detrend:
        y = y - np.mean(y)
    N = len(y)
    if window == "hann":
        w = np.hanning(N)
    else:
        w = np.ones(N)
    cg = w.mean()
    F = np.fft.rfft(y * w)
    A = 2.0 * F / (N * cg)
    A[0] /= 2.0
    f = np.fft.rfftfreq(N, dx)
    return f, A


def harmonic_amplitudes(x, y, period, n_harm=8):
    """Least-squares complex amplitudes of harmonics k/period, k=1..n_harm.
    Works on non-uniform sampling and non-integer record lengths."""
    x = np.asarray(x, float)
    y = np.asarray(y, float) - np.mean(y)
    cols = []
    for k in range(1, n_harm + 1):
        w = 2 * np.pi * k / period
        cols += [np.cos(w * x), np.sin(w * x)]
    M = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    a, b = coef[0::2], coef[1::2]
    return a - 1j * b  # amplitude * exp(i phase) for a cos(wx + phase)


def odd_even_power(x, y, pitch, n_harm=12):
    """Split periodic power at fundamental 1/(2 pitch): even harmonics are
    the normal p-periodic stitch, odd harmonics are the odd/even pass
    difference. Returns (rms_even, rms_odd)."""
    A = harmonic_amplitudes(x, y, 2 * pitch, n_harm)
    k = np.arange(1, n_harm + 1)
    p = np.abs(A) ** 2 / 2.0
    return np.sqrt(p[k % 2 == 0].sum()), np.sqrt(p[k % 2 == 1].sum())


# ----------------------------------------------------------------- folding
def detrend_moving(x, y, window):
    """Subtract a moving average of length `window` (mm) on uniform x.
    A boxcar of length L nulls every harmonic of 1/L, so use L = 2 * pitch:
    that removes all p- and 2p-periodic content from the trend estimate.
    L = pitch would null the p series but pass about 64% of a 2p
    component into the trend, cutting the odd/even signature to ~36%.
    Returns (residual, valid) where valid excludes the ends."""
    dx = x[1] - x[0]
    n = max(int(round(window / dx)), 1)
    k = np.ones(n) / n
    trend = np.convolve(y, k, mode="same")
    edge = n // 2 + 1
    valid = np.zeros_like(y, bool)
    valid[edge:-edge] = True
    return y - trend, valid


def fold(x, y, period, nbins=38, x0=0.0):
    """Phase-fold y on period; returns bin centres (mm), mean, std err."""
    ph = np.mod(np.asarray(x) - x0, period)
    edges = np.linspace(0, period, nbins + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, nbins - 1)
    cnt = np.bincount(idx, minlength=nbins)
    s = np.bincount(idx, y, nbins)
    s2 = np.bincount(idx, np.asarray(y) ** 2, nbins)
    m = s / np.maximum(cnt, 1)
    var = s2 / np.maximum(cnt, 1) - m**2
    se = np.sqrt(np.maximum(var, 0) / np.maximum(cnt, 1))
    return 0.5 * (edges[1:] + edges[:-1]), m, se


def fold_metrics(x, y, pitch, nbins=38):
    """Peak-to-peak and RMS of the p-fold, and max |half difference| of the
    2p fold (odd vs even passes)."""
    _, m, _ = fold(x, y, pitch, nbins)
    _, m2, _ = fold(x, y, 2 * pitch, 2 * nbins)
    half = m2[:nbins] - m2[nbins:]
    return {"p2p": float(m.max() - m.min()), "rms": float(np.std(m)),
            "odd_even_max": float(np.max(np.abs(half)))}


# ------------------------------------------------------- 4PP spatial kernel
def fpp_sensitivity_2d(X, Y, s=1.0):
    """Normalized collinear 4PP sensitivity S(x,y) (1/mm^2) for pins at
    -1.5s, -0.5s, 0.5s, 1.5s on the x axis. Integral over plane -> 1."""
    pins = np.array([-1.5, -0.5, 0.5, 1.5]) * s

    def J(xa, xb):
        dxa, dxb = X - xa, X - xb
        ra2 = dxa**2 + Y**2
        rb2 = dxb**2 + Y**2
        jx = (dxa / ra2 - dxb / rb2) / (2 * np.pi)
        jy = (Y / ra2 - Y / rb2) / (2 * np.pi)
        return jx, jy

    ax, ay = J(pins[0], pins[3])
    bx, by = J(pins[1], pins[2])
    return (ax * bx + ay * by) / (np.log(2) / np.pi)


def fpp_kernel_1d(s=1.0, orientation="perpendicular", dx=None, half_span=None):
    """1-D sensitivity kernel k(x) across the stripes, normalized to unit
    area. orientation 'perpendicular': probe row across the stripes;
    'parallel': probe row along the stripes. Returns x (mm), k (1/mm)."""
    dx = dx or s / 40.0
    half_span = half_span or 12 * s
    xs = np.arange(-half_span, half_span + dx / 2, dx) + dx * 0.37
    t = np.sinh(np.linspace(-np.arcsinh(400 * s / (0.01 * s)),
                            np.arcsinh(400 * s / (0.01 * s)), 6001)) * 0.01 * s
    if orientation == "perpendicular":
        X, Y = np.meshgrid(xs, t, indexing="ij")
    else:
        Y, X = np.meshgrid(xs, t, indexing="ij")  # integrate along probe axis
    S = fpp_sensitivity_2d(X, Y, s)
    k = np.trapezoid(S, t, axis=1)
    k = k / np.trapezoid(k, xs)
    return xs, k


def transfer_function(xk, k, f):
    """Real-valued transfer function H(f) = sum k cos(2 pi f x) dx for a
    symmetric kernel (sign changes are retained)."""
    xk = np.asarray(xk)
    f = np.atleast_1d(np.asarray(f, float))
    return np.array([np.trapezoid(k * np.cos(2 * np.pi * fi * xk), xk) for fi in f])


def apply_probe(x, rs, xk, k):
    """Convolve a uniformly sampled Rs line with the probe kernel."""
    dx = x[1] - x[0]
    xk_r = np.arange(xk.min(), xk.max() + dx / 2, dx)
    kr = np.interp(xk_r, xk, k)
    kr = kr / kr.sum()
    pad = len(kr)
    ypad = np.pad(rs, pad, mode="edge")
    return np.convolve(ypad, kr[::-1], mode="same")[pad:-pad]
