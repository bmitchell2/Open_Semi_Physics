"""Figures for the Semiconductor Notes page on laser-anneal stitching
monitors. All data are model-generated (semiconductor_lib.laser_stitching),
not measurements. Run: python laser_stitching_figures.py OUTDIR
(set PNG=1 to also write PNG previews)."""
import sys, os
import numpy as np
from semiconductor_lib.plotting import apply_style, COLORS, full_minify_pipeline
apply_style()
import matplotlib
matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 8.5,
                            "axes.labelsize": 8, "legend.fontsize": 7,
                            "axes.spines.top": False, "axes.spines.right": False})
import matplotlib.pyplot as plt
import matplotlib.ticker
matplotlib.rcParams['ytick.minor.visible'] = False
from semiconductor_lib import laser_stitching as ls

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
os.makedirs(OUT, exist_ok=True)
P = 3.8
L = 39 * 2 * P            # whole number of 2p periods: no FFT leakage
XT = np.arange(0, L, 0.01)
DXM = 0.1
XM = np.arange(0, L, DXM)
rng = np.random.default_rng(7)
B, R, G, GR, PU, OR = (COLORS[k] for k in ["blue", "red", "green", "gray", "purple", "orange"])


def alt(d):
    return lambda n: 1 + d * (1 - 2 * (n % 2))


def measure(rs_true, noise=0.0015, bowl=0.02, xt=XT, xm=XM, seed=None):
    r = np.interp(xm, xt, rs_true)
    rr = np.random.default_rng(seed) if seed is not None else rng
    rad = (xm - L / 2) / (L / 2)
    return r * (1 + bowl * rad**2) * (1 + noise * rr.standard_normal(len(xm)))


def analyse(y, x=XM):
    res, valid = ls.detrend_moving(x, y, 2 * P)
    return x[valid], res[valid]


def save(fig, name):
    raw = os.path.join(OUT, name + "_raw.svg")
    fig.savefig(raw, bbox_inches="tight")
    if os.environ.get("PNG"): fig.savefig(os.path.join(OUT, name + ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    out = os.path.join(OUT, name + ".svg")
    full_minify_pipeline(raw, out)
    os.remove(raw)
    import re
    t = open(out).read()
    def _q(m):  # 0.1 pt precision inside attribute values only, never in text
        v = re.sub(r"(\d+\.\d)\d+", r"\1", m.group(0))
        return re.sub(r"\.0(?=[^\d])", "", v)
    t = re.sub(r'"[^"<>]*"', _q, t)
    t = re.sub(r' transform="rotate\(-0 [\d. -]+\)"', "", t)
    t = re.sub(r' font-family="[^"]*"', "", t)
    t = t.replace("*{stroke-linejoin: round; stroke-linecap: butt}",
                  "*{stroke-linejoin:round;stroke-linecap:butt}text{font-family:'DejaVu Sans',sans-serif}")
    t = re.sub(r'(<use [^>]*?) stroke="#000000" stroke-width="[\d.]+"', r"\1", t)
    open(out, "w").write(t)


def spectrum_ax(ax, y):
    """FFT of the full detrended record trimmed to whole 2p periods."""
    r, _ = ls.detrend_moving(XM, y, 2 * P)
    n0 = int(round(2 * P / DXM)); r = r[n0:-n0]   # still integer 2p periods
    f, A = ls.amplitude_spectrum(r, DXM)
    m = (f > 0.02) & (f < 1.1)
    ax.semilogy(f[m], np.abs(A[m]) + 1e-3, color=GR, lw=0.6)
    for k in range(1, 5):
        ax.axvline(k / P, color=B, lw=0.5, ls="--", alpha=0.6)
    for k in (1, 3, 5, 7):
        ax.axvline(k / (2 * P), color=R, lw=0.5, ls=":", alpha=0.8)
    ax.set_ylim(1e-2, 30)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("spatial frequency (1/mm)")
    ax.set_ylabel("|A| (Ω, 0-pk)")


def fold_ax(ax, y):
    x, r = analyse(y)
    c, m, _ = ls.fold(x, r, P, 38)
    ax.plot(np.r_[c, c + P], np.r_[m, m], color=B, lw=1.4, label="fold @3.8 (shown twice)")
    c2, m2, _ = ls.fold(x, r, 2 * P, 76)
    ax.plot(c2, m2, color=R, lw=0.9, label="fold @7.6")
    ax.axvline(P, color=GR, lw=0.4)
    ax.set_xlim(0, 2 * P)
    ax.set_xlabel("position in 7.6 mm cycle (mm)")
    ax.set_ylabel("ΔRs (Ω)")
    ax.legend(frameon=False, loc="best")


def panel3(rs_true, title, name, ylim_sec=None):
    y = measure(rs_true)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.25), constrained_layout=True)
    seg = (XM > 140) & (XM < 140 + 4 * P)
    st = (XT > 140) & (XT < 140 + 4 * P)
    axs[0].plot(XM[seg], y[seg], color=GR, lw=0.5, label="measured")
    axs[0].plot(XT[st][::5], rs_true[st][::5], color="k", lw=1.0, label="true")
    axs[0].set_xlabel("x (mm)"); axs[0].set_ylabel("Rs (Ω/sq)")
    axs[0].legend(frameon=False, loc="upper right")
    if ylim_sec: axs[0].set_ylim(*ylim_sec)
    spectrum_ax(axs[1], y)
    fold_ax(axs[2], y)
    fig.suptitle(title, fontsize=8.5)
    save(fig, name)


# ---------------------------------------------------------------- F1 mechanism
def f1():
    x = np.arange(124, 146, 0.01)
    fig, axs = plt.subplots(2, 1, figsize=(6.0, 3.6), sharex=True, constrained_layout=True,
                            gridspec_kw={"height_ratios": [1.3, 1]})
    n, c = ls.pass_centres(x.min(), x.max())
    profs = []
    j = 0
    for k, ck in zip(n, c):
        pr = ls.beam_profile(x - ck)
        if pr.max() > 0.01:
            profs.append(pr)
        if 128 < ck < 142:
            axs[0].plot(x, pr + 1.15 * j, color=(R if k % 2 else B), lw=1.1)
            axs[0].text(x.min() - 0.3, 1.15 * j + 0.45, f"pass {k - 33}", ha="right", fontsize=7,
                        color=(R if k % 2 else B))
            j += 1
    axs[0].set_xlim(x.min(), x.max())
    axs[0].set_yticks([]); axs[0].spines["left"].set_visible(False)
    axs[0].set_title("Long-axis profiles of successive passes (odd red, even blue), W = 7.6 mm, step 3.8 mm",
                     fontsize=7.5)
    second = np.sort(np.array(profs), axis=0)[-2]
    rs = ls.simulate_rs(x)
    axs[1].plot(x, rs, color="k", lw=1.2, label="Rs")
    ax2 = axs[1].twinx(); ax2.plot(x, second, color=OR, lw=0.9)
    ax2.set_ylabel("2nd-strongest pass", color=OR); ax2.set_ylim(0, 1.1)
    ax2.spines["top"].set_visible(False)
    for ck in c:
        if x.min() < ck < x.max():
            for a in axs: a.axvline(ck, color=GR, lw=0.4, ls="--")
    axs[1].set_ylabel("Rs (Ω/sq)"); axs[1].set_xlabel("x (mm); dashed = pass centres")
    save(fig, "f01_mechanism")


def f4():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3), constrained_layout=True)
    sig = np.linspace(0.05, 1.0, 14)
    x = XT[(XT > 100) & (XT < 100 + 4 * P)]
    for g, col in [(0.0, B), (0.15, G), (0.3, R)]:
        v = [ls.fold_metrics(x, ls.simulate_rs(x, sigma=s_, gamma=g), P)["p2p"] for s_ in sig]
        axs[0].plot(sig, v, "-o", ms=2.5, color=col, lw=1, label=f"γ = {g}")
    axs[0].set_xlabel("edge rms width σ (mm)"); axs[0].set_ylabel("p-fold peak-to-peak (Ω)")
    axs[0].legend(frameon=False)
    Ws = np.linspace(7.3, 7.9, 13)
    for s_, col in [(0.1, B), (0.35, G), (0.7, R)]:
        v = [ls.fold_metrics(x, ls.simulate_rs(x, W=w, sigma=s_), P)["p2p"] for w in Ws]
        axs[1].plot(Ws, v, "-o", ms=2.5, color=col, lw=1, label=f"σ = {s_} mm")
    axs[1].axvline(2 * P, color=GR, lw=0.5, ls="--")
    axs[1].set_xlabel("beam width W (mm), step fixed 3.8"); axs[1].set_ylabel("p-fold peak-to-peak (Ω)")
    axs[1].legend(frameon=False)
    save(fig, "f04_edge_width_sweeps")


def f7():
    x = XT[(XT > 60) & (XT < 60 + 20 * P)]
    cases = [("ideal", {}), ("3% tilt", dict(tilt=0.03)), ("1% odd/even", dict(power=alt(0.01))),
             ("150 µm alt. step", dict(step_errors=lambda n: 0.15 * (n % 2))),
             ("tilt + 0.5% odd/even", dict(tilt=0.03, power=alt(0.005)))]
    ev, od = [], []
    for _, kw in cases:
        e, o = ls.odd_even_power(x, ls.simulate_rs(x, **kw), P)
        ev.append(e); od.append(o)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1.4, 1]})
    i = np.arange(len(cases))
    axs[0].bar(i - 0.2, ev, 0.4, color=B, label="even harmonics of 1/7.6 (normal stitch)")
    axs[0].bar(i + 0.2, od, 0.4, color=R, label="odd harmonics (odd/even pass difference)")
    axs[0].set_xticks(i, [c[0] for c in cases], rotation=18, ha="right")
    axs[0].set_ylabel("RMS (Ω)"); axs[0].legend(frameon=False, loc="upper left")
    d = np.linspace(0, 0.02, 9)
    for s_, col in [(0.2, B), (0.35, G), (0.6, R)]:
        v = [ls.odd_even_power(x, ls.simulate_rs(x, sigma=s_, power=alt(dd)), P)[1] for dd in d]
        axs[1].plot(d * 200, v, "-o", ms=2.5, color=col, lw=1, label=f"σ = {s_} mm")
    axs[1].set_xlabel("odd/even power mismatch, peak-to-peak (%)")
    axs[1].set_ylabel("odd-harmonic RMS (Ω)"); axs[1].legend(frameon=False)
    save(fig, "f07_harmonic_split")


def f9():
    fig, ax = plt.subplots(figsize=(4.4, 2.4), constrained_layout=True)
    pops = [("fixed 0.5% odd/even\n(±25% wafer-to-wafer)", B, "fixed"),
            ("near-2-pass oscillation,\nrandom phase per wafer", R, "osc"),
            ("no asymmetry\n(noise only)", GR, "none")]
    for i, (lab, col, mode) in enumerate(pops):
        vals = []
        for w in range(20):
            ph = rng.uniform(0, 2 * np.pi)
            if mode == "fixed":
                a = 1 + 0.25 * rng.standard_normal()
                pw = (lambda a: (lambda n: 1 + 0.0025 * a * (1 - 2 * (n % 2))))(a)
            elif mode == "osc":
                pw = (lambda p0: (lambda n: 1 + 0.0025 * np.cos(np.pi * n * 0.9995 + p0)))(ph)
            else:
                pw = None
            y = measure(ls.simulate_rs(XT, power=pw), seed=1000 + w)
            xx, r = analyse(y)
            vals.append(ls.harmonic_amplitudes(xx, r, 2 * P, 1)[0].real)
        ax.plot(np.full(20, i) + rng.uniform(-0.15, 0.15, 20), vals, "o", ms=3.5, color=col)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(3), [p[0] for p in pops], fontsize=6.5)
    ax.set_ylabel("signed 1/7.6 mm amplitude (Ω)")
    ax.set_title("Wafer-to-wafer full-beam term, 20 wafers per case", fontsize=8)
    save(fig, "f09_w2w_signed")


def f10():
    rs = ls.simulate_rs(XT, power=alt(0.005), tilt=0.02)
    y = measure(rs)
    x, r = analyse(y)
    per = np.linspace(2.5, 9.0, 650)
    p2p = [np.ptp(ls.fold(x, r, q, 30)[1]) for q in per]
    yn = measure(np.full_like(XT, 600.0))
    xn, rn = analyse(yn)
    p2pn = [np.ptp(ls.fold(xn, rn, q, 30)[1]) for q in per]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1.6, 1]})
    axs[0].plot(per, p2p, color="k", lw=0.8, label="tilt 2% + 0.5% odd/even")
    axs[0].plot(per, p2pn, color=GR, lw=0.8, label="no stitch: noise floor")
    axs[0].set_xlabel("trial fold period (mm)"); axs[0].set_ylabel("folded peak-to-peak (Ω)")
    axs[0].legend(frameon=False)
    for q, col, lab in [(3.8, B, "fold @3.800"), (3.82, R, "fold @3.820 (0.5% error)")]:
        c, m, _ = ls.fold(x, r, q, 38)
        axs[1].plot(c, m, color=col, lw=1.2, label=lab)
    axs[1].set_xlabel("position in cycle (mm)"); axs[1].set_ylabel("ΔRs (Ω)")
    axs[1].legend(frameon=False, fontsize=6.5)
    save(fig, "f10_period_scan")


def f11():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), constrained_layout=True)
    f = np.linspace(0.0, 1.2, 121)
    for o, col, lab in [("perpendicular", B, "1 mm 4PP, row ⊥ stripes"),
                        ("parallel", R, "1 mm 4PP, row ∥ stripes")]:
        xk, k = ls.fpp_kernel_1d(1.0, o)
        m = np.abs(xk) < 4
        axs[0].plot(xk[m][::4], k[m][::4], color=col, lw=1, label=lab)
        axs[1].plot(f, ls.transfer_function(xk, k, f), color=col, lw=1.2, label=lab)
    xk, k = ls.fpp_kernel_1d(0.01, "perpendicular")
    axs[1].plot(f, ls.transfer_function(xk, k, f), color=G, lw=1.2, label="10 µm micro-4PP")
    fd = {"perpendicular": [0.637, 0.483, 0.659], "parallel": [0.622, 0.252, -0.077]}
    for o, col in [("perpendicular", B), ("parallel", R)]:
        axs[1].plot([1 / 7.6, 1 / 3.8, 1 / 1.9], fd[o], "s", mfc="none", color=col, ms=5)
    axs[1].plot([], [], "s", mfc="none", color="k", ms=5, label="finite-difference check")
    for kk in range(1, 5):
        axs[1].axvline(kk / P, color=GR, lw=0.4, ls="--")
    axs[1].axhline(0, color="k", lw=0.4)
    axs[0].set_xlabel("x across stripes (mm)"); axs[0].set_ylabel("kernel k(x) (1/mm)")
    axs[0].set_ylim(-0.3, 1.5); axs[0].legend(frameon=False)
    axs[1].set_xlabel("spatial frequency (1/mm)"); axs[1].set_ylabel("H(f)")
    axs[1].legend(frameon=False, fontsize=6.5)
    save(fig, "f11_probe_mtf")


def f12():
    x = XT
    rs = ls.simulate_rs(x, tilt=0.03, power=alt(0.005))
    seg = (x > 100) & (x < 100 + 20 * P)
    xs, ts = x[seg], rs[seg]
    c, mt, _ = ls.fold(xs, ts - ts.mean(), 2 * P, 152)
    fig, ax = plt.subplots(figsize=(4.6, 2.4), constrained_layout=True)
    ax.plot(c, mt, color="k", lw=1.6, label="true")
    for o, col, lab in [("perpendicular", B, "1 mm 4PP ⊥"), ("parallel", R, "1 mm 4PP ∥")]:
        xk, k = ls.fpp_kernel_1d(1.0, o)
        meas = ls.apply_probe(xs, ts, xk, k)
        cc, mm, _ = ls.fold(xs, meas - meas.mean(), 2 * P, 152)
        ax.plot(cc, mm, color=col, lw=1, label=lab)
        if o == "perpendicular":
            A = ls.harmonic_amplitudes(xs, meas, 2 * P, 16)
            fk = np.arange(1, 17) / (2 * P)
            H = ls.transfer_function(xk, k, fk)
            keep = np.abs(H) > 0.3
            Ac = np.where(keep, A / np.where(keep, H, 1), 0)
            rec = sum((Ac[i] * np.exp(2j * np.pi * fk[i] * cc)).real for i in range(16))
            ax.plot(cc, rec, color=G, lw=1, ls="--", label="⊥ corrected (|H|>0.3)")
    ax.set_xlabel("position in 7.6 mm cycle (mm)"); ax.set_ylabel("ΔRs (Ω)")
    ax.legend(frameon=False, ncol=2, fontsize=6.5)
    save(fig, "f12_probe_on_fold")


def f13():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), constrained_layout=True)
    rs = ls.simulate_rs(XT, power=alt(0.005))
    for dx, col, lab in [(0.1, B, "0.1 mm sampling"), (1.5, R, "1.5 mm sampling")]:
        xm = np.arange(0, L, dx)
        y = np.interp(xm, XT, rs)
        f, A = ls.amplitude_spectrum(y, dx)
        m = (f > 0.02) & (f < 1.1)
        axs[0].semilogy(f[m], np.abs(A[m]) + 1e-3, color=col, lw=0.7, label=lab)
    for k in range(1, 5): axs[0].axvline(k / P, color=GR, lw=0.4, ls="--")
    axs[0].axvline(1 / 3.0, color=R, lw=0.5, ls=":")
    axs[0].set_ylim(1e-2, 30); axs[0].legend(frameon=False)
    axs[0].set_xlabel("spatial frequency (1/mm)"); axs[0].set_ylabel("|A| (Ω)")
    y = measure(rs)
    for w, col, lab in [(P, R, "detrend window 3.8 mm"), (2 * P, B, "detrend window 7.6 mm"),
                        (None, GR, "no detrend")]:
        if w is None:
            xx, r = XM, y - y.mean()
        else:
            r, v = ls.detrend_moving(XM, y, w); xx, r = XM[v], r[v]
        c, m, _ = ls.fold(xx, r, 2 * P, 76)
        axs[1].plot(c, m - m.mean(), color=col, lw=1, label=lab)
    axs[1].set_xlabel("position in 7.6 mm cycle (mm)"); axs[1].set_ylabel("ΔRs (Ω)")
    axs[1].legend(frameon=False, fontsize=6.5)
    save(fig, "f13_sampling_detrend")


def f14():
    x = np.arange(0, 2 * P * 40, 0.005)
    fig, ax = plt.subplots(figsize=(4.6, 2.4), constrained_layout=True)
    seps = np.linspace(0.02, 8, 120)
    lr = 0.05
    nk = int(lr / 0.005)
    for lab, kw, col in [("normal stitch (3% tilt)", dict(tilt=0.03), B),
                         ("plus 0.5% odd/even", dict(tilt=0.03, power=alt(0.005)), R)]:
        rs = ls.simulate_rs(x, **kw)
        r = np.convolve(rs, np.ones(nk) / nk, "same")
        out = []
        for s_ in seps:
            k = int(round(s_ / 0.005))
            a, b = r[2000:-k - 2000], r[2000 + k:-2000]
            out.append(np.std((a - b) / (0.5 * (a + b))) * 100)
        ax.plot(seps, out, color=col, lw=1.2, label=lab)
    ax.axvline(P, color=GR, lw=0.4, ls="--"); ax.axvline(2 * P, color=GR, lw=0.4, ls="--")
    ax.set_xlabel("separation of two 50 µm-long resistors across stripes (mm)")
    ax.set_ylabel("stitch mismatch σ (%)")
    ax.legend(frameon=False)
    save(fig, "f14_matching")


def f15():
    # parabolic stripes: centre lines x = c_n + a y^2; line scan along 45 deg
    a = 1.0 / 400.0
    t = np.arange(0, 200, 0.05)
    xl = 20 + t / np.sqrt(2); yl = -100 + t / np.sqrt(2)
    u = xl - a * yl**2
    xu = np.arange(u.min() - 10, u.max() + 10, 0.01)
    r1 = ls.simulate_rs(xu, power=alt(0.005))
    y = np.interp(u, xu, r1)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), constrained_layout=True)
    yy = np.linspace(-150, 150, 61)
    for c0 in np.arange(-60, 90, 3.8 * 4):
        axs[0].plot(c0 + a * yy**2, yy, color=GR, lw=0.5)
    axs[0].plot(xl, yl, color=R, lw=1.2, label="45° line scan")
    axs[0].set_aspect("equal"); axs[0].set_xlim(-60, 120); axs[0].set_ylim(-150, 150)
    axs[0].set_xlabel("x (mm)"); axs[0].set_ylabel("y (mm)")
    axs[0].set_title("every 4th stripe centre", fontsize=7.5)
    f, A = ls.amplitude_spectrum(y, 0.05, "hann")
    m = (f > 0.02) & (f < 0.7)
    axs[1].semilogy(f[m], np.abs(A[m]) * 2 + 1e-3, color=R, lw=0.7, label="FFT vs path length")
    uu = np.arange(u.min(), u.max(), 0.05)
    yu = np.interp(uu, np.sort(u), y[np.argsort(u)])
    f2, A2 = ls.amplitude_spectrum(yu, 0.05, "hann")
    m2 = (f2 > 0.02) & (f2 < 0.7)
    axs[1].semilogy(f2[m2], np.abs(A2[m2]) * 2 + 1e-3, color=B, lw=0.7, label="FFT vs stripe coordinate u")
    for k in (1, 2): axs[1].axvline(k / P, color=GR, lw=0.4, ls="--")
    axs[1].axvline(1 / (2 * P), color=GR, lw=0.4, ls=":")
    axs[1].set_ylim(1e-2, 40); axs[1].legend(frameon=False, fontsize=6.5)
    axs[1].set_xlabel("spatial frequency (1/mm)"); axs[1].set_ylabel("|A| (Ω, Hann-corrected)")
    save(fig, "f15_parabolic")


if __name__ == "__main__":
    f1()
    panel3(ls.simulate_rs(XT), "Ideal: flat-top beam, 50% overlap, all passes identical", "f02_ideal")
    panel3(ls.simulate_rs(XT, tilt=0.03), "Long-axis profile tilt (3% centre-to-edge)", "f03_tilt")
    f4()
    panel3(ls.simulate_rs(XT, power=alt(0.01)), "Odd/even pass power mismatch (±1%)", "f05_odd_even_power")
    panel3(ls.simulate_rs(XT, step_errors=lambda n: 0.15 * (n % 2)),
           "Alternating 150 µm lateral error on odd passes", "f06_alt_step")
    f7()
    panel3(ls.simulate_rs(XT, power=lambda n: 1 + 0.005 * np.cos(2 * np.pi * n / 2.3)),
           "Unsynchronized power oscillation (±0.5%, period 2.3 passes)", "f08_unsynced")
    f9(); f10(); f11(); f12(); f13(); f14(); f15()
    for fn in sorted(os.listdir(OUT)):
        print(fn, os.path.getsize(os.path.join(OUT, fn)))
