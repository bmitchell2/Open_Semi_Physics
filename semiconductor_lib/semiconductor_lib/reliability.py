"""
Gate oxide reliability models: percolation-model breakdown simulation and
Weibull extrinsic/intrinsic population separation (TDDB and RVS).

Physics/statistics basis:
  Weibull CDF: F(x) = 1 - exp(-(x/eta)^beta)
  Weibull-plot transform (linearizes the CDF): W = ln(-ln(1-F)) vs ln(x)
  A single population plots as a straight line of slope beta. A mixture
  of an extrinsic (defect-driven) and intrinsic (oxide-quality-driven)
  sub-population shows a "kink": extrinsic points break off from the
  intrinsic line at the low end of the distribution, typically with a
  shallower slope.

  Plotting-position estimator: Bernard's approximation,
      F_i = (i - 0.3) / (n + 0.4)
  the standard median-rank estimator for Weibull plots of reliability
  data.

All numeric parameters (eta, beta, sample sizes) below are illustrative
defaults chosen to produce a representative, qualitatively correct
shape -- not fits to measured process data.
"""
import numpy as np


def percolation_breakdown_step(n_traps_max, box_size=1.0, connect_radius=0.12,
                                seed=None):
    """Simulate random trap generation in a 2-D box (representing a cross
    section through the oxide) and return the trap coordinates plus the
    step index at which a connected path first spans the box top-to-bottom
    (the percolation / breakdown event). Illustrative only -- not a fit
    to any specific process's trap generation rate."""
    rng = np.random.default_rng(seed)
    traps = rng.uniform(0, box_size, size=(n_traps_max, 2))

    def spans(pts):
        if len(pts) == 0:
            return False
        # union-find style connectivity via a distance graph
        n = len(pts)
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if np.hypot(*(pts[i] - pts[j])) < connect_radius:
                    adj[i].append(j)
                    adj[j].append(i)
        bottom = [i for i in range(n) if pts[i, 1] < connect_radius]
        top = set(i for i in range(n) if pts[i, 1] > box_size - connect_radius)
        seen = set()
        stack = list(bottom)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in top:
                return True
            stack.extend(adj[node])
        return False

    for step in range(1, n_traps_max + 1):
        if spans(traps[:step]):
            return traps[:step], step
    return traps, None


def bernard_plotting_positions(n):
    """Bernard's approximation median-rank plotting positions for n
    ordered samples, i = 1..n."""
    i = np.arange(1, n + 1)
    return (i - 0.3) / (n + 0.4)


def weibull_plot_coordinates(samples):
    """Convert an array of failure times/voltages into Weibull-plot
    coordinates (ln(x), ln(-ln(1-F))), sorted ascending."""
    x = np.sort(np.asarray(samples))
    F = bernard_plotting_positions(len(x))
    return np.log(x), np.log(-np.log(1 - F))


def simulate_mixed_population(n_intrinsic, n_extrinsic, eta_intrinsic,
                               beta_intrinsic, eta_extrinsic, beta_extrinsic,
                               seed=None):
    """Simulate a mixed extrinsic + intrinsic Weibull population (e.g. a
    TDDB time-to-failure or RVS breakdown-voltage distribution). Returns
    the combined, sorted sample array. The extrinsic sub-population
    should use a smaller eta (fails earlier / at lower voltage) to
    reproduce the characteristic low-end "kink" described above."""
    rng = np.random.default_rng(seed)
    intrinsic = eta_intrinsic * rng.weibull(beta_intrinsic, size=n_intrinsic)
    extrinsic = eta_extrinsic * rng.weibull(beta_extrinsic, size=n_extrinsic)
    return np.sort(np.concatenate([intrinsic, extrinsic]))
