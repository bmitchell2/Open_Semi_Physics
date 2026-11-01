"""
srim_implant.py - Implant profiles and moments (Rp, dRp, skewness, kurtosis) in Si
using pysrim to drive SRIM/TRIM.

Setup:
  pip install pysrim numpy matplotlib
  (if pysrim errors on yaml.load, run: pip install "pyyaml<6")
  Install SRIM-2013 from srim.org and point SRIM_DIR at the folder with TRIM.exe.
  On Linux/macOS install wine; pysrim calls TRIM.exe through it automatically.

Notes:
  - TRIM targets are amorphous: no channeling, so crystalline B/P tails will be deeper.
  - BF2 is approximated as B at 10.81/48.81 of the beam energy (fluorine ignored).
  - TRIM bins the target into 100 depth bins, so the Si layer width is auto-sized
    over a few passes to keep the bins fine relative to the profile.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from srim import TRIM, Ion, Layer, Target

SRIM_DIR = os.path.expanduser("~/srim")   # folder containing TRIM.exe
OUT_DIR = "srim_runs"

# Per-element: displacement, lattice binding, surface binding energies (eV)
SI_EL = {"stoich": 1.0, "E_d": 15.0, "lattice": 2.0, "surface": 4.7}
O_EL = {"E_d": 28.0, "lattice": 3.0, "surface": 2.0}


def si_layer(width_A):
    return Layer({"Si": SI_EL}, density=2.329, width=width_A, name="Si")


def oxide_layer(width_A):
    return Layer({"Si": {**SI_EL, "stoich": 1 / 3}, "O": {**O_EL, "stoich": 2 / 3}},
                 density=2.20, width=width_A, name="SiO2")


def run_trim(symbol, energy_keV, si_width_A, screen_ox_A=0.0, n_ions=2000, tag="run"):
    """One TRIM run. Returns depth (nm, from Si surface) and ion density per unit dose."""
    ion = Ion(symbol, energy=energy_keV * 1e3)            # pysrim energy is in eV
    layers = ([oxide_layer(screen_ox_A)] if screen_ox_A > 0 else []) + [si_layer(si_width_A)]
    trim = TRIM(Target(layers), ion, number_ions=n_ions, calculation=1)
    results = trim.run(SRIM_DIR)
    out = os.path.join(OUT_DIR, tag)
    os.makedirs(out, exist_ok=True)
    TRIM.copy_output_files(SRIM_DIR, out)

    depth_A = np.asarray(results.range.depth, dtype=float)
    ions = np.asarray(results.range.ions, dtype=float)   # (atoms/cm3)/(atoms/cm2)
    bin_A = depth_A[1] - depth_A[0]
    depth_nm = (depth_A + bin_A / 2 - screen_ox_A) / 10.0  # bin centers, Si-referenced
    return depth_nm, ions


def moments(x, w):
    w = np.clip(w, 0, None)
    w = w / w.sum()
    m = np.sum(x * w)
    s = np.sqrt(np.sum((x - m) ** 2 * w))
    g = np.sum((x - m) ** 3 * w) / s ** 3
    b = np.sum((x - m) ** 4 * w) / s ** 4
    return m, s, g, b


def implant(species, energy_keV, screen_ox_nm=0.0, n_ions=2000, passes=3):
    """Auto-size the Si layer, then return profile and moments of the dopant in Si."""
    sym, E = species, energy_keV
    if species.upper() == "BF2":
        sym, E = "B", energy_keV * 10.811 / 48.81

    ox_A = screen_ox_nm * 10.0
    width_A = 30000.0                                       # generous first guess (3 um)
    for p in range(passes):
        last = p == passes - 1
        x, w = run_trim(sym, E, width_A, ox_A, n_ions if last else 300,
                        tag=f"{species}_{energy_keV}keV_ox{screen_ox_nm}nm_p{p}")
        in_si = x > 0
        m, s, _, _ = moments(x[in_si], w[in_si])
        bin_A = width_A / 100.0
        width_A = 3.0 * (m + 4 * s) * 10.0 + bin_A         # ~3x the profile extent

    in_si = x > 0
    Rp, dRp, skew, kurt = moments(x[in_si], w[in_si])
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    retained = _trapz(w[in_si], x[in_si] * 1e-7)          # fraction of dose in Si
    return dict(species=species, E=energy_keV, Rp=Rp, dRp=dRp, skew=skew,
                kurt=kurt, retained=retained, depth=x, per_dose=w)


def sweep(species_list=("B", "BF2", "P", "As", "Sb"), energies=(10, 20, 50, 100, 200),
          screen_ox_nm=0.0, n_ions=2000):
    print(f"{'Species':<8}{'E(keV)':>7}{'Rp(nm)':>9}{'dRp(nm)':>9}{'skew':>7}{'kurt':>7}")
    rows = []
    for sp in species_list:
        for E in energies:
            r = implant(sp, E, screen_ox_nm, n_ions)
            rows.append(r)
            print(f"{sp:<8}{E:>7}{r['Rp']:>9.1f}{r['dRp']:>9.1f}"
                  f"{r['skew']:>7.2f}{r['kurt']:>7.2f}")
    return rows


def plot_profiles(runs, dose_cm2=1e15):
    plt.figure(figsize=(7, 4.5))
    for r in runs:
        si = r["depth"] > 0
        plt.semilogy(r["depth"][si], r["per_dose"][si] * dose_cm2,
                     label=f"{r['species']} {r['E']} keV")
    plt.xlabel("Depth in Si (nm)")
    plt.ylabel(f"Concentration (cm$^{{-3}}$) at {dose_cm2:.0e} cm$^{{-2}}$")
    plt.ylim(dose_cm2 * 1e2, None)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("srim_profiles.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Full table sweep (25 runs x 3 passes; can take a while with 2000 ions)
    rows = sweep(n_ions=2000)

    # Example: same species, bare Si vs 10 nm screen oxide
    bare = implant("P", 30, screen_ox_nm=0)
    scr = implant("P", 30, screen_ox_nm=10)
    print(f"\nP 30 keV bare:   Rp={bare['Rp']:.1f} nm, dRp={bare['dRp']:.1f} nm")
    print(f"P 30 keV 10nm ox: Rp={scr['Rp']:.1f} nm (Si-referenced), "
          f"retained in Si={scr['retained']:.1%}")

    plot_profiles([r for r in rows if r["E"] == 50])
