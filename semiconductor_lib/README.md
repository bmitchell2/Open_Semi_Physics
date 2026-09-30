# semiconductor-lib

Shared, verified code for the Semiconductor Notes knowledge base — built so
figures and circuit diagrams don't get re-derived and re-debugged from
scratch in every conversation. Companion to Section 31 ("Visual Generation
Workflow and Tooling") of the Processing Instructions page in Notion.

## Install

Once this is pushed to a repo:

```bash
pip install git+https://github.com/<your-username>/semiconductor-lib.git
```

No credentials needed for this — cloning/installing from a public repo is a
read, not a write.

For local development:

```bash
pip install -e .
```

## Structure

- `semiconductor_lib/constants.py` — physical constants, material
  permittivities, thermal voltage.
- `semiconductor_lib/electrostatics.py` — verified 1-D ideal MOS capacitor
  electrostatics (Kingston-Neustadter/Garrett-Brattain). Threshold voltage,
  LF/HF/deep-depletion capacitance, both p- and n-type substrates.
- `semiconductor_lib/lattice.py` — diamond-cubic silicon geometry checked by
  enumerating atom positions: neighbour shells, vacant tetrahedral sites,
  planar densities for (100)/(110)/(111), packing fraction, bulk densities.
- `semiconductor_lib/dopants.py` — Si band gap / DOS / n_i temperature fits
  and the self-consistent shallow-dopant ionization solver (donors and
  acceptors, freeze-out; non-degenerate, unreliable above ~1e18 cm^-3).
- `semiconductor_lib/pnjunction.py` — abrupt p-n junction: closed-form
  depletion approximation and a nonlinear-Poisson (Boltzmann, damped Newton)
  numerical solver, with the depletion-approximation error quantified.
- `semiconductor_lib/diode.py` — biased p-n diode: forward I-V with SCR
  (n = 2) current and series resistance, local ideality factor, forward-voltage
  tempco (closed form and numerical, ~-2 mV/K), reverse leakage components vs
  temperature (diffusion ∝ ni², generation ∝ ni W), Sze-Gibbons breakdown and
  Miller multiplication, charge control (Q = I tau, C_diff), short-base transit
  time, Kingston storage time, and a finite-difference reverse-recovery
  transient (matches Kingston within 1 %). Tests: `tests/test_diode.py`.
- `semiconductor_lib/band_diagrams.py` — band diagrams along a device:
  (1) abrupt p-n junction under bias (depletion approximation, quasi-Fermi
  levels, Joyce-Dixon Fermi-level offsets so degenerate sides work; V_bi from
  band alignment); (2) quasi-2-D (Young / Liu et al.) lateral surface
  potential of a bulk nFET below threshold with laterally non-uniform
  (pocket) doping, finite-difference solve, V_T by a constant surface-
  electron-density criterion, DIBL, RSCE and off-state source barrier.
  Verified: FD matches the closed-form sinh solution to <0.5 mV; long-channel
  V_T matches `mosfet.vt_uniform` to 3 mV; DIBL grows as L shrinks and is
  reduced by pockets; pocket V_T rises as L shrinks (RSCE).
  Tests: `tests/test_band_diagrams.py`; figures:
  `examples/band_diagram_figures.py`.
- `semiconductor_lib/oxidation.py` — Deal-Grove thermal oxidation
  (thickness, time, tau, (111)/(100) orientation factor, silicon consumption).
- `semiconductor_lib/bands.py` — silicon effective masses and band-curvature
  helpers: conductivity vs. density-of-states masses (electrons and holes),
  parabolic bands, numerical curvature -> mass, cyclotron-resonance fields.
- `semiconductor_lib/carriers.py` — Fermi-Dirac occupation, 3-D density of
  states, effective DOS, numerical-integral vs. Boltzmann cross-check,
  Fermi potential, intrinsic-level offset, fixed-EF ionization estimate.
- `semiconductor_lib/transport.py` — drift/diffusion relations: Einstein
  relation, conductivity/resistivity, scattering time, drift velocity,
  diffusion length, drift-diffusion current densities.
- `semiconductor_lib/figures.py` — reproducible generators for the six
  figures on the 2026-09-28 notes (unit cell, Fermi level, ionization,
  E-k curvature, p-n junction, Deal-Grove), built from the modules above;
  `build_all(outdir)` writes minified SVGs of 7-13 KB. Docstring explains
  why mathtext must be avoided (glyph outlines double the SVG size).
- `semiconductor_lib/figures_devices.py` — generators for the 2026-09-29
  diode (forward I-V/ideality, leakage Arrhenius, stored charge and diffusion
  capacitance, reverse recovery) and MOS capacitor (band diagram at VG = 0,
  W and Qinv vs VG) figures; `build_all(outdir)` writes minified SVGs of
  6-16 KB. Tests: `tests/test_figures_devices.py`.
- `semiconductor_lib/lifetime.py` — SRH generation lifetime vs. doping,
  Zerbst transient simulation/extraction, DLTS Arrhenius extraction.
- `semiconductor_lib/reliability.py` — percolation breakdown concept model,
  Weibull extrinsic/intrinsic population separation (TDDB, RVS).
- `semiconductor_lib/plotting.py` — shared matplotlib style preset and the
  SVG minification pipeline (scour + whitespace/precision pass) that keeps
  figures under Notion's inline-attachment size limit.
- `semiconductor_lib/circuits.py` — reusable schemdraw building blocks
  (circuit schematics and block/flow diagrams in one tool).
- `tests/test_electrostatics.py` — physics validation: every claim in
  `electrostatics.py` and `lifetime.py` docstrings is checked here. Run
  once, trusted thereafter — don't re-derive these checks by hand in a
  new conversation.
- `tests/test_lattice_dopants_pn_oxidation.py` — physics validation for
  `lattice.py`, `dopants.py`, `pnjunction.py` and `oxidation.py`; each test
  encodes a number verified while writing the corresponding Semiconductor
  Notes page (neighbour shells, ionization fractions, depletion-approximation
  error, Deal-Grove worked examples).
- `tests/test_bands_carriers_transport.py` — physics validation for
  `bands.py`, `carriers.py` and `transport.py` (masses, Hu examples,
  numerical-vs-Boltzmann carrier density, Einstein relation, resistivity,
  equilibrium drift/diffusion cancellation).
- `tests/test_figures.py` — every figure builds with the expected panels,
  and the minified SVGs are well-formed and under 30 KB.
- `examples/ngspice_moscap_cv.py` — cross-check of the hand-derived
  electrostatics against a real BSIM3 compact model (ngspice + a public
  PTM 180nm model card). See the module docstring for the verification
  numbers from the most recent run.

## Running the tests

```bash
pip install -e ".[test]"
pytest tests/ -v
```

All 129 tests currently pass (as of 2026-09-30, including 12 in
`test_diode.py`, 6 in `test_figures_devices.py` and 11 in
`test_band_diagrams.py`). Two of the
electrostatics tests are worth knowing about if they ever look like they've
"regressed":

- **Cmin vs. the analytic max-depletion-width formula** is checked to 10%,
  not tighter, because the analytic formula freezes the depletion width at
  exactly `psi_s = 2*phi_F`, while the full numerical solve lets `psi_s`
  keep drifting slightly upward past threshold. That gap is real physics,
  not solver error — see the test's docstring.
- **Zerbst extraction** derives its equilibrium reference (`x_eq`) from the
  simulated data's own late-time asymptote, not from an independently
  computed formula. Using the formula instead introduces a small
  floating-point/ODE-settling mismatch that biases the fitted `tau_g` and
  `s0` by tens of percent — this was an actual bug caught while building
  this package (2026-09-06), not a hypothetical one.

One diode test is similar: **the reverse-recovery simulation is compared
with Kingston using the reverse current actually drawn during storage**
(about 3.5 % above V_R/R, because the junction still holds ~+0.7 V then).
Using the nominal V_R/R instead produces a spurious ~4 % mismatch.

## Running the ngspice cross-check

```bash
cd examples
python3 ngspice_moscap_cv.py
```

Requires `ngspice` (`apt-get install ngspice`). Verified result (see the
module docstring): the simulated accumulation/inversion plateau matches
the analytic `Cox * Area` to within 2.4%, with the small excess explained
by the model card's fixed gate-overlap capacitance — and the depletion
minimum sits at roughly 30% of that plateau, a real dip in the right place
relative to the model's threshold voltage.

## What's not in here yet

- Poly depletion, oxide-thickness/XRR, and series-resistance models
  (built and verified in the 2026-09-05 "Moscaps and veractors" session,
  not yet ported into this package as reusable functions).
- A DEVSIM-based example (recommended in Processing Instructions Section
  31 for device physics beyond what a compact model captures — not yet
  built). A drift-diffusion transient would also cross-check the
  diode.py reverse-recovery model including junction capacitance.
- More schemdraw circuit-building blocks (common-source stage,
  differential pair, current mirror, cascode — discussed but not yet
  built).
