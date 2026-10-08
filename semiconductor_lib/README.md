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
- `semiconductor_lib/leakage.py` — field-driven junction leakage: reduced
  doping N_eff, abrupt-junction peak field, uniform-field (Kane) band-to-band
  tunnelling current density with a `B_override` for model sensitivity, and
  the classical overlap-GIDL surface field (Chan et al. 1987). Absolute BTBT
  magnitude is model-dependent; the doping/field trend is the robust output.
  Tests: `tests/test_leakage_multigate.py`.
- `semiconductor_lib/multigate.py` — natural (scale) length
  λ = √(εsi t tox / N εox) for 1-4 equivalent gates (Colinge 2004) and the
  5λ minimum-gate-length estimate. Tests: `tests/test_leakage_multigate.py`.
- `semiconductor_lib/figures_leakage.py` — generators for the GIDL/pocket
  junction-field-and-BTBT figure and the multigate natural-length figure;
  minified SVGs of about 9 KB. Tests: `tests/test_figures_leakage.py`.
- `semiconductor_lib/implant_dosimetry.py` — beamline implanter dosimetry and
  angle sensitivity: charge-exchange ion survival, pressure-compensation
  K-factor (I_dose = I_meas exp(KP), US 6,657,209) with residual-error and
  fit helpers, decel energy-contamination fraction and depth ratio, and tilt
  geometry (lateral reach and shadow-edge sensitivity). Tests:
  `tests/test_implant_dosimetry.py`; figure script
  `examples/implant_dosimetry_figures.py` (~11 KB SVG).
- `semiconductor_lib/implant_anneal.py` — post-implant anneal models:
  intrinsic boron diffusivity (0.76 exp(-3.46 eV/kT)), backward-Euler 1-D
  diffusion with a decaying TED supersaturation S(t)=1+S0 exp(-t/tau) and an
  optional immobile (clustered) peak, the integrated TED budget, Arrhenius
  defect-dissolution time ratio (Ea ~3.8 eV, Stolk et al. 1997), trap-limited
  release of implanted nitrogen to a Si/SiO2 interface sink with optional
  capacity saturation (after Adam et al. 2000; Dokumaci et al. 2001), and the
  amorphous/crystalline (EOR) depth from a damage profile. Tests:
  `tests/test_implant_anneal.py` (dose conservation, TED tail spreading,
  peak pinning, interface uptake/saturation, 1000 C 10 s boron sqrt(Dt) ~3.9
  nm); figure script `examples/implant_anneal_figures.py` (10-15 KB SVGs).
- `semiconductor_lib/high_field.py` — high-field transport: Canali bulk
  velocity-field curve (Jacoboni/Canali parameters), inversion-layer
  velocity saturation (V_Dsat, I_Dsat, L-independence limit), universal
  inversion-layer mobility fit (del Alamo 6.720), effective field with
  eta = 1/2 (electrons) and 1/3 (holes), illustrative Matthiessen split.
- `semiconductor_lib/lithography.py` — Rayleigh resolution/DOF, k1 = 0.25
  two-beam limit, coherent grating aerial image from captured orders
  (on-axis or dipole), photon shot noise vs dose and wavelength,
  swing-curve / interferometric-endpoint period, Gaussian implant-mask
  transmission, Tanaka capillary-collapse stress.
- `semiconductor_lib/plasma.py` — Debye length, Bohm velocity, floating
  potential, Child-law sheath thickness, ion mean free path, ion arrival
  angle, capacitive area-ratio voltage scaling.
- `semiconductor_lib/deposition.py` — Grove two-resistance CVD rate,
  Knudsen-regime (Thiele) reactant depletion down a via or trench,
  steady-state Berg model of reactive sputtering (target poisoning and
  flow hysteresis), Stoney film stress.
- `semiconductor_lib/thermal_budget.py` — sum of D*t over a step list,
  equivalent time at a reference temperature, D*t over a ramped profile,
  interstitial Fe and Cu diffusivity and Fe solubility in Si (gettering).
- `examples/deposition_thermal_figures.py` — figures for the CVD,
  reactive-sputtering and gettering notes.
- `examples/ngspice_vsat_crosscheck.py` — I_Dsat vs L from the PTM 180 nm
  BSIM3 card versus the analytic velocity-saturation models (L_eff and
  knee-current method; results in the docstring).
- `examples/high_field_litho_plasma_figures.py` — figures for the velocity
  saturation, mobility, lithography, EUV and plasma notes.
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
- `tests/test_high_field_litho_plasma.py` — physics validation for
  `high_field.py`, `lithography.py` and `plasma.py` (velocity-saturation
  limits, universal mobility, k1 = 0.25 limit, order capture and aerial-image
  contrast, photon counts, Child-law scaling, argon floating potential).
- `tests/test_deposition_thermal_budget.py` — physics validation for
  `deposition.py` and `thermal_budget.py` (Grove limits, 1/cosh depletion
  trend and small-phi expansion, Berg metallic-to-poisoned transition and
  hysteresis removal by pumping, Stoney magnitude, boron D*t for 1 h at
  800 C versus 15 min at 900 C, Fe crossing a wafer in about an hour).
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

All 268 tests currently pass (as of 2026-10-07, including 10 in
`test_deposition_thermal_budget.py`, 21 in
`test_high_field_litho_plasma.py`, 9 in `test_implant_anneal.py`, 9 in
`test_implant_dosimetry.py`, 12 in
`test_diode.py`, 6 in `test_figures_devices.py`, 6 in
`test_leakage_multigate.py`, 3 in `test_figures_leakage.py` and 11 in
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
