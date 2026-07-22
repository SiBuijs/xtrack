# Bz-ramp coupling-amplification study (new files, not part of 000-014)

> **REMOVED 2026-07-22, at the user's request — no longer used.** All four
> files below (`bz_ramp_field.py`, `004a2_build_solenoid_bz_ramp.py`,
> `004b2_install_solenoid_bz_ramp_in_fcc_ring.py`, `scan_bz_ramp_amp.py`) were
> `git rm`'d from `examples/fcc_ee_solenoid/`. They are recoverable from git
> history (they were committed) if this line of investigation is ever
> revisited. Nothing else in the pipeline imported them, so removal was
> clean — no other script needs updating as a result. This note is kept
> below **for historical reference only**; do not expect the files it
> describes to exist.

Written 2026-07-14. Purpose: let a future session skip re-deriving why these
files exist and what they found. If a referenced file/function no longer
exists, treat these notes as stale for that detail and re-check the source.

## Motivating question

Is the x-y coupling induced by the FCC-ee detector solenoid **genuinely
small**, or does it **cancel** (e.g. between the front/back halves of the
main solenoid, or between the main and compensation solenoids), so that a
"true" measure of the coupling is being masked? To test this, we wanted to
artificially amplify the longitudinal field asymmetrically in one half of a
solenoid and see how sensitively the coupling responds — without rebuilding
the solenoid model from scratch and without breaking Maxwell's equations
(a naive rescale of `bs` in one half only, e.g. via `SplineBoris.scale_b`,
would not be div/curl-free).

## Method: Maxwell-consistent linear-in-s perturbation, added at the fit level

`bz_ramp_field.py` — `LinearBzRamp(slope, z0)`: an *exact* vacuum solution
(`div B = curl B = 0` for all r, not just paraxially):
`Bz(s) = slope * (s - z0)`, `Br(r) = -(slope/2) * r`. Because Maxwell's
equations are linear, this can be superposed onto the real solenoid field and
the sum stays Maxwell-consistent.

Key implementation insight: `extract_tapered_field_data` +
`build_splineboris_line` (in `spline_boris_setup.py`) are, for a fixed
`s_axis`/taper/spline config, a **linear map** from sampled field values to
final Hermite (`Spline4`) coefficients (fixed-knot least-squares fits,
`polyfit`, and a field-independent geometric taper). So instead of building
one composite analytic field and fitting it once, we fit the ramp *alone*
with 004a's exact config, and **add its Spline4 coefficients to the
already-saved `main_solenoid` template**, slice-by-slice. This is
mathematically identical to fitting the sum once, but never touches/reruns
004a's fit of the real solenoid.

`z0=0.0` centers the ramp on the IP, so `Bz_ramp` is an **odd function about
the IP**: negative at the upstream coil end (`s=-1.23`), zero at the IP
(`s=0`), positive at the downstream end (`s=+1.23`). Its integral over the
symmetric `MAIN_SOLENOID_S_AXIS` is therefore exactly zero — it does not
perturb `comp_scale_b` (main-vs-compensation `∫Bs ds` balance), it purely
breaks the main solenoid's front/back mirror symmetry.

`ramp_amp` normalization: `RAMP_SLOPE = ramp_amp * B0 / (L/2)`, so `Bz_ramp`
swings by `∓ramp_amp * B0` between the two physical coil ends
(`s = ∓1.23 m`). E.g. `ramp_amp=0.2` → `∓0.4 T` at the two ends (a ±20%-of-B0,
40%-peak-to-peak swing).

## New files

- **`bz_ramp_field.py`** — `LinearBzRamp` field model, described above.
- **`004a2_build_solenoid_bz_ramp.py`** — duplicates 004a's exact fit-config
  constants (must stay in sync with `004a_build_and_check_solenoids.py` or
  the coefficient-addition is no longer valid), fits the ramp alone, loads
  the untouched `main_solenoid` template from `004_solenoid_lines.json`, adds
  `bs`/`bx`/`by` coefficients slice-by-slice, runs two sanity checks
  (ramp's own order≥2 coefficients ≈0; combined SplineBoris field vs literal
  analytic sum at random points in the untapered interior — 0.28% max error,
  in line with the pipeline's own 0.66% baseline fit residual, i.e. no
  compounding error from this approach). Saves
  `004_solenoid_lines_bz_ramp_{ramp_amp:g}.json`. CLI: `--ramp-amp` (default
  0.2).
- **`004b2_install_solenoid_bz_ramp_in_fcc_ring.py`** — near-identical copy of
  `004b_install_solenoids_in_fcc_ring.py`, but installs the perturbed
  `main_solenoid` template at **`ipg` only** (`RAMP_IP_NAME = 'ipg'`); the
  other 3 IPs (`ipa`, `ipd`, `ipj`) get the standard unperturbed template.
  Saves `temp_fcc_ee_lcc_splineboris_solenoids_bz_ramp_{ramp_amp:g}.json`.
  CLI: `--ramp-amp`, `--no-plot` (skips the inspection-plot section, used by
  the scan script for speed).
- **`scan_bz_ramp_amp.py`** — scans `ramp_amp` over
  `[0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05,
  0.07, 0.1]`, driving 004a2/004b2 as subprocesses per amplitude, then
  measures **raw (pre-correction)** coupling at `ipg`: with all 4 IPs'
  solenoids off, twiss once for the reference optics (`tw0`); then with only
  `ipg`'s solenoid+compensation switched on, a local twiss across the whole
  straight section (`end_ds_start_straight_ipg` to
  `end_straight_start_ds_ipg`, `init_at=ipg` using `tw0`'s betas) reporting
  running-max `|betx2|`, `|bety1|`, `|alfx2|`, `|alfy1|`, `|dy|`, plus
  `dmux`/`dmuy` (see below). Deliberately measures **raw** coupling rather
  than running through `004c_correct_solenoids_in_fcc_ring.py`, because 004c
  *re-solves* correction knobs from scratch against whatever field is
  installed — running it here would tautologically null the match targets
  regardless of the underlying asymmetry, making it uninformative unless one
  compared knob *strengths* instead of final residuals (not done here).
  Produces `temp_bz_ramp_amp_scan.png` (3x2 grid, log-x; log-y for the four
  coupling metrics, linear-y for `dmux`/`dmuy`). CLI: `--no-show`.
  Run with: `cd examples/fcc_ee_solenoid && python scan_bz_ramp_amp.py`.

## `dmux`/`dmuy` diagnostic (phase-advance across the solenoid region)

Added to `raw_coupling()`: cuts the line (`line.cut_at_s(...)`) at exactly
`IP ± 20 m` (the solenoid+compensation hardware itself only extends to about
`±14 m`, so `±20 m` sits safely outside it in the following drift on both
sides), then reports `mux`/`muy` (normal-mode phase advance in the coupled
Edwards-Teng parametrization, units of `2π` i.e. tune) at those two exact
points, differenced. Note: no such calculation pre-existed in 004d despite
initial belief that it did — this was implemented fresh.

**Finding**: `dmux`/`dmuy` are essentially insensitive to `ramp_amp` — only a
~1e-7 relative shift even at `ramp_amp=0.5` (a huge, unphysically large
perturbation), while `betx2`/`bety1` at the same amplitudes change by
factors of several. This is physically expected, not a bug: Edwards-Teng
normal-mode phase advances shift only at **second order** in coupling
strength (an avoided-crossing-like effect), whereas the cross-plane beta
terms (`betx2`, `bety1`) respond at **first order**. So this panel confirms
the coupling is a first-order beta-mixing effect rather than a tune-shift
effect — a real (if visually undramatic) result, not a failed diagnostic.

## Raw-coupling scan findings (pre-correction, at `ipg`)

At `ramp_amp=0.2` (±20%-of-B0 swing per end), raw coupling metrics grew
46-194x relative to the unperturbed baseline — but this amplitude is itself
comparable in scale to `B0`, so a large response is somewhat expected rather
than surprising on its own. Refining with the amplitude scan above quantified
local log-log power-law crossover points (where the perturbed value first
exceeds baseline):
- `bety1_max`: crossover ≈ 1.1% of `B0`-scale ramp amplitude; roughly linear
  (no protective threshold) growth beyond that.
- `alfy1_max`: crossover ≈ 3.9%.
- `betx2_max`: shows a **destructive-interference minimum** around ≈1.5%
  (i.e. the ramp partially cancels some pre-existing asymmetry before
  overtaking it), crossing back above baseline between 2-3%.
- `alfx2_max`: stays essentially flat until ~7-10%.

Interpretation: the response is not uniform across metrics — some coupling
observables (`bety1`, `alfy1`) grow monotonically/linearly with no special
threshold, consistent with a "genuinely small, not fragile-cancellation"
picture, while `betx2` shows real destructive interference at small
amplitude, meaning at least one observable *is* sensitive to a
front/back-symmetry-breaking perturbation in a genuinely non-monotonic way.
No full 004c correction-knob-strength comparison or DA/MA/emittance impact
study has been run on the perturbed lattices yet (listed as possible next
steps, not done).

## Reproducing

```
cd examples/fcc_ee_solenoid
python scan_bz_ramp_amp.py            # full scan, opens interactive plot
python scan_bz_ramp_amp.py --no-show  # save temp_bz_ramp_amp_scan.png only
```

Single-amplitude build (e.g. to inspect one perturbed lattice by hand):
```
python 004a2_build_solenoid_bz_ramp.py --ramp-amp 0.05
python 004b2_install_solenoid_bz_ramp_in_fcc_ring.py --ramp-amp 0.05
```
