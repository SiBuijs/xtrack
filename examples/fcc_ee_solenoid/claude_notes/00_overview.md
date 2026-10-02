# fcc_ee_solenoid — directory overview (for Claude, read this first)

Written 2026-07-10, pruned 2026-10-02. Purpose: let a future session skip
re-reading every script. If a referenced file/function no longer exists, treat
these notes as stale for that detail and re-check the source.

**Branch `fcc_nonlocal_solenoid` (2026-10-02) keeps only the lattice-construction
chain.** Scripts 004aa, 004cc, 004dd, 004e-004m and 006-018 were removed, along
with the helpers `lattice_knobs.py`, `aperture_grid.py`, `aperture_study_io.py`
and `radial_steering.py`, the `results/` directory, and the `_o2` / `_mainscale`
lattice variants. The four study-only knobs `sext_amp`, `comp_b_scale`,
`main_b_scale` and `comp_b_scale_{side}_{ip}` no longer exist: 004b installs the
order-2 spline coefficients as plain element data, and the compensation field is
gated by `on_comp_sol_{ip}` alone. All of it is recoverable from git history
except notes 06 and 07, which were never committed.

## What this study is about

FCC-ee (Z-pole, 45.6 GeV e+/e-) lattice `fccee_z_lcc.json` (local-chromaticity-
correction optics, built elsewhere — not by anything in this dir) needs a
2 T x 2.46 m detector solenoid installed around each of 4 IPs
(`ipa`, `ipd`, `ipg`, `ipj`), tilted by `theta = -0.015` rad w.r.t. the beam
axis. The solenoid couples x-y motion and perturbs optics/dispersion; this
directory builds increasingly refined field models of the solenoid +
compensation scheme, installs them in the ring, corrects orbit/optics/coupling
around each IP, then runs aperture and emittance studies on the result.

Two competing solenoid **element models** are carried in parallel throughout:
- **SplineBoris** (`xt.SplineBoris`): field given by quartic (`xt.Spline4`)
  longitudinal profiles of `bs`, and per-multipole-order `bx`/`by`, integrated
  with a Boris pusher. Higher fidelity (captures multipole content up to
  sextupole-like order and beyond), more expensive. Sextupole content is now
  fixed at build time by `SEXTUPOLE_AMPLIFICATION_FACTOR` in 004a; there is no
  runtime knob for it.
- **VariableSolenoid** (`xt.VariableSolenoid`): linear `ks_profile` (2-point
  linear ramp) + one dipole kick (`knl`/`ksl`) per slice. Cheaper, linear-only.

Naming convention seen everywhere: **"SB"** = SplineBoris, **"VarSol"** =
VariableSolenoid. The case names `sb_on` / `varsol_on` / `sb_off` belonged to the
removed 009/010/014 family (`sb_off` = the same JSON with solenoids and
correctors switched off, i.e. the bare-machine baseline).

## Pipeline / file dependency order

```
fccee_z_lcc.json (external input, LCC optics, no solenoids)
   |
   |-- (superseded/legacy path, kept for reference) --
   |   000a -> temp_fcc_ee_lcc_local_solenoid.json
   |   000b -> fccee_z_lcc_local_solenoid.json      (reads 000a's temp file)
   |   001a -> temp_fcc_ee_lcc_non_local_solenoid.json
   |   001b -> fccee_z_lcc_non_local_solenoid.json  (reads 001a's temp file)
   |   002, 003: analysis/plots of the above (non_local variant)
   |
   ‾-- (current path) --
       004a: build isolated SplineBoris + VariableSolenoid element templates
             from analytic tilted field maps -> 004_solenoid_lines.json
       004b_install_solenoids_in_fcc_ring.py       (SplineBoris)
           -> temp_fcc_ee_lcc_splineboris_solenoids.json
       004b_install_varsol_solenoids_in_fcc_ring.py (VariableSolenoid)
           -> temp_fcc_ee_lcc_varsol_solenoids.json
       004c --model=splineboris|varsol: orbit+optics+coupling correction
           -> fccee_z_lcc_splineboris_solenoids_coupling_corrected.json
           -> fccee_z_lcc_varsol_solenoids_coupling_corrected.json
       004d: analysis/plots of the SplineBoris corrected lattice
```

004d is the end of the chain on this branch. It reads the SplineBoris corrected
lattice for the requested `--b0` plus, always, the untagged 2T and 3T ones as
comparisons. The varsol corrected pair is still produced by 004c but nothing
reads it any more. Everything in the 000a-003 path is legacy/exploratory and not
on the live dependency path.

Concrete filenames carry a field tag and, for SplineBoris only, an order tag:
`004_solenoid_lines_{2T,3T}.json`,
`temp_fcc_ee_lcc_{splineboris,varsol}_solenoids_{2T,3T}.json`,
`fccee_z_lcc_{splineboris,varsol}_solenoids_coupling_corrected_{2T,3T}.json`.

## Shared helper modules (used across many scripts)

- `tilted_solenoid.py` — `TiltedSolenoid(L, a, B0, theta)`: wraps
  `xt.._temp.boris_and_solenoid_map.solenoid_field.SolenoidField` (an
  axisymmetric analytic hard-edge-ish solenoid field model), rotating fields
  in/out of the tilted frame. This is the actual physical model of the main
  detector solenoid everywhere in the "current" path.
- `spline_boris_setup.py` — the real field-extraction/SplineBoris-building
  logic used by 004a (superset of the simpler inline versions duplicated in
  006/007/008). Key functions: `extract_tapered_field_data`,
  `build_splineboris_line`, `build_variable_solenoid_line`,
  `assemble_three_solenoid_system`, `symplectic_error`,
  `sample_splineboris_line[_on_s]`, `smooth_edge_taper`.
- `solenoid_params.py` — single source of truth for main/compensation
  solenoid geometry, imported by 004a/004b[_varsol]/004c/004d/009-015 (added
  2026-07-16, `--b0` CLI mechanism added 2026-07-24). `MAIN_SOLENOID_B0`
  selects the field-strength case (`2.0` or `3.0` T so far); `add_b0_argument`
  wires a `--b0` flag into each script, and `field_tag(b0)` (e.g. `3.0 ->
  '3T'`) is threaded into every filename across the pipeline so different
  cases never overwrite each other's lattice/study files.
  **2026-07-28: 009/010/013 caught up to this pattern** — until then they
  only read the module-level `MAIN_SOLENOID_B0`/`FIELD_TAG` default at import
  time (no `--b0` of their own), even though `MAIN_SOLENOID_B0` was already
  `3.0`, so they happened to already target the 3T lattices without a flag.
  Now `_build_ma_cases(tag)`/`_build_da_cases(tag)` build their case lists
  from a runtime `field_tag(args.b0)` (mirroring 014/015's
  `_build_emitt_cases`/`_build_pol_cases`), and 013 forwards its own `--b0`
  to all three subprocesses. This also exposed and fixed a real gap in
  `aperture_study_io.save_da_study`/`save_ma_study`, which had no `field_tag`
  passthrough at all (unlike `save_emitt_study`).
  **2026-08-03: `--max-transverse-order` added**, mirroring `--b0`/`field_tag`
  exactly. `order_tag(n)`/`add_max_order_argument()` in `solenoid_params.py`
  (default `n=4`, tag `''` at the default, `'_o2'` etc. otherwise) cap the
  transverse multipole order baked into each installed `xt.SplineBoris`
  element via `build_splineboris_line`'s `max_transverse_derivative_order_
  for_spline` (see `spline_boris_setup.py`) — this directly sets
  `element.multipole_order`, i.e. how many `bx`/`by` polynomial terms the
  Boris pusher evaluates per step. Lower orders (below 2, the sextupole row)
  drop that multipole content in exchange for cheaper tracking. SplineBoris-only — VariableSolenoid is linear-only
  and untouched by this flag. Field-extraction order in 004a
  (`MAX_TRANSVERSE_DERIVATIVE_ORDER`) stays fixed at 4 regardless (it's the
  cheap, one-time part; the spline-build order must stay `<=` it). Threaded
  through the same tagged-filename chain as `--b0`: 004a's
  `004_solenoid_lines_{FIELD_TAG}{ORDER_TAG}.json` -> 004b's
  `temp_fcc_ee_lcc_splineboris_solenoids_{FIELD_TAG}{ORDER_TAG}.json` ->
  004c's `fccee_z_lcc_splineboris_solenoids_coupling_corrected_
  {FIELD_TAG}{ORDER_TAG}.json` (004c's `--model varsol` path is untagged by
  order, since it has no such knob) -> 004d and 009/010/014/015 (each of
  which also tags its saved DA/MA/EMIT/POL npz+pdf outputs with the combined
  `{field_tag}{order_tag}` string). 013 forwards its own
  `--max-transverse-order` to all three subprocesses, same as `--b0`. Not
  wired into 011 (which has no `--b0` either — see its own note below).
  In 004a, dropping the order below 2 disables one diagnostic ("straight
  from SplineBoris" d^2Bx/dx^2 integral, which reads `element.bx[2]`
  directly) rather than crashing — guarded by
  `D2BX_DX2_SPLINE_DIAGNOSTIC_AVAILABLE`.
  **2026-07-27: main-solenoid half-length and first-corrector ds_start are
  now case-dependent too** (previously both were hardcoded at the 2T values
  regardless of `MAIN_SOLENOID_B0`, silently wrong for the 3T case) —
  `half_length_for_b0(b0)`/`corrector_ds_start_for_b0(b0)` look up
  `_MAIN_SOLENOID_HALF_LENGTH_BY_B0`/`_MAIN_SOLENOID_CORRECTOR_DS_START_BY_B0`
  dicts (`2.0 T -> 1.23/1.23 m`, `3.0 T -> 1.30/1.40 m`; `ds_end=2.29 m` is
  the same for both cases). 004a calls `half_length_for_b0(MAIN_SOLENOID_B0)`
  after applying the CLI `--b0` override; 004b/004b_varsol call
  `corrector_ds_start_for_b0(_args.b0)` the same way. If a new field-strength
  case is ever added, both dicts need a new entry or these functions raise
  `ValueError`.
  `PLOT_DIR` now lives in `solenoid_params.py` (moved there 2026-10-02 when
  `aperture_study_io.py` was deleted — 004d imported the whole module for that
  one constant). It is `Path.home() / "cernbox/Pictures/FCC_Solenoid_Studies"`,
  overridable via the `FCC_SOLENOID_PLOT_DIR` environment variable, which is
  what the remote/headless box uses.

## Physical/engineering constants worth remembering

- Main solenoid: `L=1.23*2=2.46 m`, `a=0.13 m` radius, `B0=2.0 T`, tilt
  `theta=-0.015 rad`.
- Compensation solenoids: `L=1.5 m`, `a=0.03 m`, `B0=1.0 T` (unscaled), one on
  each side at `COMP_SOLENOID_DISTANCE_FROM_IP = 12.0 m` from the IP; scale
  factor `comp_scale_b` chosen so `2 * comp_scale_b * comp_integral =
  -main_integral` (cancel the net `∫Bs ds`).
  Note 004a/004c and 004b are independent scripts. 004a computes `comp_scale_b`
  during template-building and bakes it into the saved
  `compensation_solenoid`/`compensation_solenoid_varsol` line templates in
  `004_solenoid_lines.json`; 004b/004c just install/correct those templates and
  do not need to recompute `comp_scale_b`.
- IP doublet quads get tilted by half the integrated solenoid rotation
  (`ksol_l_main_solenoid/2/2` each side) to compensate the solenoid's
  Larmor/coupling rotation — done identically in 000b/001b/004c.
- Correction knob chain per IP (final consolidated knob):
  `on_sol_corr_{ip}` drives `on_comp_sol_{ip}`, `on_rot_doublet_{left,right}_{ip}`,
  `on_sol_orbit_corr_{ip}`, `on_sol_optics_corr_{ip}`, and (004c/current path
  only) `on_sol_coupling_corr_{ip}`.
- `line.particle_ref.anomalous_magnetic_moment = 0.00115965218128` is set
  whenever spin/polarization-related twiss (`polarization_analysis=True`) is
  used (002, 004d, 009/010/011/014 radiative tracking setup).

## Known caveats / open items mentioned in comments

- With solenoids on, x-y coupling makes single-plane Courant-Snyder
  emittances (as computed in 014) only approximate; a follow-up could extract
  emittances from 4x4 transverse covariance eigenvalues instead (see
  014's module docstring).
- `002_analysis_and_plots.py` still points at the legacy
  `fccee_z_lcc_non_local_solenoid.json`, not the current SplineBoris/VarSol
  corrected lattices — treat 002/003 as legacy-path analysis only.
- A stray file `Untitled` (single line: `build_splineboris_line`) and a
  `__pycache__/` sit in this directory — junk, not part of the pipeline.

## Note index

Only three notes describe code that still exists on this branch; the rest are
kept for their measurements.

- `01_lattice_construction_000_004d.md` — 000a/000b/001a/001b/002/003/004a/004b
  (x2)/004c/004d in detail. **Current.** Its 004aa/004cc sections describe
  deleted scripts.
- `04_bz_ramp_coupling_amplification.md` — the linear-Bz-ramp perturbation used
  to probe whether the detector solenoid's x-y coupling is genuinely small or a
  fragile cancellation, plus raw-coupling and phase-advance scan findings. The
  scripts were `git rm`'d 2026-07-22; the note is history only.
- `06_coupling_matching_convergence.md` — why the per-scan-point coupling
  re-solve (84 skew-quad vary knobs vs 12 targets) is ill-conditioned, the SVD
  diagnosis, the `broyden=True` fix, and a ranked list of further options for
  outright non-convergence. The `max_step` fix recorded there was **backed out
  on 2026-09-05** in favour of a smaller finite-difference `step` plus tighter
  tolerances — read that section's "Superseded 2026-09-05" note before trusting
  the surrounding text. Describes 004f/004g/004j, all deleted.
  **Never committed — this file is the only copy.**
- `07_main_b_scale_scans.md` — the `main_b_scale` and per-side
  `comp_b_scale_{side}_{ip}` knobs and the 004h/004i/004j scans that used them.
  Also records which knobs 004c/004f/004h/004j each vary, and what every line
  and shading on the 004j plots means. Knobs and scripts are all gone now.
  **Never committed — this file is the only copy.**

Deleted 2026-10-02 (recoverable from git history):
`02_solenoid_model_checks_006_008.md`,
`03_aperture_emittance_studies_009_014.md`, `05_spin_polarization.md`.

Note `08_second_order_chromaticity_source.md` is referenced by the deleted 004cc
but was never written. Its subject — the Q''y amplification the half-straight
optics match produces by leaving the QD0 -> sdy1 vertical phase advance free —
is still an open question on this branch.

Removed 2026-07-15: the `kill_higher_order_{upstream,downstream}_{ip}` knob
(zeroed sextupole-and-above multipole content for one half of one IP's main
solenoid; was wired into 004b, the since-deleted `lattice_knobs.py`,
009/010/013/014 and 004d) has
been stripped from all of those files at the user's request. It was never
committed (added and removed within the same uncommitted working-tree
session), so there is no git history to recover it from. Its note
(`05_kill_higher_order_half_solenoid.md`) is deleted; if this line of
investigation (effect of higher-order multipole content on DA) is revisited,
it will need to be rebuilt from scratch.
