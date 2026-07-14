# fcc_ee_solenoid — directory overview (for Claude, read this first)

Written 2026-07-10. Purpose: let a future session skip re-reading every script.
If a referenced file/function no longer exists, treat these notes as stale for
that detail and re-check the source.

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
  sextupole-like order incl. `sext_amp` knob), more expensive.
- **VariableSolenoid** (`xt.VariableSolenoid`): linear `ks_profile` (2-point
  linear ramp) + one dipole kick (`knl`/`ksl`) per slice. Cheaper, linear-only.

Naming convention seen everywhere: **"SB"** = SplineBoris, **"VarSol"** =
VariableSolenoid. Case names `sb_on` / `varsol_on` / `sb_off` recur across
009/010/014 (`sb_off` = SplineBoris lattice with solenoids+correctors both
switched off, i.e. the "bare machine" baseline using the same JSON).

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

These two *_coupling_corrected.json files are the ones actually consumed by
009/010/011/013/014 (the "current" study family). Everything upstream of them
(000a-003) is legacy/exploratory and not on the live dependency path anymore.

   006, 007, 008: standalone model-validation studies (do NOT read/write any
       of the JSON above; they rebuild field models from scratch each time).
       See 02_solenoid_model_checks_006_008.md.

   009, 010, 011, 013, 014: aperture/emittance studies reading the two
       *_coupling_corrected.json lattices. See
       03_aperture_emittance_studies_009_014.md.
```

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
- `lattice_knobs.py` — `set_lattice_knobs(line, with_solenoids, with_correctors,
  sext_amp=1.0)`: single entry point used by 009/010/014 to flip all 4 IPs'
  `on_sol_*` / correction knobs together and set the `sext_amp` sextupole-
  amplification knob. (011 and 012 have their own inline copy/no-op instead of
  importing this — minor duplication, not a bug.)
- `aperture_grid.py` — `initial_conditions_grid(study="MA", ...)`: polar grid
  generator for momentum-acceptance initial conditions (used by 009 only; 010
  has its own DA-specific grid builder inline).
- `aperture_study_io.py` — save/reload/replot helpers for DA/MA/EMIT `.npz`
  data + PDF figures. Used by 009, 010, 014, and replotted by 012. PDFs go to
  `/home/simonfan/cernbox/Pictures/FCC_Solenoid_Studies` (outside the repo);
  `.npz` raw data go to `examples/fcc_ee_solenoid/data/`.

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
- `01_lattice_construction_000_004d.md` — 000a/000b/001a/001b/002/003/004a/004b
  (x2)/004c/004d in detail.
- `02_solenoid_model_checks_006_008.md` — 006/007/008 standalone validation
  scripts.
- `03_aperture_emittance_studies_009_014.md` — 009/010/011/012/013/014 +
  `aperture_grid.py` + `aperture_study_io.py` + `lattice_knobs.py`.
- `04_bz_ramp_coupling_amplification.md` — `bz_ramp_field.py`,
  `004a2_build_solenoid_bz_ramp.py`, `004b2_install_solenoid_bz_ramp_in_fcc_ring.py`,
  `scan_bz_ramp_amp.py`: a Maxwell-consistent linear-Bz-ramp perturbation used
  to probe whether the detector solenoid's x-y coupling is genuinely small or
  a fragile cancellation, plus raw-coupling and phase-advance (`dmux`/`dmuy`)
  scan findings.
