# fcc_ee_solenoid — directory overview (for Claude, read this first)

Written 2026-07-10. Last reviewed against the code 2026-10-14 (notes 01, 05,
09, 10 and this file updated; 02, 03, 08 only checked for missing files and
functions). Purpose: let a future session skip re-reading every script.
If a referenced file/function no longer exists, treat these notes as stale for
that detail and re-check the source.

**Lattice filenames are tagged.** Since 2026-07-24, every lattice/template
file carries `_{FIELD_TAG}{ORDER_TAG}` (e.g. `_2T`, `_3T`, `_3T_o2`), plus an
optional `_{output_tag}` (e.g. `_mainscale`) from 004b/004c's `--output-tag`.
Where an older passage writes an untagged name such as
`fccee_z_lcc_splineboris_solenoids_coupling_corrected.json`, read it as the
tagged family.

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
       004d: analysis/plots of the SplineBoris corrected lattice (needs BOTH
             the 2T and 3T corrected lattices, see note 01)

   (filenames above are shown untagged -- see the tagging note at the top)

   Side scripts (none write a lattice; most read the corrected ones):
       004cc: one-IP, right-half sandbox of the 004c correction, with W and
              phase diagnostics (notes 09/10)
       004dd: chromatic phase advance mu(s), mu'(s), mu''(s) (note 08)
       004e:  sextupole-content window study on the isolated main solenoid
              (standalone, reads no pipeline JSON)
       004f/004g/004h/004i/004j: comp_b_scale / main_b_scale scans and the
              coupling-Jacobian SVD diagnostic (notes 06/07, both MISSING
              from this checkout -- see the note index). CAVEAT: all five
              were last changed 2026-09-09 and re-solve coupling the OLD
              way (full straight,
              ~84 skew knobs vs 12 targets at both boundaries, init at
              dy_match), not 004c's per-side scheme since 2026-09-15.
       004k:  off-momentum tune footprint on a resonance diagram
       004l:  A/B of the optics match with/without the sdm1 targets
       004m:  beta functions of the old (pre mid-bend quads, pre rcond) vs
              current corrected rings, old ones read from git
       017:   tune vs delta, scan+fit vs twiss dqx/ddqx
       See "Scripts without a dedicated note" below.

These two *_coupling_corrected.json files are the ones actually consumed by
009/010/011/013/014 (the "current" study family). Everything upstream of them
(000a-003) is legacy/exploratory and not on the live dependency path anymore.

   006, 007, 008: standalone model-validation studies (do NOT read/write any
       of the JSON above; they rebuild field models from scratch each time).
       See 02_solenoid_model_checks_006_008.md.

   009, 010, 011, 013, 014, 015, 018: aperture/emittance/spin studies
       reading the two *_coupling_corrected.json lattices. See
       03_aperture_emittance_studies_009_014.md and 05_spin_polarization.md.
       **2026-09-17: 018_emittance_and_polarization.py merges 014 and 015**
       into ONE tracking run per case (they differed only in twiss flavour,
       bunch initial emittance and spin IC), and 013 now drives 018 in place
       of 014 -- so a default 013 run yields DA + MA + EMIT + POL for the
       cost of three tracking passes. 014 and 015 are still runnable
       standalone (015 has changed since; see 05_spin_polarization.md).
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
  importing this — minor duplication, not a bug.) Also `set_solenoid_offset`
  (rigid x/y shift of the 4 main solenoids) and, added 2026-07-15,
  `install_extra_sextupole(line, k2l=...)` — opt-in (no-op at `k2l=0.0`) thin
  extra sextupole inserted into each IP's main solenoid at a fixed
  SplineBoris slice boundary near s=-1.23 m from the IP; used by 009/010 via
  their `--extra-sext-strength` CLI flag. See
  `03_aperture_emittance_studies_009_014.md` for details.
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
  passthrough at all (unlike `save_emitt_study`) — see
  `03_aperture_emittance_studies_009_014.md`.
  **2026-08-03: `--max-transverse-order` added**, mirroring `--b0`/`field_tag`
  exactly. `order_tag(n)`/`add_max_order_argument()` in `solenoid_params.py`
  (default `n=4`, tag `''` at the default, `'_o2'` etc. otherwise) cap the
  transverse multipole order baked into each installed `xt.SplineBoris`
  element via `build_splineboris_line`'s `max_transverse_derivative_order_
  for_spline` (see `spline_boris_setup.py`) — this directly sets
  `element.multipole_order`, i.e. how many `bx`/`by` polynomial terms the
  Boris pusher evaluates per step. Lower orders (below 2, the sextupole row
  the 004b `sext_amp` knob scales) drop that multipole content in exchange
  for cheaper tracking. SplineBoris-only — VariableSolenoid is linear-only
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
- `aperture_grid.py` — `initial_conditions_grid(study="MA", ...)`: polar grid
  generator for momentum-acceptance initial conditions (used by 009 only; 010
  has its own DA-specific grid builder inline).
- `aperture_study_io.py` — save/reload/replot helpers for DA/MA/EMIT `.npz`
  data + PDF figures. Used by 009, 010, 014, and replotted by 012. PDFs go to
  `/home/simonfan/cernbox/Pictures/FCC_Solenoid_Studies` (outside the repo);
  `.npz` raw data go to `examples/fcc_ee_solenoid/data/`. As of 2026-07-15,
  `PLOT_DIR` is `Path.home() / "cernbox/Pictures/FCC_Solenoid_Studies"` (was
  hardcoded to `/home/simonfan/...`, which broke on other machines/users —
  same synced cernbox folder, just resolved via the current user's home dir
  now).

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
  only) `on_sol_coupling_corr_{ip}`. Since 2026-09-15 each of those three
  drives a `..._corr_left_{ip}` and a `..._corr_right_{ip}` child, which are
  the actual generated knobs (one per half-straight match, see note 01).
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
- `radial_steering.py` is a stray: it imports `dx_manual` from
  `examples/`, which does not exist in this repo, so it cannot run here. It
  is only referenced in a comment in `lattice_knobs.py`. (The stray
  `Untitled` file mentioned here before is gone.)

## Note index
- `01_lattice_construction_000_004d.md` — 000a/000b/001a/001b/002/003/004a/004b
  (x2)/004c/004d in detail.
- `02_solenoid_model_checks_006_008.md` — 006/007/008 standalone validation
  scripts.
- `03_aperture_emittance_studies_009_014.md` — 009/010/011/012/013/014 +
  `aperture_grid.py` + `aperture_study_io.py` + `lattice_knobs.py`.
- `04_bz_ramp_coupling_amplification.md` — **REMOVED 2026-07-22** (files
  `git rm`'d at user's request, no longer used; note kept for history only).
  Was a Maxwell-consistent linear-Bz-ramp perturbation used to probe whether
  the detector solenoid's x-y coupling is genuinely small or a fragile
  cancellation, plus raw-coupling and phase-advance (`dmux`/`dmuy`) scan
  findings.
- `05_spin_polarization.md` — `015_spin_polarization.py`: radiative
  spin-polarization buildup/depolarization study (Sokolov-Ternov `P_inf`/
  `tau_pol` from Twiss `polarization_analysis=True`, `tau_depol` from a
  linear fit of tracked polarization decay), structurally a sibling of
  `014_emittance_evolution.py`. Not part of the 000-014 pipeline numbering.
  **Also covers `018_emittance_and_polarization.py`** (2026-09-17), the
  merge of 014+015 into a single tracking run, why spin costs nothing when
  it is off, the `__bunchdiv<F>` provenance tag, the `--seed` determinism
  path, and the measurement that the tracked `tau_depol` is noise-dominated
  (SNR ~ 1).
- `07_main_b_scale_scans.md` — **MISSING from this checkout** and in no git
  commit, because `claude_notes/` was gitignored until 2026-10-14. It may
  still exist on another machine; see "Note history" below. The scale knobs themselves are now summarised in note 01
  (004b section). What it covered: the `main_b_scale` / per-side
  `comp_b_scale_{side}_{ip}` knobs (new in 004b/004c, `--output-tag
  mainscale` lattices) and the three scan scripts that use them:
  `004h_main_b_scale_scan.py` (004f analogue, floats the per-side comp
  knobs), `004i_comp_weight_sweep.py` (comp-knob-weight diagnostic), and
  `004j_main_b_scale_suite.py` (combined 2 T + 3 T suite: emittance / tune
  / chromaticity / `C^-` vs `main_b_scale`, plus IR/straight β /
  coupled-β / dispersion / beam-size profiles + skew-corrector strengths;
  plain skew-only unit-weight coupling re-solve). Also documents **which
  knobs 004c/004f/004h/004j each vary** (the same ~84 skew + 24 orbit
  knobs per IP in all four; only the solver settings and frozen-vs-raw
  knob handling differ) and **what every line/shading on the 004j plots
  means** — the grey dashed verticals on the IR figures mark the *orbit*
  correctors, not the coupling skews, which is the easiest thing in this
  study to misread.
- `06_coupling_matching_convergence.md` — **MISSING**, same as 07. What it
  covered: why `004f_comp_b_scale_scan.py`'s
  per-scan-point coupling re-solve (84 skew-quad vary knobs vs. 12 targets)
  is slow/sometimes fails to converge (ill-conditioned Jacobian, SVD
  diagnosis), the `004g_coupling_svd_diagnostic.py` tool built to inspect
  it, the `broyden=True` fix applied, and a ranked list of further options
  for outright non-convergence (not yet applied). Note the `max_step` fix
  recorded there as applied on 2026-09-03 was **backed out of 004j on
  2026-09-05** in favour of a smaller finite-difference `step` + much
  tighter tolerances — read that section's "Superseded 2026-09-05" note
  before trusting the surrounding text. `004f`/`004g` themselves
  are not otherwise documented in this pipeline overview yet.

- `08_second_order_chromaticity_source.md` — (2026-09-15) why the 004c
  corrected rings have Q''y ~ +1.5e4..1.8e4 (bare -149): the half-straight
  optics match leaves the vertical phase advance QD0 -> sdy1 (vertical LCC
  -I sextupole pair) free, and Q''y ~ 2.4e7 * (phase error) per side
  (ipj left dominates). Includes the phase-trombone recipe, the IP-marker
  phase pitfall, and how to read 004dd's mu''(s) (realised in the arcs, not
  sourced there). sdm1 sextupoles are chromatically irrelevant.
  **Its "Fix" section is superseded — see note 09.**

- `09_correcting_the_q2y_source.md` — (2026-09-16) the attempts to *correct*
  the note-08 source, all of which failed, and the response-matrix/null-space
  measurement of why: the −I pair needs both a phase and a strength condition,
  the half-straight optics match constrains neither, and within the null space
  of its existing targets the two trade against each other. Also: what 004cc
  is now (IP_NAME selector, per-knob decomposition, W and QD0->sdy1 phase
  printouts), how to target `wy_chrom` in a match and why it must be staged,
  and a reference section on what the Montague W functions are.
  **Its "one hypothesis checked and killed" section is wrong — see note 10.**

- `10_conditioning_the_optics_match.md` — (2026-09-16/17) **the fix that
  worked, and the end of this thread.** The Q''y problem was a *conditioning*
  problem, not a physics one: the half-straight optics match is rank-deficient
  (15 knobs, 8-9 targets, condition number 3.7e7) and `xdeps` defaults to
  `rcond=1e-14`, so it inverted the near-null directions and landed on an
  arbitrary null-space point each run — which is why Q''y flipped sign between
  otherwise identical runs. Two solver changes, no new targets:
  `OPTICS_RCOND = 1e-6` and a staged pass-1 solve (`sext` targets disabled,
  then re-enabled). Took ipg left from max|k1| 4.5e-3 / phase error 2.6e-3 to
  4.7e-5 / 5.3e-5, and made ring Q''y **predictable from the eight QD0->sdy1
  phase errors to 0.2%** via note 08's 2.42e7 calibration. Also records four
  measured dead ends (knob limits, `step=1e-6`, the `bety`-at-sdm1 target,
  tighter tolerances), why "penalty is not quality", the diagnostics added to
  004c (`max|knob|` column, `--ips`, figure 4's Q(δ) plot), and the fact that
  that 004cc carried a value-shift bug in its sextupole targets. **That bug is
  gone (2026-10-14 check: never in any commit).** 004cc now runs two
  independent correction chains, with and without the `sext` targets; see
  notes 09/10.

## Scripts without a dedicated note (summarised 2026-10-14 from docstrings)

- `000c_field_map_y_s_plane.py` — figure: main-solenoid (By, Bz) field
  arrows in the y-s plane at x=0, coloured by |B|, from `SolenoidField`.
- `004aa_splineboris_field_content.py` (2026-09-24) — two figures, main and
  compensation solenoid: Bs/Bx/By along s at x=y=0, and d^k Bx/dx^k,
  d^k By/dx^k per order k, read from the templates 004a wrote. The
  curves are the exact quartic the Boris pusher integrates
  (`_quartic_from_hermite`, checked against `get_field` to 1e-16). A
  cheaper alternative to 004a's ~100 check figures.
- `004e_sextupole_window_study.py` — zeroes the order-2 bx/by row of the
  isolated main solenoid in an s-window (default 0.7-1.245 m) and compares
  beta_y- and beta_y^(3/2)-weighted cumulative d^2Bx/dx^2 integrals against
  the nominal solenoid. d^2Bx/dx^2 is antisymmetric about the IP, so its
  plain integral cancels but a beta-weighted one need not. Standalone: reads
  and writes no pipeline JSON.
- `004k_tune_diagram_vs_delta.py` — (Qx, Qy) footprint vs delta from
  periodic 4D twisses of the 2T/3T corrected rings (+ bare), on a resonance
  diagram up to `--max-resonance-order` (default 3), in absolute tune.
  `--vs-old` adds the previous lattice generation read from git at
  `--old-rev` (default `f89a90b2c`) via `aperture_study_io.old_lattice_path`.
- `004l_sextupole_target_ab.py` (2026-09-15) — runs one IP's optics match
  with and without the sdm1 `sext` targets (mid-bend quads installed in
  both) and compares convergence, boundary residuals, bety at the sextupole
  and the worst beta-beat. Writes nothing; meant for `%run` in IPython.
  Written when the 3T match with the sext targets stopped converging
  (penalty ~57.75) and `generate_knob()` froze a near-zero knob, i.e. the
  optics correction had silently become a no-op.
- `004m_old_vs_new_beta.py` (2026-09-23) — beta functions of the old vs
  current corrected rings at 2T and 3T. Old = `OLD_REV` `f89a90b2c`, the
  newest commit where both rings are pre-mid-bend-quads and
  pre-conditioning-fix. `957c3bea4` is NOT usable as old: there the 3T
  ring was already cut and the 2T was not.
- `aperture_study_io.old_lattice_path(rev, filename)` (added 2026-09-23) —
  extracts `examples/fcc_ee_solenoid/<filename>` at a git rev into
  `data/old_lattices/{rev}_{filename}` (gitignored) via `git cat-file`,
  writing atomically through a `.part` file. Shared by 004k and 004m. Needed
  because each 004c run overwrites the corrected lattices in place.
- `009_parametric_version.py` / `009b_table_results.py` /
  `009c_table_local_sol.py` — a separate, parametric single-IP study (ipa
  only) after Boscolo, Ciarma, Burkhardt (NIM A 1083 (2026) 171135). It
  builds the solenoid from analytic `TiltedSolenoid`s with free `B0`,
  half-length, corrector `ds_start` and optional local anti-solenoids
  (`B0_local`, `l_local`), corrects it, and dumps results (incl.
  `gemitt_y`) to `results/*.json`. 009b tabulates `gemitt_y` vs B0 x
  half-length, and 009c vs B0_local x l_local. Not connected to the
  004-pipeline lattices. Despite the name, it is unrelated to
  `009_momentum_acceptance.py`.
- `017_tune_vs_delta.py` — Q(delta) two ways: scan + polynomial fit
  (order `--fit-order`, coefficient x n!), and twiss's own `dqx/ddqx`. Its
  docstring cites `xtrack.twiss._get_chromatic_functions`; after the twiss
  refactor that is in `xtrack/twiss/chromatic_functions.py`.

## Note history: why some notes are missing

`.gitignore` had `**/claude_notes/` until 2026-10-14, when it was removed at
the user's request. Notes 00-05 had been committed regardless and survived.
Everything else lived only in whichever working tree it was written in. 08-10
are in this checkout; 06 and 07 are not, and are in no commit on any branch.
Some of this work was done on a second, Linux machine (`/home/simonfan/...`,
see the `PLOT_DIR` history above), so 06/07 may still be there. Check
before rewriting them. `004d` and `004j` still cite them in comments. From
2026-10-14 on, new notes here are ordinary untracked files and need committing
like any other.

Removed 2026-07-15: the `kill_higher_order_{upstream,downstream}_{ip}` knob
(zeroed sextupole-and-above multipole content for one half of one IP's main
solenoid; was wired into 004b/`lattice_knobs.py`/009/010/013/014/004d) has
been stripped from all of those files at the user's request. It was never
committed (added and removed within the same uncommitted working-tree
session), so there is no git history to recover it from. Its note
(`05_kill_higher_order_half_solenoid.md`) is deleted; if this line of
investigation (effect of higher-order multipole content on DA) is revisited,
it will need to be rebuilt from scratch.

**History rewrite, 2026-10-14.** A third-party field map was purged from the
fork's history. Every commit on `fcc_nonlocal_paper`, `tube_approach`,
`fcc_coupling` and `fcc_nonlocal_solenoid` descending from `6d0f3cc2e`
(2026-08-28) or `37daa77df` (2026-10-09) got a new hash (57 commits);
everything older is unchanged. Hashes cited in these notes and in 004k/004m
were updated to the new values the same day. Any other hash of such a commit
(e.g. in an old message or log) no longer exists in the repo; the pre-rewrite
clone `xtrack_OLD_DO_NOT_PUSH` still has them.
