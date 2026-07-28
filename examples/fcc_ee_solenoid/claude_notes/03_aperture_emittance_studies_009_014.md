# 009-014: aperture and emittance-evolution studies

Written 2026-07-10, updated same day after the 014 fitting/plotting changes
(free-fit eq. emittance + damping rate, Twiss-vs-fit comparison box on plots).
Updated 2026-07-13: fixed a real y-axis bug in 010 (DA), then replaced the
underlying DA grid geometry itself (physical-radius circle -> independent
per-plane sigma ellipse) after the first fix made the plot technically
correct but visually useless, and added an x/y direction split to 009 (MA)
— see their sections below and the `aperture_study_io.py` section for the
filename/schema changes that went with all three.
Companion to `00_overview.md`. All of 009/010/011/013/014 read one or both of
the final corrected lattices from `01_lattice_construction_000_004d.md`:
`fccee_z_lcc_splineboris_solenoids_coupling_corrected.json` and
`fccee_z_lcc_varsol_solenoids_coupling_corrected.json`.

## Common per-script pattern (009/010/014)

All three share the same `*_CASES` list-of-dicts pattern (`name`, `model`
("SB"/"VarSol"), `lattice_json`, `with_solenoids`, `with_correctors`, `title`,
plus study-specific extras) with 3 standard cases:
- `sb_on` — SplineBoris lattice, solenoids + correctors on.
- `varsol_on` — VariableSolenoid lattice, solenoids + correctors on.
- `sb_off` — SplineBoris lattice, solenoids AND correctors off (bare-machine
  baseline; note there is no `varsol_off` — off/off collapses to the same
  physics regardless of which lattice file since solenoids are zeroed, so only
  one baseline case is kept).
Each script: loads lattice, `line.cycle("ipa")`,
`set_lattice_knobs(line, with_solenoids=..., with_correctors=..., sext_amp=...)`
(from `lattice_knobs.py`), rebuilds tracker, configures radiation
(`anomalous_magnetic_moment=0.00115965218128`, `configure_radiation(model=...)`),
tracks, saves a plot + `.npz` via `aperture_study_io.py`, has a
`--cases/--list-cases/--b0/--n-turns/--sexamp/--no-show` CLI (009 additionally
has `--directions`, see below), and a `main()` guarded by
`__name__ == "__main__"`. `013_run_da_and_ma.py` drives 009+010+014 as
subprocesses with a shared CLI.
- **`--b0` field-strength flag (009/010 added 2026-07-28, mirroring
  014/015)**: previously 009/010 built their `MA_CASES`/`DA_CASES` list once
  at import time from the module-level `solenoid_params.FIELD_TAG`
  (`MAIN_SOLENOID_B0`), so selecting a field-strength case meant editing
  `solenoid_params.py` rather than passing a flag. Now both scripts have
  `_build_ma_cases(tag)`/`_build_da_cases(tag)` (same shape as 014's
  `_build_emitt_cases`/015's `_build_pol_cases`: a `_lattice_paths(tag)`
  helper plus a case-list builder), called from `main()` with
  `tag = field_tag(args.b0)` after `add_b0_argument(parser, ...)`. Also added
  the same missing-lattice `SystemExit` message 014/015 already had
  (`_run_momentum_acceptance`/`_run_dynamic_aperture` now take a `tag` kwarg
  and check `lattice_json.exists()` before loading). `--list-cases` prints
  the resolved `Field tag: ... (--b0 ...)` line first, same as 014/015.
  `013_run_da_and_ma.py` also gained its own `--b0` (default
  `MAIN_SOLENOID_B0`) and unconditionally forwards `--b0 <value>` to all
  three subprocesses (DA/MA/EMIT) — unlike most of 013's other forwarded
  flags, this one isn't gated by `is not None` since `add_b0_argument`
  always supplies a default, so it's never `None`.
- **Extra sextupole knob (009/010 only, added 2026-07-15)**:
  `--extra-sext-strength K2L` (default `0.0`, off) calls
  `lattice_knobs.install_extra_sextupole(line, k2l=...)` right after
  `set_solenoid_offset(...)` and before the tracker is (re)built. It inserts
  one zero-length `xt.Sextupole` per IP (named `extra_sext_{ip_name}`), with
  its whole strength given as an integrated `knl[2] = k2l` kick rather than
  via `k2`+`length` — required so the insertion lands exactly on an existing
  `sol_slice_{ip}_048`/`sol_slice_{ip}_049` boundary (s ~= -1.223 m from the
  IP, the closest slice edge to the user-requested -1.23 m) without needing
  to split either neighbouring SplineBoris slice, which doesn't support
  slicing (`SplineBoris` has no `_thick_slice_class`). A no-op when
  `k2l == 0.0`, so the lattice is byte-for-byte unchanged by default. 014
  (EMIT) does not have this flag.

## 009_momentum_acceptance.py (MA study)
- **x/y direction split (added 2026-07-13)**: previously ran a single fixed
  `theta=pi/4` diagonal scan and only ever plotted/saved the x-component,
  even though `y_normalized` was tracked too — the y aperture at that
  diagonal amplitude was silently never examined. Now `MA_DIRECTIONS = {
  "x_only": theta=0.0, "y_only": theta=pi/2}` (pure on-axis scans: at
  `theta=0` the grid is on-axis x with `y_normalized=0`; at `theta=pi/2` it's
  on-axis y with `x_normalized=0`). CLI: `--directions x_only y_only`
  (`choices=list(MA_DIRECTIONS)`); if omitted, `_select_directions` defaults
  to **`x_only` only** (changed 2026-07-15 at user request — y MA is off by
  default now; pass `--directions y_only` or `--directions x_only y_only` to
  turn it on). `013_run_da_and_ma.py` forwards this via its own
  `--ma-directions` flag, which has the same x_only-only default (013 only
  passes `--directions` through to 009 when `--ma-directions` is explicitly
  given, so 009's default applies whenever it's omitted).
- Grid: `initial_conditions_grid` (from `aperture_grid.py`) — polar in
  `(x_normalized, y_normalized)` at `theta in [theta, theta]` (i.e. a **line**,
  not a fan, `nn_x_theta` defaults to 1; `theta` comes from
  `MA_DIRECTIONS[direction]`, passed as `min_x_theta=max_x_theta=theta`) with
  `nn_y_r=NN_Y_R=25`, replicated over `DELTA_INITIAL_VALUES =
  linspace(-35*ENERGY_SPREAD, +35*ENERGY_SPREAD, 51)`
  (`ENERGY_SPREAD=3.9e-4`). So each direction is really an amplitude x
  momentum-offset 2D scan, 25*51 = 1275 particles by default — running both
  directions means 2550 particles total across two separate tracked runs
  (not one combined run).
- `NEMITT_X=6.33e-5`, `NEMITT_Y=1.69e-7`, `GLOBAL_XY_LIMIT=5e-2` (5 cm —
  intentionally tight since MA only cares about longitudinal loss, not large
  transverse excursions; see 010's comment on why it uses a looser limit).
  `N_TURNS=10_000` default.
- **`max_y_r` split per direction, added 2026-07-13** (same underlying issue
  as 010/DA's y-axis, spotted by the user after the DA fix): originally
  `max_y_r=MAX_Y_R` (a single module constant, 30) was passed unchanged for
  both directions, so `y_only` scanned amplitudes only up to 30 sigma_y —
  nowhere near where the real y aperture boundary sits, since sigma_y is so
  tiny (~219x smaller than sigma_x, see 010's section). Symptom: a short
  smoke test showed `y_only` losing 0 particles while `x_only` lost ~29%, at
  a turn count where that's clearly a scan-range problem, not a genuine
  physics result. Fix, mirroring 010's `Y_AXIS_SCAN_FACTOR`: `MA_DIRECTIONS`
  now carries a per-direction `max_y_r_factor` (`x_only: 1.0`,
  `y_only: Y_AXIS_SCAN_FACTOR = 6.0`, same self-chosen/not-derived-from-
  sigma_x/sigma_y value currently used in 010), and
  `_run_momentum_acceptance` computes `max_y_r = MAX_Y_R *
  MA_DIRECTIONS[direction]["max_y_r_factor"]` before building the grid and
  saving. Verified: `sb_off`, 200-turn smoke test — `x_only` unaffected
  (`max_y_r=30`, frac_lost=0.565); `y_only` now uses `max_y_r=180` and shows
  meaningful loss structure spread across the full 0-180 sigma_y range
  (frac_lost=0.369), instead of the near-zero-loss result from before the
  fix. If `Y_AXIS_SCAN_FACTOR` is retuned in 010, consider whether 009's copy
  of the same constant should follow (they are independent module-level
  constants in separate scripts, not shared).
- `line.build_particles(method="6d", delta=tt_init.delta_init,
  x_norm=tt_init.x_normalized, y_norm=tt_init.y_normalized)`.
- Plot: `delta_init` vs the direction's own amplitude axis (`x_normalized` for
  `x_only`, `y_normalized` for `y_only` — picked via
  `MA_DIRECTIONS[direction]["axis_key"]`), survived (dots) vs lost (colored by
  `at_turn`) with a colorbar; title includes `(x-only)`/`(y-only)` and reports
  `frac_lost`/`at_turn_mean`.
- Saved via `save_ma_study(..., direction=direction, ...)` — filenames are now
  tagged `MA_X_.../MA_Y_...` (see `aperture_study_io.py` section below).

## 010_dynamic_aperture.py (DA study)
- **Different grid from 009**: `_build_da_initial_conditions` builds its own
  polar grid directly with `xp.generate_2D_polar_grid` (bypassing
  `aperture_grid.py`'s `initial_conditions_grid`, whose docstring/comment in
  010 says it "has a column-length bug for single delta" — i.e. 010
  intentionally avoids the shared helper). `theta in [0, pi/2]`
  (quarter-plane fan), per-case `nn_y_r`/`nn_x_theta` (sb_on: 25x30,
  varsol_on: 30x50, sb_off: 25x30 — VarSol gets a finer grid, presumably
  because its DA boundary needed more resolution).
- **Grid-geometry history, both changes made 2026-07-13** (worth knowing
  both steps, not just the current state, since the first "fix" looked
  correct but produced a useless plot):
  1. *Original bug*: `y_norm` was hardcoded `y_hat * 5.0`
     (comment-labeled `sigma_x_over_sigma_y` but ignoring the
     actually-computed ratio), which — measured at the IP — is way off from
     the real `sigma_x/sigma_y ~ 219` (same across sb_on/varsol_on/sb_off
     since the correction scheme preserves this IP optics design regardless
     of solenoid state; `cube_root(219)~6.03` and `sqrt(219)~14.8` don't
     match 5 either, so the 5 wasn't a disguised nonlinear function of the
     real ratio — just wrong). Root cause of the *design*, not just the
     hardcoded constant: the grid was built as a genuine **physical-radius
     circle** — `x_hat, y_hat = xp.generate_2D_polar_grid(r_range=(0,
     MAX_AMP_SIGMA_X), theta_range=(0, pi/2), ...)` gives `x_hat=r*cos(theta)`,
     `y_hat=r*sin(theta)` for `r` up to `MAX_AMP_SIGMA_X` **sigma_x-equivalent
     physical units**, then `y_norm = y_hat * sigma_x_over_sigma_y` converts
     that same physical `r` into sigma_y units for the y-plane. Correct
     conversion math, but it means the scan explores physical y offsets up
     to `MAX_AMP_SIGMA_X * sigma_x` (~320 um), which in sigma_y units is
     ~8760 — a scan range so much larger than where the DA boundary actually
     sits that the interesting structure collapsed to a sliver near y=0 on a
     linear axis (confirmed empirically: a 100-turn smoke test with this
     geometry showed frac_lost=0.91 with almost no visible boundary shape).
  2. *Geometry replaced* (not just the axis label): the scan is now a
     genuine **ellipse in normalized (sigma_x, sigma_y) units** rather than
     a physical-radius circle, so its y-extent can be chosen independently
     of the true sigma_x/sigma_y ratio.
     `_build_da_initial_conditions` calls `xp.generate_2D_polar_grid(
     r_range=(0, 1.0), theta_range=(0, pi/2), ...)` to get a **unit** circle
     (`x_unit, y_unit` in `[0,1]`), then scales each plane independently:
     `x_hat = x_unit * MAX_AMP_SIGMA_X`, `y_hat = y_unit * MAX_AMP_SIGMA_Y`.
     **`MAX_AMP_SIGMA_Y` is deliberately NOT the true sigma_x/sigma_y ratio**
     — after the first attempt (setting it equal to `MAX_AMP_SIGMA_X`, i.e. a
     true circle, 40 sigma_y) turned out to genuinely miss/undersample where
     the DA boundary sits in y, the user clarified the actual intent: they
     want a **self-chosen** (not physics-derived) y-axis amplification,
     same spirit as the original hardcoded `5.0`, just correctly reflected
     in the axis values this time. Final design:
     `Y_AXIS_SCAN_FACTOR = 5.0` (module constant, explicitly commented as
     self-chosen/not sigma_x/sigma_y) and
     `MAX_AMP_SIGMA_Y = MAX_AMP_SIGMA_X * Y_AXIS_SCAN_FACTOR` (= 200 sigma_y
     by default). This reproduces the original factor-5 plot's *visual
     shape* (a nicely-visible circular DA boundary — verified against a
     100-turn `sb_off` smoke test: frac_lost=0.224, boundary visibly curves
     from ~28 sigma_x at y=0 down through ~20 sigma_x at y~125 up to the
     y-axis around 175-200 sigma_y) while the axis now genuinely reads
     "200", not a mislabeled "40". No `sigma_x_over_sigma_y` dependency
     anywhere in the grid build — `_run_dynamic_aperture` still computes
     `sigma_x`/`sigma_y` via `_compute_beam_sizes`, purely for the
     print/plot title annotation (`sigma_x/sigma_y=219` shown there is
     informational only, unrelated to the y-axis scale of the plot itself).
     If `Y_AXIS_SCAN_FACTOR` is ever revisited, do it empirically (rerun and
     check whether the DA boundary is fully captured within the scanned
     range, as was done here) rather than trying to derive it from theory —
     that's the mistake that started this whole fix.
  The table (`tt_init`) still carries `x_hat`/`y_hat`/`x_normalized`/
  `y_normalized` columns for schema stability, but `x_hat == x_normalized`
  and `y_hat == y_normalized` always now (no separate raw-vs-converted
  values any more — both planes are already in their own true sigma units,
  scaled by `MAX_AMP_SIGMA_X`/`MAX_AMP_SIGMA_Y`, straight out of the grid
  builder). Plot y-axis label is `$\hat{y}\,[\sigma_y]$` (unchanged from the
  first fix). `set_aspect("equal", ...)` remains removed on both DA subplots
  (x and y axes are intentionally different scales now, 0-40 vs 0-200, so
  forcing equal aspect would misrepresent the plot again).
- `GLOBAL_XY_LIMIT = 1.0` (1 m, xtrack default) **intentionally looser than
  009's 5 cm** — code comment explains: large beta functions blow up amplitude
  units well past 5 cm even for physical sigma-levels ~O(30), so 5 cm would
  clip particles that should survive.
  `NEMITT_X=6.33e-5`, `NEMITT_Y=1.69e-7` (same as 009), `N_TURNS=10_000`.
- On-momentum only (`delta=0, px_norm=py_norm=0`) — no off-momentum DA scan
  yet; comment says "For off-momentum DA, repeat with delta set to selected
  momentum offsets" (not implemented).
- Two-panel plot: scatter (`x_normalized` vs `y_normalized`, colored by
  `at_turn`) + `pcolormesh` reshaped to `(nn_y_r, nn_x_theta)` — the
  pcolormesh only makes sense because the grid is a structured `(r, theta)`
  mesh, so reshape order matters if grid generation ever changes. Title
  annotates measured `sigma_x`, `sigma_y`, and `sigma_x/sigma_y` per case.
- Saved via `save_da_study` (now also stores `y_normalized` and
  `max_amp_sigma_y` alongside `x_hat`/`y_hat`/`max_amp_sigma_x` in the
  `.npz`, see `aperture_study_io.py` below).

## 011_bunch_tracking.py
- **Not part of the `*_CASES` pattern** — single hardcoded run, SplineBoris
  lattice only, solenoids+correctors on. No `save_*_study` call (doesn't use
  `aperture_study_io.py` at all), no CLI args, meant to be run/edited by hand.
- Has its own local `_set_solenoid_knobs` (duplicate of
  `lattice_knobs.set_lattice_knobs`, not imported — minor drift risk if
  `lattice_knobs.py` changes and this copy doesn't follow).
- Generates a matched Gaussian bunch (`xp.generate_matched_gaussian_bunch`,
  `engine='single-rf-harmonic'`) with **physical FCC-ee Feasibility-Study-Vol-2
  values hardcoded**, not derived from Twiss: `sigma_x=9e-6, sigma_y=40e-9,
  sigma_z=15.2e-3` (comment: 15.2 mm includes beamstrahlung; would be 5.15 mm
  without). The Twiss-derived alternative computation is left in as commented-
  out code just above (`beta_x/beta_y` from `tw`, `geom_emitt_x/y` from
  `nemitt/(beta0*gamma0)`) — useful if you want to switch back to
  self-consistent sigma values instead of the hardcoded FS-vol-2 ones.
  `bunch_intensity=1e11`, `n_part=100`.
- Shifts `particles.zeta/delta` by the **closed-orbit values at the IP**
  (`tw.zeta[0]`, `tw.delta[0]` — nonzero because of solenoid-induced orbit
  distortion; commit history shows this was a deliberately-found bug: commit
  "Error in emittance was caused by nonzero CO at IP" — **this CO-shift
  pattern is the origin of the same fix later applied inside 014's
  `_compute_geometric_emittances`**, see below).
  Also exposes (all zeroed) `x_off/px_off/y_off/py_off/delta_off/zeta_off`
  knobs for manually decentering the bunch — hook for future studies.
- Tracks in chunks of `N_PLOT=100` turns (`N_TURNS=100` total => single chunk
  by default) re-plotting phase space (`x-px`, `y-py`, `zeta-delta`) after
  each chunk via `_plot_bunch_phase_space`. `GLOBAL_XY_LIMIT=1.0`.

## 012_replot_aperture_from_data.py
- Thin CLI wrapper around `aperture_study_io.replot_from_npz` /
  `DATA_DIR`: regenerates PDFs from saved DA/MA `.npz` files without rerunning
  tracking (`--show` to also display interactively; positional `npz_files`
  or, if omitted, every `*.npz` in `data/`).
- **Only handles DA and MA** (dispatches on filename prefix in
  `replot_from_npz`) — EMIT `.npz` files (from 014) are not replottable this
  way; there's no `_plot_emitt_from_arrays` in `aperture_study_io.py`, so
  re-plotting an emittance-evolution run currently requires rerunning 014.

## 013_run_da_and_ma.py
- Orchestrator: runs 010 (DA), 009 (MA), 014 (EMIT) as subprocesses in that
  order via `subprocess.run`, forwarding a shared/overridable CLI
  (`--da-only/--ma-only/--emitt-only`, `--cases` shortcut for `--da-cases`,
  per-study `--{da,ma,emitt}-cases`/`--n-turns` overrides plus a global
  `--n-turns`, `--ma-directions` (added 2026-07-13, forwarded straight to
  009's `--directions`, default: `x_only` only as of 2026-07-15 — pass
  `--ma-directions y_only` or `--ma-directions x_only y_only` to include the
  y MA scan),
  `--n-part` for EMIT only, `--sexamp`, `--extra-sext-strength` (added
  2026-07-15, forwarded to 009/010's own `--extra-sext-strength`; not
  forwarded to 014/EMIT), `--b0` (added 2026-07-28 via
  `solenoid_params.add_b0_argument`, default `MAIN_SOLENOID_B0`; forwarded
  unconditionally to all three subprocesses since it always has a default),
  `--no-emitt` to skip EMIT,
  `--show`). No `--da-n-turns` etc. defaults to the global `--n-turns` unless
  explicitly overridden. Despite the module docstring/filename saying
  "DA and MA", it now also runs the emittance-evolution study (014) by default
  unless `--da-only`/`--ma-only`/`--no-emitt` is passed — **the filename is
  stale relative to current behavior**.

## 014_emittance_evolution.py (EMIT study) — most actively developed script
Purpose: track a matched bunch under **quantum** synchrotron radiation for
many turns, measure the geometric emittance decay per plane, and fit it
against the exponential-damping-to-equilibrium model, comparing the fit
against the Twiss-predicted equilibrium/damping.

### Physics/tracking setup
- `N_TURNS=10_000` (heavier default than 009/010's aperture scans since it's
  full 6D quantum-radiation tracking), `N_PART=1000`, `GLOBAL_XY_LIMIT=1.0`.
- `_configure_radiative_tracking`: sets `anomalous_magnetic_moment`, then
  `configure_radiation(model="mean")` + `compensate_radiation_energy_loss()`
  for the **Twiss** pass (so damping/equilibrium predictions come from a
  self-consistent mean-radiation ring), separately from the actual tracking
  which switches to `configure_radiation(model="quantum")` right before
  `line.track(...)` — i.e. two different radiation models are used
  deliberately: mean for Twiss, quantum for tracking.
- Bunch generation: `xp.generate_matched_gaussian_bunch(nemitt_x=
  tw.eq_nemitt_x/3, nemitt_y=tw.eq_nemitt_y/3, sigma_z=sqrt(tw.eq_gemitt_zeta/3
  * tw.bets0), engine='single-rf-harmonic')` — **the `/3` factor is
  deliberate**: starting well below equilibrium (1/3 of eq. emittance) so the
  exponential approach to equilibrium is clearly visible over `N_TURNS`. This
  `/3` is also exactly why the "Twiss-predicted initial emittance" in the
  fit-box (`twiss_eps_init = [eq_x, eq_y, eq_z]/3.0`, computed at
  `_run_emittance_evolution` around the `_plot_emittance_evolution_figure`
  call) is defined that way — if this generation factor ever changes from `/3`
  to something else, that line must be updated to match or the Twiss-column
  in the plot box will silently show the wrong "initial" value.
- **Closed-orbit correction** (`particles.zeta += tw.zeta[0]`,
  `particles.delta += tw.delta[0]`): same CO-nonzero-at-IP fix noted in 011's
  section above — required because the solenoid-corrected lattice still has
  small residual closed-orbit offset at the IP; without this shift the
  measured emittance would include a spurious contribution from bunch/CO
  mismatch (this was an actual bug found and fixed per git history, see
  `00_overview.md`'s commit list).
- `_compute_geometric_emittances(mon, tw, n_turns)`: for each turn, subtracts
  the **closed-orbit values at turn-0** (`x_co, px_co, ..., delta_co` all from
  `tw` row 0) before computing the Courant-Snyder invariant
  `0.5*(gamma*u^2 + 2*alfa*u*u' + beta*u'^2)` for x/y and the longitudinal
  analog `0.5*(zeta^2/bets0 + bets0*delta^2)` for zeta — this centering is the
  key correctness fix; omitting it reintroduces the CO-driven bias bug.
  Handles particle loss via `mon.state[:, t] > 0` masking (NaN emittance for
  turns where all particles are lost — doesn't happen in practice at these
  amplitudes but is defensive).

### Fitting (current behavior, changed 2026-07-10)
- `_fit_damping_rate(turns, gemitt, eps_eq_guess, eps_init)`: fits **both**
  `eps_eq` and damping rate `alpha` via `scipy.optimize.curve_fit`, with model
  `(eps_init - eps_eq) * exp(-alpha * n) + eps_eq` — **`eps_init` is fixed**
  to the actual tracked initial value (`gemitt[0]`), NOT fitted; `eps_eq_guess`
  (the Twiss equilibrium value) is only the initial guess `p0`, the fit is
  free to move away from it. `alpha` p0 guess is `2e-3` (turn^-1, positive
  sign convention — note this differs from the *old* pre-2026-07-10 version
  which fit a single signed rate `a` with `exp(+a*n)`, `a<0`; the current
  convention is `exp(-alpha*n)`, `alpha>0`, so `tau = 1/alpha` turns is a
  positive damping time). Returns `(eps_eq_fit, alpha_fit)` tuple, or
  `(nan, nan)` if fewer than 3 finite points.
- Twiss-side comparison quantities used for both printing and the plot box:
  `alpha_twiss = 2 * damp_turns` (damp_turns = `tw.damping_constants_turns`
  per plane), `tau_twiss = 1/alpha_twiss` — the factor of 2 matches the
  `_analytic_emittance` model `(eps_init-eps_eq)*exp(-2*damp_turn*n)+eps_eq`
  used for the dotted "Twiss -2λt" reference curve, i.e. `damping_constants_turns`
  from xtrack twiss is defined as amplitude damping rate, and emittance
  (~amplitude^2) damps at twice that rate.
- Per-plane console print block (symbols `x`, `y`, `ζ`): prints
  `ε_{sym},0` (fixed init, from tracking), `ε_{sym},eq` (fit, with Twiss value
  in brackets), `α_{sym}` (fit, 1/turn), `τ_{sym}` (fit, turns, with Twiss
  `damp_turns` in brackets).

### Plotting (current behavior, changed 2026-07-10)
`_plot_emittance_evolution_figure`: 3 stacked subplots (x/y/ζ), each with:
- tracked curve, dashed Twiss-eq horizontal line, dashed fit curve (using
  **fitted** `eps_eq`/`alpha`, not Twiss), dotted Twiss `-2λt` analytic curve.
- Legend pinned `loc="upper left"`.
- A bottom-right (`loc` via axes-fraction `(0.98, 0.03)`, `ha="right",
  va="bottom"`) rounded text box listing **both** fit and Twiss values for
  `ε_{sym},0`, `ε_{sym},eq`, `α_{sym}`, `τ_{sym}` (fit block first, then a
  "Twiss:" block) — deliberately placed bottom-right/legend-top-left per user
  preference, since the emittance curve empirically leaves that corner
  emptiest (decay-to-plateau shape has spare room bottom-right, spare room
  top-left before the curve rises).
- If a per-plane fit failed (`alpha_fit` non-finite), the box prints
  `fit: (failed)` instead of raising.

### Saved data (`save_emitt_study`, see `aperture_study_io.py` below)
Current schema stores `fit_eps_eq`, `fit_alpha`, `fit_tau`, `fit_eps_init`
(replacing an older single `fit_a` array from before the free-eq-emittance
fit change) — **no other script reads these keys**, so this was a safe
breaking rename; if you add a fifth study that reads EMIT `.npz` files, use
these new key names, not `fit_a`.

## aperture_grid.py — `initial_conditions_grid`
Vendored from an external `fcc_92/004_tutorial_cap_meeting/gen_grid.py`
(comment says so explicitly) so this directory has no external-repo
dependency. Only supports `study="MA"` (raises otherwise). Builds a polar
grid via `xp.generate_2D_polar_grid`, tiled over `delta_initial_values`, and
returns an `xt.Table` with `id/x_normalized/y_normalized/delta_init` plus
scalar attrs `nn_x_theta/nn_y_r/num_delta/num_particles` stapled on. Used only
by 009 (010 deliberately avoids it, see above).

## aperture_study_io.py — shared save/reload/replot layer
- `ModelTag = "SB"|"VarSol"`, `StudyTag = "DA"|"MA"|"EMIT"`.
- `BUILD_DEFAULTS = {"sexamp": 1.0, "theta": -0.015}` — `variant_suffix(**overrides)`
  appends `__sexamp{value}` to filenames only when `sexamp != 1.0` (uses
  `format_tag_float`, e.g. `2.0 -> "2p0"`, minus sign -> `"m"`), so default
  runs get clean filenames and only non-default sextupole-amplification runs
  get a distinguishing suffix. **Note (2026-07-15): this docstring line is
  stale** — `BUILD_DEFAULTS` also has `x_offset`/`y_offset` (pre-existing,
  tags `xoff`/`yoff`) and, as of 2026-07-15, `extra_sext_strength` (tag
  `xsext`, see below); check the source, not just this bullet.
- **Extra sextupole labeling (added 2026-07-15)**: `extra_sext_strength`
  (the `--extra-sext-strength` CLI knob, see below) is threaded through
  `save_da_study`/`save_ma_study` and `variant_suffix` exactly like `sexamp`/
  `x_offset`/`y_offset` — saved as a float in the `.npz`, and non-default
  (nonzero) values get a `__xsext{value}` filename tag shared by both the
  `.npz` and the PDF plot.
- `make_basename`/`make_study_stem`: filename convention
  `{STUDY}[_{tag}]_{Sol_On|Sol_Off[_Cor_Off]}_{Model}_{n_part}p_{n_turns}t_{xylimtag}{variant}`
  (n_part omitted for EMIT calls that don't pass it? — check: `save_emitt_study`
  does pass `n_part`, so EMIT filenames do include it, e.g.
  `EMIT_Sol_On_SB_1000p_10000t_xylim1m.npz`; the `data/` dir currently has
  some older EMIT files without the `p` segment, e.g.
  `EMIT_Sol_On_SB_10000t_xylim1m.npz` — **these are stale/pre-rename
  artifacts from before `n_part` was added to the EMIT stem; don't assume
  the DATA_DIR file list reflects the current naming scheme exactly**).
  `global_xy_limit_tag`: `>=1m -> "xylim{N}m"`, `>=1cm -> "xylimNcm"`, else
  `"xylimNmm"`. **`tag` param added 2026-07-13** (`make_study_stem(study, *,
  tag="", **basename_kwargs)`): when non-empty, inserted right after the
  study prefix, e.g. `MA_X_...`/`MA_Y_...` — used only by `save_ma_study` so
  far (`MA_DIRECTION_TAGS = {"x_only": "X", "y_only": "Y"}`). Because
  `replot_from_npz` dispatches on `npz_path.name.split("_", 1)[0]` (first
  underscore only), `MA_X_...`/`MA_Y_...` still resolve to `study="MA"`, so
  the tag doesn't break dispatch.
- `PLOT_DIR = Path.home() / "cernbox/Pictures/FCC_Solenoid_Studies"` (changed
  2026-07-15 from a hardcoded `/home/simonfan/...` string, which broke for
  any other user/machine even though the underlying synced cernbox folder is
  the same) — **outside the git repo**; PDFs are not version-controlled.
  `DATA_DIR = HERE/"data"` (raw `.npz`, inside the repo, currently
  gitignored-or-just-untracked — check `git status` if unsure).
- `save_da_study`/`save_ma_study`/`save_emitt_study`: each builds the stem,
  writes a dict of arrays via `np.savez`, then calls `save_figure_pdf`.
  `save_da_study` now also stores `y_normalized` and `max_amp_sigma_y`
  (2026-07-13, alongside the pre-existing `x_hat`/`y_hat`/`max_amp_sigma_x`)
  — see 010's section above for why `max_amp_sigma_y` exists as its own
  independent constant rather than being derived from `sigma_x/sigma_y`.
  `save_ma_study` now requires a `direction: Literal["x_only", "y_only"]`
  kwarg, stored as a scalar `direction` string in the `.npz` and used to pick
  the filename tag (2026-07-13 MA direction split).
  **Bug fixed 2026-07-28**: `save_da_study`/`save_ma_study` had no
  `field_tag` parameter at all (unlike `save_emitt_study`, which already
  passed it through to `make_study_stem`) — every DA/MA output filename was
  silently tagged with the module-level `solenoid_params.FIELD_TAG` default
  regardless of which lattice was actually loaded. Harmless while 009/010
  had no `--b0` of their own (module default == what was loaded), but would
  have mislabeled filenames the moment they gained a runtime `--b0` (see
  above). Both now take `field_tag: str = FIELD_TAG` and pass it to
  `make_study_stem`; 009/010 call them with `field_tag=tag` (the
  `--b0`-resolved tag for that run). Verified with short (5-turn) smoke
  runs: `--b0 3.0` produced `DA_Sol_Off_SB_3T_...`/`MA_X_Sol_Off_SB_3T_...`.
- `_plot_da_from_arrays`: reads `y_normalized` (not raw `y_hat`) for the
  y-axis, no `set_aspect("equal", ...)` (matches the live 010 plot post-fix).
- `_plot_ma_from_arrays`: reads `direction` from the `.npz` and plots
  `x_normalized` (x_only) or `y_normalized` (y_only) accordingly, title
  suffixed `(x-only)`/`(y-only)`. **Backward-compatible**: older MA `.npz`
  files saved before this change have no `direction` key —
  `"direction" in data.files` falls back to `"x_only"` so pre-existing files
  still replot correctly (verified against a real pre-split file in
  `data/`).
- `replot_from_npz`: dispatches on the `{STUDY}_` filename prefix to
  `_plot_da_from_arrays`/`_plot_ma_from_arrays` (rebuild the same figure style
  as the live scripts) — **no EMIT branch**, see 012's note above.
