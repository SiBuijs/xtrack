# 009-014: aperture and emittance-evolution studies

Written 2026-07-10, updated same day after the 014 fitting/plotting changes
(free-fit eq. emittance + damping rate, Twiss-vs-fit comparison box on plots).
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
`--cases/--list-cases/--n-turns/--sexamp/--no-show` CLI, and a `main()` guarded
by `__name__ == "__main__"`. `013_run_da_and_ma.py` drives 009+010+014 as
subprocesses with a shared CLI.

## 009_momentum_acceptance.py (MA study)
- Grid: `initial_conditions_grid` (from `aperture_grid.py`) — polar in
  `(x_normalized, y_normalized)` at `theta in [pi/4, pi/4]` (i.e. a **line**,
  not a fan, `nn_x_theta` defaults to 1) with `nn_y_r=25, max_y_r=25` sigma,
  replicated over `DELTA_INITIAL_VALUES = linspace(-35*ENERGY_SPREAD,
  +35*ENERGY_SPREAD, 51)` (`ENERGY_SPREAD=3.9e-4`). So it's really an
  amplitude x momentum-offset 2D scan, 25*51 = 1275 particles by default.
- `NEMITT_X=6.33e-5`, `NEMITT_Y=1.69e-7`, `GLOBAL_XY_LIMIT=5e-2` (5 cm —
  intentionally tight since MA only cares about longitudinal loss, not large
  transverse excursions; see 010's comment on why it uses a looser limit).
  `N_TURNS=10_000` default.
- `line.build_particles(method="6d", delta=tt_init.delta_init,
  x_norm=tt_init.x_normalized, y_norm=tt_init.y_normalized)`.
- Plot: `delta_init` vs `x_normalized`, survived (dots) vs lost (colored by
  `at_turn`) with a colorbar; title reports `frac_lost`/`at_turn_mean`.
- Saved via `save_ma_study` (see `aperture_study_io.py` section below).

## 010_dynamic_aperture.py (DA study)
- **Different grid from 009**: `_build_da_initial_conditions` builds its own
  polar grid directly with `xp.generate_2D_polar_grid` (bypassing
  `aperture_grid.py`'s `initial_conditions_grid`, whose docstring/comment in
  010 says it "has a column-length bug for single delta" — i.e. 010
  intentionally avoids the shared helper). Grid: `r in [0, MAX_AMP_SIGMA_X=40]`
  sigma_x-units, `theta in [0, pi/2]` (quarter-plane fan), per-case
  `nn_y_r`/`nn_x_theta` (sb_on: 25x30, varsol_on: 30x50, sb_off: 25x30 — VarSol
  gets a finer grid, presumably because its DA boundary needed more
  resolution). `y_norm = y_hat * 5.0` hardcodes a fixed 5:1 aspect-ratio
  scaling comment-labeled `sigma_x_over_sigma_y` even though the actual
  computed `sigma_x/sigma_y` ratio (`_compute_beam_sizes`) is only used for
  print/plot annotation, not for this scaling factor — **potential
  inconsistency if the real ratio drifts far from 5 for a given case**, worth
  checking if DA results look distorted.
- `GLOBAL_XY_LIMIT = 1.0` (1 m, xtrack default) **intentionally looser than
  009's 5 cm** — code comment explains: large beta functions blow up amplitude
  units well past 5 cm even for physical sigma-levels ~O(30), so 5 cm would
  clip particles that should survive.
  `NEMITT_X=6.33e-5`, `NEMITT_Y=1.69e-7` (same as 009), `N_TURNS=10_000`.
- On-momentum only (`delta=0, px_norm=py_norm=0`) — no off-momentum DA scan
  yet; comment says "For off-momentum DA, repeat with delta set to selected
  momentum offsets" (not implemented).
- Two-panel plot: raw scatter (`x_hat` vs `y_hat`, colored by `at_turn`) +
  `pcolormesh` reshaped to `(nn_y_r, nn_x_theta)` — the pcolormesh only makes
  sense because the grid is a structured `(r, theta)` mesh, so reshape order
  matters if grid generation ever changes.
- Saved via `save_da_study`.

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
  `--n-turns`, `--n-part` for EMIT only, `--sexamp`, `--no-emitt` to skip EMIT,
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
  get a distinguishing suffix.
- `make_basename`/`make_study_stem`: filename convention
  `{STUDY}_{Sol_On|Sol_Off[_Cor_Off]}_{Model}_{n_part}p_{n_turns}t_{xylimtag}{variant}`
  (n_part omitted for EMIT calls that don't pass it? — check: `save_emitt_study`
  does pass `n_part`, so EMIT filenames do include it, e.g.
  `EMIT_Sol_On_SB_1000p_10000t_xylim1m.npz`; the `data/` dir currently has
  some older EMIT files without the `p` segment, e.g.
  `EMIT_Sol_On_SB_10000t_xylim1m.npz` — **these are stale/pre-rename
  artifacts from before `n_part` was added to the EMIT stem; don't assume
  the DATA_DIR file list reflects the current naming scheme exactly**).
  `global_xy_limit_tag`: `>=1m -> "xylim{N}m"`, `>=1cm -> "xylimNcm"`, else
  `"xylimNmm"`.
- `PLOT_DIR = Path("/home/simonfan/cernbox/Pictures/FCC_Solenoid_Studies")` —
  **outside the git repo**, user-specific absolute path; PDFs are not
  version-controlled. `DATA_DIR = HERE/"data"` (raw `.npz`, inside the repo,
  currently gitignored-or-just-untracked — check `git status` if unsure).
- `save_da_study`/`save_ma_study`/`save_emitt_study`: each builds the stem,
  writes a dict of arrays via `np.savez`, then calls `save_figure_pdf`.
- `replot_from_npz`: dispatches on the `{STUDY}_` filename prefix to
  `_plot_da_from_arrays`/`_plot_ma_from_arrays` (rebuild the same figure style
  as the live scripts) — **no EMIT branch**, see 012's note above.
