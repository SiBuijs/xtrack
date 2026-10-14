# `007a_sls_undulator_tune_displacement.py` -- tune shift vs. undulator horizontal offset, SB vs MK (ring-inserted)

Summary of `examples/splineboris/007a_sls_undulator_tune_displacement.py` as
of 2026-08-25 (`f4e25b0cc`), for orientation before further work on this
file. This is a note about the code, not a design doc -- re-read the source
for anything load-bearing. Companion notes:
[tube_fitter_to_line.md](tube_fitter_to_line.md),
[tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md),
[sls_undulator_spin_emittance_evolution.md](sls_undulator_spin_emittance_evolution.md)
(012, the sibling spin/emittance study built on the same `to_line()`/
`to_multipole_line()` pieces),
[sls_undulator_tune_displacement_local_optics.md](sls_undulator_tune_displacement_local_optics.md)
(007b, a sibling script doing the same offset scan but reading the tune
shift off the standalone undulator's own phase advance instead of
inserting it into the ring -- see that note for how and why it differs).

**Renamed from `007_sls_undulator_tune_displacement.py`** on 2026-08-25
(`b531c668b`, "Checking second-order betas") -- same commit also added the
`betx2`/`bety1` (Edwards-Teng coupled beta) tracking and the
`beta_beat_coupled` figure, both of which this note already documented
below (written against the pre-rename filename, content unaffected by the
rename itself). All in-file paths/filenames in this note have been updated
to the new name; nothing else about the pipeline changed in that commit.

## Current state (2026-10-14) -- read this first

Everything below this section is the history of the file as of
2026-08-25..10-05. It describes `TubeFitter.to_line()`/`to_multipole_line()`,
`get_tube_fitter()` and figure sets that no longer exist. This section
describes `HEAD` (after `21d1bf1e5`, `a8e5ecc17`, `d7343fade`, `a755e8667`,
2026-10-06..14).

- **Fit (two-stage, see `00_overview.md`).** `get_undulator_fit()` (was
  `get_tube_fitter()`) loads `simona_field_map.txt` and runs
  `TubeFitter(n_frames=4441, deg=multipole_order-1=2)`, then
  `LongitudinalFitter(z[0], z[-1], end_condition="free", period=0.036)` on
  `tf.on_axis_multipoles()`, plus a second `fit()` call for
  `tf.on_axis_bs()` -> `("Bs", 0)`. The `LongitudinalFitter` is cached at
  module scope (`_undulator_fit`). `FIELD_TOL = dict(field_tol=1e-4,
  r_ref=2e-3)` is now passed at export time, to `to_line`/`to_multipole_line`.
- **Models.** SB = `LongitudinalFitter.to_line(multipole_order=3)`, elements
  `splineboris_NNN`. MK = `LongitudinalFitter.to_multipole_line(p0c=E0,
  field_at='mean')`, thick `Multipole`s `multipole_NNN`, no `Bs`. Field
  elements are found by these name prefixes. `to_multipole_line` emits **no**
  `MultipoleEdge`s, so the `isinstance(..., xt.MultipoleEdge)` filter on
  `mult_names` and the comment above it, which says edges are emitted, are
  dead.
- **`CASES`** at the top selects placements; only `'ars11_uind_0610_1'` is
  active (0210 and `both` are commented out). Both models always run.
- **Offsets.** If the measurement CSV exists, `hor_off_list` is the
  measured bump amplitudes, so simulation and measurement line up point by
  point. Otherwise it is `linspace(-0.5, 0.5 mm, 30)`. Correctors are still
  re-matched per offset.
- **New: `field_focusing()`** (SB only, `field_K_included`). Linear focusing
  K_x, K_y on the wiggling closed orbit, taken directly from
  `SplineBoris.get_field()`, including the B_s terms (`-y_w' dBs/dx`,
  `+x_w' dBs/dy`) and the 3D part of dBy/dx. For SB, these element-mean
  K_x/K_y **replace** the multipole deltaK1 in the analytic cross-check below
  (the plot labels switch from deltaK1 to K_x/K_y). The B_s-only part goes to
  `deltaq{x,y}_bs_list`, which is cached and printed at the offset nearest 0
  but not plotted. The derivation is cited as `claude_notes/undulator_linear_focusing_derivation.md`,
  which is **not in this checkout** (see `00_overview.md`). The term-by-term
  split is done in the new sibling `007c_undulator_focusing_decomposition.py`.
- **Analytic cross-check** as before for MK (K1 + K2 x_local - K2_skew y from a
  separate `to_multipole_line()` build; for SB see the previous bullet; unperturbed vs perturbed beta;
  `x_deflection_baseline` from a single-particle track). The cached dict now
  also carries `betx2/bety1` scans, `undulator_s_ranges`, `field_K_included`
  and the `_bs` lists.
- **Figures: 3 per placement, MK and SB side by side** (the old per-case
  `beta_beat`/`beta_beat_coupled`/`tune_shift` set and
  `x11ma_open_gap_drift.pdf` are gone):
  - `{place}_tune_shift_MK_vs_SB.pdf`: Twiss tune shift and its quadratic
    fit, the two cross-check curves (bare beta and perturbed beta), the
    X11MA diamonds with their own quadratic fit, and fit-coefficient boxes.
  - `{place}_tune_shift_vs_closed_gap_zeroed.pdf` (only if the CSV
    exists): simulated vs measured closed-gap tune (= `dtune_*_measured` +
    open-gap), each curve shifted to zero at x = 0, plus the open-gap
    drift for reference.
  - `{place}_beta_beat_MK_vs_SB.pdf`: Delta beta / beta around the ring, one
    curve per offset. The bare-ring beta is re-twissed at the exact s
    (`bare_ring_beta`, cached), because interpolating the cached rows is
    off by up to ~8% inside the undulator span.
  - `{place}_tune_shift_beta_beat.pdf` existed briefly and was removed in
    `a755e8667` (2026-10-14).
- `OUT_DIR` uses `$SLS_UNDULATOR_PLOT_DIR` (default
  `~/cernbox/Pictures/SLS_Undulator_Studies`); `--replot` and the per-case
  `data/{place}_{model}.npz` cache are unchanged, still with no staleness
  detection. **Caches written before 2026-10-08 came from the old fitter.
  Delete them.**

## Purpose

For the SLS ring, measures how the horizontal tune shift `ΔQx/ΔQy` depends on
a rigid horizontal misalignment (`shift_x`) of the undulator field elements,
comparing two undulator field models against each other and against an
analytic first-order-perturbation-theory cross-check.

## The 3×2 case grid

`WIGGLER_CASES` (placement) × `MODEL_LABELS` (field model), all 6 combinations
run via `compute_case(place_label, wiggler_places, model_label)` /
`plot_case(data, place_label, model_label)` (split 2026-08-19, see "Caching
and --replot" below):

- **Placement**: undulator at `ars11_uind_0210_1` only, `ars11_uind_0610_1`
  only, or both simultaneously (`'both'`).
- **Model**: `SB` = `TubeFitter.to_line()` (one `SplineBoris` element per
  polynomial piece, full spatial field integration, includes `Bs`); `MK` =
  `TubeFitter.to_multipole_line()` (coarse thick-`Multipole` per region, mean
  field only, no `Bs` -- see
  [tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md)).

The field map itself (`get_tube_fitter()`, `TubeFitter(...)`,
`n_frames=4441, deg=multipole_order-1=2, field_tol=1e-4`) is fit lazily, on
first use, and cached at module scope (a module-level `_tube_fitter` behind
`get_tube_fitter()`) for reuse across all 6 cases -- fitting is independent
of placement/model choice. "Lazily" matters: loading+fitting the raw field
map (`simona_field_map.txt`, ~1.8 GB) is the single slowest thing this
script does, so a `--replot` run where every case's data is already cached
(see "Caching and --replot" below) never touches the field map at all.

## Caching and --replot

`compute_case()` does the expensive part (build/match/scan/analytic
cross-check/coupling correction) and returns a plain dict of arrays/scalars
-- no matplotlib. `get_case_data()` wraps it with an `np.savez`/`np.load`
cache in `DATA_DIR = examples/splineboris/data/{place_label}_{model_label}.npz`:
normally it always calls `compute_case()` and overwrites the cache; pass
`--replot` on the command line and it will load from the cache instead,
*for any case whose `.npz` already exists* (falls back to computing if not).
`plot_case(data, place_label, model_label)` then builds and saves all 8
figures purely from that dict, so `--replot` (once every case has been
computed at least once) skips the entire fit/match/scan pipeline -- just
loads ~30 arrays per case and replots, useful for iterating on plot styling
without waiting for a full rerun. Unlike 012's `--replot`
([sls_undulator_spin_emittance_evolution.md](sls_undulator_spin_emittance_evolution.md)),
this is per-case (mixed cache-hit/cache-miss runs work fine) and the cache
key is only `{place_label}_{model_label}` -- if you change scan parameters
(`n_tunes`, `hor_off_list` bounds, `multipole_order`, ...) the stale cache
will NOT be detected as stale; delete the relevant `.npz` (or the whole
`data/` dir) by hand first.

## Per-case pipeline (`compute_case`)

1. Fresh `xt.load(madx_file)` + `configure_bend_model('mat-kick-mat')` per
   case (not shared across cases -- each case gets its own `line_sls`).
2. Build the undulator line (SB or MK) in its own `und_env` (both builders
   create an implicit `xt.Environment`), add 4 correctors (`corr1..4` at
   `0.02, 0.1, l_wig-0.1, l_wig-0.02` m from the start), match them once at
   zero offset (`only_orbit=True`, targets `x=px=y=py=0` at both ends).
3. Import the matched undulator into `line_sls` at each placement position
   (`env.import_line` first, then `insert()` -- direct `insert()` of the
   undulator's own-env line fails, same gotcha documented in 012's note).
4. Baseline Twiss (`tw_no_undulator`, no undulator inserted) for `qx_0`/`qy_0`
   and as the orbit-comparison reference.
5. On/off-axis orbit comparison plot (`shift_x=0` vs `0.5e-3`).
6. **The tune scan** (`n_tunes=30`, `hor_off_list` from -0.5 to +0.5 mm):
   for each `dx`, sets `shift_x=dx` on both the standalone `undulator` line
   and its copy(ies) in `line_sls`, **re-matches the 4 correctors** (see
   "Why re-match per offset" below), pushes the rematched values into the
   ring copy, then Twiss's and records `ΔQx`, `ΔQy`, orbit, `betx`/`bety`.
7. Independent analytic cross-check (see next section), plus a closest-tune-
   approach (betatron coupling) correction of that cross-check (see "Coupling
   correction" below).
8. 8 figures per case: the field along the tracked trajectory (see "Field
   along the trajectory" below), orbit comparison, orbit scan (colour-coded
   by offset), beta functions across the scan, beta beat vs. no-undulator
   baseline, the tune-shift-vs-offset plot (Twiss vs. quadratic fit vs. both
   formula variants), the tracked-minus-formula residual (both variants), and
   that same residual before/after the coupling correction. Saved as
   `{place_label}_{model_label}_{suffix}.pdf` to `OUT_DIR` -- **see
   "Hardcoded output path" below, likely broken on this machine.**

All 48 figures (8 × 6 cases) stay open (not closed inside `plot_case`) so they
can be reviewed interactively via the final `plt.show()`, in addition to being
saved as PDFs.

**Stale as of 2026-08-26 -- see "Update (2026-08-26)" below.** Down to 3
figures/case (18 total), the coupling correction is gone entirely, and a
digitized measurement overlay was added to the tune-shift figure.

## Why re-match correctors at every offset in the scan

The 4 correctors are only matched once, at `shift_x=0`. As the offset is
scanned, the field seen by the field elements changes, so those fixed
strengths stop restoring `x=px=y=py=0` at the undulator exit -- left
uncorrected, the resulting orbit distortion would leak into the rest of the
ring and contaminate the tune shift with an orbit-distortion effect on top of
the field-nonlinearity effect the scan is trying to isolate. So `opt_dx =
undulator.match(...)` is re-run (same targets, on the standalone `undulator`
line) inside the scan loop for every `dx`, and the rematched values are then
copied into the ring's corrector variables (`env[kk] = und_env[kk]`).

Contrast with [sls_undulator_spin_emittance_evolution.md](sls_undulator_spin_emittance_evolution.md)'s
(012) `--offset-x` misalignment study, which does the equivalent rematch but
applies the offset to the correctors too (correctors move with the undulator
girder) -- 007 only ever offsets the field elements, never the correctors
themselves (consistent with 007 predating that later shift-x-on-correctors
fix in 012; not revisited here).

## The analytic tune-shift cross-check

Standard first-order perturbation theory,
`ΔQ = 1/(4π) ∮ ΔK1(s)·β(s) ds`, evaluated two ways per offset (unperturbed
`tw_no_undulator` beta vs. the actual perturbed `betx_scan`/`bety_scan` beta
at that offset -- the gap between the two isolates how much of the
formula-vs-Twiss discrepancy is beta-beat feedback rather than `ΔK(s)` itself).

`ΔK1(s)` has two independent sources inside the undulator region, both
evaluated at the *actual local orbit position* `x_local(s) = x(s) - dx` (field
elements are offset, not the beam -- `shift_x` moves the magnet, so the field
must be evaluated at the particle's position relative to the shifted magnet):

1. **Quadrupole feed-down**: `mult_K1 + mult_K2 * x_local`, where `mult_K1`/
   `mult_K2` come from a dedicated `to_multipole_line()` build (built fresh
   inside `compute_case` regardless of which model -- SB or MK -- is primary for
   that case, so the cross-check is on equal footing across all 6 cases).
2. **Solenoid focusing**: a longitudinal field focuses *both* planes,
   `K_sol(s) = (Bs(s) / (2·Brho0))²`. On-axis `Bs` is ~0 here (`to_fit=False`
   per the fit report), but Maxwell's equations tie the z-varying transverse
   field to a nonzero *off-axis* `Bs` even when on-axis `Bs` is exactly zero
   (confirmed empirically: `Bz(x=0.5mm) ≈ 0.012 T` vs `Bz(x=0) = 0` exactly).
   So `Bs` must be read at the actual local orbit position, via a dedicated
   `bs_line = tube_fitter.to_line(...)` (always SplineBoris-based, since
   `to_multipole_line()` drops `Bs` entirely and the MK `undulator` therefore
   has no `Bs` to read even when MK is the primary model for that case).

Sign convention (`ΔQx = +.../(4π)`, `ΔQy = -.../(4π)` for the quadrupole term,
but **same sign in both planes** for the solenoid term, since a solenoid
focuses x and y alike) was empirically verified by inserting a small
known-`K1L` thin `Multipole` into `line_sls` and checking which sign choice
matched the resulting Twiss `qx`/`qy` shift -- not derived from a textbook
convention, so worth re-verifying if this formula is reused elsewhere with a
different sign/coordinate convention.

## Coupling correction of the analytic cross-check

The perturbation-theory formula above gives *uncoupled* tunes
(`Qx_formula = qx_0 + deltaQx_formula`, etc). The betatron-coupling closest-
tune-approach resonance pulls the real (Twiss/tracked) eigentunes apart from
those uncoupled values, so a residual between the formula and Twiss is
expected even if `ΔK1(s)`/`K_sol(s)` are exactly right -- this is corrected
for via

```
Qbar = (Qx + Qy) / 2,  Delta = Qx - Qy
Qx_obs = Qbar + sign(Delta)/2 * sqrt(Delta**2 + |C-|**2)
Qy_obs = Qbar - sign(Delta)/2 * sqrt(Delta**2 + |C-|**2)
```

applied to the **absolute** uncoupled formula tunes at each scan point (and
to the `tw_no_undulator` reference, using its own `c_minus`) before
differencing -- never added directly to an already-differenced `ΔQ`. `|C-|`
is read straight off `tw.c_minus`, which `xtrack`'s `twiss4d`/`twiss` computes
automatically whenever `mux`/`muy` are available, so this needed no new
tracking or an independent analytic `k_s(s)`/Eq. 3.83 integral -- the twiss
route alone was judged sufficient (an analytic cross-check via skew-sextupole
feed-down + the coupling part of the undulator's longitudinal field was
scoped but not implemented, should it be wanted later). Implemented as
`_coupling_corrected_tunes()` inside `compute_case`, feeding the new
`fig_tune_shift_corr` figure (residual before/after correction) and a
`[coupling check]`-prefixed console printout of `c_minus`, `Delta`, and the
residual at the largest-offset scan point, per the request that motivated
this feature.

## Field along the trajectory

The closed-orbit trajectory from `tw_onaxis`/`tw_offaxis` (`.x`/`.y`/`.s`,
already computed for the orbit-comparison plot right after) is sampled at
`n_sub=6` points per field element (`_sample_field`) and the field evaluated
there with whichever model is primary for this case:

- **SB**: `SplineBoris.get_field(x_local, y_local, s_local)` on the actual
  elements in `line_sls` -- the real fitted field, `Bx`/`By`/`Bs` all
  included.
- **MK**: no field evaluator exists on a thick `Multipole` (no spatial
  structure within the element), so the field is reconstructed from its own
  `knl`/`ksl` and the reference rigidity via `_multipole_field_bx_by()` (a
  module-level helper), matching the exact convention xtrack's own multipole
  kick uses internally (`track_magnet_kick.h::evaluate_field_from_strengths`
  / `kick_simple_single_coordinates`): `knl`/`ksl` are length-integrated, and
  `By + i*Bx = brho0/length * sum_n (knl[n]+i*ksl[n])/n! * (x+iy)**n`. `Bs`
  is always 0 for MK (no solenoid equivalent).

Cost-wise this is cheap relative to the rest of the file: the existing
analytic-cross-check scan loop already calls `bs_line[nn].get_field(...)`
once per region *per scan point* (30x), so this feature's 2 offsets x 6
samples/region is a small addition on top of that.

## `MultipoleEdge` thick-filtering fix (2026-08-25)

`to_multipole_line()` was changed the same day (`f4e25b0cc`, "Checking if
adding edges makes the differnece") to bracket each region's thick
`Multipole` with a pair of thin `xt.MultipoleEdge` kicks -- see
[tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md)'s
"Update" section for the physical rationale. Those edge elements match the
`'tubefitter'` name-prefix filter used throughout this file (so they still
get `shift_x`'d together with their parent region, correctly) but have no
`.length`/`.knl`/`.ksl`/`.get_field()`, so three places that iterate
per-field-element and read those attributes had to be narrowed to a
thick-only subset (`isinstance(..., xt.MultipoleEdge)` exclusion):
`field_element_names_thick`/`undulator_field_element_names_thick` (used by
`_sample_field()`, the field-along-trajectory plot) and `mult_names` (used
by the `mult_K1`/`mult_K2`/`mult_K2_skew`/`mult_s_mid` analytic-cross-check
arrays, previously indexed by `multipole_line.element_names` directly and
`mult_table['s'][:-1]` positionally -- now keyed by name via `mult_table['s',
nn]`, robust to the edge elements' presence in that table). The Twiss
itself (`tw.qx`/`tw.qy`) already picks up the edge kicks automatically
(they're just more elements in the line); only the two hand-rolled,
per-region loops needed the filter.

**Superseded 2026-08-26:** the edges were reverted out of `to_multipole_line()`
entirely (see [tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md)'s
"REVERTED" update -- they caused an `SVD did not converge` crash in this
script's own per-offset corrector re-match). `field_element_names_thick`/
`undulator_field_element_names_thick` were separately removed the same day
for an unrelated reason (the field-along-trajectory plot they fed was cut,
see "Update (2026-08-26)" below) -- but the `mult_names` filter above is
still present and now a harmless no-op (nothing will ever match
`isinstance(..., xt.MultipoleEdge)` again). Left in rather than cleaned up.

## Hardcoded output path -- likely broken on this machine

`OUT_DIR = Path('/home/simonfan/cernbox/Pictures/SLS_Undulator_Studies')`
([007a:67](../007a_sls_undulator_tune_displacement.py#L67)) is a **different
user's home directory on a Linux machine** (`simonfan`), not this user's
(`sietse`, macOS, `$HOME=/Users/sietse`, no `/home` tree at all). `OUT_DIR.mkdir
(parents=True, exist_ok=True)` will attempt to create `/home/simonfan/...` on
this machine and almost certainly fail (no `/home` directory / permission
denied on macOS), so **this script cannot currently save its figures on this
machine as-is**. Contrast with 012
([sls_undulator_spin_emittance_evolution.md](sls_undulator_spin_emittance_evolution.md)),
which solved exactly this cross-machine problem via a `PLOT_DIR` default of
`~/cernbox/Pictures/SLS_Undulator_Studies` overridable by the
`SLS_UNDULATOR_PLOT_DIR` env var -- 007 was evidently the origin of that
`.../cernbox/Pictures/SLS_Undulator_Studies` convention (same target
directory name) but never got the portable `PLOT_DIR`/env-var treatment 012
later added. If picking this file up again, porting 012's `PLOT_DIR` pattern
here first is probably the quickest unblock.

## History

Git history (`git log --follow -- 007a_...py`, the file was
`007_sls_undulator_tune_displacement.py` before the 2026-08-25 rename):
`5cfbcc075` (2026-05-18, "Method quite
heavy for new field map") is an old, unrelated earlier version pre-dating the
`TubeFitter` rewrite era covered by the other notes in this directory. The
file was substantially reworked on 2026-08-17/18 (`43f07e590` "Working on the
tune scan", `027d60df8` "Comparing tune shift in SB and MK undulators",
`07cc73595` "Some changes to the plots.") from an earlier single-placement,
single-model (`SB` only, `multipole_order=5`) script into the current
3-placement × 2-model `run_case()` structure with the analytic cross-check
formula added. On 2026-08-19 `run_case()` was split into
`compute_case()`/`plot_case()` with an `.npz` cache and `--replot` flag (see
"Caching and --replot" above), and a lazily-fit `get_tube_fitter()` --
motivated by full reruns being dominated by loading+fitting the ~1.8 GB field
map even when only iterating on plot styling.

## Update (2026-08-26): figures trimmed to 3, coupling correction removed, X11MA measurement overlay added

Three separate, user-requested changes landed the same session -- noted
together since they touch the same functions. Re-read `compute_case()`/
`plot_case()` directly for the current code; sections above (`8 figures per
case`, `Coupling correction of the analytic cross-check`, `Field along the
trajectory`) describe the **pre-2026-08-26** structure and are only kept for
history.

**1. Coupling correction removed entirely.** The user reported the
closest-tune-approach (`|C-|`) correction (`_coupling_corrected_tunes()`,
`c_minus` tracking, `deltaqx_formula_corr_list`/`deltaqy_formula_corr_list`
and their perturbed-beta counterparts, the `deltaqx_resid_corr_*`/
`deltaqy_resid_corr_*` residuals, the `[coupling check]` console prints, and
the entire `fig_tune_shift_corr` figure) as insignificant and asked for it
to be dropped, not just hidden. All of it is gone from `compute_case()` and
`plot_case()` -- there is no `tw.c_minus` tracking left in this script at
all. The `$C^-$-corrected` traces are also gone from the main
`fig_tune_shift` figure itself (it used to overlay 4 more marker series on
top of the Twiss/formula ones).

**2. Four more figures cut on request** (the user separately said "I no
longer want to calculate/plot the beta functions, the magnetic field along
the trajectory, the tune shift difference and the tune shift coupling
corrected" -- the last one already covered by point 1): `fig_beta_scan`
("Beta functions across the tune scan", `beta_functions.pdf`),
`fig_field_traj` ("Field along the tracked trajectory",
`field_along_trajectory.pdf` -- along with the now-dead
`_multipole_field_bx_by()` helper, `brho0`, `_sample_field()`, and the
`field_element_names_thick`/`undulator_field_element_names_thick`
thick-filtering this fed), and `fig_tune_shift_diff`
("Tracked minus calculated tune shift", `tune_shift_difference.pdf`) are all
gone. Then, in the very next turn, the user also asked to drop
`fig_orbit`/`fig_orbit_scan` (`orbit_comparison.pdf`/`orbit_scan.pdf`,
previously SB-only) -- along with the `tw_offaxis` Twiss call that existed
only to feed that plot (`tw_onaxis` is kept -- still needed for the
x-deflection-baseline Twiss-interpolation check) and the `orbit_scan_x`/
`orbit_scan_y` dict exports (the underlying lists are still collected
inside `compute_case()`, since the analytic tune-shift formula's
quadrupole/sextupole feed-down term still reads them locally -- only their
export to `plot_case()` was cut).

**Net effect:** every case now saves exactly **3 figures**
(`beta_beat.pdf`, `beta_beat_coupled.pdf`, `tune_shift.pdf`), unconditional
across both `SB` and `MK` -- the old model-dependent figure count (previously
8 for SB, 6 for MK; the `field_element_names_thick`/`MultipoleEdge`-filtering
episode above briefly touched the same code) no longer applies at all. Total
across all cases: `3 figures × 2 models × 3 placements = 18` (was 48).

**3. X11MA digitized measurement overlay.** The user supplied a screenshot of
a real accelerator measurement plot (`ΔTune x`/`ΔTune y` vs. bump amplitude,
titled "X11MA, gap = 11.5 mm") and asked for it pixel-digitized and overlaid
on this script's `tune_shift` figure. Digitized via blue-diamond-marker
blob detection (color-channel thresholding + `scipy.ndimage.label`) with
pixel-to-data calibration from the axis tick-mark pixel positions (found via
dark-pixel row/column scans of the axis spines), excluding the two legend
diamond icons (identified as the two blobs whose value was inconsistent with
the local monotonic trend of their neighbours) -- 37 points recovered per
panel, verified by overlaying the digitized points back on the source image
(every point landed exactly on a diamond). Saved to
`test_data/sls/x11ma_gap11p5mm_tune_shift_digitized.csv` (columns
`bump_amplitude_um`, `bump_amplitude_m`, `dtune_x_measured`,
`dtune_y_measured`); source screenshot at `test_data/sls/image004.png`
(gitignored, matches the existing blanket `*.png` rule).

Loaded via `load_measured_tune_shift()` (returns `None` gracefully if the
CSV is absent) and plotted in `fig_tune_shift`'s `ax1`/`ax2` as purple
diamonds, `label='X11MA, gap=11.5mm (measured)'`. Applies unconditionally to
every case (all 3 placements × both models) -- **the correspondence between
this "X11MA" beamline measurement and the simulated `ars11_uind_0210_1`/
`ars11_uind_0610_1` elements has never been confirmed**, so it's overlaid on
all of them deliberately, to let the two individual-placement figures answer
"does either one look like a match" by eye. Not yet actually resolved either
way.

Plotting mechanics changed once after first landing: initially plotted on a
separate `ax1.twinx()`/`ax2.twinx()` (independent y-scale, distinct purple
axis labels/ticks) specifically because the absolute baseline of "measured
ΔTune" vs. simulated ΔQ wasn't known to match. After seeing the first real
plot (`ars11_uind_0610_1`, `SB`), the two scales turned out to sit in the
same ballpark on their own, so the user asked to drop the twin axis and plot
both series directly on the shared `Delta Qx`/`Delta Qy` scale instead --
simpler, and lets the shapes be compared directly rather than via two
independent axes. Implemented by removing the `twinx()` calls and just
calling `ax1.plot()`/`ax2.plot()` directly, then re-invoking `ax1.legend()`/
`ax2.legend()` so the new series joins the existing legend (a fresh
`legend()` call picks up every current artist on that axes, not just new
ones).

**4. Unrelated same-session crash, traced to a different file.** While
first trying to actually run this script (on a different, Linux machine --
`/home/simonfan/...`, not this note's usual `sietse`/macOS machine), the
per-offset corrector re-match (`opt_dx.step(2)`, see "Why re-match
correctors at every offset" above) crashed with `numpy.linalg.LinAlgError:
SVD did not converge` / `Optimize - start penalty: nan`. Traced to the
`MultipoleEdge`-bracketing experiment in `to_multipole_line()` (the
`f4e25b0cc` change described in the "`MultipoleEdge` thick-filtering fix"
section above) -- reverted in `tube_fitter.py`, not in this script; see
[tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md)'s
"REVERTED" update for the full story. Nothing in this file needed changing
for that fix, only for `to_multipole_line()`'s own module.

## Not done / open

- Output path needs fixing for this machine (see above) before the script can
  actually save PDFs here.
- The `--replot` cache has no staleness detection (see "Caching and
  --replot" above) -- changing scan parameters silently serves stale cached
  data unless you delete `data/` by hand first.
- `multipole_order` is set to `3` at module scope but `tube_fitter` is built
  with `deg=multipole_order-1` (i.e. `deg=2`) -- consistent with the `deg<=2`
  practical ceiling noted elsewhere in these notes, but worth knowing this is
  one shared constant, not independently tuned per fitter call.
- As of this note's update (2026-08-25), `DATA_DIR` (`examples/splineboris/
  data/`, gitignored) does not exist on disk on this machine -- 007a hasn't
  been (re)run since the rename/edge-filtering changes above, so there is no
  cached scan data yet; the first run after picking this back up will be a
  full (slow) compute for every case, not a `--replot`.
- (2026-08-26) X11MA-vs-`ars11_uind_*` device correspondence still
  unconfirmed -- see point 3 of "Update (2026-08-26)" above.
- (2026-08-26) The `mult_names` `isinstance(..., xt.MultipoleEdge)` filter is
  dead code now that `to_multipole_line()` no longer emits `MultipoleEdge`
  elements (see the "`MultipoleEdge` thick-filtering fix" section's
  "Superseded" note) -- harmless, left in rather than cleaned up.
- (2026-08-26) The `SVD did not converge` crash (point 4 of "Update
  (2026-08-26)" above) was never root-caused beyond "the edges experiment
  did it" -- no isolated repro pinning it to a specific `dx`, corrector, or
  whether the `model='drift-kick-drift-exact'` switch (vs. the edge kicks
  themselves) was responsible. Worth revisiting if edges are attempted again.

## Update (2026-10-05): open/closed-gap raw tunes, `x11ma_open_gap_drift` figure

The user supplied a second screenshot of the same X11MA measurement showing
the raw tunes ("Open gap" blue / "Gap scan" orange) and lifetime vs bump
amplitude. Both tune series were digitized (per-column colour-weighted
centroid at the known 25 um grid, calibrated on tick marks) and appended to
`test_data/sls/x11ma_gap11p5mm_tune_shift_digitized.csv` as
`tune_{x,y}_{open,closed}_gap` (absolute, offsets Qx = 39.37, Qy = 15.22 --
the Qy offset label was illegible in the screenshot; 15.22 confirmed by the
user as the SLS vertical tune). Row-aligned with the existing rows.

**Finding:** `dtune_*_measured` (the purple diamonds) is already
closed-minus-open *point by point*: rms(meas - (closed - open)) = 4.5e-6 in
y vs 1.7e-5 for closed - mean(open); x agrees too but is pixel-limited
(~8e-5/px). So do NOT subtract the open gap again. Point-by-point is also
the right choice: the open-gap tunes drift systematically with the bump
(Qx by ~-2e-4, Qy by ~+5e-5 over -400..500 um, presumably bump orbit
through ring sextupoles), comparable to the undulator signal itself.

New `plot_measured_open_gap_drift()` (module level, called once before the
case loop, measurement only) saves `x11ma_open_gap_drift.pdf`: open-gap
Q - <Q> next to ΔQ - <ΔQ> per plane. The open-gap curve's staircase look is
pixel quantization from the digitization, not physics.

**Output path fixed (2026-10-05):** `OUT_DIR` now uses 012's pattern --
`$SLS_UNDULATOR_PLOT_DIR`, default `~/cernbox/Pictures/SLS_Undulator_Studies`
-- so the "Hardcoded output path" section above no longer applies.
