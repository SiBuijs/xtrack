# `012_sls_undulator_spin_emittance_evolution.py` -- SLS spin/emittance tracking, 7 cases (2026-08-11)

New script, modeled on `examples/fcc_ee_solenoid/014_emittance_evolution.py` /
`015_spin_polarization.py` (`fcc_da_ma` branch), but for the SLS ring instead
of FCC-ee, and combining both studies (emittance damping + spin
depolarization) into a single tracking run per case rather than two separate
scripts. Companion notes on the underlying pieces:
[tube_fitter_to_line.md](tube_fitter_to_line.md),
[tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md),
[tubefitter_canonical_and_upstream_merge.md](tubefitter_canonical_and_upstream_merge.md).

## Current state (2026-10-14) -- read this first

The study itself (7 cases, tracking, analysis, `--replot`, `--offset-x`) is
unchanged since this note. Only the field fit changed, in `a8e5ecc17`
(2026-10-08), when `FieldFitter`/`SplineBorisSequence` were deleted and the
fit was split in two (see `00_overview.md`):

- `_fit_undulator()` reads `test_data/sls/undulator_field_map.txt`, runs
  `TubeFitter(n_frames=1701, distance_unit=1e-3, deg=MULTIPOLE_ORDER-1)`,
  then `LongitudinalFitter(z[0], z[-1], points_per_element=2,
  end_condition="free", period=0.036)` on `on_axis_multipoles()` plus
  `on_axis_bs()`, and returns the `LongitudinalFitter`. (1701 frames over
  2.2 m is ~28 per 36 mm period, so ~14 elements per period.) The
  `n_frames=400` quoted below is out of date; the script already used 1701
  on 2026-08-11 (`f7675f4ff`).
- `field_tol` moved from the fitter to the exports:
  `FIELD_TOL = dict(field_tol=1e-3, r_ref=1e-3)` is passed to
  `lf.to_line(...)` and `lf.to_multipole_line(...)`.
- Read every `TubeFitter.to_line()`/`to_multipole_line()` below as
  `LongitudinalFitter.to_line()`/`to_multipole_line()`. They still return a
  `Line` in its own implicit environment, so the `env.import_line` gotcha
  (point 2 below) still applies. `SplineBorisSequence` no longer exists.
- `to_multipole_line()` still uses the element **mean** field by default
  (`field_at="mean"`; `"midpoint"` is now an option), so point 3 below still
  holds. The Hermite `param_index=4` it mentions is now column 4 of
  `LongitudinalFitter.element_params()`.
- The fixture `undulator_tubefitter_fit_pars_df` in `tests/test_splineboris.py`
  is gone; the tests use an `undulator_fit` fixture now.

## The 7 cases

1. `no_undulators` -- bare ring baseline.
2. `sb_0210` / `sb_0610` / `sb_both` -- one or two `SplineBoris`-based
   undulators (`TubeFitter.to_line()`) at `ars11_uind_0210_1` /
   `ars11_uind_0610_1`.
3. `mult_0210` / `mult_0610` / `mult_both` -- the same, but built with the
   new `TubeFitter.to_multipole_line()` (coarse thick-`Multipole`
   approximation, see the companion note).

The undulator field map is fit **once** with `TubeFitter` (`n_frames=400,
distance_unit=1e-3, deg=2, field_tol=1e-3` -- the exact recipe already
validated in `tests/test_splineboris.py`'s `undulator_tubefitter_fit_pars_df`
fixture), then both a `to_line()` and a `to_multipole_line()` version are
built and orbit-corrected once each and reused across the relevant cases.

## Two things had to be fixed that don't show up in the existing 004b/009 SLS examples

**1. `test_data/sls/sls.madx` has no RF cavities.** Confirmed by counting
element types in the loaded line -- zero `Cavity` elements. The existing
undulator examples (`004b`, `007`, `009`) never track multiple turns under
radiation, so this never mattered for them; `twiss(radiation_integrals=True)`
(the integral/perturbative estimate) doesn't need a real closed 6D orbit.
But `twiss(radiation_analysis=True)` (needed for `eq_gemitt_zeta`/`bets0`,
required to generate a matched bunch and to actually track under radiation)
requires a converged 6D closed orbit, which doesn't exist without RF --
`ClosedOrbitSearchError`. Worse: without a cavity, `configure_radiation
(model="mean")` tracking has no restoring force for the ~684 keV/turn
synchrotron-radiation energy loss, so a real multi-turn run would just
monotonically lose momentum (at that loss rate, ~4000 turns exhausts most of
a 2.7 GeV beam -- totally unphysical for a 10000-turn study).

Fix: found the precedent already in the repo,
[examples/radiation_thick/000_sls_radiation.py](../../radiation_thick/000_sls_radiation.py),
which adds a synthetic cavity for exactly this reason. Reused its exact
parameters: `vrf=1.8e6` (1.8 MV), `frf=499.6e6` (499.6 MHz), inserted at
`s=0` via `env.new('cav', 'Cavity', voltage=..., frequency=..., phase=...)`
(used `phase=np.pi` instead of the deprecated `lag=180` kwarg that script
uses). With the cavity in place, `compensate_radiation_energy_loss()`
converges normally (tapers magnet strengths to offset the energy loss) and
`twiss6d(radiation_analysis=True, polarization_analysis=True)` finds a
proper closed orbit.

**2. `TubeFitter.to_line()`/`to_multipole_line()` build their own implicit
`xt.Environment`** (unlike `SplineBorisSequence.to_line(env=env)`, which
009 uses to put the undulator directly in the ring's env). Passing the
resulting `Line` straight to `line_sls.insert(undulator_line, ...)` crashes
(`KeyError` inside `_resolve_s_positions`/`get_table()`) because `insert()`
resolves element names against `self.env`, not the undulator's own env. Fix:
`env.import_line(undulator_line, line_name=...)` first (same pattern as
`004b_undulators_in_sls_ring.py`'s `env.import_line(undulator, line_name=
'undulator')`), then insert the *imported* line object.

A further wrinkle for the `_both` cases (undulator inserted at 2 locations):
reusing the same imported line object for both `insert()` calls reuses the
same underlying element instances at two different `s` positions, which
`compensate_radiation_energy_loss()` explicitly rejects (`"Line must not
contain repeated elements..."`). Fix: `import_line()` once *per insertion
point*, each with a distinct `line_name` (`f"{kind}_undulator_{key}"`), so
each location gets its own element copies.

## What's recorded and how equilibrium polarization is fit

Per turn: geometric emittances `gemitt_x/y/zeta` (`_compute_geometric_emittances`,
ported from 014) and mean spin vector / polarization `P(t) = |<spin_vec>|`
(`_compute_polarization`, ported from 015). Bunch generated fully polarized
along y (`spin_y=1`) at equilibrium emittance, tracked under
`configure_radiation(model="quantum")`.

Equilibrium polarization, per the user's explicit request, mirrors 015's
formula (`P_eq = P_inf / (1 + tau_pol/tau_depol)`, eq. 8.37) with `P_inf`/
`tau_pol` from Twiss (`spin_polarization_inf_no_depol`/
`spin_t_pol_component_s`) -- but `tau_depol` comes from a **straight-line**
fit of `P(t)` restricted to `turn >= FIT_START_TURN` (default 2000, CLI
`--fit-start-turn`), not the exponential fit 015 later switched to (see that
branch's commit `6bad42e43`, "Fitting exponential, linear does not quite
seem to cut it"). The user asked specifically for linear decay from turn
2000, so this script uses `_fit_linear_depolarization` (015's earlier,
pre-exponential version) with the start-turn restriction added on top, to
skip whatever initial transient exists before the bunch/spin distribution
settles.

`nemitt_x`/`nemitt_y` are floored at `1e-13` before bunch generation --
`no_undulators` has essentially exact zero vertical emittance/coupling
(`eq_gemitt_y ~ 1e-45`, no undulator-induced skew coupling), and
`generate_matched_gaussian_bunch` needs a nonzero width.

## Smoke-tested, not run at production scale

All 7 cases were run end-to-end (`--n-turns 25 --n-part 15`, well under
`--fit-start-turn`, so `P_eq (derived)` correctly comes out `nan` -- not
enough post-transient turns to fit) and completed without errors, confirming
the fit/correction/insertion/tracking pipeline is sound for all 3 undulator
kinds x both single/double placement patterns.

Sanity-check numbers from that smoke run (Twiss-level, not tracking-fit --
these don't depend on turn count) came out physically sensible:
`tau_depol (Twiss)` is `inf`/`~1e17 s` for `no_undulators` (no coupling
source), drops to `~3.6e6 s` (single SplineBoris undulator) down to
`5.8e4 s` for `sb_both`, while the Multipole-model equivalents stay ~2-4
orders of magnitude larger (`~9.5e8 s` single, `~2.4e8 s` both) -- consistent
with the Multipole model discarding the longitudinal field structure /
Hermite boundary-derivative terms that drive `dn/ddelta`-based
depolarization (see the "gotcha" in
[tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md)); the
coarse model is expected to *underestimate* undulator-driven depolarization.
This is exactly the SplineBoris-vs-Multipole comparison the script exists to
make, and it's already visible at the Twiss level before any tracking-fit
noise enters.

Tracking throughput is noisy at very low particle counts (thread-overhead
dominated -- the 15-particle smoke run measured ~22-240 turns/s depending on
case, vs. an isolated 200-particle/no-undulator/no-spin benchmark done
earlier in this session at ~10,900 particle-turns/s); no reliable
particle-turns/s figure exists yet for the production defaults
(`--n-turns 10000 --n-part 1000`, x7 cases). Following 015's own precedent
of not guessing at a number it hasn't measured: the user should expect this
to take a while (likely tens of minutes to a few hours across all 7 cases)
and may want to start with a reduced `--n-turns`/`--n-part` or a `--cases`
subset before committing to a full run. Not run at production scale in this
session.

## Not done / open

- No convergence scan on `--n-turns`/`--n-part` (mirrors the same open item
  in 015's notes on the other branch).
- `results/` (`.npz` per case) is created next to the script but gitignored
  like other example run outputs; nothing has been committed there.
- Corrector positions (edges + 0.02/0.1 m insets) were reused verbatim from
  `009_sls_undulators_closed_spin_radiation.py`/`004a_build_undulator.py`
  without retuning -- the user explicitly said this was fine ("feel free to
  pick the locations of the correctors yourself").

## Update (same day): misalignment study, corrector-sharing, PDF output to cernbox

The user asked for three more things, all implemented:

**1. `--offset-x` misalignment scan.** A rigid horizontal shift applied to
the undulator field elements (`_apply_shift_x`, same mechanism as
`007_sls_undulator_tune_displacement.py`'s (renamed `007a_...py` on
2026-08-25, see
[sls_undulator_tune_displacement_007a.md](sls_undulator_tune_displacement_007a.md))
`line_sls[nn].shift_x = dx`,
which already scans exactly `-0.5e-3` to `0.5e-3` m -- the 0.5 mm the user
asked for). Applied *before* the correctors are built, so the correctors
compensate for the misalignment rather than ignoring it (unlike 007, which
matches at zero offset once and then scans the offset without rematching --
here each `--offset-x` value gets its own matched correction).

**2. Corrector-sharing strategy (`--corrector-strategy`, default
`shared`).** The user asked directly which is the "more fair" comparison
under a misalignment: match each undulator model's correctors
independently, or match once (on SplineBoris) and reuse those same
strengths on the Multipole line. They leaned towards reuse; agreed and made
it the default, for this reason: matching separately would let *each
model's own correction* re-absorb the very difference the whole script
exists to measure (the two field models predict different orbits from the
same physical misalignment; if each is independently forced back to
x=y=px=py=0 at the segment ends, that boundary-condition matching can mask
how differently the two models actually respond internally). Reusing one
model's correctors on the other instead holds the correction fixed and lets
the field-model difference show up in the tracked trajectory/emittance/
depolarization results, which is a cleaner isolation of what's being
studied. `_correct_undulator_orbit()` now takes an optional
`corrector_values` dict -- if given, it skips matching and just sets the 8
knobs directly; `_build_undulator_lines()` matches on `line_sb` first and
passes its resulting dict to `line_mult` when `corrector_strategy=="shared"`.
`--corrector-strategy separate` is kept available for an explicit A/B check
if the user wants to quantify how much this choice matters.

**3. Midpoint vs. integrated field for `to_multipole_line()`.** The user
wondered whether the Multipole model should use the field at the region's
*midpoint* (matching what a single Boris step would sample) rather than the
integrated/averaged field, given the study's focus on longitudinal-field
effects. No code change was needed here -- `to_multipole_line()` already
uses the Hermite `mean` parameter (`param_index=4`), which
[tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md)
confirms is the field *averaged over the whole region* (a true integral,
divided by region length), not a point sample at any single s -- so the
existing implementation already matches the "integrated field" interpretation
the user leaned towards. Relayed this back rather than changing anything.

**4. PDF output to cernbox, matching the fcc_ee_solenoid convention.**
Figures now save as PDF to `PLOT_DIR` (default
`~/cernbox/Pictures/SLS_Undulator_Studies`, overridable via
`SLS_UNDULATOR_PLOT_DIR`) instead of PNG under the repo's `results/` --
mirrors `examples/fcc_ee_solenoid/aperture_study_io.py`'s `PLOT_DIR`/
`FCC_SOLENOID_PLOT_DIR` pattern on `fcc_da_ma` exactly, including
`format_tag_float()` copied verbatim and the `__xoff<value>` filename tag
style (e.g. `SLS_sb_0210_1000p_10000t__xoff0p0005.pdf` for 0.5 mm). Reason
for the override env var and cernbox destination: the user runs this script
on a different machine than the one used for development, and cernbox sync
gets the PDFs onto whichever machine they're actually looking at.
`.npz` per-turn arrays still save locally next to the script
(`examples/splineboris/results/`), matching the DATA_DIR/PLOT_DIR split on
the other branch.

Smoke-tested (`--n-turns 5-10 --n-part 5-10`, both `sb_0210`/`mult_0210`, both
`shared` and `separate` corrector strategies, with `--offset-x 0.5e-3`):
PDF lands at the correct cernbox path with the expected filename tag, and
`shared` visibly skips the second `match()` call (no second "Matching:
model call" log block) while `separate` runs it. Not run at production
scale.

## Update (same day): `--replot` from saved data

The user asked to be able to replot without rerunning. The `.npz` already
saved everything needed for the raw per-turn traces, but not quite enough to
redraw the figure exactly (`fit_p0`/`fit_slope`/`fit_tau_depol_turns` and the
case `title` weren't saved, only the derived `fit_tau_depol_s` was) --
added those, plus `name`, to the `np.savez(...)` call.

New `--replot` flag: when passed (together with the same
`--cases`/`--n-turns`/`--n-part`/`--offset-x` used for the original run, so
the filename stem matches), `main()` skips `_build_undulator_lines()` and
`_run_case()` entirely and instead calls a new `_replot_case()` per
selected case, which loads `RESULTS_DIR/{stem}.npz`
(`_case_stem()` factored out of `_run_case()` so both paths compute the
same filename), rebuilds the figure via the existing `_plot_case_figure()`,
and re-saves the PDF to `PLOT_DIR`. Raises a clear `SystemExit` pointing at
the missing file if the npz for a requested case/turn-count/particle-count/
offset combination doesn't exist yet.

Smoke-tested: ran `sb_0210` normally (10 turns/10 particles), deleted the
PDF, reran with `--replot` on the identical CLI (same n-turns/n-part) --
regenerated the identical PDF near-instantly (no "Fitting SLS undulator" /
"Tracking N particles" output at all) with an identical summary table.

## Update (same day): offset now applied to the correctors too

Originally `_apply_shift_x(offset_x)` ran *before* `_correct_undulator_orbit`
inserted the 4 correctors, so only the undulator field elements got
misaligned -- the correctors stayed at `shift_x=0`. The user pointed out
xtrack elements have a built-in `shift_x`/`shift_y` alignment offset and
asked for it on the whole undulator assembly, correctors included (physically
the correctors are mounted on the same girder as the undulator and move with
it).

Fix: moved the `_apply_shift_x()` call inside `_correct_undulator_orbit()`,
right after the correctors are inserted but before matching/setting their
values -- so `offset_x` now lands on all 4 correctors as well as the field
elements, and (for the default `matched` path) the corrector-orbit match
itself is solved *with* the correctors already sitting at the offset
position. `_build_undulator_lines()` no longer calls `_apply_shift_x()`
directly; it just threads `offset_x` through to `_correct_undulator_orbit()`
for both the SplineBoris and Multipole lines.

**Correction (same day):** the paragraph above overclaimed what `shift_x`
does for the correctors -- verified empirically it's actually a **no-op**
for them, not just kick-strength-preserving. Reason: `xt.Multipole`'s
`shift_x` is xtrack's *generic* per-element misalignment field (from the
`allow_rot_and_shift=True` base-class mixin in `base_element.py` --
shift the particle into the element's local frame, apply the kick, shift
back out), which is architecturally different from `SplineBoris`'s own
`shift_x` (`splineboris.py` sets `allow_rot_and_shift = False` and defines
a custom field instead, used only to offset the field-map lookup position
inside the spatial Boris integration -- confirmed by tracing
`track_splineboris.h`: the particle's real `x` is read once at entry,
`x - shift_x` is used only for field evaluation at each substep, and the
unshifted `x` is written back at the end). For a *thin* element with only
`knl[0]`/`ksl[0]` (a position-independent kick), the generic shift-in/
shift-out exactly cancels and the kick itself doesn't depend on position
either -- so `shift_x` on our correctors changes nothing. Verified directly:
tracked a thin `xt.Multipole(knl=[...], ksl=[...])` with `shift_x=0` vs
`shift_x=0.5e-3` and got bit-identical `x`/`px`/`y`/`py` output. Left the
code as-is (harmless, and would start mattering if the correctors ever
gained quadrupole-order content) but flagging the claim as wrong here so it
doesn't get trusted later. The Multipole *undulator* elements from
`to_multipole_line()` are thick and do use this same generic mechanism, and
there it's physically meaningful (feed-down from the fitted quadrupole/
sextupole-order terms) -- consistent with SplineBoris's field-elements case,
just via a different code path under the hood.

Smoke-tested: verified all 4 correctors carry the expected `shift_x` after
`_correct_undulator_orbit(line_sb, offset_x=0.5e-3)`, matching still
converges to `~2e-12` penalty, and a full `sb_0210`/`mult_0210` smoke run
with `--offset-x 0.5e-3` completes and saves PDFs as before.
