# splineboris claude_notes — overview

Session-handoff notes for `examples/splineboris/`, on branch
`boris_symplectic_error`, written 2026-07-21. This directory studies
`xtrack.SplineBoris` / `SplineBorisSequence`
(`xtrack/beam_elements/elements_src/track_splineboris.h`,
`xtrack/_temp/splineboris/`) — a piecewise-polynomial (Hermite spline)
field-fit Boris pusher, compiled into production C code, as opposed to the
pure-Python `BorisSpatialIntegrator` studied in
`../../boris_spatial/claude_notes/`. **Read that directory's notes first** —
the canonical-vs-kinetic-momentum symplectic-error bug found and fixed there
is directly relevant to the open question here.

**Treat as a snapshot, not ground truth**: re-check the source if a
referenced file/function no longer exists, and check `git log`/`git status`
in this directory for what's changed since 2026-07-21.

## Script pipeline

- `001_splineboris_api_intro.py` … `007_sls_undulator_tune_displacement.py`:
  earlier exploratory/validation scripts (not touched this session) —
  API intro, a homogeneous rbend check, `FieldFitter` basics, undulator
  spin/radiation tracking against the SLS lattice, and an undulator tune
  displacement study.
- `006_solenoid.py` builds the same analytical solenoid used below
  (`SolenoidField(L=4, a=0.3, B0=1.5, z0=20)`), fits it with `FieldFitter`,
  and assembles it into a `Line` of `SplineBoris` elements via
  `SplineBorisSequence`; compares trajectories/spin against
  `BorisSpatialIntegrator` and `VariableSolenoid` references.
- `008_study_symplectic_error.py` — new this session. Mirrors
  `boris_spatial/005_study_symplectic_error.py`: same solenoid/reference
  particle, sweeps `steps_per_point` (→ total Boris steps) through the
  `SplineBorisSequence` built from `006`'s solenoid, and measures the
  one-turn map's symplectic deviation `‖RᵀSR−S‖` via `line.get_R_matrix`,
  plotted against the `BorisSpatialIntegrator` (canonical/corrected)
  reference curve from `005`. Caches the `FieldFitter` fit to
  `_008_results/solenoid_fit_pars.csv` (loaded via
  `SplineBorisSequence.FIELD_FIT_INDEX_COLUMNS`, skipping the refit if the
  file already exists).

## Finding: symplectic error plateaus, doesn't converge with `steps_per_point`

**Setup:** same solenoid/particle as `boris_spatial/005`, `field_tol=1e-4`,
`multipole_order=2`, `n_fit_points=5000` (matching `006`).

**Observed:** sweeping `steps_per_point` from 1 to 100 (total Boris steps
5,000 → 500,000, piece count fixed at ~1418) left `sympl_error` essentially
flat at **~27** — no `1/n_steps²`-style convergence at all. For comparison,
the *uncorrected* (`vector_potential_callable=None`) `BorisSpatialIntegrator`
plateaus at **~28** on the same solenoid (see
`../../boris_spatial/claude_notes/00_overview.md`), while the *corrected*
one reaches ~0.02 at 5,000 steps and keeps shrinking. The `SplineBoris`
result's magnitude and step-count-independence match the *uncorrected*
case far more closely than the corrected one, despite `SplineBoris` already
doing a per-element canonical↔kinetic conversion.

**Important correction to the stale note in `boris_spatial/00_overview.md`
(bottom, "Separately, still unresolved"):** that note says
`track_splineboris.h` "writes kinetic `px` straight to `LocalParticle_set_px`
without an `ax` correction" — this is **no longer true**. As of this branch,
`track_splineboris.h` already converts canonical→kinetic at each element's
entry and kinetic→canonical at its exit, using `evaluate_A` of that
element's own fitted field
([track_splineboris.h:110-116](../../../xtrack/beam_elements/elements_src/track_splineboris.h#L110-L116),
[:354-360](../../../xtrack/beam_elements/elements_src/track_splineboris.h#L354-L360)).
That note predates commits `4ef8b9688` ("Made a corrected
BorisSpatialIntegrator... reduces symplectic error") and `7a071db28`
("going to correct it on SplineBoris now as well") on this branch.

**Hypothesis (not yet verified) for the plateau:** each `SplineBoris`
element does its own local entry/exit conversion using its *own*
independently-fit Hermite coefficients. If piece `i`'s exit `Ax` (at its
local `s=length`) doesn't exactly equal piece `i+1`'s entry `Ax` (at its
local `s=0`, same physical `s`) — plausible, since each piece's vector
potential is only constrained by its own boundary `B`/derivative fit, not
by a shared running integration constant across the whole
`SplineBorisSequence` — then the exit-then-entry conversions at that
boundary don't cancel, leaving a small residual canonical-momentum offset
at *every* one of the ~500–1,400 piece boundaries. That would explain why
the error is insensitive to `steps_per_point` (which only changes steps
*within* a piece, never the piece count/boundaries) while still being of
the same order as the "no correction at all" case. Not yet checked: whether
`evaluate_A` (`splineboris_src/spline_A_field_eval.h`) integrates each
piece's vector potential from a purely local reference point, or carries an
additive constant threaded through by `SplineBorisSequence` (cf.
`_split_hermite_to_canonical` in `splineboris_sequence.py`, which does
handle continuity for the *B*-field Hermite re-expression, but not
obviously for `A`).

**Unrelated bug also found and fixed this session** (same investigation,
different code path): `track_splineboris.h`'s *radiation*-tracking block
was subtracting `LocalParticle_get_ax(part)`/`ay` a second time from a
local `px`/`py` that had already been converted to kinetic momentum at the
element's entry — a double vector-potential subtraction. Fixed by using
`px / P0`, `py / P0` directly (see git history on this branch). This does
**not** explain the plateau above — `radiation_flag=0` in the symplectic-error
study, so that code path is never entered — but is worth knowing about if
revisiting radiation with `SplineBoris`. A related staleness issue was also
noted (not fixed): `magnet_spin`
(`track_magnet_radiation.h:96-145`) reads `LocalParticle_get_px/ax(part)`
directly rather than a per-step value, but `track_splineboris.h` only calls
`LocalParticle_set_px`/`set_py` once, after the whole `n_steps` loop — so
spin precession uses the frozen entry-face momentum for every substep of a
multi-step `SplineBoris` element instead of the evolving one. Deferred by
the user for a later session.

## Next step

Verify the piece-boundary `Ax`/`Ay` continuity hypothesis directly — e.g.
evaluate `evaluate_A` at a shared boundary `s` from both the exiting and
entering piece's coefficients and compare. If confirmed, the fix likely
belongs in `SplineBorisSequence._build_elements` (or a new helper next to
`_split_hermite_to_canonical`): compute a per-piece additive offset to
`Ax`/`Ay` so they agree with the running total from the sequence's global
`s_start`, the same way piece continuity is already handled for the
`B`-field Hermite coefficients themselves.
