# Making TubeFitter canonical in the tests, and the upstream merge that interrupted it (2026-08-11)

> **HISTORICAL (checked 2026-10-14).** The 'TubeFitter canonical, FieldFitter kept for comparison' state described here ended with the 2026-10-08 rewrite (`a8e5ecc17` and `0f77ee734`, after TubeFitter simplifications on 2026-10-06): `FieldFitter` and `SplineBorisSequence` were deleted, and `TubeFitter` now only produces on-axis multipoles at its frames, while the new `LongitudinalFitter` does the quartic-B-spline fit and the `to_line()`/`to_multipole_line()` export. See `00_overview.md`. `SplineBoris`/`Spline4` now live in `xtrack/beam_elements/splineboris.py`, not `elements.py`. Several scripts and tests named below no longer exist; `tests/test_splineboris.py` now uses `undulator_fit`/`solenoid_fit` fixtures built with `TubeFitter` + `LongitudinalFitter`. The account of the 2026-08-11 upstream merge is still accurate as history.

Two things happened back to back in the same session: (1) `TubeFitter` was
made the canonical fitter for `tests/test_splineboris.py` while keeping
`FieldFitter` around for comparison, and (2) that work got interrupted by an
accidentally-started `git pull --rebase` against `upstream/main` (1316
commits ahead), which had to be aborted and redone as a plain merge, pulling
in a large, independent parallel development of `SplineBoris`/`FieldFitter`
by a colleague (`giadarol <giovanni.iadarola@cern.ch>`). Companion notes:
[tube_fitter_to_line.md](tube_fitter_to_line.md),
[fieldfitter_vs_tubefitter_comparison.md](fieldfitter_vs_tubefitter_comparison.md),
[tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md).

## Part 1: TubeFitter canonical in the tests

User's direction (after an initial "replace FieldFitter entirely" framing
was corrected): *"I'd like to make TubeFitter canonical in the tests, so
that the tests no longer depend on FieldFitter. The examples demonstrating
the difference between FieldFitter and TubeFitter, I want to keep."*

Before this, `TubeFitter` had already been confirmed (this session) to pass
`test_splineboris_variable_solenoid_radiation` at `n_frames=2000`/`kernel=
"tent"`, using only ~12% of the test's tolerance budget on the strictest
check (`dE_ds`, the radiation-loss comparison). At `n_frames=500` it fails
specifically on that `dE_ds` check -- confirming the test's existing
`n_frames=2000` choice isn't arbitrary headroom.

Changes made to
[tests/test_splineboris.py](../../../tests/test_splineboris.py):

- Deleted the `solenoid_vs_varsol_fit_pars_df` fixture (the only live
  `FieldFitter(...)` fitting call in the test file) and
  `test_splineboris_solenoid_vs_variable_solenoid` -- once its fixture
  stopped mattering, it was a byte-for-byte duplicate of the existing
  `test_splineboris_tubefitter_solenoid_vs_variable_solenoid`.
- Repointed `test_splineboris_variable_solenoid_radiation` to the existing
  `solenoid_tubefitter_fit_pars_df` fixture, building the line via
  `SplineBorisSequence(...).to_line()` the same way the TubeFitter
  comparison test already did.
- Left `undulator_fit_pars_df`, `undulator_rotated_fit_pars_df`, and
  `test_splineboris_tubefitter_vs_fieldfitter_undulator` alone -- they read
  pre-baked CSVs rather than calling `FieldFitter` live, and the latter is a
  legitimate TubeFitter-vs-FieldFitter cross-check, not redundant.
- Lightly reworded `SplineBorisSequence`'s docstrings
  ([splineboris_sequence.py](../../../xtrack/_temp/splineboris/splineboris_sequence.py))
  to describe it as consuming `df_fit_pars` generically (from either
  fitter), not "FieldFitter output" specifically.

This was executed via Plan Mode and verified (45 passed / 3 pre-existing,
unrelated `PrebuiltKernelNotFoundError` failures) -- before the merge below
folded it into a much larger reconciliation.

## Part 2: the accidental rebase, and choosing to merge instead

Mid-session, the user ran what they described as "I merged with upstream
main" -- investigation of `.git/rebase-merge/` showed this was actually a
**stuck interactive rebase**, 2 of 19 commits replayed, with an unresolved
"deleted by us: `xtrack/beam_elements/elements.py`" conflict. Nothing was
lost (`orig-head` was intact), so the first step was just reporting the
real state accurately before touching anything.

Given the two options -- resolve the rebase commit-by-commit, or abort and
do a single merge -- the user chose **abort and merge**:
`git rebase --abort` followed by `git merge upstream/main --no-edit`.

`upstream/main` turned out to be 1316 commits ahead, including independent
work by a colleague on the exact same feature area
(`SplineBoris`/`FieldFitter`). Asked directly whether the user was aware,
they set explicit ground rules for resolving conflicts:

- Accept upstream's `elements.py` -> per-file module split (including a new
  `splineboris.py`).
- OK if `SplineBorisSequence` was removed or changed upstream.
- OK to renumber examples, as long as files unique to this branch (the
  TubeFitter-comparison ones) sort *after* upstream's canonical numbering.
- Defer deeper reconciliation ("rework everything around TubeFitter") to
  later -- don't over-engineer the merge resolution itself.

## Part 3: what the merge actually changed

**Upstream (non-splineboris-specific) changes worth knowing about:**

- `xtrack/beam_elements/elements.py` (4836 lines) was flattened into
  per-element-type files; `SplineBoris`/`Spline4` now live in
  [xtrack/beam_elements/splineboris.py](../../../xtrack/beam_elements/splineboris.py).
- `Spline4`'s 5th field renamed `integral` -> `mean`; constructing with the
  old `integral=` kwarg now raises `ValueError`.
- `SplineBoris._HERMITE_SUFFIXES` renamed to `_SB_HERMITE_SUFFIXES`.
- `SplineBoris` gained real new features: `knl`/`ksl` (integrated
  multipole-kick strengths, split evenly across Boris substeps), `scale_b`
  (multiplicative field-only scale factor), backtrack/twiss support, and
  native `to_dict`/`from_dict`.
- `FieldFitter.__init__` now calls `self.fit()` automatically -- an
  explicit `.fit()` afterward is now redundant (harmless) rather than
  required.
- `FieldFitter` gained `get_spline_data()`, returning per-region Hermite
  dicts directly, as an alternative to routing through
  `SplineBorisSequence`.
- Despite a commit literally titled "Eradicating SplineBorisSequence",
  `SplineBorisSequence` was **not** actually deleted -- it still exists and
  works; that commit just gave `FieldFitter` the `get_spline_data()`
  alternative path.
- `Particles.kin_xprime`/`kin_yprime` renamed to `kin_xp`/`kin_yp`.

**The 6 conflicted files, and how each was resolved:**

1. `.gitignore` -- union of both sides' additions (`**/claude_notes/`,
   `.idea`, `.claude/`).
2. `xtrack/beam_elements/elements.py` -- deleted (`git rm`), accepting
   upstream's module split. Our branch's only real diff here (`SplineBoris`
   `to_dict`/`from_dict`, ~143 lines) was confirmed redundant with
   upstream's own (better, `knl`/`ksl`-aware) implementation.
3. `examples/splineboris/004_undulator_open_track_spin_radiation.py` /
   `005_sls_undulators_closed_spin_radiation.py` -- modify/delete conflicts
   (upstream added its own new `004a`/`004b` at those numbers). Resolved by
   renaming ours to `008_...py` / `009_...py` (`git show HEAD:<old> ><new>`
   + `git rm`/`git add`), per the "our files sort last" rule.
4. `examples/splineboris/006a_solenoid.py` -- renamed to
   `010_solenoid_tubefitter_comparison.py`; canonical
   `006_solenoid.py` restored verbatim from upstream (it had grown spin-
   tracking setup our older fork didn't have).
   `006b_solenoid_kernel_comparison.py` -> `011_solenoid_kernel_comparison.py`
   (uncontested, plain `git mv`).
5. `tests/test_splineboris.py` -- the big one, see below.
6. (Auto-merged cleanly, no conflict, but needed manual follow-up fixes:)
   [tube_fitter.py](../../../xtrack/_temp/splineboris/tube_fitter.py) --
   import path (`xtrack.beam_elements.elements` ->
   `xtrack.beam_elements.splineboris`), `mean=` rename, and
   `_SB_HERMITE_SUFFIXES` rename. `splineboris_sequence.py` and
   `field_fitter.py` merged cleanly with no manual changes needed (the
   latter already called `.fit()` internally and used the new suffix name).

**`tests/test_splineboris.py` reconstruction:** rather than resolve textual
conflict markers by hand, both versions (`HEAD` = our TubeFitter-canonical
version, `upstream/main` = the colleague's independent version) were parsed
with Python's `ast` module to extract every top-level function (and the
handful of top-level constants sitting between them, initially missed and
added via a second pass) by name, then reassembled: theirs was current for
9 shared-name functions that had gone stale on our side (outdated
`SplineBoris` field names like `hx`, `Bs_hermite`), ours was kept for the
TubeFitter-specific tests and the one deliberately-dropped FieldFitter
duplicate (`solenoid_vs_varsol_fit_pars_df` /
`test_splineboris_solenoid_vs_variable_solenoid`), and upstream's new
`test_field_fitter_get_spline_data_matches_sequence` (testing
`FieldFitter.get_spline_data()`, non-redundant new coverage) was kept
as-is.

## Verification and current state

Full suite run with `XSUITE_ALLOW_NO_PREBUILT_KERNELS=1` (a local kernel-
cache/version mismatch unrelated to the merge itself --
`xtrack` was bumped to `0.111.0` by the merge, invalidating the cached
prebuilt kernels) came back **56 passed, 0 failed** after the merge
resolution.

The merge is committed as `4bd16586d` ("Merged with main", parents
`c50601fb6` ours / `1330d50c5` upstream). At the time this note was written
`tests/test_splineboris.py` and `tube_fitter.py` had further, still-
uncommitted changes on top (`to_multipole_line()` and its tests -- see
[tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md)).

## Open follow-up (explicitly deferred by the user)

"We can rework everything around TubeFitter afterwards" -- deeper
reconciliation between `FieldFitter.get_spline_data()` and
`TubeFitter.to_line()`/`to_multipole_line()`, and further consolidation of
the two fitters' test/example surface, was explicitly punted to a later
session. Not started, not scheduled.
