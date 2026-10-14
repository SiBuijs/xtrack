# Plan: retire FieldFitter, migrate TubeFitter off pandas (2026-08-21)

> **PARTLY OVERTAKEN (checked 2026-10-14).** Goal 1's FieldFitter half is done: `FieldFitter`, `SplineBorisSequence` and their tests, examples and CSV fixtures were deleted in `a8e5ecc17` (2026-10-08), as part of a rewrite that did not follow this plan. `df_fit_pars` no longer exists: `TubeFitter` returns plain arrays (`on_axis_multipoles()` -> `(z, F, names)`), and `LongitudinalFitter` uses no pandas at all. **Still open:** `TubeFitter` itself still takes its `raw_data` as a `pd.DataFrame` with a `('X','Y','Z')` MultiIndex, and the examples and tests load maps with pandas. The 'blocked on a decision' status below is from 2026-08-21 and was not re-checked.

**Status: planned, not started. Blocked on a decision with Giovanni Iadarola.**

Companion notes: [tubefitter_canonical_and_upstream_merge.md](tubefitter_canonical_and_upstream_merge.md)
(FieldFitter's design history and the upstream merge that brought in Giovanni
Iadarola's parallel `FieldFitter`/`SplineBoris` work), [tube_fitter_to_line.md](tube_fitter_to_line.md),
[tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md).

## Motivation

Two separate goals turned out to unlock each other:

1. **Avoid a pandas dependency** in the splineboris fitting code where reasonably
   possible, using `xtrack`'s own `xt.Table` instead.
2. **Simplify `TubeFitter.df_fit_pars`**: `region_name`/`s_start`/`s_end`/
   `idx_start`/`idx_end` are duplicated across every `(field_component,
   derivative_x)` combination for a given region, even though `TubeFitter`
   (unlike `FieldFitter`) uses one shared frame grid for all fields --
   `to_line()`'s own docstring already says so
   ([tube_fitter.py:396-406](../../../xtrack/_temp/splineboris/tube_fitter.py#L396)).
   With `deg=2` that's 7 field/derivative combinations x 5 Hermite params = 35
   rows per region all carrying identical boundary metadata.

Both are blocked by the same thing: `df_fit_pars`'s schema
(`SplineBorisSequence.FIELD_FIT_INDEX_COLUMNS`,
[splineboris_sequence.py:39-48](../../../xtrack/_temp/splineboris/splineboris_sequence.py#L39))
is shared with `FieldFitter`, whose regions genuinely do differ per field/
derivative (independent peak-finding per field,
[field_fitter.py:558-604](../../../xtrack/_temp/splineboris/field_fitter.py#L558)), so
it can't be freely reshaped while both fitters are in play. Pandas usage is
also extremely lopsided: `field_fitter.py` (932 lines) is where almost all of
the hard `.xs()`/`.loc[(field, der)]`/`groupby(level=...)`/incremental
`pd.concat` MultiIndex machinery lives; `TubeFitter`'s own pandas usage is
comparatively small and self-contained (one `.xs()` call, two `groupby()`
calls on an already-flat `reset_index()`'d frame, everything else already
drops into `.to_numpy()`/numba kernels).

So: dropping `FieldFitter` removes the hardest pandas code directly, *and*
frees `TubeFitter`'s own schema to be redesigned without an external
compatibility constraint.

## Decision gate -- do not start until this is resolved

`FieldFitter` is not just this branch's code. `upstream/main` uses it directly in:
- `examples/splineboris/006_solenoid.py` (upstream's own canonical example,
  currently byte-identical on this branch, restored verbatim during the
  earlier merge)
- `examples/splineboris/003_fieldfitter_basic_usage.py` (this branch's `003a`,
  same file renamed)
- `tests/test_splineboris.py` (an upstream-authored `FieldFitter` smoke test,
  plus this branch's own `test_splineboris_tubefitter_vs_fieldfitter_undulator`
  cross-check)

It was developed independently upstream by **Giovanni Iadarola
(`giovanni.iadarola@cern.ch`, git handle `giadarol`)** in parallel with this
branch's own `FieldFitter`/`TubeFitter` work, then merged in
(`tubefitter_canonical_and_upstream_merge.md`). No prior deprecation
discussion exists in the notes -- only an explicit deferral from that session:
*"We can rework everything around TubeFitter afterwards."* This plan is that
deferred work, now being scoped out.

**Before any of the steps below: confirm with Giovanni whether `FieldFitter`
is being retired from `upstream/main` too, or only dropped from this
branch's own PR while staying alive upstream.** That answer determines
whether `006_solenoid.py` needs a `TubeFitter`-based replacement (coordinated
with upstream) or can simply diverge/be dropped from this branch's PR.

Also worth deciding at the same time: what replaces
`test_splineboris_tubefitter_vs_fieldfitter_undulator` as a fitter-correctness
cross-check once there's no second independent fitter to compare against
(e.g. validation against an analytic/known field instead).

## Scope once unblocked

**Delete:**
- `xtrack/_temp/splineboris/field_fitter.py` (932 lines)
- `examples/splineboris/003c_fieldfitter_vs_tubefitter.py` (its entire premise
  is comparing the two fitters)
- `examples/splineboris/003a_fieldfitter_basic_usage.py`, unless kept as a
  historical/upstream-parity example (decide alongside the decision gate above)

**Simplify or delete:**
- `xtrack/_temp/splineboris/splineboris_sequence.py` (306 lines) -- its core
  job is reconciling regions that differ per field/derivative across an
  arbitrary fitter's output. That situation only arises with `FieldFitter`;
  `TubeFitter.to_line()` already replaces it more simply for the
  single-shared-grid case. Once `FieldFitter` is gone, check whether anything
  still needs `SplineBorisSequence.from_csv()` (loading a previously-saved
  fit without a live `TubeFitter` object) -- if so, it needs a small
  equivalent against the new flat schema; if not, delete the whole class.

**Redesign (the actual pandas migration, in `xtrack/_temp/splineboris/tube_fitter.py`):**
- `df_raw_data` (3-level `(X, Y, Z)` MultiIndex, flat `Bx`/`By`/`Bs` columns)
  -> flat 1-D arrays or an `xt.Table` with plain `X`/`Y`/`Z`/`Bx`/`By`/`Bs`
  columns. Most existing access is already `.get_level_values()` /
  `.to_numpy()`; only `_set_df_on_axis`'s single `.xs((x0, y0), level=["X",
  "Y"])` call needs a real replacement (a boolean mask is sufficient since
  `xy_point` is fixed).
- `df_on_axis_raw` / `df_on_axis_fit` (row index `Z`, column MultiIndex
  `(field, derivative)`) -> a dict of flat arrays keyed by `(field, der)`, or
  an `xt.Table` with named columns like `Bx_0`, `Bx_1`, ... instead of a
  hierarchical column index.
- `df_fit_pars` (8-level index) -> flat/wide `xt.Table`, now free to drop
  `region_name` (confirmed write-only/dead in `TubeFitter`'s own code --
  never read back, only present for the old shared-schema contract) and to
  stop repeating `s_start`/`s_end`/`idx_start`/`idx_end` per field/derivative.
  **Open design choice, decide during implementation:** one row per region
  with columns for every `(field, der, param)` combination (very wide, but
  `multipole_order`-dependent column set), vs. one row per `(region, field,
  der)` with 5 named value columns (`val_start`, `der_start`, `val_end`,
  `der_end`, `mean`) instead of `param_index`/`param_name` (still some
  duplication of `s_start`/`s_end` per field/der, but far less schema
  coupling to fit configuration).
- `save_fit_pars()`/CSV round-trip -> `xt.Table.to_csv()`/`from_csv()`
  instead of `df.to_csv(index=True)` / `pd.read_csv(..., index_col=[...])`.
  `test_data/sls/undulator_fit_pars.csv` and `undulator_fit_pars_rotated.csv`
  are pinned to the current schema and will need regenerating.
- Examples' `pd.read_csv(...)`/`pd.DataFrame({...})` calls for loading raw
  field maps (`003b`, `003d`, `004a-c`, `006`\*, `008`-`012`) are already flat
  dict-of-arrays -- mechanical conversion to `xt.Table`, independent of
  everything else above.

**Explicitly out of scope:** `xtrack/table.py` itself already depends on
pandas (`to_tfs`/`from_tfs` via `tfs-pandas`), so this does not remove pandas
from `xtrack` as a dependency overall -- only from the splineboris fitting
code's own internals and examples.

## Suggested order of operations

1. Resolve the decision gate with Giovanni (scope of `FieldFitter` removal,
   what replaces the cross-fitter test).
2. Delete `field_fitter.py` and the now-pointless `003c` comparison example;
   handle `003a`/`006_solenoid.py` per the decision gate.
3. Prune or delete `splineboris_sequence.py`.
4. Redesign `TubeFitter`'s three internal frames per above (this is the real
   work -- do it as its own reviewable change, `df_raw_data` first since
   everything else depends on it, then the on-axis frames, then
   `df_fit_pars`).
5. Update `save_fit_pars`/CSV round-trip; regenerate the two pinned test CSV
   fixtures under the new schema.
6. Convert examples' flat `pd.read_csv`/`pd.DataFrame` calls to `xt.Table`.
7. Full `tests/test_splineboris.py` run after each of steps 2-6
   (`XSUITE_ALLOW_NO_PREBUILT_KERNELS=1`, per this branch's established
   convention).

## Verification

- `pytest tests/test_splineboris.py` green after each stage above, not just
  at the end.
- Numerically diff `df_fit_pars` (or downstream `SplineBoris` element
  parameters / tracking results for a fixed test case) before and after the
  schema redesign, to confirm the reshape is representation-only and doesn't
  change any fitted values.
- Spot-run the surviving example scripts end-to-end (at least whichever
  becomes the canonical solenoid example, `004a_build_undulator.py`, and
  `012_sls_undulator_spin_emittance_evolution.py`) to confirm they still
  produce the same plots/output after the `xt.Table` conversion.

## Open follow-up

- Exact `df_fit_pars` wide-schema shape (see "Open design choice" above) --
  not decided yet, deliberately left for implementation time.
- Whether `SplineBorisSequence` survives as a thin CSV loader or is deleted
  outright.
- What replaces `test_splineboris_tubefitter_vs_fieldfitter_undulator`.
- Whether `006_solenoid.py` diverges from `upstream/main` or gets a
  coordinated `TubeFitter`-based replacement upstream.
