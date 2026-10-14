# `TubeFitter.to_line()` -- direct SplineBoris-line builder (2026-08-11)

> **HISTORICAL (checked 2026-10-14).** `TubeFitter.to_line()` no longer exists. After the 2026-10-08 rewrite (`a8e5ecc17` and `0f77ee734`, after TubeFitter simplifications on 2026-10-06): `FieldFitter` and `SplineBorisSequence` were deleted, and `TubeFitter` now only produces on-axis multipoles at its frames, while the new `LongitudinalFitter` does the quartic-B-spline fit and the `to_line()`/`to_multipole_line()` export. See `00_overview.md`. The equivalent is `LongitudinalFitter.to_line(multipole_order, steps_per_point, field_tol, r_ref, shift_x, shift_y, radiation_flag)`: one `SplineBoris` per B-spline knot interval, named `splineboris_NNN`.

Adds a dedicated line-builder method directly on `TubeFitter`, sidestepping
`SplineBorisSequence`'s generic (and heavier) region-reconciliation logic.
Companion notes:
[fieldfitter_vs_tubefitter_comparison.md](fieldfitter_vs_tubefitter_comparison.md),
[tube_hermite_export_boundary_bug.md](tube_hermite_export_boundary_bug.md),
[hermite_to_polynomial_reconstruction_bug.md](hermite_to_polynomial_reconstruction_bug.md).

## Why a separate path from `SplineBorisSequence`

`SplineBorisSequence` was built for `FieldFitter`, where different field
components/derivatives can have independently peak-found regions that
straddle each other -- it has to collect all unique `(s, idx)` boundaries
across fields, mask which params are valid for each merged sub-region, and
re-derive Hermite data for a canonical sub-interval via
`_split_hermite_to_canonical`, which round-trips through
`hermite_to_polynomial` (the same function that had the two reconstruction
bugs documented in the companion notes above).

`TubeFitter._populate_df_fit_pars`
([tube_fitter.py:778](../../../xtrack/_temp/splineboris/tube_fitter.py#L778))
builds every `(field_component, derivative_x)` from the *same* shared
region grid (`self.frames`) -- confirmed by inspection of the code, not
just assumed. So none of `SplineBorisSequence`'s reconciliation machinery
is needed: each region's 5 stored Hermite params (`param_index` 0..4)
already map directly onto a `Spline4`'s fields (`val_start, der_start,
val_end, der_end, integral`), since `_hermite_from_bspline` returns them
in exactly that order.

## What was added

`TubeFitter.to_line(multipole_order, steps_per_point=1, shift_x=0.0,
shift_y=0.0, radiation_flag=0)` --
[tube_fitter.py](../../../xtrack/_temp/splineboris/tube_fitter.py), placed
right after `save_fit_pars`. A single flat
`df_fit_pars.groupby(["s_start", "s_end", "field_component",
"derivative_x"])` pass (not nested per-region groupby calls) builds a
`Spline4` per group directly from `param_value` (sorted by `param_index`),
assembles `by`/`bx` tuples (zero `Spline4` for any multipole order beyond
what was actually fit), and chains the resulting `SplineBoris` elements
into a plain `xt.Line`. No dependency on `SplineBorisSequence` or
`hermite_to_polynomial` at all.

## Verification

Ad hoc check (undulator field map, `n_frames=300`, `deg=2`,
`multipole_order=3`): built the line both via `to_line()` and via
`SplineBorisSequence(fitter.df_fit_pars, multipole_order=3)`, then compared
`elem.get_field(x, y, s_local)` at 500 random `(x, y, s)` points across the
whole domain (accounting for the different local-vs-global `s` origin
between the two). Results agree to `~4e-14 T` (floating-point noise, not a
real discrepancy), and per-element `length`/`n_steps` match exactly between
the two constructions. Not yet added as a permanent test in
`tests/test_splineboris.py`.

## Design note: why on `TubeFitter`, not `SplineBorisSequence`

Discussed explicitly with the user. `SplineBorisSequence` also provides
`get_field(x, y, s)` over a whole sequence, `s_starts`/`s_ends` bookkeeping,
and `from_csv` -- none of which `to_line()` reproduces (it returns a bare
`xt.Line`). Chosen to live on `TubeFitter` anyway because (a) a plain
`xt.Line` is what's actually needed for tracking, (b) each `SplineBoris`
element already has its own `get_field(x, y, s_local)`, so per-s field
queries can be done by picking the right element by hand if ever needed
(no need to reintroduce a sequence wrapper for that alone), and (c) the
user wants `TubeFitter` to stay self-contained rather than reliant on a
separate module -- notably because the user is now considering discarding
`FieldFitter` altogether (not decided or actioned yet; see below).

## Open follow-up (mentioned, not yet done)

- No permanent regression test added yet for `to_line()` vs
  `SplineBorisSequence` agreement.
- Not wired into any of the `003*` example scripts yet.
- **User is considering dropping `FieldFitter` entirely** at some point.
  If that happens, `SplineBorisSequence` itself (built specifically to
  handle `FieldFitter`'s independently-per-field regions, via
  `_split_hermite_to_canonical` and the boundary-merging/masking logic)
  would become dead code, since `to_line()` would be the only line-builder
  left in use. Not decided or scheduled -- flagged here so the connection
  isn't lost whenever that decision is revisited.
