# `TubeFitter.to_multipole_line()` -- coarse Multipole-line builder (2026-08-11)

> **HISTORICAL (checked 2026-10-14).** `TubeFitter.to_multipole_line()` no longer exists. After the 2026-10-08 rewrite (`a8e5ecc17` and `0f77ee734`, after TubeFitter simplifications on 2026-10-06): `FieldFitter` and `SplineBorisSequence` were deleted, and `TubeFitter` now only produces on-axis multipoles at its frames, while the new `LongitudinalFitter` does the quartic-B-spline fit and the `to_line()`/`to_multipole_line()` export. See `00_overview.md`. The equivalent is `LongitudinalFitter.to_multipole_line(p0c, multipole_order, q0, field_at='mean'|'midpoint', field_tol, r_ref, shift_x, shift_y)`: thick `Multipole`s `multipole_NNN` on the same nodes as `to_line()`, `knl[n] = Delta/brho0 * (By, n)`, `ksl[n] = Delta/brho0 * (Bx, n)`. B_s is dropped with a printed warning, and there are **no** `MultipoleEdge`s (the 2026-08-25 edge experiment stayed reverted). The tests `test_longitudinal_fitter_to_multipole_line_*` replace the ones named below.

Adds a second line-builder alongside `to_line()`
([tube_fitter_to_line.md](tube_fitter_to_line.md)): same region grid, but
each region becomes a thick `xt.Multipole` (a rigidity-normalized kick) built
from the region's *mean* field, instead of a full `SplineBoris` spline
element. Requested by the user directly ("derive the multipole coeffs from
the hermite coefficients that record the average field over the region"),
as a coarse comparison baseline, not a replacement for `to_line()`.

## The conversion

`df_fit_pars`'s Hermite `mean` parameter (`param_index=4`) for
`Bnorm`/`Bskew` at derivative order `n` is exactly the on-axis multipole
coefficient `d^n By/dx^n` / `d^n Bx/dx^n`, averaged over the region (see the
module docstring of
[tube_fitter.py](../../../xtrack/_temp/splineboris/tube_fitter.py) for the
`b_m`/`a_m` definitions). To turn that into `knl`/`ksl`, traced the exact
relation `xt.Multipole` itself uses internally, from
[track_magnet_kick.h::evaluate_field_from_strengths](../../../xtrack/beam_elements/elements_src/track_magnet_kick.h#L264):
evaluating that function's `By_T`/`Bx_T` output on-axis and differentiating
n times w.r.t. `x` gives

```
knl[n] = length / brho0 * mean(d^n By/dx^n)   # Bnorm
ksl[n] = length / brho0 * mean(d^n Bx/dx^n)   # Bskew
brho0  = p0c / (clight * q0)                  # [T*m], p0c in eV
```

with no extra sign flip needed -- `Bnorm`/`Bskew` are already stored in the
same convention `SplineBoris`'s own field evaluator uses directly, and this
matches the standard `knl[0] = By*L/brho0` dipole relation exactly (checked:
`p0c=1e9` eV gives `brho0=3.3356` T*m, the textbook `p[GeV/c]*3.3356`
number).

`Bs` (on-axis solenoid field) has no `Multipole` equivalent and is dropped;
a warning is printed via `[TubeFitter] to_multipole_line(): ...` if `Bs` was
actually significant enough to have been fit
(`component_to_fit[("Bs", 0)]`).

## What was added

`TubeFitter.to_multipole_line(multipole_order, p0c, q0=1.0)` in
[tube_fitter.py](../../../xtrack/_temp/splineboris/tube_fitter.py), right
after `to_line()`. Filters `df_fit_pars` to `param_index == 4` and
`field_component != "Bs"`, groups by region same as `to_line()`, and builds
one `xt.Multipole(knl=..., ksl=..., length=..., isthick=True)` per region
(orders beyond what was fit come out zero, same convention as `to_line()`'s
zero `Spline4`).

## A test-data gotcha worth remembering

First verification attempt used a synthetic field `By = B0 + G*x`,
`Bx = A0 + Askew1*x` (independent linear terms in each) and got `knl[1]` off
by exactly 2x from the naive expectation. **Not a code bug** -- that field
is not Maxwell-consistent: a real quadrupole-order transverse field must
pair `By ∝ x` with `Bx ∝ y` (normal quad) or `By ∝ y` with `Bx ∝ x` (skew
quad), because both come from the *same* shared potential coefficient
`Psi[1,1]` in the tube fit's row-`Bx`/row-`By` equations. Feeding in a field
where `Bx` has no matching `y`-dependence makes the least-squares fit split
that shared unknown roughly in half between the (mismatched) `Bx` and `By`
targets -- hence the 2x. Fixed by using the physically valid combination
`By = B0 + G*x + Gs*y`, `Bx = A0 + G*y + Gs*x`, which reproduced `G`/`Gs`
to `~1e-6` relative precision. (The standalone skew term `Askew1*x` alone
*was* fine in the first attempt, since `a_2`/`Bskew` at that order comes
from `Psi[2,0]`, which only ever appears in the row-`Bx` equation -- no
coupling to `By` to go wrong.)

## Tests added

[tests/test_splineboris.py](../../../tests/test_splineboris.py), right
after `test_splineboris_tubefitter_vs_fieldfitter_undulator`:

- `test_tubefitter_to_multipole_line_matches_mean_field` -- the
  Maxwell-consistent synthetic dipole+normal-quad+skew-quad field above,
  checking every region's `knl`/`ksl` against the analytic expectation
  (`rtol=1e-5`), including that the unfit 3rd order comes out exactly zero.
- `test_tubefitter_to_multipole_line_drops_bs_with_warning` -- confirms the
  warning fires (via `capsys`) when `Bs` was significant, and that `knl[0]`
  (the dipole) still comes through unaffected.

Both pass individually. A full-suite regression run (`XSUITE_ALLOW_NO_PREBUILT_KERNELS=1`)
was kicked off after adding them to confirm no interaction with the rest of
the file; see [tubefitter_canonical_and_upstream_merge.md](tubefitter_canonical_and_upstream_merge.md)
for the merge context these tests landed on top of.

## Not done / open

- Not yet committed (sits on top of the merge commit `4bd16586d` as
  uncommitted changes to `tube_fitter.py` and `test_splineboris.py`).
- Not wired into any example script.
- No tracking-level comparison test against `to_line()`/`SplineBoris`
  (e.g. total kick angle over a whole solenoid) -- only the per-region
  `knl`/`ksl` values themselves are checked against the analytic
  expectation for a synthetic field.

## Update (2026-08-25): bracketed with `MultipoleEdge` boundary kicks

`f4e25b0cc` ("Checking if adding edges makes the differnece") changed each
region's output from a single thick `xt.Multipole` to three elements:
`..._edge_entry` (thin `xt.MultipoleEdge`), the thick `xt.Multipole` itself
(now also `model='drift-kick-drift-exact'`, previously unset), and
`..._edge_exit`. Motivation per the module docstring: a bare thick-multipole
kick alone still discards more than "everything beyond a single field value
per region" -- it also has no boundary kick at all at a region's start/end,
where a spatially-resolved field (`SplineBoris`/`to_line()`) would give the
particle a kick from the field gradient turning on/off. The edge elements
close that gap, using the exact same `n>=1`-orders-only convention xtrack's
own `Quadrupole`/`Sextupole`/`Octupole` use internally to realize
`edge_entry_active`/`edge_exit_active`
(`slice_elements_edge.py::_parent_total_kn_ks_for_multipole_edge` -- traced
because `xt.Multipole` itself has no edge-related fields despite its
docstring mentioning them, so this was the element-agnostic way to attach
one). Edge strength is `kn_edge = knl/length`, `ks_edge = ksl/length` with
index 0 (dipole) zeroed before construction -- dipole fields don't
discontinuously turn on/off at a region boundary the way quadrupole-and-up
gradients do, so no entry/exit kick is wanted there.

**Consequence for any code that iterates `to_multipole_line()`'s
`element_names`:** the edge elements share the `'tubefitter'` name prefix
(so they still move together with their parent region under `shift_x`,
correctly) but have no `.length`/`.knl`/`.ksl`/`.get_field()` -- any
per-region loop that assumed one thick element per name (previously safe)
now needs an `isinstance(..., xt.MultipoleEdge)` exclusion first, or it will
crash (no `.length`) or silently index against the wrong table row
(`mult_table['s'][:-1]` positional slicing, previously fine when
`element_names` == regions 1:1, is now off by however many edge elements
precede the row of interest). Both call sites in
[sls_undulator_tune_displacement_007a.md](sls_undulator_tune_displacement_007a.md)
needed exactly this fix the same day -- see that note's "`MultipoleEdge`
thick-filtering fix" section for the concrete before/after.
[sls_undulator_tune_displacement_local_optics.md](sls_undulator_tune_displacement_local_optics.md)
(007b, new the same day) was written with the filter already in place from
the start.

Not yet reflected above in "The conversion"/"What was added" (those sections
still describe the pre-edge, single-`Multipole`-per-region behaviour) --
re-read `to_multipole_line()`'s current source rather than trusting this
note's older sections for the exact element layout.

## Update (2026-08-26): REVERTED -- broke corrector matching, and the two dedicated unit tests

`to_multipole_line()` is back to exactly its pre-`f4e25b0cc` form: a single
thick `xt.Multipole` per region, no `MultipoleEdge` elements, no
`model='drift-kick-drift-exact'`. Confirmed byte-identical to the
`1666b0eb4` version via `git diff 1666b0eb4 -- xtrack/_temp/splineboris/tube_fitter.py`
(empty).

**Why.** Running `007a_sls_undulator_tune_displacement.py`'s per-offset
corrector re-match (`opt_dx.step(2)` inside the `hor_off_list` scan loop, see
[sls_undulator_tune_displacement_007a.md](sls_undulator_tune_displacement_007a.md))
on the user's machine crashed with `numpy.linalg.LinAlgError: SVD did not
converge`, printing `Optimize - start penalty: nan` -- i.e. the MK-model
undulator line (built via this method) was already producing a NaN
orbit/penalty *before* the optimizer took a single step, for some offset in
the ±0.5 mm scan. The user identified the edges experiment as the likely
cause and asked to revert it; not root-caused further beyond that (e.g. no
isolated repro narrowing it to a specific `dx`, corrector, or the
`drift-kick-drift-exact` model switch vs. the edge kicks themselves) --
worth confirming if edges are attempted again.

**Independent corroborating evidence the edges change was never fully
integrated:** `tests/test_splineboris.py`'s
`test_tubefitter_to_multipole_line_matches_mean_field` and
`test_tubefitter_to_multipole_line_drops_bs_with_warning` both assert
`len(line.elements) == fitter.n_regions` and
`all(isinstance(el, xt.Multipole) for el in line.elements)` -- assertions
that the edge elements (3x elements per region, 2 of them `MultipoleEdge`)
would necessarily fail. `f4e25b0cc`'s own file list (`git show f4e25b0cc
--stat`) does not include `tests/test_splineboris.py`, confirming these
tests were never updated for the edge elements and would have been silently
broken (in a fresh env without stale `.pyc`/cached results) from that commit
onward. Both pass again post-revert
(`pytest tests/test_splineboris.py -k to_multipole_line`, 2 passed).

**Not reverted / still stale:** the `isinstance(..., xt.MultipoleEdge)`
filtering this change motivated in
[sls_undulator_tune_displacement_007a.md](sls_undulator_tune_displacement_007a.md)
(`mult_names`, and originally also `field_element_names_thick` -- since
removed for an unrelated reason, see that note) and in
[sls_undulator_tune_displacement_local_optics.md](sls_undulator_tune_displacement_local_optics.md)
(007b) is still present in both scripts. It's a harmless no-op now (nothing
will ever match `MultipoleEdge` again), not incorrect, just dead weight --
left in place since the user was asked and didn't request the cleanup.
Neither script's own `elements.py`/core-xtrack dependency needed touching
for this revert: `MultipoleEdge` itself predates this whole episode (only
git history: `7c67bd9c9`, the much earlier upstream module-flattening
merge) -- `f4e25b0cc` only ever *reused* an existing element type, so
nothing outside `tube_fitter.py` needed reverting.
