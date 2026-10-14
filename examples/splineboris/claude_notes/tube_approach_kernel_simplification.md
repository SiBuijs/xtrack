# TubeFitter: dropping higher-order kernels, tent-only (2026-08-10, kernels restored later same day)

> **HISTORICAL (checked 2026-10-14).** After the 2026-10-08 rewrite (`a8e5ecc17` and `0f77ee734`, after TubeFitter simplifications on 2026-10-06): `FieldFitter` and `SplineBorisSequence` were deleted, and `TubeFitter` now only produces on-axis multipoles at its frames, while the new `LongitudinalFitter` does the quartic-B-spline fit and the `to_line()`/`to_multipole_line()` export. See `00_overview.md`. `TubeFitter` is **tent-only** (degree-1 B-spline in z), so there are no `kernel`/`basis_order` options. Note that the kernels were dropped for good well before that rewrite, in `1666b0eb4` (2026-08-14: "TubeFitter now only uses the triangle kernel", for solve speed, which made `residual_tol` feasible on the 1.8 GB map). So the 'restored later same day' in the title was itself undone four days later. Its block-tridiagonal normal equations are assembled and solved directly (`_accumulate_block_tridiagonal`, `_solve_block_tridiagonal`). The scripts named below (`003b_tube_fitter_basic_usage.py`, `003c_fieldfitter_vs_tubefitter.py`, `003e_*`, `011_*`) no longer exist.

Session log of simplifying `TubeFitter` down to the `tent` (degree-1) B-spline
kernel only. Companion notes: [tube_approach_paper.md](tube_approach_paper.md),
[fieldfitter_vs_tubefitter_comparison.md](fieldfitter_vs_tubefitter_comparison.md).
See also [hermite_to_polynomial_reconstruction_bug.md](hermite_to_polynomial_reconstruction_bug.md)
for the (much bigger) bug this work led to finding.

## Why

The user found the `quadratic`/`cubic` kernel machinery harder to reason
about than they wanted to rely on without fully understanding it, and asked
to strip it out of the working branch (`tube_approach`) entirely for now,
defaulting hard to `tent`. The removed work (kernel-comparison plots,
`003e_tube_fitter_kernel_comparison.py`, the `kernel=` parameter) is not
lost -- it was parked on branch **`tube_approach_kernels`** to revisit later.

## What changed in `tube_fitter.py`

- Removed `KERNEL_TO_DEGREE`, the `kernel` constructor parameter, and the
  "Kernel choice" docstring section entirely.
- `self.basis_order` (previously `KERNEL_TO_DEGREE[kernel]`) was first
  hardcoded to a `BASIS_ORDER = 1` module constant, then -- at the user's
  request to keep things as clean as possible -- **removed entirely**,
  inlining the literal `1` at every one of its ~9 use sites (`k = 1` in each
  method that builds a `BSpline`).
- This let a redundant check collapse: `n_frames >= basis_order + 1` was
  identical to the already-present `n_frames >= 2` check once `basis_order`
  was pinned to `1`, so it was deleted.
- `_setup_frames`'s general clamped-knot construction (`[tube_fitter.py:453](../../../xtrack/_temp/splineboris/tube_fitter.py#L453)`)
  collapsed from the old branching/`interior = linspace(...)` logic to a
  single line:
  ```python
  self.knots = np.r_[z_min, self.frames, z_max]
  ```
  Verified bit-identical (`np.array_equal`) to the old general `k`-parametrized
  formula for `k=1`, across a range of `n_frames` -- correct because for tent,
  knots and frames coincide (interior knots = interior frames exactly), so the
  clamped vector is just each endpoint duplicated around the frames array.
- `_dof_ceiling` ([tube_fitter.py:292](../../../xtrack/_temp/splineboris/tube_fitter.py#L292))
  was separately changed (during this same session, prompted by a "what if the
  grid isn't rectangular" question) from `n_eq = 2*n_x*n_y*n_z` to
  `n_eq = 2*len(self.df_raw_data)` -- the old formula silently assumed a full
  Cartesian X/Y/Z grid; the new one uses the actual row count, correct even
  for a ragged grid. No behavior change on today's rectangular datasets
  (verified: same `n_frames=200` default result before/after).
- Removed the dead `evaluate_transverse_field` method -- unused anywhere in
  the repo; it was added for a holdout/cross-validation diagnostic in 003c
  that the user had already removed (see
  [fieldfitter_vs_tubefitter_comparison.md](fieldfitter_vs_tubefitter_comparison.md)'s
  "New TubeFitter capability" section for that history).
- `_trial_relative_residual`, `_constant_dipole_sign_check`, and the
  `__main__` self-test block no longer pass/accept `kernel=`.

## Example scripts

- `003b_tube_fitter_basic_usage.py`: dropped `kernel="quadratic"`; also
  dropped a kernel-comparison demo plot (tent/quadratic/cubic basis functions
  with dashed frame markers) that had been added earlier in the session --
  that's the piece parked on `tube_approach_kernels`.
- `003d_tube_fitter_nframes_scan.py`: dropped the `kernel = "cubic"`
  variable/argument.
- **Deleted** `003e_tube_fitter_kernel_comparison.py` entirely -- it existed
  solely to compare convergence rates across the three kernels, so has no
  purpose with only `tent` left. Fully recoverable from git history / the
  `tube_approach_kernels` branch.

## Regression: switching the default from `cubic` to `tent` broke a test

Discovered via `tests/test_splineboris.py`, not immediately obvious from the
refactor itself (see the "why this wasn't caught earlier" note below).

**`solenoid_tubefitter_fit_pars_df`** (a `pytest` fixture,
[test_splineboris.py:226](../../../tests/test_splineboris.py#L226)) constructs
its `TubeFitter` without ever passing `kernel=`, so it silently went from
implicitly-cubic to implicitly-tent. Its `n_frames=500` was tuned for cubic's
accuracy; tent needs more frames for the same fit quality. Symptom: the
fixture's own `check_trace_consistency()` gate (comparing `Psi[0,2]` against
the `-Psi[2,0]+Bs'/2` Maxwell prediction) failed at 2.92% against a 1%
threshold.

Measured `n_frames` vs `relative_rms` for this exact solenoid dataset with tent:

| n_frames | relative_rms |
|---|---|
| 500 (old) | 2.92% (fails) |
| 1000 | 1.48% |
| **2000** | **0.75% (fixed, now used)** |
| 4000 | 0.375% |
| 8000 | 21.4% (breaks -- exceeds the ~5001 unique z data points, ill-posed) |

Fix applied: bumped that fixture's `n_frames` to `2000`
([test_splineboris.py:268](../../../tests/test_splineboris.py#L268)). Also
replaced `fitter.basis_order + 1` with the literal `2` in
`test_tubefitter_residual_tol_search_reaches_target`
([test_splineboris.py:851](../../../tests/test_splineboris.py#L851)), since
`basis_order` no longer exists as an attribute.

**Verified via direct empirical check** (not just argument) that this
regression predates all of today's `self.basis_order`/knots-formula cleanup:
reconstructed the exact pre-cleanup `_setup_frames` logic via monkeypatching
and reran the fixture at `n_frames=500` -- got `relative_rms = 2.920%`,
matching the current code to 3 decimal places. So the refactor itself is
behavior-preserving; the regression traces back to the kernel-default switch
alone.

## `test_splineboris_tubefitter_solenoid_vs_variable_solenoid`: was failing, now FIXED

Even after fixing the fixture above (so it *builds* successfully), the actual
downstream test -- comparing `SplineBoris` tracking (built from this
`TubeFitter` fit) against `VariableSolenoid` tracking, `atol = 3%` of range --
still failed. Critically, **more frames made it worse, not better**:
`n_frames=2000` mismatched 9.7% of checked points; `n_frames=4000` mismatched
61.5%. This ruled out "just needs more resolution" as the explanation and was
left failing/parked at the user's request, pending investigation.

**Root cause found and fixed:** see
[hermite_to_polynomial_reconstruction_bug.md](hermite_to_polynomial_reconstruction_bug.md)
("Bug 3"). `TubeFitter._hermite_from_bspline` was pre-scaling the Hermite
`der_start`/`der_end` boundary values by the region length `L` before storing
them, but the shared `SplineBoris` field evaluator (`hermite_to_polynomial`)
already expects raw, unscaled derivatives -- the convention `FieldFitter` has
always used. The resulting double-scaling error grows with the per-region
derivative-to-length ratio, exactly matching the observed "more frames ->
worse" symptom (shrinking `L` at roughly fixed physical derivative). Fixed by
removing the spurious `* L` in
[tube_fitter.py:100,102](../../../xtrack/_temp/splineboris/tube_fitter.py#L100).
`tests/test_splineboris.py` now passes 46/49 (was 45/49); this test is one of
the ones that flipped to passing. The other 3 failures are unrelated, see
below.

## Test suite status

`pytest tests/test_splineboris.py`, after the `_hermite_from_bspline` fix:
**46 passed, 3 failing** -- all 3 pre-existing and unrelated:
`PrebuiltKernelNotFoundError` (Xsuite *compiled tracking kernel* cache
version mismatch -- `xtrack==0.109.1` cached vs `0.101.4` installed -- a
completely different "kernel" than the B-spline kernels discussed here).

(Earlier in the session, before the fix: 45 passed, 4 failing -- the same 3
plus the solenoid-vs-`VariableSolenoid` test, left failing on purpose
pending investigation.)

## Update: kernels restored (2026-08-10, same day)

After the `_hermite_from_bspline` fix above (which independently fixed the
solenoid-vs-`VariableSolenoid` test -- see
[hermite_to_polynomial_reconstruction_bug.md](hermite_to_polynomial_reconstruction_bug.md)),
and after finding the `n_frames` non-monotonicity problem (see
[tubefitter_nframes_nonmonotonic.md](tubefitter_nframes_nonmonotonic.md)),
the user asked whether restoring `quadratic`/`cubic` kernels would help, then
asked to restore them regardless (no longer blocked on "harder to reason
about" -- the reasoning that motivated stripping them, since the
`_derivative_into_region` fix means today's boundary-derivative computation
is simpler/more principled than what existed when they were removed).

**What was restored, from branch `tube_approach_kernels`:** `KERNEL_TO_DEGREE`,
the `kernel` constructor parameter (default kept as `"tent"`, not the
branch's `"cubic"`, to preserve existing default behaviour/tests), all
`k = 1` -> `k = self.basis_order` use sites, and the general clamped-knot
construction in `_setup_frames` (verified in
[tube_fitter.py:100](../../../xtrack/_temp/splineboris/tube_fitter.py#L100)-era
notes above to reduce to the tent-only formula for `k=1`).

**What was deliberately NOT restored:** the branch's `_safe_derivative_at`
(averaging left/right derivative limits, then re-scaling by `L`) -- that was
the *pre-fix*, buggy convention. The current `_derivative_into_region`
(one-sided nudge into the region, no rescaling) is correct for `tent`'s
kinks and equally correct for `quadratic`/`cubic` (their derivative is
continuous, so one-sided vs averaged give the same value there). Kept as-is.

**Measured effect** (`003c_fieldfitter_vs_tubefitter.py`, undulator map,
`n_frames=2200`, `deg=2`, `field_tol=1e-3`, full grid vs raw data):

| kernel | Bskew | Bnorm | Bs |
|---|---|---|---|
| tent | 4.456% | 5.440% | 10.647% |
| quadratic | 4.383% | 5.270% | 6.322% |
| cubic | 4.387% | 5.319% | 6.070% |

`Bskew`/`Bnorm` (the on-axis-derived, directly-exported components) barely
move -- under 0.2 percentage points. `Bs` (fit independently via `_fit_bs`,
a plain 1D B-spline through on-axis data -- same kernel, same `n_frames`)
drops by ~40% relative going from tent to quadratic/cubic.

**Superseded -- this comparison was measured before a much bigger,
independent bug was found and fixed the same day.** See
[tube_hermite_export_boundary_bug.md](tube_hermite_export_boundary_bug.md):
`TubeFitter` was exporting Hermite region boundaries snapped to the nearest
raw z-sample instead of the true frame boundaries, which happened to
corrupt the reconstruction badly right around `n_frames=2200` (next to this
dataset's 2201 raw samples). After that fix, the kernel-order comparison at
the same settings looks completely different -- the approximation-order
argument now shows up cleanly for Bskew/Bnorm too:

| kernel | Bskew | Bnorm | Bs |
|---|---|---|---|
| tent | 0.677% | 1.407% | 4.574% |
| quadratic | 0.940% | 0.121% | 0.946% |
| cubic | 0.931% | 0.030% | 0.414% |

(Bskew is a mild exception -- tent is marginally *better* there than
quadratic/cubic; not yet investigated why.) `n_frames` non-monotonicity was
also retested after the fix -- see
[tubefitter_nframes_nonmonotonic.md](tubefitter_nframes_nonmonotonic.md)'s
"root cause found" update: the middle bad patch and the Bskew/Bnorm
high-`n_frames` blowup are gone; `Bs`'s blowup remains, for a different,
still-open reason.

## Update (2026-08-14): kernel removed again, this time for good -- `TubeFitter` is tent-only

Commit `1666b0eb4` ("TubeFitter now only uses the triangle kernel...")
stripped `KERNEL_TO_DEGREE` and the `kernel` constructor parameter back out
entirely -- **not** a reversion of the restoration above, but a deliberate,
separate decision: quadratic/cubic kernels made the global sparse solve too
slow to be practical for large real field maps (the commit message cites
`simona_field_map`, ~1.8 GB), which mattered specifically because it made
`residual_tol`'s repeated-refit search (see
[fieldfitter_vs_tubefitter_comparison.md](fieldfitter_vs_tubefitter_comparison.md)'s
"n_frames sizing" section) infeasible at that scale. `011_solenoid_kernel_comparison.py`
was deleted in the same commit.

As of this update, `TubeFitter.__init__` has **no `kernel` parameter at
all** -- the longitudinal basis is unconditionally the tent (degree-1,
2-frame-wide, C^0) B-spline, documented directly in the class docstring.
Confirmed by reading `xtrack/_temp/splineboris/tube_fitter.py` directly
(`grep -n "kernel\|KERNEL_TO_DEGREE\|basis_order"` turns up nothing).
Nothing to restore from a parked branch this time -- if quadratic/cubic are
wanted again, they'd need to be re-derived, and the solve-speed cost that
motivated removing them would need addressing first (e.g. only paying for a
finer kernel on the final fit, not every trial inside a `residual_tol`
search).
