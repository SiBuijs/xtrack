# TubeFitter: Hermite export used the wrong region boundaries (found + fixed 2026-08-10)

> **HISTORICAL (checked 2026-10-14).** The Hermite export and region boundaries this note debugs are gone after the 2026-10-08 rewrite (`a8e5ecc17` and `0f77ee734`, after TubeFitter simplifications on 2026-10-06): `FieldFitter` and `SplineBorisSequence` were deleted, and `TubeFitter` now only produces on-axis multipoles at its frames, while the new `LongitudinalFitter` does the quartic-B-spline fit and the `to_line()`/`to_multipole_line()` export. See `00_overview.md`. The export now evaluates one global quartic B-spline per component on uniform nodes, so there are no per-region boundaries to get wrong.

**Status: FIXED.** This is a bigger, independent bug from
[hermite_to_polynomial_reconstruction_bug.md](hermite_to_polynomial_reconstruction_bug.md)
("Bug 3", the `* L` double-scaling fix from earlier the same day) -- found
while chasing the user's hypothesis that TubeFitter's remaining accuracy gap
vs `FieldFitter` (after that first fix, and after restoring `quadratic`/
`cubic` kernels, see
[tube_approach_kernel_simplification.md](tube_approach_kernel_simplification.md))
might be a raw-data sparsity issue.

## How this was found

The user's hypothesis was transverse (X/Y) sparsity: the undulator raw data
turned out to have only a 3x3 = 9-point transverse grid (`X,Y ∈ {-1,0,1}`
raw units) but 2201 finely-spaced Z points -- so the sparsity, if any, was
transverse, not longitudinal. Testing this directly:

- Reducing `deg` (fewer transverse multipole unknowns, more overdetermined
  per z-slice) did **not** improve on-axis Bskew/Bnorm accuracy at all
  (4.53% at `deg=0/1` vs 4.37% at `deg=2` -- deg=2 was already the best).
  Ruled out "too few transverse points for the multipole order" as the
  driver.
- Increasing `deg` past 2 (`deg=3,4`, i.e. `M=4,5`, 14-20 unknowns vs 9
  points) added nothing -- the higher-order multipole terms have magnitude
  ~1e-11 to 1e-16 (floating-point noise), i.e. there is genuinely no real
  field content above octupole order in this dataset. Not a truncation
  problem either.
- A fully **local, per-z-slice fit** (solve the 9-point transverse system
  independently at each z, with no B-spline/frame machinery at all)
  reproduces raw Bx/By to **0.016%/0.012%** -- matching TubeFitter's own
  reported internal `tube_fit_residual_rms` (0.02%/0.01%) almost exactly.
  **The transverse multipole fit itself is essentially exact.** Sparsity
  was a dead end.

That left a real puzzle: if the internal fit is accurate to ~0.01-0.02%,
why did the *downstream* SplineBoris reconstruction (what
[003c](../003c_fieldfitter_vs_tubefitter.py) measures, and the number that
matters for tracking) show 4.3-5.4% error, even **on-axis**, where der=0
should just be reading the fitted curve back out directly? Comparing stages
directly pinned it down:

| stage | Bx rel RMS vs raw |
|---|---|
| `df_on_axis_fit` (TubeFitter's own on-axis multipole eval) vs raw | 0.0134% |
| `SplineBorisSequence.get_field` (exported Hermite → Table-1 eval) vs raw | 4.3702% |
| `SplineBorisSequence.get_field` vs TubeFitter's own `df_on_axis_fit` (should be ~identical) | 4.3705% |

So the corruption happens strictly **between** the fitted on-axis curve and
its Hermite export/reconstruction -- not in the tube fit itself.

## Root cause

`_convert_to_hermite` computes each region's Hermite parameters against the
**true, continuous frame boundaries**:

```python
for i_reg in range(self.n_regions):
    s_left = float(self.frames[i_reg])
    s_right = float(self.frames[i_reg + 1])
    ...
    self._hermite[...] = _hermite_from_bspline(f_norm, s_left, s_right)
```

But `_populate_df_fit_pars` exported **different** boundaries -- the raw
data's nearest z-sample, found via `_frame_indices` (nearest-index snapping):

```python
idx_start = int(idx_extrema[i_reg])
idx_end = int(idx_extrema[i_reg + 1])
s_start = float(self.s_full[idx_start])   # WRONG -- snapped, not the true frame
s_end = float(self.s_full[idx_end])       # WRONG -- snapped, not the true frame
```

Downstream, `SplineBorisSequence._split_hermite_to_canonical`
([splineboris_sequence.py:98](../../../xtrack/_temp/splineboris/splineboris_sequence.py#L98))
reconstructs the polynomial via
`hermite_to_polynomial(piece_s_start, piece_s_end, hermite_vals)` using
*exactly* these exported (wrong) `s_start`/`s_end` as `L`. Since
`hermite_to_polynomial` scales the derivative terms by `L`
(`L * c2 * b2 + L * c4 * b4`, see
[hermite_to_polynomial_reconstruction_bug.md](hermite_to_polynomial_reconstruction_bug.md)
for the basis), using the *wrong* `L` to reconstruct coefficients that were
*derived* against a different `L` directly corrupts the polynomial -- the
same category of bug as Bug 3, but at the export boundary rather than in the
coefficient computation itself.

**Why this was invisible for simple cases (e.g. the constant-dipole check,
`solenoid_tubefitter_fit_pars_df`) but huge here:** the size of the
corruption scales with (a) how large the derivative terms are and (b) how
far the snapped boundary drifts from the true frame boundary. It's
negligible when `n_frames` is small relative to the raw z-sample count
(snapping error << frame spacing). It becomes serious specifically when
`n_frames` is *close to* the raw sample count -- exactly the regime this
session's `n_frames=2200` (chosen as the "best found" sweet spot against
2201 raw samples, see
[tubefitter_nframes_nonmonotonic.md](tubefitter_nframes_nonmonotonic.md))
sits in.

**Why it forms a smooth bump peaking in the domain centre, not a random
scatter:** `n_frames=2200` and the raw sample count `2201` are consecutive
integers -- the frame grid and the raw-sample grid drift in and out of
registration like a beat pattern over the domain (near-perfect alignment at
both edges, maximum drift at the centre), which is exactly the profile
measured:

```
idx [   0, 200): RMS err = 0.0053   idx [1000,1200): RMS err = 0.0518 (peak, domain centre)   idx [2000,2200): RMS err = 0.0049
```

This single mechanism plausibly explains **both** open problems from
[tubefitter_nframes_nonmonotonic.md](tubefitter_nframes_nonmonotonic.md):
the general downstream-vs-internal-residual gap, and the previously
unexplained "middle bad patch" (`n_frames≈1200-1500` locally worse than its
neighbours) -- both are beat-pattern artifacts of frame-count vs
raw-sample-count misalignment. Confirmed below.

## Fix applied

[tube_fitter.py:801-814](../../../xtrack/_temp/splineboris/tube_fitter.py#L801)
(`_populate_df_fit_pars`): export `s_start`/`s_end` as the true
`self.frames[i_reg]` / `self.frames[i_reg + 1]` instead of the
nearest-raw-sample-snapped `self.s_full[idx_start/idx_end]`.
`idx_start`/`idx_end` themselves are left unchanged (they only drive
`SplineBorisSequence`'s tracking step count via `n_steps = (idx_end -
idx_start) * steps_per_point`, an unrelated, harmless use).

## Verified impact

Isolated patch test (only `s_start`/`s_end`, everything else unchanged):
on-axis Bnorm rel RMS **5.310% → 0.0106%** -- matches the internal fit
residual almost exactly.

Full `003c_fieldfitter_vs_tubefitter.py` re-run (`n_frames=2200`, `deg=2`,
`field_tol=1e-3`, full grid vs raw data):

| kernel | Bskew (before → after) | Bnorm (before → after) | Bs (before → after) |
|---|---|---|---|
| tent | 4.456% → **0.677%** | 5.440% → **1.407%** | 10.647% → 4.574% |
| quadratic | 4.383% → **0.940%** | 5.270% → **0.121%** | 6.322% → 0.946% |
| cubic | 4.387% → **0.931%** | 5.319% → **0.030%** | 6.070% → 0.414% |

(FieldFitter, unaffected by this bug, for reference: Bskew 1.303%, Bnorm
2.037%, Bs 4.273%.) **TubeFitter now beats FieldFitter** on Bskew/Bnorm at
every kernel, and clearly beats it on Bnorm with quadratic/cubic. This also
means the earlier same-day kernel comparison in
[tube_approach_kernel_simplification.md](tube_approach_kernel_simplification.md)
was measured *before* this fix and understated the benefit of
quadratic/cubic kernels for Bskew/Bnorm -- with this bug gone, the
approximation-order argument for restoring the kernels shows up cleanly
(Bnorm: tent 1.407% → quadratic 0.121% → cubic 0.030%), unlike the earlier
near-flat comparison.

**`n_frames` re-scan after the fix** (tent, `deg=2`, `field_tol=1e-3`, full
grid): the previously-documented middle "bad patch" is **gone** --
Bskew/Bnorm now improve smoothly and monotonically with `n_frames`, no
local degradation at 1200-1500:

```
n_frames    Bskew    Bnorm      Bs
     100   58.644%  92.470%  89.750%
     600    1.815%   2.127%  18.650%
    1000    1.187%   1.505%  11.303%
    1200    1.137%   1.454%   9.126%   <- was the "bad patch"; now just another point on a smooth curve
    1500    1.095%   1.411%   7.769%
    1800    1.090%   1.407%   6.739%
    2000    1.090%   1.407%   5.924%
    2200    1.087%   1.407%   4.583%
    2800    1.090%   1.407%  12.045%
    4000    1.090%   1.407% 747.836%   <- Bs still blows up (separate mechanism, see below)
    4402    1.090%   1.407% 849.432%
```

**Bskew/Bnorm's high-`n_frames` blowup is also gone** -- they now plateau
flat (~1.09%/1.41%) all the way to the DOF ceiling instead of blowing up to
150-190% as before the fix.

## Still open: `Bs` blows up at high `n_frames` -- a different mechanism

`Bs` (`to_fit=False` here -- its own magnitude is below `field_tol`, so its
direct Hermite fit is exported as all-zero) still degrades badly at high
`n_frames`. This is **not** the boundary-export bug above (confirmed: this
fix touches all three field components identically). Direct check of
`SplineBorisSequence.get_field` shows Bs is *exactly* zero on-axis (as
expected, `to_fit=False`) but **nonzero off-axis**:

```
(x,y,z)=(0.0,0.0,0.0):   Bs = 0.0
(x,y,z)=(0.001,0.0,0.0): Bs = -0.0199
```

So off-axis `Bs` is entirely reconstructed by the Table-1/Schueren
evaluator from the **z-derivative** content of the Bskew/Bnorm multipole
fit (the div(B)=0 relation tying `Bs'(z)` to transverse curvature -- see
the "Conventions" docstring at the top of
[tube_fitter.py](../../../xtrack/_temp/splineboris/tube_fitter.py)), not
from its own (unfit) curve at all. Meanwhile Bskew/Bnorm **values** (der=0)
stay flat/accurate all the way to the DOF ceiling after the fix above. The
likely mechanism: differentiating an increasingly-overfit tent-spline
interpolation of noisy raw data amplifies noise sharply as `n_frames`
approaches the ~2201-raw-sample ceiling -- values are stable under
overfitting, derivatives are not. A reasonable next step would
be checking whether `quadratic`/`cubic` kernels (smoother, well-defined
derivatives everywhere vs `tent`'s one-sided nudge at kinks) soften this,
same as the still-open question in
[tubefitter_nframes_nonmonotonic.md](tubefitter_nframes_nonmonotonic.md).

## RESOLVED (2026-08-26): confirmed undulator/near-zero-`Bs`-specific -- no risk for solenoid-dominated field maps

The user asked whether this blowup would also afflict a solenoid field map,
recalling (correctly) that the undulator's on-axis `Bs` is essentially zero
here (`to_fit=False`), and suspecting a solenoid -- where `Bs` is the large,
dominant, genuinely-fit quantity -- would not hit the same failure mode.

Verified directly: reran the identical `n_frames` scan (full-grid relative
RMS, `SplineBorisSequence.get_field` vs measurement -- same method as the
table above) on the analytic paraxial solenoid used by
`tests/test_splineboris.py`'s `solenoid_tubefitter_fit_pars_df` fixture
(`SolenoidField(L=4.0, a=0.3, B0=1.5, z0=20.0)`, `SOLENOID_INTERVAL=30.0`,
5001 unique z samples -> DOF ceiling `n_frames=5000`):

```
n_frames    Bskew        Bnorm        Bs
     100   16.156%      57.108%    0.875%
    1000    0.081%       5.662%    0.0085%
    2200    0.015%       2.585%    0.0016%
    4000    0.00019%     1.528%    0.00010%
    4900    0.00013%     1.154%    0.00010%
    5000    0.00013%     0.697%    0.00010%   <- DOF ceiling
```

`Bs` (and `Bskew`) decrease smoothly and monotonically all the way to the
DOF ceiling, bottoming out at floating-point-floor-level relative error --
**no blowup, no non-monotonicity**, nothing resembling the undulator's climb
to 750-850% near its ceiling. (`Bnorm` plateaus at a nonzero ~0.7-3% floor
rather than reaching machine precision, but that's a smooth, well-behaved
floor from the transverse Laplace/trace reconstruction in `x`/`y` -- an
unrelated, non-pathological limitation, not a blowup.)

**Why this confirms the mechanism rather than contradicting it:** the fit's
own diagnostic print shows why the two datasets diverge so sharply. For
this solenoid, `Bs` (der=0) is `to_fit=True` with `rel_max=1.483` (a large,
real, directly-fit signal), while `Bskew`/`Bnorm` (der=0) are the near-zero,
unfit components -- the exact mirror image of the undulator, where `Bs` was
the near-zero, unfit, purely-derivative-reconstructed component. The
pathological mechanism (differentiating an increasingly overfit
tent-spline-in-z to regenerate a component that has *no other* signal of
its own) requires that component to be small/unfit and reconstructed
entirely from another component's z-derivative content, via the
Maxwell/Schueren machinery described in
[tube_schueren_integration.md](tube_schueren_integration.md). A solenoid's
`Bs` never enters that regime because it isn't reconstructed that way at
all -- it's fit directly from abundant on-axis data.

**Practical conclusion:** for solenoid-dominated field maps (e.g. detector-
solenoid studies), pushing `n_frames` high to chase a tighter on-axis
residual is safe -- this failure mode does not apply there. It remains a
live risk specifically for datasets where the longitudinal field is the
small, derivative-only-reconstructed component (undulators, wigglers, or
any field map where `Bs` sits below `field_tol` on-axis). The underlying
code mechanism itself was not changed/fixed by this investigation -- only
scoped: still worth guarding against (e.g. a downstream 003c-style sanity
check) whenever `Bs` is the small/unfit component, but no longer an open
question for the FCC-ee solenoid work this user cares about.

Verification script (not committed, one-off check): `solenoid_nframes_scan.py`
in the session scratchpad, adapted directly from
`solenoid_tubefitter_fit_pars_df` (raw data/grid) and `003c`'s
`spline_boris_residual` (the downstream-vs-raw comparison method).
