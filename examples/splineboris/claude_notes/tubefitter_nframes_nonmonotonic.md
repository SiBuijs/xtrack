# `TubeFitter`: downstream accuracy is non-monotonic in `n_frames`, and `residual_tol` can't see it (found 2026-08-10, middle "bad patch" + Bskew/Bnorm blowup FIXED same day, Bs blowup still open)

> **MEASURED ON THE OLD PIPELINE, NOT RE-CHECKED (2026-10-14).** Everything here predates the 2026-10-08 rewrite (`a8e5ecc17` and `0f77ee734`, after TubeFitter simplifications on 2026-10-06): `FieldFitter` and `SplineBorisSequence` were deleted, and `TubeFitter` now only produces on-axis multipoles at its frames, while the new `LongitudinalFitter` does the quartic-B-spline fit and the `to_line()`/`to_multipole_line()` export. See `00_overview.md`. The non-monotonicity was measured through the old Hermite/region export, which no longer exists, so it is unknown whether it survives. `residual_tol` and its `n_frames` search are still in `TubeFitter`. The current tool for downstream accuracy vs `n_frames` (and vs element length) is `examples/splineboris/003c_accuracy_scan.py`, whose docstring models the stage-1 error as a smoothing bias ~(k Delta_frame)^2/12 that shrinks with more frames. Re-run it before relying on the 'open, significant' status below.

**Status: open, significant.** Found while checking (per the user's request) what
part of the remaining `TubeFitter`-vs-`FieldFitter` gap -- after
[hermite_to_polynomial_reconstruction_bug.md](hermite_to_polynomial_reconstruction_bug.md)'s
fix -- was attributable to `field_tol`/significance-zeroing. It isn't; instead
this surfaced a bigger problem with how `n_frames` is chosen.

## The core finding

`003c`'s downstream metric (`SplineBorisSequence.get_field` vs raw data --
the number that actually matters for tracking, see
[hermite_to_polynomial_reconstruction_bug.md](hermite_to_polynomial_reconstruction_bug.md)'s
"How this was found") is **not monotonic in `n_frames`**, and gets
**catastrophically worse, not better, as `n_frames` approaches the DOF
ceiling** -- even though `TubeFitter`'s own internal diagnostic
(`tube_fit_residual_rms`, the thing `residual_tol` optimizes) keeps
improving the whole way there.

Full-grid relative RMS, undulator map, `deg=2`, `field_tol=1e-3`:

```
n_frames    Bskew      Bnorm       Bs
     100    58.7%      92.5%      89.9%
     600     4.5%       5.4%      27.3%   <- local optimum
    1000     4.7%       5.7%      26.3%
    1200    11.0%      14.4%      62.2%   <- local bad patch
    1500    12.6%      16.5%      75.0%
    1800     9.3%      12.0%      55.9%
    2000     6.6%       8.3%      37.5%
    2200     4.5%       5.4%      10.6%   <- best found, ~1 frame per raw z-sample
    2800     9.5%      12.4%      51.4%
    4000    94.1%     117.1%     321.3%
 4402 (DOF ceiling)  153.4%     188.5%   515.1%   <- catastrophic
```

Meanwhile `[TubeFitter] ... tube fit residual (der=0)` -- the metric
`residual_tol`'s bisection search (`_search_n_frames`,
[tube_fitter.py:328](../../../xtrack/_temp/splineboris/tube_fitter.py#L328))
optimizes -- keeps improving monotonically all the way to the ceiling (0.10%
at `n_frames≈1510`, 0.02%/0.01% at the ceiling). **In the high-`n_frames`
regime the internal residual is not just a different/easier metric than the
one that matters (as first thought when this investigation started) -- it is
actively anti-correlated with it.** A tight `residual_tol` target will drive
the search straight into the worst part of this curve, and the search has no
way to detect that it's doing so.

## Confirmed mechanisms (two of the three regimes)

**Low `n_frames` (~100): plain underfitting.** The undulator map has a real,
measurable period -- 36 raw z-samples (0.036 m) per cycle, found via
zero-crossing counting on the on-axis `Bnorm` curve. `n_frames=100` gives
only ~1.6 piecewise-linear (tent) segments per period, nowhere near enough to
represent a periodic field. Ordinary resolution/Nyquist failure, not
surprising.

**High `n_frames` (2200 → 4402): the fit runs out of independent data per
frame.** The raw data has exactly **2201 unique, uniformly-spaced Z points**
(confirmed directly: `df['Z'].unique()` size 2201, `diff` constant at 1 raw
unit). `n_frames=2200` -- essentially one frame per raw z-sample -- is right
where the high-`n_frames` regime performs best; every frame beyond that has
to be constrained mostly through the tent basis's minimal (2-frame-wide)
support coupling rather than direct data, and the downstream error climbs
smoothly and dramatically from there to the ceiling. Directly explains why
"more frames" made things worse both here and in the earlier, unrelated
`solenoid_tubefitter_fit_pars_df` fixture history noted in
[tube_approach_kernel_simplification.md](tube_approach_kernel_simplification.md)
(`n_frames=8000` "breaks -- exceeds the ~5001 unique z data points,
ill-posed" -- same mechanism, different dataset).

## Ruled out, with direct evidence, as the driver of the third regime

The middle non-monotonic patch (`n_frames≈1200-1500` distinctly worse than
neighbours at ~600, ~1000, ~2200) is real, reproducible, and **not**
explained by any of:

- **`field_tol`/significance-zeroing.** Rerunning with `field_tol=1e-12`
  (nothing gets zeroed as insignificant) changes every number above by
  <0.01 percentage points, at both `n_frames=2200` and `n_frames=4402`.
  Ruled out cleanly.
- **`Bskew_der2`/`Bnorm_der2` magnitude.** Forcing these off entirely
  (`field_tol=2.5e-3`, above their `rel_max≈1.7e-3`) likewise changes nothing
  meaningfully, at every `n_frames` tested (1000/1200/1500/2200/4402).
  Ruled out.
- **`Bnorm der=0`'s own slope (`der_start`/`der_end`) quality**, checked
  directly against a raw central-difference estimate of the true on-axis
  slope at each frame boundary: RMS slope error is 20.4 (n=1000) → 17.4
  (n=1200) → 10.5 (n=2200) T/m -- monotonically *improving*, the opposite of
  what would explain n=1200 being worse than n=1000 downstream. This was the
  leading hypothesis and it's wrong.

**What is established:** the error at the bad patch is concentrated
*off-axis*, not on-axis. On-axis relative error is nearly flat between
n=1000 and n=1200 (`Bnorm` 4.50% → 4.73%), while full-grid error nearly
triples (5.7% → 14.4% `Bnorm`, 26% → 62% `Bs`). So the mechanism is
specifically in the Maxwell/Table-1-Schueren off-axis reconstruction step
(which regenerates off-axis structure from the on-axis polynomials alone --
see the "Root cause, corrected" section of
[hermite_to_polynomial_reconstruction_bug.md](hermite_to_polynomial_reconstruction_bug.md)
for how that evaluator works), not in the on-axis fit quality itself. Which
specific ingredient of that reconstruction is responsible is **not yet
found** -- next natural probe would be inspecting the off-diagonal `Psi[p,q](z)`
curves directly (e.g. `Psi[1,1]`, `Psi[2,1]`), since those feed the off-axis
reconstruction but have no raw on-axis ground truth to check derivative
quality against the way `Bnorm`/`Bskew` der=0 do.

## Practical implication

`residual_tol` (and therefore `_search_n_frames`) should not currently be
trusted to pick a good `n_frames` without a downstream (003c-style) sanity
check -- it can silently walk into the high-`n_frames` blowup regime, and its
own diagnostic will report record-low residual while doing so. No fix
applied yet.

## Update: kernels restored, effect on this problem not yet retested (superseded below)

Later the same day, `quadratic`/`cubic` kernels were restored (previously
stripped to `tent`-only, parked on branch `tube_approach_kernels`) -- see
[tube_approach_kernel_simplification.md](tube_approach_kernel_simplification.md)'s
"Update: kernels restored" section for what changed and the measured
`n_frames=2200` comparison. Short version: `Bs` improved substantially
(~40% relative) with quadratic/cubic over tent; `Bskew`/`Bnorm` barely
moved at that particular `n_frames`.

## Update: root cause found -- the "bad patch" and the Bskew/Bnorm blowup were the SAME bug, now fixed

Chasing a follow-up hypothesis from the user (raw-data sparsity) about why
kernel order barely moved Bskew/Bnorm led to finding a real, independent
bug: `_populate_df_fit_pars` exported Hermite region boundaries
(`s_start`/`s_end`) snapped to the nearest **raw z-sample**, not the true
continuous **frame** boundaries the Hermite coefficients were actually
derived against. See
[tube_hermite_export_boundary_bug.md](tube_hermite_export_boundary_bug.md)
for the full root-cause writeup, fix, and verification. Short version:
`n_frames=2200` sits right next to the raw sample count (2201) -- close
enough that the mismatch between "true frame boundary" and "nearest raw
sample" forms a beat pattern across the domain, vanishing at the edges and
peaking in the middle. That beat pattern **is** the previously-unexplained
middle bad patch documented above, and it also explains the general
high-`n_frames` degradation of Bskew/Bnorm.

**Fixed** in
[tube_fitter.py:801-814](../../../xtrack/_temp/splineboris/tube_fitter.py#L801).
Re-running the exact scan from "The core finding" above, after the fix
(tent, `deg=2`, `field_tol=1e-3`, full grid):

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
    4000    1.090%   1.407% 747.836%   <- Bs still blows up -- different mechanism, see below
    4402    1.090%   1.407% 849.432%
```

**Bskew/Bnorm are now monotonic and flat well past their old "best found"
point** -- no bad patch, no high-`n_frames` blowup. The `residual_tol`
search's blind spot (this note's original "Practical implication") is
resolved for Bskew/Bnorm as a result -- the internal residual and the
downstream metric no longer diverge for these components. `Bs` still blows
up catastrophically near the DOF ceiling, but this is a **separate**
mechanism (Bs's off-axis reconstruction depends on the
Bskew/Bnorm fit's *derivative* content via Maxwell's equations, not on
Bskew/Bnorm's values -- differentiating an increasingly-overfit tent spline
amplifies noise even though the values stay stable) -- see
[tube_hermite_export_boundary_bug.md](tube_hermite_export_boundary_bug.md)'s
closing section for detail, including its **RESOLVED (2026-08-26)** update:
confirmed via a matching `n_frames` scan on the analytic solenoid field that
this blowup is specific to datasets (like this undulator) where `Bs` is the
small, unfit, purely-derivative-reconstructed component -- a solenoid,
where `Bs` is the dominant directly-fit quantity, shows no blowup at all,
even at its own DOF ceiling. `residual_tol` is therefore still not safe to
trust blindly for undulator/wiggler-like maps where `Bs` is small, but this
is not a concern for solenoid-dominated field maps.
