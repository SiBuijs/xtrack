# `hermite_to_polynomial` reconstruction bug (found + fixed 2026-08-10)

> **HISTORICAL (checked 2026-10-14).** The code this note debugs is gone after the 2026-10-08 rewrite (`a8e5ecc17` and `0f77ee734`, after TubeFitter simplifications on 2026-10-06): `FieldFitter` and `SplineBorisSequence` were deleted, and `TubeFitter` now only produces on-axis multipoles at its frames, while the new `LongitudinalFitter` does the quartic-B-spline fit and the `to_line()`/`to_multipole_line()` export. See `00_overview.md`. There is no Hermite conversion in `TubeFitter` any more (the per-element `Spline4` values come from `LongitudinalFitter.element_params()`), so these bugs cannot recur in this form. Commit hashes and function names below refer to the old code.

**Status: FIXED.** Turned out to be a `TubeFitter`-only bug, not a shared
evaluator bug as first suspected (see "Root cause, corrected" below) --
`FieldFitter` was never affected. Two smaller, *fixed* bugs were found along
the way and are documented below too. This is the code path that actually
gets used for tracking, so it mattered more than any of the fit-quality work
in [tube_approach_kernel_simplification.md](tube_approach_kernel_simplification.md).

## How this was found

Motivated by the user wanting to know whether `TubeFitter`'s ~1% internal
residual is actually comparable to `FieldFitter`'s fit quality. Each fitter's
own residual metric (`transverse_fit_residual_rms` /
`tube_fit_residual_rms`) measures a different sub-problem and isn't directly
comparable (see [field_fitter.md](field_fitter.md) and
[fieldfitter_vs_tubefitter_comparison.md](fieldfitter_vs_tubefitter_comparison.md)),
so instead: build a `SplineBorisSequence` from each fitter's `df_fit_pars`
and compare the actual predicted field (`seq.get_field(x, y, s)`) against
the raw measured data -- the number that actually matters for tracking. This
is now a permanent section in `003c_fieldfitter_vs_tubefitter.py`
([003c_fieldfitter_vs_tubefitter.py:69-120](../003c_fieldfitter_vs_tubefitter.py#L69-L120)).

## Bug 1 (fixed): `FieldFitter._set_raw_data` mutated the caller's DataFrame

[field_fitter.py:131](../../../xtrack/_temp/splineboris/field_fitter.py#L131)
did `self.df_raw_data = raw_data` (no `.copy()`), then rescaled
`self.df_raw_data.index` by `distance_unit` **in place** -- since
`self.df_raw_data` was literally the same object the caller passed in, this
silently mutated it. `TubeFitter`'s equivalent method already copies
correctly. Any caller reusing their original DataFrame after `fit()` (e.g.
to build query points for a residual check, exactly what happened here) got
back a double-scaled coordinate system with no warning.

**Fix applied:** `self.df_raw_data = raw_data.copy()`. Verified the caller's
DataFrame is now unchanged after `fit()`; full test suite still green
(45 passed / same 4 pre-existing-or-parked failures, see the other note).

This one first manifested as a seemingly catastrophic "off-axis `Bs` is
1000x too small" result, entirely because a query point had been silently
double-scaled by the leftover mutation. Worth remembering as a general
lesson: always suspect the test harness before the code under test when a
result looks *that* wrong.

## Bug 2 (fixed, smaller effect than expected): `_safe_derivative_at` averaged the wrong thing

[tube_fitter.py:80-88](../../../xtrack/_temp/splineboris/tube_fitter.py#L80)
(pre-fix) computed the Hermite `der_start`/`der_end` boundary values as
`0.5*(f'(s-eps) + f'(s+eps))` -- averaging the one-sided derivative *entering*
a region with the one-sided derivative *leaving the previous* region (or
*entering the next* one, at the far end). For tent, every frame boundary is
a genuine kink (adjacent linear segments generally have different slopes),
so this average is neither region's true slope.

Verified directly: at the region straddling `s=0` in the undulator fit,
the true one-sided slope entering and leaving was **126.8385...** on both
sides (correct, since a single tent region is exactly linear), but the old
averaged value gave **132.79** at the left edge and **118.81** at the right
-- both wrong.

**Fix applied:** renamed to `_derivative_into_region(f, s, eps)`, returning
`f'(s+eps)` only, called with `+eps` at `s_left` and `-eps` at `s_right`
([tube_fitter.py:80-99](../../../xtrack/_temp/splineboris/tube_fitter.py#L80)) --
i.e. always evaluate strictly *inside* the region being reconstructed.
Verified this now reproduces the true region slope exactly (both ends give
`126.8385...`, matching to the full precision checked).

**However:** rerunning the 003c residual comparison after this fix showed
almost no change in the overall numbers (35.225% before -> 35.225% after for
`Bskew`, full grid). This fix is real and correct, but turned out to be a
much smaller contributor than Bug 3 below, which dominates.

## Bug 3 (FIXED): `TubeFitter` fed `hermite_to_polynomial` derivatives in the wrong units

This is the actual, dominant cause of TubeFitter's large downstream
residuals (up to 152% relative RMS on `Bs`, full grid, vs FieldFitter's
~1-4%) -- and, confirmed, of the previously-failing
`test_splineboris_tubefitter_solenoid_vs_variable_solenoid`, documented in
[tube_approach_kernel_simplification.md](tube_approach_kernel_simplification.md),
which now passes.

**Isolated with a clean, synthetic, unambiguous test:** picked an arbitrary
exact quartic polynomial `f(s) = a0 + a1*s + a2*s^2 + a3*s^3 + a4*s^4`,
computed its *exact* Hermite data analytically (value + derivative at both
region endpoints, plus the true interval average), fed that into
`hermite_to_polynomial`
([spline_B_field_eval_python.py:10](../../../xtrack/beam_elements/splineboris_src/spline_B_field_eval_python.py#L10)),
and compared the reconstructed polynomial against the known-exact original:

| position in region | true value | reconstructed | error |
|---|---|---|---|
| start (trivial) | 0.46462625 | 0.46462625 | 0 |
| 25% through | 0.49764695 | 0.48517908 | **1.25e-2** |
| midpoint | 0.53102000 | 0.53088701 | 1.33e-4 |
| 75% through | 0.56465820 | 0.57717595 | **-1.25e-2** |
| end (trivial) | 0.59847625 | 0.59847625 | 0 |

Since the input Hermite data was *exact* (not from any fit), a correct
implementation must reproduce the original polynomial exactly everywhere,
not just at the two endpoints (which any implementation gets right trivially
-- `hermite_to_polynomial(s_start, ...)` and `(..., s_end)` map directly to
`c1`/`c3` without needing the basis-function machinery at all). It doesn't.
The error is real math, not floating-point noise: reproduced with a
synthetic *linear* function too (where it should be even more trivially
exact), and the error scales linearly with the region's derivative
magnitude:

```
slope=3.0     rel error at 25%-through = 0.0698%
slope=30.0    rel error = 0.6650%
slope=127.0   rel error = 2.4017%   (~matches real TubeFitter fit data)
slope=1270.0  rel error = 8.7943%
```

(Reproduction snippet, region `s in [0.0005, 0.0015]`, matching the real
fit's magnitudes:
```python
from xtrack.beam_elements.splineboris_src.spline_B_field_eval_python import hermite_to_polynomial
def make_hermite(s_start, s_end, slope, intercept):
    L = s_end - s_start
    val_start, val_end = intercept + slope*s_start, intercept + slope*s_end
    return (val_start, slope*L, val_end, slope*L, 0.5*(val_start+val_end))
h = make_hermite(0.0005, 0.0015, slope=127.0, intercept=0.4)
poly = hermite_to_polynomial(0.0005, 0.0015, h)
# poly(0.25*L) should equal intercept + slope*(s_start + 0.25*L) exactly -- it doesn't.
```
)

### Root cause, corrected: not a shared-evaluator bug at all -- `TubeFitter` alone used the wrong convention

Initial suspicion (below, kept for the record) was that the shared
`hermite_to_polynomial` formula's `b1..b5` basis functions or their
combination were wrong. They aren't. Re-derived each `b_i` from scratch from
its defining Hermite/integral conditions (`b1(0)=1` & all else 0 at `t=0/1`
and `∫b1=0`; `b2'(0)=1` & all else 0; etc.) and verified analytically that
every `b1..b5` in
[spline_B_field_eval_python.py:24-28](../../../xtrack/beam_elements/splineboris_src/spline_B_field_eval_python.py#L24)
satisfies its conditions exactly. The combination formula
(`poly_t = c1*b1 + L*c2*b2 + c3*b3 + L*c4*b4 + c5*b5`) is also correct --
*provided* `c2`/`c4` (`der_start`/`der_end`) are the **raw**, unscaled
derivative `f'(s)`, and `c5` is the **average** value over the region
(`integral / L`). Confirmed algebraically: `poly_t'(0) = L*c2*b2'(0) = L*c2`,
and since `poly_s'(s0) = poly_t'(0)/L`, that only equals `c2` (i.e. only
correctly reproduces `f'(s0)`) when `c2` itself is the raw derivative.

Traced via `git log -S"L * c2 * b2"` to the formula's origin (commit
`adce1686f`, the first version of this function): back then `c2`/`c4` were
explicitly documented as raw derivatives and `c5` was the raw *integral*
(needing `c5/L` in the formula, not `c5`). Commit `aeb4ec675` ("Refactor of
polynomials... store f(s_1), f'(s_1), f(s_2), f'(s_2) and the integral over
the region... more consistent") changed `FieldFitter` to compute
`average = integral/L` instead of the raw integral, and correctly updated
the formula's `c5` term to drop the now-redundant `/L`. It left the
`L*c2`/`L*c4` terms untouched -- correctly, since `FieldFitter`'s derivative
computation (`_boundary_from_finite_differences` in
[field_fitter.py:346](../../../xtrack/_temp/splineboris/field_fitter.py#L346))
was never changed and still returns the **raw** derivative to this day.

**`FieldFitter` was therefore never affected by this bug.** The break came
later, when `TubeFitter` was written: its `_hermite_from_bspline`
(pre-fix, [tube_fitter.py:95-104](../../../xtrack/_temp/splineboris/tube_fitter.py#L95))
independently chose to *pre-scale* `der_start`/`der_end` by `L` --
apparently by analogy with the "everything's in its final, already-scaled
form" spirit of the `average` convention -- without knowing the shared
formula still expected raw derivatives. That mismatch is the entire bug:
`TubeFitter`'s derivatives got multiplied by `L` twice (once explicitly at
the call site, once implicitly inside the formula).

So this was never "shared code exposed more by TubeFitter's short regions"
as first hypothesized (the `FieldFitter`-vs-`TubeFitter` region-length
argument was a red herring, or maybe suppresses a different second-order
effect) -- it's a plain unit mismatch confined entirely to `TubeFitter`.

### Fix applied

Removed the two spurious `* L` factors in `_hermite_from_bspline`
([tube_fitter.py:95-104](../../../xtrack/_temp/splineboris/tube_fitter.py#L95)),
making `TubeFitter` emit raw derivatives, matching `FieldFitter`'s
(correct, already-shared) convention:

```python
def _hermite_from_bspline(f: BSpline, s_left: float, s_right: float) -> tuple[float, ...]:
    L = s_right - s_left
    eps = 1e-6 * L
    return (
        float(f(s_left)),
        _derivative_into_region(f, s_left, eps),      # was: * L
        float(f(s_right)),
        _derivative_into_region(f, s_right, -eps),     # was: * L
        float(f.integrate(s_left, s_right)) / L,
    )
```

No changes needed to the shared evaluator (`spline_B_field_eval_python.py`,
`elements.py`, or the C backend `spline_B_field_eval.h`) -- it was correct
all along for the convention `FieldFitter` already used. Confirmed the
`_hermite` dict's contents aren't consumed anywhere else inside `TubeFitter`
expecting the old (L-scaled) convention, so this is a self-contained,
one-file fix.

**Verified:**
- Exact-cubic `BSpline` reconstruction test at production-scale region
  length (`L=0.001`, matching real tent-region sizes): relative error
  dropped from multi-percent to `~1e-11%` (residual floor set by the
  `eps`-nudged one-sided-derivative estimate, not by any remaining formula
  error).
- `tests/test_splineboris.py`: **46 passed, 3 failed** (up from 45/4).
  The 3 remaining failures are the pre-existing, unrelated
  `PrebuiltKernelNotFoundError` (stale compiled-kernel cache, see
  [tube_approach_kernel_simplification.md](tube_approach_kernel_simplification.md)).
  `test_splineboris_tubefitter_solenoid_vs_variable_solenoid` -- left
  failing on purpose pending this exact investigation -- **now passes**.

## 003c residual numbers after the fix (2026-08-10)

Rerun of the same comparison from "How this was found" above, now that Bug 3
is fixed:

```
=== SplineBoris field-prediction residual vs raw data (full grid) ===
field    fitter            RMS [T]    rel RMS
Bskew    FieldFitter    2.8179e-03     1.303%
Bnorm    FieldFitter    1.2170e-02     2.037%
Bs       FieldFitter    3.7881e-03     4.273%
Bskew    TubeFitter     9.6351e-03     4.456%
Bnorm    TubeFitter     3.2500e-02     5.440%
Bs       TubeFitter     9.4378e-03    10.647%

=== SplineBoris field-prediction residual vs raw data (on-axis only (X=0, Y=0)) ===
field    fitter            RMS [T]    rel RMS
Bskew    FieldFitter    2.6359e-03     1.230%
Bnorm    FieldFitter    1.1608e-02     1.963%
Bs       FieldFitter    5.1114e-08   100.000%
Bskew    TubeFitter     9.3658e-03     4.370%
Bnorm    TubeFitter     3.1403e-02     5.310%
Bs       TubeFitter     5.1114e-08   100.000%
```

TubeFitter's `Bs`/full-grid number alone dropped from 152% to 10.6% --
confirms Bug 3 was indeed the dominant contributor. TubeFitter is now in the
same ballpark as FieldFitter (single-digit-to-low-double-digit %), though
still consistently ~2-3x worse across every field component. That remaining
gap is now plausibly a genuine fit-quality difference between the two
approaches (global sparse tube fit vs adaptive peak/valley regions) rather
than an evaluator bug -- not yet investigated further.

The identical `Bs` on-axis row for both fitters (`5.1114e-08 T, 100.000%`)
is a **benign artifact, not a result**: on-axis `Bs` is ~0 by undulator
symmetry, so `field_rms` in the denominator
([003c_fieldfitter_vs_tubefitter.py:108](../003c_fieldfitter_vs_tubefitter.py#L108))
is just measurement/floating-point noise. Both fitters correctly predict
~0 there (below `field_tol`, not fitted), so `rel = noise/noise ≈ 100%` for
both, identically. Not evidence of a real problem -- safe to ignore that
specific row when reading this table.
