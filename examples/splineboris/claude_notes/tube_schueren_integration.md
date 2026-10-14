# Tube → Schueren integration: resolves the solenoid gap

> **ARCHITECTURE STILL CURRENT, IMPLEMENTATION HISTORICAL (checked 2026-10-14).** The design holds: `TubeFitter` passes on only the q=0/q=1 rows of `Psi`, and the Van der Schueren/Table-1 evaluator (`evaluate_B`, `xtrack/beam_elements/splineboris_src/`) regenerates the rest. `tube_fitter.py`'s docstring still cites this note, and `check_trace_consistency()` still exists. The helper functions named below (`_convert_to_hermite`, `_free_and_derived_pq`, `bs_smoothing`, ...) no longer exist. B_s now comes straight from the map's on-axis values (`TubeFitter.on_axis_bs()`), fitted by `LongitudinalFitter` like any other component. Of the two solenoid-vs-VariableSolenoid tests, only the TubeFitter one, `test_splineboris_tubefitter_solenoid_vs_variable_solenoid`, remains; the FieldFitter one, `test_splineboris_solenoid_vs_variable_solenoid`, was removed in `c50601fb6` (2026-08-11).

Implements and validates the architecture from the user's handoff note
(2026-08-07): `TubeFitter` never needs to enforce div(B) = 0 on itself, or
export any `q>=2` (trace) content, because the downstream Schueren/Table-1
field evaluator (`evaluate_B`,
[spline_B_field_eval.h](../../../xtrack/beam_elements/splineboris_src/spline_B_field_eval.h))
already regenerates all `q>=2` structure from `(a_n, b_n, b_s)` alone,
Maxwell-consistent by construction. Supersedes the diagnosis in
[transverse_bs_coupling_gap.md](transverse_bs_coupling_gap.md) and the "Why
there's no `TubeFitter`-vs-`VariableSolenoid` test" section of
[fieldfitter_vs_tubefitter_comparison.md](fieldfitter_vs_tubefitter_comparison.md)
-- both of which correctly diagnosed the *mechanism* (`Psi[0,2]` is never
exported) but incorrectly concluded this was a genuine gap requiring a fix
inside `TubeFitter` itself.

## The insight

`TubeFitter`'s `(a_m, b_m)` export ladder reads only `Psi[m,0]` (`q=0`,
skew) and `Psi[m,1]` (`q=1`, norm) -- exactly the Van der Schueren
potential's Cauchy data (`phi_0`, `phi_1`). Everything with `q>=2`,
including `Psi[0,2]` (needed to represent a solenoid's `dBy/dy`), is
**generated** by Schueren's own recurrence (order-by-order solution of
`Laplace(phi)=0` in `y`) at *evaluation* time, from `a_n`, `b_n`, and `b_s`
alone -- not fit or exported by `TubeFitter` at all. Concretely (`h=0`,
`B=-grad(Omega)`, `Omega` the tube's potential, `phi` Schueren's):

```
a_n = -n!     * Psi[n,0]      (q=0 row, from d_x^{n-1} Bx |_0)
b_n = -(n-1)! * Psi[n-1,1]    (q=1 row, from d_x^{n-1} By |_0)
b_s(z) = Bs(0,0,z)            (independent on-axis fit, unrelated to Psi[0,0])
```

so the tube fit only needs to supply the `q=0`/`q=1` rows correctly; it
never needs `Psi[0,2]` to be right, or even present.

**Why the earlier diagnosis missed this**: it implicitly assumed the
exported `(a_m, b_m)` ladder needed to *represent* `dBy/dy` directly for the
reconstructed field to be correct -- i.e. that `TubeFitter` itself had to
be internally Maxwell-consistent. It doesn't: `evaluate_B` already performs
the Laplace-consistent reconstruction downstream, using only the same
Cauchy data `FieldFitter` has always supplied (which also never fits or
exports `q>=2` terms -- `FieldFitter._fit_transverse_polynomials` only ever
touches the `y=0` line).

**Verification (this session)**: rederived the `a_n`/`b_n`/`b_s` dictionary
above from scratch (differentiating `Bx`, `By` directly from the `Psi`
potential) and confirmed it matches the code in `_convert_to_hermite`
exactly, including the `a_0 = -Psi[0,0]` / `b_s = a_0'` gauge argument for
why `Psi[0,0]` can never be seen by the `Bx`/`By` system (both `p` and `q`
are `0`, so its coefficient in every field-derivative term is always `0`).
Also rederived the trace relation `Psi[2,0] + Psi[0,2] = Bs'(z)/2` from
`Laplace(Omega)=0` at the origin and confirmed it matches
`transverse_bs_coupling_gap.md`'s.

## What changed in `tube_fitter.py`

First pass (2026-08-07, morning): decoupled the joint-Bs-fit machinery from
`y_symmetry` into a separate `couple_bs` flag (default `False` = fit `Bs`
independently, matching this note's architecture; `True` preserved the old
`transverse_bs_coupling_gap.md` elimination behavior for research use).

**Second pass (2026-08-07, afternoon): removed `couple_bs` entirely**, at
the user's request -- "I feel I will not really use `couple_bs` in
practice... it's okay to just show the deviation from Maxwell and be
honest about it." `Bs` is now *always* fit independently on-axis
(`_fit_bs`), unconditionally, exactly like `FieldFitter`. This also deletes
the code that motivated it: the `div(B)=0` elimination in
`_build_linear_system` (`d2_basis`, the `Psi[p,2]` derivation loop, the
joint Bs data/smoothing rows), `_free_and_derived_pq`'s free/derived split
(now every generated `(p, q)` pair is simply fit directly -- no `(0, 0)`
gauge term, no derived pairs), `_derived_transverse_psi`, the `bs_smoothing`
parameter, and the `deg<=2` / `fit_skew=True`-with-`y_symmetry=False`
construction-time restrictions (both existed only to bound the elimination;
with it gone, `deg` and `fit_skew` are unconstrained by `y_symmetry` again).
`transverse_bs_coupling_gap.md` is kept as a historical record of what was
tried and why -- the code it describes no longer exists.

`y_symmetry` keeps its original, narrower meaning: whether the transverse
potential basis includes even-`q` (`q>=2`) pairs at all (still fit freely
when present, purely to protect the exported `q=0`/`q=1` columns from
bias -- never exported itself, regardless of `y_symmetry`).

`TubeFitter.check_trace_consistency()` (added in the first pass) is kept
as-is: an optional, honest diagnostic (needs `y_symmetry=False`, so
`Psi[0,2]` is in the basis and fit freely) comparing the tube's own fitted
`Psi[0,2](z)` against the Laplace/`Bs'` prediction `-Psi[2,0](z) +
Bs'(z)/2`. A large gap flags under-resolved `Bx`/`By` data (raise
`n_frames`, shrink `tube_radius`, check grid sampling) -- it does **not**
indicate an export problem, since `Psi[0,2]` is never exported either way.
This -- plus the raw-data `div(B)` print below -- is the intended way to
see how well the "Schueren regenerates it downstream" assumption holds on
a given dataset, rather than trying to force it via a coupled fit.

## Raw-data div(B) print (added after the above)

Added `xtrack/_temp/splineboris/div_b_check.py:report_raw_div_b`, called from both
`FieldFitter._set_raw_data` and `TubeFitter._set_raw_data` (construction
time, before any fitting). Estimates div(B) = dBx/dx + dBy/dy + dBs/dz
directly from the raw (X, Y, Z) grid via `np.gradient` finite differences --
independent of either fitting method, since both start from the same kind
of rectangular field-map grid (`FieldFitter` just names the raw columns
`Bskew`/`Bnorm` instead of `Bx`/`By`). Prints the RMS deviation from zero,
relative to the typical individual-term magnitude
(`sqrt(mean(dBx/dx^2 + dBy/dy^2 + dBs/dz^2))`).

This is deliberately a *data*-quality check, not a *fit*-quality one: per
the "insight" above, neither fitter's exported coefficients are expected to
make div(B) vanish on their own (that's the evaluator's job downstream), so
checking div(B) against the fit output wouldn't be very informative. What
*is* informative is whether the raw field map itself is already
Maxwell-consistent -- confirmed identical (5.073e-05 T/m, 0.01%) for
`FieldFitter` and `TubeFitter` on the same analytic solenoid map (same
underlying data, different raw column names), and nonzero but small (4.32%
of the term scale) on the real SLS undulator map, consistent with that
being measured/discretized data rather than an analytic field.

## Validation

Added `test_splineboris_tubefitter_solenoid_vs_variable_solenoid`
(`tests/test_splineboris.py`), the direct analogue of the pre-existing
`test_splineboris_solenoid_vs_variable_solenoid` (FieldFitter), fitting the
same paraxial-solenoid field map with `TubeFitter` at its new defaults.
Previously (per `fieldfitter_vs_tubefitter_comparison.md`) this diverged
from `VariableSolenoid` by roughly an order of magnitude; it now passes at
the same tolerance FieldFitter needs (~3% of the trajectory-derivative
range), and `check_trace_consistency()` reports the tube's internal
`Psi[0,2]` fit agrees with the Laplace prediction to ~0.01%. All
pre-existing `TubeFitter` tests (undulator-vs-FieldFitter,
`n_frames`/`residual_tol` unit tests) still pass unchanged under the new
default -- see git history for this branch for the exact diff.
