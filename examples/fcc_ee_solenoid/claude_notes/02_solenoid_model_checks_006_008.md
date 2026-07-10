# 006-008: standalone solenoid-model validation scripts

Written 2026-07-10. Companion to `00_overview.md`. None of these three read or
write any of the `*.json` lattice files — each rebuilds field models and
element lines from scratch in-process, purely for model-to-model or
model-to-field-map comparison. Safe to ignore unless debugging the SplineBoris
fit or the Boris integrators themselves.

## 006_compare_individual_solenoid_models.py
- Builds the *isolated* main-solenoid field (`TiltedSolenoid`, same
  `L=2.46, a=0.13, B0=2.0, theta=-0.015`) and its taper/derivative extraction
  **inline** (a simplified, single-order copy of what became
  `spline_boris_setup.extract_tapered_field_data`, here capped at
  `MAX_MULTIPOLE_ORDER=1` and using plain `np.gradient` for s-derivatives
  instead of `LSQUnivariateSpline` — i.e. this predates or intentionally
  simplifies the 004a/spline_boris_setup machinery). Several boolean toggles
  control which simplifications are active:
  `USE_PIECEWISE_LINEAR_SPLINES`, `FORCE_IDEAL_SOLENOID_TRANSVERSE_FIELD`,
  `FORCE_ZERO_FIELD_AT_SOLENOID_ENDS`, `TAPER_SPLINEBORIS_FIELDS_TO_ZERO`
  (only the taper one is True by default).
- Builds 3 competing element representations of the *same* tilted solenoid
  over the same `s_axis = linspace(-2.399, 2.399, 201)`, each split into a
  left half + IP marker + right half: `SplineBoris`, `VariableSolenoid`, and
  a direct `xt.BorisSpatialIntegrator` (per-slice spatial field integration,
  no multipole expansion at all — the "ground truth" model since it just
  integrates the raw `field_model.get_field` callable).
- Open-twiss (`init_at='ip'`, `betx=0.09, bety=0.0007`) on the right-half-only
  lines for all three, comparing final `x, y, dy, betx2, bety1` — this is the
  head-to-head accuracy check of SplineBoris/VariableSolenoid against the
  BorisSpatialIntegrator ground truth for a single isolated solenoid (no
  compensation, no ring).
- Symplectic-error check (`||R^T S R - S||_2`) for the full SplineBoris line
  only.
- Prints `Bs` integral before/after taper and per-model twiss-end comparison
  table.

## 007_splineboris_tapered_solenoid_set_symplecticity.py
- Same physical setup as 006 but assembles the **full three-solenoid system**
  (comp_left + main + comp_right, 12 m spacing) for SplineBoris and
  VariableSolenoid via a locally-redefined `assemble_three_solenoid_system`
  (near-identical to the one in `spline_boris_setup.py`, but this copy has a
  bug-prone dependency on a module-level `drift_between_comp_and_main` global
  computed later in the script — works because Python closures capture by
  name at call time, but fragile if the function were extracted).
- Distinguishing feature vs 004a: adds a **matched compensation-solenoid
  scale knob** (`COMPENSATION_SCALE_KNOB = 'delta_compensation_scale'`,
  applied multiplicatively on top of the analytically-set `comp_scale_b`) and
  actually *solves* an optimizer
  (`spline_line.match(vary=Vary(delta_compensation_scale, ...),
  targets=[betx2, bety1, alfx2, alfy1 -> 0 at END])`) to null residual linear
  coupling at the far end, rather than trusting the analytic charge-balance
  alone. Reports the matched vs unmatched `delta_compensation_scale` and
  resulting total `Bs` integral.
- Computes symplectic error **and** `|det(R)|-1` error for both SplineBoris
  and VariableSolenoid full 3-solenoid systems; plots orbit, `dy`, `betx2`/
  `bety1`, and `alfx2`/`alfy1` vs `s - s_ip` for both models overlaid.
- Use this script as the reference if you need to re-tune
  `COMPENSATION_SCALE_KNOB_LIMITS = (-0.05, 0.05)` or re-check whether the
  analytic `comp_scale_b` alone (no extra matching) is sufficient — as of this
  writing the match is applied unconditionally
  (`MATCH_COMPENSATION_SOLENOID_SCALE = True`).

## 008_boris_spatial_three_solenoid_actual_field.py
- The "ground truth" companion to 007: instead of any multipole/spline
  approximation, tracks through the **literal combined analytic field**
  (`combined_field(x, y, s)`, a Python function summing `main_field_model` +
  scaled left/right `comp_field_model` contributions, each active only within
  its own half-range window) using one `xt.BorisSpatialIntegrator` per
  `SLICE_LENGTH=0.01 m` slice across the *entire* extended system
  (`s_start_system` to `s_end_system`, spanning both compensation windows
  plus margins — `MAIN_FIELD_HALF_RANGE=7.0`,
  `COMP_FIELD_HALF_RANGE = 1.0 + 3.0 = 4.0` around each comp solenoid center
  at `+-12 +-1 m`).
- `comp_scale_b` recomputed independently here via `np.trapezoid` over
  `linspace(*, 4001)` integration grids (not reusing 004a's saved value) —
  if this number ever disagrees meaningfully with 004a's, that's a sign the
  integration windows/resolutions matter (worth checking window truncation
  effects).
- No SplineBoris/VariableSolenoid comparison at all — this script only
  produces the actual-field Boris-integrated orbit/coupling/symplecticity
  numbers, meant to be compared *by eye* against 007's printed output for the
  same quantities (`x_end, y_end, dy_end, betx2_end, bety1_end,
  symplectic_error, det_r_error`).
- `include_collective=True` is required on both `get_R_matrix` and `.twiss`
  here because `BorisSpatialIntegrator` order this way needs it (a detail
  worth remembering if adapting this pattern elsewhere — plain `.twiss()`
  without `include_collective` will not work with per-slice
  `BorisSpatialIntegrator` chains built this way).
