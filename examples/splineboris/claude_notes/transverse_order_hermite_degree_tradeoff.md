# Reduced longitudinal polynomial order at higher transverse-derivative order (`004a_build_and_check_solenoids.py`)

Diagnosed 2026-08-05 while investigating a discrepancy the user pointed out
in `examples/fcc_ee_solenoid/004a_build_and_check_solenoids.py`'s figure
1300 (`d^4 B_x / dx^2 ds^2`) and the order-4 panels of its 600-series
figures. Confirms, with code and plots, a historical SplineBoris issue the
user recalled from memory: "it was favourable to decrease the Hermite
spline order as we went higher into transverse derivatives." Also connects
to the `1/dz²`-amplification-under-differentiation conditioning issue found
in `TubeFitter`'s `Bs` coupling (see the `tube_fitter.py` work on this
branch) — same underlying phenomenon, different axis.

## The mechanism

`extract_tapered_field_data` in `examples/fcc_ee_solenoid/spline_boris_setup.py`
fits each transverse-derivative term of `Bx`/`By` (order 0 = the field
itself, up to order `MAX_TRANSVERSE_DERIVATIVE_ORDER=4`) with a per-slice
longitudinal polynomial whose degree *drops* with transverse order
(`spline_boris_setup.py:298-302`, gated by
`DECREASE_S_POLY_ORDER_WITH_TRANSVERSE_ORDER`):

```python
degree = max(0, 4 - order)   # order 0 -> degree 4, order 4 -> degree 0
```

This is deliberate regularization: the raw per-order data being fit is
itself a 9-point finite difference in `x` (`offsets = -4..4`) divided by
`DERIVATIVE_STEP**order`. At `order=4` with `DERIVATIVE_STEP=5e-4`, that's
a division by `~6e-14` — any noise in the underlying field evaluation is
amplified enormously before it ever reaches the longitudinal fit. Cutting
the longitudinal polynomial's degrees of freedom at high order stops that
fit from chasing the amplified noise.

The catch: regardless of the fit `degree`, every term is unconditionally
repackaged into a full Hermite-quartic `xt.Spline4(val_start, der_start,
val_end, der_end, mean)` per slice (`spline_boris_setup.py:477-490`) — the
same 5-parameter shape SplineBoris always uses, independent of what degree
polynomial actually produced those 5 numbers. At `order=4` (`degree=0`),
the source polynomial is a flat per-slice constant, so `der_start =
der_end = 0` and `val_start = val_end = mean` — the "Hermite quartic"
degenerates into a literal piecewise-constant staircase. At `order=2`
(`degree=2`, used by the `MIXED_DERIVATIVE_SPECS` entry in `004a`), the
quadratic fit's boundary value/derivative get stretched into a quartic
shape that doesn't match the quadratic's actual curvature between slice
boundaries.

## Visible consequences (confirmed by running `004a` and inspecting the figures)

- **Order-4 pure transverse derivative** (figure 600, panel "order 4",
  `d^4 B_x / dx^4`): zooming in near `s=0` shows the SplineBoris trace
  (dashed) as a genuine piecewise-constant staircase that does not track
  the smooth field-map curve's shape *within* any slice, even though ~200
  slices should be fine-grained enough for the feature width. This is the
  direct, mechanical result of `degree=0` at this order.

- **Figure 1300** (`d^4 B_x / dx^2 ds^2`, `transverse_order=2, s_order=2`
  in `MIXED_DERIVATIVE_SPECS`): milder on its own (`degree=2`, not 0), but
  the diagnostic differentiates the built line's field two more times in
  `s` (`np.gradient`, twice) — amplifying the quartic-vs-quadratic shape
  mismatch into visible jagged overshoot right at the taper edges
  (~±20-30% spikes, e.g. a smooth ~-4700 minimum reproduced as a jagged
  dip to ~-5800). Same "differentiation amplifies whatever approximation
  error already exists" mechanism as the `1/dz²` noise amplification in
  `TubeFitter`'s `Bs`-coupling elimination — there the amplification comes
  from repeated differentiation of a spline in `z`; here from repeated
  differentiation (both in fitting, via the `1/DERIVATIVE_STEP**order`
  finite difference, and in the diagnostic, via `np.gradient`) stacked on
  top of an already-reduced-order longitudinal representation.

## This is not just a diagnostic-plot artifact

`004a`'s own printed twiss comparison shows a real, physical tracking
consequence:

```
SplineBoris:        betx2_end = +3.144e-03 m, bety1_end = +7.727e-04 m
VariableSolenoid:    betx2_end = +2.835e-17 m, bety1_end = +5.052e-19 m
```

`VariableSolenoid` (effectively exact) gives ~0 residual x/y coupling
through the three-solenoid system; `SplineBoris` leaves a nonzero
residual, consistent with the higher-order transverse terms being
under-resolved exactly where the field curves sharply (the taper edges —
same location as the figure-1300 artifacts).

## Status

Diagnosed, not fixed. This is a genuine trade-off, not simply a bug:
`degree = max(0, 4 - order)` buys noise robustness at the cost of
under-resolving real curvature at high transverse-derivative order, and
the current rule is quite aggressive (`degree=0` at order 4 — zero
resolution along `s` within a slice). Possible directions, not attempted:
finer/adaptive slicing concentrated at high-curvature regions (e.g. taper
edges) specifically for the high-order terms, or a less aggressive
degree-reduction schedule now that this failure mode is characterized.
