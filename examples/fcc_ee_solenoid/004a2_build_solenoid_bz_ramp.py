"""Build a Maxwell-consistent Bz-ramp perturbation and add it to the existing
main-solenoid SplineBoris template, without touching 004a's fit at all.

`extract_tapered_field_data` + `build_splineboris_line` are, for fixed
s_axis/taper/spline config, a *linear* map from sampled field values to the
final Hermite coefficients (every step is a fixed-knot least-squares fit or a
field-independent taper multiplication). So instead of building one composite
analytic field (main solenoid + ramp) and fitting once, we fit the ramp alone
with the exact same config 004a used, and add its Spline4 coefficients to the
already-saved `main_solenoid` template slice-by-slice. This is mathematically
identical to fitting the sum once, but never touches the existing fit of the
real solenoid.

The ramp field itself (see bz_ramp_field.py) is an *exact* vacuum solution
(div B = curl B = 0 for all r, not just paraxially): Bz = slope*(s - s0),
Br = -(slope/2)*r. Its integral over the (symmetric) main-solenoid s-axis is
exactly zero, so it does not perturb comp_scale_b / the main-vs-compensation
charge balance -- it purely breaks the main solenoid's own front/back mirror
symmetry.
"""
import argparse
from pathlib import Path

import numpy as np
import xtrack as xt

from spline_boris_setup import build_splineboris_line, extract_tapered_field_data
from bz_ramp_field import LinearBzRamp
from solenoid_params import FIELD_TAG
from tilted_solenoid import TiltedSolenoid

parser = argparse.ArgumentParser()
parser.add_argument('--ramp-amp', type=float, default=0.2)
args = parser.parse_args()

HERE = Path(__file__).parent
INPUT_LINES_JSON = HERE / f'004_solenoid_lines_{FIELD_TAG}.json'
OUTPUT_LINES_JSON = (
    HERE / f'004_solenoid_lines_{FIELD_TAG}_bz_ramp_{args.ramp_amp:g}.json')

# --- Must match 004a_build_and_check_solenoids.py exactly, or the coefficient
# addition below is no longer equivalent to fitting the summed field once. ---
THETA = -0.015
MAX_TRANSVERSE_DERIVATIVE_ORDER = 4
MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE = 4
DERIVATIVE_STEP = 5e-4
SPLINE_INTEGRAL_POINTS = 10
DECREASE_S_POLY_ORDER_WITH_TRANSVERSE_ORDER = True
S_DERIVATIVE_SPLINE_ORDER = 4
MAX_S_DERIVATIVE_PLOT_ORDER = 5
SPLINE_STEPS_PER_POINT = 10
TAPER_LENGTH = 0.15  # m
USE_NEAR_AXIS_SIMPLIFIED_MODEL = False
MAIN_SOLENOID_S_AXIS = np.linspace(-2.399, 2.399, 201)
MAIN_SOLENOID_L = 1.23 * 2
MAIN_SOLENOID_B0 = 2.0
# ---------------------------------------------------------------------------

# Ramp amplitude knob: slope chosen so Bz(s) changes by
# +-RAMP_AMP * MAIN_SOLENOID_B0 between the two ends of the physical coil
# (s = +-L/2), i.e. RAMP_AMP=1 is a full-B0-scale end-to-end swing.
RAMP_AMP = args.ramp_amp
RAMP_SLOPE = RAMP_AMP * MAIN_SOLENOID_B0 / (MAIN_SOLENOID_L / 2.0)

field_extraction_kwargs = {
    'max_transverse_derivative_order': MAX_TRANSVERSE_DERIVATIVE_ORDER,
    'derivative_step': DERIVATIVE_STEP,
    'spline_integral_points': SPLINE_INTEGRAL_POINTS,
    'taper_length': TAPER_LENGTH,
    's_derivative_spline_order': S_DERIVATIVE_SPLINE_ORDER,
    'max_s_derivative_plot_order': MAX_S_DERIVATIVE_PLOT_ORDER,
    'decrease_s_poly_order_with_transverse_order': (
        DECREASE_S_POLY_ORDER_WITH_TRANSVERSE_ORDER),
}
splineboris_build_kwargs = {
    'max_transverse_derivative_order_for_spline': (
        MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE),
    'spline_steps_per_point': SPLINE_STEPS_PER_POINT,
    'use_near_axis_simplified_model': USE_NEAR_AXIS_SIMPLIFIED_MODEL,
}

ramp_field_model = LinearBzRamp(slope=RAMP_SLOPE, z0=0.0)
ramp_field_data = extract_tapered_field_data(
    name='main_solenoid_bz_ramp',
    field_model=ramp_field_model,
    s_axis=MAIN_SOLENOID_S_AXIS,
    **field_extraction_kwargs)
line_ramp = build_splineboris_line(
    name='main_solenoid_bz_ramp',
    field_data=ramp_field_data,
    scale_b=1.0,
    sextupole_amplification_factor=1.0,
    **splineboris_build_kwargs)

# Load the existing, untouched main-solenoid template.
line_data = xt.json.load(INPUT_LINES_JSON)
line_main = xt.Line.from_dict(line_data['lines']['main_solenoid'])

assert len(line_main.elements) == len(line_ramp.elements)

combined_elements = []
for el_main, el_ramp in zip(line_main.elements, line_ramp.elements):
    assert isinstance(el_main, xt.SplineBoris) and isinstance(el_ramp, xt.SplineBoris)
    assert el_main.multipole_order == el_ramp.multipole_order
    assert np.isclose(el_main.length, el_ramp.length)
    assert np.isclose(el_main.scale_b, 1.0) and np.isclose(el_ramp.scale_b, 1.0)

    combined_elements.append(xt.SplineBoris(
        bs=xt.Spline4(*(np.array(el_main.bs) + np.array(el_ramp.bs))),
        bx=tuple(
            xt.Spline4(*(np.array(el_main.bx[order]) + np.array(el_ramp.bx[order])))
            for order in range(el_main.multipole_order)),
        by=tuple(
            xt.Spline4(*(np.array(el_main.by[order]) + np.array(el_ramp.by[order])))
            for order in range(el_main.multipole_order)),
        length=el_main.length,
        n_steps=el_main.n_steps,
        scale_b=1.0,
    ))

line_main_bz_ramp = xt.Line(
    elements=combined_elements, element_names=list(line_main.element_names))

##############################################################################
# Sanity checks -- cheap, no rebuild of the real solenoid fit required.
##############################################################################

# 1) The ramp is pure order-0/order-1 content (linear Br(r), constant slope
#    Bz); the fitted sextupole-and-above orders should be ~0.
max_high_order_coeff = max(
    np.max(np.abs(el.bx[2:])) if el.multipole_order > 2 else 0.0
    for el in line_ramp.elements
)
print(f'Ramp: max |bx/by| coefficient at order>=2 = {max_high_order_coeff:.3e} T '
      '(expect ~0)')

# 2) Spot-check the combined SplineBoris line's evaluated field against the
#    literal analytic sum (main TiltedSolenoid + LinearBzRamp) at a handful of
#    off-axis points -- confirms the "add fitted coefficients" shortcut agrees
#    with "fit the analytic sum once". Restricted to the untapered interior:
#    near the s_axis edges the fit is deliberately tapered to zero while the
#    raw analytic field is not, so edge points are not a meaningful check.
main_field_model = TiltedSolenoid(
    L=MAIN_SOLENOID_L, a=0.13, B0=MAIN_SOLENOID_B0, theta=THETA)

rng = np.random.default_rng(0)
max_rel_err = 0.0
s_lo = MAIN_SOLENOID_S_AXIS[0] + TAPER_LENGTH
s_hi = MAIN_SOLENOID_S_AXIS[-1] - TAPER_LENGTH
for _ in range(200):
    x0 = rng.uniform(-0.02, 0.02)
    y0 = rng.uniform(-0.02, 0.02)
    s0 = rng.uniform(s_lo, s_hi)

    bx_ref, by_ref, bs_ref = main_field_model.get_field(
        np.array([x0]), np.array([y0]), np.array([s0]))
    bx_ref2, by_ref2, bs_ref2 = ramp_field_model.get_field(
        np.array([x0]), np.array([y0]), np.array([s0]))
    bx_ref = bx_ref[0] + bx_ref2[0]
    by_ref = by_ref[0] + by_ref2[0]
    bs_ref = bs_ref[0] + bs_ref2[0]

    idx = np.searchsorted(MAIN_SOLENOID_S_AXIS, s0, side='right') - 1
    s_local = s0 - MAIN_SOLENOID_S_AXIS[idx]
    bx_fit, by_fit, bs_fit = line_main_bz_ramp.elements[idx].get_field(
        x0, y0, s_local)

    denom = max(abs(bx_ref), abs(by_ref), abs(bs_ref), 1e-6)
    rel_err = max(abs(bx_fit - bx_ref), abs(by_fit - by_ref),
                  abs(bs_fit - bs_ref)) / denom
    max_rel_err = max(max_rel_err, rel_err)

print(f'Combined SplineBoris vs analytic (main + ramp), untapered interior: '
      f'max rel. error over 200 random points = {max_rel_err:.3e}')

##############################################################################
# Save
##############################################################################

output_data = {
    'metadata': dict(
        line_data['metadata'],
        ramp_amp=RAMP_AMP,
        ramp_slope_T_per_m=RAMP_SLOPE,
        ramp_z0=0.0,
        ramp_bs_integral=float(np.trapezoid(
            ramp_field_data['bs'], ramp_field_data['s_axis'])),
        note=(
            'main_solenoid here is main_solenoid (from 004_solenoid_lines.json) '
            'plus a Maxwell-consistent linear-in-s Bz ramp, added at the Hermite'
            ' coefficient level (see 004a2_build_solenoid_bz_ramp.py).'
        ),
    ),
    'lines': dict(
        line_data['lines'],
        main_solenoid=line_main_bz_ramp.to_dict(),
    ),
}
xt.json.dump(output_data, OUTPUT_LINES_JSON, indent=1)
print(f'Saved combined template to {OUTPUT_LINES_JSON}')
