"""004e: effect of a localized sextupole-content window on the
beta_x-weighted d^2Bx/dx^2 driving-term integral.

Follow-up to 004a_build_and_check_solenoids.py's beta_x-weighted
d^2Bx/dx^2 diagnostic (see its final plot/printout). d^2Bx/dx^2 -- the
order-2 ("sextupole-like") transverse-x Hermite row of the installed
SplineBoris main solenoid -- is antisymmetric about the IP (s=0), which is
why its plain cumulative integral cancels between the two coil ends while a
beta_x-weighted (or beta_x^(3/2)-weighted) version need not, since beta_x(s)
is not antisymmetric.

This script builds two copies of the isolated main-solenoid SplineBoris
line:
  - 'nominal'  -- unmodified, same construction as 004a.
  - 'windowed' -- the order-ORDER (default 2) bx/by Hermite row zeroed for
    every slice overlapping [S_MIN, S_MAX] (default 0.7-1.245 m), i.e. the
    sextupole-like content is switched off only in that s-window, everything
    else (bs, lower-order bx/by) left untouched.

It then compares the beta_x- and beta_x^(3/2)-weighted cumulative integrals
of d^ORDER Bx/dx^ORDER between the two, to quantify how much of the total
driving-term integral comes specifically from the [S_MIN, S_MAX] window.
Standalone check -- does not read/write any of the 004_solenoid_lines*.json
/ fccee_z_lcc_*.json pipeline files (same spirit as 006/007/008).
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from spline_boris_setup import (
    assemble_three_solenoid_system,
    build_splineboris_line,
    extract_tapered_field_data,
    sample_splineboris_line,
)
from solenoid_params import (
    COMP_SOLENOID_A,
    COMP_SOLENOID_B0,
    COMP_SOLENOID_DISTANCE_FROM_IP,
    COMP_SOLENOID_LENGTH,
    MAIN_SOLENOID_A,
    MAIN_SOLENOID_B0,
    THETA,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    half_length_for_b0,
    order_tag,
)
from tilted_solenoid import TiltedSolenoid
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


_parser = argparse.ArgumentParser(
    description=(
        'Effect of zeroing the sextupole-like SplineBoris content in an '
        's-window on the beta_x-weighted d^2Bx/dx^2 integral.'))
add_b0_argument(_parser, default=MAIN_SOLENOID_B0)
add_max_order_argument(_parser, default=4)
_parser.add_argument(
    '--s-min', type=float, default=0.7,
    help='Start (m) of the s-window whose sextupole-like content is '
         'zeroed. Default: 0.7.')
_parser.add_argument(
    '--s-max', type=float, default=1.245,
    help='End (m) of the s-window whose sextupole-like content is '
         'zeroed. Default: 1.245.')
_parser.add_argument(
    '--order', type=int, default=2,
    help="Transverse-x Hermite order to zero in the window and to plot "
         "(2 = sextupole-like term, matching 004a's D2BX_DX2_ORDER). "
         'Default: 2.')
_parser.add_argument(
    '--no-show', action='store_true',
    help='Skip the interactive plt.show() at the end (e.g. for batch runs).')
_args = _parser.parse_args()

MAIN_SOLENOID_B0 = _args.b0
FIELD_TAG = field_tag(MAIN_SOLENOID_B0)
MAIN_SOLENOID_HALF_LENGTH = half_length_for_b0(MAIN_SOLENOID_B0)

MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE = _args.max_transverse_order
ORDER_TAG = order_tag(MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE)

ORDER = _args.order
S_MIN = _args.s_min
S_MAX = _args.s_max
assert S_MIN < S_MAX

PARTICLE = 'positron'
ENERGY0 = 45.6e9

# Field-extraction order is kept fixed at 4, same rationale as 004a: cheap,
# one-time, and must stay >= both ORDER and
# MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE.
MAX_TRANSVERSE_DERIVATIVE_ORDER = 4
DERIVATIVE_STEP = 5e-4
SPLINE_INTEGRAL_POINTS = 10
DECREASE_S_POLY_ORDER_WITH_TRANSVERSE_ORDER = True
S_DERIVATIVE_SPLINE_ORDER = 4
MAX_S_DERIVATIVE_PLOT_ORDER = 5
SPLINE_STEPS_PER_POINT = 10
TAPER_LENGTH = 0.15  # m

X_FIELD_COMPARISON = 0.0
Y_FIELD_COMPARISON = 0.0
BETX = 0.09
BETY = 0.0007
BETA_WEIGHT_EXPONENTS = [1.0, 1.5]

MAIN_SOLENOID_S_AXIS = np.linspace(-2.399, 2.399, 201)
COMP_SOLENOID_S_AXIS = np.linspace(-1.0, 1.0, 201)

assert 0 <= MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE <= xt.SplineBoris._SB_MAX_MULTIPOLE_ORDER - 1
assert MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE <= MAX_TRANSVERSE_DERIVATIVE_ORDER
assert ORDER <= MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE, (
    f'--order {ORDER} would zero a Hermite row the installed SplineBoris '
    f'does not carry (--max-transverse-order '
    f'{MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE}).')
assert MAIN_SOLENOID_S_AXIS[0] <= S_MIN and S_MAX <= MAIN_SOLENOID_S_AXIS[-1]


def cumulative_trapezoid_with_zero(y, s):
    """Cumulative trapezoidal integral of y over s, anchored at 0 for s[0]."""
    increments = np.diff(s) * (y[1:] + y[:-1]) / 2.0
    return np.concatenate(([0.0], np.cumsum(increments)))


def annotate_totals(ax, label_nominal, total_nominal, label_windowed,
                     total_windowed):
    """Framed textbox giving the endpoint (total) value of the nominal vs
    windowed cumulative-integral curve pair, top-left of ax."""
    ax.text(
        0.03, 0.97,
        f'total ({label_nominal})  = {total_nominal:+.4e}\n'
        f'total ({label_windowed}) = {total_windowed:+.4e}',
        transform=ax.transAxes, fontsize=8, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))


def zero_transverse_order_in_window(line, s_axis, order, s_min, s_max):
    """Zero the order-`order` bx/by Hermite row (all 5 Spline4 coefficients)
    for every slice of `line` whose [s_axis[ii], s_axis[ii+1]] interval
    overlaps [s_min, s_max]. Mutates `line`'s elements in place. Returns the
    number of slices zeroed and the [s_start, s_end] actually covered
    (rounded out to the nearest slice boundaries, since s_min/s_max need not
    fall exactly on the 200-slice grid)."""
    zeroed_indices = []
    for ii, element in enumerate(line.elements):
        s_start, s_end = s_axis[ii], s_axis[ii + 1]
        if s_end > s_min and s_start < s_max:
            for k in range(xt.SplineBoris._SB_NUM_COEFFS):
                element.bx[order, k] = 0.0
                element.by[order, k] = 0.0
            zeroed_indices.append(ii)
    if not zeroed_indices:
        raise ValueError(
            f'No slices overlap [{s_min}, {s_max}] -- window too narrow '
            f'relative to the slice grid, or outside the s-axis range.')
    covered_s_start = s_axis[zeroed_indices[0]]
    covered_s_end = s_axis[zeroed_indices[-1] + 1]
    return len(zeroed_indices), (covered_s_start, covered_s_end)


def transverse_derivative_on_s(line, field_data, order, derivative_step):
    """Order-`order` d^order Bx / dx^order (and By) sampled finely along
    `line`, by finite-differencing get_field in x around
    (X_FIELD_COMPARISON, Y_FIELD_COMPARISON) -- same technique 004a uses for
    its derivative_comparison_data. Returns (s, d_bx, d_by)."""
    offsets = np.arange(-4, 5)
    coefficients = SolenoidField.finite_difference_coefficients(
        offsets, order)
    bx_at_offsets = []
    by_at_offsets = []
    s_model = None
    for offset in offsets:
        s_curr, bx_curr, by_curr, _ = sample_splineboris_line(
            line=line,
            s0=field_data['s_axis'][0],
            spline_steps_per_point=SPLINE_STEPS_PER_POINT,
            x=X_FIELD_COMPARISON + offset * derivative_step,
            y=Y_FIELD_COMPARISON,
        )
        if s_model is None:
            s_model = s_curr
        bx_at_offsets.append(bx_curr)
        by_at_offsets.append(by_curr)
    bx_at_offsets = np.array(bx_at_offsets)
    by_at_offsets = np.array(by_at_offsets)
    d_bx = (
        np.tensordot(coefficients, bx_at_offsets, axes=(0, 0))
        / derivative_step**order)
    d_by = (
        np.tensordot(coefficients, by_at_offsets, axes=(0, 0))
        / derivative_step**order)
    return s_model, d_bx, d_by


# Build the two physical field models and extract the tapered field data
# (identical construction to 004a).
main_field_model = TiltedSolenoid(
    L=MAIN_SOLENOID_HALF_LENGTH * 2, a=MAIN_SOLENOID_A, B0=MAIN_SOLENOID_B0,
    theta=THETA)
comp_field_model = SolenoidField(
    L=COMP_SOLENOID_LENGTH, a=COMP_SOLENOID_A, B0=COMP_SOLENOID_B0, z0=0.0)

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

main_field_data = extract_tapered_field_data(
    name='main_solenoid',
    field_model=main_field_model,
    s_axis=MAIN_SOLENOID_S_AXIS,
    **field_extraction_kwargs)
comp_field_data = extract_tapered_field_data(
    name='compensation_solenoid',
    field_model=comp_field_model,
    s_axis=COMP_SOLENOID_S_AXIS,
    **field_extraction_kwargs)

main_bs_integral = np.trapezoid(
    main_field_data['bs'], main_field_data['s_axis'])
comp_bs_integral_unscaled = np.trapezoid(
    comp_field_data['bs'], comp_field_data['s_axis'])
comp_scale_b = -main_bs_integral / comp_bs_integral_unscaled / 2.0

particle_ref = xt.Particles(PARTICLE, energy0=ENERGY0)

splineboris_build_kwargs = {
    'max_transverse_derivative_order_for_spline': (
        MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE),
    'spline_steps_per_point': SPLINE_STEPS_PER_POINT,
    'use_near_axis_simplified_model': False,
}

# Two independent builds of the main-solenoid SplineBoris line: 'nominal'
# stays untouched, 'windowed' gets its order-ORDER bx/by Hermite row zeroed
# in [S_MIN, S_MAX]. Building twice (rather than copying) keeps their
# elements fully independent.
line_main_nominal = build_splineboris_line(
    name='main_solenoid', field_data=main_field_data, scale_b=1.0,
    **splineboris_build_kwargs)
line_main_windowed = build_splineboris_line(
    name='main_solenoid', field_data=main_field_data, scale_b=1.0,
    **splineboris_build_kwargs)

n_zeroed, (covered_s_start, covered_s_end) = zero_transverse_order_in_window(
    line_main_windowed, main_field_data['s_axis'], ORDER, S_MIN, S_MAX)

# Three-solenoid assembly (nominal main only) purely to get a realistic
# beta_x(s) profile around the IP for the weighting below -- sextupole-like
# content does not affect the linear R-matrix / twiss (a pure order>=2 term
# is nonlinear in x and drops out of the on-axis linear map), so it is safe
# to reuse this one nominal twiss for weighting both the nominal and
# windowed curves.
spline_comp_left = build_splineboris_line(
    name='spline_comp_left', field_data=comp_field_data,
    scale_b=comp_scale_b, **splineboris_build_kwargs)
spline_comp_right = build_splineboris_line(
    name='spline_comp_right', field_data=comp_field_data,
    scale_b=comp_scale_b, **splineboris_build_kwargs)
main_half_length = (
    MAIN_SOLENOID_S_AXIS[-1] - MAIN_SOLENOID_S_AXIS[0]) / 2.0
drift_between_comp_and_main = (
    COMP_SOLENOID_DISTANCE_FROM_IP - main_half_length)
line_system_nominal = assemble_three_solenoid_system(
    line_comp_left=spline_comp_left,
    line_main=line_main_nominal,
    line_comp_right=spline_comp_right,
    drift_between_comp_and_main=drift_between_comp_and_main)
line_system_nominal.particle_ref = particle_ref.copy()
tw = line_system_nominal.twiss(init_at='ip', betx=BETX, bety=BETY)
s_from_ip = tw.s - tw['s', 'ip']

# Order-ORDER d^ORDER Bx/dx^ORDER sampled finely along both lines.
s_nominal, d_bx_nominal, _ = transverse_derivative_on_s(
    line_main_nominal, main_field_data, ORDER, DERIVATIVE_STEP)
s_windowed, d_bx_windowed, _ = transverse_derivative_on_s(
    line_main_windowed, main_field_data, ORDER, DERIVATIVE_STEP)

betx_on_nominal = np.interp(s_nominal, s_from_ip, tw.betx)
betx_on_windowed = np.interp(s_windowed, s_from_ip, tw.betx)

cum_plain_nominal = cumulative_trapezoid_with_zero(d_bx_nominal, s_nominal)
cum_plain_windowed = cumulative_trapezoid_with_zero(
    d_bx_windowed, s_windowed)

cum_weighted_nominal = {}
cum_weighted_windowed = {}
for exponent in BETA_WEIGHT_EXPONENTS:
    cum_weighted_nominal[exponent] = cumulative_trapezoid_with_zero(
        betx_on_nominal**exponent * d_bx_nominal, s_nominal)
    cum_weighted_windowed[exponent] = cumulative_trapezoid_with_zero(
        betx_on_windowed**exponent * d_bx_windowed, s_windowed)

# ---------------------------- Plot ---------------------------- #
n_panels = 3 + len(BETA_WEIGHT_EXPONENTS)
fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.2 * n_panels),
                          sharex=True, num=1600)

axes[0].plot(s_from_ip, tw.betx, color='0.3')
axes[0].set_ylabel(r'$\beta_x$ [m]')

order_label = (
    rf'$\partial_x^{{{ORDER}}} B_x$' if ORDER != 1 else r'$\partial_x B_x$')
if ORDER == 0:
    order_unit = 'T'
elif ORDER == 1:
    order_unit = 'T/m'
else:
    order_unit = rf'T/m$^{{{ORDER}}}$'

axes[1].plot(s_nominal, d_bx_nominal, '-', label='nominal')
axes[1].plot(s_windowed, d_bx_windowed, '--', label='windowed')
axes[1].set_ylabel(f'{order_label} [{order_unit}]')
axes[1].legend(loc='best')

axes[2].plot(s_nominal, cum_plain_nominal, '-', label='nominal')
axes[2].plot(s_windowed, cum_plain_windowed, '--', label='windowed')
axes[2].set_ylabel(rf'$\int${order_label}$\, ds$ [T/m]')
axes[2].legend(loc='best')
annotate_totals(
    axes[2], 'nominal', cum_plain_nominal[-1],
    'windowed', cum_plain_windowed[-1])

for panel_index, exponent in enumerate(BETA_WEIGHT_EXPONENTS, start=3):
    ax = axes[panel_index]
    ax.plot(s_nominal, cum_weighted_nominal[exponent], '-', label='nominal')
    ax.plot(s_windowed, cum_weighted_windowed[exponent], '--',
            label='windowed')
    weight_label = (
        r'\beta_x' if exponent == 1.0 else rf'\beta_x^{{{exponent:g}}}')
    ax.set_ylabel(rf'$\int {weight_label}\,${order_label}$\, ds$')
    ax.legend(loc='best')
    annotate_totals(
        ax, 'nominal', cum_weighted_nominal[exponent][-1],
        'windowed', cum_weighted_windowed[exponent][-1])

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.axvspan(covered_s_start, covered_s_end, color='tab:red', alpha=0.08)
axes[-1].set_xlabel(r'$s - s_{\mathrm{ip}}$ [m]')
axes[0].set_xlim(s_nominal[0], s_nominal[-1])
fig.suptitle(
    rf'Effect of zeroing order-{ORDER} $B_x$/$B_y$ content in '
    rf'$s\in[{covered_s_start:.3f}, {covered_s_end:.3f}]$ m '
    rf'(requested $[{S_MIN:g}, {S_MAX:g}]$ m) on '
    rf'$\beta_x$-weighted {order_label} integrals')
fig.tight_layout()

print('004e sextupole-window study')
print(
    f'  b0 = {MAIN_SOLENOID_B0:g} T ({FIELD_TAG}), order = {ORDER}, '
    f'max_transverse_order = {MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE}'
    f'{ORDER_TAG}')
print(
    f'  requested window s in [{S_MIN:g}, {S_MAX:g}] m -> '
    f'{n_zeroed} slices zeroed, covering '
    f's in [{covered_s_start:.4f}, {covered_s_end:.4f}] m')
print(f'  Plain integral of d^{ORDER} Bx/dx^{ORDER} ds (T/m):')
print(f'    nominal  = {cum_plain_nominal[-1]:+.6e}')
print(f'    windowed = {cum_plain_windowed[-1]:+.6e}')
for exponent in BETA_WEIGHT_EXPONENTS:
    print(
        f'  beta_x^{exponent:g}-weighted integral of '
        f'd^{ORDER} Bx/dx^{ORDER} ds:')
    print(f'    nominal  = {cum_weighted_nominal[exponent][-1]:+.6e}')
    print(f'    windowed = {cum_weighted_windowed[exponent][-1]:+.6e}')

if not _args.no_show:
    plt.show()
