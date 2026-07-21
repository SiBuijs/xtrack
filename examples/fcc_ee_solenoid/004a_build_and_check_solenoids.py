from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from spline_boris_setup import (
    assemble_three_solenoid_system,
    build_splineboris_line,
    build_variable_solenoid_line,
    extract_tapered_field_data,
    sample_splineboris_line,
    sample_splineboris_line_on_s,
    smooth_edge_taper,
    symplectic_error,
)
from solenoid_params import (
    COMP_SOLENOID_A,
    COMP_SOLENOID_B0,
    COMP_SOLENOID_DISTANCE_FROM_IP,
    COMP_SOLENOID_LENGTH,
    FIELD_TAG,
    MAIN_SOLENOID_A,
    MAIN_SOLENOID_B0,
    MAIN_SOLENOID_HALF_LENGTH,
    THETA,
)
from tilted_solenoid import TiltedSolenoid
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack.beam_elements.splineboris_src.spline_B_field_eval_python import (
    hermite_to_polynomial,
)


HERE = Path(__file__).parent
OUTPUT_LINES_JSON = HERE / f'004_solenoid_lines_{FIELD_TAG}.json'

PARTICLE = 'positron'
ENERGY0 = 45.6e9

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
SAVE_SOLENOID_LINES_JSON = True
X_FIELD_COMPARISON = 0
Y_FIELD_COMPARISON = 0
PLOT_MAIN_SOLENOID = True
PLOT_COMPENSATION_SOLENOID = False

SEXTUPOLE_AMPLIFICATION_FACTOR = 1.0
# Sextupole amplification is applied at study time via the runtime sext_amp knob
# on the installed SplineBoris lattice (004b), not at template build time.

MIXED_DERIVATIVE_SPECS = [
    {
        'component': 'bx',
        'transverse_direction': 'x',
        'transverse_order': 2,
        's_order': 2,
    },
]

# Transverse-x derivative order for the d^2 Bx / dx^2 integral-over-s printout
# further below.
D2BX_DX2_ORDER = 2

BETX = 0.09
BETY = 0.0007

MAIN_SOLENOID_S_AXIS = np.linspace(-2.399, 2.399, 201)
COMP_SOLENOID_S_AXIS = np.linspace(-1.0, 1.0, 201)

assert MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE <= xt.SplineBoris._SB_MAX_MULTIPOLE_ORDER - 1
assert MAX_TRANSVERSE_DERIVATIVE_ORDER <= 5
assert D2BX_DX2_ORDER <= MAX_TRANSVERSE_DERIVATIVE_ORDER
assert D2BX_DX2_ORDER <= MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE
assert S_DERIVATIVE_SPLINE_ORDER == 4
assert MAX_S_DERIVATIVE_PLOT_ORDER <= 5
for spec in MIXED_DERIVATIVE_SPECS:
    assert spec['component'] in ('bx', 'by', 'bs')
    assert spec['transverse_direction'] in ('x', 'y')
    assert spec['transverse_order'] >= 0
    assert spec['s_order'] >= 0


FIELD_COMPONENT_SYMBOL = {'bs': 'B_s', 'bx': 'B_x', 'by': 'B_y'}


def latex_symbol(component):
    r"""LaTeX-wrapped symbol for a field component, e.g. 'bx' -> '$B_x$'."""
    return f'${FIELD_COMPONENT_SYMBOL[component]}$'


def _latex_partial(component, order, variable):
    r"""LaTeX (no $) for d^order component / d(variable)^order, e.g.
    _latex_partial('bx', 2, 'x') -> r'\partial_x^2 B_x'."""
    symbol = FIELD_COMPONENT_SYMBOL[component]
    if order == 0:
        return symbol
    if order == 1:
        return rf'\partial_{variable} {symbol}'
    return rf'\partial_{variable}^{order} {symbol}'


def latex_derivative_label(component, order, variable):
    r"""LaTeX label ($...$) for d^order component / d(variable)^order, e.g.
    latex_derivative_label('bx', 2, 'x') -> r'$\partial_x^2 B_x$'."""
    return f'${_latex_partial(component, order, variable)}$'


def latex_cumulative_integral_label(
        component, order, variable, unit=None, weight=None):
    r"""LaTeX label for the cumulative integral over s of d^order component /
    d(variable)^order, optionally weighted by another quantity (pass e.g.
    weight=r'\beta_x'), e.g. latex_cumulative_integral_label('bx', 2, 'x')
    -> r'$\int \partial_x^2 B_x\, ds$'."""
    integrand = _latex_partial(component, order, variable)
    if weight:
        integrand = f'{weight}\\, {integrand}'
    label = rf'$\int {integrand}\, ds$'
    if unit:
        label += f' [{unit}]'
    return label


def _latex_mixed_partial(
        component, transverse_direction, transverse_order, s_order):
    r"""LaTeX (no $) for a mixed partial derivative combining a transverse
    order and an s order, e.g. _latex_mixed_partial('bx', 'x', 2, 2) ->
    r'\partial_x^2 \partial_s^2 B_x'."""
    symbol = FIELD_COMPONENT_SYMBOL[component]
    parts = []
    for variable, order in (
            (transverse_direction, transverse_order), ('s', s_order)):
        if order == 0:
            continue
        elif order == 1:
            parts.append(rf'\partial_{variable}')
        else:
            parts.append(rf'\partial_{variable}^{order}')
    return f'{" ".join(parts)} {symbol}'.strip()


def latex_mixed_derivative_label(
        component, transverse_direction, transverse_order, s_order):
    r"""LaTeX label ($...$) for a mixed derivative, e.g.
    latex_mixed_derivative_label('bx', 'x', 2, 2) ->
    r'$\partial_x^2 \partial_s^2 B_x$'."""
    partial = _latex_mixed_partial(
        component, transverse_direction, transverse_order, s_order)
    return f'${partial}$'


def latex_mixed_cumulative_integral_label(
        component, transverse_direction, transverse_order, s_order):
    r"""LaTeX label for the cumulative integral over s of a mixed derivative,
    e.g. latex_mixed_cumulative_integral_label('bx', 'x', 2, 2) ->
    r'$\int \partial_x^2 \partial_s^2 B_x\, ds$'."""
    integrand = _latex_mixed_partial(
        component, transverse_direction, transverse_order, s_order)
    return rf'$\int {integrand}\, ds$'


def cumulative_trapezoid_with_zero(y, s):
    """Cumulative trapezoidal integral of y over s, anchored at 0 for s[0]."""
    increments = np.diff(s) * (y[1:] + y[:-1]) / 2.0
    return np.concatenate(([0.0], np.cumsum(increments)))


def plot_cumulative_integral_grid(fig_num, nrows, ncols, panels, suptitle):
    """Plot cumulative-over-s integrals of field-map vs SplineBoris curves.

    Each panel dict needs 's_field'/'y_field'/'s_model'/'y_model'/'ylabel'
    and optionally 'title'. Each subplot is annotated with a framed textbox
    giving the total (endpoint) integral for both curves.
    """
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False,
        num=fig_num)
    axes_flat = axes.ravel()
    for ax, panel in zip(axes_flat, panels):
        cum_field = cumulative_trapezoid_with_zero(
            panel['y_field'], panel['s_field'])
        cum_model = cumulative_trapezoid_with_zero(
            panel['y_model'], panel['s_model'])
        ax.plot(panel['s_field'], cum_field, '-', label='field-map data')
        ax.plot(panel['s_model'], cum_model, '--', label='SplineBoris')
        ax.set_ylabel(panel['ylabel'])
        ax.set_xlabel(r'$s$ [m]')
        if panel.get('title'):
            ax.set_title(panel['title'])
        ax.grid(True, alpha=0.3)
        ax.text(
            0.03, 0.97,
            f'total (field-map)   = {cum_field[-1]:+.4e}\n'
            f'total (SplineBoris) = {cum_model[-1]:+.4e}',
            transform=ax.transAxes, fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)
    axes_flat[0].legend(loc='lower right')
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def annotate_totals(ax, total_field, total_model):
    """Framed textbox giving the endpoint (total) value of a field-map vs
    SplineBoris cumulative-integral curve pair, placed in the top-left of ax."""
    ax.text(
        0.03, 0.97,
        f'total (field-map)   = {total_field:+.4e}\n'
        f'total (SplineBoris) = {total_model:+.4e}',
        transform=ax.transAxes, fontsize=8, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))


def select_transverse_field_values(field_data, component, direction):
    """Pick the dict-of-order transverse-derivative array from field_data
    matching one of the (component, direction) pairs used in
    transverse_plot_specs (mirrors that inline selection, for use with a
    separately-built, e.g. widened, field_data)."""
    if direction == 'x':
        if component == 'bs':
            return field_data['bs_x_pure']
        return field_data[component]
    if component == 'bs':
        return field_data['bs_y_pure']
    return field_data[f'{component}_y_pure']


def compute_field_mixed_derivative(
        field_model, s_eval, taper_length, x0, y0, component_index,
        transverse_direction, transverse_order, s_order, derivative_step):
    """Recompute one MIXED_DERIVATIVE_SPECS entry directly from field_model
    on an arbitrary s_eval grid (mirrors the inline finite-difference +
    np.gradient logic in the mixed-derivative plotting loop below, factored
    out so it can also be evaluated on a widened s-axis)."""
    taper = smooth_edge_taper(s_axis=s_eval, taper_length=taper_length)
    offsets = np.arange(-4, 5)
    coefficients = SolenoidField.finite_difference_coefficients(
        offsets, transverse_order)
    values_at_offsets = []
    for offset in offsets:
        if transverse_direction == 'x':
            x_values = np.full_like(s_eval, x0 + offset * derivative_step)
            y_values = np.full_like(s_eval, y0)
        else:
            x_values = np.full_like(s_eval, x0)
            y_values = np.full_like(s_eval, y0 + offset * derivative_step)
        values_at_offsets.append(
            field_model.get_field(x_values, y_values, s_eval)[
                component_index] * taper)
    mixed_derivative = (
        np.tensordot(coefficients, np.array(values_at_offsets), axes=(0, 0))
        / derivative_step**transverse_order
    )
    for _ in range(s_order):
        mixed_derivative = np.gradient(mixed_derivative, s_eval, edge_order=2)
    return mixed_derivative


# Build the two physical solenoid models and extract the tapered field data.
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
rigidity0 = particle_ref.rigidity0[0]


# Build the isolated SplineBoris templates that will be installed later.
splineboris_build_kwargs = {
    'max_transverse_derivative_order_for_spline': (
        MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE),
    'spline_steps_per_point': SPLINE_STEPS_PER_POINT,
    'use_near_axis_simplified_model': USE_NEAR_AXIS_SIMPLIFIED_MODEL,
}

line_main_solenoid = build_splineboris_line(
    name='main_solenoid',
    field_data=main_field_data,
    scale_b=1.0,
    sextupole_amplification_factor=SEXTUPOLE_AMPLIFICATION_FACTOR,
    **splineboris_build_kwargs)
line_compensation_solenoid = build_splineboris_line(
    name='compensation_solenoid',
    field_data=comp_field_data,
    scale_b=comp_scale_b,
    sextupole_amplification_factor=SEXTUPOLE_AMPLIFICATION_FACTOR,
    **splineboris_build_kwargs)
line_main_solenoid_varsol = build_variable_solenoid_line(
    name='main_solenoid',
    field_data=main_field_data,
    scale_b=1.0,
    rigidity0=rigidity0)
line_compensation_solenoid_varsol = build_variable_solenoid_line(
    name='compensation_solenoid',
    field_data=comp_field_data,
    scale_b=comp_scale_b,
    rigidity0=rigidity0)
line_main_solenoid.particle_ref = particle_ref.copy()
line_compensation_solenoid.particle_ref = particle_ref.copy()
line_main_solenoid_varsol.particle_ref = particle_ref.copy()
line_compensation_solenoid_varsol.particle_ref = particle_ref.copy()

if SAVE_SOLENOID_LINES_JSON:
    output_data = {
        'metadata': {
            'max_transverse_derivative_order': (
                MAX_TRANSVERSE_DERIVATIVE_ORDER),
            'max_transverse_derivative_order_for_spline': (
                MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE),
            'decrease_s_poly_order_with_transverse_order': (
                DECREASE_S_POLY_ORDER_WITH_TRANSVERSE_ORDER),
            'spline_s_polynomial_degree_rule': (
                'max(0, 4 - transverse_derivative_order)'
                if DECREASE_S_POLY_ORDER_WITH_TRANSVERSE_ORDER
                else '4'),
            'derivative_step': DERIVATIVE_STEP,
            'spline_integral_points': SPLINE_INTEGRAL_POINTS,
            's_derivative_spline_order': S_DERIVATIVE_SPLINE_ORDER,
            'max_s_derivative_plot_order': MAX_S_DERIVATIVE_PLOT_ORDER,
            'spline_steps_per_point': SPLINE_STEPS_PER_POINT,
            'taper_length': TAPER_LENGTH,
            'use_near_axis_simplified_model': (
                USE_NEAR_AXIS_SIMPLIFIED_MODEL),
            'main_bs_integral': main_bs_integral,
            'comp_bs_integral_unscaled': comp_bs_integral_unscaled,
            'comp_scale_b': comp_scale_b,
            'sextupole_amplification_factor': SEXTUPOLE_AMPLIFICATION_FACTOR,
        },
        'lines': {
            'main_solenoid': line_main_solenoid.to_dict(),
            'compensation_solenoid': line_compensation_solenoid.to_dict(),
            'main_solenoid_varsol': line_main_solenoid_varsol.to_dict(),
            'compensation_solenoid_varsol': (
                line_compensation_solenoid_varsol.to_dict()),
        },
    }
    xt.json.dump(output_data, OUTPUT_LINES_JSON, indent=1)

############################################
# ----------- Checks and plots ----------- #
############################################

# Build a complete local three-solenoid system for checks before FCC install.
main_half_length = (
    MAIN_SOLENOID_S_AXIS[-1] - MAIN_SOLENOID_S_AXIS[0]) / 2.0
drift_between_comp_and_main = (
    COMP_SOLENOID_DISTANCE_FROM_IP - main_half_length)

spline_comp_left = build_splineboris_line(
    name='spline_comp_left',
    field_data=comp_field_data,
    scale_b=comp_scale_b,
    **splineboris_build_kwargs)
spline_main = build_splineboris_line(
    name='spline_main',
    field_data=main_field_data,
    scale_b=1.0,
    **splineboris_build_kwargs)
spline_comp_right = build_splineboris_line(
    name='spline_comp_right',
    field_data=comp_field_data,
    scale_b=comp_scale_b,
    **splineboris_build_kwargs)

varsol_comp_left = build_variable_solenoid_line(
    name='varsol_comp_left',
    field_data=comp_field_data,
    scale_b=comp_scale_b,
    rigidity0=rigidity0)
varsol_main = build_variable_solenoid_line(
    name='varsol_main',
    field_data=main_field_data,
    scale_b=1.0,
    rigidity0=rigidity0)
varsol_comp_right = build_variable_solenoid_line(
    name='varsol_comp_right',
    field_data=comp_field_data,
    scale_b=comp_scale_b,
    rigidity0=rigidity0)

line_systems = {
    'SplineBoris': assemble_three_solenoid_system(
        line_comp_left=spline_comp_left,
        line_main=spline_main,
        line_comp_right=spline_comp_right,
        drift_between_comp_and_main=drift_between_comp_and_main),
    'VariableSolenoid': assemble_three_solenoid_system(
        line_comp_left=varsol_comp_left,
        line_main=varsol_main,
        line_comp_right=varsol_comp_right,
        drift_between_comp_and_main=drift_between_comp_and_main),
}
for line in line_systems.values():
    line.particle_ref = particle_ref.copy()


# Twiss checks for the local three-solenoid systems.
twiss_results = {}
twiss_results['SplineBoris'] = line_systems['SplineBoris'].twiss(
    init_at='ip', betx=BETX, bety=BETY)
twiss_results['VariableSolenoid'] = line_systems['VariableSolenoid'].twiss(
    init_at='ip', betx=BETX, bety=BETY)

main_symplectic_error = symplectic_error(
    line=line_main_solenoid, particle_ref=particle_ref)
comp_symplectic_error = symplectic_error(
    line=line_compensation_solenoid, particle_ref=particle_ref)


# Plots: compare field-map extraction against the built SplineBoris lines.
plt.close('all')

comparison_fields = {
}
if PLOT_MAIN_SOLENOID:
    comparison_fields['main_solenoid'] = {
        'field_data': main_field_data,
        'field_model': main_field_model,
        'line': line_main_solenoid,
        'scale_b': 1.0,
    }
if PLOT_COMPENSATION_SOLENOID:
    comparison_fields['compensation_solenoid'] = {
        'field_data': comp_field_data,
        'field_model': comp_field_model,
        'line': line_compensation_solenoid,
        'scale_b': comp_scale_b,
    }

# # Widened s-axis used ONLY by the cumulative-integral check plots further
# # below (see plot_cumulative_integral_grid calls), to see whether the total
# # integral is still converged within the nominal taper window or keeps
# # growing once you look further out. Does NOT feed main_field_data/
# # comp_field_data or the saved SplineBoris/VariableSolenoid templates in
# # 004_solenoid_lines.json, so it has no effect on the installed lattice.
# # Set CUMULATIVE_INTEGRAL_S_AXIS_SCALE_FACTOR = 1 to remove the widening.
# CUMULATIVE_INTEGRAL_S_AXIS_SCALE_FACTOR = 2
# cumulative_integral_s_axis_by_name = {
#     'main_solenoid': np.linspace(
#         CUMULATIVE_INTEGRAL_S_AXIS_SCALE_FACTOR * MAIN_SOLENOID_S_AXIS[0],
#         CUMULATIVE_INTEGRAL_S_AXIS_SCALE_FACTOR * MAIN_SOLENOID_S_AXIS[-1],
#         len(MAIN_SOLENOID_S_AXIS)),
#     'compensation_solenoid': np.linspace(
#         CUMULATIVE_INTEGRAL_S_AXIS_SCALE_FACTOR * COMP_SOLENOID_S_AXIS[0],
#         CUMULATIVE_INTEGRAL_S_AXIS_SCALE_FACTOR * COMP_SOLENOID_S_AXIS[-1],
#         len(COMP_SOLENOID_S_AXIS)),
# }
# cumulative_integral_field_data_by_name = {
#     name: extract_tapered_field_data(
#         name=name,
#         field_model=item['field_model'],
#         s_axis=cumulative_integral_s_axis_by_name[name],
#         **field_extraction_kwargs)
#     for name, item in comparison_fields.items()
# }

# Integral of d^2 Bx / dx^2 over s in [-2.4, 2.4] m (the full main-solenoid
# s-range), computed two ways: directly from the tapered field-map samples,
# and "straight from SplineBoris" by reading the order-D2BX_DX2_ORDER Hermite
# row (bx[D2BX_DX2_ORDER, :]) of each installed slice element -- this is the
# raw (non-1/n!-normalized) d^2Bx/dx^2(s) polynomial already stored in the
# element, reconstructed and integrated analytically per slice, with no
# finite-differencing in x needed.
d2bx_dx2_integrals = {}
for name, item in comparison_fields.items():
    field_data = item['field_data']
    scale_b = item['scale_b']
    line = item['line']

    field_map_integral = scale_b * np.trapezoid(
        field_data['bx'][D2BX_DX2_ORDER], field_data['s_axis'])

    splineboris_integral = 0.0
    for element in line.elements:
        bx_hermite = np.array(element.bx[D2BX_DX2_ORDER])
        poly = hermite_to_polynomial(0.0, element.length, bx_hermite)
        integral_poly = poly.integ()
        splineboris_integral += element.scale_b * (
            integral_poly(element.length) - integral_poly(0.0))

    d2bx_dx2_integrals[name] = {
        'field_map': field_map_integral,
        'splineboris': splineboris_integral,
    }

# sampled_lines = {}
# for name, item in comparison_fields.items():
#     field_data = item['field_data']
#     s_model, bx_model, by_model, bs_model = sample_splineboris_line(
#         line=item['line'],
#         s0=field_data['s_axis'][0],
#         spline_steps_per_point=SPLINE_STEPS_PER_POINT,
#         x=X_FIELD_COMPARISON,
#         y=Y_FIELD_COMPARISON,
#     )
#     sampled_lines[name] = {
#         's': s_model,
#         'bx': bx_model,
#         'by': by_model,
#         'bs': bs_model,
#     }
# 
# field_components = [
#     (latex_symbol('bs') + ' [T]', 'bs'),
#     (latex_symbol('bx') + ' [T]', 'bx'),
#     (latex_symbol('by') + ' [T]', 'by'),
# ]

# shared_x_axes_by_solenoid = {}

# if comparison_fields:
#     fig_fields, axes_fields = plt.subplots(
#         len(comparison_fields), 3,
#         figsize=(15, 4.0 * len(comparison_fields)),
#         squeeze=False,
#         num=1000,
#     )
# 
#     for row, (name, item) in enumerate(comparison_fields.items()):
#         field_data = item['field_data']
#         scale_b = item['scale_b']
#         field_values = {
#             'bs': scale_b * field_data['bs'],
#             'bx': scale_b * field_data['bx'][0],
#             'by': scale_b * field_data['by'][0],
#         }
#         model_values = sampled_lines[name]
# 
#         for col, (ylabel, component) in enumerate(field_components):
#             ax = axes_fields[row, col]
#             ax.plot(
#                 field_data['s_axis'], field_values[component],
#                 '-', label='field-map data')
#             ax.plot(
#                 model_values['s'], model_values[component],
#                 '--', label='SplineBoris')
#             ax.set_ylabel(f'{name}\n{ylabel}')
#             ax.grid(True, alpha=0.3)
#             if row == 0 and col == 0:
#                 ax.legend(loc='best')
# 
#         axes_to_share = axes_fields[row, :].ravel()
#         if name not in shared_x_axes_by_solenoid:
#             shared_x_axes_by_solenoid[name] = axes_to_share[0]
#         for ax in axes_to_share:
#             if ax is not shared_x_axes_by_solenoid[name]:
#                 ax.sharex(shared_x_axes_by_solenoid[name])
# 
#     for ax in axes_fields[-1, :]:
#         ax.set_xlabel(r'$s$ [m]')
#     fig_fields.suptitle(
#         r'Tapered field-map data and SplineBoris-line comparison '
#         rf'at $x$={X_FIELD_COMPARISON:g} m, $y$={Y_FIELD_COMPARISON:g} m')
#     fig_fields.tight_layout()

# s_derivative_model_data = {}
# u_integral = np.linspace(-1.0, 1.0, SPLINE_INTEGRAL_POINTS)
# splineboris_poly_order = min(4, SPLINE_INTEGRAL_POINTS - 1)
# max_s_derivative_plot_order = min(
#     MAX_S_DERIVATIVE_PLOT_ORDER,
#     5)
# 
# for name, item in comparison_fields.items():
#     field_data = item['field_data']
#     s_axis = field_data['s_axis']
#     line = item['line']
#     n_intervals = len(line.elements)
# 
#     s_model = np.empty((n_intervals, SPLINE_INTEGRAL_POINTS))
#     bx_model = np.empty_like(s_model)
#     by_model = np.empty_like(s_model)
#     bs_model = np.empty_like(s_model)
# 
#     for ii, element in enumerate(line.elements):
#         length = element.length
#         s_local = np.linspace(0.0, length, SPLINE_INTEGRAL_POINTS)
#         s_model[ii] = s_axis[ii] + s_local
#         bx_model[ii], by_model[ii], bs_model[ii] = element.get_field(
#             np.full_like(s_local, X_FIELD_COMPARISON),
#             np.full_like(s_local, Y_FIELD_COMPARISON),
#             s_local,
#         )
# 
#     s_derivative_model_data[name] = {}
#     model_values = {
#         'bx': bx_model,
#         'by': by_model,
#         'bs': bs_model,
#     }
# 
#     for derivative_order in range(max_s_derivative_plot_order + 1):
#         s_derivative_model_data[name][derivative_order] = {
#             's': s_model.ravel(),
#         }
# 
#         for component, values in model_values.items():
#             derivative_values = np.empty_like(values)
#             for ii, element in enumerate(line.elements):
#                 if derivative_order == 0:
#                     derivative_values[ii] = values[ii]
#                 else:
#                     coefficients = np.polyfit(
#                         u_integral, values[ii], splineboris_poly_order)
#                     derivative_coefficients = np.polyder(
#                         coefficients, derivative_order)
#                     derivative_values[ii] = (
#                         np.polyval(derivative_coefficients, u_integral)
#                         * (2.0 / element.length)**derivative_order)
# 
#             s_derivative_model_data[name][derivative_order][component] = (
#                 derivative_values.ravel())

# for name, item in comparison_fields.items():
#     field_data = item['field_data']
#     scale_b = item['scale_b']
#     figure_number_offset = 300 if name == 'compensation_solenoid' else 200
# 
#     for component_index, component in enumerate(('bx', 'by', 'bs')):
#         label = latex_symbol(component)
#         fig_s_derivatives, axes_s_derivatives = plt.subplots(
#             2, 3, figsize=(15, 8),
#             num=figure_number_offset + 10 * component_index)
#         axes_s_derivatives_flat = axes_s_derivatives.ravel()
#         if name not in shared_x_axes_by_solenoid:
#             shared_x_axes_by_solenoid[name] = axes_s_derivatives_flat[0]
#         for ax in axes_s_derivatives_flat:
#             if ax is not shared_x_axes_by_solenoid[name]:
#                 ax.sharex(shared_x_axes_by_solenoid[name])
# 
#         for derivative_order in range(max_s_derivative_plot_order + 1):
#             ax = axes_s_derivatives_flat[derivative_order]
#             field_plot_data = field_data['s_derivative_plot_data'][
#                 derivative_order]
#             model_plot_data = s_derivative_model_data[name][derivative_order]
#             ax.plot(
#                 field_plot_data['s'],
#                 scale_b * field_plot_data[component],
#                 '-',
#                 label='field-map data')
#             ax.plot(
#                 model_plot_data['s'],
#                 model_plot_data[component],
#                 '--',
#                 label='SplineBoris')
#             ax.set_ylabel(
#                 latex_derivative_label(component, derivative_order, 's'))
#             ax.set_xlabel(r'$s$ [m]')
#             ax.grid(True, alpha=0.3)
#             ax.set_title(f'order {derivative_order}')
# 
#         for ax in axes_s_derivatives_flat[max_s_derivative_plot_order + 1:]:
#             ax.set_visible(False)
# 
#         axes_s_derivatives_flat[0].legend(loc='best')
#         fig_s_derivatives.suptitle(
#             rf'Longitudinal derivative comparison for {name}, '
#             rf'{label} at $x$={X_FIELD_COMPARISON:g} m, '
#             rf'$y$={Y_FIELD_COMPARISON:g} m')
#         fig_s_derivatives.tight_layout()
# 
#         s_derivative_cumulative_panels = []
#         wide_field_data = cumulative_integral_field_data_by_name[name]
#         for derivative_order in range(max_s_derivative_plot_order + 1):
#             wide_plot_data = wide_field_data['s_derivative_plot_data'][
#                 derivative_order]
#             model_plot_data = s_derivative_model_data[name][derivative_order]
#             cumulative_ylabel = latex_cumulative_integral_label(
#                 component, derivative_order, 's',
#                 unit='T m' if derivative_order == 0 else None)
#             s_derivative_cumulative_panels.append({
#                 's_field': wide_plot_data['s'],
#                 'y_field': scale_b * wide_plot_data[component],
#                 's_model': model_plot_data['s'],
#                 'y_model': model_plot_data[component],
#                 'ylabel': cumulative_ylabel,
#                 'title': f'order {derivative_order}',
#             })
#         plot_cumulative_integral_grid(
#             fig_num=figure_number_offset + 10 * component_index + 5,
#             nrows=2, ncols=3, panels=s_derivative_cumulative_panels,
#             suptitle=(
#                 rf'Cumulative (over $s$) integral of longitudinal '
#                 rf'derivatives for {name}, {label} at '
#                 rf'$x$={X_FIELD_COMPARISON:g} m, '
#                 rf'$y$={Y_FIELD_COMPARISON:g} m'))

offsets = np.arange(-4, 5)
zero_offset_index = np.where(offsets == 0)[0][0]
derivative_comparison_data = {}

for name, item in comparison_fields.items():
    field_data = item['field_data']
    bx_at_offsets = []
    by_at_offsets = []
    bs_at_offsets = []
    s_model = None

    for offset in offsets:
        s_curr, bx_curr, by_curr, bs_curr = sample_splineboris_line(
            line=item['line'],
            s0=field_data['s_axis'][0],
            spline_steps_per_point=SPLINE_STEPS_PER_POINT,
            x=X_FIELD_COMPARISON + offset * DERIVATIVE_STEP,
            y=Y_FIELD_COMPARISON,
        )
        if s_model is None:
            s_model = s_curr
        bx_at_offsets.append(bx_curr)
        by_at_offsets.append(by_curr)
        bs_at_offsets.append(bs_curr)

    bx_at_offsets = np.array(bx_at_offsets)
    by_at_offsets = np.array(by_at_offsets)
    bs_at_offsets = np.array(bs_at_offsets)

    derivative_comparison_data[name] = {
        0: {
            's': s_model,
            'bx': bx_at_offsets[zero_offset_index],
            'by': by_at_offsets[zero_offset_index],
            'bs': bs_at_offsets[zero_offset_index],
        }
    }

    for order in range(1, MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
        coefficients = SolenoidField.finite_difference_coefficients(
            offsets, order)
        derivative_comparison_data[name][order] = {
            's': s_model,
            'bx': (
                np.tensordot(coefficients, bx_at_offsets, axes=(0, 0))
                / DERIVATIVE_STEP**order
            ),
            'by': (
                np.tensordot(coefficients, by_at_offsets, axes=(0, 0))
                / DERIVATIVE_STEP**order
            ),
            'bs': (
                np.tensordot(coefficients, bs_at_offsets, axes=(0, 0))
                / DERIVATIVE_STEP**order
            ),
        }

# vertical_derivative_comparison_data = {}
# 
# for name, item in comparison_fields.items():
#     field_data = item['field_data']
#     bx_at_offsets = []
#     by_at_offsets = []
#     bs_at_offsets = []
#     s_model = None
# 
#     for offset in offsets:
#         s_curr, bx_curr, by_curr, bs_curr = sample_splineboris_line(
#             line=item['line'],
#             s0=field_data['s_axis'][0],
#             spline_steps_per_point=SPLINE_STEPS_PER_POINT,
#             x=X_FIELD_COMPARISON,
#             y=Y_FIELD_COMPARISON + offset * DERIVATIVE_STEP,
#         )
#         if s_model is None:
#             s_model = s_curr
#         bx_at_offsets.append(bx_curr)
#         by_at_offsets.append(by_curr)
#         bs_at_offsets.append(bs_curr)
# 
#     bx_at_offsets = np.array(bx_at_offsets)
#     by_at_offsets = np.array(by_at_offsets)
#     bs_at_offsets = np.array(bs_at_offsets)
# 
#     vertical_derivative_comparison_data[name] = {
#         0: {
#             's': s_model,
#             'bx': bx_at_offsets[zero_offset_index],
#             'by': by_at_offsets[zero_offset_index],
#             'bs': bs_at_offsets[zero_offset_index],
#         }
#     }
# 
#     for order in range(1, MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
#         coefficients = SolenoidField.finite_difference_coefficients(
#             offsets, order)
#         vertical_derivative_comparison_data[name][order] = {
#             's': s_model,
#             'bx': (
#                 np.tensordot(coefficients, bx_at_offsets, axes=(0, 0))
#                 / DERIVATIVE_STEP**order
#             ),
#             'by': (
#                 np.tensordot(coefficients, by_at_offsets, axes=(0, 0))
#                 / DERIVATIVE_STEP**order
#             ),
#             'bs': (
#                 np.tensordot(coefficients, bs_at_offsets, axes=(0, 0))
#                 / DERIVATIVE_STEP**order
#             ),
#         }

# for name, item in comparison_fields.items():
#     field_data = item['field_data']
#     scale_b = item['scale_b']
#     figure_number_offset = 700 if name == 'compensation_solenoid' else 600
# 
#     transverse_plot_specs = [
#         ('bx', 'x', field_data['bx'], derivative_comparison_data[name]),
#         (
#             'bx', 'y',
#             field_data['bx_y_pure'],
#             vertical_derivative_comparison_data[name],
#         ),
#         ('by', 'x', field_data['by'], derivative_comparison_data[name]),
#         (
#             'by', 'y',
#             field_data['by_y_pure'],
#             vertical_derivative_comparison_data[name],
#         ),
#         (
#             'bs', 'x',
#             field_data['bs_x_pure'],
#             derivative_comparison_data[name],
#         ),
#         (
#             'bs', 'y',
#             field_data['bs_y_pure'],
#             vertical_derivative_comparison_data[name],
#         ),
#     ]
# 
#     for plot_index, (
#             component, direction, field_values_by_order,
#             model_data_by_order) in enumerate(transverse_plot_specs):
#         fig_derivatives, axes_derivatives = plt.subplots(
#             2, 3, figsize=(15, 8),
#             num=figure_number_offset + plot_index)
#         axes_derivatives_flat = axes_derivatives.ravel()
#         if name not in shared_x_axes_by_solenoid:
#             shared_x_axes_by_solenoid[name] = axes_derivatives_flat[0]
#         for ax in axes_derivatives_flat:
#             if ax is not shared_x_axes_by_solenoid[name]:
#                 ax.sharex(shared_x_axes_by_solenoid[name])
# 
#         for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
#             ax = axes_derivatives_flat[order]
#             model_data = model_data_by_order[order]
#             ax.plot(
#                 field_data['s_axis'],
#                 scale_b * field_values_by_order[order],
#                 '-',
#                 label='field-map data')
#             ax.plot(
#                 model_data['s'],
#                 model_data[component],
#                 '--',
#                 label='SplineBoris')
#             ax.set_ylabel(latex_derivative_label(component, order, direction))
#             ax.set_xlabel(r'$s$ [m]')
#             ax.grid(True, alpha=0.3)
#             ax.set_title(f'order {order}')
# 
#         for ax in axes_derivatives_flat[
#                 MAX_TRANSVERSE_DERIVATIVE_ORDER + 1:]:
#             ax.set_visible(False)
# 
#         axes_derivatives_flat[0].legend(loc='best')
#         fig_derivatives.suptitle(
#             rf'Tapered transverse derivative comparison for {name}: '
#             rf'{latex_symbol(component)} vs ${direction}$ at '
#             rf'$x$={X_FIELD_COMPARISON:g} m, $y$={Y_FIELD_COMPARISON:g} m')
#         fig_derivatives.tight_layout()
# 
#         transverse_cumulative_panels = []
#         wide_field_data = cumulative_integral_field_data_by_name[name]
#         wide_field_values_by_order = select_transverse_field_values(
#             wide_field_data, component, direction)
#         for order in range(MAX_TRANSVERSE_DERIVATIVE_ORDER + 1):
#             model_data = model_data_by_order[order]
#             cumulative_ylabel = latex_cumulative_integral_label(
#                 component, order, direction,
#                 unit='T m' if order == 0 else None)
#             transverse_cumulative_panels.append({
#                 's_field': wide_field_data['s_axis'],
#                 'y_field': scale_b * wide_field_values_by_order[order],
#                 's_model': model_data['s'],
#                 'y_model': model_data[component],
#                 'ylabel': cumulative_ylabel,
#                 'title': f'order {order}',
#             })
#         plot_cumulative_integral_grid(
#             fig_num=figure_number_offset + plot_index + 10,
#             nrows=2, ncols=3, panels=transverse_cumulative_panels,
#             suptitle=(
#                 rf'Cumulative (over $s$) integral of tapered transverse '
#                 rf'derivatives for {name}: {latex_symbol(component)} vs '
#                 rf'${direction}$ at $x$={X_FIELD_COMPARISON:g} m, '
#                 rf'$y$={Y_FIELD_COMPARISON:g} m'))

# mixed_component_index = {'bx': 0, 'by': 1, 'bs': 2}
# 
# for name, item in comparison_fields.items():
#     field_data = item['field_data']
#     field_model = item['field_model']
#     s_axis = field_data['s_axis']
#     scale_b = item['scale_b']
#     figure_number_offset = 1400 if name == 'compensation_solenoid' else 1300
#     s_mixed = field_data['s_derivative_plot_data'][0]['s']
#     taper_mixed = smooth_edge_taper(
#         s_axis=s_mixed, taper_length=TAPER_LENGTH)
# 
#     for spec_index, spec in enumerate(MIXED_DERIVATIVE_SPECS):
#         component = spec['component']
#         transverse_direction = spec['transverse_direction']
#         transverse_order = spec['transverse_order']
#         s_order = spec['s_order']
# 
#         offsets = np.arange(-4, 5)
#         coefficients = SolenoidField.finite_difference_coefficients(
#             offsets, transverse_order)
#         field_values_at_offsets = []
#         model_values_at_offsets = []
# 
#         for offset in offsets:
#             if transverse_direction == 'x':
#                 x_values = np.full_like(
#                     s_mixed,
#                     X_FIELD_COMPARISON + offset * DERIVATIVE_STEP)
#                 y_values = np.full_like(s_mixed, Y_FIELD_COMPARISON)
#             elif transverse_direction == 'y':
#                 x_values = np.full_like(s_mixed, X_FIELD_COMPARISON)
#                 y_values = np.full_like(
#                     s_mixed,
#                     Y_FIELD_COMPARISON + offset * DERIVATIVE_STEP)
#             else:
#                 raise ValueError(
#                     "transverse_direction must be either 'x' or 'y'")
# 
#             field_values_at_offsets.append(
#                 field_model.get_field(
#                     x_values, y_values, s_mixed
#                 )[mixed_component_index[component]] * taper_mixed)
#             model_values_at_offsets.append(
#                 sample_splineboris_line_on_s(
#                     line=item['line'],
#                     s_axis=s_axis,
#                     s_eval=s_mixed,
#                     x=x_values[0], y=y_values[0],
#                 )[mixed_component_index[component]])
# 
#         field_mixed_derivative = (
#             np.tensordot(
#                 coefficients,
#                 np.array(field_values_at_offsets),
#                 axes=(0, 0))
#             / DERIVATIVE_STEP**transverse_order
#         )
#         model_mixed_derivative = (
#             np.tensordot(
#                 coefficients,
#                 np.array(model_values_at_offsets),
#                 axes=(0, 0))
#             / DERIVATIVE_STEP**transverse_order
#         )
# 
#         for _ in range(s_order):
#             field_mixed_derivative = np.gradient(
#                 field_mixed_derivative, s_mixed, edge_order=2)
#             model_mixed_derivative = np.gradient(
#                 model_mixed_derivative, s_mixed, edge_order=2)
# 
#         fig_mixed, ax_mixed = plt.subplots(
#             1, 1, figsize=(10, 5),
#             num=figure_number_offset + spec_index)
#         if name in shared_x_axes_by_solenoid:
#             ax_mixed.sharex(shared_x_axes_by_solenoid[name])
#         else:
#             shared_x_axes_by_solenoid[name] = ax_mixed
# 
#         ax_mixed.plot(
#             s_mixed,
#             scale_b * field_mixed_derivative,
#             '-',
#             label='field-map data')
#         ax_mixed.plot(
#             s_mixed,
#             model_mixed_derivative,
#             '--',
#             label='SplineBoris')
#         mixed_label = latex_mixed_derivative_label(
#             component, transverse_direction, transverse_order, s_order)
#         ax_mixed.set_ylabel(mixed_label)
#         ax_mixed.set_xlabel(r'$s$ [m]')
#         ax_mixed.grid(True, alpha=0.3)
#         ax_mixed.legend(loc='best')
#         fig_mixed.suptitle(
#             rf'Mixed derivative comparison for {name}: {mixed_label}')
#         fig_mixed.tight_layout()
# 
#         wide_field_data = cumulative_integral_field_data_by_name[name]
#         s_mixed_wide = wide_field_data['s_derivative_plot_data'][0]['s']
#         field_mixed_derivative_wide = compute_field_mixed_derivative(
#             field_model=field_model,
#             s_eval=s_mixed_wide,
#             taper_length=TAPER_LENGTH,
#             x0=X_FIELD_COMPARISON, y0=Y_FIELD_COMPARISON,
#             component_index=mixed_component_index[component],
#             transverse_direction=transverse_direction,
#             transverse_order=transverse_order,
#             s_order=s_order,
#             derivative_step=DERIVATIVE_STEP)
# 
#         mixed_cumulative_label = latex_mixed_cumulative_integral_label(
#             component, transverse_direction, transverse_order, s_order)
#         mixed_cumulative_panel = {
#             's_field': s_mixed_wide,
#             'y_field': scale_b * field_mixed_derivative_wide,
#             's_model': s_mixed,
#             'y_model': model_mixed_derivative,
#             'ylabel': mixed_cumulative_label,
#         }
#         plot_cumulative_integral_grid(
#             fig_num=figure_number_offset + spec_index + 50,
#             nrows=1, ncols=1, panels=[mixed_cumulative_panel],
#             suptitle=(
#                 rf'Cumulative (over $s$) integral of mixed derivative for '
#                 rf'{name}: {mixed_cumulative_label}'))


# Plots: local three-solenoid checks. (Commented out for now to focus on
# the beta_x / d^2 Bx/dx^2 figure below; uncomment to bring this back.)
# fig_orbit_coupling, axes_orbit_coupling = plt.subplots(
#     2, 2, figsize=(12, 7), sharex=True, num=1001)
# for name, tw in twiss_results.items():
#     s_from_ip = tw.s - tw['s', 'ip']
#     axes_orbit_coupling[0, 0].plot(s_from_ip, tw.x, label=name)
#     axes_orbit_coupling[1, 0].plot(s_from_ip, tw.y, label=name)
#     axes_orbit_coupling[0, 1].plot(s_from_ip, tw.betx2, label=name)
#     axes_orbit_coupling[1, 1].plot(s_from_ip, tw.bety1, label=name)
# axes_orbit_coupling[0, 0].set_ylabel(r'$x$ [m]')
# axes_orbit_coupling[1, 0].set_ylabel(r'$y$ [m]')
# axes_orbit_coupling[0, 1].set_ylabel(r'$\beta_{x,2}$ [m]')
# axes_orbit_coupling[1, 1].set_ylabel(r'$\beta_{y,1}$ [m]')
# axes_orbit_coupling[1, 0].set_xlabel(r'$s - s_{\mathrm{ip}}$ [m]')
# axes_orbit_coupling[1, 1].set_xlabel(r'$s - s_{\mathrm{ip}}$ [m]')
# for ax in axes_orbit_coupling.ravel():
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='best')
# fig_orbit_coupling.suptitle(
#     r'Three-solenoid orbit and $\beta$ coupling, initialized at IP')
# fig_orbit_coupling.tight_layout()


# Plot: beta_x, cumulative integral of d^2 Bx/dx^2, and the cumulative
# integral of beta_x * d^2 Bx/dx^2 ds (i.e. the beta_x-weighted running
# integral of the same order-D2BX_DX2_ORDER transverse derivative), all
# sharing the s-axis of the main solenoid, computed both from the field-map
# data and from the installed SplineBoris line.
for name, item in comparison_fields.items():
    field_data = item['field_data']
    scale_b = item['scale_b']
    figure_number = 1550 if name == 'compensation_solenoid' else 1500

    s_field = field_data['s_axis']
    d2bx_dx2_field = scale_b * field_data['bx'][D2BX_DX2_ORDER]
    s_model = derivative_comparison_data[name][D2BX_DX2_ORDER]['s']
    d2bx_dx2_model = derivative_comparison_data[name][D2BX_DX2_ORDER]['bx']

    tw_spb = twiss_results['SplineBoris']
    s_from_ip_spb = tw_spb.s - tw_spb['s', 'ip']
    betx_on_field_axis = np.interp(s_field, s_from_ip_spb, tw_spb.betx)
    betx_on_model_axis = np.interp(s_model, s_from_ip_spb, tw_spb.betx)

    cum_d2bx_dx2_field = cumulative_trapezoid_with_zero(
        d2bx_dx2_field, s_field)
    cum_d2bx_dx2_model = cumulative_trapezoid_with_zero(
        d2bx_dx2_model, s_model)

    cum_betx_weighted_field = cumulative_trapezoid_with_zero(
        betx_on_field_axis * d2bx_dx2_field, s_field)
    cum_betx_weighted_model = cumulative_trapezoid_with_zero(
        betx_on_model_axis * d2bx_dx2_model, s_model)

    fig_betx_d2bx, axes_betx_d2bx = plt.subplots(
        3, 1, figsize=(8, 10), sharex=True, num=figure_number)

    for tw_name, tw in twiss_results.items():
        s_from_ip = tw.s - tw['s', 'ip']
        axes_betx_d2bx[0].plot(s_from_ip, tw.betx, label=tw_name)
    axes_betx_d2bx[0].set_ylabel(r'$\beta_x$ [m]')
    axes_betx_d2bx[0].legend(loc='best')

    axes_betx_d2bx[1].plot(s_field, cum_d2bx_dx2_field, '-',
                            label='field-map data')
    axes_betx_d2bx[1].plot(s_model, cum_d2bx_dx2_model, '--',
                            label='SplineBoris')
    axes_betx_d2bx[1].set_ylabel(
        latex_cumulative_integral_label(
            'bx', D2BX_DX2_ORDER, 'x', unit='T/m'))
    axes_betx_d2bx[1].legend(loc='best')
    annotate_totals(
        axes_betx_d2bx[1], cum_d2bx_dx2_field[-1], cum_d2bx_dx2_model[-1])

    axes_betx_d2bx[2].plot(s_field, cum_betx_weighted_field, '-',
                            label='field-map data')
    axes_betx_d2bx[2].plot(s_model, cum_betx_weighted_model, '--',
                            label='SplineBoris')
    axes_betx_d2bx[2].set_ylabel(
        latex_cumulative_integral_label(
            'bx', D2BX_DX2_ORDER, 'x', unit='T', weight=r'\beta_x'))
    axes_betx_d2bx[2].legend(loc='best')
    annotate_totals(
        axes_betx_d2bx[2], cum_betx_weighted_field[-1],
        cum_betx_weighted_model[-1])

    axes_betx_d2bx[-1].set_xlabel(r'$s - s_{\mathrm{ip}}$ [m]')
    axes_betx_d2bx[0].set_xlim(s_field[0], s_field[-1])
    for ax in axes_betx_d2bx:
        ax.grid(True, alpha=0.3)
    fig_betx_d2bx.suptitle(
        rf'$\beta_x$ and cumulative '
        rf'{latex_derivative_label("bx", D2BX_DX2_ORDER, "x")} '
        rf'integrals for {name}')
    fig_betx_d2bx.tight_layout()

print('004a build and check tapered solenoids')
print(f'  output lines json = {OUTPUT_LINES_JSON}')
print(f'  main Bs integral = {main_bs_integral:.12e} T m')
print(
    '  2 x compensation Bs integral = '
    f'{2 * comp_scale_b * comp_bs_integral_unscaled:.12e} T m')

print('  Symplectic checks:')
print(
    '    main SplineBoris line: '
    f'{main_symplectic_error:.12e}')
print(
    '    compensation SplineBoris line: '
    f'{comp_symplectic_error:.12e}')

for label, tw in twiss_results.items():
    print(f'  {label}:')
    print(
        f'    betx2_end = {tw.betx2[-1]:+.12e} m, '
        f'bety1_end = {tw.bety1[-1]:+.12e} m')

print(
    f'  Integral of d^{D2BX_DX2_ORDER} Bx / dx^{D2BX_DX2_ORDER} ds over '
    f's in [{MAIN_SOLENOID_S_AXIS[0]:.3f}, {MAIN_SOLENOID_S_AXIS[-1]:.3f}] m '
    '(units: T/m):')
for name, values in d2bx_dx2_integrals.items():
    print(f'    {name}:')
    print(f'      field-map data = {values["field_map"]:+.12e} T/m')
    print(f'      SplineBoris    = {values["splineboris"]:+.12e} T/m')

plt.show()
