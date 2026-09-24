"""What a built SplineBoris solenoid actually contains, in two figures.

004a builds the solenoid templates and checks them against the analytic field
map from every angle, which costs ~100 figures. This script reads the templates
004a already wrote (004_solenoid_lines_{FIELD_TAG}{ORDER_TAG}.json) and plots
just the content of the elements:

  * the field Bs, Bx, By along s at x = y = 0;
  * one panel per transverse multipole order k >= 1, showing d^k Bx/dx^k and
    d^k By/dx^k, up to the order the elements were actually built with.

One figure per solenoid (main, compensation), so two figures by default.

"At x = y = 0" is the beam axis, deliberately not called "on-axis": the main
solenoid is tilted by theta = -0.015 rad, so its own axis is not the beam's.
That tilt is the entire reason Bx is nonzero there at all.

Where the numbers come from
---------------------------
Each xt.SplineBoris slice stores, per transverse derivative order k, a
five-value Hermite row `bx[k]` = (val_start, der_start, val_end, der_end,
mean) describing d^k Bx/dx^k over that slice, and likewise `by[k]`; `bs` is the
same for the on-axis longitudinal field. Those rows are the *pure x-derivatives
on the axis* -- spline_boris_setup.extract_tapered_field_data fills them from
field_model.compute_pure_field_derivatives(direction='x'), so `bx[2]` really is
d^2 Bx/dx^2 and not a coefficient with a factorial in it. Verified here against
central differences of element.get_field (see CHECK_AGAINST_GET_FIELD).

The five Hermite values define a quartic over the slice, which is what the
Boris pusher integrates. _quartic_from_hermite reproduces it exactly (to 1e-16
against get_field on axis), so the curves below are the field as tracked, not a
node-to-node interpolation of it.

`scale_b` is NOT baked into the stored rows -- get_field applies it. The
compensation solenoid carries scale_b = comp_scale_b ~ -2.6, so forgetting it
would plot a +1 T solenoid where the machine has a -2.6 T one. Everything
plotted here is multiplied by it.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from solenoid_params import (
    MAIN_SOLENOID_B0,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    order_tag,
)


HERE = Path(__file__).parent

SOLENOID_CHOICES = ('main', 'compensation')
SOLENOID_LINE_NAME = {
    'main': 'main_solenoid',
    'compensation': 'compensation_solenoid',
}

# Points per slice at which the stored quartic is evaluated. The slices are
# ~2.4 cm long and the quartic is smooth inside one, so this is about
# resolution of the plotted curve, not accuracy.
POINTS_PER_SLICE = 9

# Cross-check the reconstruction against central differences of get_field at
# orders 1 and 2 (the only ones a difference stencil resolves usefully) and
# print the agreement. Cheap, and it is the check that the rows mean what the
# docstring says they mean.
CHECK_AGAINST_GET_FIELD = True

FIGURE_NUMBER = {'main': 1700, 'compensation': 1750}

# Draw every Nth SplineBoris slice boundary. 1 = all of them, which is what the
# figure is meant to show; both solenoids have 200 slices, so that is a line
# every ~2.4 cm (main) or ~1 cm (compensation) and the result reads as texture
# rather than as individual boundaries. Raise to 5 or 10 for a legible sample of
# the grid instead.
SLICE_LINE_STRIDE = 1


_parser = argparse.ArgumentParser(
    description='Plot the beam-axis field and multipole content of the built '
                'SplineBoris solenoid elements.')
add_b0_argument(_parser, default=MAIN_SOLENOID_B0)
add_max_order_argument(_parser)
_parser.add_argument(
    '--solenoid', default='both', choices=(*SOLENOID_CHOICES, 'both'),
    help='Which solenoid to plot (default: both).')
_parser.add_argument(
    '--save', action='store_true',
    help='Write the figures as PNGs to the shared plot directory.')
_parser.add_argument(
    '--no-show', action='store_true',
    help='Skip the interactive plt.show() at the end (e.g. for batch runs).')
_args = _parser.parse_args()

FIELD_TAG = field_tag(_args.b0)
ORDER_TAG = order_tag(_args.max_transverse_order)
INPUT_SOLENOID_LINES_JSON = (
    HERE / f'004_solenoid_lines_{FIELD_TAG}{ORDER_TAG}.json')

SOLENOIDS = (SOLENOID_CHOICES if _args.solenoid == 'both'
             else (_args.solenoid,))


###############################################################################
# Reading the stored splines                                                  #
###############################################################################

# Hermite -> quartic, once: the three interior coefficients solve a fixed 3x3
# system, so factor it here rather than per slice per order.
_QUARTIC_MATRIX = np.array([
    [1.0, 1.0, 1.0],        # p(1) = v1
    [2.0, 3.0, 4.0],        # p'(1) = d1 * L
    [1 / 3, 1 / 4, 1 / 5],  # mean over the slice
])


def _quartic_from_hermite(hermite, length, u):
    """Evaluate the slice quartic at u in [0, 1] (u = s_local / length).

    The five stored values -- value and ds-derivative at both ends, plus the
    mean over the slice -- determine a quartic uniquely. With
    p(u) = a0 + a1 u + a2 u^2 + a3 u^3 + a4 u^4 the boundary values fix a0 and
    a1 directly and the remaining three conditions are the linear system in
    _QUARTIC_MATRIX.
    """
    val_start, der_start, val_end, der_end, mean = hermite
    a0 = val_start
    a1 = der_start * length
    rhs = np.array([
        val_end - a0 - a1,
        der_end * length - a1,
        mean - a0 - a1 / 2,
    ])
    a2, a3, a4 = np.linalg.solve(_QUARTIC_MATRIX, rhs)
    return a0 + a1 * u + a2 * u**2 + a3 * u**3 + a4 * u**4


def _sample_solenoid(line):
    """(s, beam-axis fields, multipole rows, n_orders, boundaries) for one
    solenoid template.

    The beam-axis field is taken straight from element.get_field at x = y = 0
    -- that is what the pusher evaluates, with scale_b already applied, so
    there is nothing to reconstruct. The k >= 1 rows have no such accessor
    (get_field returns the total field, not its derivatives), so those come
    from the stored quartics, scaled by scale_b to match.

    `boundaries` are the slice edges: the start of every element plus the end
    of the last, i.e. len(elements) + 1 values.
    """
    elements = list(line.elements)
    n_orders = len(elements[0].bx)

    u = np.linspace(0.0, 1.0, POINTS_PER_SLICE)
    s_parts, on_axis_parts = [], {'bs': [], 'bx': [], 'by': []}
    rows = {('bx', k): [] for k in range(n_orders)}
    rows.update({('by', k): [] for k in range(n_orders)})

    s_start = 0.0
    boundaries = [s_start]
    for element in elements:
        length = element.length
        s_local = u * length
        s_parts.append(s_start + s_local)

        bx, by, bs = element.get_field(
            np.zeros_like(s_local), np.zeros_like(s_local), s_local)
        on_axis_parts['bx'].append(bx)
        on_axis_parts['by'].append(by)
        on_axis_parts['bs'].append(bs)

        for k in range(n_orders):
            for component, stored in (('bx', element.bx), ('by', element.by)):
                rows[component, k].append(
                    element.scale_b
                    * _quartic_from_hermite(
                        np.asarray(stored[k], dtype=float), length, u))

        s_start += length
        boundaries.append(s_start)

    s = np.concatenate(s_parts)
    on_axis = {kk: np.concatenate(vv) for kk, vv in on_axis_parts.items()}
    rows = {kk: np.concatenate(vv) for kk, vv in rows.items()}
    return s, on_axis, rows, n_orders, np.asarray(boundaries)


def _check_against_get_field(line, rows, n_orders):
    """Central-difference d^k Bx/dx^k out of get_field and compare.

    Only orders 1 and 2: a central stencil on a quartic-in-x expansion loses
    two digits per order, so beyond 2 the comparison would measure the stencil,
    not the rows. The point is to confirm the *convention* (that bx[k] is the
    plain k-th derivative, with no 1/k! and no scale_b), which orders 1 and 2
    already settle.
    """
    index = len(line.elements) // 2
    element = list(line.elements)[index]
    step = 1e-4
    u = np.linspace(0.0, 1.0, POINTS_PER_SLICE)
    s_local = u * element.length

    def bx_at(x):
        bx, _, _ = element.get_field(
            np.full_like(s_local, x), np.zeros_like(s_local), s_local)
        return bx

    numeric = {
        1: (bx_at(step) - bx_at(-step)) / (2 * step),
        2: (bx_at(step) - 2 * bx_at(0.0) + bx_at(-step)) / step**2,
    }
    out = []
    for k in (1, 2):
        if k >= n_orders:
            continue
        # Same slice, same u grid, so the rows line up by construction.
        start = index * POINTS_PER_SLICE
        reconstructed = rows['bx', k][start:start + POINTS_PER_SLICE]
        deviation = float(np.max(np.abs(reconstructed - numeric[k])))
        scale = float(np.max(np.abs(numeric[k])))
        # A relative figure is meaningless when the row itself is numerically
        # zero (the compensation solenoid's order-2 row is ~1e-11, i.e. the
        # untilted solenoid has no sextupole content at all): the ratio would
        # then just measure the difference stencil's own noise. Report the
        # deviation as absolute in that case and let the caller say so.
        if scale < 1e-6:
            out.append((k, deviation, None))
        else:
            out.append((k, deviation, deviation / scale))
    return out


###############################################################################
# Plotting                                                                    #
###############################################################################

def _derivative_label(component, k):
    symbol = r'B_x' if component == 'bx' else r'B_y'
    if k == 0:
        return f'${symbol}$'
    if k == 1:
        return fr'$\partial_x {symbol}$'
    return fr'$\partial_x^{{{k}}} {symbol}$'


def _unit_label(k):
    """Unit for an axis label (mathtext)."""
    if k == 0:
        return 'T'
    if k == 1:
        return 'T/m'
    return fr'T/m$^{{{k}}}$'


def _unit_plain(k):
    """Unit for console output -- mathtext would print as literal $ signs."""
    if k == 0:
        return 'T'
    return 'T/m' if k == 1 else f'T/m^{k}'


def _field_figure(tag, s, on_axis, rows, n_orders, scale_b, boundaries):
    """Field at x = y = 0 on top, then one panel per multipole order k >= 1."""
    n_panels = n_orders   # 1 field panel + (n_orders - 1) derivative panels
    fig, axes = plt.subplots(
        n_panels, 1, sharex=True, figsize=(8.5, 1.9 * n_panels + 1.4),
        num=FIGURE_NUMBER[tag])
    axes = np.atleast_1d(axes)

    # Bs runs to a few tesla and Bx to a few tens of millitesla (it is there
    # only because the solenoid is tilted), so they need separate axes or Bx is
    # a flat line on the baseline. Both axes stay black: the colours belong to
    # the curves, and a coloured left axis suggests Bs is the only thing on it.
    ax_bs = axes[0]
    ax_bs.plot(s, on_axis['bs'], color='C0', label=r'$B_s$')
    ax_bs.set_ylabel(r'$B_s$ [T]')

    ax_transverse = ax_bs.twinx()
    ax_transverse.plot(s, on_axis['bx'] * 1e3, color='C1', label=r'$B_x$')
    ax_transverse.plot(s, on_axis['by'] * 1e3, color='C2', linestyle='--',
                       label=r'$B_y$')
    ax_transverse.set_ylabel(r'$B_{x,y}$ [mT]')
    # No "on-axis" in the title: the solenoid is tilted by theta w.r.t. the
    # beam, so x = y = 0 is the beam axis and not the solenoid's own axis.
    ax_bs.set_title(f'{tag} solenoid, {_args.b0:g} T')
    ax_transverse.legend(
        handles=(ax_bs.get_legend_handles_labels()[0]
                 + ax_transverse.get_legend_handles_labels()[0]),
        loc='upper right', fontsize=8, ncol=3, framealpha=0.9)

    # The high orders look like staircases, and that is the model, not the
    # plot. 004a builds these with decrease_s_poly_order_with_transverse_order
    # = True and the rule max(0, 4 - transverse_order) for the s-polynomial
    # degree, so order 1 is cubic in s over a slice, order 2 quadratic, order 3
    # linear and order 4 *constant* -- a genuine piecewise-constant row, which
    # is exactly what the Boris pusher integrates. Sampling more points per
    # slice will not smooth it.
    for k in range(1, n_orders):
        ax = axes[k]
        ax.plot(s, rows['bx', k], color='C1',
                label=_derivative_label('bx', k))
        ax.plot(s, rows['by', k], color='C2', linestyle='--',
                label=_derivative_label('by', k))
        ax.set_ylabel(f'order {k}\n[{_unit_label(k)}]')
        ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    for ax in axes:
        ax.grid(True, alpha=0.35)
        ax.axhline(0.0, color='0.7', linewidth=0.8, zorder=0)
        # Slice boundaries: every element start plus the end of the last one.
        # Always include the final edge, whatever the stride, so the chain's
        # extent is marked at both ends.
        edges = boundaries[::SLICE_LINE_STRIDE]
        if edges[-1] != boundaries[-1]:
            edges = np.append(edges, boundaries[-1])
        for s_edge in edges:
            ax.axvline(s_edge, color='0.4', linewidth=0.4, linestyle='--',
                       alpha=0.7, zorder=0)
    axes[-1].set_xlabel('s [m]')

    fig.subplots_adjust(hspace=0.22, top=0.94, bottom=0.07, left=0.13,
                        right=0.88)
    fig._004aa_stem = f'{tag}_{FIELD_TAG}{ORDER_TAG}'
    return fig


###############################################################################
# Run                                                                         #
###############################################################################

if not INPUT_SOLENOID_LINES_JSON.exists():
    raise SystemExit(
        f'{INPUT_SOLENOID_LINES_JSON.name} not found -- run '
        f'004a_build_and_check_solenoids.py --b0 {_args.b0:g} first.')

_line_data = xt.json.load(INPUT_SOLENOID_LINES_JSON)
print(f'Loaded {INPUT_SOLENOID_LINES_JSON.name}')
_metadata = _line_data.get('metadata', {})
print('  max_transverse_derivative_order_for_spline = '
      f'{_metadata.get("max_transverse_derivative_order_for_spline")}, '
      f'sext_amp = {_metadata.get("sextupole_amplification_factor")}')

FIGURES = []
for _tag in SOLENOIDS:
    _line = xt.Line.from_dict(_line_data['lines'][SOLENOID_LINE_NAME[_tag]])
    _s, _on_axis, _rows, _n_orders, _boundaries = _sample_solenoid(_line)
    _scale_b = list(_line.elements)[0].scale_b

    print()
    print(f'--- {_tag} solenoid ---')
    print(f'  {len(_line.elements)} slices, {_s[-1]:.4f} m long, '
          f'orders 0..{_n_orders - 1}, scale_b = {_scale_b:.6g}')
    print(f'  at x = y = 0: peak |Bs| = {np.max(np.abs(_on_axis["bs"])):.4f} T, '
          f'peak |Bx| = {np.max(np.abs(_on_axis["bx"])) * 1e3:.4f} mT, '
          f'peak |By| = {np.max(np.abs(_on_axis["by"])) * 1e3:.4g} mT')
    print(f'  int Bs ds = {np.trapezoid(_on_axis["bs"], _s):.6f} T m')
    for _k in range(1, _n_orders):
        _bx_row, _by_row = _rows['bx', _k], _rows['by', _k]
        print(f'  order {_k}: peak |d^{_k}Bx/dx^{_k}| = '
              f'{np.max(np.abs(_bx_row)):.6g} {_unit_plain(_k)}, '
              f'integral = {np.trapezoid(_bx_row, _s):.6g} '
              f'{_unit_plain(_k)} m; '
              f'peak |By row| = {np.max(np.abs(_by_row)):.3g}')

    if CHECK_AGAINST_GET_FIELD:
        for _k, _abs, _rel in _check_against_get_field(
                _line, _rows, _n_orders):
            if _rel is None:
                print(f'  check order {_k}: row is numerically zero; '
                      f'reconstruction agrees with get_field to '
                      f'{_abs:.2e} absolute')
            else:
                print(f'  check order {_k}: reconstruction vs central '
                      f'difference of get_field, max rel. deviation '
                      f'{_rel:.2e}')

    FIGURES.append(_field_figure(
        _tag, _s, _on_axis, _rows, _n_orders, _scale_b, _boundaries))

if _args.save:
    from aperture_study_io import PLOT_DIR
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    for _fig in FIGURES:
        _path = PLOT_DIR / f'004aa_splineboris_field_{_fig._004aa_stem}.png'
        _fig.savefig(_path, dpi=200)
        print(f'saved {_path}')

if not _args.no_show:
    plt.show()
