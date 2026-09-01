"""One combined main-solenoid field-strength scan suite, for the 2 T and 3 T
detector-solenoid cases together.

Scans the ``main_b_scale`` knob (global multiplier on every main detector
solenoid's field, added by the updated
004b_install_solenoids_in_fcc_ring.py / lattice_knobs.set_lattice_knobs)
over MAIN_B_SCALE_VALUES = linspace(0.990, 1.010, 21) and, at every scan
point, re-solves the orbit- and coupling-correction knobs from scratch --
exactly the 004f_comp_b_scale_scan.py / 004h_main_b_scale_scan.py recipe,
but:

* run for both --b0 2.0 and --b0 3.0 in one process (loads
  ..._{FIELD_TAG}_mainscale.json for each; both must already be built with
  004b/004c --output-tag mainscale);
* the coupling re-solve is the plain 004f one -- only the 84
  k1s_*_sol_coupling_corr skew quads, unit weights (the per-side
  compensation-field knobs 004h floats are left pinned at 1.0 here);
* an expanded deliverable set.

As a function of main_b_scale (2 T and 3 T overlaid on shared axes):
  - equilibrium emittance shift Deps_x / Deps_y (from radiation integrals,
    relative to the main_b_scale=1.0 point -- same as 004f/004h)
  - horizontal / vertical tune qx / qy          (Twiss table, new)
  - horizontal / vertical chromaticity dqx / dqy (Twiss table, new)
  - the coupling coefficient C^- (tw.c_minus)

In the IR (+-20 m) and over the whole straight section, one curve per
main_b_scale value (curve colour = field-strength multiplier), one figure
set per field case:
  - beta_x / beta_y
  - the coupled-mode betas normalised by their primary beta,
    betx2/betx and bety1/bety
  - dispersion D_x / D_y
  - beam sizes sigma_x / sigma_y (new -- tw.get_beam_covariance with the
    fixed design emittances below)

Plus, per field case, at main_b_scale = 1.0 and with the default
(unit-weight, skew-quad-only) coupling correction: every skew coupling
corrector's integrated strength k1s*L as a fraction of the arc-cell normal
quadrupole strength <|k1 L|>_arc, plotted as thick red dots against the
host quad's longitudinal position s.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import xtrack as xt

from aperture_study_io import PLOT_DIR as _BASE_PLOT_DIR
from lattice_knobs import set_lattice_knobs
from solenoid_params import (
    COMP_SOLENOID_LENGTH,
    add_max_order_argument,
    field_tag,
    half_length_for_b0,
    order_tag,
)


_parser = argparse.ArgumentParser(
    description=(
        'Combined 2 T + 3 T main_b_scale scan suite: emittance/tune/'
        'chromaticity/C- vs main_b_scale, plus IR/straight-section beta, '
        'coupled-beta, dispersion and beam-size profiles, plus the skew '
        'coupling-corrector strengths.'))
add_max_order_argument(_parser)
_parser.add_argument(
    '--b0', type=float, nargs='+', default=[2.0, 3.0], metavar='TESLA',
    help='Main-solenoid field-strength case(s) to run (default: 2.0 3.0). '
         'Each needs its own ..._{FIELD_TAG}{ORDER_TAG}_mainscale.json '
         'built by 004b/004c --output-tag mainscale.')
_parser.add_argument(
    '--input-tag', default='mainscale',
    help='--output-tag that 004b/004c were run with to produce the lattices '
         'to load (default: "mainscale"). Pass "" for the standard untagged '
         'lattices (only works if they carry the main_b_scale knob).')
_parser.add_argument(
    '--coupling-only', action='store_true',
    help='Only re-solve the coupling (skew-quad) correction at each '
         'main_b_scale value; leave the orbit correctors frozen at the '
         'nominal main_b_scale=1.0 fit loaded from the lattice JSON. By '
         'default both orbit and coupling corrections are re-solved at each '
         'scan point.')
_parser.add_argument(
    '--no-show', action='store_true',
    help='Save the figures without opening an interactive window.')
_args = _parser.parse_args()

ORDER_TAG = order_tag(_args.max_transverse_order)
INPUT_TAG = f'_{_args.input_tag}' if _args.input_tag else ''
B0_VALUES = list(_args.b0)

HERE = Path(__file__).parent

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']
IP_PLOT = 'ipa'

# main_b_scale scan grid: 21 points, +-1 % about the nominal main-solenoid
# field (as requested). Same span/count as 004f/004h.
MAIN_B_SCALE_VALUES = np.linspace(0.990, 1.010, 21)

# Fixed design beam parameters used for the beam-size (tw.get_beam_covariance)
# panels -- deliberately NOT the per-scan-point equilibrium values, so the
# beam-size plots isolate the optics/coupling change rather than folding in
# the equilibrium-emittance shift already shown separately. Transverse
# normalised emittances match 009/010 (NEMITT_X/Y); the energy spread matches
# 009's ENERGY_SPREAD and the bunch length is the FCC-ee FS-vol-2 Z value
# (beamstrahlung-inflated, same as 011_bunch_tracking.py). Change here if a
# different working point is wanted.
NEMITT_X = 6.33e-5
NEMITT_Y = 1.69e-7
SIGMA_PZETA = 3.9e-4
SIGMA_ZETA = 15.2e-3
GEMITT_ZETA = SIGMA_ZETA * SIGMA_PZETA

# Quads carrying the orbit correctors -- for plot annotation only, same
# duplication pattern as 004f/004h.
CORRECTOR_QUADS_BY_IP = {
    'ipa': ['qd0ar.0', 'qd0br.0', 'qf1ar.0', 'qf1br.0',
            'qd0al.3', 'qd0bl.3', 'qf1al.3', 'qf1bl.3'],
    'ipd': ['qd0ar.1', 'qd0br.1', 'qf1ar.1', 'qf1br.1',
            'qd0al.0', 'qd0bl.0', 'qf1al.0', 'qf1bl.0'],
    'ipg': ['qd0ar.2', 'qd0br.2', 'qf1ar.2', 'qf1br.2',
            'qd0al.1', 'qd0bl.1', 'qf1al.1', 'qf1bl.1'],
    'ipj': ['qd0ar.3', 'qd0br.3', 'qf1ar.3', 'qf1br.3',
            'qd0al.2', 'qd0bl.2', 'qf1al.2', 'qf1bl.2'],
}

_ORBIT_CORRECTOR_SUFFIXES = (
    'acbh1', 'acbv1', 'acbh2', 'acbh3', 'acbh4', 'acbh5', 'acbh6',
    'acbv2', 'acbv3', 'acbv4', 'acbv5', 'acbv6',
)

_B0_COLORS = {2.0: 'C0', 3.0: 'C3'}


##############################################################
# Geometry / annotation helpers (identical to 004f / 004h).   #
##############################################################

def _region_s_extent(table, s_ip_ref, env_name_prefix):
    env_names = table['env_name'].astype(str)
    mask = np.char.startswith(env_names, env_name_prefix)
    s_starts = table['s_start'][mask]
    s_ends = table['s_end'][mask]
    return s_starts.min() - s_ip_ref, s_ends.max() - s_ip_ref


def _shrink_to_physical_extent(padded_extent, physical_length):
    padded_start, padded_end = padded_extent
    taper_pad = ((padded_end - padded_start) - physical_length) / 2.0
    return padded_start + taper_pad, padded_end - taper_pad


def _compute_marker_positions(table, ip_plot, b0):
    s_ip_ref = table['s', ip_plot]
    main_range = _shrink_to_physical_extent(
        _region_s_extent(table, s_ip_ref, f'sol_slice_{ip_plot}_'),
        2 * half_length_for_b0(b0),
    )
    comp_ranges = [
        _shrink_to_physical_extent(
            _region_s_extent(table, s_ip_ref,
                             f'comp_sol_slice_left_{ip_plot}_'),
            COMP_SOLENOID_LENGTH),
        _shrink_to_physical_extent(
            _region_s_extent(table, s_ip_ref,
                             f'comp_sol_slice_right_{ip_plot}_'),
            COMP_SOLENOID_LENGTH),
    ]
    corrector_positions = [
        table['s', name] - s_ip_ref
        for name in CORRECTOR_QUADS_BY_IP[ip_plot]
    ]
    return main_range, comp_ranges, corrector_positions


def _autoscale_y_to_xlim(ax, xlim, margin=0.1):
    """Rescale ax's ylim to the data actually visible within xlim (see the
    same helper in 004f_comp_b_scale_scan.py for why)."""
    y_min, y_max = np.inf, -np.inf
    for line_obj in ax.get_lines():
        xd, yd = line_obj.get_data()
        xd = np.asarray(xd)
        yd = np.asarray(yd)
        mask = (xd >= xlim[0]) & (xd <= xlim[1])
        if mask.any():
            y_min = min(y_min, np.nanmin(yd[mask]))
            y_max = max(y_max, np.nanmax(yd[mask]))
    if np.isfinite(y_min) and np.isfinite(y_max):
        span = y_max - y_min
        pad = margin * span if span > 0 else max(abs(y_max), 1.0) * margin
        ax.set_ylim(y_min - pad, y_max + pad)


def _mark_solenoid_regions(ax, main_range, comp_ranges, corrector_positions):
    ax.axvspan(*main_range, color='red', alpha=0.15, linewidth=0)
    for comp_range in comp_ranges:
        ax.axvspan(*comp_range, color='orange', alpha=0.15, linewidth=0)
    for s_pos in corrector_positions:
        ax.axvline(s_pos, color='grey', linewidth=0.8, linestyle='--')


##############################################################
# Vary-knob discovery (identical to 004f / 004h).            #
##############################################################

def _straight_section_boundary_names(ip_name):
    return (
        f'end_ds_start_straight_{ip_name}',
        f'end_straight_start_ds_{ip_name}',
    )


def _k1s_coupling_knobs_for_ip(line, table, ip_name, lattice_name):
    name_start, name_end = _straight_section_boundary_names(ip_name)
    quad_names = []
    for table_part in (
            table.rows[name_start:ip_name], table.rows[ip_name:name_end]):
        for element_type, env_name in zip(
                table_part.element_type, table_part.env_name):
            if element_type == 'Quadrupole' and env_name not in quad_names:
                quad_names.append(env_name)
    knob_names = [f'k1s_{nn}_sol_coupling_corr' for nn in quad_names]
    missing = [nn for nn in knob_names if nn not in line.vars]
    if missing:
        raise SystemExit(
            f'{lattice_name} is missing coupling-correction knob(s), e.g. '
            f'{missing[0]!r} -- it must be a lattice produced by '
            '004c_correct_solenoids_in_fcc_ring.py.'
        )
    return knob_names, quad_names


def _orbit_corrector_knobs_for_ip(line, ip_name, lattice_name):
    knob_names = [
        f'{suffix}_sol_{side}_{ip_name}'
        for side in ('right', 'left')
        for suffix in _ORBIT_CORRECTOR_SUFFIXES
    ]
    missing = [nn for nn in knob_names if nn not in line.vars]
    if missing:
        raise SystemExit(
            f'{lattice_name} is missing orbit-corrector knob(s), e.g. '
            f'{missing[0]!r}, for {ip_name}.'
        )
    return knob_names


def _arc_cell_k1l_reference(tw):
    """Median |k1*L| over the arc FODO-cell quads (qf2a.*/qd1a.*). Falls back
    to the median over all quads if that naming isn't found. Same as 004h."""
    mask = tw['element_type'] == 'Quadrupole'
    names = np.asarray([str(n) for n in tw['name']])[mask]
    k1l = np.abs(np.asarray(tw['k1l'])[mask])
    arc = np.array([n.startswith(('qf2a.', 'qd1a.')) for n in names])
    if arc.sum() < 50:
        return float(np.median(k1l)), int(mask.sum()), 'all quads'
    return float(np.median(k1l[arc])), int(arc.sum()), 'qf2a.*/qd1a.*'


##############################################################
# Per-IP orbit + coupling re-correction (the 004f recipe).    #
##############################################################

def _resolve_orbit_correction(line, ip_name, orbit_knobs):
    tw_local = line.twiss4d(strengths=True)
    opt_orbit = line.match(
        solve=False,
        betx=tw_local['betx', ip_name],
        bety=tw_local['bety', ip_name],
        init_at=ip_name,
        start=f'dy_match_l_{ip_name}',
        end=f'dy_match_r_{ip_name}',
        vary=xt.VaryList(orbit_knobs, step=1e-6),
        targets=[
            xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.END),
            xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.START),
        ])
    opt_orbit.solve()
    status = opt_orbit.target_status(ret=True)
    if not all(status.tol_met):
        print(f'  WARNING: orbit re-fit for {ip_name} did not fully converge '
              'to tolerance; using best point found.')


def _resolve_coupling_correction(line, ip_name, k1s_knobs):
    name_start, name_end = _straight_section_boundary_names(ip_name)
    tw_local = line.twiss4d(strengths=True)
    opt_coupling = line.match(
        solve=False,
        betx=tw_local['betx', ip_name],
        bety=tw_local['bety', ip_name],
        init_at=ip_name,
        start=name_start,
        end=name_end,
        n_steps_max=100,
        assert_within_tol=False,
        vary=xt.VaryList(k1s_knobs, step=1e-6),
        targets=[
            xt.TargetSet(betx2=0, bety1=0, at=xt.START, tol=5e-5),
            xt.TargetSet(betx2=0, bety1=0, at=xt.END, tol=5e-5),
            xt.TargetSet(alfx2=0, alfy1=0, at=xt.START, tol=1e-6),
            xt.TargetSet(alfx2=0, alfy1=0, at=xt.END, tol=1e-6),
            xt.TargetSet(dy=0, at=xt.START, tol=5e-5),
            xt.TargetSet(dy=0, at=xt.END, tol=5e-5),
            xt.TargetSet(dpy=0, at=xt.START, tol=1e-7),
            xt.TargetSet(dpy=0, at=xt.END, tol=1e-7),
        ])
    # rcond=0 + broyden=True: see claude_notes/06_coupling_matching_
    # convergence.md -- the ~84-skew-quad Jacobian is severely
    # ill-conditioned; broyden reuses it via cheap rank-1 updates and
    # rcond=0 keeps the tight alfx2/alfy1 directions from being truncated.
    opt_coupling.solve(rcond=0, broyden=True)
    status = opt_coupling.target_status(ret=True)
    if not all(status.tol_met):
        print(f'  WARNING: coupling re-fit for {ip_name} did not fully '
              'converge to tolerance; using best point found.')


##############################################################
# Run one field-strength case end to end.                    #
##############################################################

def run_field_case(b0):
    field_t = field_tag(b0)
    lattice_json = (
        HERE / (
            'fccee_z_lcc_splineboris_solenoids_coupling_corrected_'
            f'{field_t}{ORDER_TAG}{INPUT_TAG}.json'
        )
    )
    if not lattice_json.exists():
        raise SystemExit(
            f'{lattice_json.name} not found -- build it with\n'
            f'  python 004b_install_solenoids_in_fcc_ring.py --b0 {b0:g} '
            f'--output-tag {_args.input_tag or "mainscale"}\n'
            f'  python 004c_correct_solenoids_in_fcc_ring.py --b0 {b0:g} '
            f'--output-tag {_args.input_tag or "mainscale"}'
        )

    print(f'\n=== {field_t} main solenoid: loading {lattice_json.name} ===')
    env = xt.load(lattice_json)
    line = env.fccee_p_ring.copy(shallow=True)
    line.particle_ref.anomalous_magnetic_moment = 0.00115965218128

    if 'main_b_scale' not in line.vars:
        raise SystemExit(
            f'{lattice_json.name} has no main_b_scale knob -- rebuild 004b/004c '
            f'with --output-tag {_args.input_tag or "mainscale"} (same '
            '--b0/--max-transverse-order as here).'
        )

    line.cycle(f'end_ds_start_straight_{IP_NAMES[0]}')
    table_before_cuts = line.get_table()
    for ip_name in IP_NAMES:
        line.cut_at_s(np.arange(
            table_before_cuts['s', ip_name] + 2.4,
            table_before_cuts['s', ip_name] + 11.0, 0.2))
        line.cut_at_s(np.arange(
            table_before_cuts['s', ip_name] - 11.0,
            table_before_cuts['s', ip_name] - 2.4, 0.2))

    k1s_knobs_by_ip = {}
    quad_hosts_by_ip = {}
    for ip_name in IP_NAMES:
        knob_names, quad_names = _k1s_coupling_knobs_for_ip(
            line, table_before_cuts, ip_name, lattice_json.name)
        k1s_knobs_by_ip[ip_name] = knob_names
        quad_hosts_by_ip[ip_name] = quad_names
    orbit_knobs_by_ip = {
        ip_name: _orbit_corrector_knobs_for_ip(
            line, ip_name, lattice_json.name)
        for ip_name in IP_NAMES
    }

    main_range, comp_ranges, corrector_positions = _compute_marker_positions(
        table_before_cuts, IP_PLOT, b0)
    s_ip_ref = table_before_cuts['s', IP_PLOT]
    straight_section_s_range = (
        table_before_cuts['s', f'end_ds_start_straight_{IP_PLOT}'] - s_ip_ref,
        table_before_cuts['s', f'end_straight_start_ds_{IP_PLOT}'] - s_ip_ref,
    )

    for ip_name in IP_NAMES:
        line[f'on_sol_{ip_name}'] = 1
        line[f'on_sol_corr_{ip_name}'] = 1

    points = []
    for main_b_scale in MAIN_B_SCALE_VALUES:
        set_lattice_knobs(
            line, with_solenoids=True, with_correctors=True,
            main_b_scale=float(main_b_scale))
        # Warm-started from the previous scan point (grid is monotonic).
        for ip_name in IP_NAMES:
            if not _args.coupling_only:
                _resolve_orbit_correction(
                    line, ip_name, orbit_knobs_by_ip[ip_name])
            _resolve_coupling_correction(
                line, ip_name, k1s_knobs_by_ip[ip_name])

        k1s_values = {
            nn: float(line.vars[nn]._value)
            for ip_name in IP_NAMES for nn in k1s_knobs_by_ip[ip_name]
        }

        try:
            tw = line.twiss4d(strengths=True, radiation_integrals=True)
        except Exception as exc:  # noqa: BLE001 -- coupled optics can fail
            print(f'  WARNING: twiss failed at main_b_scale={main_b_scale:.4f}'
                  f' ({exc!r}); recording NaNs for this point.')
            points.append(dict(
                main_b_scale=float(main_b_scale), tw=None, beam_sizes=None,
                qx=np.nan, qy=np.nan, dqx=np.nan, dqy=np.nan, c_minus=np.nan,
                eq_gemitt_x=np.nan, eq_gemitt_y=np.nan, k1s_values=k1s_values))
            continue

        scalars = dict(
            qx=float(tw.qx), qy=float(tw.qy),
            dqx=float(getattr(tw, 'dqx', np.nan)),
            dqy=float(getattr(tw, 'dqy', np.nan)),
            c_minus=float(tw.c_minus),
            eq_gemitt_x=float(tw.rad_int_eq_gemitt_x),
            eq_gemitt_y=float(tw.rad_int_eq_gemitt_y),
        )
        tw.zero_at(IP_PLOT)
        beam_sizes = tw.get_beam_covariance(
            nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y, gemitt_zeta=GEMITT_ZETA)

        points.append(dict(
            main_b_scale=float(main_b_scale), tw=tw, beam_sizes=beam_sizes,
            k1s_values=k1s_values, **scalars))
        print(f'  main_b_scale={main_b_scale:+.4f}: twiss OK '
              f'(qx={scalars["qx"]:.5f} qy={scalars["qy"]:.5f} '
              f'C-={scalars["c_minus"]:.3e})')

    # Nominal (main_b_scale = 1.0) skew-corrector snapshot for the red-dot plot.
    nominal_idx = int(np.argmin(np.abs(MAIN_B_SCALE_VALUES - 1.0)))
    tw_nom = points[nominal_idx]['tw']
    skew_dots = None
    if tw_nom is not None:
        k1l_ref, n_ref, ref_label = _arc_cell_k1l_reference(tw_nom)
        L_by_name = dict(zip(np.asarray(tw_nom['name']),
                             np.asarray(tw_nom['length'])))
        s_ring = []
        ratio = []
        for ip_name in IP_NAMES:
            for knob, quad in zip(k1s_knobs_by_ip[ip_name],
                                  quad_hosts_by_ip[ip_name]):
                L = float(L_by_name.get(quad, np.nan))
                if not np.isfinite(L) or L <= 0:
                    continue
                k1s = points[nominal_idx]['k1s_values'][knob]
                s_ring.append(float(table_before_cuts['s', quad]))
                ratio.append(k1s * L / k1l_ref)
        skew_dots = dict(
            s=np.asarray(s_ring), ratio=np.asarray(ratio),
            k1l_ref=k1l_ref, n_ref=n_ref, ref_label=ref_label,
            ip_s={ip: float(table_before_cuts['s', ip]) for ip in IP_NAMES},
        )

    return dict(
        b0=b0, field_tag=field_t, points=points,
        main_range=main_range, comp_ranges=comp_ranges,
        corrector_positions=corrector_positions,
        straight_section_s_range=straight_section_s_range,
        nominal_idx=nominal_idx, skew_dots=skew_dots,
    )


##############################################################
# Plotting.                                                  #
##############################################################

_NORM = mcolors.Normalize(
    vmin=MAIN_B_SCALE_VALUES.min(), vmax=MAIN_B_SCALE_VALUES.max())
_CMAP = cm.viridis
_SM = cm.ScalarMappable(norm=_NORM, cmap=_CMAP)

# IR / straight-section s-windows.
_IR_XLIM = (-20, 20)
_STRAIGHT_XLIM = (-1400, 1400)


def _iter_profiles(case):
    """Yield (main_b_scale, tw, beam_sizes) for the scan points that twissed."""
    for pt in case['points']:
        if pt['tw'] is not None:
            yield pt['main_b_scale'], pt['tw'], pt['beam_sizes']


def _profile_title(case, suffix):
    return (f'{IP_PLOT} main solenoid ({case["b0"]:g} T) -- '
            f'main_b_scale scan{suffix}')


def _add_colorbar(fig, axs):
    fig.colorbar(_SM, ax=axs, label='main_b_scale', fraction=0.06, pad=0.03)


def _decorate_ir(axs, case):
    for ax in axs:
        _mark_solenoid_regions(
            ax, case['main_range'], case['comp_ranges'],
            case['corrector_positions'])


def _decorate_straight(axs, case):
    for ax in axs:
        for s_pos in case['straight_section_s_range']:
            ax.axvline(s_pos, color='black', linewidth=0.8, linestyle=':')


def _make_profile_fig(case, xlim, region_suffix, top_fn, bot_fn,
                      top_label, bot_label, autoscale=False):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8.0, 6.4))
    for main_b_scale, tw, beam_sizes in _iter_profiles(case):
        color = _CMAP(_NORM(main_b_scale))
        top_x, top_y = top_fn(tw, beam_sizes)
        bot_x, bot_y = bot_fn(tw, beam_sizes)
        axs[0].plot(top_x, top_y, color=color)
        axs[1].plot(bot_x, bot_y, color=color)
    axs[0].set_ylabel(top_label)
    axs[1].set_ylabel(bot_label)
    axs[0].set_title(_profile_title(case, region_suffix))
    for ax in axs:
        ax.grid(True)
    axs[-1].set_xlabel('s [m]')
    axs[-1].set_xlim(*xlim)
    if xlim == _IR_XLIM:
        _decorate_ir(axs, case)
    else:
        _decorate_straight(axs, case)
    if autoscale:
        for ax in axs:
            _autoscale_y_to_xlim(ax, xlim)
    fig.subplots_adjust(
        hspace=0.15, top=0.92, bottom=0.1, left=0.12, right=0.88)
    _add_colorbar(fig, axs)
    return fig


# Column accessors: (tw, beam_sizes) -> (x, y).
_BETA = (
    (lambda tw, bs: (tw.s, tw.betx), lambda tw, bs: (tw.s, tw.bety)),
    (r'$\beta_x$ [m]', r'$\beta_y$ [m]'), False)
_COUPLED_BETA = (
    (lambda tw, bs: (tw.s, tw.betx2 / tw.betx),
     lambda tw, bs: (tw.s, tw.bety1 / tw.bety)),
    (r'$\beta_{x2}/\beta_x$', r'$\beta_{y1}/\beta_y$'), False)
_DISP = (
    (lambda tw, bs: (tw.s, tw.dx * 1e3), lambda tw, bs: (tw.s, tw.dy * 1e3)),
    (r'$D_x$ [mm]', r'$D_y$ [mm]'), True)
_BEAMSIZE = (
    (lambda tw, bs: (bs.s, bs.sigma_x * 1e6),
     lambda tw, bs: (bs.s, bs.sigma_y * 1e6)),
    (r'$\sigma_x$ [$\mu$m]', r'$\sigma_y$ [$\mu$m]'), True)

_PROFILE_SPECS = [
    ('beta', _BETA),
    ('coupled_beta', _COUPLED_BETA),
    ('dispersion', _DISP),
    ('beam_size', _BEAMSIZE),
]


def _skew_dot_fig(case):
    sd = case['skew_dots']
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.axhline(0.0, color='0.5', linewidth=0.8)
    ax.plot(sd['s'], sd['ratio'], linestyle='none', marker='o',
            markersize=8, color='red', label='skew coupling correctors')
    for ip_name, s_ip in sd['ip_s'].items():
        ax.axvline(s_ip, color='0.7', linewidth=0.8, linestyle='--')
        ax.text(s_ip, 1.0, f' {ip_name}', transform=ax.get_xaxis_transform(),
                va='top', ha='left', fontsize=8, color='0.4')
    ax.set_xlabel('s [m]  (host quadrupole position)')
    ax.set_ylabel(r'$k_{1s}L \,/\, \langle |k_1 L|\rangle_{\mathrm{arc}}$')
    ax.set_title(
        f'Skew coupling-corrector integrated strength ({case["b0"]:g} T main '
        f'solenoid, main_b_scale = 1.0, unit-weight correction)\n'
        r'relative to arc-cell $\langle|k_1 L|\rangle$ = '
        f'{sd["k1l_ref"]:.3e} 1/m (median over {sd["n_ref"]} '
        f'{sd["ref_label"]} quads)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _scalar_overlay_fig(cases, panel_specs, suptitle):
    """panel_specs: list of (key, ylabel, scale). One panel each, 2 T and 3 T
    overlaid."""
    n = len(panel_specs)
    fig, axs = plt.subplots(n, 1, sharex=True, figsize=(7.0, 2.6 * n + 0.6))
    if n == 1:
        axs = [axs]
    for ax, (key, ylabel, scale) in zip(axs, panel_specs):
        for case in cases:
            x = MAIN_B_SCALE_VALUES
            y = np.array([pt[key] for pt in case['points']]) * scale
            ax.plot(x, y, '-o', color=_B0_COLORS.get(case['b0'], None),
                    label=f'{case["b0"]:g} T')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)
    axs[-1].set_xlabel('main_b_scale')
    axs[0].set_title(suptitle)
    fig.tight_layout()
    return fig


def _emittance_overlay_fig(cases):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.0, 6.4))
    for case in cases:
        idx = case['nominal_idx']
        ex = np.array([pt['eq_gemitt_x'] for pt in case['points']])
        ey = np.array([pt['eq_gemitt_y'] for pt in case['points']])
        color = _B0_COLORS.get(case['b0'], None)
        axs[0].plot(MAIN_B_SCALE_VALUES, (ex - ex[idx]) * 1e9, '-o',
                    color=color, label=f'{case["b0"]:g} T')
        axs[1].plot(MAIN_B_SCALE_VALUES, (ey - ey[idx]) * 1e12, '-o',
                    color=color, label=f'{case["b0"]:g} T')
    axs[0].set_ylabel(r'$\Delta\varepsilon_{x,\mathrm{eq}}$ [nm]')
    axs[1].set_ylabel(r'$\Delta\varepsilon_{y,\mathrm{eq}}$ [pm]')
    axs[1].set_xlabel('main_b_scale')
    axs[0].set_title(
        'Equilibrium emittance shift vs main_b_scale '
        '(relative to main_b_scale = 1.0)')
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    return fig


##############################################################
# Main.                                                      #
##############################################################

def main():
    cases = [run_field_case(b0) for b0 in B0_VALUES]

    plt.close('all')
    figs = {}  # stem -> figure

    # --- Scalar quantities vs main_b_scale, 2 T + 3 T overlaid. ---
    figs['eq_emittance_shift_vs_main_b_scale'] = _emittance_overlay_fig(cases)
    figs['tunes_vs_main_b_scale'] = _scalar_overlay_fig(
        cases,
        [('qx', r'$q_x$', 1.0), ('qy', r'$q_y$', 1.0)],
        'Betatron tunes vs main_b_scale')
    figs['chromaticity_vs_main_b_scale'] = _scalar_overlay_fig(
        cases,
        [('dqx', r"$Q'_x$", 1.0), ('dqy', r"$Q'_y$", 1.0)],
        'Linear chromaticity vs main_b_scale')
    figs['c_minus_vs_main_b_scale'] = _scalar_overlay_fig(
        cases, [('c_minus', r'$C^-$', 1.0)],
        r'Coupling coefficient $C^-$ vs main_b_scale')

    # --- Per-field-case s-profiles (colour = main_b_scale). ---
    for case in cases:
        ft = case['field_tag']
        for name, (fns, labels, autoscale) in _PROFILE_SPECS:
            top_fn, bot_fn = fns
            top_label, bot_label = labels
            figs[f'IR_{name}_{ft}'] = _make_profile_fig(
                case, _IR_XLIM, ' (interaction region)', top_fn, bot_fn,
                top_label, bot_label, autoscale=autoscale)
            figs[f'full_straight_{name}_{ft}'] = _make_profile_fig(
                case, _STRAIGHT_XLIM, ' (full straight section)',
                top_fn, bot_fn, top_label, bot_label, autoscale=autoscale)
        if case['skew_dots'] is not None:
            figs[f'skew_corrector_strength_{ft}'] = _skew_dot_fig(case)

    # --- Save. ---
    plot_dir = _BASE_PLOT_DIR / 'Coupling_Studies' / 'main_b_scale_suite'
    plot_dir.mkdir(parents=True, exist_ok=True)
    scan_tag = (f'{round(MAIN_B_SCALE_VALUES.min() * 1000)}-'
                f'{round(MAIN_B_SCALE_VALUES.max() * 1000)}')
    for stem, fig in figs.items():
        path = plot_dir / f'{stem}_scan{scan_tag}.pdf'
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved plot: {path}')

    if not _args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
