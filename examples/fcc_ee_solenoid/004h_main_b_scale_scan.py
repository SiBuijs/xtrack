"""Scan the main_b_scale knob (global multiplier on every main detector
solenoid's field, added to 004b_install_solenoids_in_fcc_ring.py /
lattice_knobs.set_lattice_knobs) and, at every scan point, re-solve the
orbit- and coupling-correction knobs -- exactly as
004f_comp_b_scale_scan.py does for comp_b_scale, but scanning the *main*
solenoid strength instead of the compensation solenoids.

Two things differ from 004f beyond the scanned knob:

1. The coupling re-solve's vary set is AUGMENTED with the per-side, per-IP
   compensation-solenoid field knobs comp_b_scale_{left,right}_{ip} (also
   added by the updated 004b). The correction is then free to retune the
   compensation-solenoid strength on each side, in addition to the
   k1s_*_sol_coupling_corr skew quads, to null the same 12 coupling targets.
   Letting the compensation strength float breaks the exact net-int(Bs)
   cancellation the nominal design assumes -- that trade-off is part of what
   this scan is meant to expose.

2. It reports a statistic on how hard the skew coupling correctors are being
   driven relative to the ring's normal-quadrupole strength scale (see
   _skew_corrector_stats below): the largest equivalent quad roll angle, the
   RMS integrated skew strength as a fraction of an arc-cell quad, and a
   difference-resonance "drive" proxy comparable to |C_-|.

Requires a corrected lattice built with the main_b_scale and
comp_b_scale_{left,right}_{ip} knobs, i.e. a
004b_install_solenoids_in_fcc_ring.py / 004c_correct_solenoids_in_fcc_ring.py
rebuild done with --output-tag after those knobs were added. By default this
script loads the ..._{FIELD_TAG}{ORDER_TAG}_mainscale.json produced that way
(override with --input-tag). See the runtime check below.
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
    MAIN_SOLENOID_B0,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    half_length_for_b0,
    order_tag,
)


_parser = argparse.ArgumentParser(
    description=(
        'Scan main_b_scale (main-solenoid field multiplier), re-solving the '
        'orbit + coupling corrections at each point, with the per-side '
        'compensation-solenoid knobs added as coupling-match parameters.'))
add_b0_argument(_parser, default=MAIN_SOLENOID_B0)
add_max_order_argument(_parser)
_parser.add_argument(
    '--input-tag', default='mainscale',
    help='--output-tag that 004b/004c were run with to produce the lattice '
         'to load (default: "mainscale"). Pass "" to load the standard '
         'untagged lattice (only works if it was rebuilt with the '
         'main_b_scale / comp_b_scale_{side}_{ip} knobs).')
_parser.add_argument(
    '--coupling-only', action='store_true',
    help='Only re-solve the coupling correction (skew quads + per-side comp '
         'knobs) at each main_b_scale value; leave the orbit correctors '
         'frozen at the nominal main_b_scale=1.0 fit loaded from the lattice '
         'JSON. By default both the orbit and coupling corrections are '
         're-solved at each scan point.')
_parser.add_argument(
    '--no-show', action='store_true',
    help='Save the figures without opening an interactive window '
         '(skip the blocking plt.show()).')
_parser.add_argument(
    '--comp-weight', type=float, default=1e4,
    help='Weight applied to the per-side comp_b_scale_{left,right}_{ip} '
         'knobs in the coupling re-solve. xdeps maps x = knob/weight, so a '
         'larger weight makes the solver more willing to spend these knobs. '
         'weight <~1e3 is a no-op (a fractional-percent compensation-field '
         'change is a ~1e4 weaker coupling handle than a skew quad); '
         '~1e4 makes them visibly engage (see 004i_comp_weight_sweep.py). '
         'Default: 1e4.')
_args = _parser.parse_args()
FIELD_TAG = field_tag(_args.b0)
ORDER_TAG = order_tag(_args.max_transverse_order)
INPUT_TAG = f'_{_args.input_tag}' if _args.input_tag else ''

HERE = Path(__file__).parent
INPUT_LATTICE_JSON = (
    HERE / (
        'fccee_z_lcc_splineboris_solenoids_coupling_corrected_'
        f'{FIELD_TAG}{ORDER_TAG}{INPUT_TAG}.json'
    )
)

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']
IP_PLOT = 'ipa'

# main_b_scale values, evenly spaced around 1.0 (nominal main-solenoid
# field). Same span and point count as 004f_comp_b_scale_scan.py's
# COMP_B_SCALE_VALUES -- "scan the main solenoid over a similar range".
MAIN_B_SCALE_VALUES = np.linspace(0.990, 1.010, 21)

# Per-side, per-IP compensation-solenoid field knobs, added to the coupling
# re-solve's vary set at each scan point (see _resolve_coupling_correction).
COMP_SCALE_KNOBS_BY_IP = {
    ip_name: [f'comp_b_scale_left_{ip_name}', f'comp_b_scale_right_{ip_name}']
    for ip_name in IP_NAMES
}

MAIN_SOLENOID_FIELD_LABEL = f'{_args.b0:g} T'

# Quads carrying the orbit correctors -- copied here for plot annotation
# only, same duplication pattern as 004f_comp_b_scale_scan.py.
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


################################
# Load and prepare the lattice #
################################

env = xt.load(INPUT_LATTICE_JSON)

line = env.fccee_p_ring.copy(shallow=True)
line.particle_ref.anomalous_magnetic_moment = 0.00115965218128

if 'main_b_scale' not in line.vars:
    raise SystemExit(
        f'{INPUT_LATTICE_JSON.name} has no main_b_scale knob -- it was built '
        'before that knob was added to 004b_install_solenoids_in_fcc_ring.py. '
        'Rebuild it via `python 004b_install_solenoids_in_fcc_ring.py '
        '--output-tag mainscale` followed by `python '
        '004c_correct_solenoids_in_fcc_ring.py --output-tag mainscale` (same '
        '--b0/--max-transverse-order flags as here, if non-default), then '
        'rerun this script.'
    )
_missing_comp = [
    nn for ip_name in IP_NAMES for nn in COMP_SCALE_KNOBS_BY_IP[ip_name]
    if nn not in line.vars
]
if _missing_comp:
    raise SystemExit(
        f'{INPUT_LATTICE_JSON.name} is missing per-side compensation knob(s), '
        f'e.g. {_missing_comp[0]!r} -- rebuild 004b/004c with --output-tag as '
        'above (the updated 004b creates comp_b_scale_{left,right}_{ip}).'
    )

line.cycle(f'end_ds_start_straight_{IP_NAMES[0]}')
table_before_cuts = line.get_table()
for ip_name in IP_NAMES:
    s_cut_right = np.arange(
        table_before_cuts['s', ip_name] + 2.4,
        table_before_cuts['s', ip_name] + 11.0,
        0.2,
    )
    line.cut_at_s(s_cut_right)

    s_cut_left = np.arange(
        table_before_cuts['s', ip_name] - 11.0,
        table_before_cuts['s', ip_name] - 2.4,
        0.2,
    )
    line.cut_at_s(s_cut_left)


##############################################################
# Element s-positions (in the tw.s frame) for plot annotation #
##############################################################
# Same approach as 004f_comp_b_scale_scan.py / 004d_analysis_and_plots.py.

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
            _region_s_extent(table, s_ip_ref, f'comp_sol_slice_left_{ip_plot}_'),
            COMP_SOLENOID_LENGTH),
        _shrink_to_physical_extent(
            _region_s_extent(table, s_ip_ref, f'comp_sol_slice_right_{ip_plot}_'),
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
        mask = (xd >= xlim[0]) & (xd <= xlim[1])
        if mask.any():
            y_min = min(y_min, np.min(yd[mask]))
            y_max = max(y_max, np.max(yd[mask]))
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


MAIN_SOLENOID_S_RANGE, COMP_SOLENOID_S_RANGES, CORRECTOR_QUAD_S_POSITIONS = (
    _compute_marker_positions(table_before_cuts, IP_PLOT, _args.b0)
)

_straight_section_s_ip_ref = table_before_cuts['s', IP_PLOT]
STRAIGHT_SECTION_S_RANGE = (
    table_before_cuts['s', f'end_ds_start_straight_{IP_PLOT}']
    - _straight_section_s_ip_ref,
    table_before_cuts['s', f'end_straight_start_ds_{IP_PLOT}']
    - _straight_section_s_ip_ref,
)


#####################################################################
# Per-IP orbit + coupling re-correction, one main_b_scale at a time. #
# Same targets/vary as 004c's opt_orbit/opt_coupling, re-solved      #
# fresh at each scan point (warm-started across points), with the    #
# per-side comp_b_scale knobs added to the coupling vary set.        #
#####################################################################

def _straight_section_boundary_names(ip_name):
    return (
        f'end_ds_start_straight_{ip_name}',
        f'end_straight_start_ds_{ip_name}',
    )


def _k1s_coupling_knobs_for_ip(table, ip_name):
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
            f'{INPUT_LATTICE_JSON.name} is missing coupling-correction knob(s), '
            f'e.g. {missing[0]!r} -- it must be a lattice produced by '
            '004c_correct_solenoids_in_fcc_ring.py.'
        )
    return knob_names


K1S_KNOBS_BY_IP = {
    ip_name: _k1s_coupling_knobs_for_ip(table_before_cuts, ip_name)
    for ip_name in IP_NAMES
}

_ORBIT_CORRECTOR_SUFFIXES = (
    'acbh1', 'acbv1', 'acbh2', 'acbh3', 'acbh4', 'acbh5', 'acbh6',
    'acbv2', 'acbv3', 'acbv4', 'acbv5', 'acbv6',
)


def _orbit_corrector_knobs_for_ip(ip_name):
    knob_names = [
        f'{suffix}_sol_{side}_{ip_name}'
        for side in ('right', 'left')
        for suffix in _ORBIT_CORRECTOR_SUFFIXES
    ]
    missing = [nn for nn in knob_names if nn not in line.vars]
    if missing:
        raise SystemExit(
            f'{INPUT_LATTICE_JSON.name} is missing orbit-corrector knob(s), '
            f'e.g. {missing[0]!r}, for {ip_name}.'
        )
    return knob_names


ORBIT_CORRECTOR_KNOBS_BY_IP = {
    ip_name: _orbit_corrector_knobs_for_ip(ip_name) for ip_name in IP_NAMES
}


def _knob_value_table(title, knob_names, values_by_scale):
    """Print one row per knob, one column per MAIN_B_SCALE_VALUES entry."""
    col_width = 13
    header = f'{"knob":<42s}' + ''.join(
        f'{cb:>{col_width}.4f}' for cb in MAIN_B_SCALE_VALUES)
    print(f'\n{title}')
    print(header)
    print('-' * len(header))
    for nn in knob_names:
        row = f'{nn:<42s}' + ''.join(
            f'{values[nn]:>{col_width}.3e}' for values in values_by_scale)
        print(row)


def _resolve_orbit_correction(ip_name):
    """Re-solve the acbh/acbv_sol_* dipole correctors for this IP against the
    current (main_b_scale-perturbed) optics -- same start/end/targets as
    004c's opt_orbit block. Skipped when --coupling-only is passed."""
    tw_local = line.twiss4d(strengths=True)
    opt_orbit = line.match(
        solve=False,
        betx=tw_local['betx', ip_name],
        bety=tw_local['bety', ip_name],
        init_at=ip_name,
        start=f'dy_match_l_{ip_name}',
        end=f'dy_match_r_{ip_name}',
        vary=xt.VaryList(ORBIT_CORRECTOR_KNOBS_BY_IP[ip_name], step=1e-6),
        targets=[
            xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.END),
            xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.START),
        ])
    opt_orbit.solve()
    status = opt_orbit.target_status(ret=True)
    if not all(status.tol_met):
        print(f'  WARNING: orbit re-fit for {ip_name} did not fully converge '
              'to tolerance; using best point found.')


def _resolve_coupling_correction(ip_name):
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
        vary=[
            xt.VaryList(K1S_KNOBS_BY_IP[ip_name], step=1e-6),
            # Per-side compensation-solenoid field knobs. A fractional-percent
            # change in compensation field is a ~1e4 weaker linear-coupling
            # handle than a k1s skew quad, so with unit weight the least-
            # squares solver ignores these entirely while skew authority
            # holds (which it does across the whole scan). xdeps maps
            # x = knob/weight, so the solver's Jacobian column for a knob is
            # (physical column) * weight -- --comp-weight (default 1e4) lifts
            # these to ~the median skew-quad column so the solver actually
            # spends them, exposing the skew-vs-comp trade-off. limits keep
            # the field physical; no max_step (with a large weight a knob-
            # unit max_step clips almost every step). See
            # 004i_comp_weight_sweep.py for the weight-dependence.
            xt.VaryList(
                COMP_SCALE_KNOBS_BY_IP[ip_name],
                step=1e-4, limits=(0.5, 1.5), weight=_args.comp_weight),
        ],
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
    # Same rcond=0 + broyden=True as 004f (see claude_notes/06): the
    # ~84-skew-quad Jacobian is severely ill-conditioned; broyden reuses it
    # via cheap rank-1 updates, rcond=0 keeps the tight alfx2/alfy1
    # directions from being truncated. The 2 comp knobs per IP are
    # strong-authority (large singular values) and don't change that
    # picture -- rerun 004g_coupling_svd_diagnostic.py with the comp knobs
    # included to inspect the new spectrum.
    opt_coupling.solve(rcond=0, broyden=True)
    status = opt_coupling.target_status(ret=True)
    if not all(status.tol_met):
        print(f'  WARNING: coupling re-fit for {ip_name} did not fully '
              'converge to tolerance; using best point found.')


##############################################################
# Skew-corrector-strength statistic (see module docstring).   #
##############################################################

def _arc_cell_k1l_reference(twiss_or_line_table):
    """Median |k1*L| over the arc FODO-cell quads (qf2a.*/qd1a.* -- by far
    the most numerous quads in the ring; the ring median is identical to any
    superset of them). Falls back to the median over all quads if that
    naming isn't found."""
    tt = twiss_or_line_table
    mask = tt['element_type'] == 'Quadrupole'
    names = np.asarray([str(n) for n in tt['name']])[mask]
    k1l = np.abs(np.asarray(tt['k1l'])[mask])
    arc = np.array([n.startswith(('qf2a.', 'qd1a.')) for n in names])
    if arc.sum() < 50:
        return float(np.median(k1l)), int(mask.sum()), 'all quads'
    return float(np.median(k1l[arc])), int(arc.sum()), 'qf2a.*/qd1a.*'


def _skew_corrector_stats(tw, k1s_knobs, k1l_ref):
    """Given a twiss (with strengths=True, so k1l/length available) and one
    IP's list of k1s_*_sol_coupling_corr knob names, return summary numbers
    on how hard those skew correctors are being driven.

    - max_roll_mrad / rms_roll_mrad: |0.5*atan2(k1s, k1_host)|, the
      equivalent roll angle of the host quad; hosts with negligible own
      gradient (|k1l| < 0.1*k1l_ref) are excluded from the roll stats to
      avoid divide-by-near-zero.
    - max_k1sL_ratio / rms_k1sL_ratio: |k1s*L| as a fraction of an arc-cell
      quad's |k1*L|.
    - sum_drive: sum_i |k1s_i*L_i|*sqrt(betx_i*bety_i)/(2*pi), an upper
      bound on the skew system's contribution to the |C_-| difference
      resonance (dimensionless, directly comparable to |C_-|).
    """
    names = np.asarray(tw['name'])
    k1l_by_name = dict(zip(names, np.asarray(tw['k1l'])))
    L_by_name = dict(zip(names, np.asarray(tw['length'])))
    betx_by_name = dict(zip(names, np.asarray(tw['betx'])))
    bety_by_name = dict(zip(names, np.asarray(tw['bety'])))

    rolls, k1sL, drive = [], [], []
    for knob in k1s_knobs:
        quad = knob[len('k1s_'):-len('_sol_coupling_corr')]
        L = float(L_by_name.get(quad, np.nan))
        k1l_host = float(k1l_by_name.get(quad, np.nan))
        k1s = float(line[knob])
        if not np.isfinite(L) or L <= 0:
            continue
        k1sL.append(k1s * L)
        if np.isfinite(k1l_host) and abs(k1l_host) > 0.1 * k1l_ref:
            # k1s / k1 = tan(2*phi) for a quad rolled by phi -> phi is the
            # equivalent roll. Use arctan of the ratio (NOT arctan2, which
            # returns +-pi/2 for a zero-gradient host and wraps to pi for a
            # defocusing k1 < 0 host).
            rolls.append(0.5 * np.arctan(k1s * L / k1l_host))
        bx = float(betx_by_name.get(quad, np.nan))
        by = float(bety_by_name.get(quad, np.nan))
        if np.isfinite(bx) and np.isfinite(by):
            drive.append(abs(k1s * L) * np.sqrt(bx * by) / (2.0 * np.pi))

    rolls = np.asarray(rolls)
    k1sL = np.asarray(k1sL)
    drive = np.asarray(drive)
    return dict(
        max_roll_mrad=(np.max(np.abs(rolls)) * 1e3 if rolls.size else np.nan),
        rms_roll_mrad=(np.sqrt(np.mean(rolls ** 2)) * 1e3
                       if rolls.size else np.nan),
        max_k1sL_ratio=(np.max(np.abs(k1sL)) / k1l_ref
                        if k1sL.size else np.nan),
        rms_k1sL_ratio=(np.sqrt(np.mean(k1sL ** 2)) / k1l_ref
                        if k1sL.size else np.nan),
        sum_drive=float(np.sum(drive)) if drive.size else np.nan,
    )


############################################
# Twiss once per main_b_scale, solenoids on #
############################################

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 1
    line[f'on_sol_corr_{ip_name}'] = 1

TWISS_BY_MAIN_B_SCALE = []
K1S_VALUES_BY_MAIN_B_SCALE = []
ORBIT_VALUES_BY_MAIN_B_SCALE = []
COMP_SCALE_VALUES_BY_MAIN_B_SCALE = []
STATS_BY_MAIN_B_SCALE = []       # list (per scan point) of {ip_name: statdict}

for main_b_scale in MAIN_B_SCALE_VALUES:
    set_lattice_knobs(
        line, with_solenoids=True, with_correctors=True,
        main_b_scale=float(main_b_scale))
    # Warm-started from the previous scan point (MAIN_B_SCALE_VALUES is
    # monotonic), not reset to the nominal main_b_scale=1.0 solution.
    for ip_name in IP_NAMES:
        if not _args.coupling_only:
            _resolve_orbit_correction(ip_name)
        _resolve_coupling_correction(ip_name)

    tw = line.twiss4d(strengths=True, radiation_integrals=True)

    k1l_ref, n_ref, ref_label = _arc_cell_k1l_reference(tw)
    stats = {
        ip_name: _skew_corrector_stats(tw, K1S_KNOBS_BY_IP[ip_name], k1l_ref)
        for ip_name in IP_NAMES
    }
    STATS_BY_MAIN_B_SCALE.append(stats)

    tw.zero_at(IP_PLOT)
    TWISS_BY_MAIN_B_SCALE.append(tw)
    K1S_VALUES_BY_MAIN_B_SCALE.append({
        nn: float(line.vars[nn]._value)
        for ip_name in IP_NAMES for nn in K1S_KNOBS_BY_IP[ip_name]})
    ORBIT_VALUES_BY_MAIN_B_SCALE.append({
        nn: float(line.vars[nn]._value)
        for ip_name in IP_NAMES for nn in ORBIT_CORRECTOR_KNOBS_BY_IP[ip_name]})
    COMP_SCALE_VALUES_BY_MAIN_B_SCALE.append({
        nn: float(line.vars[nn]._value)
        for ip_name in IP_NAMES for nn in COMP_SCALE_KNOBS_BY_IP[ip_name]})
    print(f'main_b_scale={main_b_scale:+.4f}: twiss OK '
          f'(orbit{"" if not _args.coupling_only else " frozen"}, coupling '
          f'+ per-side comp knobs re-solved)')

print(f'\nArc-cell |k1*L| reference = {k1l_ref:.4g} 1/m '
      f'(median over {n_ref} {ref_label} quads)')


###################################################################
# Corrector-knob value tables (one row per knob, one column per   #
# MAIN_B_SCALE_VALUES entry), printed separately per IP.          #
###################################################################

for ip_name in IP_NAMES:
    _knob_value_table(
        f'Skew-quad coupling-correction knobs (k1s_*_sol_coupling_corr) '
        f'-- {ip_name} -- re-solved at each main_b_scale',
        K1S_KNOBS_BY_IP[ip_name], K1S_VALUES_BY_MAIN_B_SCALE)

for ip_name in IP_NAMES:
    _knob_value_table(
        f'Per-side compensation-solenoid knobs (comp_b_scale_{{side}}_{ip_name}) '
        f'-- {ip_name} -- re-solved at each main_b_scale',
        COMP_SCALE_KNOBS_BY_IP[ip_name], COMP_SCALE_VALUES_BY_MAIN_B_SCALE)

_orbit_table_note = (
    'frozen at the nominal main_b_scale=1.0 fit (--coupling-only)'
    if _args.coupling_only else 're-solved at each main_b_scale')
for ip_name in IP_NAMES:
    _knob_value_table(
        f'Orbit-corrector knobs (acbh/acbv_sol_*) -- {ip_name} -- '
        f'{_orbit_table_note}',
        ORBIT_CORRECTOR_KNOBS_BY_IP[ip_name], ORBIT_VALUES_BY_MAIN_B_SCALE)


#########
# Plots #
#########

plt.close('all')

_norm = mcolors.Normalize(
    vmin=MAIN_B_SCALE_VALUES.min(), vmax=MAIN_B_SCALE_VALUES.max())
_cmap = cm.viridis
_sm = cm.ScalarMappable(norm=_norm, cmap=_cmap)

_IP_COLORS = {'ipa': 'C0', 'ipd': 'C1', 'ipg': 'C2', 'ipj': 'C3'}


def _scan_title(title_suffix):
    return (f'{IP_PLOT} main solenoid ({MAIN_SOLENOID_FIELD_LABEL}) -- '
            f'main_b_scale scan{title_suffix}')


def _plot_betx2_bety1_scan(axs, xlim, title_suffix):
    for main_b_scale, tw in zip(MAIN_B_SCALE_VALUES, TWISS_BY_MAIN_B_SCALE):
        color = _cmap(_norm(main_b_scale))
        axs[0].plot(tw.s, tw.betx2 / tw.betx, color=color)
        axs[1].plot(tw.s, tw.bety1 / tw.bety, color=color)
    axs[0].set_ylabel(r'$\beta_{x2}/\beta_x$')
    axs[1].set_ylabel(r'$\beta_{y1}/\beta_y$')
    axs[0].set_title(_scan_title(title_suffix))
    for ax in axs:
        ax.grid(True)
    axs[-1].set_xlabel('s [m]')
    axs[-1].set_xlim(*xlim)


def _plot_beta_scan(axs, xlim, title_suffix):
    for main_b_scale, tw in zip(MAIN_B_SCALE_VALUES, TWISS_BY_MAIN_B_SCALE):
        color = _cmap(_norm(main_b_scale))
        axs[0].plot(tw.s, tw.betx, color=color)
        axs[1].plot(tw.s, tw.bety, color=color)
    axs[0].set_ylabel(r'$\beta_x$ [m]')
    axs[1].set_ylabel(r'$\beta_y$ [m]')
    axs[0].set_title(_scan_title(title_suffix))
    for ax in axs:
        ax.grid(True)
    axs[-1].set_xlabel('s [m]')
    axs[-1].set_xlim(*xlim)


def _plot_dispersion_scan(axs, xlim, title_suffix):
    for main_b_scale, tw in zip(MAIN_B_SCALE_VALUES, TWISS_BY_MAIN_B_SCALE):
        color = _cmap(_norm(main_b_scale))
        axs[0].plot(tw.s, tw.dx * 1e3, color=color)
        axs[1].plot(tw.s, tw.dy * 1e3, color=color)
    axs[0].set_ylabel(r'$D_x$ [mm]')
    axs[1].set_ylabel(r'$D_y$ [mm]')
    axs[0].set_title(_scan_title(title_suffix))
    for ax in axs:
        ax.grid(True)
    axs[-1].set_xlabel('s [m]')
    axs[-1].set_xlim(*xlim)
    for ax in axs:
        _autoscale_y_to_xlim(ax, xlim)


fig1, axs1 = plt.subplots(2, 1, sharex=True, figsize=(8.0, 6.4))
_plot_betx2_bety1_scan(axs1, (-20, 20), ' (interaction region)')
for ax in axs1:
    _mark_solenoid_regions(
        ax, MAIN_SOLENOID_S_RANGE, COMP_SOLENOID_S_RANGES,
        CORRECTOR_QUAD_S_POSITIONS)
fig1.subplots_adjust(hspace=0.15, top=0.92, bottom=0.1, left=0.12, right=0.88)
fig1.colorbar(_sm, ax=axs1, label='main_b_scale', fraction=0.06, pad=0.03)

fig2, axs2 = plt.subplots(2, 1, sharex=True, figsize=(8.0, 6.4))
_plot_betx2_bety1_scan(axs2, (-1400, 1400), ' (full straight section)')
for ax in axs2:
    for s_pos in STRAIGHT_SECTION_S_RANGE:
        ax.axvline(s_pos, color='black', linewidth=0.8, linestyle=':')
fig2.subplots_adjust(hspace=0.15, top=0.92, bottom=0.1, left=0.12, right=0.88)
fig2.colorbar(_sm, ax=axs2, label='main_b_scale', fraction=0.06, pad=0.03)

fig_beta_ir, axs_beta_ir = plt.subplots(2, 1, sharex=True, figsize=(8.0, 6.4))
_plot_beta_scan(axs_beta_ir, (-20, 20), ' (interaction region)')
for ax in axs_beta_ir:
    _mark_solenoid_regions(
        ax, MAIN_SOLENOID_S_RANGE, COMP_SOLENOID_S_RANGES,
        CORRECTOR_QUAD_S_POSITIONS)
fig_beta_ir.subplots_adjust(
    hspace=0.15, top=0.92, bottom=0.1, left=0.12, right=0.88)
fig_beta_ir.colorbar(
    _sm, ax=axs_beta_ir, label='main_b_scale', fraction=0.06, pad=0.03)

fig_beta_full, axs_beta_full = plt.subplots(
    2, 1, sharex=True, figsize=(8.0, 6.4))
_plot_beta_scan(axs_beta_full, (-1400, 1400), ' (full straight section)')
for ax in axs_beta_full:
    for s_pos in STRAIGHT_SECTION_S_RANGE:
        ax.axvline(s_pos, color='black', linewidth=0.8, linestyle=':')
fig_beta_full.subplots_adjust(
    hspace=0.15, top=0.92, bottom=0.1, left=0.12, right=0.88)
fig_beta_full.colorbar(
    _sm, ax=axs_beta_full, label='main_b_scale', fraction=0.06, pad=0.03)

fig_disp_ir, axs_disp_ir = plt.subplots(2, 1, sharex=True, figsize=(8.0, 6.4))
_plot_dispersion_scan(axs_disp_ir, (-20, 20), ' (interaction region)')
for ax in axs_disp_ir:
    _mark_solenoid_regions(
        ax, MAIN_SOLENOID_S_RANGE, COMP_SOLENOID_S_RANGES,
        CORRECTOR_QUAD_S_POSITIONS)
fig_disp_ir.subplots_adjust(
    hspace=0.15, top=0.92, bottom=0.1, left=0.12, right=0.88)
fig_disp_ir.colorbar(
    _sm, ax=axs_disp_ir, label='main_b_scale', fraction=0.06, pad=0.03)

fig_disp_full, axs_disp_full = plt.subplots(
    2, 1, sharex=True, figsize=(8.0, 6.4))
_plot_dispersion_scan(axs_disp_full, (-1400, 1400), ' (full straight section)')
for ax in axs_disp_full:
    for s_pos in STRAIGHT_SECTION_S_RANGE:
        ax.axvline(s_pos, color='black', linewidth=0.8, linestyle=':')
fig_disp_full.subplots_adjust(
    hspace=0.15, top=0.92, bottom=0.1, left=0.12, right=0.88)
fig_disp_full.colorbar(
    _sm, ax=axs_disp_full, label='main_b_scale', fraction=0.06, pad=0.03)


# C_minus: whole-ring scalar per twiss.
C_MINUS_VALUES = np.array([tw.c_minus for tw in TWISS_BY_MAIN_B_SCALE])

fig3, ax3 = plt.subplots(figsize=(6.4, 4.8))
ax3.plot(MAIN_B_SCALE_VALUES, C_MINUS_VALUES, '-o', color='C0')
ax3.set_xlabel('main_b_scale')
ax3.set_ylabel(r'$C_-$')
ax3.set_title(
    f'Closest-tune-approach coupling coefficient vs main_b_scale '
    f'({MAIN_SOLENOID_FIELD_LABEL} main solenoid)')
ax3.grid(True)
fig3.tight_layout()


# Equilibrium emittances (whole-ring scalars), from radiation_integrals=True.
EQ_GEMITT_X_VALUES = np.array(
    [tw.rad_int_eq_gemitt_x for tw in TWISS_BY_MAIN_B_SCALE])
EQ_GEMITT_Y_VALUES = np.array(
    [tw.rad_int_eq_gemitt_y for tw in TWISS_BY_MAIN_B_SCALE])

_nominal_idx = int(np.argmin(np.abs(MAIN_B_SCALE_VALUES - 1.0)))
if not np.isclose(MAIN_B_SCALE_VALUES[_nominal_idx], 1.0):
    print(f'WARNING: main_b_scale=1.0 not in MAIN_B_SCALE_VALUES -- using '
          f'closest value {MAIN_B_SCALE_VALUES[_nominal_idx]:.4f} as the '
          'equilibrium-emittance baseline instead.')
DELTA_EQ_GEMITT_X_VALUES = (
    EQ_GEMITT_X_VALUES - EQ_GEMITT_X_VALUES[_nominal_idx])
DELTA_EQ_GEMITT_Y_VALUES = (
    EQ_GEMITT_Y_VALUES - EQ_GEMITT_Y_VALUES[_nominal_idx])

fig_emit, (ax_emit_x, ax_emit_y) = plt.subplots(
    2, 1, sharex=True, figsize=(6.4, 6.4))
ax_emit_x.plot(
    MAIN_B_SCALE_VALUES, DELTA_EQ_GEMITT_X_VALUES * 1e9, '-o', color='C0')
ax_emit_x.set_ylabel(r'$\Delta\varepsilon_{x,\mathrm{eq}}$ [nm]')
ax_emit_x.set_title(
    f'Equilibrium emittance shift vs main_b_scale (relative to '
    f'main_b_scale=1.0, {MAIN_SOLENOID_FIELD_LABEL} main solenoid)')
ax_emit_x.grid(True)
ax_emit_y.plot(
    MAIN_B_SCALE_VALUES, DELTA_EQ_GEMITT_Y_VALUES * 1e12, '-o', color='C1')
ax_emit_y.set_xlabel('main_b_scale')
ax_emit_y.set_ylabel(r'$\Delta\varepsilon_{y,\mathrm{eq}}$ [pm]')
ax_emit_y.grid(True)
fig_emit.tight_layout()


# ---- Skew-corrector-strength statistic vs main_b_scale (one line per IP) ----

def _stat_series(key):
    return {
        ip_name: np.array(
            [STATS_BY_MAIN_B_SCALE[i][ip_name][key]
             for i in range(len(MAIN_B_SCALE_VALUES))])
        for ip_name in IP_NAMES
    }


_max_roll = _stat_series('max_roll_mrad')
_rms_roll = _stat_series('rms_roll_mrad')
_max_ratio = _stat_series('max_k1sL_ratio')
_rms_ratio = _stat_series('rms_k1sL_ratio')
_sum_drive = _stat_series('sum_drive')

fig_stat, (ax_roll, ax_ratio) = plt.subplots(
    2, 1, sharex=True, figsize=(7.5, 7.0))
for ip_name in IP_NAMES:
    c = _IP_COLORS[ip_name]
    ax_roll.plot(MAIN_B_SCALE_VALUES, _max_roll[ip_name], '-o', color=c,
                 label=f'{ip_name} max')
    ax_roll.plot(MAIN_B_SCALE_VALUES, _rms_roll[ip_name], '--', color=c,
                 label=f'{ip_name} rms')
    ax_ratio.plot(MAIN_B_SCALE_VALUES, _max_ratio[ip_name], '-o', color=c,
                  label=f'{ip_name} max')
    ax_ratio.plot(MAIN_B_SCALE_VALUES, _rms_ratio[ip_name], '--', color=c,
                  label=f'{ip_name} rms')
ax_roll.set_ylabel('equivalent host-quad\nroll angle [mrad]')
ax_roll.set_title(
    'Skew coupling correctors vs normal-quad strength -- main_b_scale scan\n'
    r'roll $= \frac{1}{2}\arctan(k_{1s}/k_1)$ on each host quad; '
    r'ratio $= |k_{1s}L| / \mathrm{median}_\mathrm{arc}|k_1 L|$')
ax_ratio.set_ylabel(r'$|k_{1s}L|\,/\,$arc-cell$\,|k_1 L|$')
ax_ratio.set_xlabel('main_b_scale')
for ax in (ax_roll, ax_ratio):
    ax.grid(True)
    ax.legend(fontsize=7, ncol=4, loc='best')
fig_stat.tight_layout()

fig_drive, (ax_drive, ax_cm) = plt.subplots(
    2, 1, sharex=True, figsize=(7.0, 6.4))
for ip_name in IP_NAMES:
    ax_drive.plot(MAIN_B_SCALE_VALUES, _sum_drive[ip_name], '-o',
                  color=_IP_COLORS[ip_name], label=ip_name)
ax_drive.set_ylabel(
    r'$\sum_i |k_{1s,i} L_i|\sqrt{\beta_{x,i}\beta_{y,i}}\,/\,2\pi$')
ax_drive.set_title(
    'Skew-corrector difference-resonance drive vs main_b_scale\n'
    '(per-IP sum, upper bound on the |C_-| contribution) and whole-ring '
    r'$C_-$')
ax_drive.grid(True)
ax_drive.legend(fontsize=8, loc='best')
ax_cm.plot(MAIN_B_SCALE_VALUES, C_MINUS_VALUES, '-o', color='k')
ax_cm.set_ylabel(r'$C_-$ (whole ring)')
ax_cm.set_xlabel('main_b_scale')
ax_cm.grid(True)
fig_drive.tight_layout()

# ---- Fitted per-side comp_b_scale knobs vs main_b_scale ----
fig_comp, axs_comp = plt.subplots(2, 2, sharex=True, figsize=(9.0, 6.4))
for ax, ip_name in zip(axs_comp.flat, IP_NAMES):
    left_key = f'comp_b_scale_left_{ip_name}'
    right_key = f'comp_b_scale_right_{ip_name}'
    left = np.array([v[left_key] for v in COMP_SCALE_VALUES_BY_MAIN_B_SCALE])
    right = np.array([v[right_key] for v in COMP_SCALE_VALUES_BY_MAIN_B_SCALE])
    ax.plot(MAIN_B_SCALE_VALUES, left, '-o', color='C0', label='left')
    ax.plot(MAIN_B_SCALE_VALUES, right, '-s', color='C1', label='right')
    ax.axhline(1.0, color='0.6', linewidth=0.8)
    ax.set_title(ip_name)
    ax.grid(True)
    ax.legend(fontsize=8, loc='best')
for ax in axs_comp[-1, :]:
    ax.set_xlabel('main_b_scale')
for ax in axs_comp[:, 0]:
    ax.set_ylabel('fitted comp field scale')
fig_comp.suptitle(
    'Per-side compensation-solenoid field scale chosen by the coupling '
    f'match vs main_b_scale ({MAIN_SOLENOID_FIELD_LABEL} main solenoid)')
fig_comp.tight_layout()

print(f'Loaded {INPUT_LATTICE_JSON}')
print(f'main_b_scale values: {MAIN_B_SCALE_VALUES}')


#####################################################################
# Save every figure under Coupling_Studies/main_b_scale_scan/, with  #
# a 'main_b_scale' marker in every stem so nothing collides with     #
# 004f_comp_b_scale_scan.py's output in Coupling_Studies/.           #
#####################################################################

COUPLING_STUDIES_PLOT_DIR = (
    _BASE_PLOT_DIR / 'Coupling_Studies' / 'main_b_scale_scan')
COUPLING_STUDIES_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _scan_range_tag(values):
    return f'{round(values.min() * 1000)}-{round(values.max() * 1000)}'


# Fold the input-lattice tag and (non-default) comp-knob weight into the
# scan tag so runs with different lattices / weights do not overwrite each
# other's figures.
_WEIGHT_TAG = '' if _args.comp_weight == 1e4 else f'_w{_args.comp_weight:g}'
_SCAN_TAG = (
    f'{_scan_range_tag(MAIN_B_SCALE_VALUES)}{INPUT_TAG}{_WEIGHT_TAG}')


def _save_fig(fig, stem):
    path = COUPLING_STUDIES_PLOT_DIR / f'{stem}.pdf'
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved plot: {path}')


_save_fig(fig1, f'IR_coupled_beta_main_b_scale_{FIELD_TAG}_scan{_SCAN_TAG}')
_save_fig(
    fig2, f'full_straight_coupled_beta_main_b_scale_{FIELD_TAG}_scan{_SCAN_TAG}')
_save_fig(fig_beta_ir, f'IR_beta_main_b_scale_{FIELD_TAG}_scan{_SCAN_TAG}')
_save_fig(
    fig_beta_full, f'full_straight_beta_main_b_scale_{FIELD_TAG}_scan{_SCAN_TAG}')
_save_fig(
    fig_disp_ir, f'IR_dispersion_main_b_scale_{FIELD_TAG}_scan{_SCAN_TAG}')
_save_fig(
    fig_disp_full,
    f'full_straight_dispersion_main_b_scale_{FIELD_TAG}_scan{_SCAN_TAG}')
_save_fig(fig3, f'c_minus_vs_main_b_scale_{FIELD_TAG}_scan{_SCAN_TAG}')
_save_fig(
    fig_emit,
    f'eq_emittance_shift_vs_main_b_scale_{FIELD_TAG}_scan{_SCAN_TAG}')
_save_fig(
    fig_stat,
    f'skew_corrector_strength_vs_main_b_scale_{FIELD_TAG}_scan{_SCAN_TAG}')
_save_fig(
    fig_drive,
    f'coupling_drive_vs_main_b_scale_{FIELD_TAG}_scan{_SCAN_TAG}')
_save_fig(
    fig_comp,
    f'comp_b_scale_knobs_vs_main_b_scale_{FIELD_TAG}_scan{_SCAN_TAG}')

if not _args.no_show:
    plt.show()
