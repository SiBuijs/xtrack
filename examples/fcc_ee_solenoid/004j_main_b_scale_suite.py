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

As a function of main_b_scale (2 T and 3 T overlaid on shared axes; every
quantity shown as a difference from the bare ring -- the same lattice with
the solenoid structure and all of its corrections switched off,
set_lattice_knobs(with_solenoids=False, with_correctors=False)):
  - equilibrium emittance shift Deps_x / Deps_y, from a 6D radiative Twiss
    (radiation_analysis=True -> tw.eq_gemitt_x/y, the Chao formalism: the
    equilibrium of the one-turn map with SR damping and excitation, which
    needs the RF on and treats x-y coupling properly). eps_zeta and the
    energy loss per turn are stored and printed too. Radiation integrals are
    not used anywhere in this script. See _chao_equilibrium_emittances.
  - horizontal / vertical tune shift Dqx / Dqy   (Twiss table)
  - horizontal / vertical chromaticity shift DQ'x / DQ'y (Twiss table)
  - the coupling coefficient shift DC^- (tw.c_minus; ~ the absolute value,
    the bare ring being essentially uncoupled)

In the IR (+-20 m) and over the entire accelerator (full ring circumference),
one curve per
main_b_scale value (curve colour = field-strength multiplier), one figure
set per field case:
  - beta_x / beta_y
  - the coupled-mode betas normalised by their primary beta,
    betx2/betx and bety1/bety
  - the coupled-mode betas un-normalised, betx2 / bety1 [m]
  - dispersion D_x / D_y
  - beam sizes sigma_x / sigma_y (tw.get_beam_covariance with the fixed
    design emittances below)
  - phase advance mu_x / mu_y  (units of 2*pi, referenced to the IP)
  - closed orbit x / y

Plus, per field case, at main_b_scale = 1.0 and with the default
(unit-weight, skew-quad-only) coupling correction: every skew coupling
corrector's integrated strength k1s*L as a fraction of the integrated
normal gradient |k1*L| of the quadrupole that corrector is attached to
(004c adds each k1s as a skew component on an existing quad), plotted as
thick red dots against that host quad's longitudinal position s, with the
equivalent host-quad roll angle 0.5*atan(k1s/k1) on the right-hand axis.

Every run (without --replot) pickles the full scan result (per field case:
per-scan-point scalars, packed s-profile arrays, the skew-quad snapshot and
the bare-ring baseline) to data/main_b_scale_suite_<...>.pkl -- labelled by
--b0/--input-tag/--max-transverse-order/--coupling-only and the scan grid,
see _data_path(). --replot reloads that file and only re-runs the plotting
code, skipping the expensive per-point orbit/coupling re-solve entirely.
"""

from pathlib import Path
from types import SimpleNamespace
import argparse
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText
import numpy as np
import xtrack as xt

from aperture_study_io import DATA_DIR as _DATA_DIR
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
        'chromaticity/C- vs main_b_scale, plus IR/entire-accelerator beta, '
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
    '--no-correctors', action='store_true',
    help='Turn off the actively-solved orbit-corrector dipoles and the '
         'coupling skew-quad correction (and skip re-solving them at every '
         'scan point -- their strengths are pinned to exactly 0 instead of '
         'the nominal main_b_scale=1.0 fit loaded from the lattice JSON), '
         'and turn off the optics-rematch knob. The compensation solenoids '
         '(on_comp_sol_{ip}) and the doublet-quad rotation/tilt '
         '(on_rot_doublet_{left,right}_{ip}, which compensates the main '
         "solenoid's own Larmor rotation rather than being an actively-"
         'solved corrector) are left on. Incompatible with --coupling-only '
         '(which only makes sense when correctors are being re-solved).')
_parser.add_argument(
    '--no-show', action='store_true',
    help='Save the figures without opening an interactive window.')
_parser.add_argument(
    '--replot', action='store_true',
    help='Skip the (expensive) scan entirely and reload the scan data saved '
         'by a previous run -- matched on --b0/--input-tag/'
         '--max-transverse-order/--coupling-only/--no-correctors and the '
         'MAIN_B_SCALE_VALUES grid -- to only regenerate the plots. Every '
         'non-replot run saves its data automatically; see _data_path() for '
         'the exact file.')
_args = _parser.parse_args()

if _args.no_correctors and _args.coupling_only:
    raise SystemExit('--no-correctors and --coupling-only are incompatible: '
                      'with correctors off there is nothing left to '
                      "re-solve, coupling-only or otherwise.")

ORDER_TAG = order_tag(_args.max_transverse_order)
INPUT_TAG = f'_{_args.input_tag}' if _args.input_tag else ''
B0_VALUES = list(_args.b0)

HERE = Path(__file__).parent

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']
IP_PLOT = 'ipa'

# main_b_scale scan grid: 21 points, +-1 % about the nominal main-solenoid
# field (as requested). Same span/count as 004f/004h.
MAIN_B_SCALE_VALUES = np.linspace(0.995, 1.005, 21)

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
# Saved-data labelling (for --replot).                       #
##############################################################

def _scan_tag():
    return (f'{round(MAIN_B_SCALE_VALUES.min() * 1000)}-'
            f'{round(MAIN_B_SCALE_VALUES.max() * 1000)}')


def _data_path():
    """Path for this run's pickled scan data -- labelled by every knob that
    changes what's in it, so a mismatched --replot (wrong --b0/--input-tag/
    --max-transverse-order/--coupling-only/--no-correctors) misses the file
    instead of silently loading the wrong scan."""
    b0_tag = ''.join(field_tag(b0) for b0 in B0_VALUES)
    extra = (
        '_nocorr' if _args.no_correctors else
        '_couplingonly' if _args.coupling_only else '')
    return _DATA_DIR / (
        f'main_b_scale_suite_{b0_tag}{ORDER_TAG}{INPUT_TAG}{extra}'
        f'_scan{_scan_tag()}.pkl')


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


def _arc_cell_k1l_reference(table_attr):
    """Median |k1*L| over the arc FODO-cell quads (qf2a.*/qd1a.*). Falls back
    to the median over all quads if that naming isn't found. Same as 004h.
    Only used here to set the "host quad is effectively unpowered" threshold
    in _build_skew_dots (arc quads are never sliced by the near-IP cut_at_s,
    so no host aggregation is needed for this one)."""
    mask = table_attr['element_type'] == 'Quadrupole'
    names = np.asarray([str(n) for n in table_attr['name']])[mask]
    k1l = np.abs(np.asarray(table_attr['k1l'])[mask])
    arc = np.array([n.startswith(('qf2a.', 'qd1a.')) for n in names])
    if arc.sum() < 50:
        return float(np.median(k1l)), int(mask.sum()), 'all quads'
    return float(np.median(k1l[arc])), int(arc.sum()), 'qf2a.*/qd1a.*'


def _host_attr_maps(table_attr):
    """Integrated gradient and length of every element, summed over its
    slices: -> ({host name: k1*L}, {host name: L}).

    Needed because the near-IP `line.cut_at_s` calls slice the thick
    quadrupoles within +-11 m of each IP into `<name>..0`, `<name>..1`, ...
    rows whose `parent_name` is the original element -- the host name itself
    then no longer appears in the table's `name` column at all. Summing by
    parent recovers the host quad's own k1*L (and L) whether or not it was
    sliced, for all 84 coupling-corrector hosts per IP.
    """
    names = np.asarray([str(n) for n in table_attr['name']])
    parents = np.asarray([str(n) for n in table_attr['parent_name']])
    host_names = np.where(np.isin(parents, ('None', '')), names, parents)
    k1l = np.asarray(table_attr['k1l'], dtype=float)
    lengths = np.asarray(table_attr['length'], dtype=float)

    k1l_by_host, l_by_host = {}, {}
    for host, k1l_i, l_i in zip(host_names, k1l, lengths):
        k1l_by_host[host] = k1l_by_host.get(host, 0.0) + k1l_i
        l_by_host[host] = l_by_host.get(host, 0.0) + l_i
    return k1l_by_host, l_by_host


def _build_skew_dots(table_attr, table_uncut, s_ip_ref, k1s_knobs,
                     quad_hosts, k1s_values):
    """Per skew coupling corrector: its integrated strength k1s*L, the host
    quadrupole's own integrated gradient k1*L, and the host's position
    relative to the IP.

    Every corrector is a k1s component added onto an existing quadrupole
    (004c does `env[quad].k1s += env.ref[knob]`), so `k1s*L / |k1*L|_host`
    is a meaningful per-corrector measure of how hard it is driven relative
    to the magnet carrying it (= tan of twice the equivalent host roll).
    The arc-cell reference is kept only to flag hosts whose own gradient is
    ~0 (the unpowered qf1c/qf1d spares), for which that ratio is meaningless.
    """
    k1l_ref, n_ref, ref_label = _arc_cell_k1l_reference(table_attr)
    k1l_by_host, l_by_host = _host_attr_maps(table_attr)

    names, s_rel, k1s_l, k1l_host = [], [], [], []
    for knob, quad in zip(k1s_knobs, quad_hosts):
        length = float(l_by_host.get(quad, np.nan))
        if not np.isfinite(length) or length <= 0:
            print(f'  WARNING: no length found for skew-corrector host quad '
                  f'{quad!r}; dropping it from the skew-corrector figure.')
            continue
        names.append(quad)
        s_rel.append(float(table_uncut['s', quad]) - s_ip_ref)
        k1s_l.append(k1s_values[knob] * length)
        k1l_host.append(float(k1l_by_host.get(quad, np.nan)))

    return dict(
        name=np.asarray(names), s=np.asarray(s_rel),
        k1s_l=np.asarray(k1s_l), k1l_host=np.asarray(k1l_host),
        k1l_ref=k1l_ref, n_ref=n_ref, ref_label=ref_label,
    )


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
# Equilibrium emittances from the 6D radiative Twiss (Chao). #
##############################################################

# Cavity attributes compensate_radiation_energy_loss() writes to (it restores
# voltage/frequency/harmonic itself, but leaves phase_taper/lag_taper set), so
# that each call starts from the pristine RF state rather than from the
# previous scan point's compensation.
_CAVITY_STATE_FIELDS = (
    'voltage', 'frequency', 'lag', 'phase', 'lag_taper', 'phase_taper')

# What _chao_equilibrium_emittances returns, and the NaNs it falls back to.
_CHAO_FIELDS = (
    'chao_eq_gemitt_x', 'chao_eq_gemitt_y', 'chao_eq_gemitt_zeta',
    'chao_energy_loss')
_CHAO_NANS = {ff: np.nan for ff in _CHAO_FIELDS}


def _chao_equilibrium_emittances(line):
    """Equilibrium emittances straight out of a 6D radiative Twiss.

    `twiss(method='6d', radiation_analysis=True)` builds the one-turn map
    with radiation damping and the SR excitation, and reads the equilibrium
    emittances off its eigen-decomposition (the Chao formalism, xtrack's
    `eq_gemitt_x/y/zeta`). It needs the RF on and the longitudinal plane in
    the map, and in exchange it treats x-y coupling and vertical excitation
    properly instead of plane by plane.

    This replaced the radiation integrals (`rad_int_eq_gemitt_*`), which are
    no longer used anywhere here. They agreed on eps_x to ~1 % and on the
    bare uncoupled ring to 0.03 %, but their eps_y is built from the
    vertical dispersion alone and misses the horizontal excitation coupled
    into mode 2: with the solenoid correction on it came out ~20 % low, and
    at 2 T with --no-correctors (C^- ~ 3e-2) ~30x low -- i.e. wrong exactly
    where this study looks.

    Requires the lattice's cavity to be present and radiation configured.
    The line is put in the radiative/tapered state only for the duration of
    the twiss and restored afterwards (in a `finally`, so a failed point
    can't leave the scan's 4D matching in a radiative state):
      - `configure_radiation(model=None)` switches the radiation flags back
        off. The `delta_taper` values `compensate_radiation_energy_loss`
        wrote onto ~16k magnets are then inert -- the tracking code only
        applies them `if (radiation_flag)` -- and are left in place
        (verified: the 4D twiss after the restore reproduces qx/qy/c_minus
        of the one before it to machine precision).
      - the cavity's own RF state is snapshotted and written back.
    """
    cavity_names = [nn for nn in line.element_names
                    if isinstance(line[nn], xt.Cavity)]
    if not cavity_names:
        print('  WARNING: no Cavity in the line -- cannot compute the 6D '
              'radiative equilibrium emittances.')
        return dict(_CHAO_NANS)
    rf_snapshot = {
        nn: {ff: getattr(line[nn], ff) for ff in _CAVITY_STATE_FIELDS}
        for nn in cavity_names}

    try:
        line.configure_radiation(model='mean')
        line.compensate_radiation_energy_loss(verbose=False)
        tw6d = line.twiss(method='6d', radiation_analysis=True)
        return dict(
            chao_eq_gemitt_x=float(tw6d.eq_gemitt_x),
            chao_eq_gemitt_y=float(tw6d.eq_gemitt_y),
            chao_eq_gemitt_zeta=float(tw6d.eq_gemitt_zeta),
            chao_energy_loss=float(tw6d.energy_loss),
        )
    except Exception as exc:  # noqa: BLE001 -- radiative 6D twiss can fail
        print(f'  WARNING: 6D radiative twiss failed ({exc!r}); recording '
              'NaN equilibrium emittances for this point.')
        return dict(_CHAO_NANS)
    finally:
        line.configure_radiation(model=None)
        for nn in cavity_names:
            for ff, value in rf_snapshot[nn].items():
                setattr(line[nn], ff, value)


##############################################################
# Packing a twiss/beam-size table down to what the s-profile plots use.  #
##############################################################

# Columns the _PROFILE_SPECS column-accessor lambdas actually read (see
# _BETA/_COUPLED_BETA/.../_PHASE below). Reducing to just these before
# storing each scan point drops the Line/env references a full TwissTable
# carries, so the saved --replot data is a plain, picklable stack of float
# ndarrays instead of ~1 GB of live-object graph per case.
_TW_PROFILE_FIELDS = (
    's', 'betx', 'bety', 'betx2', 'bety1', 'x', 'y', 'dx', 'dy', 'mux', 'muy')
_BS_PROFILE_FIELDS = ('s', 'sigma_x', 'sigma_y')


def _pack_tw(tw):
    return SimpleNamespace(**{
        f: np.asarray(getattr(tw, f), dtype=float)
        for f in _TW_PROFILE_FIELDS})


def _pack_beam_sizes(beam_sizes):
    return SimpleNamespace(**{
        f: np.asarray(getattr(beam_sizes, f), dtype=float)
        for f in _BS_PROFILE_FIELDS})


##############################################################
# Run one field-strength case end to end.                    #
##############################################################

def _load_case_line(b0):
    """Load one field case's corrected lattice and return the cycled line."""
    lattice_json = HERE / (
        'fccee_z_lcc_splineboris_solenoids_coupling_corrected_'
        f'{field_tag(b0)}{ORDER_TAG}{INPUT_TAG}.json')
    if not lattice_json.exists():
        raise SystemExit(
            f'{lattice_json.name} not found -- build it with\n'
            f'  python 004b_install_solenoids_in_fcc_ring.py --b0 {b0:g} '
            f'--output-tag {_args.input_tag or "mainscale"}\n'
            f'  python 004c_correct_solenoids_in_fcc_ring.py --b0 {b0:g} '
            f'--output-tag {_args.input_tag or "mainscale"}'
        )

    correctors_note = (
        ' [correctors OFF, comp. solenoid ON]' if _args.no_correctors else '')
    print(f'\n=== {field_tag(b0)} main solenoid: loading {lattice_json.name}'
          f'{correctors_note} ===')
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
    return line, lattice_json


def _set_scan_point_knobs(line, main_b_scale, orbit_knobs_by_ip,
                          k1s_knobs_by_ip):
    """Put the lattice in the state one scan point is evaluated in."""
    set_lattice_knobs(
        line, with_solenoids=True, with_correctors=not _args.no_correctors,
        main_b_scale=float(main_b_scale))
    if _args.no_correctors:
        # set_lattice_knobs(with_correctors=False) also turned off the
        # compensation solenoids and the doublet-quad rotation/tilt --
        # neither is an actively-solved "corrector" here (the tilt
        # compensates the main solenoid's own Larmor rotation), so turn
        # them back on. The orbit-corrector dipoles and coupling
        # skew-quads stay pinned to exactly 0, instead of being re-solved
        # at every scan point.
        for ip_name in IP_NAMES:
            line[f'on_comp_sol_{ip_name}'] = 1
            line[f'on_rot_doublet_left_{ip_name}'] = 1
            line[f'on_rot_doublet_right_{ip_name}'] = 1
            for nn in orbit_knobs_by_ip[ip_name]:
                line[nn] = 0.0
            for nn in k1s_knobs_by_ip[ip_name]:
                line[nn] = 0.0


def run_field_case(b0):
    field_t = field_tag(b0)
    line, lattice_json = _load_case_line(b0)
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
    # Full-ring s-extent in the same IP_PLOT-zeroed frame as straight_section_
    # s_range/tw.zero_at(IP_PLOT) above -- table_before_cuts is read right
    # after line.cycle(...), before any cuts, so its 's' column already runs
    # from 0 to the full ring length.
    ring_s_range = (
        float(table_before_cuts['s'].min()) - s_ip_ref,
        float(table_before_cuts['s'].max()) - s_ip_ref,
    )

    # Bare-ring baseline: solenoid structure + every correction off. The
    # scalar-vs-main_b_scale plots are shown as differences from this.
    # Computed before the scan loop so the k1s_*_sol_coupling_corr vars are
    # still their original on_sol_coupling_corr-gated expressions (which
    # line.match would later replace with constants).
    set_lattice_knobs(line, with_solenoids=False, with_correctors=False)
    tw_bare = line.twiss4d(strengths=True)
    baseline = dict(
        qx=float(tw_bare.qx), qy=float(tw_bare.qy),
        dqx=float(getattr(tw_bare, 'dqx', np.nan)),
        dqy=float(getattr(tw_bare, 'dqy', np.nan)),
        c_minus=float(tw_bare.c_minus),
        **_chao_equilibrium_emittances(line),
    )
    print(f'  bare ring: qx={baseline["qx"]:.5f} qy={baseline["qy"]:.5f} '
          f"dqx={baseline['dqx']:.3f} dqy={baseline['dqy']:.3f} "
          f'C-={baseline["c_minus"]:.3e}')
    print(f'    eq. emittance (6D radiative twiss, Chao): '
          f'ex={baseline["chao_eq_gemitt_x"]:.5e} '
          f'ey={baseline["chao_eq_gemitt_y"]:.5e} '
          f'ez={baseline["chao_eq_gemitt_zeta"]:.5e} '
          f'U0={baseline["chao_energy_loss"] * 1e-6:.3f} MeV')

    for ip_name in IP_NAMES:
        line[f'on_sol_{ip_name}'] = 1
        line[f'on_sol_corr_{ip_name}'] = 1

    # Nominal (main_b_scale = 1.0) scan-point index -- its skew-corrector
    # snapshot is captured inline below, while the line is still in that
    # scan point's knob state.
    nominal_idx = int(np.argmin(np.abs(MAIN_B_SCALE_VALUES - 1.0)))
    skew_dots = None

    points = []
    for i, main_b_scale in enumerate(MAIN_B_SCALE_VALUES):
        _set_scan_point_knobs(
            line, main_b_scale, orbit_knobs_by_ip, k1s_knobs_by_ip)
        if not _args.no_correctors:
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
            tw = line.twiss4d(strengths=True)
        except Exception as exc:  # noqa: BLE001 -- coupled optics can fail
            print(f'  WARNING: twiss failed at main_b_scale={main_b_scale:.4f}'
                  f' ({exc!r}); recording NaNs for this point.')
            points.append(dict(
                main_b_scale=float(main_b_scale), tw=None, beam_sizes=None,
                qx=np.nan, qy=np.nan, dqx=np.nan, dqy=np.nan, c_minus=np.nan,
                k1s_values=k1s_values, **_CHAO_NANS))
            continue

        scalars = dict(
            qx=float(tw.qx), qy=float(tw.qy),
            dqx=float(getattr(tw, 'dqx', np.nan)),
            dqy=float(getattr(tw, 'dqy', np.nan)),
            c_minus=float(tw.c_minus),
            **_chao_equilibrium_emittances(line),
        )

        if i == nominal_idx:
            # One IP only (IP_PLOT): the per-IP skew-corrector solutions are
            # near-identical across the 4 IPs, so a single IP's straight
            # section is enough and keeps the s-axis readable. Host gradients
            # come from a fresh attr table rather than from `tw`, so that the
            # slices the near-IP cut_at_s produced can be summed back onto
            # their parent quad (see _host_attr_maps).
            skew_dots = _build_skew_dots(
                line.get_table(attr=True), table_before_cuts, s_ip_ref,
                k1s_knobs_by_ip[IP_PLOT], quad_hosts_by_ip[IP_PLOT],
                k1s_values)

        tw.zero_at(IP_PLOT)
        beam_sizes = tw.get_beam_covariance(
            nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y, gemitt_zeta=GEMITT_ZETA)

        points.append(dict(
            main_b_scale=float(main_b_scale), tw=_pack_tw(tw),
            beam_sizes=_pack_beam_sizes(beam_sizes),
            k1s_values=k1s_values, **scalars))
        print(f'  main_b_scale={main_b_scale:+.4f}: twiss OK '
              f'(qx={scalars["qx"]:.5f} qy={scalars["qy"]:.5f} '
              f'C-={scalars["c_minus"]:.3e}) '
              f'eq. emitt ex={scalars["chao_eq_gemitt_x"]:.4e} '
              f'ey={scalars["chao_eq_gemitt_y"]:.4e} '
              f'ez={scalars["chao_eq_gemitt_zeta"]:.4e}')

    return dict(
        b0=b0, field_tag=field_t, points=points,
        main_range=main_range, comp_ranges=comp_ranges,
        corrector_positions=corrector_positions,
        straight_section_s_range=straight_section_s_range,
        ring_s_range=ring_s_range,
        nominal_idx=nominal_idx, skew_dots=skew_dots, baseline=baseline,
    )


def _rebuild_skew_dots(case):
    """Recompute a pre-2026-09-03 run's skew-corrector snapshot in the
    current (host-quad-relative) format.

    Runs saved before that stored only k1s*L / <|k1L|>_arc ratios, with no
    host gradients -- and silently dropped the ~14 quads per IP that the
    near-IP cut_at_s had sliced, i.e. the whole final-focus doublet. Both
    are recoverable without redoing the (hours-long) scan: the nominal scan
    point's k1s knob values are in the saved data, and the host quads' own
    k1*L depends only on the lattice and the knob state at main_b_scale=1.0,
    neither of which the scan changes. Costs one lattice load per case.
    """
    line, lattice_json = _load_case_line(case['b0'])
    table = line.get_table()
    knobs_by_ip, hosts_by_ip = {}, {}
    for ip_name in IP_NAMES:
        knobs_by_ip[ip_name], hosts_by_ip[ip_name] = (
            _k1s_coupling_knobs_for_ip(
                line, table, ip_name, lattice_json.name))
    orbit_knobs_by_ip = {
        ip_name: _orbit_corrector_knobs_for_ip(
            line, ip_name, lattice_json.name)
        for ip_name in IP_NAMES
    }
    for ip_name in IP_NAMES:
        line[f'on_sol_{ip_name}'] = 1
        line[f'on_sol_corr_{ip_name}'] = 1
    _set_scan_point_knobs(line, 1.0, orbit_knobs_by_ip, knobs_by_ip)

    return _build_skew_dots(
        line.get_table(attr=True), table, table['s', IP_PLOT],
        knobs_by_ip[IP_PLOT], hosts_by_ip[IP_PLOT],
        case['points'][case['nominal_idx']]['k1s_values'])


##############################################################
# Plotting.                                                  #
##############################################################

_NORM = mcolors.Normalize(
    vmin=MAIN_B_SCALE_VALUES.min(), vmax=MAIN_B_SCALE_VALUES.max())
_CMAP = cm.viridis
_SM = cm.ScalarMappable(norm=_NORM, cmap=_CMAP)

# IR s-window. The "entire accelerator" window is per-case (case['ring_s_
# range']), since it depends on the actual ring length, not a fixed constant.
_IR_XLIM = (-20, 20)


def _iter_profiles(case):
    """Yield (main_b_scale, tw, beam_sizes) for the scan points that twissed."""
    for pt in case['points']:
        if pt['tw'] is not None:
            yield pt['main_b_scale'], pt['tw'], pt['beam_sizes']


def _profile_title(case, suffix):
    correctors_note = ' [correctors off]' if _args.no_correctors else ''
    return (f'{IP_PLOT} main solenoid ({case["b0"]:g} T) -- '
            f'main_b_scale scan{suffix}{correctors_note}')


def _add_colorbar(fig, axs):
    fig.colorbar(_SM, ax=axs, label='main_b_scale', fraction=0.06, pad=0.03)


def _decorate_ir(axs, case):
    for ax in axs:
        _mark_solenoid_regions(
            ax, case['main_range'], case['comp_ranges'],
            case['corrector_positions'])


def _decorate_ring(axs, case):
    """Mark IP_PLOT's local straight-section boundary on a full-ring plot,
    for orientation (everything outside those two lines is arc)."""
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
    # Autoscale y to the visible x-window BEFORE adding the axvspan/axvline
    # annotations -- an axvline is a Line2D with y-data [0, 1], which
    # _autoscale_y_to_xlim would otherwise fold into the min/max and blow
    # the vertical scale out (making e.g. the mm-scale IR dispersion panel
    # a flat sliver).
    if autoscale:
        for ax in axs:
            _autoscale_y_to_xlim(ax, xlim)
    if xlim == _IR_XLIM:
        _decorate_ir(axs, case)
    else:
        _decorate_ring(axs, case)
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
_COUPLED_BETA_ABS = (
    (lambda tw, bs: (tw.s, tw.betx2), lambda tw, bs: (tw.s, tw.bety1)),
    (r'$\beta_{x2}$ [m]', r'$\beta_{y1}$ [m]'), True)
# Closed orbit from the twiss (tw.x / tw.y). Not cyclical, but plotted
# against the IP-referenced s from zero_at like everything else.
_ORBIT = (
    (lambda tw, bs: (tw.s, tw.x * 1e3), lambda tw, bs: (tw.s, tw.y * 1e3)),
    (r'$x$ [mm]', r'$y$ [mm]'), True)
_DISP = (
    (lambda tw, bs: (tw.s, tw.dx * 1e3), lambda tw, bs: (tw.s, tw.dy * 1e3)),
    (r'$D_x$ [mm]', r'$D_y$ [mm]'), True)
_BEAMSIZE = (
    (lambda tw, bs: (bs.s, bs.sigma_x * 1e6),
     lambda tw, bs: (bs.s, bs.sigma_y * 1e6)),
    (r'$\sigma_x$ [$\mu$m]', r'$\sigma_y$ [$\mu$m]'), True)
# mux/muy are in units of 2*pi and are cyclical quantities, so tw.zero_at
# (IP_PLOT) has already referenced them to zero at the IP -- the profile is
# the phase advance accumulated from the IP.
_PHASE = (
    (lambda tw, bs: (tw.s, tw.mux), lambda tw, bs: (tw.s, tw.muy)),
    (r'$\mu_x\,/\,2\pi$', r'$\mu_y\,/\,2\pi$'), True)

_PROFILE_SPECS = [
    ('beta', _BETA),
    ('coupled_beta', _COUPLED_BETA),
    ('coupled_beta_abs', _COUPLED_BETA_ABS),
    ('dispersion', _DISP),
    ('beam_size', _BEAMSIZE),
    ('phase_advance', _PHASE),
    ('closed_orbit', _ORBIT),
]


# Host quads whose own |k1*L| is below this fraction of the arc-cell median
# are treated as unpowered and listed separately rather than plotted: their
# k1s/k1 is dominated by dividing by ~0 (up to 4e-2, i.e. a 20 mrad
# "equivalent roll", from |k1s*L| < 1e-6 1/m -- the weakest correctors in
# the whole set), which would flatten the y-scale for everything else.
# The threshold sits in a clean two-decade gap in the actual lattices: the
# four unpowered qf1c/qf1d spares either side of the IP run 0.0006-0.011 of
# the arc-cell median (2 T and 3 T), while the weakest genuinely powered
# host, qf17l.3/qd9l.3, is at 0.065. (004h uses 0.1 for its roll statistics,
# which would also drop those two.)
_WEAK_HOST_FRACTION = 0.03


def _skew_dot_fig(case):
    sd = case['skew_dots']
    k1l_host = np.abs(sd['k1l_host'])
    weak = ~(k1l_host > _WEAK_HOST_FRACTION * sd['k1l_ref'])
    ratio = np.full(k1l_host.shape, np.nan)
    ratio[~weak] = sd['k1s_l'][~weak] / k1l_host[~weak]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.axhline(0.0, color='0.5', linewidth=0.8)
    ax.axvline(0.0, color='0.7', linewidth=0.8, linestyle='--')
    ax.text(0.0, 1.0, f' {IP_PLOT}', transform=ax.get_xaxis_transform(),
            va='top', ha='left', fontsize=8, color='0.4')
    ax.plot(sd['s'][~weak], ratio[~weak], linestyle='none', marker='o',
            markersize=8, color='red')
    ax.set_xlabel(r'$s - s_{\mathrm{IP}}$ [m]  (host quadrupole position)')
    ax.set_ylabel(r'$k_{1s}L \,/\, |k_1 L|_{\mathrm{host\ quad}}$')
    ax.set_title(
        f'{IP_PLOT} skew coupling-corrector integrated strength '
        f'({case["b0"]:g} T main solenoid, main_b_scale = 1.0, '
        f'unit-weight correction)\n'
        'relative to the integrated gradient of the quadrupole each '
        'corrector sits on')
    # Same number read as the equivalent roll of the host quad: a quad rolled
    # by phi has k1s/k1 = tan(2 phi).
    sec = ax.secondary_yaxis(
        'right',
        functions=(lambda r: 0.5 * np.arctan(r) * 1e3,
                   lambda phi: np.tan(2.0 * phi * 1e-3)))
    sec.set_ylabel('equivalent host-quad roll [mrad]')
    if weak.any():
        ax.text(
            0.5, -0.22,
            f'not shown: {int(weak.sum())} corrector(s) on unpowered host '
            r'quads ($|k_1L|_{\mathrm{main}} < $'
            f'{_WEAK_HOST_FRACTION:.0%} of the arc-cell median), all with '
            r'$|k_{1s}L| \leq $'
            f'{np.max(np.abs(sd["k1s_l"][weak])):.1e} 1/m:\n'
            + ', '.join(sd['name'][weak]),
            transform=ax.transAxes, ha='center', va='top', fontsize=7,
            color='0.35')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _fmt_bare(v):
    v = float(v)
    if v == 0.0 or 1e-3 <= abs(v) < 1e5:
        return f'{v:.5f}'
    return f'{v:.4e}'


def _place_legend_clear_of_bare_ring_box(ax):
    """The bare-ring box is anchored upper-left, and `loc='best'` doesn't see
    AnchoredText artists -- on a rising curve it puts the legend right on top
    of it. Pin the legend opposite and add headroom so neither lands on the
    data."""
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ymargin(0.30)
    ax.autoscale_view()


def _add_bare_ring_box(ax, rows, loc='upper left'):
    """rows: list of (label, formatted_value_str). Framed reference box in a
    corner of the axes showing the bare-ring value(s) each curve is a
    difference from."""
    text = 'bare ring\n' + '\n'.join(f'{lab}: {val}' for lab, val in rows)
    at = AnchoredText(text, loc=loc, frameon=True, pad=0.4, borderpad=0.4,
                      prop=dict(size=7, family='monospace'))
    at.patch.set(boxstyle='round', facecolor='white', edgecolor='0.6',
                 alpha=0.9)
    at.set_zorder(5)
    ax.add_artist(at)


def _scalar_overlay_fig(cases, panel_specs, suptitle):
    """panel_specs: list of (key, ylabel, scale). One panel each, 2 T and 3 T
    overlaid. Each quantity is plotted as (scan value - bare-ring value)."""
    n = len(panel_specs)
    fig, axs = plt.subplots(n, 1, sharex=True, figsize=(7.0, 2.6 * n + 0.6))
    if n == 1:
        axs = [axs]
    for ax, (key, ylabel, scale) in zip(axs, panel_specs):
        for case in cases:
            x = MAIN_B_SCALE_VALUES
            base = case['baseline'][key]
            y = (np.array([pt[key] for pt in case['points']]) - base) * scale
            ax.plot(x, y, '-o', color=_B0_COLORS.get(case['b0'], None),
                    label=f'{case["b0"]:g} T')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        _place_legend_clear_of_bare_ring_box(ax)
        _add_bare_ring_box(
            ax, [(f'{case["b0"]:g} T', _fmt_bare(case['baseline'][key]))
                 for case in cases])
    axs[-1].set_xlabel('main_b_scale')
    axs[0].set_title(suptitle)
    fig.tight_layout()
    return fig


def _emittance_overlay_fig(cases):
    """Returns None (and says so) for scan data saved before the 6D
    radiative-Twiss emittances went in on 2026-09-03: those runs only stored
    the radiation-integral values, which are no longer plotted. Every other
    figure still replots from such a file."""
    key_x, key_y = 'chao_eq_gemitt_x', 'chao_eq_gemitt_y'
    if not all(key_x in case['baseline'] for case in cases):
        print('NOTE: the loaded scan data predates the 6D radiative-Twiss '
              'equilibrium emittances, so the emittance figure is skipped. '
              'Re-run without --replot to produce it.')
        return None
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.0, 6.4))
    for case in cases:
        bx = case['baseline'][key_x]
        by = case['baseline'][key_y]
        ex = np.array([pt[key_x] for pt in case['points']])
        ey = np.array([pt[key_y] for pt in case['points']])
        color = _B0_COLORS.get(case['b0'], None)
        axs[0].plot(MAIN_B_SCALE_VALUES, (ex - bx) * 1e9, '-o',
                    color=color, label=f'{case["b0"]:g} T')
        axs[1].plot(MAIN_B_SCALE_VALUES, (ey - by) * 1e12, '-o',
                    color=color, label=f'{case["b0"]:g} T')
    axs[0].set_ylabel(r'$\Delta\varepsilon_{x,\mathrm{eq}}$ [nm]')
    axs[1].set_ylabel(r'$\Delta\varepsilon_{y,\mathrm{eq}}$ [pm]')
    axs[1].set_xlabel('main_b_scale')
    axs[0].set_title(
        'Equilibrium emittance shift vs main_b_scale '
        '(relative to the bare ring)\n6D radiative Twiss, Chao formalism')
    _add_bare_ring_box(
        axs[0], [(f'{case["b0"]:g} T',
                  _fmt_bare(case['baseline'][key_x] * 1e9) + ' nm')
                 for case in cases])
    _add_bare_ring_box(
        axs[1], [(f'{case["b0"]:g} T',
                  _fmt_bare(case['baseline'][key_y] * 1e12) + ' pm')
                 for case in cases])
    for ax in axs:
        ax.grid(True)
        _place_legend_clear_of_bare_ring_box(ax)
    fig.tight_layout()
    return fig


##############################################################
# Main.                                                      #
##############################################################

def main():
    data_path = _data_path()
    if _args.replot:
        if not data_path.exists():
            raise SystemExit(
                f'--replot: no saved scan data at {data_path}\n'
                'Run the suite once without --replot (with the same --b0 / '
                '--input-tag / --max-transverse-order / --coupling-only) to '
                'create it first.')
        print(f'--replot: loading saved scan data from {data_path}')
        with open(data_path, 'rb') as f:
            cases = pickle.load(f)
        for case in cases:
            if case['skew_dots'] is not None and 'k1l_host' not in (
                    case['skew_dots']):
                print(f'--replot: {case["field_tag"]} skew-corrector data is '
                      'in the old arc-relative format; recomputing the host '
                      'quadrupole gradients from the lattice.')
                case['skew_dots'] = _rebuild_skew_dots(case)
    else:
        cases = [run_field_case(b0) for b0 in B0_VALUES]
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(data_path, 'wb') as f:
            pickle.dump(cases, f)
        print(f'Saved scan data: {data_path}')

    plt.close('all')
    figs = {}  # stem -> figure

    # --- Scalar quantities vs main_b_scale, 2 T + 3 T overlaid. ---
    emittance_fig = _emittance_overlay_fig(cases)
    if emittance_fig is not None:
        figs['eq_emittance_shift_vs_main_b_scale'] = emittance_fig
    figs['tunes_vs_main_b_scale'] = _scalar_overlay_fig(
        cases,
        [('qx', r'$\Delta q_x$', 1.0), ('qy', r'$\Delta q_y$', 1.0)],
        'Betatron tune shift vs main_b_scale (relative to the bare ring)')
    figs['chromaticity_vs_main_b_scale'] = _scalar_overlay_fig(
        cases,
        [('dqx', r"$\Delta Q'_x$", 1.0), ('dqy', r"$\Delta Q'_y$", 1.0)],
        'Linear chromaticity shift vs main_b_scale (relative to the bare ring)')
    figs['c_minus_vs_main_b_scale'] = _scalar_overlay_fig(
        cases, [('c_minus', r'$\Delta C^-$', 1.0)],
        r'Coupling coefficient shift $\Delta C^-$ vs main_b_scale '
        '(relative to the bare ring)')

    # --- Per-field-case s-profiles (colour = main_b_scale). ---
    for case in cases:
        ft = case['field_tag']
        for name, (fns, labels, autoscale) in _PROFILE_SPECS:
            top_fn, bot_fn = fns
            top_label, bot_label = labels
            figs[f'IR_{name}_{ft}'] = _make_profile_fig(
                case, _IR_XLIM, ' (interaction region)', top_fn, bot_fn,
                top_label, bot_label, autoscale=autoscale)
            figs[f'full_ring_{name}_{ft}'] = _make_profile_fig(
                case, case['ring_s_range'], ' (entire accelerator)',
                top_fn, bot_fn, top_label, bot_label, autoscale=autoscale)
        if case['skew_dots'] is not None:
            figs[f'skew_corrector_strength_{ft}'] = _skew_dot_fig(case)

    # --- Save. ---
    plot_dir = _BASE_PLOT_DIR / 'Coupling_Studies' / 'main_b_scale_suite'
    plot_dir.mkdir(parents=True, exist_ok=True)
    scan_tag = _scan_tag()
    for stem, fig in figs.items():
        path = plot_dir / f'{stem}_scan{scan_tag}.pdf'
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved plot: {path}')

    if not _args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
