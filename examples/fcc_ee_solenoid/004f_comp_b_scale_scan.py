"""Scan the comp_b_scale knob (see lattice_knobs.set_lattice_knobs and
004b_install_solenoids_in_fcc_ring.py) and plot betx2/betx and bety1/bety
(the coupled-mode beta functions, normalized by the primary-mode beta at the
same s, so the plotted ratio reflects coupling strength rather than local
optics amplitude) for each value, over the same two s-ranges
004d_analysis_and_plots.py uses: the local
interaction-region view (+-20 m) and the full straight section out to the
coupling-correction target (+-1362 m or so, from
end_ds_start_straight_{ip}/end_straight_start_ds_{ip}).

comp_b_scale=1.0 is nominal compensation (net main+compensation solenoid
integral cancels, as designed by 004a/004b). The coupling-correction knobs
(on_sol_coupling_corr_{ip}) were first solved in 004c at that nominal value,
but here their underlying k1s_*_sol_coupling_corr skew-quad knobs are
re-solved fresh at every comp_b_scale value (see _resolve_coupling_correction
below, same targets/vary set as 004c's opt_coupling block), so the plots show
how well the coupling correction can track a compensation solenoid thrown
off-balance, rather than how much residual coupling appears under a
correction that assumes it is exact (the latter is what the (removed)
Bz-ramp coupling study did with a fixed correction -- see
claude_notes/04_bz_ramp_coupling_amplification.md -- and what this script
itself did before skew-quad re-correction was added here).

Requires a corrected lattice built with the comp_b_scale knob, i.e. a
004b_install_solenoids_in_fcc_ring.py / 004c_correct_solenoids_in_fcc_ring.py
rebuild done after that knob was added -- see the runtime check below.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import xtrack as xt

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
        'Scan comp_b_scale and plot betx2/bety1 over the IR and full '
        'straight-section regions.'))
add_b0_argument(_parser, default=MAIN_SOLENOID_B0)
add_max_order_argument(_parser)
_args = _parser.parse_args()
FIELD_TAG = field_tag(_args.b0)
ORDER_TAG = order_tag(_args.max_transverse_order)

HERE = Path(__file__).parent
INPUT_LATTICE_JSON = (
    HERE / (
        'fccee_z_lcc_splineboris_solenoids_coupling_corrected_'
        f'{FIELD_TAG}{ORDER_TAG}.json'
    )
)

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']
IP_PLOT = 'ipa'

# Five comp_b_scale values, evenly spaced between 0.95 and 1.05. Kept small
# (rather than the original 11) because each point now re-solves the
# on_sol_coupling_corr_{ip} skew-quad knobs for all 4 IPs -- a single
# opt_coupling.solve(rcond=3e-3) pass took ~8.5 minutes per IP per scale
# value in testing (84 vary knobs, numerical Jacobian), so an 11-point x
# 4-IP scan would take on the order of hours even with the warm-start
# continuation across scale values below.
COMP_B_SCALE_VALUES = np.linspace(0.95, 1.05, 5)

# Quads carrying the orbit correctors (corr_1..4_{left,right}_on_quad in
# 004c_correct_solenoids_in_fcc_ring.py's per-IP `config` dict) -- copied here
# for plot annotation only, same duplication pattern as the config dict
# itself and as 004d_analysis_and_plots.py (see
# claude_notes/01_lattice_construction_000_004d.md).
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

# Work on a copy, so extra cuts used only for plotting do not alter the
# environment loaded from JSON.
line = env.fccee_p_ring.copy(shallow=True)
line.particle_ref.anomalous_magnetic_moment = 0.00115965218128

if 'comp_b_scale' not in line.vars:
    raise SystemExit(
        f'{INPUT_LATTICE_JSON.name} has no comp_b_scale knob -- it was '
        'built before that knob was added to '
        '004b_install_solenoids_in_fcc_ring.py. Rebuild it via '
        '`python 004b_install_solenoids_in_fcc_ring.py` followed by '
        '`python 004c_correct_solenoids_in_fcc_ring.py` (same --b0/'
        '--max-transverse-order flags as here, if non-default), then rerun '
        'this script.'
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
# Same approach as 004d_analysis_and_plots.py: read the actual installed
# SplineBoris slice-chain extents from the table and shrink them to the
# physical (engineering) device length, since the slice chain is padded with
# a field taper beyond the real solenoid -- see 004a/spline_boris_setup.py.

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


def _mark_solenoid_regions(ax, main_range, comp_ranges, corrector_positions):
    s_start, s_end = main_range
    ax.axvline(s_start, color='red', linewidth=0.8, linestyle='--')
    ax.axvline(s_end, color='red', linewidth=0.8, linestyle='--')
    for s_start, s_end in comp_ranges:
        ax.axvline(s_start, color='orange', linewidth=0.8, linestyle='--')
        ax.axvline(s_end, color='orange', linewidth=0.8, linestyle='--')
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
# Per-IP skew-quad (coupling) re-correction, one comp_b_scale at a  #
# time -- same targets/vary set as 004c_correct_solenoids_in_fcc_   #
# ring.py's opt_coupling block, but re-solved fresh at each scan    #
# point instead of staying frozen at its comp_b_scale=1.0 solution. #
#####################################################################
# This changes what the scan measures relative to the module docstring above:
# instead of showing how much residual coupling appears under a correction
# that assumes comp_b_scale=1.0 is exact, it shows how well the correction
# can track a compensation solenoid thrown off-balance, and whether the
# skew-quad knobs run out of strength/authority to do so.

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
            '004c_correct_solenoids_in_fcc_ring.py (which creates the '
            'k1s_*_sol_coupling_corr vars this scan re-solves).'
        )
    return knob_names


# Built from table_before_cuts (not the cut `line`) so the quad names match
# exactly what 004c used to build the k1s_*_sol_coupling_corr knob names --
# line.cut_at_s above only splits drift-like regions far from these quads,
# but the pre-cut table removes any doubt.
K1S_KNOBS_BY_IP = {
    ip_name: _k1s_coupling_knobs_for_ip(table_before_cuts, ip_name)
    for ip_name in IP_NAMES
}


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
        # See 004c: the first pass can leave 1-2 targets just outside tol
        # even though the knob is well-behaved; take_best (solve()'s
        # default) keeps the best point found regardless.
        assert_within_tol=False,
        vary=xt.VaryList(K1S_KNOBS_BY_IP[ip_name], step=1e-6),
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
    # Same rank-deficient-Jacobian truncation as 004c (many more k1s knobs
    # than targets).
    opt_coupling.solve(rcond=3e-3)
    status = opt_coupling.target_status(ret=True)
    if not all(status.tol_met):
        print(f'  WARNING: on_sol_coupling_corr_{ip_name} re-fit did not '
              'fully converge to tolerance; using best point found.')


############################################
# Twiss once per comp_b_scale, solenoids on #
############################################

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 1
    line[f'on_sol_corr_{ip_name}'] = 1

TWISS_BY_COMP_B_SCALE = []
for comp_b_scale in COMP_B_SCALE_VALUES:
    set_lattice_knobs(
        line, with_solenoids=True, with_correctors=True,
        comp_b_scale=float(comp_b_scale))
    # Warm-started from whichever comp_b_scale was solved previously (the
    # scan is monotonic in COMP_B_SCALE_VALUES), rather than reset to the
    # nominal comp_b_scale=1.0 solution each time.
    for ip_name in IP_NAMES:
        _resolve_coupling_correction(ip_name)
    tw = line.twiss4d(strengths=True)
    tw.zero_at(IP_PLOT)
    TWISS_BY_COMP_B_SCALE.append(tw)
    print(f'comp_b_scale={comp_b_scale:+.2f}: twiss OK (skew quads re-solved)')


#########
# Plots #
#########

plt.close('all')

_norm = mcolors.Normalize(
    vmin=COMP_B_SCALE_VALUES.min(), vmax=COMP_B_SCALE_VALUES.max())
_cmap = cm.viridis
_sm = cm.ScalarMappable(norm=_norm, cmap=_cmap)


def _plot_betx2_bety1_scan(axs, xlim, title_suffix):
    for comp_b_scale, tw in zip(COMP_B_SCALE_VALUES, TWISS_BY_COMP_B_SCALE):
        color = _cmap(_norm(comp_b_scale))
        axs[0].plot(tw.s, tw.betx2 / tw.betx, color=color)
        axs[1].plot(tw.s, tw.bety1 / tw.bety, color=color)
    axs[0].set_ylabel(r'$\beta_{x2}/\beta_x$')
    axs[1].set_ylabel(r'$\beta_{y1}/\beta_y$')
    axs[0].set_title(f'{IP_PLOT} main solenoid -- comp_b_scale scan{title_suffix}')
    for ax in axs:
        ax.grid(True)
    axs[-1].set_xlabel('s [m]')
    axs[-1].set_xlim(*xlim)


fig1, axs1 = plt.subplots(2, 1, sharex=True, figsize=(8.0, 6.4))
_plot_betx2_bety1_scan(axs1, (-20, 20), ' (interaction region)')
for ax in axs1:
    _mark_solenoid_regions(
        ax, MAIN_SOLENOID_S_RANGE, COMP_SOLENOID_S_RANGES,
        CORRECTOR_QUAD_S_POSITIONS)
fig1.subplots_adjust(hspace=0.15, top=0.92, bottom=0.1, left=0.12, right=0.88)
fig1.colorbar(
    _sm, ax=axs1, label='comp_b_scale', fraction=0.06, pad=0.03)

fig2, axs2 = plt.subplots(2, 1, sharex=True, figsize=(8.0, 6.4))
_plot_betx2_bety1_scan(axs2, (-1400, 1400), ' (full straight section)')
for ax in axs2:
    for s_pos in STRAIGHT_SECTION_S_RANGE:
        ax.axvline(s_pos, color='black', linewidth=0.8, linestyle=':')
fig2.subplots_adjust(hspace=0.15, top=0.92, bottom=0.1, left=0.12, right=0.88)
fig2.colorbar(
    _sm, ax=axs2, label='comp_b_scale', fraction=0.06, pad=0.03)

# C_minus (closest-tune-approach coupling coefficient) is a single
# whole-ring scalar per twiss, not an s-dependent quantity like betx2/bety1
# above -- it is already integrated around the entire ring (all 4 IPs'
# solenoids perturbed together by the same comp_b_scale, since comp_b_scale
# is one global knob, see 004b_install_solenoids_in_fcc_ring.py), so this
# plot is not specific to IP_PLOT the way fig1/fig2 are.
C_MINUS_VALUES = np.array([tw.c_minus for tw in TWISS_BY_COMP_B_SCALE])

fig3, ax3 = plt.subplots(figsize=(6.4, 4.8))
ax3.plot(COMP_B_SCALE_VALUES, C_MINUS_VALUES, '-o', color='C0')
ax3.set_xlabel('comp_b_scale')
ax3.set_ylabel(r'$C_-$')
ax3.set_title('Closest-tune-approach coupling coefficient vs comp_b_scale')
ax3.grid(True)
fig3.tight_layout()

print(f'Loaded {INPUT_LATTICE_JSON}')
print(f'comp_b_scale values: {COMP_B_SCALE_VALUES}')

plt.show()
