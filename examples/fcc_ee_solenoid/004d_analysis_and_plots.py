from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from matplotlib.patches import Patch

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
    description='Analyze and plot the corrected SplineBoris FCC lattice.')
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

# Quads carrying the orbit correctors (corr_1..4_{left,right}_on_quad in
# 004c_correct_solenoids_in_fcc_ring.py's per-IP `config` dict) -- copied here
# for plot annotation only, same duplication pattern as the config dict
# itself (see claude_notes/01_lattice_construction_000_004d.md).
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

# The full final-focus doublet rotated between the main and compensation
# solenoids (doublet_quad_left/right in 004c_correct_solenoids_in_fcc_ring.py's
# per-IP `config` dict -- rot_s_rad = +-phi_rot_doublet there). This is a
# superset of CORRECTOR_QUADS_BY_IP above: qd0c/qf1c/qf1d are rotated too but
# do not carry an orbit corrector, so they were previously not marked at all.
DOUBLET_QUADS_BY_IP = {
    'ipa': ['qd0ar.0', 'qd0br.0', 'qd0cr.0', 'qf1ar.0', 'qf1br.0',
            'qf1cr.0', 'qf1dr.0',
            'qd0al.3', 'qd0bl.3', 'qd0cl.3', 'qf1al.3', 'qf1bl.3',
            'qf1cl.3', 'qf1dl.3'],
    'ipd': ['qd0ar.1', 'qd0br.1', 'qd0cr.1', 'qf1ar.1', 'qf1br.1',
            'qf1cr.1', 'qf1dr.1',
            'qd0al.0', 'qd0bl.0', 'qd0cl.0', 'qf1al.0', 'qf1bl.0',
            'qf1cl.0', 'qf1dl.0'],
    'ipg': ['qd0ar.2', 'qd0br.2', 'qd0cr.2', 'qf1ar.2', 'qf1br.2',
            'qf1cr.2', 'qf1dr.2',
            'qd0al.1', 'qd0bl.1', 'qd0cl.1', 'qf1al.1', 'qf1bl.1',
            'qf1cl.1', 'qf1dl.1'],
    'ipj': ['qd0ar.3', 'qd0br.3', 'qd0cr.3', 'qf1ar.3', 'qf1br.3',
            'qf1cr.3', 'qf1dr.3',
            'qd0al.2', 'qd0bl.2', 'qd0cl.2', 'qf1al.2', 'qf1bl.2',
            'qf1cl.2', 'qf1dl.2'],
}


################################
# Load and prepare the lattice #
################################

env = xt.load(INPUT_LATTICE_JSON)

# Work on a copy, so extra cuts used only for plotting do not alter the
# environment loaded from JSON.
line = env.fccee_p_ring.copy(shallow=True)
line.particle_ref.anomalous_magnetic_moment = 0.00115965218128

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


#####################################
# Twiss with solenoids/corrections off #
#####################################

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 0
    line[f'on_sol_corr_{ip_name}'] = 0

tw_off = line.twiss6d(strengths=True)


####################################
# Twiss with solenoids/corrections on #
####################################

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 1
    line[f'on_sol_corr_{ip_name}'] = 1

tw4d = line.twiss4d(
    strengths=True,
    polarization_analysis=True,
    radiation_integrals=True,
)
tw = line.twiss6d(strengths=True, polarization_analysis=True)
two = line.twiss(betx=tw_off.betx[0], bety=tw_off.bety[0])


######################
# Radiation analysis #
######################

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()
tw_rad = line.twiss6d(strengths=True, radiation_analysis=True)

energy_eV = (
    tw_rad.ptau * line.particle_ref.p0c[0]
    + line.particle_ref.energy0[0]
)
dE_eV = -np.diff(energy_eV, append=energy_eV[-1])
length = tw.length
mask_len = length > 0
dE_ds_eV_per_m = np.zeros_like(dE_eV)
dE_ds_eV_per_m[mask_len] = dE_eV[mask_len] / length[mask_len]
dE_ds_eV_per_m[dE_ds_eV_per_m < 0] = 0.0

tw_off.zero_at(IP_PLOT)
tw.zero_at(IP_PLOT)


##############################################################
# Element s-positions (in the tw.s frame) for plot annotation #
##############################################################

def _region_s_extent(table, s_ip_ref, env_name_prefix):
    """(s_start, s_end) of all elements in `table` whose env_name starts with
    the given prefix, shifted into a frame zeroed at s_ip_ref. This is the
    extent of the installed SplineBoris slice chain, which is padded with a
    field taper beyond the physical device -- see _shrink_to_physical_extent
    below."""
    env_names = table['env_name'].astype(str)
    mask = np.char.startswith(env_names, env_name_prefix)
    s_starts = table['s_start'][mask]
    s_ends = table['s_end'][mask]
    return s_starts.min() - s_ip_ref, s_ends.max() - s_ip_ref


def _shrink_to_physical_extent(padded_extent, physical_length):
    """The installed slice chain spans a field-sampling axis that is padded
    symmetrically beyond the physical device (so the fitted field can taper
    to exactly 0 at the array ends -- see 004a/spline_boris_setup.py). The
    physical device itself is centered in that padded region, so shrink
    symmetrically down to its true (engineering) length."""
    padded_start, padded_end = padded_extent
    taper_pad = ((padded_end - padded_start) - physical_length) / 2.0
    return padded_start + taper_pad, padded_end - taper_pad


def _compute_marker_positions(table, ip_plot, b0):
    """(main_solenoid_range, comp_solenoid_ranges, corrector_quad_positions)
    for the given lattice `table` (as returned by line.get_table(), taken
    before any cuts), in a frame zeroed at ip_plot -- matches the frame
    produced by tw.zero_at(ip_plot) for a twiss computed on the same line
    before it was cut, since cutting only adds markers in drift regions and
    does not move existing elements."""
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
    doublet_positions = [
        table['s', name] - s_ip_ref
        for name in DOUBLET_QUADS_BY_IP[ip_plot]
    ]
    return main_range, comp_ranges, corrector_positions, doublet_positions


(MAIN_SOLENOID_S_RANGE, COMP_SOLENOID_S_RANGES, CORRECTOR_QUAD_S_POSITIONS,
 DOUBLET_QUAD_S_POSITIONS) = (
    _compute_marker_positions(table_before_cuts, IP_PLOT, _args.b0)
)


def _mark_solenoid_regions(
        ax, main_range, comp_ranges, corrector_positions, doublet_positions):
    """Mark main solenoid / compensation solenoid extents (shaded spans),
    the rotated final-focus doublet quads (thin purple dotted lines), and
    corrector-quad locations (dashed grey lines, a subset of the doublet) on
    an s-axis plot -- shading matches 004h_main_b_scale_scan.py's (and
    onward) style."""
    ax.axvspan(*main_range, color='red', alpha=0.15, linewidth=0)
    for comp_range in comp_ranges:
        ax.axvspan(*comp_range, color='orange', alpha=0.15, linewidth=0)
    for s_pos in doublet_positions:
        ax.axvline(s_pos, color='purple', linewidth=0.6, linestyle=':')
    for s_pos in corrector_positions:
        ax.axvline(s_pos, color='grey', linewidth=0.8, linestyle='--')


#########
# Plots #
#########

plt.close('all')

fig1 = plt.figure(figsize=(6.4, 4.8 * 1.8))
ax1 = fig1.add_subplot(5, 1, 1)
tw_off.plot(ax=ax1)

ax2 = fig1.add_subplot(5, 1, 2, sharex=ax1)
ax2.plot(tw.s, tw.bs)
ax2.set_ylabel(r'$B_s$ [T]')
ax2.grid(True)

ax3 = fig1.add_subplot(5, 1, 3, sharex=ax1)
ax3.plot(tw.s, tw.y * 1e3)
ax3.set_ylabel('y [mm]')
ax3.set_ylim(-0.2, 0.2)
ax3.grid(True)

ax4 = fig1.add_subplot(5, 1, 4, sharex=ax1)
ax4.plot(tw.s, tw.dy * 1e3)
ax4.set_ylabel(r'$D_y$ [mm]')
ax4.set_ylim(-0.2, 0.2)
ax4.grid(True)

ax5 = fig1.add_subplot(5, 1, 5, sharex=ax1)
ax5.plot(tw.s, tw.betx2, label=r'$\beta_{x2}$')
ax5.plot(tw.s, tw.bety1, label=r'$\beta_{y1}$')
ax5.set_ylabel(r'$\beta_{x2,y1}$')
ax5.legend(loc='best')
ax5.grid(True)

for _ax in (ax1, ax2, ax3, ax4, ax5):
    _mark_solenoid_regions(
        _ax, MAIN_SOLENOID_S_RANGE, COMP_SOLENOID_S_RANGES,
        CORRECTOR_QUAD_S_POSITIONS, DOUBLET_QUAD_S_POSITIONS)

ax1.set_xlabel('')
ax5.set_xlabel('s [m]')
fig1.subplots_adjust(hspace=0.25, top=0.95, bottom=0.06, left=0.14)
ax5.set_xlim(-20, 20)

fig2 = plt.figure(figsize=(6.4, 4.8 * 1.8))
ax1 = fig2.add_subplot(5, 1, 1)
tw_off.plot(ax=ax1)

ax2 = fig2.add_subplot(5, 1, 2, sharex=ax1)
ax2.plot(tw.s, tw.bs)
ax2.set_ylabel(r'$B_s$ [T]')
ax2.grid(True)

ax3 = fig2.add_subplot(5, 1, 3, sharex=ax1)
ax3.plot(tw.s, dE_ds_eV_per_m / 1e6)
ax3.set_ylabel(r'dE/ds [MeV/m]')
ax3.grid(True)

ax4 = fig2.add_subplot(5, 1, 4, sharex=ax1)
ax4.plot(tw.s, tw.spin_y)
ax4.set_ylabel(r'spin y')
ax4.grid(True)

ax5 = fig2.add_subplot(5, 1, 5, sharex=ax1)
ax5.plot(tw.s, tw.spin_x, label='spin x')
ax5.plot(tw.s, tw.spin_z, label='spin z')
ax5.set_ylabel(r'spin x, z')
ax5.legend(loc='best')
ax5.grid(True)

for _ax in (ax1, ax2, ax3, ax4, ax5):
    _mark_solenoid_regions(
        _ax, MAIN_SOLENOID_S_RANGE, COMP_SOLENOID_S_RANGES,
        CORRECTOR_QUAD_S_POSITIONS, DOUBLET_QUAD_S_POSITIONS)

ax1.set_xlabel('')
ax5.set_xlabel('s [m]')
fig2.subplots_adjust(hspace=0.25, top=0.95, bottom=0.06, left=0.14)
ax5.set_xlim(-20, 20)

###########################################
# 2T vs 3T betx2/bety1 comparison figure #
###########################################

COMPARISON_TAGS = ['2T', '3T']
COMPARISON_B0_BY_TAG = {'2T': 2.0, '3T': 3.0}


def _twiss_on_for_tag(tag):
    """Coupled-optics twiss (solenoids+corrections on) and marker positions
    for a given field tag. The main-solenoid half-length differs between the
    2T and 3T cases (see solenoid_params.half_length_for_b0), so markers are
    recomputed per tag rather than reusing the primary case's."""
    if tag == FIELD_TAG:
        return tw, (
            MAIN_SOLENOID_S_RANGE, COMP_SOLENOID_S_RANGES,
            CORRECTOR_QUAD_S_POSITIONS, DOUBLET_QUAD_S_POSITIONS)

    input_json = (
        HERE / f'fccee_z_lcc_splineboris_solenoids_coupling_corrected_{tag}.json'
    )
    env_cmp = xt.load(input_json)
    line_cmp = env_cmp.fccee_p_ring.copy(shallow=True)
    line_cmp.particle_ref.anomalous_magnetic_moment = 0.00115965218128

    line_cmp.cycle(f'end_ds_start_straight_{IP_NAMES[0]}')
    table_cmp = line_cmp.get_table()
    for ip_name in IP_NAMES:
        s_cut_right = np.arange(
            table_cmp['s', ip_name] + 2.4, table_cmp['s', ip_name] + 11.0, 0.2,
        )
        line_cmp.cut_at_s(s_cut_right)

        s_cut_left = np.arange(
            table_cmp['s', ip_name] - 11.0, table_cmp['s', ip_name] - 2.4, 0.2,
        )
        line_cmp.cut_at_s(s_cut_left)

    for ip_name in IP_NAMES:
        line_cmp[f'on_sol_{ip_name}'] = 1
        line_cmp[f'on_sol_corr_{ip_name}'] = 1

    tw_cmp = line_cmp.twiss4d(strengths=True)
    tw_cmp.zero_at(IP_PLOT)
    markers_cmp = _compute_marker_positions(
        table_cmp, IP_PLOT, COMPARISON_B0_BY_TAG[tag])
    return tw_cmp, markers_cmp


# Computed once per tag and reused for both the local-region figure (fig3)
# and the full-straight-section figure (fig4) below, since each call rebuilds
# and re-twisses a comparison lattice and is not cheap to repeat.
_TAG_RESULTS = [_twiss_on_for_tag(tag) for tag in COMPARISON_TAGS]

fig3, axs3 = plt.subplots(
    len(COMPARISON_TAGS), 1, sharex=True, figsize=(6.4, 4.8),
)
for ax, tag, (tw_tag, markers_tag) in zip(axs3, COMPARISON_TAGS, _TAG_RESULTS):
    ax.plot(tw_tag.s, tw_tag.betx2, label=r'$\beta_{x2}$')
    ax.plot(tw_tag.s, tw_tag.bety1, label=r'$\beta_{y1}$')
    ax.set_ylabel(r'$\beta_{x2,y1}$')
    ax.set_title(f'{tag} main solenoid')
    ax.legend(loc='best')
    ax.grid(True)
    _mark_solenoid_regions(ax, *markers_tag)
axs3[-1].set_xlabel('s [m]')
axs3[-1].set_xlim(-20, 20)
fig3.subplots_adjust(hspace=0.3, top=0.92, bottom=0.1, left=0.14)

#############################################################
# Same betx2/bety1 comparison, zoomed out to the coupling-  #
# correction target at the straight-section boundary        #
#############################################################

# The on_sol_coupling_corr_{ip} knob (004c) only forces betx2=bety1=0 exactly
# at these two markers, not anywhere nearer the IP -- see
# claude_notes/01_lattice_construction_000_004d.md. table_before_cuts shares
# tw's zeroed-at-IP_PLOT frame via the same shift used in
# _compute_marker_positions above.
_straight_section_s_ip_ref = table_before_cuts['s', IP_PLOT]
STRAIGHT_SECTION_S_RANGE = (
    table_before_cuts['s', f'end_ds_start_straight_{IP_PLOT}']
    - _straight_section_s_ip_ref,
    table_before_cuts['s', f'end_straight_start_ds_{IP_PLOT}']
    - _straight_section_s_ip_ref,
)

fig4, axs4 = plt.subplots(
    len(COMPARISON_TAGS), 1, sharex=True, figsize=(6.4, 4.8),
)
for ax, tag, (tw_tag, _markers_tag) in zip(axs4, COMPARISON_TAGS, _TAG_RESULTS):
    ax.plot(tw_tag.s, tw_tag.betx2, label=r'$\beta_{x2}$')
    ax.plot(tw_tag.s, tw_tag.bety1, label=r'$\beta_{y1}$')
    ax.set_ylabel(r'$\beta_{x2,y1}$')
    ax.set_title(f'{tag} main solenoid (full straight section)')
    ax.legend(loc='best')
    ax.grid(True)
    for s_pos in STRAIGHT_SECTION_S_RANGE:
        ax.axvline(s_pos, color='black', linewidth=0.8, linestyle=':')
axs4[-1].set_xlabel('s [m]')
axs4[-1].set_xlim(-1400, 1400)
fig4.subplots_adjust(hspace=0.3, top=0.92, bottom=0.1, left=0.14)

##############################################################
# beta_x/beta_y with and without solenoids, and the beta-beat #
##############################################################

# Extra elements highlighted with a green shaded span on the beta-comparison
# figures below: two back-to-back pairs from the sdm1 family either side of
# ipa, each sextupole 0.3 m long.
#   sdm1l.6/.7  at s - s_ipa ~= -210.7 m
#   sdm1r.0/.1  at s - s_ipa ~= +105.3 m -- these sit right at the exit of
#               qd4r.0, where the vertical beta-beat peaks (see the max-beat
#               print at the end of this script).
# Both pairs are inside ipa's straight section but well outside the +-20 m IR
# window the other figures use, which is why this figure is drawn at both
# ranges (see BETA_COMPARISON_RANGES).
SEXTUPOLES_TO_MARK = ['sdm1l.6', 'sdm1l.7', 'sdm1r.0', 'sdm1r.1']
SEXTUPOLE_LEGEND_LABEL = 'sdm1l.6/7, sdm1r.0/1'


def _element_s_spans(table, s_ip_ref, names):
    """[(s_start, s_end), ...] for the named elements, shifted into the frame
    zeroed at s_ip_ref -- same frame convention as _compute_marker_positions."""
    return [
        (float(table['s_start', nn]) - s_ip_ref,
         float(table['s_end', nn]) - s_ip_ref)
        for nn in names
    ]


SEXTUPOLE_S_SPANS = _element_s_spans(
    table_before_cuts, table_before_cuts['s', IP_PLOT], SEXTUPOLES_TO_MARK)


def _autoscale_y_to_xlim(ax, xlim, margin=0.1):
    """Rescale ax's ylim to the data actually visible within xlim.

    Must be called BEFORE the axvspan/axvline decoration: an axvline is a
    Line2D with y-data [0, 1], which would otherwise be folded into the
    min/max and flatten the panel (same helper, and the same gotcha, as
    004f/004j -- see claude_notes/07_main_b_scale_scans.md).
    """
    log_scale = ax.get_yscale() == 'log'
    y_min, y_max = np.inf, -np.inf
    for line_obj in ax.get_lines():
        xd = np.asarray(line_obj.get_xdata(), dtype=float)
        yd = np.asarray(line_obj.get_ydata(), dtype=float)
        if xd.size != yd.size:
            continue
        mask = (xd >= xlim[0]) & (xd <= xlim[1]) & np.isfinite(yd)
        if log_scale:
            mask &= yd > 0
        if mask.any():
            y_min = min(y_min, float(np.min(yd[mask])))
            y_max = max(y_max, float(np.max(yd[mask])))
    if not (np.isfinite(y_min) and np.isfinite(y_max)):
        return
    if log_scale:
        # Generous factors rather than a fractional margin -- this is a
        # decade-spanning axis, and the extra room keeps the legend clear.
        ax.set_ylim(y_min / 5.0, y_max * 5.0)
    else:
        span = y_max - y_min
        pad = margin * span if span > 0 else max(abs(y_max), 1.0) * margin
        ax.set_ylim(y_min - pad, y_max + pad)


def _mark_sextupoles(ax, spans):
    """Shade the marked sextupoles green. Each is only 0.3 m long, i.e. well
    under one pixel on the full-straight-section x-range, so a thin centre
    line is drawn too -- without it the span is invisible when zoomed out."""
    for s_start, s_end in spans:
        ax.axvspan(s_start, s_end, color='green', alpha=0.25, linewidth=0)
        ax.axvline(0.5 * (s_start + s_end), color='green', linewidth=0.8,
                   alpha=0.6)


# tw_off and tw are twissed on the same already-cut line (only knob *values*
# change between them), so they share an element grid and tw_off.betx can be
# subtracted from tw.betx directly. Guard anyway, and interpolate onto tw.s
# if that ever stops holding.
if np.array_equal(np.asarray(tw.name), np.asarray(tw_off.name)):
    BETX_OFF, BETY_OFF = tw_off.betx, tw_off.bety
else:
    print('NOTE: solenoid-on/off twiss grids differ; interpolating for the '
          'beta-beat.')
    BETX_OFF = np.interp(tw.s, tw_off.s, tw_off.betx)
    BETY_OFF = np.interp(tw.s, tw_off.s, tw_off.bety)

# Relative beta-beat (the standard definition). For the raw difference in
# metres instead, drop the division and the 100 and relabel the y-axis --
# note a raw difference is dominated by wherever beta is largest, which over
# a full straight section spans several orders of magnitude.
BEAT_X = (tw.betx - BETX_OFF) / BETX_OFF * 100.0
BEAT_Y = (tw.bety - BETY_OFF) / BETY_OFF * 100.0

# (xlim, title suffix, mark the IR solenoid/quad annotations?)
BETA_COMPARISON_RANGES = [
    ((-20, 20), 'IR', True),
    ((-1400, 1400), 'full straight section', False),
]


def _beta_comparison_fig(xlim, title_suffix, mark_ir_regions):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.0, 5.6))

    # beta is the Edwards-Teng mode-1/mode-2 beta once the solenoids couple
    # the planes, so "beta_x with solenoids" is really beta_{x1}.
    axs[0].plot(tw.s, tw.betx, color='C0', label=r'$\beta_x$ (solenoids on)')
    axs[0].plot(tw.s, tw.bety, color='C1', label=r'$\beta_y$ (solenoids on)')
    axs[0].plot(tw.s, BETX_OFF, color='C0', linestyle='--',
                label=r'$\beta_x$ (solenoids off)')
    axs[0].plot(tw.s, BETY_OFF, color='C1', linestyle='--',
                label=r'$\beta_y$ (solenoids off)')
    axs[0].set_ylabel(r'$\beta_{x,y}$ [m]')
    # beta spans ~1e-3 m at the IP to ~1e3 m in the straight -- log or the
    # vertical beta is a flat line on the floor of the plot.
    axs[0].set_yscale('log')
    axs[0].set_title(
        f'{IP_PLOT}: beta functions with/without solenoids '
        f'({_args.b0:g} T) -- {title_suffix}')

    axs[1].axhline(0.0, color='0.6', linewidth=0.8)
    axs[1].plot(tw.s, BEAT_X, color='C0', label=r'$\Delta\beta_x/\beta_x$')
    axs[1].plot(tw.s, BEAT_Y, color='C1', label=r'$\Delta\beta_y/\beta_y$')
    axs[1].set_ylabel(r'$\Delta\beta/\beta$ [%]')
    axs[1].set_xlabel('s [m]')

    # xlim and the y-autoscale both have to happen before the decoration --
    # see _autoscale_y_to_xlim. Without the autoscale the IR panel inherits
    # the whole ring's beat range (the ~100 % beta_y spike ~100 m from the
    # IP) and the IR structure is squashed to a flat line.
    axs[1].set_xlim(*xlim)
    for ax in axs:
        _autoscale_y_to_xlim(ax, xlim)

    for ax in axs:
        ax.grid(True)
        if mark_ir_regions:
            _mark_solenoid_regions(
                ax, MAIN_SOLENOID_S_RANGE, COMP_SOLENOID_S_RANGES,
                CORRECTOR_QUAD_S_POSITIONS, DOUBLET_QUAD_S_POSITIONS)
        else:
            for s_pos in STRAIGHT_SECTION_S_RANGE:
                ax.axvline(s_pos, color='black', linewidth=0.8, linestyle=':')
        _mark_sextupoles(ax, SEXTUPOLE_S_SPANS)

    sextupole_handle = Patch(
        facecolor='green', alpha=0.25, label=SEXTUPOLE_LEGEND_LABEL)
    axs[0].legend(
        handles=axs[0].get_legend_handles_labels()[0] + [sextupole_handle],
        loc='lower left', fontsize=8, ncol=2, framealpha=0.9)
    axs[1].legend(loc='upper right', fontsize=8, framealpha=0.9)

    fig.subplots_adjust(hspace=0.12, top=0.93, bottom=0.1, left=0.12)
    return fig


BETA_COMPARISON_FIGS = [
    _beta_comparison_fig(*spec) for spec in BETA_COMPARISON_RANGES
]

print(f'Loaded {INPUT_LATTICE_JSON}')
print(f'tw4d qx = {tw4d.qx:.12g}, qy = {tw4d.qy:.12g}')
print(f'tw6d qx = {tw.qx:.12g}, qy = {tw.qy:.12g}, qs = {tw.qs:.12g}')

_in_straight = (
    (tw.s >= STRAIGHT_SECTION_S_RANGE[0])
    & (tw.s <= STRAIGHT_SECTION_S_RANGE[1])
)
print(f'max |dbetx/betx| = {np.nanmax(np.abs(BEAT_X[_in_straight])):.4g} %, '
      f'max |dbety/bety| = {np.nanmax(np.abs(BEAT_Y[_in_straight])):.4g} % '
      f'(over {IP_PLOT}\'s straight section)')

plt.show()
