from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

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
    return main_range, comp_ranges, corrector_positions


MAIN_SOLENOID_S_RANGE, COMP_SOLENOID_S_RANGES, CORRECTOR_QUAD_S_POSITIONS = (
    _compute_marker_positions(table_before_cuts, IP_PLOT, _args.b0)
)


def _mark_solenoid_regions(ax, main_range, comp_ranges, corrector_positions):
    """Mark main solenoid / compensation solenoid extents and corrector-quad
    locations on an s-axis plot (vertical lines only, no legend entries)."""
    s_start, s_end = main_range
    ax.axvline(s_start, color='red', linewidth=0.8, linestyle='--')
    ax.axvline(s_end, color='red', linewidth=0.8, linestyle='--')
    for s_start, s_end in comp_ranges:
        ax.axvline(s_start, color='orange', linewidth=0.8, linestyle='--')
        ax.axvline(s_end, color='orange', linewidth=0.8, linestyle='--')
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
        CORRECTOR_QUAD_S_POSITIONS)

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
        CORRECTOR_QUAD_S_POSITIONS)

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
            CORRECTOR_QUAD_S_POSITIONS)

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

print(f'Loaded {INPUT_LATTICE_JSON}')
print(f'tw4d qx = {tw4d.qx:.12g}, qy = {tw4d.qy:.12g}')
print(f'tw6d qx = {tw.qx:.12g}, qy = {tw.qy:.12g}, qs = {tw.qs:.12g}')

plt.show()
