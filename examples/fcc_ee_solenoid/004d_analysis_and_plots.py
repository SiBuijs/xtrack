from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from aperture_study_io import PLOT_DIR
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

# Every figure is drawn at FIGURE_SCALE times its old linear size, so 2.0 is
# four times the area. This is NOT a literal magnification -- it buys room, and
# what it spends that room on is deliberate:
#
# - Scales: the rcParams below, covering ticks, axis labels and titles
#   (matplotlib's defaults for those are relative to font.size), plus the
#   explicit fontsize= calls that go through _fs. These keep the proportions
#   the small figures had, so the plot itself stays as legible as it was.
# - Does NOT scale: the legends and the |C^-| frame, left at absolute point
#   sizes (8 and 7). They are the reason for the extra room, so they take a
#   smaller share of it and hand the difference to the curves.
# - Does NOT scale: line widths, left at their absolute point values. On the
#   bigger canvas they come out relatively finer, which reads as crisper.
#
# For a literal magnification instead, wrap the legend/frame font sizes in _fs
# and scale 'lines.linewidth' here.
FIGURE_SCALE = 2.0

plt.rcParams['font.size'] = (
    plt.rcParamsDefault['font.size'] * FIGURE_SCALE)


def _fs(points):
    """An explicit point size, scaled with the figure."""
    return points * FIGURE_SCALE


def _figsize(width, height):
    """A figsize in the old units, scaled with the figure."""
    return (width * FIGURE_SCALE, height * FIGURE_SCALE)

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


# The three families of correction magnets installed in the straight by 004c.
# on_sol_corr_{ip} is their parent knob, but it drives two other things as
# well -- on_comp_sol_{ip} and on_rot_doublet_{side}_{ip} (004c:945-952) -- so
# zeroing the parent would switch the compensation solenoids off too. These
# three are the corrector magnets proper.
CORRECTION_KINDS = ('orbit', 'optics', 'coupling')


def _twiss_without_correctors(line_to_use):
    """Twiss with the straight's corrector magnets off, everything else on.

    Main and compensation solenoids stay powered and the final-focus doublet
    stays rotated; only the orbit, optics and coupling correction knobs go to
    zero. That isolates what the correctors in the straight are doing, rather
    than what the whole compensation scheme does.

    The knobs are restored to their expressions, not to 1, and that is the
    whole subtlety here. 004c wires each child as the *expression*
    `on_sol_{kind}_corr_{ip} = 'on_sol_corr_{ip}'`; assigning a number
    overwrites that expression permanently, so a later `on_sol_corr_{ip} = 1`
    would silently fail to switch these back on and every twiss after this one
    would be quietly wrong. Re-assigning the string re-attaches the child to
    its parent (the same trick 004cc's set_correction_state uses).
    """
    for ip_name in IP_NAMES:
        for kind in CORRECTION_KINDS:
            line_to_use[f'on_sol_{kind}_corr_{ip_name}'] = 0
    tw_result = line_to_use.twiss4d(strengths=True)
    for ip_name in IP_NAMES:
        for kind in CORRECTION_KINDS:
            line_to_use[f'on_sol_{kind}_corr_{ip_name}'] = (
                f'on_sol_corr_{ip_name}')
    tw_result.zero_at(IP_PLOT)
    return tw_result


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

# coupling_edw_teng adds the Edwards-Teng decoupling of the one-turn map:
# r11..r22_edw_teng, the decoupled-mode Twiss parameters, and the coupling
# RDTs f1001/f1010 (plotted below). It is only valid on a periodic twiss.
# Cost is a Python loop re-deriving the decoupling at every element, measured
# at ~9 us/element, so about a second for this ring -- cheap enough to ask for
# here, still not something to put inside a match loop.
tw4d = line.twiss4d(
    strengths=True,
    polarization_analysis=True,
    radiation_integrals=True,
    coupling_edw_teng=True,
)
tw = line.twiss6d(strengths=True, polarization_analysis=True)
two = line.twiss(betx=tw_off.betx[0], bety=tw_off.bety[0])


##############################################
# Twiss with the straight's correctors off   #
##############################################

tw_nocorr = _twiss_without_correctors(line)


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
# tw4d is plotted alongside tw in the coupling-RDT figures below, so it has to
# share the same zeroed-at-IP_PLOT s frame.
tw4d.zero_at(IP_PLOT)


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


def _element_s_spans(table, s_ip_ref, names):
    """[(s_start, s_end), ...] for the named elements, shifted into the frame
    zeroed at s_ip_ref -- same frame convention as _compute_ir_markers."""
    return [
        (float(table['s_start', nn]) - s_ip_ref,
         float(table['s_end', nn]) - s_ip_ref)
        for nn in names
    ]


def _peak_bs(tw, s_start, s_end):
    """Signed B_s [T] of largest magnitude on tw's grid inside [s_start,
    s_end]. The solenoid field is read back from the twiss rather than taken
    from solenoid_params, because the compensation field is not a parameter
    this script has: 004a scales it at build time so that the net integrated
    field (main + 2x compensation) cancels for whatever MAIN_SOLENOID_B0 is
    in use."""
    s = np.asarray(tw.s)
    bs = np.asarray(tw.bs)
    mask = (s >= s_start) & (s <= s_end)
    if not mask.any():
        return 0.0
    bs_in = bs[mask]
    return float(bs_in[np.argmax(np.abs(bs_in))])


class IrMarkers:
    """The IR annotation for one lattice case: strength-scaled bars for the
    solenoids and the rotated final-focus doublet quads, plus the s-positions
    of the orbit correctors.

    Each bar is (s_start, s_end, strength, color); strengths are B_s [T] for
    the solenoids and k1 [m^-2] for the quads, each family normalised to its
    own strongest element at draw time (see _mark_solenoid_regions), so bar
    heights are comparable within a family but not between the two.
    """

    def __init__(self, solenoid_bars, quad_bars, corrector_s):
        self.solenoid_bars = solenoid_bars
        self.quad_bars = quad_bars
        self.corrector_s = corrector_s


def _compute_ir_markers(table, uncut_line, tw, ip_plot, b0):
    """Build the IrMarkers for one case.

    `table` is line.get_table() taken before any cuts, in a frame zeroed at
    ip_plot -- this matches the frame produced by tw.zero_at(ip_plot) for a
    twiss computed on the same line before it was cut, since cutting only adds
    markers in drift regions and does not move existing elements.

    `uncut_line` is where the quad strengths are read from, and must be the
    line *before* cut_at_s: the plotting cuts fall inside the doublet quads
    (the +-2.4..11 m windows overlap them), which replaces e.g. 'qd0ar.0' by a
    chain of 'qd0ar.0..N' slices, so the name lookup would fail on the cut
    line.
    """
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
    solenoid_bars = [
        (*main_range, _peak_bs(tw, *main_range), 'red'),
    ] + [
        (*comp_range, _peak_bs(tw, *comp_range), 'orange')
        for comp_range in comp_ranges
    ]

    quad_bars = [
        (s_start, s_end, float(uncut_line[name].k1), 'purple')
        for name, (s_start, s_end) in zip(
            DOUBLET_QUADS_BY_IP[ip_plot],
            _element_s_spans(table, s_ip_ref, DOUBLET_QUADS_BY_IP[ip_plot]))
    ]

    # The orbit correctors are not elements of their own: 004c adds them as
    # knl[0]/ksl[0] on the host quad itself, which (num_multipole_kicks=200,
    # yoshida4) spreads the kick over the whole quad body rather than putting
    # it at one end. A line at the quad centre therefore marks *which* quads
    # host a corrector, not where inside them the kick sits.
    corrector_s = [
        0.5 * (s_start + s_end)
        for s_start, s_end in _element_s_spans(
            table, s_ip_ref, CORRECTOR_QUADS_BY_IP[ip_plot])
    ]
    return IrMarkers(solenoid_bars, quad_bars, corrector_s)


IR_MARKERS = _compute_ir_markers(
    table_before_cuts, env.fccee_p_ring, tw, IP_PLOT, _args.b0)


# Marker bars are drawn in axes-fraction coordinates, so they keep the same
# visual size on every panel whatever its data range and whether it is log or
# linear. Baseline near the bottom of the panel, signed about it the way the
# default xtrack twiss plot signs its lattice strip: focusing up, defocusing
# down.
MARKER_BAR_BASELINE = 0.16          # axes fraction, the zero-strength line
MARKER_BAR_MAX_HALF_HEIGHT = 0.14   # axes fraction, for the strongest element
# qf1c/qf1d are rotated with the rest of the doublet but run at k1 ~ 1e-5,
# i.e. off; without a floor their bars would be invisible and the doublet would
# look like it is missing elements. Keep them as a hairline instead.
MARKER_BAR_MIN_HALF_HEIGHT = 0.006  # axes fraction


def _strength_bar(ax, s_start, s_end, strength, strength_max, color):
    """One bar, signed about MARKER_BAR_BASELINE and scaled so that
    strength_max reaches MARKER_BAR_MAX_HALF_HEIGHT."""
    if strength_max <= 0.0:
        return
    half = MARKER_BAR_MAX_HALF_HEIGHT * strength / strength_max
    if abs(half) < MARKER_BAR_MIN_HALF_HEIGHT:
        half = MARKER_BAR_MIN_HALF_HEIGHT * (-1.0 if strength < 0.0 else 1.0)
    ax.axvspan(
        s_start, s_end,
        ymin=min(MARKER_BAR_BASELINE, MARKER_BAR_BASELINE + half),
        ymax=max(MARKER_BAR_BASELINE, MARKER_BAR_BASELINE + half),
        color=color, alpha=0.35, linewidth=0)


def _mark_solenoid_regions(ax, markers):
    """Draw `markers` on an s-axis plot: main solenoid (red) / compensation
    solenoids (orange) / rotated final-focus doublet quads (purple) as
    rectangles whose height scales with the element's strength, and the
    corrector-hosting quads as dashed grey full-height lines."""
    solenoid_max = max(abs(bar[2]) for bar in markers.solenoid_bars)
    quad_max = max(abs(bar[2]) for bar in markers.quad_bars)
    for bars, strength_max in ((markers.solenoid_bars, solenoid_max),
                               (markers.quad_bars, quad_max)):
        for s_start, s_end, strength, color in bars:
            _strength_bar(ax, s_start, s_end, strength, strength_max, color)
    for s_pos in markers.corrector_s:
        ax.axvline(s_pos, color='grey', linewidth=0.8, linestyle='--')


#########
# Plots #
#########

plt.close('all')

fig1 = plt.figure(figsize=_figsize(6.4, 4.8 * 1.8))
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
# Normalised by the main beta of the same plane, so the panel reads as "what
# fraction of the motion in this plane belongs to the other mode" rather than
# as a length. Note betx IS betx1 and bety IS bety2 (twiss.py:1414-1415), so
# these are betx2/betx1 and bety1/bety2 -- the two are exact mirrors of each
# other, which an unnormalised pair is not: betx2 and bety1 are metres of very
# different size simply because betx and bety are.
ax5.plot(tw.s, tw.betx2 / tw.betx, label=r'$\beta_{x2}/\beta_x$')
ax5.plot(tw.s, tw.bety1 / tw.bety, label=r'$\beta_{y1}/\beta_y$')
ax5.set_ylabel(r'$\beta_{x2}/\beta_x$,  $\beta_{y1}/\beta_y$')
ax5.legend(loc='best')
ax5.grid(True)

for _ax in (ax1, ax2, ax3, ax4, ax5):
    _mark_solenoid_regions(_ax, IR_MARKERS)

ax1.set_xlabel('')
ax5.set_xlabel('s [m]')
fig1.subplots_adjust(hspace=0.25, top=0.95, bottom=0.06, left=0.14)
ax5.set_xlim(-20, 20)

fig2 = plt.figure(figsize=_figsize(6.4, 4.8 * 1.8))
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
    _mark_solenoid_regions(_ax, IR_MARKERS)

ax1.set_xlabel('')
ax5.set_xlabel('s [m]')
fig2.subplots_adjust(hspace=0.25, top=0.95, bottom=0.06, left=0.14)
ax5.set_xlim(-20, 20)

###########################################
# 2T vs 3T betx2/bety1 comparison figure #
###########################################

COMPARISON_TAGS = ['2T', '3T']
COMPARISON_B0_BY_TAG = {'2T': 2.0, '3T': 3.0}


def _twiss_for_tag(tag):
    """(twiss on, twiss off, markers, twiss-on-4d-with-Edwards-Teng) for
    a given field tag. The main-solenoid half-length differs between the
    2T and 3T cases (see solenoid_params.half_length_for_b0), so markers are
    recomputed per tag rather than reusing the primary case's.

    The off case is twissed on the same lattice with the knobs set to 0, so it
    lands on the same s grid as the on case and the two can be overlaid
    directly. It is the uncoupled baseline: with no solenoid and no skew
    correction there is nothing in the ring to couple the planes, so betx2 and
    bety1 should sit at numerical zero throughout.

    The fourth entry carries the coupling RDTs for the summary box: the two
    twisses above are 6d for the primary tag, and coupling_edw_teng belongs on a
    4d twiss (see the block comment above RDT_CASES), so it is a separate object
    rather than a flag on those. For the primary tag it is the tw4d computed at
    the top of the script; for the others it is the on-case twiss, which is 4d
    already and only needs the flag."""
    if tag == FIELD_TAG:
        return tw, tw_off, IR_MARKERS, tw4d, tw_nocorr

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
        line_cmp[f'on_sol_{ip_name}'] = 0
        line_cmp[f'on_sol_corr_{ip_name}'] = 0

    tw_cmp_off = line_cmp.twiss4d(strengths=True)
    tw_cmp_off.zero_at(IP_PLOT)

    for ip_name in IP_NAMES:
        line_cmp[f'on_sol_{ip_name}'] = 1
        line_cmp[f'on_sol_corr_{ip_name}'] = 1

    tw_cmp = line_cmp.twiss4d(strengths=True, coupling_edw_teng=True)
    tw_cmp.zero_at(IP_PLOT)
    markers_cmp = _compute_ir_markers(
        table_cmp, env_cmp.fccee_p_ring, tw_cmp, IP_PLOT,
        COMPARISON_B0_BY_TAG[tag])
    # Correctors off, solenoids and doublet rotations still on. Taken last so
    # the knob round-trip inside the helper cannot affect the twisses above.
    tw_cmp_nocorr = _twiss_without_correctors(line_cmp)
    return tw_cmp, tw_cmp_off, markers_cmp, tw_cmp, tw_cmp_nocorr


# Computed once per tag and reused for both the local-region figure (fig3)
# and the full-straight-section figure (fig4) below, since each call rebuilds
# and re-twisses a comparison lattice and is not cheap to repeat.
_TAG_RESULTS = [_twiss_for_tag(tag) for tag in COMPARISON_TAGS]

# The on_sol_coupling_corr_{ip} knob (004c) only forces betx2=bety1=0 exactly
# at these two markers, not anywhere nearer the IP -- see
# claude_notes/01_lattice_construction_000_004d.md. table_before_cuts shares
# tw's zeroed-at-IP_PLOT frame via the same shift used in
# _compute_ir_markers above.
_straight_section_s_ip_ref = table_before_cuts['s', IP_PLOT]
STRAIGHT_SECTION_S_RANGE = (
    table_before_cuts['s', f'end_ds_start_straight_{IP_PLOT}']
    - _straight_section_s_ip_ref,
    table_before_cuts['s', f'end_straight_start_ds_{IP_PLOT}']
    - _straight_section_s_ip_ref,
)


def _box_under_legend(fig, entries):
    """Put a small framed text box directly under each panel's legend, sharing
    its right edge.

    The legend's height depends on font metrics, on its ncol, and on the
    backend, so rather than guess an offset this draws once and reads each
    legend's actual extent. The axes are placed by subplots_adjust and not by
    tight/constrained layout, so nothing moves between this draw and the later
    show/savefig and the placement holds. Call it after subplots_adjust.

    `entries` is an iterable of (ax, legend, text).
    """
    fig.canvas.draw()
    for ax, leg, text in entries:
        leg_box = leg.get_window_extent().transformed(ax.transAxes.inverted())
        # ha right pins the frame to the legend's right edge; multialignment
        # left keeps the values lined up as a column inside it.
        ax.text(leg_box.x1, leg_box.y0 - 0.05, text, transform=ax.transAxes,
                ha='right', va='top', fontsize=7, multialignment='left',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='0.6', linewidth=0.8, alpha=0.9))


def _rdt_ring_average(tw_et, rdt):
    r"""Length-weighted ring average of a coupling RDT: sum(|f| dl) / C.

    Weighted by dl = diff(s), not by the table's `length` column, which also
    drops thin elements from the sum -- they get dl = 0. That is deliberate for
    the eight corr_sol_* correctors: they are thin Multipoles carrying
    length = 1.0 m while occupying zero s, so summing `length` would both
    overshoot the circumference by exactly 8 m and hand each of them a metre of
    weight at a point where it is not clear which f applies anyway. diff(s) sums
    to line_length to 3e-8.

    The weighting is what makes this number mean anything. _twiss_for_tag cuts
    the IR into 0.2 m slices and leaves the arcs alone, so an unweighted mean
    over table rows oversamples the few hundred metres around the IPs by an
    order of magnitude -- it comes out about 10x the length-weighted value
    (0.078 against 0.0072 for |f1001| on the 3T ring).

    |f| rather than f: the complex sum partly cancels by phase and answers a
    different question (what the ring drives in total, 3.1e-3 here against
    7.2e-3). This is the mean strength of the driving term, and it is the |f|
    that the panels plot.

    The solenoid-body NaNs (see the block comment above RDT_CASES) are dropped
    from the numerator, the denominator staying the full circumference. 12 m of
    90.6 km, so a 1.3e-4 effect; the masking is here only so the sum is a number
    rather than NaN.
    """
    f = np.abs(np.asarray(rdt))
    # append=nan: the last row is _end_point and opens no interval.
    dl = np.diff(np.asarray(tw_et.s), append=np.nan)
    # dl > 0 is what excludes the thin elements. Weight 0 would contribute 0
    # anyway, so this changes no number; it is here to make the exclusion
    # explicit rather than an accident of the weighting.
    ok = np.isfinite(f) & np.isfinite(dl) & (dl > 0)
    return float(np.sum(f[ok] * dl[ok]) / tw_et.line_length)


def _coupling_summary_text(tw_on, tw_off):
    r"""The closest-tune-approach scalar for one panel of the betx2/bety1
    figure. It is the global partner of the curves beside it: those show where
    the coupling sits, this says what the whole ring ends up with. `c_minus` is
    itself an s-average (trapz over the local coefficient / line_length).

    The ring-averaged RDTs used to be quoted here too; they are still printed
    at the end of the script, via _rdt_ring_average.

    Caveat, see the solenoid W-matrix note: `c_minus` is built from W_matrix
    entries, which inside the solenoid bodies are in kinetic momenta. Those
    slices are ~12 m of a 91 km ring so their weight here is negligible, but
    this number says nothing about the IR specifically.
    """
    return (fr'$|C^-|$ = {tw_on.c_minus:.2e}   '
            fr'[solenoids off: {tw_off.c_minus:.1e}]')


# Shared top of the betx2/betx and bety1/bety axes, on every panel of both
# figures. Fixed rather than autoscaled so the 2 T and 3 T panels can be read
# against each other; 0.03 clears the largest ratio anywhere in the ring
# (2.24e-2, inside the 3 T solenoid bodies) with room for the legend. If a
# future case overshoots it the figure widens the axis and prints a note
# instead of clipping.
BETX2_BETY1_YLIM_TOP = 0.03


def _betx2_bety1_fig(xlim, title_suffix, mark_ir_regions, tags=None):
    """One panel per field tag, each overlaying the coupled-mode betas with
    the solenoids on and off. Drawn twice, at the IR window and at the full
    straight section (the two calls below), which is why this is a function
    rather than the two near-identical loops it used to be.

    `tags` selects which field cases get a panel; passing a single one gives
    the one-panel version saved per case under plots/."""
    tags = list(tags or COMPARISON_TAGS)
    results = [_TAG_RESULTS[COMPARISON_TAGS.index(t)] for t in tags]
    # squeeze=False so a single-tag figure still yields an array to iterate.
    fig, axs = plt.subplots(
        len(tags), 1, sharex=True, squeeze=False,
        figsize=_figsize(6.4, 2.4 * len(tags)),
    )
    axs = axs[:, 0]
    c_minus_boxes = []
    visible_max = {}
    for ax, tag, (tw_tag, tw_off_tag, markers_tag, tw_et_tag,
                  _tw_nocorr_tag) in zip(axs, tags, results):
        # Each coupling beta over the main beta of its own plane (betx is
        # betx1 and bety is bety2, twiss.py:1414-1415, so these are
        # betx2/betx1 and bety1/bety2). Dimensionless, and the two planes
        # become directly comparable: as metres, betx2 and bety1 differ mostly
        # because betx and bety do, which says nothing about the coupling.
        ax.plot(tw_tag.s, tw_tag.betx2 / tw_tag.betx, color='C0',
                label=r'$\beta_{x2}/\beta_x$ (solenoids on)')
        ax.plot(tw_tag.s, tw_tag.bety1 / tw_tag.bety, color='C1',
                label=r'$\beta_{y1}/\beta_y$ (solenoids on)')
        # The uncoupled baseline -- see _twiss_for_tag. Expect these to lie on
        # zero: they are here to show what the solenoid does against a ring
        # that has no other coupling source, not because they carry structure.
        ax.plot(tw_off_tag.s, tw_off_tag.betx2 / tw_off_tag.betx, color='C0',
                linestyle='--', linewidth=1.0,
                label=r'$\beta_{x2}/\beta_x$ (solenoids off)')
        ax.plot(tw_off_tag.s, tw_off_tag.bety1 / tw_off_tag.bety, color='C1',
                linestyle='--', linewidth=1.0,
                label=r'$\beta_{y1}/\beta_y$ (solenoids off)')
        ax.set_ylabel(r'$\beta_{x2}/\beta_x$,  $\beta_{y1}/\beta_y$')
        ax.set_title(f'{tag} main solenoid{title_suffix}')
        # Largest ratio actually inside this figure's s-window. Taken from the
        # arrays rather than from the drawn lines, because _mark_solenoid_
        # regions adds axvlines whose y-data is [0, 1] and would swamp any
        # measurement made off the axes (the same trap _autoscale_y_to_xlim
        # documents). Matplotlib's own autoscale is not window-aware at all --
        # it fits the whole ring, so without this the IR panel's scale is set
        # by whatever the arcs do.
        in_window = ((np.asarray(tw_tag.s) >= xlim[0])
                     & (np.asarray(tw_tag.s) <= xlim[1]))
        visible_max[ax] = max(
            np.nanmax(np.asarray(tw_tag.betx2 / tw_tag.betx)[in_window]),
            np.nanmax(np.asarray(tw_tag.bety1 / tw_tag.bety)[in_window]))
        leg = ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
        c_minus_boxes.append(
            (ax, leg, _coupling_summary_text(tw_tag, tw_off_tag)))
        ax.grid(True)
        # Every panel carries its own s ticks and label, so that any one of
        # them can be cropped out and used on its own (same reasoning as in
        # _beta_comparison_fig); sharex would otherwise blank all but the last.
        ax.tick_params(labelbottom=True)
        ax.set_xlabel('s [m]')
        if mark_ir_regions:
            _mark_solenoid_regions(ax, markers_tag)
        else:
            for s_pos in STRAIGHT_SECTION_S_RANGE:
                ax.axvline(s_pos, color='black', linewidth=0.8, linestyle=':')
    axs[-1].set_xlim(*xlim)
    # One fixed top shared by every panel, so the 2 T and 3 T curves can be
    # compared by eye instead of against two different scales. It also leaves
    # room for the upper-right legend and the ring-average box hanging under
    # it, which otherwise land on the betx2/betx peaks at s ~ +-2.5 m.
    #
    # Ratios are non-negative, so the floor sits just under zero (enough to
    # keep the near-zero bety1/bety trace off the spine).
    top = BETX2_BETY1_YLIM_TOP
    exceeded = {tag: visible_max[ax]
                for ax, tag in zip(axs, tags)
                if visible_max[ax] > top}
    if exceeded:
        # Never clip silently: a fixed axis that hides the peak would misreport
        # the coupling. Fall back to fitting the data and say so.
        top = max(exceeded.values()) * 1.15
        print(f'NOTE: betx2/betx exceeds BETX2_BETY1_YLIM_TOP '
              f'({BETX2_BETY1_YLIM_TOP:g}) at ' + ', '.join(
                  f'{tag} ({value:.4g})' for tag, value in exceeded.items())
              + f'; y-axis widened to {top:.4g}.')
    for ax in axs:
        ax.set_ylim(-0.03 * top, top)
    fig.subplots_adjust(hspace=0.45, top=0.92, bottom=0.1, left=0.14)
    _box_under_legend(fig, c_minus_boxes)
    return fig


fig3 = _betx2_bety1_fig((-20, 20), '', True)

#############################################################
# Same betx2/bety1 comparison, zoomed out to the coupling-  #
# correction target at the straight-section boundary        #
#############################################################

fig4 = _betx2_bety1_fig((-1400, 1400), ' (full straight section)', False)

###############################################################
# Same two views with the straight's correctors switched off  #
#                                                             #
# Solenoids (main and compensation) stay on and the doublet   #
# stays rotated -- only on_sol_{orbit,optics,coupling}_corr   #
# go to zero, see _twiss_without_correctors. These are their  #
# own figures rather than extra curves on the ones above      #
# because the coupling runs ~30x larger here and would need   #
# the fixed 0.03 axis thrown away to fit.                     #
###############################################################


def _nocorr_fig(xlim, title_suffix, mark_ir_regions, quantity, tags=None):
    """One panel per field tag, correctors off.

    `quantity` is 'beta' for betx/bety in metres, or 'coupling' for the two
    normalised mode ratios. The two share everything but the curves and the
    y-scaling, and both are wanted at the same two s-windows, so they are one
    function with a switch rather than four near-identical bodies.

    `tags` selects which field cases get a panel; passing a single one gives
    the one-panel version saved per case under plots/.
    """
    tags = list(tags or COMPARISON_TAGS)
    results = [_TAG_RESULTS[COMPARISON_TAGS.index(t)] for t in tags]
    fig, axs = plt.subplots(
        len(tags), 1, sharex=True, squeeze=False,
        figsize=_figsize(6.4, 2.4 * len(tags)),
    )
    axs = axs[:, 0]
    c_minus_boxes = []
    for ax, tag, (_tw, _tw_off, markers_tag, _tw_et, tw_nc) in zip(
            axs, tags, results):
        ax_disp = None
        if quantity == 'beta':
            ax.plot(tw_nc.s, tw_nc.betx, color='C0', label=r'$\beta_x$')
            ax.plot(tw_nc.s, tw_nc.bety, color='C1', label=r'$\beta_y$')
            ax.set_ylabel(r'$\beta_{x,y}$ [m]')
            curves = (np.asarray(tw_nc.betx, dtype=float),
                      np.asarray(tw_nc.bety, dtype=float))
            # Dispersion on its own right-hand axis and in mm, as in
            # _beta_comparison_fig: it changes sign, and D_y is the
            # solenoid-driven one. C2/C3 continue the cycle after the betas.
            ax_disp = ax.twinx()
            ax_disp.plot(tw_nc.s, tw_nc.dx * 1e3, color='C2', linewidth=1.0,
                         label=r'$D_x$')
            ax_disp.plot(tw_nc.s, tw_nc.dy * 1e3, color='C3', linewidth=1.0,
                         label=r'$D_y$')
            ax_disp.set_ylabel(r'$D_{x,y}$ [mm]')
        else:
            ax.plot(tw_nc.s, tw_nc.betx2 / tw_nc.betx, color='C0',
                    label=r'$\beta_{x2}/\beta_x$')
            ax.plot(tw_nc.s, tw_nc.bety1 / tw_nc.bety, color='C1',
                    label=r'$\beta_{y1}/\beta_y$')
            ax.set_ylabel(r'$\beta_{x2}/\beta_x$,  $\beta_{y1}/\beta_y$')
            curves = (np.asarray(tw_nc.betx2 / tw_nc.betx, dtype=float),
                      np.asarray(tw_nc.bety1 / tw_nc.bety, dtype=float))

        ax.set_title(f'{tag} main solenoid, correctors off{title_suffix}')
        if ax_disp is None:
            leg = ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        else:
            # One legend for both axes, drawn on the twin: it is on top, so a
            # legend belonging to ax would be overdrawn by the D traces.
            leg = ax_disp.legend(
                handles=(ax.get_legend_handles_labels()[0]
                         + ax_disp.get_legend_handles_labels()[0]),
                loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
        if quantity == 'coupling':
            # Coupling panels only -- |C^-| says nothing about betx/bety.
            # c_minus needs no extra twiss flag, it is on any periodic twiss
            # with lattice functions. The corrected value goes underneath for
            # scale, with the symbol repeated rather than the line indented:
            # the prefix is mathtext, whose width has nothing to do with the
            # width of the spaces that would have to stand in for it, so an
            # indent leaves the two numbers visibly misaligned.
            c_minus_boxes.append((ax, leg, '\n'.join((
                fr'$|C^-|$ = {tw_nc.c_minus:.2e}  (correctors off)',
                fr'$|C^-|$ = {_tw.c_minus:.2e}  (corrected)',
            ))))
        ax.grid(True)
        ax.tick_params(labelbottom=True)
        ax.set_xlabel('s [m]')

        # Scale to the window, from the arrays rather than the drawn lines --
        # _mark_solenoid_regions' axvlines carry y-data [0, 1] and matplotlib's
        # own autoscale fits the whole ring, neither of which is wanted here.
        # Both quantities are non-negative, so the floor is pinned at zero.
        in_window = ((np.asarray(tw_nc.s) >= xlim[0])
                     & (np.asarray(tw_nc.s) <= xlim[1]))
        top = max(np.nanmax(curve[in_window]) for curve in curves) * 1.35
        ax.set_ylim(-0.03 * top, top)

        if ax_disp is not None:
            # Dispersion gets the same window treatment, but it changes sign,
            # so the floor cannot be pinned at zero -- pad both ends instead.
            disp = np.concatenate([
                np.asarray(tw_nc.dx, dtype=float)[in_window] * 1e3,
                np.asarray(tw_nc.dy, dtype=float)[in_window] * 1e3,
            ])
            d_lo, d_hi = np.nanmin(disp), np.nanmax(disp)
            d_pad = 0.1 * max(d_hi - d_lo, 1e-12)
            # Headroom on top only, to keep the two-column legend off the
            # traces; the legend sits upper right on both axes.
            ax_disp.set_ylim(d_lo - d_pad, d_hi + d_pad + 0.35 * (d_hi - d_lo))

        if mark_ir_regions:
            _mark_solenoid_regions(ax, markers_tag)
        else:
            for s_pos in STRAIGHT_SECTION_S_RANGE:
                ax.axvline(s_pos, color='black', linewidth=0.8, linestyle=':')
    axs[-1].set_xlim(*xlim)
    fig.subplots_adjust(hspace=0.45, top=0.92, bottom=0.1, left=0.14)
    if c_minus_boxes:
        _box_under_legend(fig, c_minus_boxes)
    return fig


fig4a = _nocorr_fig((-20, 20), '', True, 'beta')
fig4b = _nocorr_fig((-1400, 1400), ' (full straight)', False, 'beta')
fig4c = _nocorr_fig((-20, 20), '', True, 'coupling')
fig4d = _nocorr_fig((-1400, 1400), ' (full straight)', False, 'coupling')

##############################################################
# Coupling RDTs from the Edwards-Teng decoupling of the ring #
##############################################################

# betx2/bety1 above answer "is the ring decoupled *at this s*". |f1001| and
# |f1010| answer "how much resonance driving is left anywhere". Those are
# different questions: on_sol_coupling_corr_{ip} only pins betx2=bety1=0 at
# two markers (see the comment above STRAIGHT_SECTION_S_RANGE), so the local
# diagnostic can read clean while driving terms survive elsewhere in the ring.
#
# f1001 drives the difference resonance qx - qy (emittance sharing, the one
# that matters for the vertical emittance budget); f1010 drives the sum
# resonance qx + qy.
#
# Drawn from tw4d rather than tw: xtrack's Edwards-Teng block builds its
# one-turn rotation with only the qx/qy 2x2 blocks filled, i.e. the 4d map,
# and everything in xtrack that exercises coupling_edw_teng (the example in
# examples/coupling_edwards_teng and tests/test_coupling_edwards_teng.py) is
# 4d for that reason. Note also that xsuite's sign convention differs from
# MAD-X's by f1001 -> -conj(f1001); only |f1001| is plotted here, which is
# insensitive to that, but keep it in mind before comparing phases to MAD-X.
#
# IMPORTANT -- these curves have holes, and the holes are in the IR. Measured
# on the corrected 3T ring: 1208 of 32573 points come back NaN, in exactly
# eight contiguous blocks of 151, i.e. two per IP, centred at +-13.0 m and
# spanning the compensating-solenoid bodies.
#
# Traced to the momentum convention of tw4d.W_matrix, not to Edwards-Teng.
# _get_edwards_teng_initial takes sqrt(det(B + conj(C)) + (tr A - tr D)^2 / 4)
# (twiss.py, the `coeff` line). That argument is a similarity invariant of the
# one-turn map -- it equals (cos(2 pi qx) - cos(2 pi qy))^2, here 5.147179e-02
# -- so it cannot legitimately vary with s at all, let alone change sign. It
# holds that value to 7 digits everywhere outside the solenoids and falls to
# about -3.5e+03 inside them.
#
# What breaks is W itself: inside the solenoid body W is reported in kinetic
# momenta, and the kinetic<->canonical shift (px -= Bs/(2 Brho) y, py += ...x)
# is not symplectic. Measured residual max|W^T S W - S| over the transverse
# 4x4 is 2 * (Bs/(2 Brho)) * sqrt(betx bety) -- ratio 2.000, correlation
# 0.9997 over all 2400 in-field points -- against 3e-09 outside. So W Rot
# W^-1 is not a similarity transform there, the invariant stops being
# invariant, and sqrt() of the negative result is the NaN. Undoing the shift
# restores the discriminant to 5.1470e-02 (scratchpad et_fix.py).
#
# Consequences: (1) |f1001| is valid ring-wide and in the arcs but says
# nothing inside the solenoid bodies; (2) betx/bety/betx2/bety1 in those same
# slices come off the same kinetic-momentum W, so the IR panels of the figures
# above are in that convention too -- fine for judging whether the coupling
# correction zeroes them, but do not read them as canonical Twiss functions.

RDT_CASES = (
    (r'$|f_{1001}|$  (difference resonance, $q_x - q_y$)', tw4d.f1001,
     r'|f_{1001}|'),
    (r'$|f_{1010}|$  (sum resonance, $q_x + q_y$)', tw4d.f1010,
     r'|f_{1010}|'),
)


def _coupling_rdt_fig(xlim, title_suffix, mark_ir_regions):
    """The two coupling RDTs along s, drawn at the same two s-ranges as the
    betx2/bety1 figures above so the local and global coupling diagnostics can
    be read against each other panel by panel."""
    fig, axs = plt.subplots(len(RDT_CASES), 1, sharex=True, figsize=_figsize(6.4, 4.8))
    rdt_boxes = []
    for ax, (label, rdt, symbol) in zip(axs, RDT_CASES):
        # Labelled so these panels carry a top-right legend like the
        # betx2/bety1 ones, with the ring-average frame hanging under it. The
        # label adds what the ylabel does not say: which lattice this is.
        ax.plot(tw4d.s, np.abs(rdt), color='C0',
                label=fr'${symbol}$ ({FIELD_TAG}, solenoids on)')
        ax.set_ylabel(label, fontsize=_fs(9))
        ax.set_title(f'coupling RDT{title_suffix}', fontsize=_fs(10))
        ax.grid(True)
        leg = ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        # The ring average behind the curve: the same number in both s-windows
        # and in both panels of the betx2/bety1 figure, since a ring average
        # does not depend on what is currently plotted. That is the point of
        # showing it here -- it says how much of the curve the window shows.
        # Number outside the mathtext: inside it, "7.24e-03" typesets as an
        # italic e and a spaced minus.
        rdt_boxes.append((ax, leg,
                          fr'ring average $\langle{symbol}\rangle$ = '
                          fr'{_rdt_ring_average(tw4d, rdt):.2e}'))
        # Per-panel ticks and label, so a single panel can be cropped out and
        # used on its own -- same reasoning as _betx2_bety1_fig.
        ax.tick_params(labelbottom=True)
        ax.set_xlabel('s [m]')
        if mark_ir_regions:
            _mark_solenoid_regions(ax, IR_MARKERS)
        else:
            for s_pos in STRAIGHT_SECTION_S_RANGE:
                ax.axvline(s_pos, color='black', linewidth=0.8, linestyle=':')
    axs[-1].set_xlim(*xlim)
    fig.subplots_adjust(hspace=0.45, top=0.92, bottom=0.1, left=0.16)
    _box_under_legend(fig, rdt_boxes)
    return fig


fig5 = _coupling_rdt_fig((-20, 20), '', True)
fig6 = _coupling_rdt_fig((-1400, 1400), ' (full straight section)', False)

##############################################################
# beta_x/beta_y with and without solenoids, and the beta-beat #
##############################################################

# These figures are drawn at two s-ranges (see BETA_COMPARISON_RANGES): the
# +-20 m IR window the other figures use, and the full straight section. The
# wide one is what shows the sdm1 sextupole pairs either side of ipa --
# sdm1l.6/.7 at s - s_ipa ~= -210.7 m and sdm1r.0/.1 at ~= +105.3 m, the
# latter right at the exit of qd4r.0 where the vertical beta-beat peaks (see
# the max-beat print at the end of this script). They used to be shaded green
# here; the shading was dropped, so the region has to be read off the s-axis.


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


# tw_off and tw are twissed on the same already-cut line (only knob *values*
# change between them), so they share an element grid and tw_off.betx can be
# subtracted from tw.betx directly. Guard anyway, and interpolate onto tw.s
# if that ever stops holding.
if np.array_equal(np.asarray(tw.name), np.asarray(tw_off.name)):
    BETX_OFF, BETY_OFF = tw_off.betx, tw_off.bety
    DX_OFF, DY_OFF = tw_off.dx, tw_off.dy
else:
    print('NOTE: solenoid-on/off twiss grids differ; interpolating for the '
          'beta-beat.')
    BETX_OFF = np.interp(tw.s, tw_off.s, tw_off.betx)
    BETY_OFF = np.interp(tw.s, tw_off.s, tw_off.bety)
    DX_OFF = np.interp(tw.s, tw_off.s, tw_off.dx)
    DY_OFF = np.interp(tw.s, tw_off.s, tw_off.dy)

# Relative beta-beat (the standard definition). For the raw difference in
# metres instead, drop the division and the 100 and relabel the y-axis --
# note a raw difference is dominated by wherever beta is largest, which over
# a full straight section spans several orders of magnitude.
BEAT_X = (tw.betx - BETX_OFF) / BETX_OFF * 100.0
BEAT_Y = (tw.bety - BETY_OFF) / BETY_OFF * 100.0

# (xlim, title suffix, mark the IR solenoid/quad annotations?, beta y-scale,
#  legend location)
# Both ranges are linear, by request -- the log version of the IR window was
# dropped. What that costs is worth knowing: a log axis was the only one that
# resolved the IP waist (betx* ~ 0.09 m, bety* ~ 0.7 mm, i.e. 4-7 decades below
# the peaks), so on these panels the waist collapses onto zero. In exchange the
# bety double hump (5.2e3 -> 1.5e4, only half a decade, and flat on a log axis)
# and the ~5 % betx hump now read as the familiar twin peaks.
#
# The same trade applies to the full straight section, measured on the 3T ring:
# the bety doublet spike (~1.5e4 m, within a few m of the IP) sets the scale, so
# betx -- which peaks near 1.5e3 m in the arcs -- is compressed onto the bottom
# tenth of the panel. What survives is the arc bety modulation and the
# dispersion.
#
# yscale is kept as a per-figure setting rather than hard-coded, so a log panel
# can be added back with one entry. The legend location stays per-figure too:
# on a linear axis the free corner is top left for the wide range (only the
# doublet spike at s ~ 0 reaches the top, while the bottom right carries the arc
# bety and D_x traces), which is the opposite of where the log version wanted
# it.
BETA_COMPARISON_RANGES = [
    ((-20, 20), 'IR', True, 'linear', 'upper right'),
    ((-1400, 1400), 'full straight section', False, 'linear', 'upper left'),
]


def _beta_comparison_fig(xlim, title_suffix, mark_ir_regions, yscale,
                         legend_loc):
    # Betas and dispersion only. The Delta-beta/beta panel that used to sit
    # underneath is gone; the two numbers worth having from it are still
    # printed at the end of the script (max |dbetx/betx| over the straight),
    # and BEAT_X/BEAT_Y are still computed for that print.
    fig, ax_beta = plt.subplots(1, 1, figsize=_figsize(7.0, 3.6))

    # beta is the Edwards-Teng mode-1/mode-2 beta once the solenoids couple
    # the planes, so "beta_x with solenoids" is really beta_{x1}.
    ax_beta.plot(tw.s, tw.betx, color='C0', label=r'$\beta_x$ (solenoids on)')
    ax_beta.plot(tw.s, tw.bety, color='C1', label=r'$\beta_y$ (solenoids on)')
    ax_beta.plot(tw.s, BETX_OFF, color='C0', linestyle='--',
                 label=r'$\beta_x$ (solenoids off)')
    ax_beta.plot(tw.s, BETY_OFF, color='C1', linestyle='--',
                 label=r'$\beta_y$ (solenoids off)')
    ax_beta.set_ylabel(r'$\beta_{x,y}$ [m]')
    ax_beta.set_yscale(yscale)
    ax_beta.set_xlabel('s [m]')
    ax_beta.set_title(
        f'{IP_PLOT}: beta functions with/without solenoids '
        f'({_args.b0:g} T) -- {title_suffix}')

    # Dispersion on its own right-hand axis, always linear and in mm: it
    # changes sign (so a log axis is out) and D_y is the solenoid-driven one,
    # of order 0.1 mm -- see fig1's D_y panel. Both planes share the axis, so
    # if D_x runs far larger than D_y here, D_y reads as a flat line; that is
    # the honest picture of the two magnitudes, not a plotting failure.
    # C2/C3 continue the default colour cycle after the betas' C0/C1. C3 does
    # brush against the red main-solenoid bar, but that is a pale translucent
    # patch against a saturated line, so the two do not read alike.
    ax_disp = ax_beta.twinx()
    ax_disp.plot(tw.s, tw.dx * 1e3, color='C2', linewidth=1.0,
                 label=r'$D_x$ (solenoids on)')
    ax_disp.plot(tw.s, tw.dy * 1e3, color='C3', linewidth=1.0,
                 label=r'$D_y$ (solenoids on)')
    ax_disp.plot(tw.s, DX_OFF * 1e3, color='C2', linewidth=1.0, linestyle='--',
                 label=r'$D_x$ (solenoids off)')
    ax_disp.plot(tw.s, DY_OFF * 1e3, color='C3', linewidth=1.0, linestyle='--',
                 label=r'$D_y$ (solenoids off)')
    ax_disp.set_ylabel(r'$D_{x,y}$ [mm]')

    # xlim and the y-autoscale both have to happen before the decoration --
    # see _autoscale_y_to_xlim. Without the autoscale the IR panel inherits
    # the whole ring's range and its structure is squashed to a flat line.
    ax_beta.set_xlim(*xlim)
    for ax in (ax_beta, ax_disp):
        _autoscale_y_to_xlim(ax, xlim)
    if yscale == 'linear':
        # beta is positive-definite, so pin the floor at 0 -- the autoscale's
        # 10 % pad would otherwise hang the axis slightly below zero. The extra
        # headroom on top keeps the legend clear of the beta_y peaks, which on
        # a linear axis run all the way to the top of the panel -- so it is
        # only worth paying for when the legend is actually up there.
        headroom = 1.3 if legend_loc.startswith('upper') else 1.05
        ax_beta.set_ylim(bottom=0.0, top=ax_beta.get_ylim()[1] * headroom)

    ax_beta.grid(True)
    if mark_ir_regions:
        _mark_solenoid_regions(ax_beta, IR_MARKERS)
    else:
        for s_pos in STRAIGHT_SECTION_S_RANGE:
            ax_beta.axvline(s_pos, color='black', linewidth=0.8,
                            linestyle=':')

    # One legend for the whole panel: beta (left axis) and dispersion (right
    # axis). It has to be drawn on the twin axes, not on ax_beta -- the twin is
    # on top, so a legend belonging to ax_beta would be overdrawn by the
    # dispersion traces. Two columns keep eight entries down to four rows.
    ax_disp.legend(
        handles=(ax_beta.get_legend_handles_labels()[0]
                 + ax_disp.get_legend_handles_labels()[0]),
        loc=legend_loc, fontsize=8, ncol=2, framealpha=0.9)

    fig.subplots_adjust(top=0.91, bottom=0.14, left=0.12, right=0.88)
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

# Ring-wide counterpart to the local betx2/bety1 check: these are the numbers
# to watch when judging whether the coupling correction is doing its job
# everywhere and not just at the two markers it targets.
_in_straight_4d = (
    (tw4d.s >= STRAIGHT_SECTION_S_RANGE[0])
    & (tw4d.s <= STRAIGHT_SECTION_S_RANGE[1])
)
for _rdt_name, _rdt in (('f1001', tw4d.f1001), ('f1010', tw4d.f1010)):
    _abs = np.abs(_rdt)
    # nan-aware throughout: the solenoid bodies come back NaN by construction,
    # see the block comment above RDT_CASES. Report the count so a change in
    # it is visible rather than silently absorbed into the statistics.
    _n_nan = int((~np.isfinite(_abs)).sum())
    print(f'|{_rdt_name}|: max = {np.nanmax(_abs):.4g} (ring), '
          f'{np.nanmax(_abs[_in_straight_4d]):.4g} ({IP_PLOT} straight), '
          f'median = {np.nanmedian(_abs):.4g} (ring), '
          f'length-weighted ring average = '
          f'{_rdt_ring_average(tw4d, _rdt):.4g}, '
          f'{_n_nan} NaN pts (solenoid bodies)')

# What the straight's correctors are worth, as numbers behind fig4a-fig4d.
# Solenoids and doublet rotations stay on in the "off" column; only
# on_sol_{orbit,optics,coupling}_corr go to zero.
print()
print('--- correctors off (solenoids and doublet rotations still on) ---')
print(f'{"case":16s} {"qx":>11s} {"qy":>11s} {"max bety [m]":>14s} '
      f'{"max betx2/betx":>15s}')
for _tag, (_tw_c, _, _, _, _tw_nc) in zip(COMPARISON_TAGS, _TAG_RESULTS):
    _win = ((np.asarray(_tw_nc.s) >= -20) & (np.asarray(_tw_nc.s) <= 20))
    _ratio = np.asarray(_tw_nc.betx2 / _tw_nc.betx, dtype=float)
    print(f'{_tag + " corrected":16s} {_tw_c.qx:11.5f} {_tw_c.qy:11.5f}')
    print(f'{_tag + " no corr":16s} {_tw_nc.qx:11.5f} {_tw_nc.qy:11.5f} '
          f'{np.nanmax(np.asarray(_tw_nc.bety, dtype=float)[_win]):14.4g} '
          f'{np.nanmax(_ratio[_win]):15.4g}   (IR window)')


##############################################
# Save the beta / coupled-beta figures       #
#                                            #
# "coroff" = the straight's corrector knobs  #
# off; no suffix = fully corrected.          #
##############################################

COUPLING_STUDIES_PLOT_DIR = PLOT_DIR / 'Coupling_Studies'
COUPLING_STUDIES_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Note the two betas_*.pdf below come from BETA_COMPARISON_FIGS, which are
# drawn for the primary field tag only -- unlike the other six, which carry a
# 2 T and a 3 T panel each. A run with a different --b0 therefore overwrites
# those two with the other case's curves while leaving the rest unchanged.
BETA_COMPARISON_FIGS[0].savefig(
    COUPLING_STUDIES_PLOT_DIR / 'betas_IR.pdf', bbox_inches='tight')
BETA_COMPARISON_FIGS[1].savefig(
    COUPLING_STUDIES_PLOT_DIR / 'betas_straight.pdf', bbox_inches='tight')
fig4a.savefig(
    COUPLING_STUDIES_PLOT_DIR / 'betas_IR_coroff.pdf', bbox_inches='tight')
fig4b.savefig(
    COUPLING_STUDIES_PLOT_DIR / 'betas_straight_coroff.pdf',
    bbox_inches='tight')
fig3.savefig(
    COUPLING_STUDIES_PLOT_DIR / 'coupledbetas_IR.pdf', bbox_inches='tight')
fig4.savefig(
    COUPLING_STUDIES_PLOT_DIR / 'coupledbetas_straight.pdf',
    bbox_inches='tight')
fig4c.savefig(
    COUPLING_STUDIES_PLOT_DIR / 'coupledbetas_IR_coroff.pdf',
    bbox_inches='tight')
fig4d.savefig(
    COUPLING_STUDIES_PLOT_DIR / 'coupledbetas_straight_coroff.pdf',
    bbox_inches='tight')
print()
print(f'Saved 8 combined figures to {COUPLING_STUDIES_PLOT_DIR}')


#################################################
# The same plots, one field case per file       #
#################################################

SINGLE_CASE_PLOT_DIR = COUPLING_STUDIES_PLOT_DIR / 'plots'
SINGLE_CASE_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _save_single(fig_to_save, name):
    """Write a one-panel figure and close it.

    Closing matters: these exist only as files. Left open they would add a
    dozen windows to the plt.show() at the end, which is exactly the pile-up
    that makes 004a unpleasant to run interactively.
    """
    fig_to_save.savefig(SINGLE_CASE_PLOT_DIR / name, bbox_inches='tight')
    plt.close(fig_to_save)


_save_single(_betx2_bety1_fig((-20, 20), '', True, tags=['2T']),
             'coupledbetas_IR_2T.pdf')
_save_single(_betx2_bety1_fig((-20, 20), '', True, tags=['3T']),
             'coupledbetas_IR_3T.pdf')
_save_single(
    _betx2_bety1_fig((-1400, 1400), ' (full straight section)', False,
                     tags=['2T']),
    'coupledbetas_straight_2T.pdf')
_save_single(
    _betx2_bety1_fig((-1400, 1400), ' (full straight section)', False,
                     tags=['3T']),
    'coupledbetas_straight_3T.pdf')

_save_single(_nocorr_fig((-20, 20), '', True, 'beta', tags=['2T']),
             'betas_IR_coroff_2T.pdf')
_save_single(_nocorr_fig((-20, 20), '', True, 'beta', tags=['3T']),
             'betas_IR_coroff_3T.pdf')
_save_single(
    _nocorr_fig((-1400, 1400), ' (full straight)', False, 'beta',
                tags=['2T']),
    'betas_straight_coroff_2T.pdf')
_save_single(
    _nocorr_fig((-1400, 1400), ' (full straight)', False, 'beta',
                tags=['3T']),
    'betas_straight_coroff_3T.pdf')

_save_single(_nocorr_fig((-20, 20), '', True, 'coupling', tags=['2T']),
             'coupledbetas_IR_coroff_2T.pdf')
_save_single(_nocorr_fig((-20, 20), '', True, 'coupling', tags=['3T']),
             'coupledbetas_IR_coroff_3T.pdf')
_save_single(
    _nocorr_fig((-1400, 1400), ' (full straight)', False, 'coupling',
                tags=['2T']),
    'coupledbetas_straight_coroff_2T.pdf')
_save_single(
    _nocorr_fig((-1400, 1400), ' (full straight)', False, 'coupling',
                tags=['3T']),
    'coupledbetas_straight_coroff_3T.pdf')

# BETA_COMPARISON_FIGS is drawn for the primary field tag only, so these two
# already are single-case plots -- they just need the tag in the name. That
# also removes the overwrite noted at the combined-figure block above: a run
# at each --b0 now leaves one file per case instead of clobbering.
BETA_COMPARISON_FIGS[0].savefig(
    SINGLE_CASE_PLOT_DIR / f'betas_IR_{FIELD_TAG}.pdf', bbox_inches='tight')
BETA_COMPARISON_FIGS[1].savefig(
    SINGLE_CASE_PLOT_DIR / f'betas_straight_{FIELD_TAG}.pdf',
    bbox_inches='tight')

print(f'Saved 14 single-case figures to {SINGLE_CASE_PLOT_DIR}')

plt.show()
