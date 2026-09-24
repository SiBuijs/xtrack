"""Beta functions of the old and current corrected rings, at 2 T and 3 T.

Four lattices, all SplineBoris, all with solenoids and their corrections on:

    old 2 T / old 3 T          from git, at OLD_REV
    current 2 T / current 3 T  from the working tree

Two things changed between the two generations, and this script exists to show
what they did to the optics:

1. The mid-bend quads (004c, commit f4f9a77b9): the six bends framing each
   IP are cut in half and a zero-length quad `qbmid_<bend>` goes at each
   cut, giving the half-straight optics match local handles between the IP and
   the nearest chromatic sextupole. 24 bends, 72 quads.
2. The optics-match conditioning fix (OPTICS_RCOND = 1e-6 plus a staged pass-1
   solve; see claude_notes/10_conditioning_the_optics_match.md). Before it the
   half-straight match was rank-deficient with no SVD truncation and landed on
   an arbitrary null-space point each run.

Why OLD_REV is not 8d46a1fbe, which is where this was first looked for: at that
commit the 3 T lattice had ALREADY been regenerated with the cut bends (72
`qbmid_`), while the 2 T had not -- so the pair is not internally consistent.
f6dfc5087 (2026-09-01) is the newest commit at which both are still pre-cut and
pre-conditioning-fix. The 2 T blob is byte-identical at the two commits anyway
(written 2026-07-21 by 6f04ddb9b and never regenerated in between), so only the
3 T file actually differs between the two choices.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from aperture_study_io import old_lattice_path


HERE = Path(__file__).parent

# The newest commit whose 2 T *and* 3 T corrected lattices both predate the
# mid-bend cuts and the optics-match conditioning fix. See the module docstring
# for why this is not 8d46a1fbe.
DEFAULT_OLD_REV = 'f6dfc5087'

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']

# Chromatic sextupole nearest the IP on each side -- the element the current
# optics match actually targets (betx/bety/dy, tag='sext'). Copied from
# config[ip]['sext_left'] / ['sext_right'] in
# 004c_correct_solenoids_in_fcc_ring.py, which is the source of truth; same
# duplication pattern as 004d's CORRECTOR_QUADS_BY_IP.
#
# Note this is the correction's *target*, not the diagnosed driver of the large
# Q''y. That is the sdy1 -I partner's phase advance from QD0, which the match
# leaves free (claude_notes/08-10). sdm1 is nonetheless the right marker for a
# beta figure: 004c records bety at sdm1r.0 going 1.47 m bare -> 2.87 m (2 T)
# -> 310 m (3 T), which is exactly the bump these panels should show.
TARGET_SEXTUPOLES_BY_IP = {
    'ipa': ['sdm1l.7', 'sdm1r.0'],
    'ipd': ['sdm1l.1', 'sdm1r.2'],
    'ipg': ['sdm1l.3', 'sdm1r.4'],
    'ipj': ['sdm1l.5', 'sdm1r.6'],
}

LATTICE_NAME_TEMPLATE = (
    'fccee_z_lcc_splineboris_solenoids_coupling_corrected_{tag}.json')

# The six bends framing each IP that the current scheme cuts at their centre to
# install a zero-length quadrupole `qbmid_<bend>`. Three upstream and three
# downstream, in beam order. Copied from
# config[ip]['bend_for_mid_quad_correction'] in
# 004c_correct_solenoids_in_fcc_ring.py, which is the source of truth -- same
# duplication pattern as TARGET_SEXTUPOLES_BY_IP above.
#
# These bends are the whole reason the cut exists: they fill essentially all the
# space between the final-focus doublet and the chromatic sextupole on each side
# (upstream ~60 m each, downstream ~25 m each), so without a handle inside them
# the half-straight optics match has nothing to pull on between the IP and its
# target. The old generation at OLD_REV has them whole.
CUT_BENDS_BY_IP = {
    'ipa': ['b0cl.3', 'b0bl.3', 'b0al.3', 'b1ra.0', 'b1rb.0', 'b1rc.0'],
    'ipd': ['b0cl.0', 'b0bl.0', 'b0al.0', 'b1ra.1', 'b1rb.1', 'b1rc.1'],
    'ipg': ['b0cl.1', 'b0bl.1', 'b0al.1', 'b1ra.2', 'b1rb.2', 'b1rc.2'],
    'ipj': ['b0cl.2', 'b0bl.2', 'b0al.2', 'b1ra.3', 'b1rb.3', 'b1rc.3'],
}

BEND_MID_QUAD_PREFIX = 'qbmid_'    # 004c's BEND_MID_QUAD_PREFIX

# Half-width of the zoom window drawn around each targeted sextupole.
SEXT_ZOOM_HALF_WIDTH = 30.0    # m

# beta_x blue / beta_y orange follows 004d; 2 T dashed / 3 T solid separates the
# two field cases within a panel. Old vs current is the panel split, since four
# styles is exactly what colour x linestyle gives.
PLANE_COLOR = {'betx': 'C0', 'bety': 'C1'}
TAG_LINESTYLE = {'2T': '--', '3T': '-'}
COMPARISON_TAGS = ['2T', '3T']
GENERATIONS = ['old', 'current']

# What the two generations are called in every user-facing string. The internal
# keys stay 'old'/'current' because they also select the lattice source (git
# blob vs working tree); only the display name changes.
GENERATION_LABEL = {'old': 'before', 'current': 'after'}


_parser = argparse.ArgumentParser(
    description='Compare beta functions of the old and current corrected '
                'rings, at 2 T and 3 T.')
_parser.add_argument(
    '--old-rev', default=DEFAULT_OLD_REV,
    help='Git revision to read the old lattices from '
         f'(default: {DEFAULT_OLD_REV}).')
_parser.add_argument(
    '--ip', default='ipa', choices=sorted(TARGET_SEXTUPOLES_BY_IP),
    help='IP to centre the s-axis on (default: ipa).')
_parser.add_argument(
    '--save', action='store_true',
    help='Write the figures as PNGs to the shared plot directory.')
_parser.add_argument(
    '--no-show', action='store_true',
    help='Skip the interactive plt.show() at the end (e.g. for batch runs).')
_args = _parser.parse_args()

OLD_REV = _args.old_rev
IP_PLOT = _args.ip


###############################################################################
# Getting the old lattices out of git                                         #
###############################################################################

def _lattice_path(generation, tag):
    """Path to the corrected SplineBoris lattice for one case."""
    filename = LATTICE_NAME_TEMPLATE.format(tag=tag)
    if generation == 'current':
        path = HERE / filename
        if not path.exists():
            raise SystemExit(f'{path} not found -- run 004c for the {tag} case.')
        return path
    return old_lattice_path(OLD_REV, filename)


###############################################################################
# Loading and twissing                                                        #
###############################################################################

def _load_and_twiss(json_path, label):
    """(twiss, pre-cut table) for one corrected lattice, solenoids on.

    The same recipe as 004d's _twiss_for_tag: cycle so the line starts at the
    end of ipa's upstream dispersion suppressor, cut the +-11 m around each IP
    into 0.2 m slices so the IR curves are resolved, switch the solenoids and
    their corrections on, and zero the s-axis at IP_PLOT.

    Deliberately NOT following 004d in one respect: 004d twisses its primary
    field tag in 6d and the comparison tags in 4d. All four cases here get the
    same twiss4d, or the curves would not be comparable across panels.

    The table is taken before the cuts -- element names like 'qd0ar.0' become
    chains of 'qd0ar.0..N' afterwards, so marker lookups have to use the
    uncut one (same gotcha as 004d's _compute_ir_markers).
    """
    env = xt.load(json_path)
    line = env.fccee_p_ring.copy(shallow=True)

    line.cycle(f'end_ds_start_straight_{IP_NAMES[0]}')
    table = line.get_table()

    for ip_name in IP_NAMES:
        line.cut_at_s(np.arange(table['s', ip_name] + 2.4,
                                table['s', ip_name] + 11.0, 0.2))
        line.cut_at_s(np.arange(table['s', ip_name] - 11.0,
                                table['s', ip_name] - 2.4, 0.2))

    # Also resolve the zoom windows around the targeted sextupoles. Without
    # these the twiss grid out there is one row per element, tens of metres
    # apart, and the zoom panels draw beta as a few straight segments between
    # arc magnets -- which understates the real shape of the bump the whole
    # figure is about.
    #
    # Cuts landing inside a targeted sextupole are dropped: cut_at_s would
    # replace 'sdm1r.0' by a chain of 'sdm1r.0..N' and the beta lookups in the
    # summary below would stop resolving.
    _sext_extents = [
        (float(table['s_start', name]), float(table['s_end', name]))
        for name in TARGET_SEXTUPOLES_BY_IP[IP_PLOT] if name in table.name
    ]
    for s_start, s_end in _sext_extents:
        s_mid = 0.5 * (s_start + s_end)
        s_cuts = np.arange(s_mid - SEXT_ZOOM_HALF_WIDTH,
                           s_mid + SEXT_ZOOM_HALF_WIDTH, 0.5)
        keep = np.ones(s_cuts.shape, dtype=bool)
        for lo, hi in _sext_extents:
            keep &= ~((s_cuts > lo) & (s_cuts < hi))
        line.cut_at_s(s_cuts[keep])

    # The old lattices are a different vintage, so check the knobs are there
    # rather than letting a missing one surface as a confusing twiss failure.
    for ip_name in IP_NAMES:
        for knob in (f'on_sol_{ip_name}', f'on_sol_corr_{ip_name}'):
            if knob not in line.vars:
                raise SystemExit(
                    f'{label}: {json_path.name} has no knob {knob!r}. That '
                    'lattice predates the current correction scheme; pick a '
                    'later --old-rev.')
        line[f'on_sol_{ip_name}'] = 1
        line[f'on_sol_corr_{ip_name}'] = 1

    tw = line.twiss4d(strengths=True)
    tw.zero_at(IP_PLOT)
    return tw, table


print(f'IP {IP_PLOT}, old revision {OLD_REV}')
TWISS = {}
TABLES = {}
for _generation in GENERATIONS:
    for _tag in COMPARISON_TAGS:
        _label = f'{GENERATION_LABEL[_generation]} {_tag}'
        _path = _lattice_path(_generation, _tag)
        print(f'loading {_label:12s} {_path.name}')
        try:
            TWISS[_generation, _tag], TABLES[_generation, _tag] = (
                _load_and_twiss(_path, _label))
        except SystemExit:
            raise
        except Exception as exc:
            # The "before" ring is the badly-corrected one; if it will not close,
            # say which case failed and keep the other three rather than
            # losing the whole figure set.
            print(f'  FAILED to twiss {_label}: {type(exc).__name__}: {exc}')

if not TWISS:
    raise SystemExit('No lattice twissed successfully; nothing to plot.')


###############################################################################
# Where the targeted sextupoles sit                                           #
###############################################################################

def _sextupole_spans(table):
    """[(name, s_start, s_end)] for the sextupoles the match targets at
    IP_PLOT, in the s-frame zeroed at IP_PLOT.

    sdm1* elements arrive unsliced in the corrected lattices (16 plain names,
    no '..N' suffixes), so the table lookup is direct. Guarded anyway, since a
    future 004c that slices them would otherwise silently drop the markers.
    """
    s_ip = table['s', IP_PLOT]
    spans = []
    for name in TARGET_SEXTUPOLES_BY_IP[IP_PLOT]:
        if name not in table.name:
            print(f'NOTE: {name!r} not found in the table; marker skipped. '
                  'Has the IR sextupole naming or slicing changed?')
            continue
        spans.append((name,
                      float(table['s_start', name]) - s_ip,
                      float(table['s_end', name]) - s_ip))
    return spans


def _bend_spans(table):
    """[(name, s_start, s_end, n_parts, s_mid_quad)] for the six bends the
    current scheme cuts at IP_PLOT, in the s-frame zeroed at IP_PLOT.

    Resolved per generation, because the whole point is that the two
    generations disagree: at OLD_REV each bend is a single element with no trim
    quad, while the current lattice has it split and carries a `qbmid_<bend>`
    at the centre. `s_mid_quad` is None when that quad is absent.

    The extent is taken as min(s_start)..max(s_end) over every piece whose name
    is the bend or one of its `..N` slices, so a bend that later gets sliced
    further still reports its true physical span.
    """
    s_ip = table['s', IP_PLOT]
    names = list(table.name)
    spans = []
    for name in CUT_BENDS_BY_IP[IP_PLOT]:
        parts = [n for n in names if n == name or n.startswith(name + '..')]
        if not parts:
            print(f'NOTE: bend {name!r} not found in the table; skipped. '
                  'Has the near-IP bend naming changed?')
            continue
        quad = BEND_MID_QUAD_PREFIX + name
        spans.append((
            name,
            min(float(table['s_start', n]) for n in parts) - s_ip,
            max(float(table['s_end', n]) for n in parts) - s_ip,
            len(parts),
            float(table['s', quad]) - s_ip if quad in names else None,
        ))
    return spans


# Resolved once and reused on both panels: the two generations share the arc
# geometry, so the sextupoles sit at the same s in all four cases. Prefer a
# current-generation table, falling back to whatever loaded.
_MARKER_KEY = next(
    (key for key in TABLES if key[0] == 'current'), next(iter(TABLES)))
SEXT_SPANS = _sextupole_spans(TABLES[_MARKER_KEY])

# Per generation -- the cut and the quads only exist in the current one.
# Either tag's table will do within a generation; they share the arc geometry.
BEND_SPANS = {}
for _generation in GENERATIONS:
    _key = next((k for k in TABLES if k[0] == _generation), None)
    if _key is not None:
        BEND_SPANS[_generation] = _bend_spans(TABLES[_key])

STRAIGHT_TABLE = TABLES[_MARKER_KEY]
_s_ip_ref = STRAIGHT_TABLE['s', IP_PLOT]
STRAIGHT_SECTION_S_RANGE = (
    float(STRAIGHT_TABLE['s', f'end_ds_start_straight_{IP_PLOT}']) - _s_ip_ref,
    float(STRAIGHT_TABLE['s', f'end_straight_start_ds_{IP_PLOT}']) - _s_ip_ref,
)


###############################################################################
# Plotting                                                                    #
###############################################################################

def _autoscale_y_to_xlim(ax, xlim, margin=0.1):
    """Rescale ax's ylim to the data actually visible within xlim.

    Must be called BEFORE the axvspan/axvline decoration: an axvline is a
    Line2D with y-data [0, 1], which would otherwise be folded into the
    min/max and flatten the panel. Copied from 004d (which is a script, not an
    importable module) -- see the note there and in claude_notes/07.
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
        ax.set_ylim(y_min / 5.0, y_max * 5.0)
    else:
        span = y_max - y_min
        pad = margin * span if span > 0 else max(abs(y_max), 1.0) * margin
        ax.set_ylim(y_min - pad, y_max + pad)


def _beta_figure(xlim, title_suffix, yscale, stem):
    """One figure: the "before" ring on top, "after" below, four curves each.

    sharey is the point of the layout. Without it each panel autoscales to its
    own data and a bump that is two orders of magnitude larger in one
    generation reads as the same size in both.
    """
    fig, axs = plt.subplots(len(GENERATIONS), 1, sharex=True, sharey=True,
                            figsize=(7.5, 6.4))

    for ax, generation in zip(axs, GENERATIONS):
        for tag in COMPARISON_TAGS:
            tw = TWISS.get((generation, tag))
            if tw is None:
                continue
            for plane, symbol in (('betx', r'\beta_x'), ('bety', r'\beta_y')):
                ax.plot(tw.s, tw[plane], color=PLANE_COLOR[plane],
                        linestyle=TAG_LINESTYLE[tag], linewidth=1.2,
                        label=fr'${symbol}$ ({tag})')
        ax.set_ylabel(r'$\beta_{x,y}$ [m]')
        ax.set_yscale(yscale)

    # Two lines: on the wider windows the one-line form runs off the figure.
    axs[0].set_title(
        f'{IP_PLOT}: beta functions, before ({OLD_REV}) vs after'
        f'\n{title_suffix}')
    axs[-1].set_xlabel('s [m]')

    # Both panels keep their s ticks, so either can be cropped out and used on
    # its own; the hspace below makes room for the extra label.
    axs[0].tick_params(labelbottom=True)
    axs[0].set_xlabel('s [m]')

    # xlim and the autoscale both before the decoration -- see
    # _autoscale_y_to_xlim. sharey means scaling one panel scales both, so this
    # has to consider every panel's data: do it per panel and keep the union.
    axs[-1].set_xlim(*xlim)
    y_lo, y_hi = np.inf, -np.inf
    for ax in axs:
        _autoscale_y_to_xlim(ax, xlim)
        lo, hi = ax.get_ylim()
        y_lo, y_hi = min(y_lo, lo), max(y_hi, hi)
    if np.isfinite(y_lo) and np.isfinite(y_hi):
        if yscale == 'linear':
            # beta is positive-definite, so pin the floor at 0 rather than let
            # the 10 % pad hang the axis below zero.
            y_lo, y_hi = 0.0, y_hi * 1.25
        else:
            # Extra headroom for the legend: on the log IR panel the beta_y
            # plateau runs along the top of the axis and the legend would sit
            # on it.
            y_hi *= 8.0
        axs[0].set_ylim(y_lo, y_hi)

    for ax, generation in zip(axs, GENERATIONS):
        ax.grid(True, alpha=0.4)
        for name, s_start, s_end in SEXT_SPANS:
            # A thin sextupole would give a zero-width span and draw nothing,
            # so fall back to a line in that case.
            if s_end > s_start:
                ax.axvspan(s_start, s_end, color='C2', alpha=0.25, zorder=0)
            else:
                ax.axvline(s_start, color='C2', alpha=0.6, zorder=0)
        ax.text(0.01, 0.95, GENERATION_LABEL[generation],
                transform=ax.transAxes,
                ha='left', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='0.6', alpha=0.85))
        ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    # Name the marked sextupoles once, under the figure rather than inside a
    # panel -- in axes coordinates it lands on the curves on every range where
    # beta is small at the bottom left, which is most of them. Only the ones
    # actually inside this figure's xlim are named.
    visible = [name for name, s_start, s_end in SEXT_SPANS
               if s_end >= xlim[0] and s_start <= xlim[1]]
    if visible:
        fig.text(0.11, 0.015,
                 'green band: sextupole targeted by the optics correction '
                 f'({", ".join(visible)})',
                 ha='left', va='bottom', fontsize=7, color='0.3')

    fig.subplots_adjust(hspace=0.28, top=0.93, bottom=0.11, left=0.11,
                        right=0.97)
    fig._004m_stem = stem
    return fig


# Layout-strip geometry, in the layout panel's own 0..1 axes coordinates. One
# row per figure now that the generations are drawn separately.
_LAYOUT_ROW_Y = 0.45
_LAYOUT_ROW_H = 0.30


def _beta_max_in_window(xlim):
    """Largest beta of any case inside xlim.

    Used to give the two cut-bend figures one shared vertical scale. They are
    separate figures, so `sharey` cannot do it, and letting each autoscale
    would draw the two generations at different magnifications -- exactly the
    failure mode `_beta_figure`'s sharey exists to avoid.
    """
    top = 0.0
    for generation in GENERATIONS:
        for tag in COMPARISON_TAGS:
            tw = TWISS.get((generation, tag))
            if tw is None:
                continue
            in_window = ((np.asarray(tw.s, dtype=float) >= xlim[0])
                         & (np.asarray(tw.s, dtype=float) <= xlim[1]))
            for plane in ('betx', 'bety'):
                values = np.asarray(tw[plane], dtype=float)[in_window]
                if values.size:
                    top = max(top, float(np.nanmax(values)))
    return top


def _bend_layout_figure(xlim, generation, y_top):
    """One generation's beta functions over the near-IP window, with its six
    bends drawn underneath as a layout strip.

    This is the geometry behind the correction: the three upstream bends
    (~60 m each) and three downstream bends (~25 m each) take up nearly all the
    distance between the final-focus doublet and the chromatic sextupole the
    half-straight match targets. Whole, as at OLD_REV, they are dead space with
    no quadrupole handle in them; cut at the centre they each carry a
    zero-length `qbmid_` quad, which is what gives the match something
    local to vary between the IP and its target.

    `y_top` is passed in rather than fitted here so the before/after pair share
    one vertical scale across two separate figures.
    """
    label = GENERATION_LABEL[generation]
    fig, (ax_beta, ax_layout) = plt.subplots(
        2, 1, sharex=True, figsize=(9.5, 6.6),
        gridspec_kw=dict(height_ratios=[3.0, 1.0]))

    for tag in COMPARISON_TAGS:
        tw = TWISS.get((generation, tag))
        if tw is None:
            continue
        for plane, symbol in (('betx', r'\beta_x'), ('bety', r'\beta_y')):
            ax_beta.plot(
                tw.s, tw[plane], color=PLANE_COLOR[plane],
                linestyle=TAG_LINESTYLE[tag], linewidth=1.2,
                label=fr'${symbol}$ ({tag})')
    ax_beta.set_ylabel(r'$\beta_{x,y}$ [m]')
    title = (f'{IP_PLOT}: the six bends cut for the mid-bend quadrupoles'
             f'\n{label}' + (f' ({OLD_REV})' if generation == 'old' else ''))
    ax_beta.set_title(title)

    # beta is positive-definite, so the floor is pinned at 0 rather than left
    # hanging below it by a pad; the shared top already carries the 1.35 the
    # legend needs.
    ax_beta.set_xlim(*xlim)
    ax_beta.set_ylim(0.0, y_top)
    ax_beta.grid(True, alpha=0.4)
    for _name, s_start, s_end in SEXT_SPANS:
        if s_end > s_start:
            ax_beta.axvspan(s_start, s_end, color='C2', alpha=0.25, zorder=0)
        else:
            ax_beta.axvline(s_start, color='C2', alpha=0.6, zorder=0)
    ax_beta.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    # sharex hides the upper panel's tick labels by default. Put them back, as
    # the other figures in this file do, so the beta panel can be read (or
    # cropped out and used) without the layout strip underneath it.
    ax_beta.tick_params(labelbottom=True)
    ax_beta.set_xlabel('s [m]')

    # --- the layout strip ------------------------------------------------- #
    ax_layout.set_ylim(0.0, 1.0)
    ax_layout.set_yticks([])
    for spine in ('left', 'right', 'top'):
        ax_layout.spines[spine].set_visible(False)
    ax_layout.set_xlabel('s [m]')

    spans = BEND_SPANS.get(generation)
    y_centre = _LAYOUT_ROW_Y
    if spans is None:
        ax_layout.text(xlim[0], y_centre, f'  {label}: not loaded',
                       va='center', fontsize=8, color='0.5')
    else:
        for _name, s_start, s_end, _n_parts, s_quad in spans:
            ax_layout.add_patch(plt.Rectangle(
                (s_start, y_centre - 0.5 * _LAYOUT_ROW_H), s_end - s_start,
                _LAYOUT_ROW_H, facecolor='C0', alpha=0.35,
                edgecolor='C0', linewidth=0.8, zorder=2))
            if s_quad is not None:
                # The cut, and the quad sitting in it.
                ax_layout.plot(
                    [s_quad, s_quad],
                    [y_centre - 0.5 * _LAYOUT_ROW_H - 0.08,
                     y_centre + 0.5 * _LAYOUT_ROW_H + 0.08],
                    color='C3', linewidth=1.4, zorder=3)
        n_quads = sum(1 for s in spans if s[4] is not None)
        ax_layout.text(
            xlim[0] + 0.005 * (xlim[1] - xlim[0]),
            y_centre + 0.5 * _LAYOUT_ROW_H + 0.14,
            f'{label}: {len(spans)} bends, {n_quads} quads',
            va='bottom', ha='left', fontsize=8, color='0.25')

    # Same green as the beta panel, so the sextupole reads across both.
    for _name, s_start, s_end in SEXT_SPANS:
        if s_end > s_start:
            ax_layout.axvspan(s_start, s_end, color='C2', alpha=0.25, zorder=0)
        else:
            ax_layout.axvline(s_start, color='C2', alpha=0.6, zorder=0)
    ax_layout.axvline(0.0, color='0.3', linewidth=0.8, linestyle=':', zorder=1)
    ax_layout.text(0.0, 0.03, ' IP', va='bottom', ha='left', fontsize=8,
                   color='0.3')

    fig.text(0.11, 0.015,
             'blue: bends framing the IP   '
             'red: mid-bend cut carrying the $qbmid\\_$ quadrupole   '
             'green: sextupole targeted by the optics correction',
             ha='left', va='bottom', fontsize=7, color='0.3')

    # Roomier hspace than a bare shared-x pair would need: the beta panel now
    # carries its own tick labels and s-axis label between the two panels.
    fig.subplots_adjust(hspace=0.30, top=0.90, bottom=0.13, left=0.09,
                        right=0.97)
    fig._004m_stem = f'cut_bends_{label}'
    return fig


# The IR window has to be log: the IP waist (betx* ~ 0.09 m, bety* ~ 0.7 mm)
# sits 4-7 decades below the doublet peaks and collapses onto zero on a linear
# axis. Everything wider is linear, where the beta bumps read as bumps.
RANGES = [
    ((-20.0, 20.0), 'IR', 'log', 'ir'),
    ((*STRAIGHT_SECTION_S_RANGE,), 'full straight section', 'linear',
     'straight'),
]

# Wide window around the IP, out far enough to contain both targeted
# sextupoles -- and with them the six bends, which fill nearly all the space in
# between. Log, for the same reason the IR panel is: this window still holds the
# IP waist (bety* ~ 0.7 mm) and the doublet peaks (~1e4 m), so on a linear axis
# everything outside the doublet is pinned to the floor.
if SEXT_SPANS:
    _wide_lo = min(s_start for _, s_start, _ in SEXT_SPANS)
    _wide_hi = max(s_end for _, _, s_end in SEXT_SPANS)
    # Enough pad that the sextupole bands sit inside the frame rather than
    # under the spines.
    _wide_pad = 0.06 * (_wide_hi - _wide_lo)
    RANGES.append((
        (_wide_lo - _wide_pad, _wide_hi + _wide_pad),
        'IP out to the targeted sextupoles', 'linear', 'ip_to_sext'))
# One zoom per targeted sextupole, derived from the resolved positions so the
# windows follow automatically if --ip changes.
for _name, _s_start, _s_end in SEXT_SPANS:
    _s_mid = 0.5 * (_s_start + _s_end)
    RANGES.append((
        (_s_mid - SEXT_ZOOM_HALF_WIDTH, _s_mid + SEXT_ZOOM_HALF_WIDTH),
        f'zoom on {_name}', 'linear', f'zoom_{_name.replace(".", "_")}'))

FIGURES = [_beta_figure(*spec) for spec in RANGES]

# The cut-bend layout figure spans the bends and the sextupoles together --
# which in practice is the same window as 'ip_to_sext', since the bends sit
# between the doublet and the sextupoles.
_LAYOUT_EDGES = (
    [s for spans in BEND_SPANS.values() for _, s, _, _, _ in spans]
    + [s for spans in BEND_SPANS.values() for _, _, s, _, _ in spans]
    + [s for _, s, _ in SEXT_SPANS] + [s for _, _, s in SEXT_SPANS]
)
if _LAYOUT_EDGES:
    _lo, _hi = min(_LAYOUT_EDGES), max(_LAYOUT_EDGES)
    _pad = 0.06 * (_hi - _lo)
    _LAYOUT_XLIM = (_lo - _pad, _hi + _pad)
    # One shared top for both figures; the 1.35 is the legend's headroom.
    _LAYOUT_YTOP = _beta_max_in_window(_LAYOUT_XLIM) * 1.35
    FIGURES += [_bend_layout_figure(_LAYOUT_XLIM, gen, _LAYOUT_YTOP)
                for gen in GENERATIONS]
else:
    print('NOTE: no bends or sextupoles resolved; cut-bend figure skipped.')


###############################################################################
# Numbers behind the figures                                                  #
###############################################################################

print()
print('--- tunes (twiss4d, solenoids and corrections on) ---')
print(f'{"case":14s} {"qx":>12s} {"qy":>12s}')
for generation in GENERATIONS:
    for tag in COMPARISON_TAGS:
        tw = TWISS.get((generation, tag))
        if tw is None:
            continue
        print(f'{GENERATION_LABEL[generation] + " " + tag:14s} {tw.qx:12.6f} {tw.qy:12.6f}')

print()
print('--- beta at the targeted sextupoles ---')
for name, _, _ in SEXT_SPANS:
    print(f'  {name}')
    print(f'    {"case":14s} {"betx [m]":>12s} {"bety [m]":>12s}')
    for generation in GENERATIONS:
        for tag in COMPARISON_TAGS:
            tw = TWISS.get((generation, tag))
            if tw is None:
                continue
            # The sextupole is far from the IP cuts, so it survives as one row.
            print(f'    {GENERATION_LABEL[generation] + " " + tag:14s} '
                  f'{tw["betx", name]:12.4f} {tw["bety", name]:12.4f}')

print()
print('--- bends cut for the mid-bend quadrupoles ---')
for generation in GENERATIONS:
    spans = BEND_SPANS.get(generation)
    if spans is None:
        continue
    n_quads = sum(1 for s in spans if s[4] is not None)
    print(f'  {GENERATION_LABEL[generation]}: {len(spans)} bends, '
          f'{n_quads} quads')
    print(f'    {"bend":10s} {"s_start":>10s} {"s_end":>10s} {"L [m]":>8s} '
          f'{"pieces":>7s} {"qbmid @":>10s}')
    for name, s_start, s_end, n_parts, s_quad in spans:
        quad_s = f'{s_quad:10.3f}' if s_quad is not None else f'{"--":>10s}'
        print(f'    {name:10s} {s_start:10.3f} {s_end:10.3f} '
              f'{s_end - s_start:8.3f} {n_parts:7d} {quad_s}')

print()
print(f'--- max beta over {IP_PLOT}\'s straight section '
      f'({STRAIGHT_SECTION_S_RANGE[0]:.1f} to '
      f'{STRAIGHT_SECTION_S_RANGE[1]:.1f} m) ---')
print(f'{"case":14s} {"max betx [m]":>14s} {"max bety [m]":>14s}')
for generation in GENERATIONS:
    for tag in COMPARISON_TAGS:
        tw = TWISS.get((generation, tag))
        if tw is None:
            continue
        in_straight = ((tw.s >= STRAIGHT_SECTION_S_RANGE[0])
                       & (tw.s <= STRAIGHT_SECTION_S_RANGE[1]))
        print(f'{GENERATION_LABEL[generation] + " " + tag:14s} '
              f'{np.nanmax(tw.betx[in_straight]):14.4g} '
              f'{np.nanmax(tw.bety[in_straight]):14.4g}')


if _args.save:
    from aperture_study_io import PLOT_DIR
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    for fig in FIGURES:
        path = PLOT_DIR / f'004m_old_vs_new_beta_{fig._004m_stem}_{IP_PLOT}.png'
        fig.savefig(path, dpi=200)
        print(f'saved {path}')

if not _args.no_show:
    plt.show()
