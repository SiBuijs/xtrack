"""Beta functions of the old and current corrected rings, at 2 T and 3 T.

Four lattices, all SplineBoris, all with solenoids and their corrections on:

    old 2 T / old 3 T          from git, at OLD_REV
    current 2 T / current 3 T  from the working tree

Two things changed between the two generations, and this script exists to show
what they did to the optics:

1. The mid-bend trim quads (004c, commit f4f9a77b9): the six bends framing each
   IP are cut in half and a zero-length trim quad `qbmid_<bend>` goes at each
   cut, giving the half-straight optics match local handles between the IP and
   the nearest chromatic sextupole. 24 bends, 72 trim quads.
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
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt


HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]

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

OLD_LATTICE_CACHE = HERE / 'data' / 'old_lattices'

# Half-width of the zoom window drawn around each targeted sextupole.
SEXT_ZOOM_HALF_WIDTH = 30.0    # m

# beta_x blue / beta_y orange follows 004d; 2 T dashed / 3 T solid separates the
# two field cases within a panel. Old vs current is the panel split, since four
# styles is exactly what colour x linestyle gives.
PLANE_COLOR = {'betx': 'C0', 'bety': 'C1'}
TAG_LINESTYLE = {'2T': '--', '3T': '-'}
COMPARISON_TAGS = ['2T', '3T']
GENERATIONS = ['old', 'current']


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

def _old_lattice_path(rev, filename):
    """Extract examples/fcc_ee_solenoid/<filename> at <rev> into the cache.

    Each 004c run overwrote the corrected lattices in place, so the old ones
    exist only as git blobs -- there is no file on disk to point at. This is a
    plain read of history: the working tree is never touched, and the cache
    lives under data/, which is gitignored, so it cannot be committed by
    accident.

    `git cat-file -p` rather than `git show`: no pager and no auto-decoration
    on a 15 MB blob. The file is written to a .part name and renamed, so an
    interrupted run cannot leave a truncated JSON behind that the next run
    would happily load.
    """
    out = OLD_LATTICE_CACHE / f'{rev}_{filename}'
    if out.exists():
        return out
    OLD_LATTICE_CACHE.mkdir(parents=True, exist_ok=True)
    spec = f'{rev}:examples/fcc_ee_solenoid/{filename}'
    print(f'  extracting {spec} -> {out.relative_to(HERE)}')
    tmp = out.with_name(out.name + '.part')
    try:
        with open(tmp, 'wb') as fh:
            subprocess.run(['git', 'cat-file', '-p', spec], cwd=REPO_ROOT,
                           stdout=fh, check=True)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f'Could not read {spec} from git (exit {exc.returncode}). Check '
            f'that {rev!r} is a valid revision and that the lattice existed '
            'there.') from exc
    tmp.replace(out)
    return out


def _lattice_path(generation, tag):
    """Path to the corrected SplineBoris lattice for one case."""
    filename = LATTICE_NAME_TEMPLATE.format(tag=tag)
    if generation == 'current':
        path = HERE / filename
        if not path.exists():
            raise SystemExit(f'{path} not found -- run 004c for the {tag} case.')
        return path
    return _old_lattice_path(OLD_REV, filename)


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
        _label = f'{_generation} {_tag}'
        _path = _lattice_path(_generation, _tag)
        print(f'loading {_label:12s} {_path.name}')
        try:
            TWISS[_generation, _tag], TABLES[_generation, _tag] = (
                _load_and_twiss(_path, _label))
        except SystemExit:
            raise
        except Exception as exc:
            # The old ring is the badly-corrected one; if it will not close,
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


# Resolved once and reused on both panels: the two generations share the arc
# geometry, so the sextupoles sit at the same s in all four cases. Prefer a
# current-generation table, falling back to whatever loaded.
_MARKER_KEY = next(
    (key for key in TABLES if key[0] == 'current'), next(iter(TABLES)))
SEXT_SPANS = _sextupole_spans(TABLES[_MARKER_KEY])

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
    """One figure: old ring on top, current ring below, four curves each.

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

    axs[0].set_title(
        f'{IP_PLOT}: beta functions, old ({OLD_REV}) vs current ring '
        f'-- {title_suffix}')
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
        ax.text(0.01, 0.95, f'{generation} ring', transform=ax.transAxes,
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


# The IR window has to be log: the IP waist (betx* ~ 0.09 m, bety* ~ 0.7 mm)
# sits 4-7 decades below the doublet peaks and collapses onto zero on a linear
# axis. Everything wider is linear, where the beta bumps read as bumps.
RANGES = [
    ((-20.0, 20.0), 'IR', 'log', 'ir'),
    ((*STRAIGHT_SECTION_S_RANGE,), 'full straight section', 'linear',
     'straight'),
]
# One zoom per targeted sextupole, derived from the resolved positions so the
# windows follow automatically if --ip changes.
for _name, _s_start, _s_end in SEXT_SPANS:
    _s_mid = 0.5 * (_s_start + _s_end)
    RANGES.append((
        (_s_mid - SEXT_ZOOM_HALF_WIDTH, _s_mid + SEXT_ZOOM_HALF_WIDTH),
        f'zoom on {_name}', 'linear', f'zoom_{_name.replace(".", "_")}'))

FIGURES = [_beta_figure(*spec) for spec in RANGES]


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
        print(f'{generation + " " + tag:14s} {tw.qx:12.6f} {tw.qy:12.6f}')

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
            print(f'    {generation + " " + tag:14s} '
                  f'{tw["betx", name]:12.4f} {tw["bety", name]:12.4f}')

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
        print(f'{generation + " " + tag:14s} '
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
