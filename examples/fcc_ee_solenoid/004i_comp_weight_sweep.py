"""Weight scan for the per-side compensation-solenoid knobs in the coupling
re-solve used by 004h_main_b_scale_scan.py.

At a fixed main-solenoid detuning (--main-b-scale, default 1.03) and one IP
(--ip-name, default ipa), re-solve the coupling correction -- the 84
k1s_*_sol_coupling_corr skew quads together with the two per-side
comp_b_scale_{left,right}_{ip} compensation-field knobs -- for a range of
weights applied to the comp knobs, and plot every skew corrector's
integrated strength against its longitudinal position, expressed as a
fraction of the arc-cell quadrupole strength <|k1*L|>_arc.

Motivation: with unit weight the least-squares solver ignores the comp
knobs entirely (a 0.01%-scale compensation-field change is a ~3-4 orders
weaker linear-coupling handle than a k1s skew quad), so they stay pinned at
1.0. Only a large weight (>~1e4) makes the solver actually spend them.
xdeps maps x = knob / weight, so the solver's Jacobian column for a knob is
(physical column) * weight -- weight ~1e4 lifts the comp knobs to roughly
the median skew-quad column. This script shows (a) whether cranking that
weight redistributes the skew correctors along s, and (b) the left/right
asymmetry and saturation of the fitted comp-field response.

Note: 004g_coupling_svd_diagnostic.py already occupies the 004g slot (SVD
decomposition of this same coupling Jacobian); this is 004i.

Run:
    python 004i_comp_weight_sweep.py --ip-name ipa --main-b-scale 1.03 --no-show
    python 004i_comp_weight_sweep.py --weights 0 1e3 1e4 1e5 --no-show
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
    COMP_SOLENOID_DISTANCE_FROM_IP,
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
        'Sweep the weight on the per-side compensation-solenoid knobs in '
        "004h's coupling re-solve, and plot the skew correctors' strength "
        '(relative to the arc-cell quad strength) vs s.'))
add_b0_argument(_parser, default=MAIN_SOLENOID_B0)
add_max_order_argument(_parser)
_parser.add_argument(
    '--input-tag', default='mainscale',
    help='--output-tag that 004b/004c were run with to produce the lattice '
         'to load (default: "mainscale"). Pass "" for the standard untagged '
         'lattice (needs the main_b_scale / comp_b_scale_{side}_{ip} knobs).')
_parser.add_argument(
    '--ip-name', default='ipa',
    help='IP whose coupling correction is re-solved (default: ipa).')
_parser.add_argument(
    '--main-b-scale', type=float, default=1.03,
    help='Fixed main-solenoid field multiplier for the sweep (default: '
         '1.03 -- a 3%% detuning, large enough that the correction is '
         'clearly working).')
_parser.add_argument(
    '--weights', type=float, nargs='+',
    default=[0.0, 1e2, 1e3, 1e4, 3e4, 1e5],
    help='Comp-knob weights to sweep. 0 means "comp knobs not in the vary '
         'set at all" (skew quads only, the 004c/004f behaviour). '
         'Default: 0 1e2 1e3 1e4 3e4 1e5.')
_parser.add_argument(
    '--no-orbit', action='store_true',
    help='Skip the per-IP orbit re-solve before the coupling sweep (leave '
         'the acbh/acbv_sol_* correctors at their nominal main_b_scale=1.0 '
         'values). By default the orbit is re-solved once (weight-'
         'independent) so the coupling sweep sees a corrected orbit.')
_parser.add_argument(
    '--no-show', action='store_true',
    help='Save the figures without opening an interactive window.')
_args = _parser.parse_args()

FIELD_TAG = field_tag(_args.b0)
ORDER_TAG = order_tag(_args.max_transverse_order)
INPUT_TAG = f'_{_args.input_tag}' if _args.input_tag else ''
IP_NAME = _args.ip_name
MAIN_B_SCALE = _args.main_b_scale
WEIGHTS = list(_args.weights)

HERE = Path(__file__).parent
INPUT_LATTICE_JSON = (
    HERE / (
        'fccee_z_lcc_splineboris_solenoids_coupling_corrected_'
        f'{FIELD_TAG}{ORDER_TAG}{INPUT_TAG}.json'
    )
)


################################
# Load and prepare the lattice #
################################

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring.copy(shallow=True)
line.particle_ref.anomalous_magnetic_moment = 0.00115965218128

for _knob in ('main_b_scale',
              f'comp_b_scale_left_{IP_NAME}', f'comp_b_scale_right_{IP_NAME}'):
    if _knob not in line.vars:
        raise SystemExit(
            f'{INPUT_LATTICE_JSON.name} has no {_knob!r} knob -- rebuild the '
            'lattice with `python 004b_install_solenoids_in_fcc_ring.py '
            '--output-tag mainscale` then `python '
            '004c_correct_solenoids_in_fcc_ring.py --output-tag mainscale` '
            '(same --b0/--max-transverse-order as here).'
        )

line[f'on_sol_{IP_NAME}'] = 1
line[f'on_sol_corr_{IP_NAME}'] = 1
set_lattice_knobs(
    line, with_solenoids=True, with_correctors=True, main_b_scale=MAIN_B_SCALE)

# Cycle so table.rows[start:ip] / [ip:end] below don't wrap the ring start.
line.cycle(f'end_ds_start_straight_{IP_NAME}')


########################################################
# Vary-knob discovery (same as 004h / 004g).            #
########################################################

def _straight_section_boundary_names(ip_name):
    return (f'end_ds_start_straight_{ip_name}',
            f'end_straight_start_ds_{ip_name}')


NAME_START, NAME_END = _straight_section_boundary_names(IP_NAME)

_table = line.get_table()
_quad_hosts = []
for _tp in (_table.rows[NAME_START:IP_NAME], _table.rows[IP_NAME:NAME_END]):
    for _et, _en in zip(_tp.element_type, _tp.env_name):
        if _et == 'Quadrupole' and _en not in _quad_hosts:
            _quad_hosts.append(_en)

K1S_KNOBS = [f'k1s_{nn}_sol_coupling_corr' for nn in _quad_hosts]
_missing = [nn for nn in K1S_KNOBS if nn not in line.vars]
if _missing:
    raise SystemExit(
        f'{INPUT_LATTICE_JSON.name} is missing coupling-correction knob(s), '
        f'e.g. {_missing[0]!r} -- it must be a 004c-produced lattice.')

COMP_KNOBS = [f'comp_b_scale_left_{IP_NAME}', f'comp_b_scale_right_{IP_NAME}']

_ORBIT_SUFFIXES = (
    'acbh1', 'acbv1', 'acbh2', 'acbh3', 'acbh4', 'acbh5', 'acbh6',
    'acbv2', 'acbv3', 'acbv4', 'acbv5', 'acbv6',
)
ORBIT_KNOBS = [
    f'{suffix}_sol_{side}_{IP_NAME}'
    for side in ('right', 'left') for suffix in _ORBIT_SUFFIXES
]

print(f'{len(K1S_KNOBS)} skew-quad knobs, {len(COMP_KNOBS)} per-side comp '
      f'knobs, {len(ORBIT_KNOBS)} orbit-corrector knobs for {IP_NAME}')


########################################################
# Host-quad geometry + arc-cell reference strength.     #
########################################################

_tw0 = line.twiss4d(strengths=True)
_names = np.asarray([str(n) for n in _tw0['name']])
_k1l = np.asarray(_tw0['k1l'])
_L = np.asarray(_tw0['length'])
_s = np.asarray(_tw0['s'])
_et = np.asarray(_tw0['element_type'])

_arc = (_et == 'Quadrupole') & np.array(
    [n.startswith(('qf2a.', 'qd1a.')) for n in _names])
if _arc.sum() < 50:
    K1L_REF = float(np.median(np.abs(_k1l[(_et == 'Quadrupole')])))
    K1L_REF_LABEL = 'all-quad median'
else:
    K1L_REF = float(np.median(np.abs(_k1l[_arc])))
    K1L_REF_LABEL = f'arc-cell (qf2a/qd1a) median, n={int(_arc.sum())}'

_k1l_by_name = dict(zip(_names, _k1l))
_L_by_name = dict(zip(_names, _L))
_s_by_name = dict(zip(_names, _s))
S_IP = float(_tw0['s', IP_NAME])

HOST_S = np.array([_s_by_name[h] - S_IP for h in _quad_hosts])
HOST_L = np.array([_L_by_name[h] for h in _quad_hosts])
HOST_K1L = np.array([_k1l_by_name[h] for h in _quad_hosts])

print(f'Arc-cell |k1*L| reference = {K1L_REF:.4e} 1/m  ({K1L_REF_LABEL})')


########################################################
# Freeze skew knobs to constants, snapshot nominal.     #
########################################################
# After 004c the k1s_*_sol_coupling_corr vars are expressions gated by
# on_sol_coupling_corr_{ip}; line.match(vary=[...]) would replace them with
# constants on the first solve anyway. Freeze them up front so every weight
# in the sweep restarts from the same nominal skew solution (cold, not
# warm-started across weights -- we want an apples-to-apples comparison).

for _k in K1S_KNOBS:
    line[_k] = float(line[_k])
K1S_NOMINAL = {k: float(line[k]) for k in K1S_KNOBS}


def _restore_nominal():
    for k, v in K1S_NOMINAL.items():
        line[k] = v
    for c in COMP_KNOBS:
        line[c] = 1.0


########################################################
# Optional one-off orbit re-solve (weight-independent). #
########################################################

def _resolve_orbit():
    tw_local = line.twiss4d(strengths=True)
    opt = line.match(
        solve=False,
        betx=tw_local['betx', IP_NAME], bety=tw_local['bety', IP_NAME],
        init_at=IP_NAME,
        start=f'dy_match_l_{IP_NAME}', end=f'dy_match_r_{IP_NAME}',
        vary=xt.VaryList(ORBIT_KNOBS, step=1e-6),
        targets=[
            xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.END),
            xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.START),
        ])
    opt.solve()
    if not all(opt.target_status(ret=True).tol_met):
        print('  WARNING: orbit re-solve did not fully converge; using best '
              'point found.')


if not _args.no_orbit:
    print(f'Re-solving orbit correction for {IP_NAME} at '
          f'main_b_scale={MAIN_B_SCALE:g} ...')
    _resolve_orbit()


########################################################
# The weight sweep.                                     #
########################################################

def _resolve_coupling(weight):
    """Re-solve the coupling correction. weight == 0 -> comp knobs are not
    in the vary set at all (skew quads only)."""
    tw_local = line.twiss4d(strengths=True)
    vary = [xt.VaryList(K1S_KNOBS, step=1e-6)]
    if weight != 0.0:
        vary.append(xt.VaryList(
            COMP_KNOBS, step=1e-4, limits=(0.5, 1.5), weight=weight))
    opt = line.match(
        solve=False,
        betx=tw_local['betx', IP_NAME], bety=tw_local['bety', IP_NAME],
        init_at=IP_NAME, start=NAME_START, end=NAME_END,
        n_steps_max=100, assert_within_tol=False,
        vary=vary,
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
    opt.solve(rcond=0, broyden=True)
    status = opt.target_status(ret=True)
    return int(np.sum(status.tol_met)), int(len(status.tol_met))


RESULTS = []          # one dict per weight
for weight in WEIGHTS:
    _restore_nominal()
    n_met, n_tot = _resolve_coupling(weight)
    k1s_vals = np.array([float(line[k]) for k in K1S_KNOBS])
    comp_left = float(line[COMP_KNOBS[0]])
    comp_right = float(line[COMP_KNOBS[1]])
    tw_after = line.twiss4d(strengths=True)
    RESULTS.append(dict(
        weight=weight,
        k1s=k1s_vals,
        k1sL_over_ref=k1s_vals * HOST_L / K1L_REF,
        comp_left=comp_left,
        comp_right=comp_right,
        c_minus=float(tw_after.c_minus),
        n_met=n_met, n_tot=n_tot,
    ))
    print(f'  weight={weight:>8g}: comp L={comp_left:.6f} R={comp_right:.6f}  '
          f'|k1sL/ref| rms={np.sqrt(np.mean((k1s_vals*HOST_L/K1L_REF)**2)):.2e} '
          f'max={np.abs(k1s_vals*HOST_L/K1L_REF).max():.2e}  '
          f'C-={tw_after.c_minus:.3e}  targets {n_met}/{n_tot}')


########################################################
# Print summary table.                                 #
########################################################

print('\nSummary (IP={}, main_b_scale={:g})'.format(IP_NAME, MAIN_B_SCALE))
print(f'{"weight":>10}  {"comp_left":>10}  {"comp_right":>10}  '
      f'{"rms|k1sL/ref|":>13}  {"max|k1sL/ref|":>13}  {"C_-":>11}  targets')
for r in RESULTS:
    rms = np.sqrt(np.mean(r['k1sL_over_ref'] ** 2))
    mx = np.abs(r['k1sL_over_ref']).max()
    wlabel = 'pinned' if r['weight'] == 0.0 else f'{r["weight"]:g}'
    print(f'{wlabel:>10}  {r["comp_left"]:>10.6f}  {r["comp_right"]:>10.6f}  '
          f'{rms:>13.3e}  {mx:>13.3e}  {r["c_minus"]:>11.3e}  '
          f'{r["n_met"]}/{r["n_tot"]}')


#########
# Plots #
#########

plt.close('all')

_num_weights = [w for w in WEIGHTS if w != 0.0]
if _num_weights:
    _wnorm = mcolors.LogNorm(vmin=min(_num_weights), vmax=max(_num_weights))
    _wcmap = cm.viridis
else:
    _wnorm = _wcmap = None


def _weight_color(weight):
    if weight == 0.0:
        return '0.15'
    return _wcmap(_wnorm(weight))


def _weight_label(weight):
    return 'comp pinned' if weight == 0.0 else f'weight = {weight:g}'


COMP_SPANS = [
    (-COMP_SOLENOID_DISTANCE_FROM_IP - COMP_SOLENOID_LENGTH / 2,
     -COMP_SOLENOID_DISTANCE_FROM_IP + COMP_SOLENOID_LENGTH / 2),
    (COMP_SOLENOID_DISTANCE_FROM_IP - COMP_SOLENOID_LENGTH / 2,
     COMP_SOLENOID_DISTANCE_FROM_IP + COMP_SOLENOID_LENGTH / 2),
]
# Main detector solenoid: physical full length = 2 * half_length_for_b0(b0)
# (2.6 m at 3 T), centred on the IP. Same red shading convention as
# 004f/004h's _mark_solenoid_regions.
_MAIN_HALF_L = half_length_for_b0(_args.b0)
MAIN_SPAN = (-_MAIN_HALF_L, _MAIN_HALF_L)


def _shade_solenoids(ax, xlim):
    if MAIN_SPAN[1] > xlim[0] and MAIN_SPAN[0] < xlim[1]:
        ax.axvspan(*MAIN_SPAN, color='red', alpha=0.15, linewidth=0, zorder=0)
    for span in COMP_SPANS:
        if span[1] > xlim[0] and span[0] < xlim[1]:
            ax.axvspan(*span, color='orange', alpha=0.18, linewidth=0,
                       zorder=0)


def _plot_correctors_vs_s(ax, xlim):
    _shade_solenoids(ax, xlim)
    ax.axhline(0.0, color='0.5', linewidth=0.8, zorder=1)
    order = np.argsort([0 if w == 0.0 else 1 for w in WEIGHTS])
    for idx in order:
        r = RESULTS[idx]
        color = _weight_color(r['weight'])
        inwin = (HOST_S >= xlim[0]) & (HOST_S <= xlim[1])
        ax.plot(HOST_S[inwin], r['k1sL_over_ref'][inwin],
                marker='o', markersize=3.5, linewidth=0.9,
                color=color, label=_weight_label(r['weight']),
                zorder=3 if r['weight'] == 0.0 else 2)
    ax.set_ylabel(r'$k_{1s}L \,/\, \langle |k_1 L|\rangle_{\mathrm{arc}}$')
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.3)


_host_span = (HOST_S.min(), HOST_S.max())
_ir_win = (-60.0, 60.0)

fig1, axs1 = plt.subplots(2, 1, figsize=(9.5, 7.5))
_plot_correctors_vs_s(axs1[0], _ir_win)
axs1[0].set_title(
    f'{IP_NAME}: skew coupling-corrector strength vs s, comp-knob weight '
    f'sweep\n(main solenoid {_args.b0:g} T, main_b_scale = {MAIN_B_SCALE:g}; '
    f'strength relative to arc-cell $\\langle|k_1 L|\\rangle$ = '
    f'{K1L_REF:.3e} 1/m)\n'
    'shaded: main solenoid (red), compensation solenoids (orange)  --  '
    'interaction region')
axs1[0].legend(fontsize=8, ncol=2, loc='best')
_plot_correctors_vs_s(axs1[1], _host_span)
axs1[1].set_title('full straight section')
axs1[1].set_xlabel(r'$s - s_{\mathrm{IP}}$ [m]')
fig1.tight_layout()

# Difference vs the pinned case (makes the tiny redistribution visible).
_pinned = next((r for r in RESULTS if r['weight'] == 0.0), None)
fig2 = None
if _pinned is not None and len(RESULTS) > 1:
    fig2, axs2 = plt.subplots(2, 1, figsize=(9.5, 7.5))
    for ax, win, ttl in zip(
            axs2, (_ir_win, _host_span),
            ('interaction region', 'full straight section')):
        _shade_solenoids(ax, win)
        ax.axhline(0.0, color='0.5', linewidth=0.8)
        for r in RESULTS:
            if r['weight'] == 0.0:
                continue
            d = r['k1sL_over_ref'] - _pinned['k1sL_over_ref']
            inwin = (HOST_S >= win[0]) & (HOST_S <= win[1])
            ax.plot(HOST_S[inwin], d[inwin], marker='o', markersize=3.5,
                    linewidth=0.9, color=_weight_color(r['weight']),
                    label=_weight_label(r['weight']))
        ax.set_ylabel(r'$\Delta(k_{1s}L)/\langle|k_1L|\rangle_{\mathrm{arc}}$'
                      '\n(vs comp pinned)')
        ax.set_xlim(*win)
        ax.grid(True, alpha=0.3)
        ax.set_title(ttl)
    axs2[0].set_title(
        f'{IP_NAME}: change in skew-corrector strength vs the comp-pinned '
        f'solution\n(main_b_scale = {MAIN_B_SCALE:g}; shaded: main solenoid '
        '(red), compensation solenoids (orange))  --  interaction region')
    axs2[0].legend(fontsize=8, ncol=2, loc='best')
    axs2[1].set_xlabel(r'$s - s_{\mathrm{IP}}$ [m]')
    fig2.tight_layout()

# Fitted comp-field response vs weight.
fig3, ax3 = plt.subplots(figsize=(7.0, 4.8))
_ws = np.array([r['weight'] for r in RESULTS])
_cl = np.array([r['comp_left'] for r in RESULTS])
_cr = np.array([r['comp_right'] for r in RESULTS])
_pos = _ws.copy()
_pos[_pos == 0.0] = (min(_num_weights) / 10.0) if _num_weights else 1.0
ax3.plot(_pos, _cl, '-o', color='C0', label='comp_b_scale_left')
ax3.plot(_pos, _cr, '-s', color='C1', label='comp_b_scale_right')
ax3.axhline(1.0, color='0.6', linewidth=0.8)
ax3.set_xscale('log')
ax3.set_xlabel('comp-knob weight  (leftmost point = comp pinned / not varied)')
ax3.set_ylabel('fitted compensation-field scale')
ax3.set_title(
    f'{IP_NAME}: fitted per-side compensation-field scale vs comp-knob '
    f'weight\n(main solenoid {_args.b0:g} T, main_b_scale = {MAIN_B_SCALE:g})')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best')
fig3.tight_layout()


#########
# Save  #
#########

PLOT_SUBDIR = _BASE_PLOT_DIR / 'Coupling_Studies' / 'comp_weight_sweep'
PLOT_SUBDIR.mkdir(parents=True, exist_ok=True)

_mbs_tag = f'{round(MAIN_B_SCALE * 1000)}'
_stem = f'{IP_NAME}_{FIELD_TAG}_mbs{_mbs_tag}'


def _save(fig, name):
    if fig is None:
        return
    path = PLOT_SUBDIR / f'{name}.pdf'
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved plot: {path}')


_save(fig1, f'skew_corrector_strength_vs_s__comp_weight_sweep__{_stem}')
_save(fig2, f'skew_corrector_delta_vs_s__comp_weight_sweep__{_stem}')
_save(fig3, f'fitted_comp_field_vs_weight__{_stem}')

print(f'\nLoaded {INPUT_LATTICE_JSON}')
if not _args.no_show:
    plt.show()
