"""Diagnostic: SVD mode decomposition of the coupling-correction matching
Jacobian used by 004f_comp_b_scale_scan.py's _resolve_coupling_correction.

Motivation: that re-solve varies ~84 k1s_*_sol_coupling_corr skew-quad knobs
against only ~12 coupling targets (betx2/bety1/alfx2/alfy1/dy/dpy at the two
straight-section boundaries), which is heavily underdetermined and gives a
poorly-conditioned Jacobian (condition number ~1e5). 004f already guards
against this with `rcond=3e-3` and `broyden=True` on the actual solve (see
the comments there), but neither of those explains *why* the Jacobian looks
the way it does. This script answers that by building the exact same
Optimize object for one (ip_name, comp_b_scale) point, evaluating its
Jacobian once (no solve -- purely diagnostic), and decomposing it via SVD:

- The singular values themselves say which of the ~12 independent target
  directions ("modes") are well-determined (large singular value) vs.
  noise-dominated (small) -- the basis for choosing an SVD truncation
  (rcond/sing_val_cutoff) for the real solve.
- Each mode's right-singular-vector (a combination of all ~84 vary knobs)
  says which few knobs actually carry that mode, and its left-singular-
  vector (a combination of the ~12 targets) says which target it controls.
  A singular value is a property of the whole matrix, not of one magnet --
  but this decomposition recovers, per mode, which handful of magnets
  dominate it.

Run standalone (loads its own lattice/knobs; does not import 004f):
    python 004g_coupling_svd_diagnostic.py [--ip-name ipa]
        [--comp-b-scale 0.998] [--top-n 6] [--no-show]
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from lattice_knobs import set_lattice_knobs
from solenoid_params import (
    MAIN_SOLENOID_B0,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    order_tag,
)


_parser = argparse.ArgumentParser(
    description=(
        'Decompose the coupling-correction matching Jacobian (see '
        '004f_comp_b_scale_scan.py) via SVD for one IP/comp_b_scale point, '
        'and show which vary knobs and targets dominate each mode.'))
add_b0_argument(_parser, default=MAIN_SOLENOID_B0)
add_max_order_argument(_parser)
_parser.add_argument(
    '--ip-name', default='ipa',
    help='IP to inspect (default: ipa).')
_parser.add_argument(
    '--comp-b-scale', type=float, default=0.998,
    help='comp_b_scale value to build the Jacobian at (default: 0.998 -- '
         'mildly off-nominal, so the correction knobs are not all sitting '
         'at a trivial value).')
_parser.add_argument(
    '--top-n', type=int, default=6,
    help='Number of dominant knobs to print/highlight per mode '
         '(default: 6).')
_parser.add_argument(
    '--no-show', action='store_true',
    help='Save figures to PNG in this directory instead of opening an '
         'interactive window.')
_args = _parser.parse_args()
FIELD_TAG = field_tag(_args.b0)
ORDER_TAG = order_tag(_args.max_transverse_order)
IP_NAME = _args.ip_name

HERE = Path(__file__).parent
INPUT_LATTICE_JSON = (
    HERE / (
        'fccee_z_lcc_splineboris_solenoids_coupling_corrected_'
        f'{FIELD_TAG}{ORDER_TAG}.json'
    )
)


################################
# Load and prepare the lattice #
################################
# Same loading/knob setup as 004f_comp_b_scale_scan.py, but only touching the
# one IP being inspected.

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring.copy(shallow=True)
line.particle_ref.anomalous_magnetic_moment = 0.00115965218128

if 'comp_b_scale' not in line.vars:
    raise SystemExit(
        f'{INPUT_LATTICE_JSON.name} has no comp_b_scale knob -- see '
        '004f_comp_b_scale_scan.py for the rebuild instructions.'
    )

line[f'on_sol_{IP_NAME}'] = 1
line[f'on_sol_corr_{IP_NAME}'] = 1
set_lattice_knobs(
    line, with_solenoids=True, with_correctors=True,
    comp_b_scale=_args.comp_b_scale)

# Required so table.rows[name_start:ip_name]/[ip_name:name_end] below don't
# wrap around the ring's default start point -- same reason
# 004f_comp_b_scale_scan.py cycles before building table_before_cuts.
line.cycle(f'end_ds_start_straight_{IP_NAME}')


###############################################################
# Same k1s_*_sol_coupling_corr vary-knob discovery as 004f's   #
# _k1s_coupling_knobs_for_ip / _straight_section_boundary_names. #
###############################################################

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
            f'{INPUT_LATTICE_JSON.name} is missing coupling-correction '
            f'knob(s), e.g. {missing[0]!r} -- it must be a lattice produced '
            'by 004c_correct_solenoids_in_fcc_ring.py.'
        )
    return knob_names


table = line.get_table()
K1S_KNOBS = _k1s_coupling_knobs_for_ip(table, IP_NAME)
print(f'{len(K1S_KNOBS)} skew-quad vary knobs for {IP_NAME}')


######################################################################
# Build the same opt_coupling Optimize object as 004f's               #
# _resolve_coupling_correction (identical targets/vary), but only     #
# evaluate its Jacobian once (run_jacobian(1)) -- no solve, this is   #
# purely diagnostic.                                                  #
######################################################################

name_start, name_end = _straight_section_boundary_names(IP_NAME)
tw_local = line.twiss4d(strengths=True)
opt_coupling = line.match(
    solve=False,
    betx=tw_local['betx', IP_NAME],
    bety=tw_local['bety', IP_NAME],
    init_at=IP_NAME,
    start=name_start,
    end=name_end,
    n_steps_max=100,
    assert_within_tol=False,
    vary=xt.VaryList(K1S_KNOBS, step=1e-6),
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
opt_coupling.run_jacobian(1)
svd = opt_coupling.solver._last_jac_svd

n_modes = len(svd.s)
target_descs = [str(t) for t in opt_coupling.targets]
print(f'{n_modes} singular values (targets), n_vary={len(K1S_KNOBS)}, '
      f'condition number = {svd.s[0] / svd.s[-1]:.3e}')


#########################################################
# Decompose each mode: dominant knobs + dominant target #
#########################################################

def _short_target_label(desc):
    """Trim a Target's __repr__, e.g.
    "Target(('betx2', 'end_straight_start_ds_ipa'), val=0, ...)", down to
    'betx2 (R)' -- R/L here means the end_straight_start_ds_{ip}/
    end_ds_start_straight_{ip} boundary (right/left of the IP, matching
    004f's CORRECTOR_QUADS_BY_IP right/left convention)."""
    inner = desc.split('(', 2)[2].split(')', 1)[0]
    quantity, at_name = [s.strip(" '") for s in inner.split(',', 1)]
    side = 'R' if at_name.startswith('end_straight_start') else 'L'
    return f'{quantity} ({side})'


mode_rows = []
for i in range(n_modes):
    vh_row = svd.Vh[i, :]
    knob_order = np.argsort(-np.abs(vh_row))[:_args.top_n]
    u_col = svd.U[:, i]
    dominant_target_idx = int(np.argmax(np.abs(u_col)))
    mode_rows.append(dict(
        index=i,
        sigma=svd.s[i],
        target_label=_short_target_label(target_descs[dominant_target_idx]),
        target_alignment=u_col[dominant_target_idx],
        knob_names=[K1S_KNOBS[j] for j in knob_order],
        knob_weights=vh_row[knob_order],
    ))

print()
print(f'{"mode":>4} {"sigma":>12} {"target":>14} {"align":>7}   '
      f'top-{_args.top_n} knobs (Vh weight)')
print('-' * 100)
for row in mode_rows:
    knob_str = ', '.join(
        f'{nn}={ww:+.2f}'
        for nn, ww in zip(row['knob_names'], row['knob_weights']))
    print(f'{row["index"]:>4} {row["sigma"]:>12.4e} {row["target_label"]:>14} '
          f'{row["target_alignment"]:>+7.3f}   {knob_str}')

# Suggest a sing_val_cutoff from the largest ratio (log-)gap between
# consecutive singular values -- see the discussion this script grew out of.
_gaps = svd.s[:-1] / svd.s[1:]
_cutoff_suggestion = int(np.argmax(_gaps)) + 1
print(f'\nLargest gap in the spectrum is after mode '
      f'{_cutoff_suggestion - 1} (ratio {_gaps[_cutoff_suggestion - 1]:.1f}x) '
      f'-- a candidate sing_val_cutoff={_cutoff_suggestion}.')


##########
# Plots  #
##########

plt.close('all')

_target_families = sorted(
    {row['target_label'].split(' ')[0] for row in mode_rows})
_cmap = plt.get_cmap('tab10')
_family_color = {fam: _cmap(i % 10) for i, fam in enumerate(_target_families)}
_colors = [_family_color[row['target_label'].split(' ')[0]]
           for row in mode_rows]

fig1, ax1 = plt.subplots(figsize=(7.5, 4.8))
ax1.bar(range(n_modes), [row['sigma'] for row in mode_rows], color=_colors)
ax1.set_yscale('log')
ax1.set_xticks(range(n_modes))
ax1.set_xticklabels(
    [row['target_label'] for row in mode_rows], rotation=45, ha='right')
ax1.set_ylabel(r'singular value $\sigma$')
ax1.set_title(
    f'{IP_NAME} coupling-correction Jacobian -- singular value spectrum\n'
    f'(comp_b_scale={_args.comp_b_scale:g}, {len(K1S_KNOBS)} vary knobs, '
    f'{n_modes} targets, cond={svd.s[0] / svd.s[-1]:.2e})')
_handles = [plt.Rectangle((0, 0), 1, 1, color=_family_color[f])
            for f in _target_families]
ax1.legend(_handles, _target_families, title='dominant target', fontsize=8)
fig1.tight_layout()

# Heatmap of |Vh|: rows = modes (loud -> quiet, SVD order), columns = knobs,
# sorted by which mode they dominate -- clusters knobs that share a mode
# together, visualizing the near-block-diagonal, per-family-redundant
# structure described in the printed table above.
abs_vh = np.abs(svd.Vh)  # (n_modes, n_vary)
knob_dominant_mode = np.argmax(abs_vh, axis=0)
knob_order_for_heatmap = np.lexsort(
    (-abs_vh.max(axis=0), knob_dominant_mode))

fig2, ax2 = plt.subplots(figsize=(12.0, 4.8))
im = ax2.imshow(
    abs_vh[:, knob_order_for_heatmap], aspect='auto', cmap='viridis', vmin=0)
ax2.set_yticks(range(n_modes))
ax2.set_yticklabels([row['target_label'] for row in mode_rows])
ax2.set_xlabel(
    f'{len(K1S_KNOBS)} skew-quad vary knobs (k1s_*_sol_coupling_corr), '
    'sorted by dominant mode')
ax2.set_xticks([])
ax2.set_title(
    f'{IP_NAME} coupling-correction Jacobian -- |right-singular-vector| '
    'knob composition per mode')
fig2.colorbar(
    im, ax=ax2, label='|Vh| (knob weight in mode)', fraction=0.03, pad=0.02)
fig2.tight_layout()

if _args.no_show:
    spectrum_png = HERE / f'004g_svd_spectrum_{IP_NAME}.png'
    heatmap_png = HERE / f'004g_svd_knob_heatmap_{IP_NAME}.png'
    fig1.savefig(spectrum_png, dpi=150)
    fig2.savefig(heatmap_png, dpi=150)
    print(f'Saved {spectrum_png.name}, {heatmap_png.name}')
else:
    plt.show()
