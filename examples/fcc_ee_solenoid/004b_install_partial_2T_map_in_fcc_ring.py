"""
Install the 2 T BNL detector-solenoid field map in the FCC ring, in place of
the analytic SplineBoris main solenoid of the SB-2T case (004a + 004b).

"Partial": only the part of the map up to 2.399 m from the IP is used, the
same slot as the SB-2T main solenoid line (004a MAIN_SOLENOID_S_AXIS), so the
final-focus quads QD0A/B/C (2.40-5.85 m) and their thin correctors stay as
they are. The map has no quadrupole or dipole-corrector field (checked
2026-10-16), and 98.9% of its int(Bz) lies within 2.4 m. The full 6 m version
will be 004b_install_2T_map_in_fcc_ring.py.

Steps:
  1. The map covers one side of the IP (z >= 0) in the beam frame. The other
     side is B(x, y, -z) = B(-x, -y, z): the solenoid is tilted by the 15 mrad
     half crossing angle, so a plain z -> -z flip would be wrong. The mirrored
     map is fitted as one solenoid (TubeFitter + LongitudinalFitter), which
     keeps the field smooth through the IP.
  2. The fitted on-axis components are tapered to zero over the last 0.15 m
     on each side, with the same smooth step as the SB solenoids (004a).
  3. Compensation solenoids: the SB-2T ones, scaled so that the net int(Bs)
     still cancels (the map's integral is ~13% larger than SB-2T's).
  4. Installation in the four IPs as in 004b (same knobs, the first corrector
     overlaid on the solenoid slices, the correctors at the compensation
     solenoids). 004c --model partialmap then adds the mid-bend trims and
     solves the correction.

The map is read from $FCC_SOLENOID_FIELD_MAP, falling back to the CERNBox
copy, as in examples/splineboris/006b_fcc_ee_solenoid.py.
"""

from pathlib import Path
import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import xtrack as xt
from xtrack._temp.splineboris import TubeFitter, LongitudinalFitter

from aperture_study_io import PLOT_DIR, pdf_metadata
from solenoid_params import (
    COMP_SOLENOID_DISTANCE_FROM_IP,
    COMPENSATION_CORRECTOR_LENGTH,
    COMPENSATION_CORRECTOR_MARKER_DS,
    MAIN_SOLENOID_CORRECTOR_DS_END,
    corrector_ds_start_for_b0,
    field_tag,
)
from spline_boris_setup import smooth_edge_taper, symplectic_error


_parser = argparse.ArgumentParser(
    description='Install the 2 T BNL field map (first 2.4 m from each IP) '
                'in the FCC ring.')
_parser.add_argument(
    '--output-tag', default='',
    help='Optional extra suffix appended to the output lattice filename. '
         '004c_correct_solenoids_in_fcc_ring.py --model partialmap must be '
         'run with the same --output-tag to pick up this file.')
_parser.add_argument(
    '--no-show', action='store_true', help='Do not open the plot windows.')
_args = _parser.parse_args()
OUT_TAG = f'_{_args.output_tag}' if _args.output_tag else ''

B0 = 2.0
FIELD_TAG = field_tag(B0)
MAIN_SOLENOID_CORRECTOR_DS_START = corrector_ds_start_for_b0(B0)

HERE = Path(__file__).parent
INPUT_LATTICE_JSON = HERE / 'fccee_z_lcc.json'
INPUT_SOLENOID_LINES_JSON = HERE / f'004_solenoid_lines_{FIELD_TAG}.json'
OUTPUT_LATTICE_JSON = (
    HERE / f'temp_fcc_ee_lcc_partialmap_solenoids_{FIELD_TAG}{OUT_TAG}.json')

# The BNL map is third-party data and must not be committed (CLAUDE.md).
FIELD_MAP_ENV_VAR = 'FCC_SOLENOID_FIELD_MAP'
DEFAULT_FIELD_MAP = (Path.home() / 'cernbox' / 'Solenoid_Field_Maps'
                     / 'Field_map_2T_BNL_August_14_2026.csv')

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']
SOLENOID_INSERTION_S_TOL = 1e-8

# Same slot and taper as the SB-2T main solenoid (004a MAIN_SOLENOID_S_AXIS,
# TAPER_LENGTH).
MAP_HALF_LENGTH = 2.399  # m
TAPER_LENGTH = 0.15  # m

# Fit settings as in 006b: transverse derivatives up to 4, 3 map planes
# (1 cm apart) per element, components below 1e-4 of the largest field at
# 10 mm dropped.
DEG = 4
POINTS_PER_ELEMENT = 3
FIELD_TOL = 1e-4
R_REF = 0.01

# Boris step length [m]: that of the SB-2T main solenoid (10 steps per
# 2.399 cm element), so tracking costs about the same. The check below
# compares it with 1 mm steps.
STEP_LENGTH = 0.0024
STEP_LENGTH_FINE = 0.001

PARTICLE_REF = xt.Particles('positron', energy0=45.6e9)


#############################
# Load and mirror the map   #
#############################

map_path = Path(os.environ.get(FIELD_MAP_ENV_VAR, DEFAULT_FIELD_MAP)).expanduser()
if not map_path.is_file():
    raise SystemExit(
        f'BNL FCC-ee solenoid field map not found at {map_path}.\n'
        f'Set {FIELD_MAP_ENV_VAR} to the path of '
        f'Field_map_2T_BNL_August_14_2026.csv (default: {DEFAULT_FIELD_MAP}). '
        'The map is not distributed with this repository.')
df_map = (
    pd.read_csv(map_path, usecols=['Xb', 'Yb', 'Zb', 'Bx', 'By', 'Bz'])
    .rename(columns={'Xb': 'X', 'Yb': 'Y', 'Zb': 'Z', 'Bz': 'Bs'}))
# Snap the coordinates to the grid (+ 0.0 turns -0.0 into 0.0), so mirrored
# points land exactly on grid points.
for col in 'XYZ':
    df_map[col] = df_map[col].round(6) + 0.0

# Planes up to 2.40 m: the fit uses those inside +-MAP_HALF_LENGTH, the Boris
# reference interpolates up to the line ends.
df_right = df_map[df_map.Z <= 2.40 + 1e-9]
df_left = df_right[df_right.Z > 0].copy()
df_left[['X', 'Y', 'Z']] *= -1
df_left[['X', 'Y', 'Z']] += 0.0
df_mirrored = (pd.concat([df_left, df_right])
               .set_index(['X', 'Y', 'Z']).sort_index())

# The mirror is only seamless if the z = 0 plane itself satisfies
# B(x, y, 0) = B(-x, -y, 0).
plane0 = df_right[df_right.Z == 0].set_index(['X', 'Y'])[['Bx', 'By', 'Bs']]
plane0_flipped = plane0.copy()
plane0_flipped.index = pd.MultiIndex.from_arrays(
    [-plane0.index.get_level_values(0) + 0.0,
     -plane0.index.get_level_values(1) + 0.0], names=['X', 'Y'])
asym0 = (plane0 - plane0_flipped.reindex(plane0.index)).abs().max()
print('IP plane, max |B(x, y, 0) - B(-x, -y, 0)| [T]:')
for comp in ('Bx', 'By', 'Bs'):
    print(f'  {comp}: {asym0[comp]:.3e}')


##############
# Fit        #
##############

df_fit = df_mirrored[
    np.abs(df_mirrored.index.get_level_values('Z')) <= MAP_HALF_LENGTH]
n_planes = df_fit.index.get_level_values('Z').nunique()
print(f'Fitting {n_planes} planes, z = +-{MAP_HALF_LENGTH} m')

fitter = TubeFitter(raw_data=df_fit, n_frames=n_planes, distance_unit=1,
                    deg=DEG)
fitter.fit()
z_frames, F_frames, names = fitter.on_axis_multipoles()
z_bs, bs_on_axis = fitter.on_axis_bs()


def taper_at(z):
    # smooth_edge_taper puts the ends of the step at the ends of its axis.
    axis = np.r_[-MAP_HALF_LENGTH, z, MAP_HALF_LENGTH]
    return smooth_edge_taper(s_axis=axis, taper_length=TAPER_LENGTH)[1:-1]


lf = LongitudinalFitter(-MAP_HALF_LENGTH, MAP_HALF_LENGTH,
                        points_per_element=POINTS_PER_ELEMENT,
                        end_condition='zero')
lf.fit(z_frames, F_frames * taper_at(z_frames)[:, None], names)
lf.fit(z_bs, bs_on_axis * taper_at(z_bs), [('Bs', 0)])
print(f'{lf.n_elements} elements of {lf.delta * 100:.4f} cm')


def build_map_line(step_length):
    line = lf.to_line(multipole_order=DEG + 1, field_tol=FIELD_TOL,
                      r_ref=R_REF)
    n_steps = math.ceil(lf.delta / step_length - 1e-9)
    for element in line.elements:
        element.n_steps = n_steps
    line.particle_ref = PARTICLE_REF.copy()
    return line, n_steps


line_map, n_steps = build_map_line(STEP_LENGTH)
line_map_fine, n_steps_fine = build_map_line(STEP_LENGTH_FINE)
print(f'Boris steps per element: {n_steps} (fine check: {n_steps_fine})')
print('Exported components:', lf._kept_names(FIELD_TOL, R_REF))

map_bs_integral = sum(el.bs[4] * el.length for el in line_map.elements)
print(f'int Bs: fitted line {map_bs_integral:.6f} T m, map (untapered) '
      f'{np.trapezoid(bs_on_axis, z_bs):.6f} T m')


###################################
# Fit quality: on-axis components #
###################################

# Map gradients by central differences on the 2 mm grid.
H = 0.002
z_grid = np.unique(df_fit.index.get_level_values('Z'))


def map_at(x, y, comp):
    return df_fit.xs((x, y), level=['X', 'Y'])[comp].reindex(z_grid).to_numpy()


map_bx0 = map_at(0.0, 0.0, 'Bx')
map_bs0 = map_at(0.0, 0.0, 'Bs')
map_dbx_dx = (map_at(H, 0.0, 'Bx') - map_at(-H, 0.0, 'Bx')) / (2 * H)
map_dby_dy = (map_at(0.0, H, 'By') - map_at(0.0, -H, 'By')) / (2 * H)
map_dby_dx = (map_at(H, 0.0, 'By') - map_at(-H, 0.0, 'By')) / (2 * H)
# Skew-quadrupole part of the gradient: (dBx/dx - dBy/dy) / 2. The
# solenoidal (radial) part is the average, -Bs'/2.
map_skew = (map_dbx_dx - map_dby_dy) / 2

s_fine = np.linspace(-MAP_HALF_LENGTH, MAP_HALF_LENGTH, 4001)


def fitted(name, der=0):
    if name not in lf.splines:
        return np.zeros_like(s_fine)
    spline = lf.splines[name]
    return (spline.derivative(der) if der else spline)(s_fine)


# div B = 0 on axis: dBy/dy = -dBx/dx - dBs/ds, so the skew part is
# dBx/dx + Bs'/2.
fit_skew = fitted(('Bx', 1)) + fitted(('Bs', 0), der=1) / 2

sb_template = xt.Line.from_dict(
    xt.json.load(INPUT_SOLENOID_LINES_JSON)['lines']['main_solenoid'])
sb_tt = sb_template.get_table().rows[
    sb_template.get_table().element_type == 'SplineBoris']
sb_s_mid = (sb_tt.s_start + sb_tt.s_end) / 2 - sb_template.get_length() / 2
sb_bs_mean = np.array([sb_template[nn].bs[4] for nn in sb_tt.name])

fig_fit, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
axes[0].plot(z_grid, map_bs0, '.', ms=3, color='0.5', label='map')
axes[0].plot(s_fine, fitted(('Bs', 0)), '-', color='C0',
             label='fit (tapered)')
axes[0].plot(sb_s_mid, sb_bs_mean, '-', color='C3', lw=1,
             label='SB-2T (element means)')
axes[0].set_ylabel('$B_s$ [T]')
axes[1].plot(z_grid, map_bx0, '.', ms=3, color='0.5', label='map')
axes[1].plot(s_fine, fitted(('Bx', 0)), '-', color='C0', label='fit')
axes[1].set_ylabel('$B_x$ on axis [T]')
axes[2].plot(z_grid, map_dbx_dx, '.', ms=3, color='0.5', label='map')
axes[2].plot(s_fine, fitted(('Bx', 1)), '-', color='C0', label='fit')
axes[2].set_ylabel(r'$\partial B_x/\partial x$ [T/m]')
axes[3].plot(z_grid, map_skew, '.', ms=3, color='0.5', label='map')
axes[3].plot(s_fine, fit_skew, '-', color='C0', label='fit')
axes[3].plot(z_grid, map_dby_dx, '.', ms=2, color='C2',
             label=r'map $\partial B_y/\partial x$ (normal)')
axes[3].set_ylabel('skew gradient [T/m]')
axes[3].set_xlabel('s from IP [m]')
for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
fig_fit.suptitle('BNL 2 T map, mirrored and fitted (|s| <= 2.399 m)')
fig_fit.tight_layout()
FIT_PLOT_PATH = PLOT_DIR / f'004b_partialmap_fit_{FIELD_TAG}.pdf'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
fig_fit.savefig(FIT_PLOT_PATH, metadata=pdf_metadata())
print(f'Saved {FIT_PLOT_PATH}')


#################################################
# Fit quality: tracking against Boris (raw map) #
#################################################

# Reference: Boris integration through the mirrored raw map, cubic
# interpolation, multiplied by the same taper as the fit.
x_ax, y_ax, z_ax = (np.unique(df_mirrored.index.get_level_values(c))
                    for c in 'XYZ')
grid_shape = (len(x_ax), len(y_ax), len(z_ax))
interps = [RegularGridInterpolator(
    (x_ax, y_ax, z_ax), df_mirrored[c].to_numpy().reshape(grid_shape),
    method='cubic') for c in ('Bx', 'By', 'Bs')]


def get_field(x, y, z):
    pts = np.column_stack(np.broadcast_arrays(x, y, z))
    taper = taper_at(np.atleast_1d(np.broadcast_to(z, pts[:, 0].shape)))
    return tuple(f(pts) * taper for f in interps)


boris = xt.BorisSpatialIntegrator(
    fieldmap_callable=get_field, s_start=-MAP_HALF_LENGTH,
    s_end=MAP_HALF_LENGTH, n_steps=int(round(2 * MAP_HALF_LENGTH / 0.0005)))

delta = np.array([0.0, 1e-2])
p0 = PARTICLE_REF.copy()
p0 = xt.Particles(mass0=p0.mass0, q0=p0.q0, energy0=p0.energy0[0],
                  x=1e-3, px=-1e-4 * (1 + delta), y=1e-3, py=2e-4,
                  delta=delta)
p_boris = p0.copy()
boris.track(p_boris)
print(f'Tracking check, {len(delta)} particles (delta = {delta}), '
      f'from s = -{MAP_HALF_LENGTH} m:')
for label, line in (('SB map line', line_map), ('fine steps', line_map_fine)):
    line.build_tracker()
    p_sb = p0.copy()
    line.track(p_sb)
    for coord in ('x', 'px', 'y', 'py'):
        diff = getattr(p_sb, coord) - getattr(p_boris, coord)
        print(f'  {label:12s} {coord:>2} - Boris = {diff}')

print(f'Symplecticity error (|R^T S R - S|_2): '
      f'{symplectic_error(line=line_map, particle_ref=PARTICLE_REF):.3e}')


#####################################################
# Compensation solenoids: SB-2T, rescaled integral  #
#####################################################

line_data = xt.json.load(INPUT_SOLENOID_LINES_JSON)
comp_solenoid_template = xt.Line.from_dict(
    line_data['lines']['compensation_solenoid'])
comp_bs_integral = sum(
    el.bs[4] * el.length * el.scale_b for el in comp_solenoid_template.elements
    if isinstance(el, xt.SplineBoris))
comp_rescale = map_bs_integral / (-2 * comp_bs_integral)
for el in comp_solenoid_template.elements:
    if isinstance(el, xt.SplineBoris):
        el.scale_b *= comp_rescale
print(f'Compensation solenoids: int Bs per solenoid {comp_bs_integral:.6f} '
      f'-> {comp_bs_integral * comp_rescale:.6f} T m '
      f'(x {comp_rescale:.6f}); net with the map: '
      f'{map_bs_integral + 2 * comp_bs_integral * comp_rescale:.3e} T m')

main_solenoid_template = line_map


#########################################################
# Install independent solenoid clones at IPs (as 004b)  #
#########################################################

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring

env['sext_amp'] = 1.0
# Global field multipliers, as in 004b.
env['comp_b_scale'] = 1.0
env['main_b_scale'] = 1.0

for ip_name in IP_NAMES:

    line.cycle(f'end_ds_start_straight_{ip_name}')
    table_before_insertion = line.get_table()
    s_ip = table_before_insertion['s', ip_name]

    print(f'Installing field-map solenoids and correctors around {ip_name}')

    env[f'on_sol_{ip_name}'] = 0
    env[f'on_comp_sol_{ip_name}'] = 0
    env[f'comp_b_scale_left_{ip_name}'] = 1.0
    env[f'comp_b_scale_right_{ip_name}'] = 1.0

    main_knob_ref = (
        env.ref[f'on_sol_{ip_name}'] * env.ref['main_b_scale'])
    comp_left_knob_ref = (
        env.ref[f'on_comp_sol_{ip_name}'] * env.ref['comp_b_scale']
        * env.ref[f'comp_b_scale_left_{ip_name}'])
    comp_right_knob_ref = (
        env.ref[f'on_comp_sol_{ip_name}'] * env.ref['comp_b_scale']
        * env.ref[f'comp_b_scale_right_{ip_name}'])
    solenoid_lines = {}
    clone_specs = [
        ('main', main_solenoid_template,
         f'sol_slice_{ip_name}', main_knob_ref),
        ('comp_left', comp_solenoid_template,
         f'comp_sol_slice_left_{ip_name}', comp_left_knob_ref),
        ('comp_right', comp_solenoid_template,
         f'comp_sol_slice_right_{ip_name}', comp_right_knob_ref),
    ]

    for clone_name, template_line, element_prefix, knob_ref in clone_specs:
        element_names = []
        name_width = len(str(max(0, len(template_line.element_names) - 1)))

        for ii, template_element in enumerate(template_line.elements):
            element_name = f'{element_prefix}_{ii:0{name_width}d}'
            cloned_element = template_element.copy()

            saved_scale_b = getattr(cloned_element, 'scale_b', 1.0)
            if saved_scale_b is None:
                saved_scale_b = 1.0

            env.elements[element_name] = cloned_element
            env.ref[element_name].scale_b = float(saved_scale_b) * knob_ref

            if (
                isinstance(cloned_element, xt.SplineBoris)
                and cloned_element.multipole_order >= 3
            ):
                sext_ref = env.ref['sext_amp']
                ref = env.ref[element_name]
                for k in range(5):
                    ref.bx[2, k] = float(cloned_element.bx[2, k]) * sext_ref
                    ref.by[2, k] = float(cloned_element.by[2, k]) * sext_ref

            element_names.append(element_name)

        solenoid_lines[clone_name] = env.new_line(components=element_names)

    line.remove(ip_name)
    line.insert([
        env.place(solenoid_lines['main'], anchor='center', at=s_ip),
        env.place(ip_name, at=s_ip),
        env.place(solenoid_lines['comp_left'], anchor='end',
                  at=-COMP_SOLENOID_DISTANCE_FROM_IP, from_=ip_name),
        env.place(solenoid_lines['comp_right'], anchor='start',
                  at=COMP_SOLENOID_DISTANCE_FROM_IP, from_=ip_name),
    ], s_tol=SOLENOID_INSERTION_S_TOL)

    # The first corrector on each side is distributed over the solenoid
    # slices between 1.23 and 2.29 m from the IP, as in 004b.
    env[f'acbh1_sol_right_{ip_name}'] = 0
    env[f'acbv1_sol_right_{ip_name}'] = 0
    env[f'acbh1_sol_left_{ip_name}'] = 0
    env[f'acbv1_sol_left_{ip_name}'] = 0

    table_region = line.get_table().rows[
        f'end_ds_start_straight_{ip_name}':f'end_straight_start_ds_{ip_name}']
    s_ip = table_region['s', ip_name]

    for side, s_lo, s_hi in (
            ('right', s_ip + MAIN_SOLENOID_CORRECTOR_DS_START,
             s_ip + MAIN_SOLENOID_CORRECTOR_DS_END),
            ('left', s_ip - MAIN_SOLENOID_CORRECTOR_DS_END,
             s_ip - MAIN_SOLENOID_CORRECTOR_DS_START)):
        table_corrector = table_region.rows[s_lo:s_hi:'s']
        assert np.all(table_corrector.element_type == 'SplineBoris')
        assert all(
            env_name.startswith(f'sol_slice_{ip_name}_')
            for env_name in table_corrector.env_name)
        length_corrector = (
            table_corrector['s_end'][-1] - table_corrector['s_start'][0])
        for env_name in table_corrector.env_name:
            element = env.get(env_name)
            env.ref[env_name].knl[0] += (
                env.ref[f'acbh1_sol_{side}_{ip_name}']
                / length_corrector * element.length)
            env.ref[env_name].ksl[0] += (
                env.ref[f'acbv1_sol_{side}_{ip_name}']
                / length_corrector * element.length)

    line.insert([
        env.new(
            f'dy_match_r_{ip_name}', xt.Marker,
            at=COMPENSATION_CORRECTOR_MARKER_DS,
            from_=ip_name,
        ),
        env.new(
            f'dy_match_l_{ip_name}', xt.Marker,
            at=-COMPENSATION_CORRECTOR_MARKER_DS,
            from_=ip_name,
        ),
        env.new(
            f'corr_sol_right_{ip_name}', xt.Multipole,
            length=COMPENSATION_CORRECTOR_LENGTH,
            isthick=False,
            anchor='end',
            at=0,
            from_=f'dy_match_r_{ip_name}@start',
        ),
        env.new(
            f'corr_sol_left_{ip_name}', xt.Multipole,
            length=COMPENSATION_CORRECTOR_LENGTH,
            isthick=False,
            anchor='start',
            at=0,
            from_=f'dy_match_l_{ip_name}@end',
        ),
    ])


########################
# Save installed lattice #
########################

env.to_json(OUTPUT_LATTICE_JSON)
print(f'Wrote {OUTPUT_LATTICE_JSON}')


###############################################
# Check: net int(Bs) in the ring, solenoids on #
###############################################

for ip_name in IP_NAMES:
    env[f'on_sol_{ip_name}'] = 1.0
    env[f'on_comp_sol_{ip_name}'] = 1.0
bs_main = bs_comp = 0.0
for name in line.element_names:
    if name.startswith(('sol_slice_', 'comp_sol_slice_')):
        el = env.get(name)
        integral = el.bs[4] * el.length * el.scale_b
        if name.startswith('sol_slice_'):
            bs_main += integral
        else:
            bs_comp += integral
print(f'Ring int(Bs): main {bs_main:.9g}, compensation {bs_comp:.9g}, '
      f'total {bs_main + bs_comp:.3e} T m')

if not _args.no_show:
    plt.show()
