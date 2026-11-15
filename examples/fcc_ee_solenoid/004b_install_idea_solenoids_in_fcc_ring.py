"""
Install the IDEA detector solenoid (004a_build_idea_solenoid.py: main +
screening total as BFieldExpansion elements, compensation solenoids) in the
FCC ring, as 004b does for the analytic solenoids, with the first corrector
split into 010a-style segments. Exploratory, not in the paper's pipeline
(claude_notes/18).

- The main line is centred on each IP: |s| <= z_m (2.30 m for edge 1.40 m),
  inside the 2.40 m drift to QD0; the IP marker goes on its central element
  boundary. Compensation solenoids at +-COMP_SOLENOID_DISTANCE_FROM_IP.
- Knobs as 004b: on_sol_{ip} * main_b_scale on the kscale of the main
  elements, on_comp_sol_{ip} * comp_b_scale * comp_b_scale_{side}_{ip} on
  the compensation ones, sext_amp on the sextupole row (ksc[2]).
- First corrector as 010a: every segment in the template's metadata
  (first_corr_segments, s ranges from the IP and weights) gets the knobs
  acb{h,v}1_seg{n}_sol_{side}_{ip} = weight x acb{h,v}1_sol_{side}_{ip},
  spread as knl[0]/ksl[0] over the elements inside the segment in
  proportion to their length. 010a has only the vertical corrector; the
  horizontal one sits on the same segments because 004c varies both.
  BFieldExpansion multiplies knl/ksl by kscale, so the corrector strengths
  are only right at on_sol = main_b_scale = 1 (bfe_setup.check_bfe_nominal_
  field), as for --model bfe.
- dy_match markers at +-COMPENSATION_CORRECTOR_MARKER_DS and the thin
  corr_sol correctors, as 004b.

Then: 004c_correct_solenoids_in_fcc_ring.py --model idea_bfe --b0 <b0>
--output-tag <idea_case_tag>, e.g. --b0 2.5 --output-tag e1p40_eps0p75.
"""
from pathlib import Path
import argparse

import numpy as np
import xtrack as xt

from bfe_setup import (
    coefficient_attrs,
    get_scale,
    is_field_element,
    ksol_integral,
    scale_attr,
    transverse_order,
)
from solenoid_params import (
    COMP_SOLENOID_DISTANCE_FROM_IP,
    COMPENSATION_CORRECTOR_LENGTH,
    COMPENSATION_CORRECTOR_MARKER_DS,
    IDEA_DEFAULT_TAPER_LENGTH,
    field_tag,
    idea_case_tag,
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument('--b0', type=float, default=2.5)
parser.add_argument('--edge', type=float, default=1.40)
parser.add_argument('--eps', type=float, default=0.75)
parser.add_argument('--taper-length', type=float,
                    default=IDEA_DEFAULT_TAPER_LENGTH)
args = parser.parse_args()

CASE_TAG = idea_case_tag(args.edge, args.eps, args.taper_length)
TAG = f'{field_tag(args.b0)}_{CASE_TAG}'
HERE = Path(__file__).parent
INPUT_LATTICE_JSON = HERE / 'fccee_z_lcc.json'
INPUT_TEMPLATE_JSON = HERE / f'004_idea_solenoid_lines_{TAG}.json'
OUTPUT_LATTICE_JSON = HERE / f'temp_fcc_ee_lcc_idea_bfe_solenoids_{TAG}.json'

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']
SOLENOID_INSERTION_S_TOL = 1e-8

if not INPUT_TEMPLATE_JSON.is_file():
    raise SystemExit(f'{INPUT_TEMPLATE_JSON.name} not found: run '
                     '004a_build_idea_solenoid.py with the same --b0, --edge, '
                     '--eps, --taper-length first.')
template_data = xt.json.load(INPUT_TEMPLATE_JSON)
metadata = template_data['metadata']
main_template = xt.Line.from_dict(template_data['lines']['main_solenoid'])
comp_template = xt.Line.from_dict(
    template_data['lines']['compensation_solenoid'])
corr_segments = metadata['first_corr_segments']
print(f'IDEA {metadata["b0"]:g} T, edge {metadata["edge"]:.2f} m, eps '
      f'{metadata["eps"]:g}: {len(main_template.elements)} BFieldExpansion '
      f'elements, |s| <= {metadata["z_match"]:.2f} m; first-corrector '
      'segments ' + ', '.join(f'{seg["s_range"][0]:.3f}-{seg["s_range"][1]:.3f}'
                              f' m (weight {seg["weight"]:.4f})'
                              for seg in corr_segments))

#########################################################
# Install independent solenoid clones at the IPs (004b) #
#########################################################

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring
tw_bare = line.twiss4d()

env['sext_amp'] = 1.0
env['comp_b_scale'] = 1.0
env['main_b_scale'] = 1.0

for ip_name in IP_NAMES:

    line.cycle(f'end_ds_start_straight_{ip_name}')
    s_ip = line.get_table()['s', ip_name]
    print(f'Installing the IDEA solenoid and correctors around {ip_name}')

    env[f'on_sol_{ip_name}'] = 0
    env[f'on_comp_sol_{ip_name}'] = 0
    env[f'comp_b_scale_left_{ip_name}'] = 1.0
    env[f'comp_b_scale_right_{ip_name}'] = 1.0

    main_knob_ref = env.ref[f'on_sol_{ip_name}'] * env.ref['main_b_scale']
    comp_left_knob_ref = (
        env.ref[f'on_comp_sol_{ip_name}'] * env.ref['comp_b_scale']
        * env.ref[f'comp_b_scale_left_{ip_name}'])
    comp_right_knob_ref = (
        env.ref[f'on_comp_sol_{ip_name}'] * env.ref['comp_b_scale']
        * env.ref[f'comp_b_scale_right_{ip_name}'])
    solenoid_lines = {}
    for clone_name, template_line, element_prefix, knob_ref in (
            ('main', main_template, f'sol_slice_{ip_name}', main_knob_ref),
            ('comp_left', comp_template, f'comp_sol_slice_left_{ip_name}',
             comp_left_knob_ref),
            ('comp_right', comp_template, f'comp_sol_slice_right_{ip_name}',
             comp_right_knob_ref)):
        element_names = []
        name_width = len(str(max(0, len(template_line.element_names) - 1)))
        for ii, template_element in enumerate(template_line.elements):
            element_name = f'{element_prefix}_{ii:0{name_width}d}'
            cloned_element = template_element.copy()
            env.elements[element_name] = cloned_element
            element_names.append(element_name)
            if not is_field_element(cloned_element):
                continue
            setattr(env.ref[element_name], scale_attr(cloned_element),
                    get_scale(cloned_element) * knob_ref)
            if transverse_order(cloned_element) >= 2:
                ref = env.ref[element_name]
                for attr in coefficient_attrs(cloned_element):
                    values = np.asarray(getattr(cloned_element, attr))
                    if values.ndim < 2 or values.shape[0] <= 2:
                        continue                    # e.g. knc = 0 here
                    ref_attr = getattr(ref, attr)
                    for k in range(values.shape[1]):
                        ref_attr[2, k] = (float(values[2, k])
                                          * env.ref['sext_amp'])
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

    # First corrector, 010a-style segments on the main elements.
    table_region = line.get_table().rows[
        f'end_ds_start_straight_{ip_name}':f'end_straight_start_ds_{ip_name}']
    s_ip = table_region['s', ip_name]
    s_mid = 0.5 * (table_region.s_start + table_region.s_end) - s_ip
    for side, sign in (('right', 1), ('left', -1)):
        for plane, attr in (('h', 'knl'), ('v', 'ksl')):
            env[f'acb{plane}1_sol_{side}_{ip_name}'] = 0
        for nn, seg in enumerate(corr_segments, start=1):
            lo, hi = sorted(sign * np.asarray(seg['s_range'], dtype=float))
            rows = table_region.rows[(s_mid > lo) & (s_mid < hi)]
            assert len(rows.name) > 0
            assert all(name.startswith(f'sol_slice_{ip_name}_')
                       for name in rows.env_name)
            lengths = np.array([env.get(name).length
                                for name in rows.env_name])
            seg_length = hi - lo
            assert np.isclose(lengths.sum(), seg_length, rtol=0, atol=1e-9), (
                ip_name, side, nn, lengths.sum(), seg_length)
            for plane, attr in (('h', 'knl'), ('v', 'ksl')):
                parent = f'acb{plane}1_sol_{side}_{ip_name}'
                knob = f'acb{plane}1_seg{nn}_sol_{side}_{ip_name}'
                if 'strength' in seg:
                    env[knob] = seg['strength']
                else:
                    env[knob] = seg['weight'] * env.ref[parent]
                for name, length in zip(rows.env_name, lengths):
                    getattr(env.ref[name], attr)[0] += (
                        env.ref[knob] * length / seg_length)

    line.insert([
        env.new(f'dy_match_r_{ip_name}', xt.Marker,
                at=COMPENSATION_CORRECTOR_MARKER_DS, from_=ip_name),
        env.new(f'dy_match_l_{ip_name}', xt.Marker,
                at=-COMPENSATION_CORRECTOR_MARKER_DS, from_=ip_name),
        env.new(f'corr_sol_right_{ip_name}', xt.Multipole,
                length=COMPENSATION_CORRECTOR_LENGTH, isthick=False,
                anchor='end', at=0, from_=f'dy_match_r_{ip_name}@start'),
        env.new(f'corr_sol_left_{ip_name}', xt.Multipole,
                length=COMPENSATION_CORRECTOR_LENGTH, isthick=False,
                anchor='start', at=0, from_=f'dy_match_l_{ip_name}@end'),
    ])

env.to_json(OUTPUT_LATTICE_JSON)
print(f'Wrote {OUTPUT_LATTICE_JSON}')

##########
# Checks #
##########

# Solenoids off: the bare optics.
tw_off = line.twiss4d()
dbet = max(abs(tw_off['bet' + p, ip] / tw_bare['bet' + p, ip] - 1)
           for p in 'xy' for ip in IP_NAMES)
print(f'Solenoids off: Qx {tw_off.qx:.10f} (bare {tw_bare.qx:.10f}), Qy '
      f'{tw_off.qy:.10f} (bare {tw_bare.qy:.10f}), max beta beat at the IPs '
      f'{dbet:.1e}')

# Solenoids on: the net int(Bs) of the ring.
rigidity0 = line.particle_ref.rigidity0[0]
for ip_name in IP_NAMES:
    env[f'on_sol_{ip_name}'] = 1.0
    env[f'on_comp_sol_{ip_name}'] = 1.0
bs_main = bs_comp = 0.0
for name in line.element_names:
    if name.startswith(('sol_slice_', 'comp_sol_slice_')):
        integral = ksol_integral(env.get(name), rigidity0=rigidity0) * rigidity0
        if name.startswith('sol_slice_'):
            bs_main += integral
        else:
            bs_comp += integral
print(f'Ring int(Bs), solenoids on: main {bs_main:.9g}, compensation '
      f'{bs_comp:.9g}, total {bs_main + bs_comp:.3e} T m')
