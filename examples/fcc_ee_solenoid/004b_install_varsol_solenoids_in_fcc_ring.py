from pathlib import Path

import numpy as np
import xtrack as xt

from solenoid_params import (
    COMP_SOLENOID_DISTANCE_FROM_IP,
    COMPENSATION_CORRECTOR_LENGTH,
    COMPENSATION_CORRECTOR_MARKER_DS,
    FIELD_TAG,
    MAIN_SOLENOID_CORRECTOR_DS_END,
    MAIN_SOLENOID_CORRECTOR_DS_START,
)


HERE = Path(__file__).parent
INPUT_LATTICE_JSON = HERE / 'fccee_z_lcc.json'
INPUT_SOLENOID_LINES_JSON = HERE / f'004_solenoid_lines_{FIELD_TAG}.json'
OUTPUT_LATTICE_JSON = HERE / f'temp_fcc_ee_lcc_varsol_solenoids_{FIELD_TAG}.json'

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']

SOLENOID_INSERTION_S_TOL = 1e-8


def _apply_solenoid_knob_to_element(env, element_name, knob_ref):
    element = env.get(element_name)
    ref = env.ref[element_name]
    ref.ks_profile[0] = float(element.ks_profile[0]) * knob_ref
    ref.ks_profile[1] = float(element.ks_profile[1]) * knob_ref
    knl = element.knl
    ksl = element.ksl
    if knl is not None and len(knl) > 0:
        ref.knl[0] = float(knl[0]) * knob_ref
    if ksl is not None and len(ksl) > 0:
        ref.ksl[0] = float(ksl[0]) * knob_ref


######################################################
# Load the FCC lattice and the isolated solenoid lines #
######################################################

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring

line_data = xt.json.load(INPUT_SOLENOID_LINES_JSON)
solenoid_templates = {
    name: xt.Line.from_dict(line_dict)
    for name, line_dict in line_data['lines'].items()
}

main_solenoid_template = solenoid_templates['main_solenoid_varsol']
comp_solenoid_template = solenoid_templates['compensation_solenoid_varsol']


###########################################################
# Install independent VariableSolenoid solenoid clones at IPs #
###########################################################

for ip_name in IP_NAMES:

    line.cycle(f'end_ds_start_straight_{ip_name}')
    table_before_insertion = line.get_table()
    s_ip = table_before_insertion['s', ip_name]

    print(f'Installing VariableSolenoid solenoids and correctors around {ip_name}')

    env[f'on_sol_{ip_name}'] = 0
    env[f'on_comp_sol_{ip_name}'] = 0

    solenoid_lines = {}
    clone_specs = [
        ('main', main_solenoid_template,
         f'sol_slice_{ip_name}', env.ref[f'on_sol_{ip_name}']),
        ('comp_left', comp_solenoid_template,
         f'comp_sol_slice_left_{ip_name}', env.ref[f'on_comp_sol_{ip_name}']),
        ('comp_right', comp_solenoid_template,
         f'comp_sol_slice_right_{ip_name}', env.ref[f'on_comp_sol_{ip_name}']),
    ]

    for clone_name, template_line, element_prefix, knob_ref in clone_specs:
        element_names = []
        name_width = len(str(max(0, len(template_line.element_names) - 1)))

        for ii, template_element in enumerate(template_line.elements):
            element_name = f'{element_prefix}_{ii:0{name_width}d}'
            env.elements[element_name] = template_element.copy()
            _apply_solenoid_knob_to_element(env, element_name, knob_ref)
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

    env[f'acbh1_sol_right_{ip_name}'] = 0
    env[f'acbv1_sol_right_{ip_name}'] = 0
    env[f'acbh1_sol_left_{ip_name}'] = 0
    env[f'acbv1_sol_left_{ip_name}'] = 0

    table_region = line.get_table().rows[
        f'end_ds_start_straight_{ip_name}':f'end_straight_start_ds_{ip_name}']
    s_ip = table_region['s', ip_name]

    table_corrector_right = table_region.rows[
        s_ip + MAIN_SOLENOID_CORRECTOR_DS_START:
        s_ip + MAIN_SOLENOID_CORRECTOR_DS_END:'s']
    assert np.all(table_corrector_right.element_type == 'VariableSolenoid')
    assert all(
        env_name.startswith(f'sol_slice_{ip_name}_')
        for env_name in table_corrector_right.env_name)
    length_corrector_right = (
        table_corrector_right['s_end'][-1]
        - table_corrector_right['s_start'][0])

    for env_name in table_corrector_right.env_name:
        element = env.get(env_name)
        env.ref[env_name].knl[0] += (
            env.ref[f'acbh1_sol_right_{ip_name}']
            / length_corrector_right * element.length)
        env.ref[env_name].ksl[0] += (
            env.ref[f'acbv1_sol_right_{ip_name}']
            / length_corrector_right * element.length)

    table_corrector_left = table_region.rows[
        s_ip - MAIN_SOLENOID_CORRECTOR_DS_END:
        s_ip - MAIN_SOLENOID_CORRECTOR_DS_START:'s']
    assert np.all(table_corrector_left.element_type == 'VariableSolenoid')
    assert all(
        env_name.startswith(f'sol_slice_{ip_name}_')
        for env_name in table_corrector_left.env_name)
    length_corrector_left = (
        table_corrector_left['s_end'][-1]
        - table_corrector_left['s_start'][0])

    for env_name in table_corrector_left.env_name:
        element = env.get(env_name)
        env.ref[env_name].knl[0] += (
            env.ref[f'acbh1_sol_left_{ip_name}']
            / length_corrector_left * element.length)
        env.ref[env_name].ksl[0] += (
            env.ref[f'acbv1_sol_left_{ip_name}']
            / length_corrector_left * element.length)

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
