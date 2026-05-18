import os
import argparse
import xtrack as xt

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

if configuration == 'z':
    madx_fname = 'FCC_v106a_2_0.madx'
    p0c = 45.6e9  # in eV
elif configuration == 't':
    madx_fname = 'FCC_v106b_2_0.madx'
    p0c = 182.5e9  # in eV
elif configuration == 'w':
    madx_fname = 'FCC_v106a_W_2_0.madx'
    p0c = 80.0e9  # in eV
else:
    raise ValueError(f'Unknown configuration {configuration}')

lines_to_kill = {}
lines_to_kill['z'] = [
    'arc_sufl',
    'arc_sufr',
    'marc_sufl',
    'marc_sufr',
    'sec_l_rfb',
    'sec_r_rfb',
    'sec_rfb',
    'arc_rfb',
    'rfb_sector',
    'fcc_sector',
    'ring_rfb',
    'ring_dsff',
    'ring_b'
]
lines_to_kill['t'] = [
    'arc_sufl',
    'arc_sufr',
    'marc_sufl',
    'marc_sufr',
    'sec_l_rfa',
    'sec_r_rfa',
    'sec_rfa',
    'arc_rfa',
    'rfa_sector',
    'fcc_sector',
    'ring_rfa',
    'ring_dsff',
    'ring_a'
]
lines_to_kill['w'] = [
    'arc_sufl',
    'arc_sufr',
    'marc_sufl',
    'marc_sufr',
    'sec_l_rfb',
    'sec_r_rfb',
    'sec_rfb',
    'arc_rfb',
    'rfb_sector',
    'fcc_sector',
    'ring_rfb',
    'ring_dsff',
    'ring_b'
]
additional_settings = {}
additional_settings['z'] = {
    'rf_on': 1.0,
    'cs_frac': 0.55,
    'cs_comp': 0.55,
    'sxt_on': 1.0,
    'rf_v_400': '90. * rf_on',
    'rf_harmon_400': 121200,
    'rf_lag_400': 0.5
}
additional_settings['t'] = {
    'rf_on': 1.0,
    'cs_frac': 0.4,
    'cs_comp': 0.4,
    'sxt_on': 1.0,
    'rf_v_400': '1.93e3/2. * rf_on',
    'rf_harmon_400': 121200,
    'rf_lag_400': 0.5,
    'rf_v_800': '8.08e3/2. * rf_on',
    'rf_harmon_800': 2*121200,
    'rf_lag_800': 0.5
}
additional_settings['w'] = {
    'rf_on': 1.0,
    'cs_frac': 0.45,
    'cs_comp': 0.45,
    'sxt_on': 1.0,
    'rf_v_400': '1000. * rf_on',
    'rf_harmon_400': 121200,
    'rf_lag_400': 0.5
}
madx_path = f'../lattices/{configuration}/source/{madx_fname}'

# make a temp folder if needed
temp_folder = f'../lattices/{configuration}/_temp'
os.makedirs(temp_folder, exist_ok=True)

with open(madx_path, 'r') as fid:
    madx_lines = fid.readlines()


for ii, ll in enumerate(madx_lines):
    madx_lines[ii] = madx_lines[ii].replace("'", "_prime")
    if 'return' in ll.lower():
        madx_lines[ii] = '! ' + ll

mad_str = ''.join(madx_lines)
lines = mad_str.split("\n")
for ii, ll in enumerate(lines):
    to_comment=any([ll.lower().startswith(ltk) for ltk in lines_to_kill[configuration]])
    if to_comment:
        lines[ii] = '! ' + ll

mad_str = '\n'.join(lines)
mad_str = mad_str.replace(" ", "")
mad_str = mad_str.lower()
mad_str = mad_str.replace('line:=', 'line=')

with open(f'{temp_folder}/temp_madx_for_xsuite.txt', 'w') as fid:
    fid.write(mad_str)

env = xt.load(string=mad_str, format='madx')

env.ring_ptr.set_particle_ref('positron', p0c=p0c)
env.lines['fccee_p_ring'] = env.ring_ptr
del env.lines['ring_ptr']

for ll in env.lines.values():
    ll.composer = None

env.vars.update(additional_settings[configuration])

env['circumference'] = env['fccee_p_ring'].get_length()
env['rf400'].frequency = '299792458 / circumference * rf_harmon_400'
env['rf400'].voltage = 'rf_v_400 * 1e6'
env['rf400'].lag = 'rf_lag_400 * 360.'
if configuration == 't':
    env['rf800'].frequency = '299792458 / circumference * rf_harmon_800'
    env['rf800'].voltage = 'rf_v_800 * 1e6'
    env['rf800'].lag = 'rf_lag_800 * 360.'

env.to_json(temp_folder + '/env_no_bpms.json')
