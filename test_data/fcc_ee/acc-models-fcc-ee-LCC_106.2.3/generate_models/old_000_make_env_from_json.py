import os
import argparse
import xtrack as xt
import xobjects as xo
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

if configuration == 'z':
    source_fname = f'../lattices/z/source/lcc_106a_Z_scrabON_srON_opt.json'
elif configuration == 't':
    source_fname = f'../lattices/t/source/LCC_tt_scrabON_srON_opt.json'

line = xt.load(source_fname)

line.replace_all_repeated_elements()

# replace element names containing ':' with '.' (to have names compatible with madx)
tt = line.get_table()
replacements = {}
for nn in tqdm(tt.env_name, desc='Replacing colons in element names'):
    if ':' in nn:
        new_nn = nn.replace(':', '.')
        assert new_nn not in line.env.elements, f'Element {new_nn} already exists in the environment.'
        line.env.new(new_nn, nn) # clone
        line[new_nn].prototype = None
        replacements[nn] = new_nn

line.discard_tracker()
new_element_names = []
for nn in line.element_names:
    new_element_names.append(replacements.get(nn, nn)) # get replaced name if any or keep old
line.element_names = new_element_names

for nn in tqdm(replacements.keys(), desc='Removing old elements with colons'):
    line.env.elements.remove(nn)

# make a temp folder if needed
temp_folder = f'../lattices/{configuration}/_temp'
os.makedirs(temp_folder, exist_ok=True)

# Set exact drift with flag
line.config['XTRACK_USE_EXACT_DRIFTS'] = False
line.configure_drift_model('exact')

# Switch off radiation, tapering and reset cavity lag
line.configure_radiation(None)
for nn in line.element_names:
    if hasattr(line[nn], 'delta_taper'):
        line[nn].delta_taper = 0.0
    if hasattr(line[nn], 'lag_taper'):
        line[nn].lag_taper = 0.0

line['rf400'].lag = 180  # degrees

env = line.env
env['fccee_p_ring'] = line

env['rf_harmon_400'] = 121200
env['rf_lag_400'] = 0.5
env['circumference'] = env['fccee_p_ring'].get_length()
env['rf400'].frequency = '299792458 / circumference * rf_harmon_400'
env['rf400'].lag = 'rf_lag_400 * 360.'


if configuration == 't':
    env['rf400.0'].frequency = '299792458 / circumference * rf_harmon_400'
    env['rf400.0'].lag = 'rf_lag_400 * 360.'
    env['rf_harmon_800'] = 121200 * 2
    env['rf_lag_800'] = 0.5
    env['rf800'].frequency = '299792458 / circumference * rf_harmon_800'
    env['rf800'].lag = 'rf_lag_800 * 360.'
    env['rf800.0'].frequency = '299792458 / circumference * rf_harmon_800'
    env['rf800.0'].lag = 'rf_lag_800 * 360.'

env.to_json(temp_folder + '/env_no_bpms.json')

