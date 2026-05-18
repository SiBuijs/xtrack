import argparse
import xtrack as xt

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

temp_folder = f'../lattices/{configuration}/_temp'

env = xt.load(f'{temp_folder}/env_with_split_dip_bpms_corrs.json')

# Make a composer
env.fccee_p_ring.composer = env.new_builder()
comp = env.fccee_p_ring.composer
tt=env.fccee_p_ring.get_table(attr=True)
for nn in tt.name:
    if nn == '_end_point':
        continue
    if tt['element_type', nn] == 'Drift':
        continue
    comp.place(nn, at=tt['s_center', nn])

# Identify and zero strengths
tt_var = env.vars.get_table()
tt_strengths = tt_var.rows['k.*']
for nn in tt_strengths.name:
    env[nn] = 0.0

# Write lattice file
from xtrack._temp.python_lattice_writer import lattice_py_generation as lpg
content = lpg.gen_py_lattice(env)

with open(temp_folder + '/lattice.py', 'w') as ff:
    ff.write(content)

# Write strengths file
statement = []
for nn in tt_strengths.name:
    if tt_strengths['expr', nn] is not None:
        statement.append(f'env["{nn}"] = "{tt_strengths["expr", nn]}"')
    else:
        statement.append(f'env["{nn}"] = {tt_strengths["value", nn]}')

preamble = '\n'.join([
    'import xtrack as xt',
    'env = xt.get_environment()',
    'env.vars.default_to_zero=True',
    ''
    ])

postamble = '\n'.join([
    '',
    'env.vars.default_to_zero=False',
    ''
    ])

strength_file_content = preamble + '\n'.join(statement) + postamble

with open(temp_folder + '/strengths.py', 'w') as ff:
    ff.write(strength_file_content)
