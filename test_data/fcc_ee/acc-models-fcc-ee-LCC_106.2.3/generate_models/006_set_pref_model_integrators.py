import xtrack as xt
import argparse

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

temp_folder = f'../lattices/{configuration}/_temp'

workload = [
    ('env_with_split_dip_bpms_corrs.json', 'env_final.json'),
    ('env_with_bpms_and_corrs.json', 'env_final_merged_dipoles.json'),
]

for fin, fout in workload:
    env = xt.load(f'{temp_folder}/{fin}')

    # Set preferred model integrators
    tt = env.fccee_p_ring.get_table()
    tt_bend = tt.rows[(tt.element_type=='Bend') | (tt.element_type=='RBend')]
    tt_quad = tt.rows[(tt.element_type=='Quadrupole')]
    tt_sext = tt.rows[(tt.element_type=='Sextupole')]
    tt_doublet = tt_quad.rows['qd0a.*|qd0b.*|qd0c.*|qf1a.*|qf1b.*|qf1c.*|qf1d.*']

    env.set(tt_bend, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')
    env.set(tt_quad, integrator='uniform', num_multipole_kicks=5, model='mat-kick-mat')
    env.set(tt_sext, integrator='yoshida4', num_multipole_kicks=1, model='drift-kick-drift-expanded')
    env.set(tt_doublet, integrator='yoshida4', num_multipole_kicks=200, model='drift-kick-drift-exact')

    if configuration == 'z':
        env.fccee_p_ring.set_particle_ref('positron', energy0=45.6e9)
    elif configuration == 't':
        env.fccee_p_ring.set_particle_ref('positron', energy0=182.5e9)
    elif configuration == 'w':
        env.fccee_p_ring.set_particle_ref('positron', energy0=80.0e9)
    else:
        raise ValueError(f'Unknown configuration {configuration}')

    env.to_json(f'{temp_folder}/{fout}')

# Add integrators and particle ref to python lattice files
with open(f'{temp_folder}/lattice.py', 'r') as fid:
    lattice_py = fid.read()

particle_ref_str = ''
if configuration == 'z':
    particle_ref_str = "env.fccee_p_ring.set_particle_ref('positron', energy0=45.6e9)"
elif configuration == 't':
    particle_ref_str = "env.fccee_p_ring.set_particle_ref('positron', energy0=182.5e9)"
elif configuration == 'w':
    particle_ref_str = "env.fccee_p_ring.set_particle_ref('positron', energy0=80.0e9)"
else:
    raise ValueError(f'Unknown configuration {configuration}')

integrators_str = '''
# Set appropriate models and integrators
env.fccee_p_ring.configure_drift_model('exact')
tt = env.fccee_p_ring.get_table()
tt_bend = tt.rows[(tt.element_type=='Bend') | (tt.element_type=='RBend')]
tt_quad = tt.rows[(tt.element_type=='Quadrupole')]
tt_sext = tt.rows[(tt.element_type=='Sextupole')]
tt_doublet = tt_quad.rows['qd0a.*|qd0b.*|qd0c.*|qf1a.*|qf1b.*|qf1c.*|qf1d.*']

env.set(tt_bend, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')
env.set(tt_quad, integrator='uniform', num_multipole_kicks=5, model='mat-kick-mat')
env.set(tt_sext, integrator='yoshida4', num_multipole_kicks=1, model='drift-kick-drift-expanded')
env.set(tt_doublet, integrator='yoshida4', num_multipole_kicks=200, model='drift-kick-drift-exact')
'''

lattice_py += '\n' + particle_ref_str + '\n' + integrators_str

with open(f'{temp_folder}/lattice_final.py', 'w') as fid:
    fid.write(lattice_py)
