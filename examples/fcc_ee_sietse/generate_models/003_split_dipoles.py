import argparse
import xtrack as xt

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

temp_folder = f'../lattices/{configuration}/_temp'

env = xt.load(f'{temp_folder}/env_with_bpms_and_corrs.json')
line = env.fccee_p_ring

tt = line.get_table(attr=True)
tt_arc_dip = tt.rows['dl1.*']

# Check no repeated elements
tt_no_generated_drifts = tt.rows[~tt.rows.mask['\\|\\|drift_.*']]
assert len(set(tt_no_generated_drifts.env_name)) == len(tt_no_generated_drifts.env_name)

# Check all arc dipoles have the same length
assert len(set(tt_arc_dip.length)) == 1
assert len(set(tt_arc_dip.angle_rad)) == 1

env['l_bellow']=0.17
env['dl_len_mod']='(dl_len-2.0*l_bellow)/3.'
env['ang0_mod']= 'ang0/3.'
env.new('dl1as', xt.RBend, length_straight='dl_len_mod', angle='ang0_mod',k0_from_h=True)

new_line = env.new_line(length=tt['s'][-1], compose=True)
for nn in tt_arc_dip.name:
    s_start = tt['s_start', nn]
    s_center = tt['s_center', nn]
    s_end = tt['s_end', nn]
    prefix = nn.split('.')[0]
    suffix = nn.split('.')[-1]
    new_line.new(prefix+'a.'+suffix, 'dl1as', anchor='start', at=s_start)
    new_line.new(prefix+'b.'+suffix, 'dl1as', anchor='center', at=s_center)
    new_line.new(prefix+'c.'+suffix, 'dl1as', anchor='end', at=s_end)

set_delete = set(tt_arc_dip.name)
for nn in tt.name:
    if nn in set_delete:
        continue
    if nn == '_end_point':
        continue
    if tt['element_type', nn] == 'Drift':
        continue
    s_start = tt['s_start', nn]
    new_line.place(nn, at=s_start, anchor='start')

new_line.end_compose()

new_line.particle_ref = line.particle_ref

del env.lines['fccee_p_ring']
env['fccee_p_ring'] = new_line

prototypes = set()
for nn in new_line.element_names:
    if env[nn].prototype is not None:
        prototypes.add(env[nn].prototype)

added = True
while added:
    added = False
    for pp in list(prototypes):
        pp_proto = env[pp].prototype
        if pp_proto is not None and pp_proto not in prototypes:
            prototypes.add(pp_proto)
            added = True

print('Delete unused elements:')
for nn in list(env.elements.keys()):
    if nn not in new_line.element_names and nn not in prototypes:
        del env[nn]
print('Done.')

env.to_json(f'{temp_folder}/env_with_split_dip_bpms_corrs.json')
