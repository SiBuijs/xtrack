import argparse
import xtrack as xt
import math
import numpy as np
parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

temp_folder = f'../lattices/{configuration}/_temp'

env = xt.load(f'{temp_folder}/env_with_bpms_and_corrs_and_depol.json')
line = env.fccee_p_ring

tw4d=line.twiss4d()
print("Tunes before split:",tw4d.qx,tw4d.qy)
tt = line.get_table(attr=True)
tt_arc_dip = tt.rows['dl1.*']

tt_arc_start = tt.rows['end_ds_start_arc_iph.*']
tt_arc_end = tt.rows['end_arc_start_ds_iph.*']

lower = tt_arc_end['s']
upper = tt_arc_start['s']
#print(lower,upper)
#print(tt.rows['ip.*'])
circumference = tt['s'][-1]

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

d_shift = math.tan(env['ang0_mod']/2.)/2.*0.35
#print("shift segment length",d_shift)
new_line = env.new_line(length=tt['s'][-1], compose=True)
total_offset = 0
offset=0
to_be_skipped=[]
for nn in tt_arc_dip.name:
    ss = tt['s_center', nn]
    in_interval = any(l <= ss <= u for l, u in zip(lower, upper))
    if in_interval==False:
        to_be_skipped.append(nn)
        sector = (math.floor(tt['s_center', nn]*8./circumference))%2.
        if sector==0:
            offset=d_shift
        else:
            offset=-d_shift
        s_start = tt['s_start', nn]
        s_center = tt['s_center', nn]
        s_end = tt['s_end', nn]
        prefix = nn.split('.')[0]
        suffix = nn.split('.')[-1]
        total_offset+=offset
        new_line.new(prefix+'a.'+suffix, 'dl1as', anchor='start', at=s_start+total_offset)
        total_offset+=2.*offset
        new_line.new(prefix+'b.'+suffix, 'dl1as', anchor='center', at=s_center+total_offset)
        total_offset+=2.*offset
        new_line.new(prefix+'c.'+suffix, 'dl1as', anchor='end', at=s_end+total_offset)
        total_offset+=offset

#print("balance must be 0:",total_offset,total_offset/d_shift)

assert math.isclose(0, total_offset, abs_tol=1.0e-2)
#print("number of DL1's that got split:",len(to_be_skipped))
total_offset = 0
set_delete = set(to_be_skipped)
for nn in tt.name:
    if nn in set_delete:
        sector = (math.floor(tt['s_start', nn]*8./circumference))%2.
        if sector==0:
            total_offset+=6.0*d_shift
        else:
            total_offset+=-6.0*d_shift
        continue
    if nn == '_end_point':
        continue
    if tt['element_type', nn] == 'Drift':
        continue
    s_start = tt['s_start', nn]
    new_line.place(nn, at=s_start+total_offset, anchor='start')

#print("balance must be 0:",total_offset)

assert math.isclose(0, total_offset, abs_tol=1.0e-2)
new_line.end_compose()
#print(np.array(new_line.get_table().rows['ip.*']['s']-tt.rows['ip.*']['s']))
assert np.allclose(np.array(new_line.get_table().rows['ip.*']['s']-tt.rows['ip.*']['s']),0,atol=2.0e-8)
new_line.particle_ref = line.particle_ref


tw4dn=new_line.twiss4d()
print("Tunes after split:",tw4dn.qx,tw4dn.qy)

opt = new_line.match(
    method='4d', # <- passed to twiss
    vary=[
        xt.VaryList(['kqd1', 'kqf2'], step=1e-8, tag='quads')
    ],
    targets = [
        xt.TargetSet(qx=tw4d.qx, qy=tw4d.qy, tol=1e-6, tag='tune'),
    ])
#opt.target_status()
tw4dnn=new_line.twiss4d()
print("Tunes after rematch:",tw4dnn.qx,tw4dnn.qy)


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
