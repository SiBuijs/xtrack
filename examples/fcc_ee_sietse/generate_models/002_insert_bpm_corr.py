import argparse
import xtrack as xt
import math

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

exclude_bpm = ['qd0br','qd0cr','qf1br','qf1cr','qf1dr',
               'qd0bl','qd0cl','qf1bl','qf1cl','qf1dl']
exclude_corr = ['qd1am','qf3a']

temp_folder = f'../lattices/{configuration}/_temp'

env = xt.load(f'{temp_folder}/env_no_bpms_merged_quads_markers.json')
line = env.fccee_p_ring

tw4d=line.twiss4d()
print("Tunes before splitting dipoles:",tw4d.qx,tw4d.qy)

# add start and end markers as the logic below does not like start/end with a quad
line.insert('_temp_start_ring', xt.Marker(), at=0)
line.append('_temp_end_ring', xt.Marker())

tt = line.get_table()
circumference = tt['s'][-1]
assert tt.name[-1] == '_end_point'
tt = tt.rows[:-1]
tt_quad = tt.rows[tt.element_type == 'Quadrupole']

env.new('corrector_h', xt.Multipole, knl=[0], ksl=[0], length=0)
env.new('corrector_v', xt.Multipole, knl=[0], ksl=[0], length=0)
env.new('bpm', xt.Marker)
for nn in tt_quad.name:
    if not any(fragment in nn for fragment in exclude_bpm):
        env.new('bpm_' + nn, 'bpm')
    if not any(fragment in nn for fragment in exclude_corr):
        env.new('hcor_' + nn, 'corrector_h')
        env.new('vcor_' + nn, 'corrector_v')

print_out=False
prev=-100
cc = 0
names = list(enumerate(tt['name']))

new_places = []
for index, name in reversed(names):
    if tt.element_type[index] == "Quadrupole":
        cc +=1
        if (prev-index)==-1:
            print("split quad!")
        prev = index
        octant_index = math.floor(tt['s_start'][index]/circumference*8)%2
        lower = max(0,index-10)
        upper = min(len(tt['name']),index+10)
        sext_index = -100
        stream = 0
        for ii in range(index,upper):
            if (tt.element_type[ii] == 'Sextupole') and abs(tt['s'][index]-tt['s'][ii])<4.5:
                if print_out:
                    print("found neighboring sextupole (r)",name,tt['name'][ii])
                stream=1
                sext_index=ii
                break
        for ii in range(lower,index):
            if (tt.element_type[ii] == 'Sextupole') and abs(tt['s'][index]-tt['s'][ii])<4.5:
                if print_out:
                    print("found neighboring sextupole (l)",name,tt['name'][ii])
                stream=-1
                sext_index=ii
        if stream!=0: # sext found. place BPM and vcorrector there
            if stream==1:
                #print("trying to insert BPM close to ",line3.element_names[sext_index])
                if not any(fragment in name for fragment in exclude_bpm):
                    new_places.append(env.place("bpm_"+name, at=tt['s_start'][sext_index], anchor='start'))
                else:
                    if print_out:
                        print(name," BPM excluded")
                if not any(fragment in name for fragment in exclude_corr):
                    new_places.append(env.place("hcor_"+name, at=tt['s_start'][sext_index], anchor='start'))
                    new_places.append(env.place("vcor_"+name, at=tt['s_start'][sext_index], anchor='start'))
                else:
                    if print_out:
                        print(name," CORR excluded")
            else:
                if not any(fragment in name for fragment in exclude_bpm):
                    new_places.append(env.place("bpm_"+name, at=tt['s_start'][sext_index+1], anchor='start'))
                else:
                    if print_out:
                        print(name," BPM excluded")
                if not any(fragment in name for fragment in exclude_corr):
                    new_places.append(env.place("hcor_"+name, at=tt['s_start'][sext_index+1], anchor='start'))
                    new_places.append(env.place("vcor_"+name, at=tt['s_start'][sext_index+1], anchor='start'))
                else:
                    if print_out:
                        print(name," CORR excluded")
        else:
            aind = -1
            if (octant_index==0):
                aind =index
            else:
                aind=index+1
            if not any(fragment in name for fragment in exclude_bpm):
                #line3.insert_element("bpm_"+name,at=aind)
                new_places.append(env.place("bpm_"+name, at=tt['s_start'][aind], anchor='start'))
            else:
                if print_out:
                    print(name," BPM excluded")
            if not any(fragment in name for fragment in exclude_corr):
                #line3.insert_element("hcor_"+name,at=aind)
                new_places.append(env.place("hcor_"+name, at=tt['s_start'][aind], anchor='start'))
                #line3.insert_element("vcor_"+name,at=aind)
                new_places.append(env.place("vcor_"+name, at=tt['s_start'][aind], anchor='start'))
            else:
                if print_out:
                    print(name," CORR excluded")

print('Start_insertion:')
line.insert(new_places)
print('Done_insertion.')

# remove temp markers
line.remove('_temp_start_ring')
line.remove('_temp_end_ring')
env.elements.remove('_temp_start_ring')
env.elements.remove('_temp_end_ring')

env.to_json(f'{temp_folder}/env_with_bpms_and_corrs.json')

