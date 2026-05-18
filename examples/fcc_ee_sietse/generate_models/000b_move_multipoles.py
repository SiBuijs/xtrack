import argparse
import xtrack as xt
import xobjects as xo

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

temp_folder = f'../lattices/{configuration}/_temp'

env = xt.load(f'{temp_folder}/env_no_bpms.json')

for nn in list(env.lines.keys()):
    if nn != 'fccee_p_ring':
        del env[nn]

line = env.fccee_p_ring
line.replace_all_repeated_elements()

list_move_right = ['ocx1l','ocx2l','ocy1l','ocy2l','decdl']
list_move_left = ['ocx1r','ocx2r','ocy1r','ocy2r','decdr']

element_list_temp = line.element_names
for ii in range(1):
    for pp,thing in enumerate(element_list_temp):
        if any(s in thing for s in list_move_right):
            if isinstance(line.element_dict[element_list_temp[pp+1]],xt.Sextupole) or isinstance(line.element_dict[element_list_temp[pp+2]],xt.Sextupole):
#                print(element_list_temp[pp+1],thing)
                element_list_temp[pp], element_list_temp[pp + 1] = element_list_temp[pp + 1], element_list_temp[pp]
#print("moving things right DONE. Now moving things left")
for ii in range(5):
#    print(".")
    for pp,thing in enumerate(element_list_temp):
        if any(s in thing for s in list_move_left):
            if isinstance(line.element_dict[element_list_temp[pp-1]],xt.Sextupole) or isinstance(line.element_dict[element_list_temp[pp-2]],xt.Sextupole):
#                print(element_list_temp[pp-1],thing)
                element_list_temp[pp], element_list_temp[pp - 1] = element_list_temp[pp - 1], element_list_temp[pp]
#print("moving things left DONE.")

env['fccee_p_ring'].element_names=element_list_temp

# del env['fccee_p_ring']
# env['fccee_p_ring'] = line
env.to_json(temp_folder + '/env_no_bpms_moved_multipoles.json')
