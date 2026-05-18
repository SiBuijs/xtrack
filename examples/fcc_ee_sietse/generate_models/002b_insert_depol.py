import argparse
import xtrack as xt
import math

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

temp_folder = f'../lattices/{configuration}/_temp'

env = xt.load(f'{temp_folder}/env_with_bpms_and_corrs.json')
line = env.fccee_p_ring

tww = line.twiss4d()

depol1_loc_4l_name = 'vcor_qd1a.319'
depol2_loc_4l_name = 'vcor_qd2c.0'
depol3_loc_4l_name = 'vcor_qd4c.0'

depol1_loc_4r_name = 'vcor_qd4c.1'
depol2_loc_4r_name = 'vcor_qd2c.1'
depol3_loc_4r_name = 'vcor_qd1a.328'

depol1_loc_8l_name = 'vcor_qd1a.751'
depol2_loc_8l_name = 'vcor_qd2d.0'
depol3_loc_8l_name = 'vcor_qd4d.0'

depol1_loc_8r_name = 'vcor_qd4d.1'
depol2_loc_8r_name = 'vcor_qd2d.1'
depol3_loc_8r_name = 'vcor_qd1a.760'

print("DEPOL locations")
for thing in [depol1_loc_4l_name,depol2_loc_4l_name,depol3_loc_4l_name,depol1_loc_4r_name,depol2_loc_4r_name,depol3_loc_4r_name,depol1_loc_8l_name,depol2_loc_8l_name,depol3_loc_8l_name,depol1_loc_8r_name,depol2_loc_8r_name,depol3_loc_8r_name]:
    print(thing,tww['s',thing],tww['mux',thing],tww['muy',thing],tww['betx',thing],tww['bety',thing],tww['dx',thing])

line.insert(f'sdepol1_l8', xt.Multipole(ksl=[0]), at=depol1_loc_8l_name)
line.insert(f'sdepol2_l8', xt.Multipole(ksl=[0]), at=depol2_loc_8l_name)
line.insert(f'sdepol3_l8', xt.Multipole(ksl=[0]), at=depol3_loc_8l_name)

line.insert(f'sdepol1_r8', xt.Multipole(ksl=[0]), at=depol1_loc_8r_name)
line.insert(f'sdepol2_r8', xt.Multipole(ksl=[0]), at=depol2_loc_8r_name)
line.insert(f'sdepol3_r8', xt.Multipole(ksl=[0]), at=depol3_loc_8r_name)
#for thing in [depol1_loc_4l_name,depol2_loc_4l_name,depol3_loc_4l_name,depol1_loc_4r_name,depol2_loc_4r_name,depol3_loc_4r_name]:
#    print(thing,tww['s',thing])

line.insert(f'sdepol1_l4', xt.Multipole(ksl=[0]), at=depol1_loc_4l_name)
line.insert(f'sdepol2_l4', xt.Multipole(ksl=[0]), at=depol2_loc_4l_name)
line.insert(f'sdepol3_l4', xt.Multipole(ksl=[0]), at=depol3_loc_4l_name)

line.insert(f'sdepol1_r4', xt.Multipole(ksl=[0]), at=depol1_loc_4r_name)
line.insert(f'sdepol2_r4', xt.Multipole(ksl=[0]), at=depol2_loc_4r_name)
line.insert(f'sdepol3_r4', xt.Multipole(ksl=[0]), at=depol3_loc_4r_name)

env.to_json(f'{temp_folder}/env_with_bpms_and_corrs_and_depol.json')

