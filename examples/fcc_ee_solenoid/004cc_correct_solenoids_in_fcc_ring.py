from pathlib import Path

import matplotlib.pyplot as plt
import xtrack as xt


HERE = Path(__file__).parent

# 3 T SplineBoris solenoids, as installed by 004b_install_solenoids_in_fcc_ring.py.
INPUT_LATTICE_JSON = HERE / 'temp_fcc_ee_lcc_splineboris_solenoids_3T.json'

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']
IP_NAME = 'ipa'
NAME_END = f'end_straight_start_ds_{IP_NAME}'


###################################
# Load installed solenoid lattice #
###################################

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring

# Bare ring: all main and compensation solenoids off.
for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 0
    line[f'on_comp_sol_{ip_name}'] = 0


############################
# Ring twiss, optics at IP #
############################

tw0 = line.twiss4d(strengths=True)

print(f'Bare ring optics at {IP_NAME}:')
for kk in ['betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx']:
    print(f'    {kk:5s} = {tw0[kk, IP_NAME]: .6e}')


#################################################
# Line from the IP to the end of the straight   #
#################################################

line_ds = line.select(start=IP_NAME, end=NAME_END, name=f'{IP_NAME}_ds')
if line_ds.particle_ref is None:
    line_ds.particle_ref = line.particle_ref.copy()

tw_ds = line_ds.twiss4d(init=tw0.get_twiss_init(IP_NAME), strengths=True)

print(f'\nLine {IP_NAME} -> {NAME_END}: length = {line_ds.get_length():.4f} m')
print(f'Optics at {NAME_END} (open-line twiss vs ring twiss):')
for kk in ['betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx']:
    print(f'    {kk:5s}  line = {tw_ds[kk, NAME_END]: .6e}   '
          f'ring = {tw0[kk, NAME_END]: .6e}')


##########
# Plot   #
##########

plt.close('all')
tw_ds.plot()
plt.suptitle(f'Bare optics {IP_NAME} -> {NAME_END}  ({INPUT_LATTICE_JSON.name})')
plt.show()
