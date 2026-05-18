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

env = xt.load(f'{temp_folder}/env_final_merged_dipoles.json')
line = env.fccee_p_ring

tw4d=line.twiss4d()
print("creating table:")
print("Tunes:",tw4d.qx,tw4d.qy)

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()
tw = line.twiss(method="6d",eneloss_and_damping = True, compute_chromatic_properties=True)

parameter_table = {}


# Configuration and kinematics
energy = round(line.particle_ref.energy[0]/1.0e9,3)
mode_dict = {}
mode_dict['z'] = {'coupling':0.001,'mode':'z','energy':45.6,'number of bunches':12000,'rf_freq':400e6,'h':121200,'xi_y_max':0.09}
mode_dict['tt'] = {'coupling':0.001,'mode':'tt','energy':182.5,'number of bunches':68,'rf_freq':800e6,'h':121200*2,'xi_y_max':0.09}
mode_dict['w'] = {'coupling':0.001,'mode':'w','energy':80.0,'number of bunches':2110,'rf_freq':400e6,'h':121200,'xi_y_max':0.1}


for thing in mode_dict.keys():
    if abs(energy-mode_dict[thing]['energy'])<1.:
        act_mode=mode_dict[thing]['mode']
        kappa=mode_dict[thing]['coupling']
        rf_freq=mode_dict[thing]['rf_freq']
        harm=mode_dict[thing]['h']
        ximax=mode_dict[thing]['xi_y_max']

parameter_table['mode']=[act_mode,'']
parameter_table['beam energy'] = [energy,'GeV/c^2']
parameter_table['relativistic gamma'] = [int(round(line.particle_ref.energy[0]/line.particle_ref.gamma0[0],0)),'']
parameter_table['circumference'] = [tw.s[-1],'m']
parameter_table['revolution frequency'] = [299792458/tw.s[-1],'Hz']
#k0 = float(line.element_dict['dl1a'].k0._value)
#parameter_table['bending radius'] = [1.0/k0,'m']
parameter_table['energy loss per turn'] = [tw.eneloss_turn/1.0e9,'GeV']
parameter_table['SR power single particle'] = [parameter_table['revolution frequency'][0]*tw.eneloss_turn/1.0e9*1.6022e-10,'W']
parameter_table['SR power beam'] = [50,'MW']
parameter_table['number of bunches'] = [mode_dict[act_mode]['number of bunches'],'']
parameter_table['bunch population'] = [parameter_table['SR power beam'][0]*1.0e6/parameter_table['SR power single particle'][0]/parameter_table['number of bunches'][0]/1.0e10,'1E10']
parameter_table['horizontal emittance'] = [tw.eq_gemitt_x*1.0e9,'nm']
parameter_table['vertical emittance'] = [tw.eq_gemitt_x*kappa*1.0e12,'pm']
parameter_table['ey/ex'] = [kappa,""]
parameter_table['momentum compaction factor'] = [tw.momentum_compaction_factor*1e6,"1.0E-6"]
parameter_table['horizontal beta*'] = [tw['betx','ipa'],'m']
parameter_table['vertical beta*'] = [tw['bety','ipa'],'m']
parameter_table['horizontal tune'] = [tw.qx,'']
parameter_table['vertical tune'] = [tw.qy,'']
parameter_table['longitudinal tune'] = [tw.qs,'']
parameter_table['horizontal chromaticity'] = [tw.dqx,'']
parameter_table['vertical chromaticity'] = [tw.dqy,'']
V_400 = 0
V_800 = 0
f_400 = 0
f_800 = 0
for thing in line.element_names:
    if isinstance(line.element_dict[thing],xt.Cavity):
        print(line.element_dict[thing])
        if abs(line.element_dict[thing].frequency-400e6)<1.0e7:
            V_400+=line.element_dict[thing].voltage
            f_400 = line.element_dict[thing].frequency
        if abs(line.element_dict[thing].frequency-800e6)<1.0e7:
            V_800+=line.element_dict[thing].voltage
            f_800 = line.element_dict[thing].frequency
parameter_table['harmonic number'] = [round(f_400/(299792458/tw.s[-1])),'']
parameter_table['RF voltage 400 MHz'] = [V_400/1.0e6,"MV"]
parameter_table['RF voltage 800 MHz'] = [V_800/1.0e6,"MV"]
parameter_table['RF frequency 400 MHz'] = [f_400/1e6,"MHz"]
parameter_table['RF frequency 800 MHz'] = [f_800/1e6,"MHz"]

from MachineAndBeamParametersGenerator import MachineAndBeamParametersGenerator
param_gen = MachineAndBeamParametersGenerator()

param_gen.n_bunches = parameter_table['number of bunches'][0]
param_gen.momentum_compaction = parameter_table['momentum compaction factor'][0]*1.0e-6
param_gen.circumference = parameter_table['circumference'][0]
param_gen.voltage = (parameter_table['RF voltage 800 MHz'][0]+parameter_table['RF voltage 400 MHz'][0])*1.0e6
param_gen.RF_frequency = rf_freq
param_gen.harmonic_number = harm
param_gen.Xing = 0.03
param_gen.energy = parameter_table['beam energy'][0]
param_gen.energy_loss_per_turn = parameter_table['energy loss per turn'][0] # GeV
param_gen.n_ip = 4
param_gen.beta_x = parameter_table['horizontal beta*'][0]
param_gen.bending_radius = 10021.0
param_gen.arc_cell_length = 104
param_gen._epsilon_x = tw.eq_gemitt_x

param_gen.a_hg = 1.3
param_gen.adjust_hourglass_to_beta_y(beta_y=parameter_table['vertical beta*'][0])
param_gen.intensity_per_bunch =  parameter_table['bunch population'][0]*1.0e10
param_gen.adjust_bunch_intensity_to_max_SR_power(max_SR_power=50E6)
param_gen.alpha_xy = 1.0/kappa
param_gen.adjust_vertical_emittance_to_xi_y_max(xi_y_max=ximax)


print(f"  Momentum compaction: {param_gen.momentum_compaction:.3e}")
print(f"  Crossing angle: {param_gen.Xing*1e3:.1f} mrad")
print(f"  RF voltage: {param_gen.voltage:.3e}")
print(f"  xi_x: {param_gen.xi_x():.4f}")
print(f"  xi_y: {param_gen.xi_y():.4f}")
print(f"  Qs: {param_gen.q_s():.4f}")
print(f"  beta s: {param_gen.beta_s():.4f}")
print(f"  sigma_z: {param_gen.sigma_z()*1e3:.1f} mm")
print(f"  sigma_z_SR: {param_gen.sigma_z_SR()*1e3:.1f} mm")
print(f"  sigma_delta: {param_gen.sigma_delta()*1e2:.3f} %")
print(f"  sigma_delta_SR: {param_gen.sigma_delta_SR()*1e2:.3f} %")
print(f"  tau_z: {param_gen.tau_z():.1f} turns")
print(f"  beta_x: {param_gen.beta_x*1e3:.2f} mm")
print(f"  beta_y: {param_gen.beta_y()*1e3:.3f} mm")
print(f"  emittance x: {param_gen.epsilon_x()*1e9:.3f} nm")
print(f"  emittance y: {param_gen.epsilon_y()*1e12:.3f} pm")
print(f"  momentum acceptance: {param_gen.momentum_acceptance()*100:.4f} %")
print(f"  bunch intensity: {param_gen.intensity_per_bunch:.2e} particles")
print(f"  instantaneous luminosity: {param_gen.luminosity()/1e34:.3f} x 10^34 cm^-2 s^-1")
print(f"  natural spin tune spread / Qs: {param_gen.natural_spin_tune_spread()/param_gen.q_s():.4f}")
print(f"  interaction_length: {param_gen.interaction_length()*1e3:.3f} mm (err {param_gen.error_on_interaction_length()*100:.2f}%)")
print(f"  beta*y/interaction_length: {param_gen.a_hg}")
print(f"  Piwinski: {param_gen.piwinski()}")
print(f"  4xi_x/Qs: {4*param_gen.xi_x()/param_gen.q_s():.4f}")
print(f"  Sync. rad power: {param_gen.tot_SR_power()*1E-6:.1f} MW")
print(f"  X-Y emittance ratio: {param_gen.alpha_xy:.3f}")



parameter_table['bunch length'] = [param_gen.sigma_z()*1e3,"mm"]
parameter_table['bunch length SR'] = [param_gen.sigma_z_SR()*1e3,"mm"]
parameter_table['energy spread'] = [param_gen.sigma_delta()*1e2,"%"]
parameter_table['energy spread SR'] = [param_gen.sigma_delta_SR()*1e2,"%"]
parameter_table['vertical emittance'] = [param_gen.epsilon_y()*1.0e12,'pm']

parameter_table['ey/ex'] = [1.0/param_gen.alpha_xy,""]
parameter_table['longitudinal damping time']=[param_gen.tau_z(),"turns"]
parameter_table['horizontal BB parameter']=[param_gen.xi_x(),""]
parameter_table['vertical BB parameter']=[param_gen.xi_y(),""]
parameter_table['4xi_x/Qs'] = [4*param_gen.xi_x()/param_gen.q_s(),""]
parameter_table['SpinTuneSpread/Qs'] = [param_gen.natural_spin_tune_spread()/param_gen.q_s(),""]
parameter_table['inst. luminosity'] = [param_gen.luminosity()/1e34,"1E34 Hz/cm^2"]

del parameter_table['SR power single particle']
del parameter_table['relativistic gamma']
del parameter_table['harmonic number']

#for thing in parameter_table.keys():
#    unit = parameter_table[thing][1]
#    value =  parameter_table[thing][0]
#    if isinstance(value,str)==False:
#        value =  parameter_table[thing][0]
#    print(f"{thing} ({unit}):\t{value}")

import pandas as pd

print(
    pd.DataFrame.from_dict(parameter_table, orient="index", columns=["value","unit"])
)

df = pd.DataFrame.from_dict(parameter_table, orient="index", columns=["value","unit"])
with open(f'../parameter_tables/{configuration}_parameter_table.txt', "w") as f:
    f.write(df.to_string())
with open(f'../lattices/{configuration}/parameter_table.txt', "w") as f:
    f.write(df.to_string())
