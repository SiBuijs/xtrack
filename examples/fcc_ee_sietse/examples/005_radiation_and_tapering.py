import matplotlib.pyplot as plt

import xtrack as xt

env = xt.Environment()

# Load lattice from python files
env = xt.Environment()
env.call('../lattices/z/fccee_z_lattice.py')
env.call('../lattices/z/fccee_z_strengths.py')
line = env['fccee_p_ring']

# Get thin line
line = env.fccee_p_ring

tt = line.get_table()
tt_cav = tt.rows[tt.element_type == 'Cavity']

tw0 = line.twiss4d()

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw = line.twiss(radiation_analysis=True, delta_chrom=1e-12)

plt.close('all')
tw.plot('delta', lattice=False)
plt.show()
