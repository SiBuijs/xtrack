import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Load lattice from python files
env = xt.Environment()
env.call('../lattices/z/fccee_z_lattice.py')
env.call('../lattices/z/fccee_z_strengths.py')

# # Alternatively load from json
# line = xt.load('../lattices/z/fccee_z.json').fccee_p_ring

line = env['fccee_p_ring']

# Survey
sv = line.survey()
sv.plot()
