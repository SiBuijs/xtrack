import xtrack as xt

env = xt.Environment()

# Load lattice from python files
env = xt.Environment()
env.call('../lattices/z/fccee_z_lattice.py')
env.call('../lattices/z/fccee_z_strengths.py')
line = env['fccee_p_ring']

line = env.fccee_p_ring.copy(shallow=True)

line.replace_all_repeated_elements()
tw = line.twiss4d()

tw_ng = line.madng_twiss(rdts=['f1200', 'f2100', 'f3000'])

tw_ng.plot('abs(f1200) abs(f2100) abs(f3000)')
