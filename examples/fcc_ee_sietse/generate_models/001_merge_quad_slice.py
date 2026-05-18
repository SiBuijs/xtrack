import argparse
import xtrack as xt
import xobjects as xo

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

temp_folder = f'../lattices/{configuration}/_temp'

env = xt.load(f'{temp_folder}/env_no_bpms_moved_multipoles.json')

for nn in list(env.lines.keys()):
    if nn != 'fccee_p_ring':
        del env[nn]

line = env.fccee_p_ring

#tw4d = line.twiss4d()
#print("Tunes after initial loading",tw4d.qx,tw4d.qy)

line.replace_all_repeated_elements()

tt = line.get_table()
circumference = tt['s'][-1]

mask_ip_imag = tt.rows.mask['ipimag.*']
mask_ips = (tt.rows.mask['ip.*'] & ~mask_ip_imag)

tt_no_markers = tt.rows[(tt.element_type != 'Marker') | mask_ips]

to_delete = []

# Identify consecutive couples of quadrupoles
for ii in range(len(tt_no_markers)-1):
    if (tt_no_markers.element_type[ii] == 'Quadrupole'
        and tt_no_markers.element_type[ii+1] == 'Quadrupole'):

        nn1 = tt_no_markers.name[ii]
        nn2 = tt_no_markers.name[ii+1]

        if ((env.ref[nn1].k1._value == env.ref[nn2].k1._value)
            and (env.ref[nn1].length._value == env.ref[nn2].length._value)
            and (env.ref[nn1].k1._expr == env.ref[nn2].k1._expr)
            and (env.ref[nn1].length._expr == env.ref[nn2].length._expr)):

            lexpr = env.ref[nn2].length._expr
            lvalue = env.ref[nn2].length._value

            if lexpr is not None:
                env[nn1].length = env.ref[nn1].length._expr * 2
            else:
                env[nn1].length = lvalue * 2
            env[nn1].prototype = None
            to_delete.append(nn2)
# Identify consecutive couples of sextupoles. The first option covers a quartet of sextupoles     
for ii in range(len(tt_no_markers)-1):
    if (tt_no_markers.element_type[ii] == 'Sextupole'
        and tt_no_markers.element_type[ii+3] == 'Sextupole'):

        nn1 = tt_no_markers.name[ii]
        nn2 = tt_no_markers.name[ii+3]

        nn12 = tt_no_markers.name[ii+1]
        nn13 = tt_no_markers.name[ii+2]
        
        if ((env.ref[nn1].k2._value == env.ref[nn2].k2._value)
            and (env.ref[nn1].length._value == env.ref[nn2].length._value)
            and (env.ref[nn1].k2._expr == env.ref[nn2].k2._expr)
            and (env.ref[nn1].length._expr == env.ref[nn2].length._expr)):
            # print("joining sext quartet.")
            lexpr = env.ref[nn2].length._expr
            lvalue = env.ref[nn2].length._value

            if lexpr is not None:
                env[nn1].length = env.ref[nn1].length._expr * 4
            else:
                env[nn1].length = lvalue * 4
            env[nn1].prototype = None
            to_delete.append(nn2)
            to_delete.append(nn12)
            to_delete.append(nn13)
            
    elif (tt_no_markers.element_type[ii] == 'Sextupole'
        and tt_no_markers.element_type[ii+1] == 'Sextupole'):

        nn1 = tt_no_markers.name[ii]
        nn2 = tt_no_markers.name[ii+1]

        if ((env.ref[nn1].k2._value == env.ref[nn2].k2._value)
            and (env.ref[nn1].length._value == env.ref[nn2].length._value)
            and (env.ref[nn1].k2._expr == env.ref[nn2].k2._expr)
            and (env.ref[nn1].length._expr == env.ref[nn2].length._expr)):
            # print("joining sext doublet.")
            lexpr = env.ref[nn2].length._expr
            lvalue = env.ref[nn2].length._value

            if lexpr is not None:
                env[nn1].length = env.ref[nn1].length._expr * 2
            else:
                env[nn1].length = lvalue * 2
            env[nn1].prototype = None
            to_delete.append(nn2)  

new_line = env.new_line(length=circumference, compose=True)

tt_no_marker_no_drift = tt_no_markers.rows[tt_no_markers.element_type != 'Drift']
assert tt_no_marker_no_drift.name[-1] == '_end_point'
tt_no_marker_no_drift = tt_no_marker_no_drift.rows[:-1]

for nn in tt_no_marker_no_drift.name:
    if nn in to_delete:
        continue
    new_line.place(nn, at=tt_no_marker_no_drift['s_start', nn], anchor='start')

new_line.end_compose()
new_line.particle_ref = line.particle_ref

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

# we identify IPA (the one before the injection) and cycle there // MJ: not needed. already IP upstream of INJ
#tt = new_line.get_table()
#if configuration == 'z':
#    xo.assert_allclose(tt['s', 'ip.1'], 34529.5, atol=1.)
#    new_line.cycle('ip.1')
#elif configuration == 't':
#    xo.assert_allclose(tt['s', 'ip'], 34004.7, atol=1.)
#    new_line.cycle('ip')

del env['fccee_p_ring']
env['fccee_p_ring'] = new_line

env.to_json(temp_folder + '/env_no_bpms_merged_quads.json')
