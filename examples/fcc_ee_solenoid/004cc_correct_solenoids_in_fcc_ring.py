from pathlib import Path

import matplotlib.pyplot as plt
import xtrack as xt


HERE = Path(__file__).parent

# 3 T SplineBoris solenoids, as installed by 004b_install_solenoids_in_fcc_ring.py.
INPUT_LATTICE_JSON = HERE / 'temp_fcc_ee_lcc_splineboris_solenoids_3T.json'



###################################
# Load installed solenoid lattice #
###################################

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring

# Bare ring: main and compensation solenoids off (the other IPs' solenoids are
# already off in the saved lattice).
line['on_sol_ipa'] = 0
line['on_comp_sol_ipa'] = 0


############################
# Ring twiss, optics at IP #
############################

tw0 = line.twiss4d(strengths=True)

print('Bare ring optics at ipa:')
for kk in ['betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx']:
    print(f'    {kk:5s} = {tw0[kk, "ipa"]: .6e}')


#################################################
# Line from the IP to the end of the straight   #
#################################################

line_ds = line.select(start='ipa', end='end_straight_start_ds_ipa', name='ipa_ds')
if line_ds.particle_ref is None:
    line_ds.particle_ref = line.particle_ref.copy()

tw_no_solenoid = line_ds.twiss4d(init=tw0.get_twiss_init('ipa'), strengths=True)

print(f'\nLine ipa -> EoS: '
      f'length = {line_ds.get_length():.4f} m')
print('Optics at EoS:')
for kk in ['betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx']:
    print(f'    {kk:5s}  line = {tw_no_solenoid[kk, "end_straight_start_ds_ipa"]: .6e}   '
          f'ring = {tw0[kk, "end_straight_start_ds_ipa"]: .6e}')


##########
# Plot   #
##########

plt.close('all')
tw_no_solenoid.plot()
plt.axvline(x=tw_no_solenoid.rows['sdm1r.0']['s'][0], color='k', ls='--', lw=1)
plt.suptitle(f'Bare optics IPa to EoS (3T)')
plt.show(block=False)



####################################################
# Correction configuration copied from 005g setup #
####################################################

quad_for_optics_correction = [
    'qd0ar.0', 'qd0br.0', 'qd0cr.0', 'qf1ar.0', 'qf1br.0',
    'qf1cr.0', 'qf1dr.0', 'qf2r.0', 'qd3r.0', 'qd4r.0',
    'qf5r.0', 'qd6r.0', 'qd6l.3', 'qf5l.3', 'qd4l.3',
    'qd3l.3', 'qf2l.3', 'qf1dl.3', 'qf1cl.3', 'qf1bl.3',
    'qf1al.3', 'qd0cl.3', 'qd0bl.3', 'qd0al.3',
]
doublet_quad_left = [
    'qd0al.3', 'qd0bl.3', 'qd0cl.3', 'qf1al.3', 'qf1bl.3',
    'qf1cl.3', 'qf1dl.3',
]
doublet_quad_right = [
    'qd0ar.0', 'qd0br.0', 'qd0cr.0', 'qf1ar.0', 'qf1br.0',
    'qf1cr.0', 'qf1dr.0',
]
corr_1_right_on_quad = 'qd0ar.0'
corr_2_right_on_quad = 'qd0br.0'
corr_3_right_on_quad = 'qf1ar.0'
corr_4_right_on_quad = 'qf1br.0'
corr_1_left_on_quad = 'qd0al.3'
corr_2_left_on_quad = 'qd0bl.3'
corr_3_left_on_quad = 'qf1al.3'
corr_4_left_on_quad = 'qf1bl.3'

name_start = 'end_ds_start_straight_ipa'
name_end = 'end_straight_start_ds_ipa'


###########################
# Build correction at ipa #
###########################

# Turn on the solenoid system being corrected.
line['on_sol_ipa'] = 1
line['on_comp_sol_ipa'] = 1

tw_uncorrected = line_ds.twiss4d(init=tw0.get_twiss_init('ipa'), strengths=True)

# Integrated ksol of the main solenoid.
ksol_l_main_solenoid = 0.0
rigidity0 = line.particle_ref.rigidity0[0]
table_solenoid_region = line.get_table().rows['dy_match_l_ipa':'dy_match_r_ipa']
for nn in table_solenoid_region.name:
    if table_solenoid_region['element_type', nn] == 'SplineBoris':
        element = env.get(table_solenoid_region['env_name', nn])
        ksol_l_main_solenoid += (
            element.scale_b * element.bs[4] * element.length / rigidity0)

# Rotate the final doublets by half of the main-solenoid rotation.
env['phi_rot_doublet_ipa'] = (ksol_l_main_solenoid / 2) / 2
env['on_rot_doublet_left_ipa'] = 1
env['on_rot_doublet_right_ipa'] = 1
for nn in doublet_quad_left:
    env[nn].rot_s_rad = (
        +env.ref['phi_rot_doublet_ipa'] * env.ref['on_rot_doublet_left_ipa'])
for nn in doublet_quad_right:
    env[nn].rot_s_rad = (
        -env.ref['phi_rot_doublet_ipa'] * env.ref['on_rot_doublet_right_ipa'])

# Orbit corrector knobs. The first pair was installed inside the main
# solenoid by 004b; the others are attached here to nearby quadrupoles and
# to the dedicated compensation-solenoid correctors.
for side in ['right', 'left']:
    for ii in range(2, 7):
        env[f'acbh{ii}_sol_{side}_ipa'] = 0
        env[f'acbv{ii}_sol_{side}_ipa'] = 0

env[corr_1_right_on_quad].knl[0] += env.ref['acbh2_sol_right_ipa']
env[corr_2_right_on_quad].knl[0] += env.ref['acbh3_sol_right_ipa']
env[corr_3_right_on_quad].knl[0] += env.ref['acbh4_sol_right_ipa']
env[corr_4_right_on_quad].knl[0] += env.ref['acbh5_sol_right_ipa']
env['corr_sol_right_ipa'].knl[0] += env.ref['acbh6_sol_right_ipa']

env[corr_1_left_on_quad].knl[0] += env.ref['acbh2_sol_left_ipa']
env[corr_2_left_on_quad].knl[0] += env.ref['acbh3_sol_left_ipa']
env[corr_3_left_on_quad].knl[0] += env.ref['acbh4_sol_left_ipa']
env[corr_4_left_on_quad].knl[0] += env.ref['acbh5_sol_left_ipa']
env['corr_sol_left_ipa'].knl[0] += env.ref['acbh6_sol_left_ipa']

env[corr_1_right_on_quad].ksl[0] += env.ref['acbv2_sol_right_ipa']
env[corr_2_right_on_quad].ksl[0] += env.ref['acbv3_sol_right_ipa']
env[corr_3_right_on_quad].ksl[0] += env.ref['acbv4_sol_right_ipa']
env[corr_4_right_on_quad].ksl[0] += env.ref['acbv5_sol_right_ipa']
env['corr_sol_right_ipa'].ksl[0] += env.ref['acbv6_sol_right_ipa']

env[corr_1_left_on_quad].ksl[0] += env.ref['acbv2_sol_left_ipa']
env[corr_2_left_on_quad].ksl[0] += env.ref['acbv3_sol_left_ipa']
env[corr_3_left_on_quad].ksl[0] += env.ref['acbv4_sol_left_ipa']
env[corr_4_left_on_quad].ksl[0] += env.ref['acbv5_sol_left_ipa']
env['corr_sol_left_ipa'].ksl[0] += env.ref['acbv6_sol_left_ipa']

# Match orbit and vertical dispersion across the solenoid region.
opt_orbit = line.match_knob(
    knob_name='on_sol_orbit_corr_ipa',
    run=False,
    betx=tw0['betx', 'ipa'],
    bety=tw0['bety', 'ipa'],
    start='dy_match_l_ipa',
    end='dy_match_r_ipa',
    init_at='ipa',
    vary=xt.VaryList([
        'acbh1_sol_right_ipa', 'acbv1_sol_right_ipa',
        'acbh2_sol_right_ipa', 'acbh3_sol_right_ipa',
        'acbh4_sol_right_ipa', 'acbh5_sol_right_ipa',
        'acbh6_sol_right_ipa', 'acbv2_sol_right_ipa',
        'acbv3_sol_right_ipa', 'acbv4_sol_right_ipa',
        'acbv5_sol_right_ipa', 'acbv6_sol_right_ipa',
        'acbh1_sol_left_ipa', 'acbv1_sol_left_ipa',
        'acbh2_sol_left_ipa', 'acbh3_sol_left_ipa',
        'acbh4_sol_left_ipa', 'acbh5_sol_left_ipa',
        'acbh6_sol_left_ipa', 'acbv2_sol_left_ipa',
        'acbv3_sol_left_ipa', 'acbv4_sol_left_ipa',
        'acbv5_sol_left_ipa', 'acbv6_sol_left_ipa',
    ], step=1e-6),
    targets=[
        xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.END),
        xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.START),
    ])
opt_orbit.solve()

two = line.twiss(
    strengths=True,
    init_at='ipa',
    betx=tw0['betx', 'ipa'],
    bety=tw0['bety', 'ipa'],
)

# Match optics and horizontal dispersion with normal quadrupole trims.
k1_knobs = []
for nn in quad_for_optics_correction:
    nn_knob = f'k1_{nn}_sol_corr'
    env[nn_knob] = 0
    env[nn].k1 += env.ref[nn_knob]
    k1_knobs.append(nn_knob)

# Skew quadrupole knobs for the additional linear-coupling/vertical-
# dispersion correction. Use all quadrupoles from the left edge to the right
# edge of the straight.
table_for_skew = line.get_table().rows[name_start:name_end]
k1s_quads_for_coupling_correction = []
for element_type, env_name in zip(
        table_for_skew.element_type, table_for_skew.env_name):
    if (element_type == 'Quadrupole'
            and env_name not in k1s_quads_for_coupling_correction):
        k1s_quads_for_coupling_correction.append(env_name)

k1s_knobs = []
for nn in k1s_quads_for_coupling_correction:
    nn_knob = f'k1s_{nn}_sol_coupling_corr'
    env[nn_knob] = 0
    env[nn].k1s += env.ref[nn_knob]
    k1s_knobs.append(nn_knob)

opt_optics = line.match_knob(
    knob_name='on_sol_optics_corr_ipa',
    run=False,
    betx=tw0['betx', 'ipa'],
    bety=tw0['bety', 'ipa'],
    init_at='ipa',
    start=name_start,
    end=name_end,
    vary=xt.VaryList(k1_knobs, step=1e-6),
    targets=[
        xt.TargetSet(
            betx=tw0['betx', name_start],
            bety=tw0['bety', name_start],
            tol=1e-5,
            at=xt.START),
        xt.TargetSet(
            alfx=tw0['alfx', name_start],
            alfy=tw0['alfy', name_start],
            tol=1e-8,
            at=xt.START),
        xt.TargetSet(
            dx=tw0['dx', name_start],
            dpx=tw0['dpx', name_start],
            tol=1e-8,
            at=xt.START),
        xt.TargetSet(
            betx=tw0['betx', name_end],
            bety=tw0['bety', name_end],
            tol=1e-5,
            at=xt.END),
        xt.TargetSet(
            alfx=tw0['alfx', name_end],
            alfy=tw0['alfy', name_end],
            tol=1e-8,
            at=xt.END),
        xt.TargetSet(
            dx=tw0['dx', name_end],
            dpx=tw0['dpx', name_end],
            tol=1e-8,
            at=xt.END),
    ])
opt_optics.solve()



# Try an additional correction of linear coupling and vertical dispersion
# at the straight-section edges using the skew quadrupole knobs.
opt_coupling = line.match_knob(
    knob_name='on_sol_coupling_corr_ipa',
    run=False,
    betx=tw0['betx', 'ipa'],
    bety=tw0['bety', 'ipa'],
    init_at='ipa',
    start=name_start,
    end=name_end,
    vary=xt.VaryList(k1s_knobs, step=1e-6),
    targets=[
        xt.TargetSet(betx2=0, bety1=0, at=xt.START, tol=5e-5),
        xt.TargetSet(betx2=0, bety1=0, at=xt.END, tol=5e-5),
        xt.TargetSet(alfx2=0, alfy1=0, at=xt.START, tol=1e-6),
        xt.TargetSet(alfx2=0, alfy1=0, at=xt.END, tol=1e-6),
        xt.TargetSet(dy=0, at=xt.START, tol=5e-5),
        xt.TargetSet(dy=0, at=xt.END, tol=5e-5),
        xt.TargetSet(dpy=0, at=xt.START, tol=1e-7),
        xt.TargetSet(dpy=0, at=xt.END, tol=1e-7),
    ])

opt_coupling.solve()

# Iterate to improve consistency of orbit and optics corrections.
opt_orbit.solve()
opt_optics.solve()
opt_coupling.solve()
opt_orbit.solve()
opt_optics.solve()

opt_orbit.generate_knob()
opt_optics.generate_knob()
opt_coupling.generate_knob()

# One user knob turns on compensation solenoid, doublet rotations, and all
# generated correction knobs.
line['on_sol_corr_ipa'] = 0
line['on_comp_sol_ipa'] = 'on_sol_corr_ipa'
line['on_rot_doublet_right_ipa'] = 'on_sol_corr_ipa'
line['on_rot_doublet_left_ipa'] = 'on_sol_corr_ipa'
line['on_sol_orbit_corr_ipa'] = 'on_sol_corr_ipa'
line['on_sol_optics_corr_ipa'] = 'on_sol_corr_ipa'
line['on_sol_coupling_corr_ipa'] = 'on_sol_corr_ipa'

# Leave the main solenoid off.
line['on_sol_ipa'] = 0
