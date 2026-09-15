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


#####################################################
# Mid-bend quadrupoles for chromaticity correction  #
#####################################################

# Cut the first three bends downstream of ipa in half and put a thin
# quadrupole at each cut. Only line_ds is modified, the ring keeps whole bends.
bends_for_mid_quad = ['b1ra.0', 'b1rb.0', 'b1rc.0']

table_ds = line_ds.get_table()
line_ds.cut_at_s([table_ds['s_center', nn] for nn in bends_for_mid_quad])

#The knob is k1l [1/m].
quad_for_chromaticity_correction = []
mid_quad_names = []
mid_quad_places = []
for bend_name in bends_for_mid_quad:
    quad_name = f'qbmid_{bend_name}'
    knob_name = f'k1l_{quad_name}_chrom_corr'
    env.elements[quad_name] = xt.Multipole(knl=[0.0, 0.0], length=0.0)
    env[knob_name] = 0
    env[quad_name].knl[1] = env.ref[knob_name]
    quad_for_chromaticity_correction.append(knob_name)
    mid_quad_names.append(quad_name)
    mid_quad_places.append(env.place(
        quad_name, at=0, from_=f'{bend_name}..1',
        anchor='start', from_anchor='start'))

line_ds.insert(mid_quad_places)

####################################################
# Correction configuration copied from 005g setup #
####################################################

# Only the elements downstream of ipa, i.e. those inside line_ds.
quad_for_optics_correction = [
    'qd0ar.0', 'qd0br.0', 'qd0cr.0', 'qf1ar.0', 'qf1br.0',
    'qf1cr.0', 'qf1dr.0', 'qf2r.0', 'qd3r.0', 'qd4r.0',
    'qf5r.0', 'qd6r.0',
]
doublet_quad_right = [
    'qd0ar.0', 'qd0br.0', 'qd0cr.0', 'qf1ar.0', 'qf1br.0',
    'qf1cr.0', 'qf1dr.0',
]
corr_1_right_on_quad = 'qd0ar.0'
corr_2_right_on_quad = 'qd0br.0'
corr_3_right_on_quad = 'qf1ar.0'
corr_4_right_on_quad = 'qf1br.0'

name_end = 'end_straight_start_ds_ipa'

# All matches start at ipa (the start of line_ds) with the bare-ring optics.
init_ipa = tw0.get_twiss_init('ipa')


###########################
# Build correction at ipa #
###########################

# Turn on the solenoid system being corrected.
line['on_sol_ipa'] = 1
line['on_comp_sol_ipa'] = 1

tw_uncorrected = line_ds.twiss4d(init=init_ipa, strengths=True)

# Integrated ksol of the full main solenoid (both sides of the IP).
ksol_l_main_solenoid = 0.0
rigidity0 = line.particle_ref.rigidity0[0]
table_solenoid_region = line.get_table().rows['dy_match_l_ipa':'dy_match_r_ipa']
for nn in table_solenoid_region.name:
    if table_solenoid_region['element_type', nn] == 'SplineBoris':
        element = env.get(table_solenoid_region['env_name', nn])
        ksol_l_main_solenoid += (
            element.scale_b * element.bs[4] * element.length / rigidity0)

# Rotate the final doublet by half of the main-solenoid rotation.
env['phi_rot_doublet_ipa'] = (ksol_l_main_solenoid / 2) / 2
env['on_rot_doublet_right_ipa'] = 1
for nn in doublet_quad_right:
    env[nn].rot_s_rad = (
        -env.ref['phi_rot_doublet_ipa'] * env.ref['on_rot_doublet_right_ipa'])

# Orbit corrector knobs. The first pair was installed inside the main
# solenoid by 004b; the others are attached here to nearby quadrupoles and
# to the dedicated compensation-solenoid corrector.
for ii in range(2, 7):
    env[f'acbh{ii}_sol_right_ipa'] = 0
    env[f'acbv{ii}_sol_right_ipa'] = 0

env[corr_1_right_on_quad].knl[0] += env.ref['acbh2_sol_right_ipa']
env[corr_2_right_on_quad].knl[0] += env.ref['acbh3_sol_right_ipa']
env[corr_3_right_on_quad].knl[0] += env.ref['acbh4_sol_right_ipa']
env[corr_4_right_on_quad].knl[0] += env.ref['acbh5_sol_right_ipa']
env['corr_sol_right_ipa'].knl[0] += env.ref['acbh6_sol_right_ipa']

env[corr_1_right_on_quad].ksl[0] += env.ref['acbv2_sol_right_ipa']
env[corr_2_right_on_quad].ksl[0] += env.ref['acbv3_sol_right_ipa']
env[corr_3_right_on_quad].ksl[0] += env.ref['acbv4_sol_right_ipa']
env[corr_4_right_on_quad].ksl[0] += env.ref['acbv5_sol_right_ipa']
env['corr_sol_right_ipa'].ksl[0] += env.ref['acbv6_sol_right_ipa']

# Match orbit and vertical dispersion at the downstream end of the solenoid
# region.
opt_orbit = line_ds.match_knob(
    knob_name='on_sol_orbit_corr_ipa',
    run=False,
    assert_within_tol=False,
    init=init_ipa,
    start='ipa',
    end='dy_match_r_ipa',
    vary=xt.VaryList([
        'acbh1_sol_right_ipa', 'acbv1_sol_right_ipa',
        'acbh2_sol_right_ipa', 'acbh3_sol_right_ipa',
        'acbh4_sol_right_ipa', 'acbh5_sol_right_ipa',
        'acbh6_sol_right_ipa', 'acbv2_sol_right_ipa',
        'acbv3_sol_right_ipa', 'acbv4_sol_right_ipa',
        'acbv5_sol_right_ipa', 'acbv6_sol_right_ipa',
    ], step=1e-6),
    targets=[
        xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.END),
    ])

print("\nOrbit correction:")
opt_orbit.solve()

two = line_ds.twiss(strengths=True, init=init_ipa)

# Match optics and horizontal dispersion with normal quadrupole trims.
k1_knobs = []
for nn in quad_for_optics_correction:
    nn_knob = f'k1_{nn}_sol_corr'
    env[nn_knob] = 0
    env[nn].k1 += env.ref[nn_knob]
    k1_knobs.append(nn_knob)

# Skew quadrupole knobs for the additional linear-coupling/vertical-
# dispersion correction. Use all quadrupoles in line_ds.
table_for_skew = line_ds.get_table()
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

sext_for_chromaticity_correction = 'sdm1r.0'

opt_optics = line_ds.match_knob(
    knob_name='on_sol_optics_corr_ipa',
    run=False,
    assert_within_tol=False,
    init=init_ipa,
    start='ipa',
    end=name_end,
    vary=[xt.VaryList(k1_knobs, tag='main', step=1e-6),
          xt.VaryList(quad_for_chromaticity_correction, tag='added', step=1e-6)],
    targets=[
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
        xt.TargetSet(
            betx=tw0['betx', sext_for_chromaticity_correction],
            bety=tw0['bety', sext_for_chromaticity_correction],
            tol=1e-5,
            tag='sext',
            at=sext_for_chromaticity_correction),

    ])
print("\nOptics correction:")
opt_optics.solve()



# Try an additional correction of linear coupling and vertical dispersion
# at the end of the straight using the skew quadrupole knobs.
opt_coupling = line_ds.match_knob(
    knob_name='on_sol_coupling_corr_ipa',
    run=False,
    assert_within_tol=False,
    init=init_ipa,
    start='ipa',
    end=name_end,
    vary=xt.VaryList(k1s_knobs, step=1e-6),
    targets=[
        xt.TargetSet(betx2=0, bety1=0, at=xt.END, tol=5e-5),
        xt.TargetSet(alfx2=0, alfy1=0, at=xt.END, tol=1e-6),
        xt.TargetSet(dy=0, at=xt.END, tol=5e-5),
        xt.TargetSet(dpy=0, at=xt.END, tol=1e-7),
    ])

print("\nCoupling correction:")
opt_coupling.solve(rcond=3e-3)

# Iterate to improve consistency of orbit and optics corrections.
print("\nOrbit correction:")
opt_orbit.solve()
print("\nOptics correction:")
opt_optics.solve()
print("\nCoupling correction:")
opt_coupling.solve(rcond=3e-3)
print("\nOrbit correction:")
opt_orbit.solve()
print("\nOptics correction:")
opt_optics.solve()

opt_orbit.generate_knob()
opt_optics.generate_knob()
opt_coupling.generate_knob()

# One user knob turns on compensation solenoid, doublet rotation, and all
# generated correction knobs.
line['on_sol_corr_ipa'] = 0
line['on_comp_sol_ipa'] = 'on_sol_corr_ipa'
line['on_rot_doublet_right_ipa'] = 'on_sol_corr_ipa'
line['on_sol_orbit_corr_ipa'] = 'on_sol_corr_ipa'
line['on_sol_optics_corr_ipa'] = 'on_sol_corr_ipa'
line['on_sol_coupling_corr_ipa'] = 'on_sol_corr_ipa'
line['on_sol_chrom_corr_ipa'] = 'on_sol_corr_ipa'


#####################
# Corrected optics  #
#####################

line['on_sol_ipa'] = 1
line['on_sol_corr_ipa'] = 1

tw_corrected = line_ds.twiss4d(init=init_ipa, strengths=True)

tw_corrected.plot()
plt.axvline(x=tw_corrected.rows['sdm1r.0']['s'][0], color='k', lw=1.5)
for optics_quad in quad_for_optics_correction:
    plt.axvline(x=tw_corrected.rows[optics_quad]['s'][0], color='b', ls='-.', lw=1.5, alpha=0.5)
for coupling_quad in k1s_quads_for_coupling_correction:
    plt.axvline(x=tw_corrected.rows[coupling_quad]['s'][0], color='r', ls=':', lw=1.5, alpha=0.5)
for corr_quad in [corr_1_right_on_quad, corr_2_right_on_quad, corr_3_right_on_quad, corr_4_right_on_quad]:
    plt.axvline(x=tw_corrected.rows[corr_quad]['s'][0], color='g', ls='--', lw=1.5, alpha=0.5)
for mid_quad in mid_quad_names:
    plt.axvline(x=tw_corrected.rows[mid_quad]['s'][0], color='m', ls='--', lw=1.5, alpha=0.5)
plt.suptitle(f'Corrected optics IPa to EoS (3T)')
plt.show(block=False)