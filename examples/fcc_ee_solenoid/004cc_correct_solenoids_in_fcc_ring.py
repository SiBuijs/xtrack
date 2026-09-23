from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt


HERE = Path(__file__).parent

# 3 T SplineBoris solenoids, as installed by 004b_install_solenoids_in_fcc_ring.py.
INPUT_LATTICE_JSON = HERE / 'temp_fcc_ee_lcc_splineboris_solenoids_3T.json'


############################################
# IP to work on -- change this one string  #
############################################

# The IR is repeated at all four IPs: the element names carry a numeric suffix
# (qd0ar.0 at ipa, .1 at ipd, .2 at ipg, .3 at ipj) and the knobs carry the IP
# name. Everything below is derived from IP_NAME, so this is the only line to
# edit.
#
# Note that only the RIGHT half straight is built here, from the IP to the end
# of the straight downstream. In the full 004c correction the large QD0 -> sdy1
# phase errors sat on the LEFT sides (ipj left, +7.1e-4), so changing this
# string does not by itself reproduce the ipj Q'' blowup. See
# claude_notes/08_second_order_chromaticity_source.md.
IP_NAME = 'ipa'

# Element families of the half straight, in beam order; resolved to the actual
# names for this IP once the line is cut, by elements_in_straight() below.
QUAD_FAMILIES_FOR_OPTICS = (
    'qd0ar', 'qd0br', 'qd0cr', 'qf1ar', 'qf1br', 'qf1cr', 'qf1dr',
    'qf2r', 'qd3r', 'qd4r', 'qf5r', 'qd6r')
N_DOUBLET_QUADS = 7      # the first N of the list above are the FF doublet
CORR_QUAD_FAMILIES = ('qd0ar', 'qd0br', 'qf1ar', 'qf1br')
BEND_FAMILIES_FOR_MID_QUAD = ('b1ra', 'b1rb', 'b1rc')
SEXT_FAMILY_FOR_CHROM = 'sdm1r'
SEXT_FAMILY_FOR_LCC = 'sdy1r'    # -I partner of the doublet, phase target below

# Calibrated in the full 004c ring with qy trombones (note 08): the change in
# ring Q''y per unit of QD0 -> sdy1 vertical phase error on one side of one IP,
# in tune units. Used only to turn the measured phase error into an expected
# Q''y, so it is an order-of-magnitude check, not a prediction of this line.
DQ2Y_PER_PHASE_ERROR = 2.42e7

name_end = f'end_straight_start_ds_{IP_NAME}'


###################################
# Load installed solenoid lattice #
###################################

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring

# Bare ring: main and compensation solenoids off (the other IPs' solenoids are
# already off in the saved lattice).
line[f'on_sol_{IP_NAME}'] = 0
line[f'on_comp_sol_{IP_NAME}'] = 0


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

line_ds = line.select(start=IP_NAME, end=name_end, name=f'{IP_NAME}_ds')
if line_ds.particle_ref is None:
    line_ds.particle_ref = line.particle_ref.copy()


def elements_in_straight(*families):
    """Element names of the given families inside this half straight.

    Once the line is cut to one half straight, the family name (the element
    name without its trailing IP suffix) identifies the element. Families with
    more than one element in the straight -- sdm1r has two -- resolve to the
    first in beam order, which is the one 004c targets.
    """
    first_by_family = {}
    for nn in line_ds.get_table().name:
        first_by_family.setdefault(nn.rsplit('.', 1)[0], nn)
    missing = [ff for ff in families if ff not in first_by_family]
    if missing:
        raise SystemExit(f'{missing} not found in the {IP_NAME} half straight: '
                         'the IR element naming must have changed.')
    return [first_by_family[ff] for ff in families]


quad_for_optics_correction = elements_in_straight(*QUAD_FAMILIES_FOR_OPTICS)
doublet_quad_right = quad_for_optics_correction[:N_DOUBLET_QUADS]
(corr_1_right_on_quad, corr_2_right_on_quad,
 corr_3_right_on_quad, corr_4_right_on_quad) = elements_in_straight(
     *CORR_QUAD_FAMILIES)
bends_for_mid_quad = elements_in_straight(*BEND_FAMILIES_FOR_MID_QUAD)
sext_for_chromaticity_correction, = elements_in_straight(SEXT_FAMILY_FOR_CHROM)
sext_for_lcc, = elements_in_straight(SEXT_FAMILY_FOR_LCC)

# chrom=True adds the Montague chromatic functions (wx_chrom, wy_chrom), which
# the optics match below targets at the end of the straight. Their bare values
# are the reference: the local chromatic correction brings W_y from ~5e3 inside
# the doublet back to ~2 here, and the whole point of the target is that the
# corrected lattice must do the same.
tw_no_solenoid = line_ds.twiss4d(init=tw0.get_twiss_init(IP_NAME),
                                 strengths=True, chrom=True)
wx_chrom_bare_end = tw_no_solenoid.wx_chrom[-1]
wy_chrom_bare_end = tw_no_solenoid.wy_chrom[-1]

print(f'\nLine {IP_NAME} -> EoS: '
      f'length = {line_ds.get_length():.4f} m')
print('Optics at EoS:')
for kk in ['betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx']:
    print(f'    {kk:5s}  line = {tw_no_solenoid[kk, name_end]: .6e}   '
          f'ring = {tw0[kk, name_end]: .6e}')


##########
# Plot   #
##########

plt.close('all')
tw_no_solenoid.plot()
plt.axvline(x=tw_no_solenoid.rows[sext_for_chromaticity_correction]['s'][0],
            color='k', ls='--', lw=1)
plt.suptitle(f'Bare optics {IP_NAME} to EoS (3T)')
plt.show(block=False)


#####################################################
# Mid-bend quadrupoles for chromaticity correction  #
#####################################################

# Cut the first three bends downstream of the IP in half and put a thin
# quadrupole at each cut. Only line_ds is modified, the ring keeps whole bends.
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

# All matches start at the IP (the start of line_ds) with the bare-ring optics.
init_ip = tw0.get_twiss_init(IP_NAME)


##########################
# Build correction at IP #
##########################

# Turn on the solenoid system being corrected.
line[f'on_sol_{IP_NAME}'] = 1
line[f'on_comp_sol_{IP_NAME}'] = 1

tw_uncorrected = line_ds.twiss4d(init=init_ip, strengths=True)

# Integrated ksol of the full main solenoid (both sides of the IP).
ksol_l_main_solenoid = 0.0
rigidity0 = line.particle_ref.rigidity0[0]
table_solenoid_region = line.get_table().rows[f'dy_match_l_{IP_NAME}':f'dy_match_r_{IP_NAME}']
for nn in table_solenoid_region.name:
    if table_solenoid_region['element_type', nn] == 'SplineBoris':
        element = env.get(table_solenoid_region['env_name', nn])
        ksol_l_main_solenoid += (
            element.scale_b * element.bs[4] * element.length / rigidity0)

# Rotate the final doublet by half of the main-solenoid rotation.
env[f'phi_rot_doublet_{IP_NAME}'] = (ksol_l_main_solenoid / 2) / 2
env[f'on_rot_doublet_right_{IP_NAME}'] = 1
for nn in doublet_quad_right:
    env[nn].rot_s_rad = (
        -env.ref[f'phi_rot_doublet_{IP_NAME}'] * env.ref[f'on_rot_doublet_right_{IP_NAME}'])

# Orbit corrector knobs. The first pair was installed inside the main
# solenoid by 004b; the others are attached here to nearby quadrupoles and
# to the dedicated compensation-solenoid corrector.
for ii in range(2, 7):
    env[f'acbh{ii}_sol_right_{IP_NAME}'] = 0
    env[f'acbv{ii}_sol_right_{IP_NAME}'] = 0

env[corr_1_right_on_quad].knl[0] += env.ref[f'acbh2_sol_right_{IP_NAME}']
env[corr_2_right_on_quad].knl[0] += env.ref[f'acbh3_sol_right_{IP_NAME}']
env[corr_3_right_on_quad].knl[0] += env.ref[f'acbh4_sol_right_{IP_NAME}']
env[corr_4_right_on_quad].knl[0] += env.ref[f'acbh5_sol_right_{IP_NAME}']
env[f'corr_sol_right_{IP_NAME}'].knl[0] += env.ref[f'acbh6_sol_right_{IP_NAME}']

env[corr_1_right_on_quad].ksl[0] += env.ref[f'acbv2_sol_right_{IP_NAME}']
env[corr_2_right_on_quad].ksl[0] += env.ref[f'acbv3_sol_right_{IP_NAME}']
env[corr_3_right_on_quad].ksl[0] += env.ref[f'acbv4_sol_right_{IP_NAME}']
env[corr_4_right_on_quad].ksl[0] += env.ref[f'acbv5_sol_right_{IP_NAME}']
env[f'corr_sol_right_{IP_NAME}'].ksl[0] += env.ref[f'acbv6_sol_right_{IP_NAME}']

# Match orbit and vertical dispersion at the downstream end of the solenoid
# region.
opt_orbit = line_ds.match_knob(
    knob_name=f'on_sol_orbit_corr_{IP_NAME}',
    run=False,
    assert_within_tol=False,
    init=init_ip,
    start=IP_NAME,
    end=f'dy_match_r_{IP_NAME}',
    vary=xt.VaryList([
        f'acbh1_sol_right_{IP_NAME}', f'acbv1_sol_right_{IP_NAME}',
        f'acbh2_sol_right_{IP_NAME}', f'acbh3_sol_right_{IP_NAME}',
        f'acbh4_sol_right_{IP_NAME}', f'acbh5_sol_right_{IP_NAME}',
        f'acbh6_sol_right_{IP_NAME}', f'acbv2_sol_right_{IP_NAME}',
        f'acbv3_sol_right_{IP_NAME}', f'acbv4_sol_right_{IP_NAME}',
        f'acbv5_sol_right_{IP_NAME}', f'acbv6_sol_right_{IP_NAME}',
    ], step=1e-6),
    targets=[
        xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.END),
    ])

print("\nOrbit correction:")
opt_orbit.solve()

two = line_ds.twiss(strengths=True, init=init_ip)

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

# Tolerance on the W targets. A leak of dW past the end of the straight costs
# roughly q_F/(4 pi) * dW of ring Q''y, with q_F ~ 5500 the doublet's chromatic
# kick, so dW = 0.1 is worth about 45 units of Q''y per side. See
# claude_notes/08_second_order_chromaticity_source.md.
W_CHROM_TOL = 0.1

# chrom=True makes every merit evaluation compute the chromatic functions, i.e.
# three twisses of the half straight instead of one. That buys the two W targets
# below, which constrain the chromatic mismatch the straight hands to the arcs
# directly, rather than through the QD0 -> sdy1 phase that causes it. Targeting
# |W| also covers a sextupole *strength* mismatch, which the phase does not:
# beta_y and D_x at the sdy sextupoles are not targeted anywhere in this match.
# Note that |W| is a magnitude, so this bounds the leak without pinning its
# phase -- enough, since Q'' is bounded by the magnitude.
opt_optics = line_ds.match_knob(
    knob_name=f'on_sol_optics_corr_{IP_NAME}',
    run=False,
    assert_within_tol=False,
    init=init_ip,
    start=IP_NAME,
    end=name_end,
    chrom=True,
    vary=[xt.VaryList(k1_knobs, tag='main', step=1e-6),
          xt.VaryList(quad_for_chromaticity_correction, tag='added', step=1e-6)],
    targets=[
        xt.TargetSet(
            wx_chrom=wx_chrom_bare_end,
            wy_chrom=wy_chrom_bare_end,
            tol=W_CHROM_TOL,
            tag='wchrom',
            at=xt.END),
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
            dy=tw0['dy', sext_for_chromaticity_correction],
            tol=1e-5,
            tag='sext',
            at=sext_for_chromaticity_correction),

    ])
# Two stages. W is unbounded and strongly nonlinear: in the trial states the
# solver visits on its way from the uncorrected optics, wy_chrom at the end of
# the straight runs to 1e6 and more, and it then dominates the merit function
# and drives the step into nonsense (a single-stage solve wanders to a penalty
# of 1e8 and take_best hands back the starting point, i.e. no correction at
# all). Converging the linear targets first puts the solver in a region where W
# is O(10) and well behaved, and it also makes the failure mode safe: the worst
# take_best can now do is return the linear solution.
# The staged solve itself is in solve_correction_chain() below, which is run
# once per sextupole-target phase (see the two phases after it).


# Try an additional correction of linear coupling and vertical dispersion
# at the end of the straight using the skew quadrupole knobs.
opt_coupling = line_ds.match_knob(
    knob_name=f'on_sol_coupling_corr_{IP_NAME}',
    run=False,
    assert_within_tol=False,
    init=init_ip,
    start=IP_NAME,
    end=name_end,
    vary=xt.VaryList(k1s_knobs, step=1e-6),
    targets=[
        xt.TargetSet(betx2=0, bety1=0, at=xt.END, tol=5e-5),
        xt.TargetSet(alfx2=0, alfy1=0, at=xt.END, tol=1e-6),
        xt.TargetSet(dy=0, at=xt.END, tol=5e-5),
        xt.TargetSet(dpy=0, at=xt.END, tol=1e-7),
    ])

VARY_KNOB_NAMES = [vv.name for opt in (opt_orbit, opt_optics, opt_coupling)
                   for vv in opt.vary]


def reset_vary_knobs():
    """Put every correction vary knob back to zero, i.e. back to the
    uncorrected solenoid optics, so that the next chain starts from the same
    place the first one did."""
    for nn in VARY_KNOB_NAMES:
        line.vars[nn] = 0.0


def solve_correction_chain(label, optics_rcond=None):
    """One full orbit / optics / coupling correction chain, in the order the
    script has always used.

    The optics solve inside it is itself staged on the 'wchrom' targets, for
    the reason given at the optics match above: W is unbounded and has to be
    approached from a converged linear solution.

    `optics_rcond` truncates the small singular values of the optics Jacobian,
    as 004c's OPTICS_RCOND does. Only used for the sext-targets-off chain,
    which is the more underdetermined of the two (8 targets against 15 knobs
    instead of 11) and the one that otherwise drifts off its boundary targets.
    """
    optics_solve_kwargs = ({} if optics_rcond is None
                           else {'rcond': optics_rcond})
    print(f'\n[{label}] Orbit correction:')
    opt_orbit.solve()
    print(f'\n[{label}] Optics correction (stage 1: linear targets only):')
    opt_optics.disable(target='wchrom')
    opt_optics.solve(**optics_solve_kwargs)
    print(f'\n[{label}] Optics correction (stage 2: with the W targets):')
    opt_optics.enable(target='wchrom')
    opt_optics.solve(**optics_solve_kwargs)
    opt_optics.target_status()
    print(f'\n[{label}] Coupling correction:')
    opt_coupling.solve(rcond=3e-3)
    # Iterate to improve consistency of orbit and optics corrections.
    print(f'\n[{label}] Orbit correction:')
    opt_orbit.solve()
    print(f'\n[{label}] Optics correction:')
    opt_optics.solve(**optics_solve_kwargs)
    print(f'\n[{label}] Coupling correction:')
    opt_coupling.solve(rcond=3e-3)
    print(f'\n[{label}] Orbit correction:')
    opt_orbit.solve()
    print(f'\n[{label}] Optics correction:')
    opt_optics.solve(**optics_solve_kwargs)
    opt_optics.target_status()


##############################################################
# The correction, solved with and without the sext targets   #
##############################################################

# Two INDEPENDENT chains, each started from all vary knobs at zero, i.e. from
# the uncorrected solenoid optics. They differ in exactly one thing: whether
# the 'sext' targets (betx/bety/dy at sdm1r, the chromatic sextupole nearest
# the IP) are active. Chain A matches only the straight boundary (plus W);
# chain B matches the sextupole as well.
#
# They must be independent rather than sequential. Running them as two stages
# of one solve -- boundary-only first, then continuing with the sext targets
# enabled, the way 004c's OPTICS_STAGE_SEXT does on the ring -- was tried here
# (2026-09-21) and FAILS badly: the boundary-only solution (bety ~ 537 m at
# sdm1r) is a trap the solver never gets out of, and the final state ends 72%
# off on betx at the straight end (276 against 160) with wy_chrom at 127
# instead of 2, where solving with the sext targets on from the start
# converges them all. Whatever makes the staging work in 004c, it does not
# carry over to this match, which also carries the W targets.
#
# Chain B is therefore the original code path, and is what the generated knobs
# and everything downstream come from. Chain A exists only for the comparison
# plot below.

print('\n' + '=' * 70)
print('Chain A: optics match WITHOUT the sextupole targets (straight edges '
      'only)')
print('=' * 70)
reset_vary_knobs()
opt_optics.disable(target='sext')
# See solve_correction_chain: the boundary-only problem is the more
# underdetermined of the two, and with the xdeps default rcond it drifts off
# the very targets it is supposed to be holding (betx at the straight end
# 1.4% off in the 2026-09-21 run), which would make the comparison at the
# sextupole impossible to attribute. Truncating as 004c does keeps it on them.
solve_correction_chain('sext targets off', optics_rcond=1e-6)
tw_corrected_no_sext = line_ds.twiss4d(init=init_ip, strengths=True)

print('\n' + '=' * 70)
print('Chain B: the same correction WITH the sextupole targets (the one the '
      'generated knobs come from)')
print('=' * 70)
reset_vary_knobs()
opt_optics.enable(target='sext')
solve_correction_chain('sext targets on')

opt_orbit.generate_knob()
opt_optics.generate_knob()
opt_coupling.generate_knob()

# One user knob turns on compensation solenoid, doublet rotation, and all
# generated correction knobs.
line[f'on_sol_corr_{IP_NAME}'] = 0
line[f'on_comp_sol_{IP_NAME}'] = f'on_sol_corr_{IP_NAME}'
line[f'on_rot_doublet_right_{IP_NAME}'] = f'on_sol_corr_{IP_NAME}'
line[f'on_sol_orbit_corr_{IP_NAME}'] = f'on_sol_corr_{IP_NAME}'
line[f'on_sol_optics_corr_{IP_NAME}'] = f'on_sol_corr_{IP_NAME}'
line[f'on_sol_coupling_corr_{IP_NAME}'] = f'on_sol_corr_{IP_NAME}'
line[f'on_sol_chrom_corr_{IP_NAME}'] = f'on_sol_corr_{IP_NAME}'


#####################
# Corrected optics  #
#####################

line[f'on_sol_{IP_NAME}'] = 1
line[f'on_sol_corr_{IP_NAME}'] = 1

tw_corrected = line_ds.twiss4d(init=init_ip, strengths=True)

tw_corrected.plot()
plt.axvline(x=tw_corrected.rows[sext_for_chromaticity_correction]['s'][0],
            color='k', lw=1.5)
for optics_quad in quad_for_optics_correction:
    plt.axvline(x=tw_corrected.rows[optics_quad]['s'][0], color='b', ls='-.', lw=1.5, alpha=0.5)
for coupling_quad in k1s_quads_for_coupling_correction:
    plt.axvline(x=tw_corrected.rows[coupling_quad]['s'][0], color='r', ls=':', lw=1.5, alpha=0.5)
for corr_quad in [corr_1_right_on_quad, corr_2_right_on_quad, corr_3_right_on_quad, corr_4_right_on_quad]:
    plt.axvline(x=tw_corrected.rows[corr_quad]['s'][0], color='g', ls='--', lw=1.5, alpha=0.5)
for mid_quad in mid_quad_names:
    plt.axvline(x=tw_corrected.rows[mid_quad]['s'][0], color='m', ls='--', lw=1.5, alpha=0.5)
plt.suptitle(f'Corrected optics {IP_NAME} to EoS (3T)')
plt.show(block=False)


###########################################################
# Beta functions with and without the sextupole targets   #
###########################################################

# What the 'sext' targets buy, read at the sextupole they target and along the
# whole half straight. Linear y, as in the other beta figures of this study.
# Note the dynamic range this costs: beta runs from the IP waist (bety* =
# 0.7 mm) to ~1.5e4 m in the doublet, so on a linear axis the waist and the
# whole low-beta region sit on the baseline and only the peaks are legible.
# The comparison this figure is for -- the three cases at the sextupole, where
# beta is a few m -- is read from the printed table above and from the bety
# separation in the IR panel, not from the waist.

s_sext = tw_corrected.rows[sext_for_chromaticity_correction]['s'][0]

# Default colours and widths, as in the other figures of this study. The two
# corrected cases lie on top of the bare one wherever the correction works, so
# they are separated by dash pattern rather than by weight: bare solid
# underneath, sext-corrected dashed, not-sext-corrected dotted.
BETA_CASES = (
    ('bare (no solenoid)', tw_no_solenoid, 'C0', '-'),
    ('corrected, sext targets off', tw_corrected_no_sext, 'C1', ':'),
    ('corrected, sext targets on', tw_corrected, 'C2', '--'),
)

print('\nOptics at the targeted sextupole '
      f'({sext_for_chromaticity_correction}):')
print(f'  {"case":30s} {"betx [m]":>12s} {"bety [m]":>12s} {"dy [m]":>12s}')
for case_label, tw_case, _, _ in BETA_CASES:
    print(f'  {case_label:30s} '
          f'{tw_case["betx", sext_for_chromaticity_correction]:12.5f} '
          f'{tw_case["bety", sext_for_chromaticity_correction]:12.5f} '
          f'{tw_case["dy", sext_for_chromaticity_correction]:12.3e}')

# Two s-ranges, same reasoning as 004d's BETA_COMPARISON_RANGES: the IR out to
# a little past the targeted sextupole is where the correction acts and where
# the two cases differ, while the full half straight shows that they have
# re-merged by the time the beam reaches the arcs.
BETA_PLOT_RANGES = (
    ((-2.0, 1.35 * s_sext), f'IR ({IP_NAME} to just past '
                            f'{sext_for_chromaticity_correction})'),
    (None, f'{IP_NAME} to EoS'),
)


def beta_ylim_for_xlim(plane, xlim, margin=1.05):
    """Y-limits from the data actually inside `xlim` -- on a shared
    full-straight axis the arc peaks would otherwise set the scale for the IR
    zoom too. Linear axis, so the bottom is pinned at 0 rather than at the
    smallest positive value."""
    hi = -np.inf
    for _, tw_case, _, _ in BETA_CASES:
        beta = np.asarray(tw_case[f'bet{plane}'])
        s_case = np.asarray(tw_case.s)
        mask = (np.ones_like(s_case, dtype=bool) if xlim is None
                else (s_case >= xlim[0]) & (s_case <= xlim[1]))
        beta_in = beta[mask & np.isfinite(beta)]
        if beta_in.size:
            hi = max(hi, beta_in.max())
    if not np.isfinite(hi):
        return None
    return 0.0, hi * margin


def beta_comparison_fig(xlim, title_suffix):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 7.5))
    for ax, plane in zip(axs, ('x', 'y')):
        for case_label, tw_case, color, linestyle in BETA_CASES:
            ax.plot(tw_case.s, tw_case[f'bet{plane}'], color=color,
                    linestyle=linestyle, label=case_label)
        ax.axvline(x=s_sext, color='k', lw=1.5)
        ax.set_ylabel(rf'$\beta_{plane}$ [m]')
        ax.grid(True, alpha=0.3)
        ylim = beta_ylim_for_xlim(plane, xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
    if xlim is not None:
        axs[0].set_xlim(*xlim)
    axs[0].annotate(
        sext_for_chromaticity_correction, xy=(s_sext, 0.98),
        xycoords=('data', 'axes fraction'), ha='right', va='top',
        fontsize=8, rotation=90, xytext=(-3, 0), textcoords='offset points')
    axs[0].legend(loc='lower right', fontsize=9)
    axs[-1].set_xlabel(f'$s$ from {IP_NAME} [m]')
    fig.suptitle(
        f'Effect of the sextupole optics targets, {title_suffix} (3T)')
    fig.tight_layout()
    return fig


figs_beta = [beta_comparison_fig(xlim, title_suffix)
             for xlim, title_suffix in BETA_PLOT_RANGES]
plt.show(block=False)


##################################################
# Chromatic phase advance across the straight    #
##################################################

# Same calculation as 004dd_chromatic_phase_advance.py, but for the half
# straight only: twiss at a range of deltas, keep mux(s) and muy(s) at every
# element, and fit
#
#     mu(s, delta) ~ mu(s) + mu'(s) delta + mu''(s)/2 delta^2
#
# with the primes true derivatives with respect to delta. Over the full ring
# the end values are Q, Q' and Q''; over the straight they are that segment's
# own contribution to them.
#
# Two caveats, since Q'' is a global quantity:
#  - The end-of-straight mu'' is NOT the ring Q''. The ring value also gets the
#    arcs, and, more importantly, the chromatic mismatch that the straight
#    leaves behind (W_x, W_y) is what makes the arcs contribute. See
#    claude_notes/08_second_order_chromaticity_source.md.
#  - detuning.py's Chromaticity stores coef / n!, so its q{x,y}_derivatives[2]
#    (printed as d2qx/d2qy by 004c/004j) is Q''/4 in the convention above.
#
# The off-momentum initial conditions at the IP are taken from a closed twiss of
# the *bare* ring at each delta, i.e. the real off-momentum optics of the
# reference lattice (at delta = 1e-3 bety* is already ~13% off its on-momentum
# value, so a dispersion-only orbit guess would not do). The same inits are
# used for both cases, so the comparison isolates what the straight itself
# does to the chromatic phase advance.

N_DELTA = 21          # as get_nonlinear_chromaticity / 004dd
DELTA_MAX = 1e-3      # idem
deltas = np.linspace(-DELTA_MAX, DELTA_MAX, N_DELTA)

# Knob states per case. 'sol' and 'corr' are the two top-level knobs; naming a
# correction part (orbit / optics / coupling) overrides that one alone, which is
# how the decomposition below is done. The last three cases answer "which part
# of the correction leaves the chromatic mismatch behind": the optics re-match
# is the suspect, since it restores beta/alpha/D at the straight edge without
# constraining the QD0 -> sdy1 phase.
CORRECTION_PARTS = ('orbit', 'optics', 'coupling')
PHASE_CASES = {
    'bare':        dict(sol=0, corr=0),
    'corrected':   dict(sol=1, corr=1),
    'no orbit':    dict(sol=1, corr=1, orbit=0),
    'no optics':   dict(sol=1, corr=1, optics=0),
    'no coupling': dict(sol=1, corr=1, coupling=0),
}


def set_correction_state(sol, corr, **parts):
    """Set the solenoid and correction knobs for one scan case.

    The three per-part knobs are normally expressions in on_sol_corr_<ip>.
    Overriding one replaces its expression with a constant, so the expression
    is written back here on every call and only the parts named in `parts` end
    up overridden.
    """
    line[f'on_sol_{IP_NAME}'] = sol
    line[f'on_sol_corr_{IP_NAME}'] = corr
    for part in CORRECTION_PARTS:
        line[f'on_sol_{part}_corr_{IP_NAME}'] = parts.get(
            part, f'on_sol_corr_{IP_NAME}')


def fit_quadratic(deltas, mu):
    """Fit mu = c0 + c1 delta + c2 delta^2 at every s, vectorised over s.

    mu has shape (n_delta, n_rows). Fitted in u = delta / delta_max for
    conditioning, then rescaled. Returns (mu, mu', mu'', rms residual), each
    of shape (n_rows,).
    """
    delta_max = np.max(np.abs(deltas))
    u = deltas / delta_max
    coef = np.polynomial.polynomial.polyfit(u, mu, 2)
    residual = mu - np.polynomial.polynomial.polyval(u, coef).T
    return (coef[0], coef[1] / delta_max, 2 * coef[2] / delta_max**2,
            np.sqrt(np.mean(residual**2, axis=0)))


# Off-momentum initial conditions at the IP from the bare ring.
print(f'\nBare-ring closed twiss at {N_DELTA} deltas '
      f'(for the off-momentum initial conditions at {IP_NAME}):')
set_correction_state(**PHASE_CASES['bare'])
inits_vs_delta = []
for delta in deltas:
    co_guess = line.particle_ref.copy()
    co_guess.x = delta * tw0.dx[0]
    co_guess.px = delta * tw0.dpx[0]
    co_guess.y = delta * tw0.dy[0]
    co_guess.py = delta * tw0.dpy[0]
    tw_ring = line.twiss4d(delta0=delta, co_guess=co_guess)
    inits_vs_delta.append(tw_ring.get_twiss_init(IP_NAME))
    print(f'    delta={delta:+.2e}  qx={tw_ring.qx:.6f}  qy={tw_ring.qy:.6f}  '
          f'betx*={tw_ring["betx", IP_NAME]:.4e}  '
          f'bety*={tw_ring["bety", IP_NAME]:.4e}')

# Phase advance across the straight, per case. The twiss init carries the IP's
# absolute phase from the ring, so mu is referenced to the start of line_ds.
phase_fits = {}
chrom_twiss = {}
for case, state in PHASE_CASES.items():
    print(f'Scanning the straight, case "{case}"')
    set_correction_state(**state)
    mu = {}
    for ii, init in enumerate(inits_vs_delta):
        tw_delta = line_ds.twiss4d(init=init)
        if not mu:
            mu = {plane: np.zeros((N_DELTA, len(tw_delta)))
                  for plane in ('x', 'y')}
        mu['x'][ii] = tw_delta.mux - tw_delta.mux[0]
        mu['y'][ii] = tw_delta.muy - tw_delta.muy[0]
    phase_fits[case] = {plane: fit_quadratic(deltas, mu[plane])
                        for plane in ('x', 'y')}
    s_straight = tw_delta.s.copy()
    # Chromatic amplitude functions W = |dbeta/ddelta| / beta-ish, on momentum.
    # This is the quantity the local chromatic correction is there to zero:
    # W_y reaches ~5e3 inside the doublet and the sdy pair has to bring it back
    # to ~0 by the end of the straight. What leaks out is what makes the arcs
    # accumulate mu'' downstream, so W at the end of the straight, not mu''
    # here, is the diagnostic for the IR itself.
    chrom_twiss[case] = line_ds.twiss4d(init=init_ip, chrom=True)

set_correction_state(**PHASE_CASES['corrected'])


####################################
# End-of-straight phase advance    #
####################################

print(f'\nChromatic phase advance {IP_NAME} -> {name_end}, '
      f'|delta| <= {DELTA_MAX:.0e}, {N_DELTA} points')
print("(primes are derivatives w.r.t. delta; mu''/4 is the convention of "
      "004c's d2q print)")
for case in PHASE_CASES:
    for plane in ('x', 'y'):
        mu_fit, dmu, d2mu, res = phase_fits[case][plane]
        print(f"  {case:12s} {plane}:  mu={mu_fit[-1]:11.6f}  mu'={dmu[-1]:10.4f}  "
              f"mu''={d2mu[-1]:12.4f}  mu''/4={d2mu[-1] / 4:11.4f}  "
              f"rms fit residual={res[-1]:.2e} (max over s {np.max(res):.2e})")
for plane in ('x', 'y'):
    diff = [phase_fits['corrected'][plane][kk][-1]
            - phase_fits['bare'][plane][kk][-1] for kk in range(3)]
    print(f"  corr - bare {plane}:  dmu={diff[0]:+11.6f}  "
          f"dmu'={diff[1]:+10.4f}  dmu''={diff[2]:+12.4f}")

# W at the end of the straight: the chromatic mismatch handed to the arcs.
# The bare local chromatic correction leaves ~0 here; whatever the corrected
# lattice leaves is what the arcs turn into ring Q''.
print(f'\nChromatic amplitude functions at {name_end} '
      '(bare LCC leaves ~0; a leak here is what drives the ring Q\'\'):')
for case in PHASE_CASES:
    tw_chrom = chrom_twiss[case]
    print(f'  {case:12s} wx_chrom={tw_chrom.wx_chrom[-1]:10.3f}  '
          f'wy_chrom={tw_chrom.wy_chrom[-1]:10.3f}   '
          f'(max over the straight: {tw_chrom.wx_chrom.max():.3e} / '
          f'{tw_chrom.wy_chrom.max():.3e})')


##############################################
# QD0 -> sdy1 phase: the control variable    #
##############################################

# The local chromatic correction needs mu_y(QD0 -> sdy1) = 1/2 exactly (the -I
# pair): the doublet's chromatic kick and the sextupole's then cancel instead of
# adding at an angle. The 004c optics match restores beta/alpha/D at the
# straight edge but does not constrain this phase, which is what leaves W behind
# and drives the ring Q''y. Everything above is a consequence of this number.
print(f'\nPhase advance {doublet_quad_right[0]} -> {sext_for_lcc} '
      '(the -I condition of the local chromatic correction):')
for case in PHASE_CASES:
    tw_chrom = chrom_twiss[case]
    parts = []
    for plane in ('x', 'y'):
        mu_range = (tw_chrom[f'mu{plane}', sext_for_lcc]
                    - tw_chrom[f'mu{plane}', doublet_quad_right[0]])
        parts.append(f'mu{plane}={mu_range:.6f}')
    print(f'  {case:12s} ' + '   '.join(parts))

muy_bare = (chrom_twiss['bare']['muy', sext_for_lcc]
            - chrom_twiss['bare']['muy', doublet_quad_right[0]])
print(f'  bare value is the reference ({muy_bare:.6f}); '
      'errors against it, and the ring Q\'\'y they would imply:')
for case in PHASE_CASES:
    if case == 'bare':
        continue
    tw_chrom = chrom_twiss[case]
    error = (tw_chrom['muy', sext_for_lcc]
             - tw_chrom['muy', doublet_quad_right[0]] - muy_bare)
    print(f'  {case:12s} phase error {error:+.3e}  ->  '
          f'dQ2y ~ {error * DQ2Y_PER_PHASE_ERROR:+9.0f} '
          f'(one side of one IP, {DQ2Y_PER_PHASE_ERROR:.2e} per unit)')


######################################
# Plot mu, mu', mu'' vs s (straight) #
######################################

# Vertical markers: the local-chromatic-correction sextupoles sdy* (green,
# these are the -I partners of the final doublet) and sdm1r.0 (black), the
# sextupole whose optics the match above constrains.
table_straight = line_ds.get_table()
sdy_s = {}
for element_type, nn in zip(table_straight.element_type, table_straight.name):
    if element_type == 'Sextupole' and nn.startswith('sdy'):
        sdy_s.setdefault(nn.split('.')[0], table_straight['s', nn])

CASE_STYLE = {
    'bare': dict(color='tab:gray', lw=1.4),
    'corrected': dict(color='tab:red', lw=1.4),
    'no orbit': dict(color='tab:blue', lw=0.9, ls='--'),
    'no optics': dict(color='tab:green', lw=0.9, ls='--'),
    'no coupling': dict(color='tab:purple', lw=0.9, ls=':'),
}
ROW_LABELS = (r'$\mu_{p}$', r"$\mu_{p}'$", r"$\mu_{p}''$")


def symlog_threshold(curves):
    """Linear-region half-width for a symlog y axis on mu' / mu''.

    Off momentum the final-doublet waists move, which shifts mu by a lot over
    a few tens of metres and shifts it back again. mu' and mu'' therefore
    swing to O(1e5) inside the doublet while the accumulated end-of-straight
    values are O(1e2): those excursions largely cancel. A linear axis shows
    only the excursions, so use symlog, linear up to the end-of-straight
    magnitude and logarithmic above it.
    """
    end_value = max(np.abs(curve[-1]) for curve in curves)
    if end_value <= 1.0:
        return 1.0
    return 10 ** np.floor(np.log10(end_value))


fig_phase, axs_phase = plt.subplots(4, 2, sharex=True, figsize=(12, 11))
for jj, plane in enumerate(('x', 'y')):
    for kk in range(4):
        ax = axs_phase[kk, jj]
        for case in PHASE_CASES:
            if kk < 3:
                ax.plot(s_straight, phase_fits[case][plane][kk], label=case,
                        **CASE_STYLE[case])
            else:
                ax.semilogy(chrom_twiss[case].s,
                            chrom_twiss[case][f'w{plane}_chrom'], label=case,
                            **CASE_STYLE[case])
        if kk < 3:
            ax.set_ylabel(ROW_LABELS[kk].replace('p', plane))
        else:
            ax.set_ylabel(f'$W_{plane}$')
        if 0 < kk < 3:
            ax.set_yscale('symlog', linthresh=symlog_threshold(
                [phase_fits[case][plane][kk] for case in PHASE_CASES]))
            ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.axvline(table_straight['s', sext_for_chromaticity_correction],
                   color='k', ls='--', lw=0.8, alpha=0.6)
        for s_sext in sdy_s.values():
            ax.axvline(s_sext, color='tab:green', ls='--', lw=0.8, alpha=0.8)
        ax.set_xlim(s_straight[0], s_straight[-1])
    for name_sext, s_sext in sdy_s.items():
        axs_phase[0, jj].text(s_sext, 1.02, name_sext,
                              transform=axs_phase[0, jj].get_xaxis_transform(),
                              ha='center', va='bottom', fontsize=8)
    axs_phase[-1, jj].set_xlabel('s [m]')
axs_phase[0, 0].legend(loc='upper left')
fig_phase.suptitle(
    f'Chromatic phase advance across the {IP_NAME} straight (3T)\n'
    f'(black: {sext_for_chromaticity_correction}, green: sdy sextupoles; '
    "the mu' and mu'' rows are symlog, linear near the end-of-straight value.\n"
    'The bottom row is what the local chromatic correction is for: '
    'W must come back to ~0 by the end of the straight)')
fig_phase.tight_layout()
plt.show(block=False)