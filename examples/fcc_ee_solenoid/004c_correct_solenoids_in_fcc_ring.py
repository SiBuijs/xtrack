from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import xtrack as xt

from solenoid_params import (
    MAIN_SOLENOID_B0,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    order_tag,
)


HERE = Path(__file__).parent

parser = argparse.ArgumentParser(
    description='Correct solenoids in the FCC ring lattice.'
)
parser.add_argument(
    '--model',
    choices=['splineboris', 'varsol'],
    default='splineboris',
    help='Solenoid model to correct (default: splineboris).',
)
add_b0_argument(parser, default=MAIN_SOLENOID_B0)
add_max_order_argument(parser)
parser.add_argument(
    '--output-tag', default='',
    help='Optional extra suffix appended to BOTH the input temp lattice and '
         'the output corrected lattice filenames (e.g. "mainscale"). Must '
         'match the --output-tag passed to 004b_install_solenoids_in_fcc_'
         'ring.py. Empty (default) keeps the standard filenames.')
args = parser.parse_args()

FIELD_TAG = field_tag(args.b0)
# The transverse-order cap only applies to the SplineBoris model (VarSol is
# linear-only, see 00_overview.md) -- its tag is left out of the varsol
# filenames so --max-transverse-order has no effect on --model varsol.
ORDER_TAG = order_tag(args.max_transverse_order)
OUT_TAG = f'_{args.output_tag}' if args.output_tag else ''
_MODEL_LATTICE_PATHS = {
    'splineboris': (
        f'temp_fcc_ee_lcc_splineboris_solenoids_{FIELD_TAG}{ORDER_TAG}{OUT_TAG}.json',
        f'fccee_z_lcc_splineboris_solenoids_coupling_corrected_{FIELD_TAG}{ORDER_TAG}{OUT_TAG}.json',
    ),
    'varsol': (
        f'temp_fcc_ee_lcc_varsol_solenoids_{FIELD_TAG}{OUT_TAG}.json',
        f'fccee_z_lcc_varsol_solenoids_coupling_corrected_{FIELD_TAG}{OUT_TAG}.json',
    ),
}

INPUT_LATTICE_JSON = HERE / _MODEL_LATTICE_PATHS[args.model][0]
OUTPUT_LATTICE_JSON = HERE / _MODEL_LATTICE_PATHS[args.model][1]

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']


def measure_ksol_l_main_solenoid(line, env, ip_name):
    ksol_l = 0.0
    rigidity0 = line.particle_ref.rigidity0[0]
    table_solenoid_region = line.get_table().rows[
        'dy_match_l_' + ip_name: 'dy_match_r_' + ip_name]

    for nn in table_solenoid_region.name:
        element_type = table_solenoid_region['element_type', nn]
        element = env.get(table_solenoid_region['env_name', nn])

        if element_type == 'VariableSolenoid':
            ksol_l += element.ks_profile.mean() * element.length
        elif element_type == 'SplineBoris':
            ksol_l += element.scale_b * element.bs[4] * element.length / rigidity0

    return ksol_l


#####################################
# Load installed solenoid lattice #
#####################################

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring


##################################################
# Correction configuration copied from 005g setup #
##################################################

config = {}
config['ipa'] = {
    'quad_for_optics_correction': [
        'qd0ar.0', 'qd0br.0', 'qd0cr.0', 'qf1ar.0', 'qf1br.0',
        'qf1cr.0', 'qf1dr.0', 'qf2r.0', 'qd3r.0', 'qd4r.0',
        'qf5r.0', 'qd6r.0', 'qd6l.3', 'qf5l.3', 'qd4l.3',
        'qd3l.3', 'qf2l.3', 'qf1dl.3', 'qf1cl.3', 'qf1bl.3',
        'qf1al.3', 'qd0cl.3', 'qd0bl.3', 'qd0al.3',
    ],
    'doublet_quad_left': [
        'qd0al.3', 'qd0bl.3', 'qd0cl.3', 'qf1al.3', 'qf1bl.3',
        'qf1cl.3', 'qf1dl.3',
    ],
    'doublet_quad_right': [
        'qd0ar.0', 'qd0br.0', 'qd0cr.0', 'qf1ar.0', 'qf1br.0',
        'qf1cr.0', 'qf1dr.0',
    ],
    'corr_1_right_on_quad': 'qd0ar.0',
    'corr_2_right_on_quad': 'qd0br.0',
    'corr_3_right_on_quad': 'qf1ar.0',
    'corr_4_right_on_quad': 'qf1br.0',
    'corr_1_left_on_quad': 'qd0al.3',
    'corr_2_left_on_quad': 'qd0bl.3',
    'corr_3_left_on_quad': 'qf1al.3',
    'corr_4_left_on_quad': 'qf1bl.3',
    'bend_for_mid_quad_correction': [
        'b0cl.3', 'b0bl.3', 'b0al.3',   # upstream, beam order
        'b1ra.0', 'b1rb.0', 'b1rc.0',   # downstream, beam order
    ],
    # Downstream only -- the upstream partner (sdm1l.6) is not
    # targeted; see the opt_optics targets below.
    'sext_for_optics_correction': 'sdm1r.0',
}
config['ipd'] = {
    'quad_for_optics_correction': [
        'qd0ar.1', 'qd0br.1', 'qd0cr.1', 'qf1ar.1', 'qf1br.1',
        'qf1cr.1', 'qf1dr.1', 'qf2r.1', 'qd3r.1', 'qd4r.1',
        'qf5r.1', 'qd6r.1', 'qd6l.0', 'qf5l.0', 'qd4l.0',
        'qd3l.0', 'qf2l.0', 'qf1dl.0', 'qf1cl.0', 'qf1bl.0',
        'qf1al.0', 'qd0cl.0', 'qd0bl.0', 'qd0al.0',
    ],
    'doublet_quad_left': [
        'qd0al.0', 'qd0bl.0', 'qd0cl.0', 'qf1al.0', 'qf1bl.0',
        'qf1cl.0', 'qf1dl.0',
    ],
    'doublet_quad_right': [
        'qd0ar.1', 'qd0br.1', 'qd0cr.1', 'qf1ar.1', 'qf1br.1',
        'qf1cr.1', 'qf1dr.1',
    ],
    'corr_1_right_on_quad': 'qd0ar.1',
    'corr_2_right_on_quad': 'qd0br.1',
    'corr_3_right_on_quad': 'qf1ar.1',
    'corr_4_right_on_quad': 'qf1br.1',
    'corr_1_left_on_quad': 'qd0al.0',
    'corr_2_left_on_quad': 'qd0bl.0',
    'corr_3_left_on_quad': 'qf1al.0',
    'corr_4_left_on_quad': 'qf1bl.0',
    'bend_for_mid_quad_correction': [
        'b0cl.0', 'b0bl.0', 'b0al.0',   # upstream, beam order
        'b1ra.1', 'b1rb.1', 'b1rc.1',   # downstream, beam order
    ],
    # Downstream only -- the upstream partner (sdm1l.0) is not
    # targeted; see the opt_optics targets below.
    'sext_for_optics_correction': 'sdm1r.2',
}
config['ipg'] = {
    'quad_for_optics_correction': [
        'qd0ar.2', 'qd0br.2', 'qd0cr.2', 'qf1ar.2', 'qf1br.2',
        'qf1cr.2', 'qf1dr.2', 'qf2r.2', 'qd3r.2', 'qd4r.2',
        'qf5r.2', 'qd6r.2', 'qd6l.1', 'qf5l.1', 'qd4l.1',
        'qd3l.1', 'qf2l.1', 'qf1dl.1', 'qf1cl.1', 'qf1bl.1',
        'qf1al.1', 'qd0cl.1', 'qd0bl.1', 'qd0al.1',
    ],
    'doublet_quad_left': [
        'qd0al.1', 'qd0bl.1', 'qd0cl.1', 'qf1al.1', 'qf1bl.1',
        'qf1cl.1', 'qf1dl.1',
    ],
    'doublet_quad_right': [
        'qd0ar.2', 'qd0br.2', 'qd0cr.2', 'qf1ar.2', 'qf1br.2',
        'qf1cr.2', 'qf1dr.2',
    ],
    'corr_1_right_on_quad': 'qd0ar.2',
    'corr_2_right_on_quad': 'qd0br.2',
    'corr_3_right_on_quad': 'qf1ar.2',
    'corr_4_right_on_quad': 'qf1br.2',
    'corr_1_left_on_quad': 'qd0al.1',
    'corr_2_left_on_quad': 'qd0bl.1',
    'corr_3_left_on_quad': 'qf1al.1',
    'corr_4_left_on_quad': 'qf1bl.1',
    'bend_for_mid_quad_correction': [
        'b0cl.1', 'b0bl.1', 'b0al.1',   # upstream, beam order
        'b1ra.2', 'b1rb.2', 'b1rc.2',   # downstream, beam order
    ],
    # Downstream only -- the upstream partner (sdm1l.2) is not
    # targeted; see the opt_optics targets below.
    'sext_for_optics_correction': 'sdm1r.4',
}
config['ipj'] = {
    'quad_for_optics_correction': [
        'qd0ar.3', 'qd0br.3', 'qd0cr.3', 'qf1ar.3', 'qf1br.3',
        'qf1cr.3', 'qf1dr.3', 'qf2r.3', 'qd3r.3', 'qd4r.3',
        'qf5r.3', 'qd6r.3', 'qd6l.2', 'qf5l.2', 'qd4l.2',
        'qd3l.2', 'qf2l.2', 'qf1dl.2', 'qf1cl.2', 'qf1bl.2',
        'qf1al.2', 'qd0cl.2', 'qd0bl.2', 'qd0al.2',
    ],
    'doublet_quad_left': [
        'qd0al.2', 'qd0bl.2', 'qd0cl.2', 'qf1al.2', 'qf1bl.2',
        'qf1cl.2', 'qf1dl.2',
    ],
    'doublet_quad_right': [
        'qd0ar.3', 'qd0br.3', 'qd0cr.3', 'qf1ar.3', 'qf1br.3',
        'qf1cr.3', 'qf1dr.3',
    ],
    'corr_1_right_on_quad': 'qd0ar.3',
    'corr_2_right_on_quad': 'qd0br.3',
    'corr_3_right_on_quad': 'qf1ar.3',
    'corr_4_right_on_quad': 'qf1br.3',
    'corr_1_left_on_quad': 'qd0al.2',
    'corr_2_left_on_quad': 'qd0bl.2',
    'corr_3_left_on_quad': 'qf1al.2',
    'corr_4_left_on_quad': 'qf1bl.2',
    'bend_for_mid_quad_correction': [
        'b0cl.2', 'b0bl.2', 'b0al.2',   # upstream, beam order
        'b1ra.3', 'b1rb.3', 'b1rc.3',   # downstream, beam order
    ],
    # Downstream only -- the upstream partner (sdm1l.4) is not
    # targeted; see the opt_optics targets below.
    'sext_for_optics_correction': 'sdm1r.6',
}


###############################################################################
# Mid-bend trim quadrupoles                                                   #
#                                                                             #
# opt_optics below constrains the optics only at the two straight-section     #
# boundary markers, ~1360 m either side of the IP, so a purely local          #
# beta-beat in between is unpenalised. With the detector solenoid on exactly  #
# such a bump appears downstream of every IP: bety at sdm1r.0 goes 1.47 m     #
# (bare) -> 2.87 m (2 T) -> 310 m (3 T), peaking at the b1rc exit / qd4r and  #
# recovering by qd6r, while the upstream side (sdm1l.6) is untouched. That    #
# ~200x vertical beta sits on a chromatic sextupole and is the suspected      #
# driver of the second-order chromaticity.                                    #
#                                                                             #
# To give the optics match local handles inside that region, each of the six  #
# bends framing the IP (config[ip]['bend_for_mid_quad_correction']) is cut in #
# half here and a zero-length trim quadrupole is placed at the cut. The       #
# correctors are wired into opt_optics' vary list further down, so they are   #
# driven by on_sol_optics_corr_{ip} and are exactly zero whenever the         #
# corrections are switched off.                                              #
#                                                                             #
# Done ONCE, up front: cut_at_s() and the s_center lookups work in the        #
# current s frame, and the per-IP loop below re-cycles the line (which        #
# re-bases s) on every pass. Everything after this section is name-based and  #
# survives those cycles.                                                     #
###############################################################################

BEND_MID_QUAD_PREFIX = 'qbmid_'


def bend_mid_quad_name(bend_name):
    """Element name of the trim quadrupole at the centre of `bend_name`."""
    return BEND_MID_QUAD_PREFIX + bend_name


_table_before_mid_quad_cuts = line.get_table()
_s_mid_bend_cuts = []
for ip_name in IP_NAMES:
    for bend_name in config[ip_name]['bend_for_mid_quad_correction']:
        if bend_name not in _table_before_mid_quad_cuts.name:
            raise SystemExit(
                f'{INPUT_LATTICE_JSON.name} has no element {bend_name!r} -- the '
                'near-IP bend naming changed; update '
                "config[...]['bend_for_mid_quad_correction'].")
        element_type = _table_before_mid_quad_cuts['element_type', bend_name]
        if element_type != 'RBend':
            raise SystemExit(
                f'{bend_name!r} is a {element_type!r}, not an unsliced RBend -- '
                'the mid-bend trim quads assume the bends arrive thick and '
                'unsliced from 004b_install_solenoids_in_fcc_ring.py.')
        s_start = _table_before_mid_quad_cuts['s_start', bend_name]
        s_end = _table_before_mid_quad_cuts['s_end', bend_name]
        if not s_end > s_start:
            raise SystemExit(
                f'{bend_name!r} spans the line start (s_start={s_start}, '
                f's_end={s_end}) -- cycle the line away from it before cutting.')
        # s_center is exactly (s_start + s_end) / 2. For an RBend the table s is
        # the arc length, so this is the true geometric midpoint, i.e. half the
        # bend angle either side.
        _s_mid_bend_cuts.append(
            _table_before_mid_quad_cuts['s_center', bend_name])

# cut_at_s advances a single iterator over `s` in lockstep with the elements
# (see Line._elements_intersecting_s), so `s` has to be ascending or cuts are
# silently dropped.
line.cut_at_s(sorted(_s_mid_bend_cuts))

# One batched insert: line.insert() rebuilds the whole ~30k-element line on
# every call (~2.3 s), so 24 separate calls would cost ~55 s instead of ~2.3 s.
_mid_quad_places = []
for ip_name in IP_NAMES:
    for bend_name in config[ip_name]['bend_for_mid_quad_correction']:
        downstream_half = f'{bend_name}..1'
        if downstream_half not in line.element_names:
            raise SystemExit(
                f'cut_at_s did not split {bend_name!r} (no {downstream_half!r}) '
                '-- the requested cut probably coincided with an existing slice '
                'boundary.')
        mid_quad_name = bend_mid_quad_name(bend_name)
        # A real xt.Quadrupole, so that it shows up as a quadrupole (and not as
        # a thin multipole with vertical/horizontal kick handles) in every
        # get_table / plot of the saved lattice. The catch is that
        # 004f/004g/004h/004i/004j all re-derive their coupling-corrector host
        # list from the saved lattice by selecting element_type == 'Quadrupole'
        # between the straight-section boundaries, then SystemExit if the
        # matching k1s_*_sol_coupling_corr var is missing -- so these would be
        # picked up as skew hosts. All five therefore skip names starting with
        # BEND_MID_QUAD_PREFIX; if you add another such scan, skip them there
        # too. Zero length also means k1/k1s are dead, so the strength goes on
        # the integrated knl[1] and no skew handle is offered at all: 004f-j
        # could not discover one, and would re-solve coupling with a smaller
        # knob set than the one used to build the lattice.
        env.elements[mid_quad_name] = xt.Multipole(knl=[0.0, 0.0], length=0.0)
        _mid_quad_places.append(env.place(
            mid_quad_name, at=0, from_=downstream_half,
            anchor='start', from_anchor='start'))

line.insert(_mid_quad_places)
print(f'Installed {len(_mid_quad_places)} mid-bend trim quadrupoles '
      f'({len(IP_NAMES)} IPs x '
      f'{len(config[IP_NAMES[0]]["bend_for_mid_quad_correction"])} bends)')

################################
# Build one correction per IP  #
################################

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 0
    line[f'on_comp_sol_{ip_name}'] = 0

optimizers = {}
for ip_name in IP_NAMES:

    print(f'IP {ip_name}:')
    line.cycle(f'end_ds_start_straight_{ip_name}')

    # Reference optics with all solenoids off at this IP.
    tw0 = line.twiss4d(strengths=True)

    # Turn on only the solenoid system being corrected.
    line[f'on_sol_{ip_name}'] = 1
    line[f'on_comp_sol_{ip_name}'] = 1

    quad_for_optics_correction = config[ip_name]['quad_for_optics_correction']
    doublet_quad_left = config[ip_name]['doublet_quad_left']
    doublet_quad_right = config[ip_name]['doublet_quad_right']
    corr_1_right_on_quad = config[ip_name]['corr_1_right_on_quad']
    corr_2_right_on_quad = config[ip_name]['corr_2_right_on_quad']
    corr_3_right_on_quad = config[ip_name]['corr_3_right_on_quad']
    corr_4_right_on_quad = config[ip_name]['corr_4_right_on_quad']
    corr_1_left_on_quad = config[ip_name]['corr_1_left_on_quad']
    corr_2_left_on_quad = config[ip_name]['corr_2_left_on_quad']
    corr_3_left_on_quad = config[ip_name]['corr_3_left_on_quad']
    corr_4_left_on_quad = config[ip_name]['corr_4_left_on_quad']

    # Rotate the final doublets by half of the main-solenoid rotation.
    ksol_l_main_solenoid = measure_ksol_l_main_solenoid(line, env, ip_name)
    env[f'phi_rot_doublet_{ip_name}'] = (ksol_l_main_solenoid / 2) / 2
    env[f'on_rot_doublet_left_{ip_name}'] = 1
    env[f'on_rot_doublet_right_{ip_name}'] = 1
    for nn in doublet_quad_left:
        env[nn].rot_s_rad = (
            +env.ref[f'phi_rot_doublet_{ip_name}']
            * env.ref[f'on_rot_doublet_left_{ip_name}'])
    for nn in doublet_quad_right:
        env[nn].rot_s_rad = (
            -env.ref[f'phi_rot_doublet_{ip_name}']
            * env.ref[f'on_rot_doublet_right_{ip_name}'])

    # Orbit corrector knobs. The first pair was installed inside the main
    # solenoid by 004b; the others are attached here to nearby quadrupoles and
    # to the dedicated compensation-solenoid correctors.
    env[f'acbh2_sol_right_{ip_name}'] = 0
    env[f'acbh3_sol_right_{ip_name}'] = 0
    env[f'acbh4_sol_right_{ip_name}'] = 0
    env[f'acbh5_sol_right_{ip_name}'] = 0
    env[f'acbh6_sol_right_{ip_name}'] = 0
    env[f'acbv2_sol_right_{ip_name}'] = 0
    env[f'acbv3_sol_right_{ip_name}'] = 0
    env[f'acbv4_sol_right_{ip_name}'] = 0
    env[f'acbv5_sol_right_{ip_name}'] = 0
    env[f'acbv6_sol_right_{ip_name}'] = 0
    env[f'acbh2_sol_left_{ip_name}'] = 0
    env[f'acbh3_sol_left_{ip_name}'] = 0
    env[f'acbh4_sol_left_{ip_name}'] = 0
    env[f'acbh5_sol_left_{ip_name}'] = 0
    env[f'acbh6_sol_left_{ip_name}'] = 0
    env[f'acbv2_sol_left_{ip_name}'] = 0
    env[f'acbv3_sol_left_{ip_name}'] = 0
    env[f'acbv4_sol_left_{ip_name}'] = 0
    env[f'acbv5_sol_left_{ip_name}'] = 0
    env[f'acbv6_sol_left_{ip_name}'] = 0

    env[corr_1_right_on_quad].knl[0] += env.ref[f'acbh2_sol_right_{ip_name}']
    env[corr_2_right_on_quad].knl[0] += env.ref[f'acbh3_sol_right_{ip_name}']
    env[corr_3_right_on_quad].knl[0] += env.ref[f'acbh4_sol_right_{ip_name}']
    env[corr_4_right_on_quad].knl[0] += env.ref[f'acbh5_sol_right_{ip_name}']
    env[f'corr_sol_right_{ip_name}'].knl[0] += (
        env.ref[f'acbh6_sol_right_{ip_name}'])

    env[corr_1_left_on_quad].knl[0] += env.ref[f'acbh2_sol_left_{ip_name}']
    env[corr_2_left_on_quad].knl[0] += env.ref[f'acbh3_sol_left_{ip_name}']
    env[corr_3_left_on_quad].knl[0] += env.ref[f'acbh4_sol_left_{ip_name}']
    env[corr_4_left_on_quad].knl[0] += env.ref[f'acbh5_sol_left_{ip_name}']
    env[f'corr_sol_left_{ip_name}'].knl[0] += (
        env.ref[f'acbh6_sol_left_{ip_name}'])

    env[corr_1_right_on_quad].ksl[0] += env.ref[f'acbv2_sol_right_{ip_name}']
    env[corr_2_right_on_quad].ksl[0] += env.ref[f'acbv3_sol_right_{ip_name}']
    env[corr_3_right_on_quad].ksl[0] += env.ref[f'acbv4_sol_right_{ip_name}']
    env[corr_4_right_on_quad].ksl[0] += env.ref[f'acbv5_sol_right_{ip_name}']
    env[f'corr_sol_right_{ip_name}'].ksl[0] += (
        env.ref[f'acbv6_sol_right_{ip_name}'])

    env[corr_1_left_on_quad].ksl[0] += env.ref[f'acbv2_sol_left_{ip_name}']
    env[corr_2_left_on_quad].ksl[0] += env.ref[f'acbv3_sol_left_{ip_name}']
    env[corr_3_left_on_quad].ksl[0] += env.ref[f'acbv4_sol_left_{ip_name}']
    env[corr_4_left_on_quad].ksl[0] += env.ref[f'acbv5_sol_left_{ip_name}']
    env[f'corr_sol_left_{ip_name}'].ksl[0] += (
        env.ref[f'acbv6_sol_left_{ip_name}'])

    # Match orbit and vertical dispersion across the solenoid region.
    opt_orbit = line.match_knob(
        knob_name=f'on_sol_orbit_corr_{ip_name}',
        run=False,
        betx=tw0['betx', ip_name],
        bety=tw0['bety', ip_name],
        start=f'dy_match_l_{ip_name}',
        end=f'dy_match_r_{ip_name}',
        init_at=ip_name,
        vary=xt.VaryList([
            f'acbh1_sol_right_{ip_name}', f'acbv1_sol_right_{ip_name}',
            f'acbh2_sol_right_{ip_name}', f'acbh3_sol_right_{ip_name}',
            f'acbh4_sol_right_{ip_name}', f'acbh5_sol_right_{ip_name}',
            f'acbh6_sol_right_{ip_name}', f'acbv2_sol_right_{ip_name}',
            f'acbv3_sol_right_{ip_name}', f'acbv4_sol_right_{ip_name}',
            f'acbv5_sol_right_{ip_name}', f'acbv6_sol_right_{ip_name}',
            f'acbh1_sol_left_{ip_name}', f'acbv1_sol_left_{ip_name}',
            f'acbh2_sol_left_{ip_name}', f'acbh3_sol_left_{ip_name}',
            f'acbh4_sol_left_{ip_name}', f'acbh5_sol_left_{ip_name}',
            f'acbh6_sol_left_{ip_name}', f'acbv2_sol_left_{ip_name}',
            f'acbv3_sol_left_{ip_name}', f'acbv4_sol_left_{ip_name}',
            f'acbv5_sol_left_{ip_name}', f'acbv6_sol_left_{ip_name}',
        ], step=1e-6),
        targets=[
            xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.END),
            xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.START),
        ])
    opt_orbit.solve()

    two = line.twiss(
        strengths=True,
        init_at=ip_name,
        betx=tw0['betx', ip_name],
        bety=tw0['bety', ip_name],
    )

    # Match optics and horizontal dispersion with normal quadrupole trims.
    k1_knobs = []
    for nn in quad_for_optics_correction:
        nn_knob = f'k1_{nn}_sol_corr'
        env[nn_knob] = 0
        env[nn].k1 += env.ref[nn_knob]
        k1_knobs.append(nn_knob)

    # Mid-bend trim quads (installed above the per-IP loop). These are
    # zero-length, so the strength has to go on the integrated knl[1]; k1 is
    # dead at zero length -- same reasoning as
    # lattice_knobs.install_extra_sextupole. Units therefore differ from the
    # trims above (k1l [1/m] rather than k1 [1/m^2]), but over a 25-60 m bend a
    # k1l of 1e-6 is a distributed k1 of ~2-4e-8, so the shared step=1e-6 in the
    # VaryList below is a comparable perturbation for both families. Each bend
    # belongs to exactly one IP, so there is no cross-IP double-attachment.
    for bend_name in config[ip_name]['bend_for_mid_quad_correction']:
        nn = bend_mid_quad_name(bend_name)
        nn_knob = f'k1_{nn}_sol_corr'
        env[nn_knob] = 0
        env[nn].knl[1] += env.ref[nn_knob]
        k1_knobs.append(nn_knob)

    name_start = f'end_ds_start_straight_{ip_name}'
    name_end = f'end_straight_start_ds_{ip_name}'

    # Skew quadrupole knobs for the additional linear-coupling/vertical-
    # dispersion correction. Use all quadrupoles from the IP to the right edge
    # and from the left edge to the IP.
    table_for_skew = line.get_table()
    k1s_quads_for_coupling_correction = []
    for table_part in (
            table_for_skew.rows[name_start:ip_name],
            table_for_skew.rows[ip_name:name_end]):
        for element_type, env_name in zip(
                table_part.element_type, table_part.env_name):
            if (
                    element_type == 'Quadrupole'
                    and env_name not in k1s_quads_for_coupling_correction):
                k1s_quads_for_coupling_correction.append(env_name)

    k1s_knobs = []
    for nn in k1s_quads_for_coupling_correction:
        nn_knob = f'k1s_{nn}_sol_coupling_corr'
        env[nn_knob] = 0
        env[nn].k1s += env.ref[nn_knob]
        k1s_knobs.append(nn_knob)

    sext_corr = config[ip_name]['sext_for_optics_correction']

    opt_optics = line.match_knob(
        knob_name=f'on_sol_optics_corr_{ip_name}',
        run=False,
        betx=tw0['betx', ip_name],
        bety=tw0['bety', ip_name],
        init_at=ip_name,
        start=name_start,
        end=name_end,
        # 30 vary knobs (24 quad trims + 6 mid-bend trims) against 20 targets.
        # Slightly above the default 20, chosen from a measured convergence
        # trace (2 T, ipa): the solve takes full Newton steps throughout
        # (bisection alpha=0 on 59 of 60 steps, no max_step clipping) and the
        # Jacobian is already converged -- sweeping the finite-difference step
        # below over 1e-6..1e-3 gives byte-identical results. What is left is
        # conditioning: the Jacobian is full rank but ill-conditioned (cond
        # ~5.9e6 measured while both sextupoles were still targeted, 1.23e6 at
        # the start point now that only the downstream one is), so after a fast
        # phase (penalty 19.9 -> 0.33 in six steps) the remaining stiff
        # direction only decays ~8 % per step. That direction is the boundary
        # START_betx, not the sextupole: bety at sext_corr is converged to
        # six digits by step ~10 and does not move thereafter, while
        # START_betx keeps improving (1.0e-3 relative at 10 steps, 1.9e-4 at
        # 30, 1.8e-5 at 60). 30 halves the cost of the three opt_optics solves
        # per IP versus 60 with no change to the quantity this correction
        # exists to fix. Raise it again if the 3 T case (a 310 m bump rather
        # than 2.9 m) needs a longer fast phase -- that has not been measured.
        n_steps_max=30,
        # Same reasoning as opt_coupling below: the first solve() can leave a
        # target marginally outside tol before the iterate pass further down
        # gets to run, and solve()'s take_best keeps the best point either way.
        assert_within_tol=False,
        vary=xt.VaryList(k1_knobs, step=1e-7),
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
            # Restore the full Twiss vector at the sdm1 sextupole downstream
            # of the IP. The boundary targets above leave a purely local
            # beta-beat unpenalised; this pins the one place where it actually
            # hurts. alfx/alfy are targeted as well as betx/bety because qd4r
            # -- the measured peak of the bump -- sits only ~1.75 m upstream of
            # sext_corr, and the six mid-bend correctors are strongly
            # non-local: pinning beta alone at one point does not forbid a
            # large bump that happens to cross the right value there with the
            # wrong slope, whereas pinning (beta, alfa) forces the whole span
            # from the sextupole to the boundary to nominal. Only the first
            # member of the contiguous sextupole pair is targeted -- the .0/.1
            # optics differ by ~4 %, so the second adds targets without adding
            # information.
            #
            # DOWNSTREAM ONLY, deliberately. The upstream partner (sdm1l.*) was
            # targeted too at first, and it made things worse: the solenoid
            # barely disturbs it (bety 6.485 -> 6.447, 0.6 %), so those four
            # near-trivial constraints bought nothing while crowding the bottom
            # of the Jacobian spectrum -- measured at the start point on ipa,
            # dropping them takes the directions with sigma < 4 from four to
            # two at unchanged condition number (1.23e6). A/B at 2 T on ipa,
            # one opt_optics.solve(), bety at sdm1r.0 against a 1.471115 m
            # nominal:
            #
            #   both sides targeted, 6 quads   1.505363
            #   downstream only,     6 quads   1.472736   <- this file
            #   both sides,          3 quads   1.480550
            #   downstream only,     3 quads   1.486639
            #
            # The last two rows are why all six mid-bend quads are kept even
            # though only the downstream sextupole is targeted: removing the
            # three upstream ones costs an order of magnitude on the result.
            # Note this is NOT because they do work at the optimum -- they
            # converge to ~1e-6 against ~1e-3 for the downstream three, i.e.
            # essentially unused -- nor because of start-point conditioning,
            # which is identical with and without them (cond 1.23e6, same
            # spectrum to four digits). The effect is on the descent path:
            # dropping three columns changes the pseudo-inverse step direction
            # at every iteration, and this solve does not reach a true optimum
            # in its step budget (the stiff START_betx direction is still
            # decaying ~8 %/step when it stops), so a different path ends at a
            # different point. Keep them; they cost three knobs and no
            # conditioning.
            #
            # tol is absolute in xtrack: 1e-5 at the boundaries is 2e-8 relative
            # on bety=514 m, while the same 1e-5 at sdm1r.0 (bety=1.47 m) is
            # 7e-6 relative, i.e. the new targets are deliberately looser in
            # relative terms than the ones that already converge.
            #
            # No explicit weights: a weight on the sextupole beta targets was
            # tried at 1, 10 and 100 and the converged residues were identical
            # to three significant figures, so the solve is not sitting at a
            # weighted optimum -- it stops at a structural convergence limit of
            # this knob set. Adding the local targets costs some boundary
            # precision (START_betx goes from ~1e-9 absolute in the unmodified
            # script to ~2.8e-3, i.e. 1.8e-5 relative).
            xt.TargetSet(
                betx=tw0['betx', sext_corr],
                bety=tw0['bety', sext_corr],
                tol=1e-5,
                at=sext_corr),
            xt.TargetSet(
                alfx=tw0['alfx', sext_corr],
                alfy=tw0['alfy', sext_corr],
                tol=1e-6,
                at=sext_corr),
        ])
    # If the WARNING further down fires, escalate in this order before
    # restructuring anything: opt_optics.solve(rcond=1e-4) -> solve(rcond=1e-3,
    # broyden=True) (what 004h already uses for its coupling re-fit) -> relax
    # the two sextupole tolerances above to 1e-4 (beta) / 1e-5 (alfa) -> only
    # then a separate knob or a two-stage match.
    opt_optics.solve()



    # Try an additional correction of linear coupling and vertical dispersion
    # at the straight-section edges using the skew quadrupole knobs.
    opt_coupling = line.match_knob(
        knob_name=f'on_sol_coupling_corr_{ip_name}',
        run=False,
        betx=tw0['betx', ip_name],
        bety=tw0['bety', ip_name],
        init_at=ip_name,
        start=name_start,
        end=name_end,
        n_steps_max=100,
        # The first solve() below (before the orbit/optics/coupling iterate
        # loop has a chance to run again) can leave 1-2 of the 12 targets
        # just outside tol (seen at 3 T for the VariableSolenoid model, e.g.
        # ipa's END_betx2 at ~5e-4 vs tol 5e-5) even though the knob is
        # otherwise well-behaved and the *next* iterate pass below brings
        # every target within tol. Without this, the strict single-pass
        # assertion raises before that second pass ever runs. take_best
        # (solve()'s default) already keeps the best point found either way.
        assert_within_tol=False,
        vary=xt.VaryList(k1s_knobs, step=1e-7),
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

    # The coupling knob has ~80 skew-quad vary knobs but only 8 targets, so
    # the Jacobian is heavily rank-deficient; with the default rcond the
    # pseudo-inverse chases numerically-noisy near-null directions and the
    # solve stalls/oscillates instead of converging (seen after the main
    # solenoid was raised to 3 T). Truncating small singular values fixes it.
    opt_coupling.solve(rcond=3e-3)

    # Iterate to improve consistency of orbit and optics corrections.
    opt_orbit.solve()
    opt_optics.solve()
    opt_coupling.solve(rcond=3e-3)
    opt_orbit.solve()
    opt_optics.solve()

    _optics_status = opt_optics.target_status(ret=True)
    if not all(_optics_status.tol_met):
        print(f'WARNING: on_sol_optics_corr_{ip_name} did not fully '
              f'converge to tolerance; using best point found.')
        opt_optics.target_mismatch()

    _coupling_status = opt_coupling.target_status(ret=True)
    if not all(_coupling_status.tol_met):
        print(f'WARNING: on_sol_coupling_corr_{ip_name} did not fully '
              f'converge to tolerance; using best point found.')
        opt_coupling.target_mismatch()

    opt_orbit.generate_knob()
    opt_optics.generate_knob()
    opt_coupling.generate_knob()

    optimizers[f'{ip_name}_orbit'] = opt_orbit
    optimizers[f'{ip_name}_optics'] = opt_optics
    optimizers[f'{ip_name}_coupling'] = opt_coupling

    # One user knob turns on compensation solenoid, doublet rotations, and all
    # generated correction knobs for this IP.
    line[f'on_sol_corr_{ip_name}'] = 0
    line[f'on_comp_sol_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_rot_doublet_right_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_rot_doublet_left_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_sol_orbit_corr_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_sol_optics_corr_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_sol_coupling_corr_{ip_name}'] = f'on_sol_corr_{ip_name}'

    # Leave the main solenoid off while preparing the next IP.
    line[f'on_sol_{ip_name}'] = 0


######################
# Save corrected line #
######################

line.cycle('ipa')

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 0
    line[f'on_sol_corr_{ip_name}'] = 0

tw_off = line.twiss4d(strengths=True, zero_at='ipg')

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 1
    line[f'on_sol_corr_{ip_name}'] = 1

tw_on_corr = line.twiss4d(strengths=True, zero_at='ipg')

env.to_json(OUTPUT_LATTICE_JSON)
print(f'Wrote {OUTPUT_LATTICE_JSON}')


################
# Check plots  #
################

plt.close('all')

fig1 = plt.figure(1)
tw_on_corr.rows[-20:20:'s'].plot('betx2 bety1', figure=fig1)

fig2 = plt.figure(2)
tw_on_corr.rows[-20:20:'s'].plot('x y', figure=fig2)

fig3 = plt.figure(3)
ax = fig3.add_subplot(3, 1, 1)
tw_off.plot(ax=ax)

ax2 = fig3.add_subplot(3, 1, 2, sharex=ax)
ax2.plot(tw_off.s, tw_off.muy - tw_on_corr.muy, label='muy error')
ax2.legend(loc='best')

ax3 = fig3.add_subplot(3, 1, 3, sharex=ax)
ax3.plot(tw_off.s, tw_off.muy)
ax3.plot(tw_on_corr.s, tw_on_corr.muy, label='muy with solenoid')
ax3.legend(loc='best')

plt.show()
