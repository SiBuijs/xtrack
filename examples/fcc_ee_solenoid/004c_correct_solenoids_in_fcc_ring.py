from pathlib import Path
import argparse
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
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
args = parser.parse_args()

FIELD_TAG = field_tag(args.b0)
# The order tag only applies to SplineBoris (VariableSolenoid is linear).
ORDER_TAG = order_tag(args.max_transverse_order)
_MODEL_LATTICE_PATHS = {
    'splineboris': (
        f'temp_fcc_ee_lcc_splineboris_solenoids_{FIELD_TAG}{ORDER_TAG}.json',
        f'fccee_z_lcc_splineboris_solenoids_coupling_corrected_{FIELD_TAG}{ORDER_TAG}.json',
    ),
    'varsol': (
        f'temp_fcc_ee_lcc_varsol_solenoids_{FIELD_TAG}.json',
        f'fccee_z_lcc_varsol_solenoids_coupling_corrected_{FIELD_TAG}.json',
    ),
}

INPUT_LATTICE_JSON = HERE / _MODEL_LATTICE_PATHS[args.model][0]
OUTPUT_LATTICE_JSON = HERE / _MODEL_LATTICE_PATHS[args.model][1]

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']

# get_nonlinear_chromaticity lives in the sibling nonlinear_tunes example.
sys.path.insert(0, str(HERE.parent / 'nonlinear_tunes'))
from detuning import get_nonlinear_chromaticity  # noqa: E402


def report_nonlinear_chromaticity(line, label):
    """Print and return Q' and d2q = Q''/4 (detuning.py's convention).
    Returns None if the off-momentum twiss fails."""
    try:
        chrom = get_nonlinear_chromaticity(
            line, npoints=21, order=2)
    except Exception as exc:  # noqa: BLE001 -- off-momentum twiss can fail
        print(f'  {label:38s} chromaticity unavailable ({type(exc).__name__}: '
              f'{exc})')
        return None
    dqx, dqy = float(chrom.qx_derivatives[1]), float(chrom.qy_derivatives[1])
    d2qx, d2qy = float(chrom.qx_derivatives[2]), float(chrom.qy_derivatives[2])
    print(f"  {label:38s} Q'x={dqx:11.4f}  Q'y={dqy:11.4f}   "
          f"d2qx={d2qx:13.4f}  d2qy={d2qy:13.4f}")
    # Polynomial coefficients of the fit Q(delta), for the plot below.
    factorials = np.array([math.factorial(ii) for ii in range(chrom.order + 1)])
    return dict(dqx=dqx, dqy=dqy, d2qx=d2qx, d2qy=d2qy,
                label=label,
                deltas=np.asarray(chrom.deltas),
                qx=np.asarray(chrom.qx), qy=np.asarray(chrom.qy),
                qx_coef=np.asarray(chrom.qx_derivatives) * factorials,
                qy_coef=np.asarray(chrom.qy_derivatives) * factorials)


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


###################################
# Load installed solenoid lattice #
###################################

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring


############################
# Correction configuration #
############################

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
    # Chromatic sextupole nearest to the IP, targeted by the optics match.
    'sext_left': 'sdm1l.7',
    'sext_right': 'sdm1r.0',
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
    'sext_left': 'sdm1l.1',
    'sext_right': 'sdm1r.2',
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
    'sext_left': 'sdm1l.3',
    'sext_right': 'sdm1r.4',
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
    'sext_left': 'sdm1l.5',
    'sext_right': 'sdm1r.6',
}


#############################
# Mid-bend trim quadrupoles #
#############################

# Matching the optics only at the straight-section boundaries leaves a local
# beta bump in between (bety at sdm1r.0: 1.47 m bare -> 310 m at 3 T). Thin
# trim quads at the centre of the three bends on each side of the IP give the
# optics match local handles to remove it.

BEND_MID_QUAD_PREFIX = 'qbmid_'


def bend_mid_quad_name(bend_name):
    """Element name of the trim quadrupole at the centre of `bend_name`."""
    return BEND_MID_QUAD_PREFIX + bend_name


# One insert for all quads, as each call rebuilds the whole line. Inserting
# at the bend centre cuts the bend in two.
_mid_quad_places = []
for ip_name in IP_NAMES:
    for bend_name in config[ip_name]['bend_for_mid_quad_correction']:
        mid_quad_name = bend_mid_quad_name(bend_name)
        # Thin, so the strength goes on knl[1] (k1 has no effect).
        env.elements[mid_quad_name] = xt.Multipole(knl=[0.0, 0.0], length=0.0)
        _mid_quad_places.append(env.place(mid_quad_name, at=0, from_=bend_name))

line.insert(_mid_quad_places)
print(f'Installed {len(_mid_quad_places)} mid-bend trim quadrupoles '
      f'({len(IP_NAMES)} IPs x '
      f'{len(config[IP_NAMES[0]]["bend_for_mid_quad_correction"])} bends)')

###############################
# Build one correction per IP #
###############################

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 0
    line[f'on_comp_sol_{ip_name}'] = 0


############################################
# Bare reference optics, all solenoids off #
############################################

# Each correction is a pair of half-straight matches starting from the bare
# optics at the IP: forward to the end of the straight, backward to the start.
tw0 = line.twiss4d(strengths=True)
init_ip = {ip_name: tw0.get_twiss_init(ip_name) for ip_name in IP_NAMES}

SIDES = ('left', 'right')


def unique_quadrupoles(table_part):
    """Names of the quadrupoles in a table slice, in order, without duplicates."""
    names = []
    for element_type, env_name in zip(
            table_part.element_type, table_part.env_name):
        if element_type == 'Quadrupole' and env_name not in names:
            names.append(env_name)
    return names


def doublet_to_sdy1_range(table_half, ip_name, side):
    """Start and end element, in beam order, of the QD0 -> sdy1 phase advance
    on this side of the IP."""
    names = np.asarray(table_half.name)
    sdy1 = [nn for nn, element_type in zip(names, table_half.element_type)
            if element_type == 'Sextupole' and nn.startswith('sdy1')]
    if not sdy1:
        raise SystemExit(f'No sdy1 sextupole found on the {side} side of '
                         f'{ip_name} -- the IR sextupole naming changed.')
    if side == 'right':
        return str(config[ip_name]['doublet_quad_right'][0]), str(sdy1[0])
    qd0_left = config[ip_name]['doublet_quad_left'][0]
    ii_qd0 = np.flatnonzero(names == qd0_left)[-1]
    return str(sdy1[0]), str(names[ii_qd0 + 1])


optimizers = {}
phase_ranges = {}
for ip_name in IP_NAMES:

    print(f'IP {ip_name}:')

    # Turn on only the solenoid system being corrected.
    line[f'on_sol_{ip_name}'] = 1
    line[f'on_comp_sol_{ip_name}'] = 1

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

    # Quads are listed right side first, bends left side first.
    quad_for_optics_correction = config[ip_name]['quad_for_optics_correction']
    n_quad_half = len(quad_for_optics_correction) // 2
    bend_for_mid_quad = config[ip_name]['bend_for_mid_quad_correction']
    n_bend_half = len(bend_for_mid_quad) // 2
    quads_per_side = {
        'right': quad_for_optics_correction[:n_quad_half],
        'left': quad_for_optics_correction[n_quad_half:],
    }
    bends_per_side = {
        'left': bend_for_mid_quad[:n_bend_half],
        'right': bend_for_mid_quad[n_bend_half:],
    }

    name_start = f'end_ds_start_straight_{ip_name}'
    name_end = f'end_straight_start_ds_{ip_name}'

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

    # Orbit corrector knobs. acb*1 is in the main solenoid (004b); the others
    # go on nearby quads and the compensation-solenoid correctors.
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

    table_for_skew = line.get_table()

    opt_orbit = {}
    opt_optics = {}
    opt_coupling = {}
    for side in SIDES:

        # The left half is twissed backward from the IP, so its targets are
        # at START.
        if side == 'right':
            orbit_range = dict(start=ip_name, end=f'dy_match_r_{ip_name}')
            optics_range = dict(start=ip_name, end=name_end)
            at_boundary = xt.END
            name_boundary = name_end
            table_half = table_for_skew.rows[ip_name:name_end]
        else:
            orbit_range = dict(start=f'dy_match_l_{ip_name}', end=ip_name)
            optics_range = dict(start=name_start, end=ip_name)
            at_boundary = xt.START
            name_boundary = name_start
            table_half = table_for_skew.rows[name_start:ip_name]
        sext_corr = config[ip_name][f'sext_{side}']

        # Orbit and vertical dispersion at the edge of the solenoid region.
        opt_orbit[side] = line.match_knob(
            knob_name=f'on_sol_orbit_corr_{side}_{ip_name}',
            name=f'{ip_name}/orbit_{side}',
            run=False,
            assert_within_tol=False,
            init=init_ip[ip_name],
            **orbit_range,
            vary=xt.VaryList([
                f'acbh1_sol_{side}_{ip_name}', f'acbv1_sol_{side}_{ip_name}',
                f'acbh2_sol_{side}_{ip_name}', f'acbh3_sol_{side}_{ip_name}',
                f'acbh4_sol_{side}_{ip_name}', f'acbh5_sol_{side}_{ip_name}',
                f'acbh6_sol_{side}_{ip_name}', f'acbv2_sol_{side}_{ip_name}',
                f'acbv3_sol_{side}_{ip_name}', f'acbv4_sol_{side}_{ip_name}',
                f'acbv5_sol_{side}_{ip_name}', f'acbv6_sol_{side}_{ip_name}',
            ], step=1e-6),
            targets=[
                xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0,
                             at=at_boundary),
            ])

        # Normal quadrupole trims on this side of the straight.
        k1_knobs = []
        for nn in quads_per_side[side]:
            nn_knob = f'k1_{nn}_sol_corr'
            env[nn_knob] = 0
            env[nn].k1 += env.ref[nn_knob]
            k1_knobs.append(nn_knob)

        # Mid-bend trim quads (integrated k1l on knl[1]).
        for bend_name in bends_per_side[side]:
            nn = bend_mid_quad_name(bend_name)
            nn_knob = f'k1_{nn}_sol_corr'
            env[nn_knob] = 0
            env[nn].knl[1] += env.ref[nn_knob]
            k1_knobs.append(nn_knob)

        # Optics and horizontal dispersion at the straight-section boundary,
        # and betx/bety at the nearest sdm1 sextupole.
        ph_start, ph_end = doublet_to_sdy1_range(table_half, ip_name, side)
        phase_ranges[ip_name, side] = (ph_start, ph_end)
        opt_optics[side] = line.match_knob(
            knob_name=f'on_sol_optics_corr_{side}_{ip_name}',
            name=f'{ip_name}/optics_{side}',
            run=False,
            assert_within_tol=False,
            init=init_ip[ip_name],
            **optics_range,
            n_steps_max=100,
            vary=xt.VaryList(k1_knobs, step=1e-7),
            targets=[
                xt.TargetSet(
                    betx=tw0['betx', name_boundary],
                    bety=tw0['bety', name_boundary],
                    tol=1e-5,
                    at=at_boundary),
                xt.TargetSet(
                    alfx=tw0['alfx', name_boundary],
                    alfy=tw0['alfy', name_boundary],
                    tol=1e-7,
                    at=at_boundary),
                xt.TargetSet(
                    dx=tw0['dx', name_boundary],
                    dpx=tw0['dpx', name_boundary],
                    tol=1e-8,
                    at=at_boundary),
                xt.TargetSet(
                    betx=tw0['betx', sext_corr],
                    bety=tw0['bety', sext_corr],
                    tol=1e-5,
                    tag='sext',
                    at=sext_corr),
            ])

        # Coupling and vertical dispersion, with skew trims on all quads in
        # this half.
        k1s_knobs = []
        for nn in unique_quadrupoles(table_half):
            nn_knob = f'k1s_{nn}_sol_coupling_corr'
            env[nn_knob] = 0
            env[nn].k1s += env.ref[nn_knob]
            k1s_knobs.append(nn_knob)

        opt_coupling[side] = line.match_knob(
            knob_name=f'on_sol_coupling_corr_{side}_{ip_name}',
            name=f'{ip_name}/coupling_{side}',
            run=False,
            assert_within_tol=False,
            init=init_ip[ip_name],
            **optics_range,
            n_steps_max=100,
            vary=xt.VaryList(k1s_knobs, step=1e-7),
            targets=[
                xt.TargetSet(betx2=0, bety1=0, at=at_boundary, tol=5e-5),
                xt.TargetSet(alfx2=0, alfy1=0, at=at_boundary, tol=1e-6),
                xt.TargetSet(dy=0, at=at_boundary, tol=5e-5),
                xt.TargetSet(dpy=0, at=at_boundary, tol=1e-7),
            ])

    # The two halves are independent. The optics and coupling matches are
    # underdetermined, so rcond cuts their near-null directions.
    for side in SIDES:
        opt_orbit[side].solve()
        # First optics solve in two stages: boundary targets only, then all.
        opt_optics[side].disable(target='sext')
        opt_optics[side].solve(rcond=1e-6)
        opt_optics[side].enable(target='sext')
        opt_optics[side].solve(rcond=1e-6)
        opt_coupling[side].solve(rcond=3e-3)
        opt_orbit[side].solve()
        opt_optics[side].solve(rcond=1e-6)
        opt_coupling[side].solve(rcond=3e-3)
        opt_orbit[side].solve()
        opt_optics[side].solve(rcond=1e-6)

    # Show unconverged targets. Must run before generate_knob().
    for side in SIDES:
        for opt in (opt_orbit[side], opt_optics[side], opt_coupling[side]):
            if not all(opt.target_status(ret=True).tol_met):
                print(f'{opt.name} did not converge:')
                opt.target_mismatch()

    for side in SIDES:
        opt_orbit[side].generate_knob()
        opt_optics[side].generate_knob()
        opt_coupling[side].generate_knob()
        optimizers[f'{ip_name}_orbit_{side}'] = opt_orbit[side]
        optimizers[f'{ip_name}_optics_{side}'] = opt_optics[side]
        optimizers[f'{ip_name}_coupling_{side}'] = opt_coupling[side]

    # One knob for the compensation solenoids, doublet rotation and all
    # corrections of this IP.
    line[f'on_sol_corr_{ip_name}'] = 0
    line[f'on_comp_sol_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_rot_doublet_right_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_rot_doublet_left_{ip_name}'] = f'on_sol_corr_{ip_name}'
    for kind in ('orbit', 'optics', 'coupling'):
        line[f'on_sol_{kind}_corr_{ip_name}'] = f'on_sol_corr_{ip_name}'
        for side in SIDES:
            line[f'on_sol_{kind}_corr_{side}_{ip_name}'] = (
                f'on_sol_{kind}_corr_{ip_name}')

    # Leave the main solenoid off while preparing the next IP.
    line[f'on_sol_{ip_name}'] = 0


#######################
# Save corrected line #
#######################

line.cycle('ipa')

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 0
    line[f'on_sol_corr_{ip_name}'] = 0

tw_off = line.twiss4d(strengths=True, zero_at='ipg')

# Chromaticity of the full ring, before and after the correction.
print()
print('--- Non-linear chromaticity (21-point delta sweep, order 2) ---')
chrom_off = report_nonlinear_chromaticity(
    line, 'BEFORE: all solenoids off')

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 1
    line[f'on_sol_corr_{ip_name}'] = 1

tw_on_corr = line.twiss4d(strengths=True, zero_at='ipg')

# Optics at the targeted sextupoles and orbit at the IP.
print()
print('--- Optics at the targeted sextupoles: bare -> solenoids on, corrected ---')
for ip_name in IP_NAMES:
    for side in SIDES:
        sext_corr = config[ip_name][f'sext_{side}']
        print(f'  {ip_name} {side:5s} {sext_corr:8s} '
              f'betx {tw0["betx", sext_corr]:9.5f} -> '
              f'{tw_on_corr["betx", sext_corr]:9.5f}   '
              f'bety {tw0["bety", sext_corr]:9.5f} -> '
              f'{tw_on_corr["bety", sext_corr]:9.5f}')
    print(f'  {ip_name} orbit at IP: x={tw_on_corr["x", ip_name]: .3e}  '
          f'y={tw_on_corr["y", ip_name]: .3e}')

# QD0 -> sdy1 phase advance (not targeted).
print()
print('--- Phase advance QD0 <-> sdy1: bare -> solenoids on, corrected ---')
for (ip_name, side), (ph_start, ph_end) in phase_ranges.items():
    parts = []
    for mu, q0, q1 in (('mux', tw0.qx, tw_on_corr.qx),
                       ('muy', tw0.qy, tw_on_corr.qy)):
        mu_bare = np.mod(tw0[mu, ph_end] - tw0[mu, ph_start], q0)
        mu_corr = np.mod(tw_on_corr[mu, ph_end] - tw_on_corr[mu, ph_start], q1)
        parts.append(f'{mu} {mu_bare:.6f} -> {mu_corr:.6f} '
                     f'({mu_corr - mu_bare:+.1e})')
    print(f'  {ip_name} {side:5s} {ph_start} -> {ph_end}: ' + '   '.join(parts))

chrom_on_corr = report_nonlinear_chromaticity(
    line, 'AFTER:  solenoids on, corrections on')

if chrom_off is not None and chrom_on_corr is not None:
    print(f'  {"change (AFTER - BEFORE)":38s} '
          f'd(d2qx)={chrom_on_corr["d2qx"] - chrom_off["d2qx"]:+13.4f}  '
          f'd(d2qy)={chrom_on_corr["d2qy"] - chrom_off["d2qy"]:+13.4f}')

env.to_json(OUTPUT_LATTICE_JSON)
print(f'Wrote {OUTPUT_LATTICE_JSON}')


###############
# Check plots #
###############

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

# Q(delta) with its fit (top), and with the constant and linear part removed
# so the quadratic term is visible (bottom).
_chrom_cases = [(cc, st) for cc, st in ((chrom_off, '--'),
                                        (chrom_on_corr, '-'))
                if cc is not None]
if _chrom_cases:
    fig4, axs4 = plt.subplots(2, 2, sharex=True, num=4, figsize=(11, 7))
    # Delta range of the sweep done by detuning.py.
    _d_lim = max(abs(cc['deltas']).max() for cc, _ in _chrom_cases)
    _delta_fine = np.linspace(-_d_lim, _d_lim, 401)
    for _col, _plane in enumerate(('x', 'y')):
        _ax_abs, _ax_res = axs4[0, _col], axs4[1, _col]
        for _chrom, _style in _chrom_cases:
            _q = _chrom[f'q{_plane}']
            _coef = _chrom[f'q{_plane}_coef']
            _d2q = _chrom[f'd2q{_plane}']
            _tag = _chrom['label'].split(':')[0]
            _fit = np.polynomial.polynomial.polyval(_delta_fine, _coef)
            _line, = _ax_abs.plot(_delta_fine, _fit, _style,
                                  label=f'{_tag} fit')
            _ax_abs.plot(_chrom['deltas'], _q, 'o', ms=4,
                         color=_line.get_color())
            # Constant and linear part of the fit.
            _lin_fine = _coef[0] + _coef[1] * _delta_fine
            _lin_pts = _coef[0] + _coef[1] * _chrom['deltas']
            _ax_res.plot(_delta_fine, _fit - _lin_fine, _style,
                         color=_line.get_color(),
                         label=f'{_tag}: d2q{_plane}={_d2q:.4g} '
                               f"(Q''={4 * _d2q:.4g})")
            _ax_res.plot(_chrom['deltas'], _q - _lin_pts, 'o', ms=4,
                         color=_line.get_color())
        _ax_abs.set_ylabel(f'Q{_plane}')
        _ax_abs.set_title(f'Q{_plane} vs delta')
        _ax_res.set_ylabel(f'Q{_plane} - (Q{_plane}0 + Q\'{_plane} delta)')
        _ax_res.set_xlabel('delta')
        _ax_res.set_title('quadratic part only')
        for _a in (_ax_abs, _ax_res):
            _a.grid(alpha=0.3)
            _a.legend(loc='best', fontsize=8)
            # Keep the delta tick labels from overlapping.
            _a.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    fig4.suptitle(f'Non-linear chromaticity, {FIELD_TAG} (21 points)')
    fig4.tight_layout()

plt.show()
