from pathlib import Path
import argparse
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
parser.add_argument(
    '--output-tag', default='',
    help='Optional extra suffix appended to BOTH the input temp lattice and '
         'the output corrected lattice filenames (e.g. "mainscale"). Must '
         'match the --output-tag passed to 004b_install_solenoids_in_fcc_'
         'ring.py. Empty (default) keeps the standard filenames.')
parser.add_argument(
    '--no-chromaticity', action='store_true',
    help='Skip the second-order chromaticity report before/after correction. '
         'Each report costs npoints+1 full-ring 4D twisses, so the two reports '
         'are ~44 extra twisses.')
parser.add_argument(
    '--chromaticity-points', type=int, default=21, metavar='N',
    help='Number of off-momentum points in the chromaticity fit '
         '(default: 21, matching 004j).')
parser.add_argument(
    '--optics-step', type=float, default=None, metavar='H',
    help='Override OPTICS_STEP, the finite-difference step of the optics '
         'vary knobs. For tuning the matching setup.')
parser.add_argument(
    '--optics-rcond', type=float, default=None, metavar='R',
    help='Override OPTICS_RCOND, the singular-value cutoff of the half-'
         'straight optics solves. 0 disables truncation (xdeps default). For '
         'tuning the matching setup.')
parser.add_argument(
    '--ips', default=None, metavar='LIST',
    help='Comma-separated subset of IPs to correct, e.g. "ipg" or "ipd,ipg". '
         'For tuning the matching setup: the other IPs keep their solenoids '
         'off, so the resulting lattice is NOT a valid corrected ring and the '
         'chromaticity numbers are not comparable with a full run. Default: '
         'all four.')
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
ALL_IP_NAMES = list(IP_NAMES)
if args.ips is not None:
    IP_NAMES = [nn.strip() for nn in args.ips.split(',') if nn.strip()]
    unknown = [nn for nn in IP_NAMES if nn not in ALL_IP_NAMES]
    if unknown:
        raise SystemExit(f'Unknown IP(s) {unknown}; choose from {ALL_IP_NAMES}.')
# A partial run leaves the other IPs uncorrected, so it must not overwrite the
# corrected lattice that the downstream scripts read.
WRITE_OUTPUT_LATTICE = IP_NAMES == ALL_IP_NAMES

# get_nonlinear_chromaticity lives in the sibling nonlinear_tunes example, not
# in xtrack proper. Same import idiom and same function as 004j, so the d2qx /
# d2qy printed here are directly comparable with 004j's scan columns.
sys.path.insert(0, str(HERE.parent / 'nonlinear_tunes'))
from detuning import get_nonlinear_chromaticity  # noqa: E402


def report_nonlinear_chromaticity(line, label):
    """Print Q', Q'' for the ring in its current knob state.

    The suspected payoff of the whole local-optics correction is the
    second-order chromaticity, so it is measured directly rather than inferred
    from the beta bump. Uses the order=2 fit from detuning.py, whose
    q{x,y}_derivatives[n] is already divided by n! -- so index 1 is Q' and
    index 2 is Q''/2, exactly what 004j records as d2qx/d2qy.

    Off-momentum twisses need a closed orbit, which the solenoid-on /
    correction-off state does not have, so a failure is reported rather than
    raised.
    """
    if args.no_chromaticity:
        return None
    try:
        chrom = get_nonlinear_chromaticity(
            line, npoints=args.chromaticity_points, order=2)
    except Exception as exc:  # noqa: BLE001 -- off-momentum twiss can fail
        print(f'  {label:38s} chromaticity unavailable ({type(exc).__name__}: '
              f'{exc})')
        return None
    dqx, dqy = float(chrom.qx_derivatives[1]), float(chrom.qy_derivatives[1])
    d2qx, d2qy = float(chrom.qx_derivatives[2]), float(chrom.qy_derivatives[2])
    print(f"  {label:38s} Q'x={dqx:11.4f}  Q'y={dqy:11.4f}   "
          f"d2qx={d2qx:13.4f}  d2qy={d2qy:13.4f}")
    return dict(dqx=dqx, dqy=dqy, d2qx=d2qx, d2qy=d2qy)


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
    # Chromatic sextupole nearest to the IP on each side of the straight;
    # targeted by the half-straight optics matches below.
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


###############################################################################
# Mid-bend trim quadrupoles                                                   #
#                                                                             #
# Matching the optics only at the straight-section boundaries, ~1360 m either #
# side of the IP, leaves a purely local beta-beat in between unpenalised.     #
# With the detector solenoid on exactly such a bump appears downstream of     #
# every IP: bety at sdm1r.0 goes 1.47 m (bare) -> 2.87 m (2 T) -> 310 m (3 T).#
# That vertical beta sits on a chromatic sextupole and was the suspected      #
# driver of the second-order chromaticity. It turned out not to be: sdm1 sits #
# at beta of a few metres; the driver is the QD0 -> sdy1 phase advance, which #
# the optics matches below leave free (measured, not targeted -- see the note #
# at the optics match and claude_notes/09).                                   #
#                                                                             #
# To give the half-straight optics matches local handles between the IP and  #
# the nearest sextupole, each of the six bends framing the IP                 #
# (config[ip]['bend_for_mid_quad_correction']: three upstream, three          #
# downstream) is cut in half here and a zero-length trim quadrupole is placed #
# at the cut. They are cut in the full ring, so the chromaticity measured at  #
# the end includes them. All of this is done once, up front, before any       #
# optics is computed; everything after this section is name-based.           #
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

def solve_and_report(opt, label, **solve_kwargs):
    """Run ``opt.solve(**solve_kwargs)`` with a banner saying what is matched.

    xdeps only prints ``Optimize - start/end penalty``, which makes a log with
    eight solves per IP impossible to attribute. This wraps each solve in a
    header naming the knob, the matching range and the pass number, and
    follows it with an explicit OK/INCOMPLETE verdict plus the mismatching
    targets, so a stalled solve is identifiable from the log alone.
    """
    print()
    print('=' * 78)
    print(f'MATCHING: {label}')
    print(f'    knob    : {opt.knob_name}')
    print(f'    vary    : {len(opt.vary)} knobs, {len(opt.targets)} targets')
    if solve_kwargs:
        print(f'    solve   : {solve_kwargs}')
    print('=' * 78)

    try:
        opt.solve(**solve_kwargs)
    except Exception:
        print(f'MATCH FAILED (exception raised): {label}')
        print('    target state below is the restored start point '
              '(restore_if_fail=True):')
        opt.target_mismatch()
        raise

    tt = opt.target_status(ret=True)
    tol_met = np.asarray(tt.tol_met, dtype=bool)
    n_bad = int((~tol_met).sum())
    if n_bad:
        print(f'MATCH INCOMPLETE: {label} -- {n_bad}/{len(tol_met)} targets '
              'outside tolerance:')
        opt.target_mismatch()
    else:
        print(f'MATCH OK: {label} -- all {len(tol_met)} targets within '
              'tolerance.')


###############################################
# Bare reference optics, all solenoids off  #
###############################################

# Every correction below is a pair of half-straight matches that start from
# the bare-ring optics at the IP: the downstream half is twissed forward from
# the IP to the end of the straight, the upstream half backward from the IP to
# the start of the straight. Both halves are pinned to the same bare Twiss
# vector at the IP, so they are independent of each other, and together they
# map the bare optics at the start of the straight onto the bare optics at the
# end. The corrections of IPs already done are gated by on_sol_corr_{ip} = 0,
# so this single bare twiss is the reference for all four IPs.
tw0 = line.twiss4d(strengths=True)
init_ip = {ip_name: tw0.get_twiss_init(ip_name) for ip_name in IP_NAMES}

SIDES = ('left', 'right')

# Broyden setting of the optics solves. False: a full finite-difference
# Jacobian at every step, the slowest but most robust option. An integer n
# takes a full Jacobian at the first step of every solve() and every n + 1
# steps, with rank-1 updates in between (see the solve loop); True must not be
# used. False was needed by the phase targets that have since been removed, so
# a small integer may now be worth trying for speed.
OPTICS_BROYDEN = False

# Regularisation of the half-straight optics solves.
#
# The optics match has 15 vary knobs against 9 targets, so 6 directions in knob
# space are unconstrained, and the constrained ones are very unevenly weighted:
# the singular values of the 9x15 Jacobian span [1.4e6 ... 3.9e-2], a condition
# number of 3.7e7, with a clean gap of ~200 between the 7th (9.6) and the 8th
# (4.9e-2). Solving with the xdeps default rcond=1e-14 inverts those two
# near-null directions and multiplies the residual by 1/sigma, so once the
# residual is small the Newton step is enormous: in the run of 2026-09-16 the
# ipd-left pass-3 solve went penalty 0.546 -> 80.0, backtracked only as far as
# 0.722 (still uphill), accepted it, and then jumped to 7.2e3. Over three
# passes the trims drifted to 26x their proper size (max |k1| 7.7e-2 against
# 2.9e-3 for a good solution), which moved the ring tune by 24 units and left
# the lattice unstable at delta = 1e-3.
#
# rcond=1e-6 cuts exactly the two directions below the gap and keeps the other
# seven; it is the same remedy already applied to the coupling solves below,
# whose Jacobian is rank-deficient for the same reason.
OPTICS_RCOND = 1e-6

# Hard bounds on the trims were tried as a second, solver-independent guard
# (a converged correction needs only |k1| ~ 3e-4 to 3e-3) and made things
# worse, so they are off by default. Measured on ipg, 3 T, end penalty of the
# three optics passes:
#
#     limits          rcond      left                      right
#     none            1e-6       9.8e-6  7.0e-6  5.6e-6    9.8e-5  3.4e-5  3.0e-5
#     +-1e-2/2e-3     none       5.3e-5  8.8e-3  2.2e-2    28.5    28.6    28.6
#     +-1e-2/2e-3     1e-6       9.8e-6  7.0e-6  5.6e-6    28.0    28.1    28.1
#
# i.e. the truncation alone is what fixes the solve, and the bounds by
# themselves break ipg right completely: the first unregularised Newton step
# slams several knobs into their limits, and the clipped point is one the
# solver never recovers from. Set them to a tuple to re-enable.
OPTICS_K1_LIMIT_QUAD = None
OPTICS_K1_LIMIT_MIDBEND = None

# Finite-difference step of the optics vary knobs.
OPTICS_STEP = 1e-7

# Also pin bety at the sdm1 sextupole in the half-straight optics match.
SEXT_BETY_TARGET = True

if args.optics_rcond is not None:
    OPTICS_RCOND = args.optics_rcond or None
if args.optics_step is not None:
    OPTICS_STEP = args.optics_step

# Omitted entirely rather than passed as None, so that rcond=0 reproduces the
# xdeps default path exactly.
OPTICS_SOLVE_KWARGS = (
    {'rcond': OPTICS_RCOND} if OPTICS_RCOND is not None else {})

# Tolerance on alfx/alfy at the straight-section boundary. alf is O(1.5) there,
# so this is a relative tolerance of ~1e-7; the previous 1e-8 was below what
# the solve can deliver once the weak Jacobian directions are truncated, and
# reported matches as INCOMPLETE at residues of 2-5e-8 that are physically
# irrelevant (a beta-beat of the same 1e-8 relative size).
OPTICS_ALF_TOL = 1e-7


def unique_quadrupoles(table_part):
    """Env names of the Quadrupoles in a table slice, in order, without
    duplicates. The mid-bend trims are zero-length Multipoles and are therefore
    not picked up as skew hosts."""
    names = []
    for element_type, env_name in zip(
            table_part.element_type, table_part.env_name):
        if element_type == 'Quadrupole' and env_name not in names:
            names.append(env_name)
    return names


def doublet_to_sdy1_range(table_half, ip_name, side):
    """(start, end) element names, in beam order, of the phase advance between
    the QD0 end facing the IP and the first slice of the sdy1 sextupole on this
    side of the IP: QD0R entrance -> sdy1r, or sdy1l -> QD0L exit."""
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

    # The config lists are ordered right side first for the quads and left
    # side (beam order) first for the bends.
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

    table_for_skew = line.get_table()

    opt_orbit = {}
    opt_optics = {}
    opt_coupling = {}
    labels = {}
    for side in SIDES:

        # Geometry of this half-straight. Both halves start from the bare
        # optics at the IP (init_ip); for the left half the IP is the END of
        # the range, so the twiss runs backward and the targets sit at START.
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

        # Match orbit and vertical dispersion at the edge of the solenoid
        # region on this side.
        labels['orbit', side] = (
            f'{ip_name} {side.upper()} ORBIT + vertical dispersion '
            f'(x/px/y/py/dy/dpy=0 at dy_match_{side[0]}_{ip_name}, '
            f'12 orbit correctors)')
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
        k1_knobs_quad = list(k1_knobs)

        # Mid-bend trim quads between the IP and the sextupole on this side.
        # These are zero-length, so the strength has to go on the integrated
        # knl[1]; k1 is dead at zero length -- same reasoning as
        # lattice_knobs.install_extra_sextupole. Units therefore differ from
        # the trims above (k1l [1/m] rather than k1 [1/m^2]), but over a
        # 25-60 m bend a k1l of 1e-7 is a distributed k1 of ~2-4e-9, so the
        # shared step=1e-7 in the VaryList below is a comparable perturbation
        # for both families.
        for bend_name in bends_per_side[side]:
            nn = bend_mid_quad_name(bend_name)
            nn_knob = f'k1_{nn}_sol_corr'
            env[nn_knob] = 0
            env[nn].knl[1] += env.ref[nn_knob]
            k1_knobs.append(nn_knob)
        k1_knobs_midbend = [nn for nn in k1_knobs if nn not in k1_knobs_quad]

        # Match optics and horizontal dispersion at the straight-section
        # boundary and betx/dy/dpy at the sdm1 sextupole nearest to the IP.
        #
        # NOTE: this leaves the QD0 -> sdy1 phase advance free, and that is
        # the known source of the huge second-order chromaticity. sdy1/sdy2
        # are a -I sextupole pair at bety ~ 1e4 m that cancels the vertical
        # chromatic kick of the final doublet only if that phase advance is
        # exactly pi; the boundary targets restore beta/alpha but say nothing
        # about it. The match leaves errors up to 7e-4 (ipj left) and Q''y ~
        # +1.8e4 (bare -149), at ~2.4e7 in Q''y per unit phase error per side.
        # The phase is measured (not targeted) in the report below.
        #
        # Adding mux/muy TargetRelPhaseAdvance here was tried and FAILED: mux
        # is essentially uncontrollable with these knobs, and the direction
        # that corrects muy drifts bety/dx at the sdy sextupoles, which breaks
        # the strength half of the -I cancellation and makes the chromatic
        # leak ~4x worse. Do not put it back as it was. See
        # claude_notes/09_correcting_the_q2y_source.md.
        ph_start, ph_end = doublet_to_sdy1_range(table_half, ip_name, side)
        phase_ranges[ip_name, side] = (ph_start, ph_end)
        labels['optics', side] = (
            f'{ip_name} {side.upper()} OPTICS + horizontal dispersion '
            f'(betx/bety/alfx/alfy/dx/dpx at {name_boundary}, betx/dy/dpy at '
            f'{sext_corr}{" +bety" if SEXT_BETY_TARGET else ""}; '
            f'normal-quad + mid-bend trims)')
        opt_optics[side] = line.match_knob(
            knob_name=f'on_sol_optics_corr_{side}_{ip_name}',
            name=f'{ip_name}/optics_{side}',
            run=False,
            assert_within_tol=False,
            init=init_ip[ip_name],
            **optics_range,
            n_steps_max=100,
            vary=[
                xt.VaryList(k1_knobs_quad, step=OPTICS_STEP,
                            limits=OPTICS_K1_LIMIT_QUAD),
                xt.VaryList(k1_knobs_midbend, step=OPTICS_STEP,
                            limits=OPTICS_K1_LIMIT_MIDBEND),
            ],
            targets=[
                xt.TargetSet(
                    betx=tw0['betx', name_boundary],
                    bety=tw0['bety', name_boundary],
                    tol=1e-5,
                    at=at_boundary),
                xt.TargetSet(
                    alfx=tw0['alfx', name_boundary],
                    alfy=tw0['alfy', name_boundary],
                    tol=OPTICS_ALF_TOL,
                    at=at_boundary),
                xt.TargetSet(
                    dx=tw0['dx', name_boundary],
                    dpx=tw0['dpx', name_boundary],
                    tol=1e-8,
                    at=at_boundary),
                xt.TargetSet(
                    betx=tw0['betx', sext_corr],
                    bety=tw0['bety', sext_corr],
                    #dy=tw0['dy', sext_corr],
                    tol=1e-5,
                    at=sext_corr),
                # bety at sdm1 is left free by the targets above and drifts
                # hard: 1.47 -> 23-25 m in the 3 T run of 2026-09-16. Pinning
                # it spends one of the 6 null-space directions; whether that
                # buys anything in ring Q''y is what SEXT_BETY_TARGET tests.
                *([xt.Target('bety', tw0['bety', sext_corr], tol=1e-5,
                             at=sext_corr)] if SEXT_BETY_TARGET else []),
            ])

        # Skew quadrupole knobs for the linear-coupling/vertical-dispersion
        # correction: all quadrupoles on this side, IP to boundary.
        k1s_knobs = []
        for nn in unique_quadrupoles(table_half):
            nn_knob = f'k1s_{nn}_sol_coupling_corr'
            env[nn_knob] = 0
            env[nn].k1s += env.ref[nn_knob]
            k1s_knobs.append(nn_knob)

        labels['coupling', side] = (
            f'{ip_name} {side.upper()} COUPLING + vertical dispersion '
            f'(betx2/bety1/alfx2/alfy1/dy/dpy=0 at {name_boundary}; '
            f'skew-quad trims)')
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

    # The two halves share no knobs and are pinned to the same bare optics at
    # the IP, so each side is iterated on its own. The coupling knobs have
    # ~40 skew-quad vary knobs against 6 targets, so the Jacobian is heavily
    # rank-deficient; truncating small singular values (rcond) keeps the
    # pseudo-inverse from chasing numerically-noisy near-null directions.
    # The optics matches use Broyden rank-1 Jacobian updates between full
    # finite-difference Jacobians (one twiss per vary knob, 15 per half).
    # broyden=True is not usable here: xdeps then builds a single Jacobian at
    # the first step and keeps updating it, also across solve() calls, so
    # later passes start from a Jacobian that predates the orbit and coupling
    # solves. That stalled at penalty ~1e-3 when the phase targets were in, and
    # a pass starting at the knob point of the last Jacobian divides by
    # |dx|^2 = 0 (NaN Jacobian, "SVD did not converge"). An integer n takes a
    # full Jacobian at step 0 of every solve() and every n+1 steps after.
    for side in SIDES:
        solve_and_report(opt_orbit[side], labels['orbit', side] + ' [pass 1/3]')
        solve_and_report(opt_optics[side], labels['optics', side] + ' [pass 1/3]',
                         broyden=OPTICS_BROYDEN, **OPTICS_SOLVE_KWARGS)
        solve_and_report(opt_coupling[side],
                         labels['coupling', side] + ' [pass 1/2]', rcond=3e-3)
        solve_and_report(opt_orbit[side], labels['orbit', side] + ' [pass 2/3]')
        solve_and_report(opt_optics[side], labels['optics', side] + ' [pass 2/3]',
                         broyden=OPTICS_BROYDEN, **OPTICS_SOLVE_KWARGS)
        solve_and_report(opt_coupling[side],
                         labels['coupling', side] + ' [pass 2/2]', rcond=3e-3)
        solve_and_report(opt_orbit[side], labels['orbit', side] + ' [pass 3/3]')
        solve_and_report(opt_optics[side], labels['optics', side] + ' [pass 3/3]',
                         broyden=OPTICS_BROYDEN, **OPTICS_SOLVE_KWARGS)

    # Final state of every knob after the iterate pass. Must run before
    # generate_knob(): target_status() afterwards evaluates the optimizer on
    # the already-generated knob state and corrupts it.
    print()
    print(f'--- IP {ip_name}: summary of generated correction knobs ---')
    for side in SIDES:
        for _knob_label, _opt in (('orbit', opt_orbit[side]),
                                  ('optics', opt_optics[side]),
                                  ('coupling', opt_coupling[side])):
            _status = _opt.target_status(ret=True)
            _tol_met = np.asarray(_status.tol_met, dtype=bool)
            _n_bad = int((~_tol_met).sum())
            _penalty = _opt.log()['penalty'][-1]
            # Largest trim the solve ended on. A converged half-straight needs
            # |k1| of a few 1e-4 to 3e-3; anything far above that means the
            # solve wandered off in a weak Jacobian direction rather than
            # correcting anything (see OPTICS_RCOND).
            _kmax = max((abs(line.vars[_vv.name]._value) for _vv in _opt.vary),
                        default=0.0)
            print(f'    {_knob_label:9s} {_opt.knob_name:38s} '
                  f'penalty={_penalty:.4g}  max|knob|={_kmax:.2e}  '
                  f'{len(_tol_met) - _n_bad}/{len(_tol_met)} targets in tol')
            if _n_bad:
                print(f'WARNING: {_opt.knob_name} did not fully converge to '
                      f'tolerance; using best point found.')
                _opt.target_mismatch()

    for side in SIDES:
        opt_orbit[side].generate_knob()
        opt_optics[side].generate_knob()
        opt_coupling[side].generate_knob()
        optimizers[f'{ip_name}_orbit_{side}'] = opt_orbit[side]
        optimizers[f'{ip_name}_optics_{side}'] = opt_optics[side]
        optimizers[f'{ip_name}_coupling_{side}'] = opt_coupling[side]

    # One user knob turns on compensation solenoid, doublet rotations, and all
    # generated correction knobs for this IP. The per-side knobs hang off the
    # per-IP on_sol_{orbit,optics,coupling}_corr_{ip} knobs, which keeps the
    # knob names used by the downstream scripts.
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


######################
# Save corrected line #
######################

line.cycle('ipa')

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 0
    line[f'on_sol_corr_{ip_name}'] = 0

tw_off = line.twiss4d(strengths=True, zero_at='ipg')

# Second-order chromaticity before/after the correction. Ring-level rather
# than per-IP: the intermediate states inside the loop (one IP's solenoid on,
# its correction not yet generated) have no closed orbit, so an off-momentum
# twiss sweep cannot be run there.
if not args.no_chromaticity:
    print()
    print(f'--- Non-linear chromaticity '
          f'({args.chromaticity_points}-point delta sweep, order 2) ---')
chrom_off = report_nonlinear_chromaticity(
    line, 'BEFORE: all solenoids off')

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 1
    line[f'on_sol_corr_{ip_name}'] = 1

tw_on_corr = line.twiss4d(strengths=True, zero_at='ipg')

# Local check of what the half-straight optics matches target: the optics at
# the chromatic sextupoles either side of each IP, and the orbit at the IP
# (which both halves pin to the bare closed orbit).
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

# The phase advance the optics matches pin, measured on the closed ring. An
# error of 1e-6 in muy is worth ~25 in Q''y.
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

if WRITE_OUTPUT_LATTICE:
    env.to_json(OUTPUT_LATTICE_JSON)
    print(f'Wrote {OUTPUT_LATTICE_JSON}')
else:
    print(f'Partial run (--ips {",".join(IP_NAMES)}): '
          f'NOT writing {OUTPUT_LATTICE_JSON.name}')


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
