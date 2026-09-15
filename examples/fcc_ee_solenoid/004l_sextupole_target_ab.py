"""A/B the optics match with and without the sextupole targets.

004c's optics match (`on_sol_optics_corr_<ip>`) was changed in two ways at
once: local targets were added on the chromatic sextupole downstream of the
IP, and six zero-length trim quadrupoles were installed at the centre of the
near-IP bends to give the match local handles. At 3 T that match then stopped
converging (penalty ~57.75, essentially unchanged over its whole step budget),
and `generate_knob()` froze a near-zero knob into the saved lattice -- i.e.
the optics correction silently became a no-op.

This script runs the *same* IP through both versions of the optics match and
reports, side by side:

  * whether the optics match converged, and its final penalty,
  * the residual at the two straight-section boundary markers (what the match
    exists to restore),
  * bety at the chromatic sextupole and the worst vertical beta-beat anywhere
    in the straight (what the sextupole targets were added to fix).

The question it answers is whether the sextupole targets bought a flatter
bump at the cost of the boundary match, or whether they simply broke the
solve.

Both variants run on an identical lattice: the mid-bend trim quadrupoles are
installed in every case, and the 'old' variant just never varies them (they
are zero-length and zero-strength, so they have no effect when not varied).
That keeps the geometry out of the comparison.

The coupling match is held at the *current* 004c settings in every variant, so
that it is not a second confound; only the optics match differs.

Nothing is written to disk. The twiss tables are left as plain module-level
variables, so running this from IPython puts them straight in your namespace:

    In [1]: %run 004l_sextupole_target_ab.py --quick

    In [2]: tw_bare                # closed ring, all solenoids off
    In [3]: tw_old, tw_new         # closed ring, each variant's correction
    In [4]: s0 = tw_old['s', SEXT_CORR]
    In [5]: tw_old.rows[(s0-40):(s0+40):'s'].plot('betx bety')

    In [6]: res_old['d2qy'], res_new['sext_bety']

One `tw_<variant>` per variant run, plus `tw_<variant>_seg` -- the segment
twiss in the frame the match optimises in (propagated from the IP with bare
initial conditions), as opposed to the physical closed-ring solution in
`tw_<variant>`. `tw_<variant>` is None when that state has no closed orbit.
`res_<variant>` holds the summary numbers, and SEXT_CORR / NAME_START /
NAME_END the element names the match uses.

Examples
--------
    python 004l_sextupole_target_ab.py                  # ipa, old vs new, 3 T
    python 004l_sextupole_target_ab.py --ip ipd
    python 004l_sextupole_target_ab.py --b0 2           # the case that works
    python 004l_sextupole_target_ab.py --variants old new targets_only
    python 004l_sextupole_target_ab.py --dry-run        # validate setup only
"""

from pathlib import Path
import argparse
import ast
import sys

import numpy as np
import xtrack as xt

from solenoid_params import (
    BEND_MID_QUAD_PREFIX,
    MAIN_SOLENOID_B0,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    order_tag,
)

HERE = Path(__file__).parent
SOURCE_SCRIPT = HERE / '004c_correct_solenoids_in_fcc_ring.py'

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']

# get_nonlinear_chromaticity lives in the sibling nonlinear_tunes example, not
# in xtrack proper. Same import and same function as 004c/004j, so d2qx/d2qy
# here are directly comparable with theirs.
sys.path.insert(0, str(HERE.parent / 'nonlinear_tunes'))
from detuning import get_nonlinear_chromaticity  # noqa: E402

# Bare-lattice chromaticity is identical for every variant (same lattice, all
# solenoids off) but each variant reloads the lattice, so cache it rather than
# paying for the delta sweep once per variant.
_BARE_CHROM_CACHE = {}

# Populated as the variants run (see run_variant): tw_bare, tw_<variant>,
# tw_<variant>_seg and res_<variant> are injected into the module namespace so
# that `%run` from IPython leaves them in the interactive namespace. They are
# live TwissTable objects straight off line.twiss(), not copies.
tw_bare = None
SEXT_CORR = NAME_START = NAME_END = None


###############################################################################
# Variant definitions                                                         #
#                                                                             #
# Only the optics match differs between variants. 'old' reproduces the optics #
# match as it was before the sextupole work (boundary targets only, the 24    #
# named quads, step 1e-6, xtrack's default 20-step budget); 'new' is what     #
# 004c does today. 'targets_only' is the controlled middle case: the new knob #
# set and solver settings, but boundary targets only -- run it when old and   #
# new disagree, to say which of the two changes is responsible.               #
#                                                                             #
# assert_within_tol is False everywhere so that a failed match is *measured*  #
# rather than raised; the original would have raised here.                    #
###############################################################################

VARIANTS = {
    'old': dict(
        label='OLD  (boundary targets, 24 quads, step 1e-6)',
        vary_mid_bend_quads=False,
        target_sextupole=False,
        vary_step=1e-6,
        n_steps_max=None,          # xtrack default (20)
    ),
    'new': dict(
        label='NEW  (boundary+sextupole, 30 quads, step 1e-7)',
        vary_mid_bend_quads=True,
        target_sextupole=True,
        vary_step=1e-7,
        n_steps_max=30,
    ),
    'targets_only': dict(
        label='CTRL (boundary only, 30 quads, step 1e-7)',
        vary_mid_bend_quads=True,
        target_sextupole=False,
        vary_step=1e-7,
        n_steps_max=30,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--model', choices=['splineboris', 'varsol'], default='splineboris',
        help='Solenoid model to correct (default: splineboris).')
    add_b0_argument(parser, default=MAIN_SOLENOID_B0)
    add_max_order_argument(parser)
    parser.add_argument(
        '--output-tag', default='',
        help='Extra suffix on the input lattice filename; must match the '
             '--output-tag passed to 004b.')
    parser.add_argument(
        '--ip', default='ipa', choices=IP_NAMES,
        help='Which IP to A/B (default: ipa). One IP only -- this is a '
             'diagnostic, not a lattice builder; nothing is saved to a '
             'corrected lattice.')
    parser.add_argument(
        '--variants', nargs='+', default=['old', 'new'],
        choices=sorted(VARIANTS),
        help="Variants to run (default: old new).")
    parser.add_argument(
        '--quick', action='store_true',
        help='Run only orbit->optics once instead of 004c\'s full '
             'orbit/optics/coupling iterate sequence. Much faster, but not '
             'what 004c actually does.')
    parser.add_argument(
        '--no-chromaticity', action='store_true',
        help='Skip the second-order chromaticity sweeps (each is '
             '--chromaticity-points+1 full-ring 4D twisses).')
    parser.add_argument(
        '--chromaticity-points', type=int, default=21, metavar='N',
        help='Off-momentum points in the chromaticity fit (default: 21, '
             'matching 004c/004j).')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Load the lattice, install the trim quads and build the '
             'optimizers, then stop without solving. Validates the setup '
             'cheaply.')
    return parser.parse_args()


args = parse_args()

FIELD_TAG = field_tag(args.b0)
ORDER_TAG = order_tag(args.max_transverse_order)
OUT_TAG = f'_{args.output_tag}' if args.output_tag else ''
if args.model == 'splineboris':
    INPUT_LATTICE_JSON = HERE / (
        f'temp_fcc_ee_lcc_splineboris_solenoids_{FIELD_TAG}{ORDER_TAG}'
        f'{OUT_TAG}.json')
else:
    INPUT_LATTICE_JSON = HERE / (
        f'temp_fcc_ee_lcc_varsol_solenoids_{FIELD_TAG}{OUT_TAG}.json')


def load_config_from_004c():
    """Read the per-IP `config` dict straight out of 004c's source.

    The correction configuration (which quads are trimmed, which bends are
    cut, which sextupole is targeted) is a pure literal in 004c. Parsing it
    rather than copying it means this A/B cannot silently drift away from the
    script it is testing -- which matters, because a stale copy here would
    make the comparison meaningless in exactly the way that is hardest to
    notice. No code from 004c is executed.
    """
    tree = ast.parse(SOURCE_SCRIPT.read_text())
    config = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == 'config'):
            key = ast.literal_eval(target.slice)
            config[key] = ast.literal_eval(node.value)
    missing = [ip for ip in IP_NAMES if ip not in config]
    if missing:
        raise SystemExit(
            f'Could not read config for {missing} from {SOURCE_SCRIPT.name} '
            '-- its config block changed shape; update load_config_from_004c.')
    return config


def bend_mid_quad_name(bend_name):
    return BEND_MID_QUAD_PREFIX + bend_name


def install_mid_bend_quads(line, env, config):
    """Cut the near-IP bends in half and insert a zero-length trim quad.

    Same operation as 004c's "Mid-bend trim quadrupoles" section, including
    the single batched insert (line.insert rebuilds the whole line per call).
    """
    table = line.get_table()
    s_cuts = []
    for ip_name in IP_NAMES:
        for bend_name in config[ip_name]['bend_for_mid_quad_correction']:
            if bend_name not in table.name:
                raise SystemExit(
                    f'{INPUT_LATTICE_JSON.name} has no element {bend_name!r}.')
            if table['element_type', bend_name] != 'RBend':
                raise SystemExit(
                    f'{bend_name!r} is a '
                    f'{table["element_type", bend_name]!r}, not an unsliced '
                    'RBend.')
            s_cuts.append(table['s_center', bend_name])
    line.cut_at_s(sorted(s_cuts))

    places = []
    for ip_name in IP_NAMES:
        for bend_name in config[ip_name]['bend_for_mid_quad_correction']:
            downstream_half = f'{bend_name}..1'
            if downstream_half not in line.element_names:
                raise SystemExit(
                    f'cut_at_s did not split {bend_name!r}.')
            name = bend_mid_quad_name(bend_name)
            env.elements[name] = xt.Multipole(knl=[0.0, 0.0], length=0.0)
            places.append(env.place(name, at=0, from_=downstream_half,
                                    anchor='start', from_anchor='start'))
    line.insert(places)
    return len(places)


def measure_ksol_l_main_solenoid(line, env, ip_name):
    """Integrated solenoid strength over the IP region (copied from 004c)."""
    ksol_l = 0.0
    rigidity0 = line.particle_ref.rigidity0[0]
    region = line.get_table().rows[
        'dy_match_l_' + ip_name: 'dy_match_r_' + ip_name]
    for nn in region.name:
        element_type = region['element_type', nn]
        element = env.get(region['env_name', nn])
        if element_type == 'VariableSolenoid':
            ksol_l += element.ks_profile.mean() * element.length
        elif element_type == 'SplineBoris':
            ksol_l += (element.scale_b * element.bs[4] * element.length
                       / rigidity0)
    return ksol_l


def nonlinear_chromaticity(line, label):
    """Q' and Q''/2 for the ring in its current knob state, or None.

    Off-momentum twisses need a closed orbit, which a solenoid-on /
    correction-inert state does not necessarily have, so a failure is
    reported rather than raised.
    """
    if args.no_chromaticity:
        return None
    try:
        chrom = get_nonlinear_chromaticity(
            line, npoints=args.chromaticity_points, order=2)
    except Exception as exc:  # noqa: BLE001 -- off-momentum twiss can fail
        print(f'    {label}: chromaticity unavailable '
              f'({type(exc).__name__}: {exc})')
        return None
    out = dict(dqx=float(chrom.qx_derivatives[1]),
               dqy=float(chrom.qy_derivatives[1]),
               d2qx=float(chrom.qx_derivatives[2]),
               d2qy=float(chrom.qy_derivatives[2]))
    print(f"    {label}: Q'x={out['dqx']:.4f} Q'y={out['dqy']:.4f} "
          f"d2qx={out['d2qx']:.4f} d2qy={out['d2qy']:.4f}")
    return out


def build_corrections(line, env, config, ip_name, variant):
    """Wire up one IP exactly as 004c does, with `variant`'s optics match.

    Returns (opt_orbit, opt_optics, opt_coupling, tw0, context).
    """
    line.cycle(f'end_ds_start_straight_{ip_name}')
    tw0 = line.twiss4d(strengths=True)

    # Bare reference chromaticity, while every solenoid is still off.
    cache_key = (FIELD_TAG, ip_name)
    if cache_key not in _BARE_CHROM_CACHE:
        _BARE_CHROM_CACHE[cache_key] = nonlinear_chromaticity(
            line, 'BEFORE (bare, all solenoids off)')
    chrom_bare = _BARE_CHROM_CACHE[cache_key]

    line[f'on_sol_{ip_name}'] = 1
    line[f'on_comp_sol_{ip_name}'] = 1

    cfg = config[ip_name]

    # --- doublet rotation ------------------------------------------------
    ksol_l = measure_ksol_l_main_solenoid(line, env, ip_name)
    env[f'phi_rot_doublet_{ip_name}'] = (ksol_l / 2) / 2
    env[f'on_rot_doublet_left_{ip_name}'] = 1
    env[f'on_rot_doublet_right_{ip_name}'] = 1
    for nn in cfg['doublet_quad_left']:
        env[nn].rot_s_rad = (+env.ref[f'phi_rot_doublet_{ip_name}']
                             * env.ref[f'on_rot_doublet_left_{ip_name}'])
    for nn in cfg['doublet_quad_right']:
        env[nn].rot_s_rad = (-env.ref[f'phi_rot_doublet_{ip_name}']
                             * env.ref[f'on_rot_doublet_right_{ip_name}'])

    # --- orbit correctors -------------------------------------------------
    hosts_right = [cfg['corr_1_right_on_quad'], cfg['corr_2_right_on_quad'],
                   cfg['corr_3_right_on_quad'], cfg['corr_4_right_on_quad'],
                   f'corr_sol_right_{ip_name}']
    hosts_left = [cfg['corr_1_left_on_quad'], cfg['corr_2_left_on_quad'],
                  cfg['corr_3_left_on_quad'], cfg['corr_4_left_on_quad'],
                  f'corr_sol_left_{ip_name}']
    orbit_knobs = []
    for side, hosts in (('right', hosts_right), ('left', hosts_left)):
        for plane, attr in (('h', 'knl'), ('v', 'ksl')):
            for idx, host in enumerate(hosts, start=2):
                knob = f'acb{plane}{idx}_sol_{side}_{ip_name}'
                env[knob] = 0
                getattr(env[host], attr)[0] += env.ref[knob]
                orbit_knobs.append(knob)
    # acbh1/acbv1 were installed inside the main solenoid by 004b.
    vary_orbit = ([f'acbh1_sol_right_{ip_name}', f'acbv1_sol_right_{ip_name}']
                  + [k for k in orbit_knobs if k.endswith(f'right_{ip_name}')]
                  + [f'acbh1_sol_left_{ip_name}', f'acbv1_sol_left_{ip_name}']
                  + [k for k in orbit_knobs if k.endswith(f'left_{ip_name}')])

    opt_orbit = line.match_knob(
        knob_name=f'on_sol_orbit_corr_{ip_name}',
        name=f'{ip_name}/orbit', run=False,
        betx=tw0['betx', ip_name], bety=tw0['bety', ip_name],
        start=f'dy_match_l_{ip_name}', end=f'dy_match_r_{ip_name}',
        init_at=ip_name,
        assert_within_tol=False,
        vary=xt.VaryList(vary_orbit, step=1e-6),
        targets=[
            xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.END),
            xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.START),
        ])

    # --- normal-quad trims ------------------------------------------------
    k1_knobs = []
    for nn in cfg['quad_for_optics_correction']:
        knob = f'k1_{nn}_sol_corr'
        env[knob] = 0
        env[nn].k1 += env.ref[knob]
        k1_knobs.append(knob)

    # Always create the mid-bend knobs so the lattice is identical across
    # variants; only add them to the vary list when the variant asks for it.
    mid_bend_knobs = []
    for bend_name in cfg['bend_for_mid_quad_correction']:
        nn = bend_mid_quad_name(bend_name)
        knob = f'k1_{nn}_sol_corr'
        env[knob] = 0
        env[nn].knl[1] += env.ref[knob]
        mid_bend_knobs.append(knob)

    vary_optics = list(k1_knobs)
    if variant['vary_mid_bend_quads']:
        vary_optics += mid_bend_knobs

    name_start = f'end_ds_start_straight_{ip_name}'
    name_end = f'end_straight_start_ds_{ip_name}'
    sext_corr = cfg['sext_for_optics_correction']

    # --- skew trims (identical in every variant) --------------------------
    table = line.get_table()
    k1s_hosts = []
    for part in (table.rows[name_start:ip_name], table.rows[ip_name:name_end]):
        for element_type, env_name in zip(part.element_type, part.env_name):
            if (element_type == 'Quadrupole'
                    and not env_name.startswith(BEND_MID_QUAD_PREFIX)
                    and env_name not in k1s_hosts):
                k1s_hosts.append(env_name)
    k1s_knobs = []
    for nn in k1s_hosts:
        knob = f'k1s_{nn}_sol_coupling_corr'
        env[knob] = 0
        env[nn].k1s += env.ref[knob]
        k1s_knobs.append(knob)

    # --- optics match (the thing under test) ------------------------------
    targets = [
        xt.TargetSet(betx=tw0['betx', name_start], bety=tw0['bety', name_start],
                     tol=1e-5, at=xt.START),
        xt.TargetSet(alfx=tw0['alfx', name_start], alfy=tw0['alfy', name_start],
                     tol=1e-8, at=xt.START),
        xt.TargetSet(dx=tw0['dx', name_start], dpx=tw0['dpx', name_start],
                     tol=1e-8, at=xt.START),
        xt.TargetSet(betx=tw0['betx', name_end], bety=tw0['bety', name_end],
                     tol=1e-5, at=xt.END),
        xt.TargetSet(alfx=tw0['alfx', name_end], alfy=tw0['alfy', name_end],
                     tol=1e-8, at=xt.END),
        xt.TargetSet(dx=tw0['dx', name_end], dpx=tw0['dpx', name_end],
                     tol=1e-8, at=xt.END),
    ]
    if variant['target_sextupole']:
        targets += [
            xt.TargetSet(betx=tw0['betx', sext_corr],
                         bety=tw0['bety', sext_corr], tol=1e-5, at=sext_corr),
            xt.TargetSet(alfx=tw0['alfx', sext_corr],
                         alfy=tw0['alfy', sext_corr], tol=1e-6, at=sext_corr),
        ]

    optics_kwargs = {}
    if variant['n_steps_max'] is not None:
        optics_kwargs['n_steps_max'] = variant['n_steps_max']
    opt_optics = line.match_knob(
        knob_name=f'on_sol_optics_corr_{ip_name}',
        name=f'{ip_name}/optics', run=False,
        betx=tw0['betx', ip_name], bety=tw0['bety', ip_name],
        init_at=ip_name, start=name_start, end=name_end,
        assert_within_tol=False,
        vary=xt.VaryList(vary_optics, step=variant['vary_step']),
        targets=targets, **optics_kwargs)

    # --- coupling match (held at current 004c settings) -------------------
    opt_coupling = line.match_knob(
        knob_name=f'on_sol_coupling_corr_{ip_name}',
        name=f'{ip_name}/coupling', run=False,
        betx=tw0['betx', ip_name], bety=tw0['bety', ip_name],
        init_at=ip_name, start=name_start, end=name_end,
        n_steps_max=100, assert_within_tol=False,
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

    context = dict(tw0=tw0, chrom_bare=chrom_bare,
                   name_start=name_start, name_end=name_end,
                   sext_corr=sext_corr, n_vary_optics=len(vary_optics),
                   n_targets_optics=len(opt_optics.targets),
                   n_vary_coupling=len(k1s_knobs))
    return opt_orbit, opt_optics, opt_coupling, context


def solve_sequence(opt_orbit, opt_optics, opt_coupling, quick):
    """004c's solve order, or just orbit->optics when `quick`."""
    def run(opt, what, **kw):
        print(f'    solving {what} ...', flush=True)
        opt.solve(**kw)

    run(opt_orbit, 'orbit   [1/3]')
    run(opt_optics, 'optics  [1/3]')
    if quick:
        return
    run(opt_coupling, 'coupling [1/2]', rcond=3e-3)
    run(opt_orbit, 'orbit   [2/3]')
    run(opt_optics, 'optics  [2/3]')
    run(opt_coupling, 'coupling [2/2]', rcond=3e-3)
    run(opt_orbit, 'orbit   [3/3]')
    run(opt_optics, 'optics  [3/3]')


BOUNDARY_QUANTITIES = ['betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx']


def measure(line, opt_orbit, opt_optics, opt_coupling, context, ip_name):
    """Collect the numbers the A/B is about, after the solves have run."""
    tw0 = context['tw0']
    name_start, name_end = context['name_start'], context['name_end']
    sext_corr = context['sext_corr']

    out = {}
    for tag, opt in (('orbit', opt_orbit), ('optics', opt_optics),
                     ('coupling', opt_coupling)):
        status = opt.target_status(ret=True)
        tol_met = np.asarray(status.tol_met, dtype=bool)
        out[f'{tag}_penalty'] = float(opt.log()['penalty'][-1])
        out[f'{tag}_n_targets'] = int(tol_met.size)
        out[f'{tag}_n_in_tol'] = int(tol_met.sum())

    # Optics state in the same frame the match itself uses: propagated from
    # the IP with the bare initial conditions, over the matched range.
    # strengths=True so that .plot() can draw the lattice strip (Bend/Quad/
    # Sext bands) when you look at this table interactively.
    # NOTE: this is the CORRECTED lattice only because generate_knob() is
    # never called in this script -- after solve() the auxiliary variables
    # still hold their solved values, so the line sits at full correction
    # strength. Adding a generate_knob() call would reset the knob to
    # knob_value_start=0 and silently turn `tw` into the uncorrected optics.
    tw = line.twiss4d(strengths=True, init_at=ip_name,
                      betx=tw0['betx', ip_name], bety=tw0['bety', ip_name],
                      start=name_start, end=name_end)

    # Closed-ring solution for the same state. The segment twiss above is the
    # frame the match optimises in (propagated from the IP with bare initial
    # conditions); this one is the physical ring, and is what the
    # chromaticity sweep needs. It can fail when the correction is inert.
    try:
        tw_ring = line.twiss4d(strengths=True)
    except Exception as exc:  # noqa: BLE001 -- no closed orbit is a result
        print(f'    closed-ring twiss unavailable '
              f'({type(exc).__name__}: {exc})')
        tw_ring = None

    chrom_corr = (nonlinear_chromaticity(line, 'AFTER  (solenoid + correction)')
                  if tw_ring is not None else None)

    for marker in (name_start, name_end):
        side = 'start' if marker == name_start else 'end'
        for q in BOUNDARY_QUANTITIES:
            out[f'{side}_{q}_residual'] = float(tw[q, marker] - tw0[q, marker])
            out[f'{side}_{q}_nominal'] = float(tw0[q, marker])

    out['sext_bety'] = float(tw['bety', sext_corr])
    out['sext_bety_nominal'] = float(tw0['bety', sext_corr])
    out['sext_bety_ratio'] = out['sext_bety'] / out['sext_bety_nominal']

    # Worst vertical beta-beat anywhere in the straight.
    bare = {n: b for n, b in zip(tw0.rows[name_start:name_end].name,
                                 tw0.rows[name_start:name_end].bety)
            if b > 1e-9}
    beats = [(tw.bety[i] / bare[n], n, float(tw.bety[i]), float(bare[n]))
             for i, n in enumerate(tw.name) if n in bare]
    if beats:
        worst = max(beats)
        out['worst_bety_beat'] = float(worst[0])
        out['worst_bety_beat_at'] = worst[1]
        out['worst_bety_value'] = worst[2]
        out['worst_bety_nominal'] = worst[3]
        out['peak_bety'] = float(max(b[2] for b in beats))

    chrom_bare = context['chrom_bare']
    for tag, chrom in (('bare', chrom_bare), ('corr', chrom_corr)):
        for key in ('dqx', 'dqy', 'd2qx', 'd2qy'):
            out[f'{tag}_{key}'] = (float(chrom[key]) if chrom is not None
                                   else float('nan'))
    if chrom_bare is not None and chrom_corr is not None:
        out['delta_d2qx'] = chrom_corr['d2qx'] - chrom_bare['d2qx']
        out['delta_d2qy'] = chrom_corr['d2qy'] - chrom_bare['d2qy']
    else:
        out['delta_d2qx'] = out['delta_d2qy'] = float('nan')

    # Live tables, kept out of `out` proper so the summary stays JSON-clean.
    out['_tables'] = {
        'bare': tw0,
        'corrected': tw,
        'corrected_ring': tw_ring,
    }
    return out


def run_variant(variant_key, config, ip_name):
    variant = VARIANTS[variant_key]
    print()
    print('#' * 78)
    print(f'# VARIANT {variant_key}: {variant["label"]}')
    print(f'# IP {ip_name}, lattice {INPUT_LATTICE_JSON.name}')
    print('#' * 78)

    # Fresh load per variant: match_knob adds permanent auxiliary-variable
    # terms to the element expressions, so re-running on one line object would
    # stack the second variant's knobs on top of the first's.
    env = xt.load(INPUT_LATTICE_JSON)
    line = env.fccee_p_ring
    n_installed = install_mid_bend_quads(line, env, config)
    print(f'  installed {n_installed} mid-bend trim quadrupoles')

    for ip in IP_NAMES:
        line[f'on_sol_{ip}'] = 0
        line[f'on_comp_sol_{ip}'] = 0

    opt_orbit, opt_optics, opt_coupling, context = build_corrections(
        line, env, config, ip_name, variant)
    print(f'  optics match: {context["n_vary_optics"]} vary knobs, '
          f'{context["n_targets_optics"]} targets '
          f'(step={variant["vary_step"]:g}, '
          f'n_steps_max={variant["n_steps_max"] or "default"})')
    print(f'  coupling match: {context["n_vary_coupling"]} vary knobs')

    if args.dry_run:
        print('  --dry-run: stopping before solve')
        return None

    solve_sequence(opt_orbit, opt_optics, opt_coupling, args.quick)
    result = measure(line, opt_orbit, opt_optics, opt_coupling,
                     context, ip_name)
    result['variant'] = variant_key
    result['label'] = variant['label']
    result['n_vary_optics'] = context['n_vary_optics']
    result['n_targets_optics'] = context['n_targets_optics']

    # Inject plain module-level names rather than collecting into a dict, so
    # that `%run` from IPython hands you `tw_old` / `tw_new` directly. The
    # variant key is a fixed identifier from VARIANTS, not user text.
    tables = result.pop('_tables')
    g = globals()
    g.setdefault('tw_bare', None)
    if g['tw_bare'] is None:
        g['tw_bare'] = tables['bare']
    g[f'tw_{variant_key}'] = tables['corrected_ring']
    g[f'tw_{variant_key}_seg'] = tables['corrected']
    g[f'res_{variant_key}'] = result
    g['SEXT_CORR'] = context['sext_corr']
    g['NAME_START'] = context['name_start']
    g['NAME_END'] = context['name_end']

    names = ['tw_bare', f'tw_{variant_key}_seg', f'res_{variant_key}']
    if tables['corrected_ring'] is not None:
        names.insert(1, f'tw_{variant_key}')
    else:
        print(f'  tw_{variant_key} is None (no closed orbit for this state)')
    print(f"  in namespace: {', '.join(names)}")
    return result


def print_comparison(results):
    if not results:
        return
    keys = [r['variant'] for r in results]
    w = 22
    def row(name, fmt, get):
        cells = ''.join(f'{fmt.format(get(r)):>{w}}' for r in results)
        print(f'  {name:34s}{cells}')

    print()
    print('=' * (36 + w * len(results)))
    print(f'  {"":34s}' + ''.join(f'{k:>{w}}' for k in keys))
    print('=' * (36 + w * len(results)))
    row('optics: vary knobs', '{:d}', lambda r: r['n_vary_optics'])
    row('optics: targets', '{:d}', lambda r: r['n_targets_optics'])
    row('optics: targets within tol', '{:s}',
        lambda r: f"{r['optics_n_in_tol']}/{r['optics_n_targets']}")
    row('optics: final penalty', '{:.6g}', lambda r: r['optics_penalty'])
    print()
    row('bety at sextupole [m]', '{:.4f}', lambda r: r['sext_bety'])
    row('  ... / nominal', '{:.2f}x', lambda r: r['sext_bety_ratio'])
    row('worst bety beat in straight', '{:.2f}x',
        lambda r: r['worst_bety_beat'])
    row('peak bety in straight [m]', '{:.1f}', lambda r: r['peak_bety'])
    print()
    row('d2qx bare', '{:.4f}', lambda r: r['bare_d2qx'])
    row('d2qx corrected', '{:.4f}', lambda r: r['corr_d2qx'])
    row('  change', '{:+.4f}', lambda r: r['delta_d2qx'])
    row('d2qy bare', '{:.4f}', lambda r: r['bare_d2qy'])
    row('d2qy corrected', '{:.4f}', lambda r: r['corr_d2qy'])
    row('  change', '{:+.4f}', lambda r: r['delta_d2qy'])
    print()
    for side in ('start', 'end'):
        for q in BOUNDARY_QUANTITIES:
            row(f'{side} {q} residual', '{:.3e}',
                lambda r, s=side, qq=q: r[f'{s}_{qq}_residual'])
        print()
    print('=' * (36 + w * len(results)))
    print('  "residual" is matched minus nominal at the straight-section')
    print('  boundary markers -- the optics match targets these to 1e-5 (beta)')
    print('  and 1e-8 (alfa, dx, dpx).')
    print('  d2q{x,y} is Q\'\'/2 (detuning.py convention, as in 004c/004j);')
    print('  NaN means the state had no closed orbit for the delta sweep.')


def main():
    config = load_config_from_004c()
    if not INPUT_LATTICE_JSON.exists():
        raise SystemExit(
            f'{INPUT_LATTICE_JSON} not found -- run '
            f'004b_install_solenoids_in_fcc_ring.py --b0 {args.b0:g} first.')

    results = []
    for variant_key in args.variants:
        result = run_variant(variant_key, config, args.ip)
        if result is not None:
            results.append(result)

    print_comparison(results)


if __name__ == '__main__':
    main()
    # tw_bare / tw_<variant> / res_<variant> are module-level, so after `%run`
    # from IPython they are sitting in the interactive namespace.
