"""Check the SplineBoris -> BFieldExpansion conversion on the 004a templates.

1. Field agreement, slice by slice, on and off axis.
2. Tracking one solenoid (main and compensation): BFE exit coordinates vs SB,
   and the step-count convergence of each against its own fine-step result.
3. Symplecticity error of the solenoid map and tracking time per pass.

Usage: python bfe_check_conversion.py --b0 2.0 [--max-transverse-order N]
"""
import argparse
import time

import numpy as np
import xtrack as xt

from bfe_setup import sb_line_to_bfe_line, bfe_field_tesla
from solenoid_params import field_tag, order_tag, add_max_order_argument

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--b0", type=float, default=2.0, choices=[2.0, 3.0])
add_max_order_argument(parser)
args = parser.parse_args()

template_json = (f'004_solenoid_lines_{field_tag(args.b0)}'
                 f'{order_tag(args.max_transverse_order)}.json')
print(f'Templates: {template_json}')
templates = xt.json.load(template_json)['lines']

COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')
SB_FINE, BFE_FINE = 200, 80
STEP_SCAN = [1, 2, 3, 5, 10, 20]


def exit_coords(line, particles):
    p = particles.copy()
    line.track(p)
    assert np.all(p.state > 0), 'particle lost'
    return np.array([getattr(p, c) for c in COORDS])


def set_steps(line, n):
    for el in line.elements:
        if isinstance(el, xt.SplineBoris):
            el.n_steps = n
        elif isinstance(el, xt.BFieldExpansion):
            el.num_integration_steps = n


for name in ('main_solenoid', 'compensation_solenoid'):
    print(f'\n=== {name} ===')
    sb = xt.Line.from_dict(templates[name])
    rigidity0 = sb.particle_ref.rigidity0[0]
    bfe = sb_line_to_bfe_line(sb, rigidity0=rigidity0)
    sb.build_tracker()
    bfe.build_tracker()

    # 1. Field agreement
    worst = {c: 0.0 for c in ('Bx', 'By', 'Bs')}
    bs_max = 0.0
    for el_sb, el_bfe in zip(sb.elements, bfe.elements):
        if not isinstance(el_sb, xt.SplineBoris):
            continue
        s_loc = np.linspace(0, el_sb.length, 7)
        for x, y in [(0, 0), (2e-3, 0), (0, 2e-3), (3e-3, -2e-3), (1e-2, 1e-2)]:
            xs = np.full_like(s_loc, x)
            ys = np.full_like(s_loc, y)
            b_sb = np.asarray(el_sb.get_field(xs, ys, s_loc))
            b_bfe = bfe_field_tesla(el_bfe, xs, ys, s_loc, rigidity0=rigidity0)
            for c, v_sb, v_bfe in zip(('Bx', 'By', 'Bs'), b_sb, b_bfe):
                worst[c] = max(worst[c], np.max(np.abs(v_sb - v_bfe)))
            bs_max = max(bs_max, np.max(np.abs(b_sb[2])))
    print(f'max |B_SB - B_BFE| over slices, |x|,|y| <= 1 cm  '
          f'(max |Bs| = {bs_max:.3f} T):')
    for c, v in worst.items():
        print(f'  {c}: {v:.3e} T')

    # 2. Tracking
    particles = sb.build_particles(
        x=[0, 1e-4, 0, 5e-4, -3e-4, 1e-3],
        px=[0, 0, 1e-5, -2e-5, 3e-5, 0],
        y=[0, 0, 1e-4, 2e-4, -5e-4, 1e-3],
        py=[0, 2e-5, 0, 1e-5, -1e-5, 0],
        delta=[0, 0, 0, 1e-3, -1e-2, 1e-2])

    set_steps(sb, SB_FINE)
    ref_sb = exit_coords(sb, particles)
    set_steps(bfe, BFE_FINE)
    ref_bfe = exit_coords(bfe, particles)
    diff = np.max(np.abs(ref_sb - ref_bfe), axis=1)
    print(f'converged exit difference SB({SB_FINE}) vs BFE({BFE_FINE}) steps/slice:')
    print('  ' + '  '.join(f'{c}={d:.2e}' for c, d in zip(COORDS, diff)))

    print('step convergence, max |exit - own fine result| (x, px, y, py):')
    print(f'  {"steps":>5}  {"SB":>10}  {"BFE":>10}  {"t_SB [ms]":>10}  {"t_BFE [ms]":>10}')
    for n in STEP_SCAN:
        out = []
        for line, ref in ((sb, ref_sb), (bfe, ref_bfe)):
            set_steps(line, n)
            exit_coords(line, particles)  # warm-up
            t0 = time.perf_counter()
            c = exit_coords(line, particles)
            out.append((np.max(np.abs(c - ref)[:4]),
                        1e3 * (time.perf_counter() - t0)))
        print(f'  {n:>5}  {out[0][0]:>10.2e}  {out[1][0]:>10.2e}  '
              f'{out[0][1]:>10.2f}  {out[1][1]:>10.2f}')

    # 3. Symplecticity at the template step counts
    set_steps(sb, 10)
    set_steps(bfe, 10)
    S = xt.linear_normal_form.S
    for label, line in (('SB', sb), ('BFE', bfe)):
        R = line.get_R_matrix(particle_on_co=line.particle_ref.copy())['R_matrix']
        err = np.linalg.norm(R.T @ S @ R - S, ord=2)
        print(f'symplecticity ||R^T S R - S|| ({label}, 10 steps): {err:.3e}')
