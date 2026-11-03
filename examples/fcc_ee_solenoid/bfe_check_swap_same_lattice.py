"""Same corrected lattice, SB slices vs their BFE conversion (knobs frozen).

Separates the element-model difference from the correction-solution
difference in the 004n SB-vs-BFE comparison: the SB-corrected ring is
evaluated as is, then with every SplineBoris slice replaced by
bfe_setup.splineboris_to_bfe at the current knob values. Exploratory.

Usage: python bfe_check_swap_same_lattice.py LATTICE_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np
import xtrack as xt

from bfe_setup import DEFAULT_BFE_STEPS, _hermite_row
from lattice_knobs import set_lattice_knobs

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / 'nonlinear_tunes'))
from detuning import get_nonlinear_chromaticity  # noqa: E402

env = xt.load(HERE / sys.argv[1])
line = env.fccee_p_ring
set_lattice_knobs(line, with_solenoids=True, with_correctors=True)
rigidity0 = line.particle_ref.rigidity0[0]


def frozen_bfe(el):
    """BFE with the SB slice's current scale_b and corrector kicks baked in.
    Kicks are divided by the scale since BFE multiplies knl/ksl by kscale."""
    scale = 1.0 if el.scale_b is None else float(el.scale_b)
    knl = np.asarray(el.knl, dtype=float)
    ksl = np.asarray(el.ksl, dtype=float)
    if (np.any(knl) or np.any(ksl)) and scale == 0:
        raise ValueError('corrector kick on a switched-off slice')
    L = float(el.length)
    return xt.BFieldExpansion(
        length=L, kscale=scale, s_start=0.0, pkin_const=True,
        num_integration_steps=DEFAULT_BFE_STEPS,
        ksolc=_hermite_row(np.asarray(el.bs), L, rigidity0),
        ksc=np.array([_hermite_row(r, L, rigidity0)
                      for r in np.atleast_2d(np.asarray(el.bx))]),
        knc=np.array([_hermite_row(r, L, rigidity0)
                      for r in np.atleast_2d(np.asarray(el.by))]),
        knl=knl / (scale if scale else 1.0),
        ksl=ksl / (scale if scale else 1.0))


def figures(ln, label):
    tw = ln.twiss4d()
    chrom = get_nonlinear_chromaticity(ln, npoints=21, order=2)
    row = dict(case=label, qx=float(tw.qx), qy=float(tw.qy),
               c_minus=float(tw.c_minus),
               dqx=float(chrom.qx_derivatives[1]),
               dqy=float(chrom.qy_derivatives[1]),
               q2x=4 * float(chrom.qx_derivatives[2]),
               q2y=4 * float(chrom.qy_derivatives[2]))
    print(json.dumps(row), flush=True)
    return tw


# Evaluate the SB ring first, then replace each SplineBoris slice in the
# environment by its frozen BFE twin. No knob changes follow, so no expression
# fires on the replaced objects.
tw_a = figures(line, 'SB lattice, SB slices')
n_swapped = 0
for nn in list(line.element_names):
    el = line[nn]
    if isinstance(el, xt.SplineBoris):
        env.element_dict[nn] = frozen_bfe(el)
        n_swapped += 1
line.discard_tracker()
print(f'swapped {n_swapped} SplineBoris slices', flush=True)
tw_b = figures(line, 'SB lattice, BFE slices (frozen knobs)')
for k in ('betx', 'bety'):
    print(f'max rel d{k}: {np.max(np.abs(tw_b[k] / tw_a[k] - 1)):.2e}')
