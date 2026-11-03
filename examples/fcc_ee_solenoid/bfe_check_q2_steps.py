"""Q'' of a corrected lattice vs integrator steps per solenoid slice.

Tests whether the SB-vs-BFE Q'' difference in 004n is SplineBoris' Boris
step error: recomputes Q', Q'' (004n's 21-point fit, true d2Q/ddelta2) with
the solenoid slices at several step counts. Exploratory.

Usage: python bfe_check_q2_steps.py LATTICE_JSON STEPS [STEPS ...]
"""
import json
import sys
from pathlib import Path

import xtrack as xt

from lattice_knobs import set_lattice_knobs

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / 'nonlinear_tunes'))
from detuning import get_nonlinear_chromaticity  # noqa: E402

lattice = sys.argv[1]
steps_list = [int(v) for v in sys.argv[2:]]
env = xt.load(HERE / lattice)
line = env.fccee_p_ring
set_lattice_knobs(line, with_solenoids=True, with_correctors=True)
out = []
for steps in steps_list:
    for nn in line.element_names:
        el = line[nn]
        if isinstance(el, xt.SplineBoris):
            el.n_steps = steps
        elif isinstance(el, xt.BFieldExpansion):
            el.num_integration_steps = steps
    tw = line.twiss4d()
    chrom = get_nonlinear_chromaticity(line, npoints=21, order=2)
    row = dict(lattice=lattice, steps=steps, qx=float(tw.qx), qy=float(tw.qy),
               dqx=float(chrom.qx_derivatives[1]),
               dqy=float(chrom.qy_derivatives[1]),
               q2x=4 * float(chrom.qx_derivatives[2]),
               q2y=4 * float(chrom.qy_derivatives[2]))
    print(json.dumps(row), flush=True)
