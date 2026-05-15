"""
SLS undulator, same field definition as in ``004_undulator_open_track_spin_radiation``,
tracked first with the default Strang-split Boris kernel and then with the
symplectic HelicalBoris kernel (analytic constant-B map per substep). The two
kernels share the same field evaluator, particle bookkeeping, and matched
corrector values; only the per-step integrator differs.

Radiation is intentionally disabled so the comparison isolates the integrator
itself. After running twiss on both kernels, a table of absolute and relative
deviations is printed for the most relevant coordinates and spin components.
"""

import xtrack as xt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xtrack._temp.splineboris.field_fitter import FieldFitter
from xtrack._temp.splineboris.splineboris_sequence import SplineBorisSequence


multipole_order = 3

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)

madx_file = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'sls.madx'
env = xt.load(str(madx_file))
line_sls = env.ring

line_sls.configure_bend_model(core='mat-kick-mat')

line_sls.particle_ref = p0.copy()

BASE_DIR = Path(__file__).resolve().parent

field_map_path = BASE_DIR.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
df_raw_data = pd.read_csv(
    field_map_path,
    sep=r"\s+",
    header=None,
    names=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
).set_index(["X", "Y", "Z"])

distance_unit = 0.001

field_fitter = FieldFitter(
    raw_data=df_raw_data,
    xy_point=(0, 0),
    distance_unit=distance_unit,
    min_region_size=10,
    deg=multipole_order-1,
)

seq = SplineBorisSequence(
    df_fit_pars=field_fitter.df_fit_pars,
    multipole_order=multipole_order,
    steps_per_point=1,
)

piecewise_undulator = seq.to_line(env=env)
l_wig = seq.length

piecewise_undulator.build_tracker()

piecewise_undulator.particle_ref = p0.copy()

env['k0l_corr1'] = 0.
env['k0l_corr2'] = 0.
env['k0l_corr3'] = 0.
env['k0l_corr4'] = 0.
env['k0sl_corr1'] = 0.
env['k0sl_corr2'] = 0.
env['k0sl_corr3'] = 0.
env['k0sl_corr4'] = 0.

env.new('corr1', xt.Multipole, knl=['k0l_corr1'], ksl=['k0sl_corr1'])
env.new('corr2', xt.Multipole, knl=['k0l_corr2'], ksl=['k0sl_corr2'])
env.new('corr3', xt.Multipole, knl=['k0l_corr3'], ksl=['k0sl_corr3'])
env.new('corr4', xt.Multipole, knl=['k0l_corr4'], ksl=['k0sl_corr4'])

piecewise_undulator.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
], s_tol=5e-3)

# Match correctors on the default Boris kernel; subsequent helical twiss reuses
# these corrector strengths so the deviations between the two kernels are due to
# the integrator alone, not to different orbits.
opt = piecewise_undulator.match(
    solve=False,
    betx=0, bety=0,
    only_orbit=True,
    include_collective=True,
    vary=xt.VaryList(['k0l_corr1', 'k0sl_corr1',
                      'k0l_corr2', 'k0sl_corr2',
                      'k0l_corr3', 'k0sl_corr3',
                      'k0l_corr4', 'k0sl_corr4',
                      ], step=1e-6),
    targets=[
        xt.Target(lambda tw: np.mean(tw.x), value=0.0, tol=1e-8, tag='avg_orbit'),
        xt.Target(lambda tw: np.mean(tw.y), value=0.0, tol=1e-8, tag='avg_orbit'),
        xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
        ],
)
opt.step(2)


piecewise_undulator.particle_ref.anomalous_magnetic_moment = 0.00115965218128

# ---------------------------------------------------------------------------
# Twiss with the default (Strang-split) Boris kernel
# ---------------------------------------------------------------------------
for _name in piecewise_undulator.element_names:
    _el = piecewise_undulator[_name]
    if isinstance(_el, xt.SplineBoris):
        _el.use_helical_boris = False

tw_boris = piecewise_undulator.twiss4d(
    betx=1, bety=1,
    include_collective=True,
    spin=True,
    spin_x=0.5, spin_y=0.25, spin_z=0.25,
)

# ---------------------------------------------------------------------------
# Twiss with the symplectic HelicalBoris kernel (same matched correctors)
# ---------------------------------------------------------------------------
for _name in piecewise_undulator.element_names:
    _el = piecewise_undulator[_name]
    if isinstance(_el, xt.SplineBoris):
        _el.use_helical_boris = True

tw_helic = piecewise_undulator.twiss4d(
    betx=1, bety=1,
    include_collective=True,
    spin=True,
    spin_x=0.5, spin_y=0.25, spin_z=0.25,
)

# ---------------------------------------------------------------------------
# Comparison table: max/mean absolute and relative deviations
# ---------------------------------------------------------------------------
columns = ['x', 'px', 'y', 'py', 'delta', 'spin_x', 'spin_y', 'spin_z']
rows = []
for col in columns:
    b = np.asarray(getattr(tw_boris, col), dtype=float)
    h = np.asarray(getattr(tw_helic, col), dtype=float)
    diff = h - b
    scale = np.max(np.abs(b)) if np.max(np.abs(b)) > 0 else 1.0
    rows.append({
        'column': col,
        'max|boris|': np.max(np.abs(b)),
        'max|h-b|':   np.max(np.abs(diff)),
        'mean|h-b|':  np.mean(np.abs(diff)),
        'max rel.':   np.max(np.abs(diff)) / scale,
    })

df_compare = pd.DataFrame(rows).set_index('column')
fmt = {
    'max|boris|': '{:.3e}'.format,
    'max|h-b|':   '{:.3e}'.format,
    'mean|h-b|':  '{:.3e}'.format,
    'max rel.':   '{:.3e}'.format,
}
print('\nDeviations: HelicalBoris vs SplineBoris (same correctors, same matched orbit)')
print(df_compare.to_string(formatters=fmt))

# ---------------------------------------------------------------------------
# Plots overlaying both kernels
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(4, 1, figsize=(9, 12), sharex=True)

axs[0].plot(tw_boris.s, tw_boris.x, label='SplineBoris', lw=2)
axs[0].plot(tw_helic.s, tw_helic.x, '--', label='HelicalBoris', lw=1.8)
axs[0].set_ylabel('x [m]')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

axs[1].plot(tw_boris.s, tw_boris.y, label='SplineBoris', lw=2)
axs[1].plot(tw_helic.s, tw_helic.y, '--', label='HelicalBoris', lw=1.8)
axs[1].set_ylabel('y [m]')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

axs[2].plot(tw_boris.s, tw_boris.x - tw_helic.x, label='dx (helical - boris)')
axs[2].plot(tw_boris.s, tw_boris.y - tw_helic.y, label='dy (helical - boris)')
axs[2].set_ylabel('coordinate diff [m]')
axs[2].legend()
axs[2].grid(True, alpha=0.3)

axs[3].plot(tw_boris.s, tw_boris.spin_x, label='spin_x (Boris)')
axs[3].plot(tw_helic.s, tw_helic.spin_x, '--', label='spin_x (Helical)')
axs[3].plot(tw_boris.s, tw_boris.spin_y, label='spin_y (Boris)')
axs[3].plot(tw_helic.s, tw_helic.spin_y, '--', label='spin_y (Helical)')
axs[3].plot(tw_boris.s, tw_boris.spin_z, label='spin_z (Boris)')
axs[3].plot(tw_helic.s, tw_helic.spin_z, '--', label='spin_z (Helical)')
axs[3].set_ylabel('spin')
axs[3].set_xlabel('s [m]')
axs[3].legend(ncol=3)
axs[3].grid(True, alpha=0.3)

fig.suptitle('SLS undulator: SplineBoris (solid) vs HelicalBoris (dashed)')
plt.tight_layout()
plt.show()
