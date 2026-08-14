import numpy as np
import pandas as pd
import xtrack as xt
import matplotlib.pyplot as plt

from xtrack._temp.splineboris.tube_fitter import TubeFitter

MULTIPOLE_ORDER = 3

SHIFT_Y = 0.0  # [m]
N_SHIFT_X = 31
SHIFT_X_VALUES = np.linspace(-0.00002, 0.00001, N_SHIFT_X)  # [m]

# Wiggler locations instrumented -- just the two ars11 straights (same
# default as 004b_undulators_in_sls_ring.py).
WIGGLER_PLACES = [
    'ars11_uind_0210_1',
    'ars11_uind_0610_1',
]

##################################################
# Fit the undulator field map (TubeFitter), once #
##################################################

field_map_path = "../../test_data/sls/undulator_field_map.txt"
df_raw_data = pd.read_csv(
    field_map_path,
    sep=r"\s+",
    header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])

fitter = TubeFitter(
    raw_data=df_raw_data,
    distance_unit=0.001,  # dataset uses mm
    n_frames=1701,
    deg=MULTIPOLE_ORDER - 1,
)
fitter.fit()


def _build_undulator_lines(shift_x):
    # Build + correct the two standalone undulator models (SplineBoris and
    # Multipole) at a given horizontal offset, reusing the SAME (already
    # matched) correctors on both -- so the comparison isolates the
    # field-model difference instead of letting each model's own correction
    # reabsorb it (same rationale as 004a_build_undulator.py /
    # 004b_undulators_in_sls_ring.py).
    und_env = xt.Environment()
    und_env.set_particle_ref('positron', p0c=2.7e9)

    undulator_line = fitter.to_line(multipole_order=MULTIPOLE_ORDER, steps_per_point=1,
                                     shift_x=shift_x, shift_y=SHIFT_Y)
    undulator = und_env.import_line(undulator_line, line_name='undulator')

    for kk in ('k0l_corr1', 'k0l_corr2', 'k0l_corr3', 'k0l_corr4',
               'k0sl_corr1', 'k0sl_corr2', 'k0sl_corr3', 'k0sl_corr4'):
        und_env[kk] = 0.
    und_env.new('corr1', xt.Multipole, knl=['k0l_corr1'], ksl=['k0sl_corr1'])
    und_env.new('corr2', xt.Multipole, knl=['k0l_corr2'], ksl=['k0sl_corr2'])
    und_env.new('corr3', xt.Multipole, knl=['k0l_corr3'], ksl=['k0sl_corr3'])
    und_env.new('corr4', xt.Multipole, knl=['k0l_corr4'], ksl=['k0sl_corr4'])

    l_undulator = undulator.get_length()
    undulator.insert([
        und_env.place('corr1', at=0.02),
        und_env.place('corr2', at=0.1),
        und_env.place('corr3', at=l_undulator - 0.1),
        und_env.place('corr4', at=l_undulator - 0.02),
    ], s_tol=5e-3)

    opt = undulator.match(
        solve=False,
        betx=1, bety=1,
        include_collective=True,
        vary=xt.VaryList(['k0l_corr1', 'k0sl_corr1',
                          'k0l_corr2', 'k0sl_corr2',
                          'k0l_corr3', 'k0sl_corr3',
                          'k0l_corr4', 'k0sl_corr4',
                          ], step=1e-6),
        targets=[
            xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
            xt.TargetSet(x=0., y=0, at='corr2'),
            xt.TargetSet(x=0., y=0, at='corr3'),
            ],
    )
    opt.solve()

    multipole_undulator_line = fitter.to_multipole_line(
        multipole_order=MULTIPOLE_ORDER, p0c=2.7e9, field_at='midpoint',
        shift_x=shift_x, shift_y=SHIFT_Y)
    undulator_mult = und_env.import_line(multipole_undulator_line, line_name='undulator_mult')
    undulator_mult.insert([
        und_env.place('corr1', at=0.02),
        und_env.place('corr2', at=0.1),
        und_env.place('corr3', at=l_undulator - 0.1),
        und_env.place('corr4', at=l_undulator - 0.02),
    ], s_tol=5e-3)

    return {'splineboris': undulator, 'multipole': undulator_mult}


def _c_minus_for_shift(shift_x):
    undulator_lines = _build_undulator_lines(shift_x)

    c_minus = {}
    for kind, und_line in undulator_lines.items():
        madx_file = '../../test_data/sls/sls.madx'
        env = xt.load(madx_file)
        line_sls = env.lines['ring']
        line_sls.set_particle_ref('positron', p0c=2.7e9)
        tt = line_sls.get_table()

        insertions = []
        for i, wig_place in enumerate(WIGGLER_PLACES):
            # Fresh import per placement -- an element repeated at multiple
            # positions in the same line confuses downstream analyses (see
            # 012_sls_undulator_spin_emittance_evolution.py's _insert_undulators).
            import_name = f'undulator_{kind}_{i}'
            env.import_line(und_line, line_name=import_name)
            insertions.append(
                env.place(env[import_name], anchor='start', at=tt['s_start', wig_place]))
        line_sls.insert(insertions)

        tw_sls = line_sls.twiss4d()
        c_minus[kind] = tw_sls.c_minus

    return c_minus


##################################################
# Scan the horizontal offset and collect C^-     #
##################################################

c_minus_vs_shift = {'splineboris': [], 'multipole': []}

for shift_x in SHIFT_X_VALUES:
    print("=" * 80)
    print(f"shift_x = {shift_x:+.4e} m")
    print("=" * 80)
    c_minus = _c_minus_for_shift(shift_x)
    for kind in ('splineboris', 'multipole'):
        print(f"  {kind:12s} C^- = {c_minus[kind]:.6e}")
        c_minus_vs_shift[kind].append(c_minus[kind])

##################################################
# Plot: C^- vs horizontal offset, both models,   #
# plus their difference (Multipole - SplineBoris) #
##################################################

c_minus_diff = (np.array(c_minus_vs_shift['multipole'])
                 - np.array(c_minus_vs_shift['splineboris']))

fig, (ax_cmin, ax_diff) = plt.subplots(
    2, 1, figsize=(8, 9), sharex=True, constrained_layout=True)

ax_cmin.plot(SHIFT_X_VALUES * 1e3, c_minus_vs_shift['splineboris'], 'o-', label='SplineBoris')
ax_cmin.plot(SHIFT_X_VALUES * 1e3, c_minus_vs_shift['multipole'], 's-.', label='Multipole')
ax_cmin.set_ylabel(r'$C^-$')
ax_cmin.set_title(r'Coupling coefficient $C^-$ vs undulator horizontal offset')
ax_cmin.legend()
ax_cmin.grid()

ax_diff.plot(SHIFT_X_VALUES * 1e3, c_minus_diff, 'd-', color='tab:green')
ax_diff.axhline(0, color='0.5', lw=0.8)
ax_diff.set_xlabel('undulator horizontal offset [mm]')
ax_diff.set_ylabel(r'$C^-_{\mathrm{Multipole}} - C^-_{\mathrm{SplineBoris}}$')
ax_diff.set_title('Difference between the two field models')
ax_diff.grid()

fig.savefig('splineboris_sls_c_minus_vs_offset.png', dpi=200, bbox_inches='tight')

plt.show()
