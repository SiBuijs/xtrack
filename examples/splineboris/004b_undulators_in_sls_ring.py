import numpy as np
import pandas as pd
import xtrack as xt
import matplotlib.pyplot as plt

from xtrack._temp.splineboris.tube_fitter import TubeFitter

MULTIPOLE_ORDER = 3
ANOMALOUS_MAGNETIC_MOMENT = 1.15965218076e-3

# Rigid transverse misalignment applied to the undulator field elements
# (both models) -- change these to scan different misalignments (same
# mechanism as 004a_build_undulator.py's SHIFT_X/SHIFT_Y).
SHIFT_X = 0.0005  # [m]
SHIFT_Y = 0.0  # [m]

# Wiggler locations instrumented by default -- just the two ars11 straights.
# Full 11-location set (all wigglers in the ring) kept below for reference.
WIGGLER_PLACES = [
    'ars11_uind_0210_1',
    'ars11_uind_0610_1',
]
# WIGGLER_PLACES = [
#     'ars02_uind_0500_1', 'ars03_uind_0380_1', 'ars04_uind_0500_1',
#     'ars05_uind_0650_1', 'ars06_uind_0500_1', 'ars07_uind_0200_1',
#     'ars08_uind_0500_1', 'ars09_uind_0790_1', 'ars11_uind_0210_1',
#     'ars11_uind_0610_1', 'ars12_uind_0500_1',
# ]

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

#############################################################################
# Build + correct the two standalone undulator models (SplineBoris and     #
# Multipole), reusing the SAME (already matched) correctors on both -- so  #
# the comparison below isolates the field-model difference instead of     #
# letting each model's own correction reabsorb it (same rationale as       #
# 004a_build_undulator.py / 012_sls_undulator_spin_emittance_evolution.py) #
#############################################################################

und_env = xt.Environment()
und_env.set_particle_ref('positron', p0c=2.7e9)

undulator_line = fitter.to_line(multipole_order=MULTIPOLE_ORDER, steps_per_point=1,
                                 shift_x=SHIFT_X, shift_y=SHIFT_Y)
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
    shift_x=SHIFT_X, shift_y=SHIFT_Y)
undulator_mult = und_env.import_line(multipole_undulator_line, line_name='undulator_mult')
undulator_mult.insert([
    und_env.place('corr1', at=0.02),
    und_env.place('corr2', at=0.1),
    und_env.place('corr3', at=l_undulator - 0.1),
    und_env.place('corr4', at=l_undulator - 0.02),
], s_tol=5e-3)

undulator_lines = {'splineboris': undulator, 'multipole': undulator_mult}

##############################################################
# Insert into a fresh copy of the SLS ring and twiss, once   #
# per undulator model                                        #
##############################################################

def _build_and_twiss(kind):
    madx_file = '../../test_data/sls/sls.madx'
    env = xt.load(madx_file)
    line_sls = env.lines['ring']
    line_sls.set_particle_ref('positron', p0c=2.7e9)
    tt = line_sls.get_table()

    und_line = undulator_lines[kind]
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

    line_sls.particle_ref.anomalous_magnetic_moment = ANOMALOUS_MAGNETIC_MOMENT
    tw_sls = line_sls.twiss4d(radiation_integrals=True, spin=True, polarization_analysis=True)
    return line_sls, tw_sls


results = {}
for kind in ('splineboris', 'multipole'):
    print("=" * 80)
    print(f"SLS WITH UNDULATORS -- {kind}")
    print("=" * 80)
    line_sls, tw_sls = _build_and_twiss(kind)
    results[kind] = (line_sls, tw_sls)

    print(f"Tunes:")
    print(f"  qx = {tw_sls.qx:.4e}")
    print(f"  qy = {tw_sls.qy:.4e}")
    print(f"  qs = {tw_sls.qs:.4e}")
    print()
    print(f"Chromaticity:")
    print(f"  dqx = {tw_sls.dqx:.4e}")
    print(f"  dqy = {tw_sls.dqy:.4e}")
    print()
    print(f"Damping constants per second:")
    print(f"  alpha_x = {tw_sls.rad_int_damping_constant_x_s:.4e}")
    print(f"  alpha_y = {tw_sls.rad_int_damping_constant_y_s:.4e}")
    print(f"  alpha_zeta = {tw_sls.rad_int_damping_constant_zeta_s:.4e}")
    print()
    print(f"Equilibrium emittances:")
    print(f"  eq_gemitt_x = {tw_sls.rad_int_eq_gemitt_x:.4e}")
    print(f"  eq_gemitt_y = {tw_sls.rad_int_eq_gemitt_y:.4e}")
    print()
    print(f"C^-: {tw_sls.c_minus:.4e}")
    print()
    print(f"Spin polarization: {tw_sls.spin_polarization_eq}")
    print()

print("=" * 80)
print("C^- SUMMARY")
print("=" * 80)
for kind in ('splineboris', 'multipole'):
    print(f"  {kind:12s} C^- = {results[kind][1].c_minus:.6e}")

#############################################
# Plot: closed orbit, SplineBoris vs        #
# Multipole (same figure as before, but now #
# comparing both undulator models)          #
#############################################

tw_sb = results['splineboris'][1]
tw_mult = results['multipole'][1]

fig_co = plt.figure(1, figsize=(10, 6))
ax_co = fig_co.gca()
ax_co.plot(tw_sb.s, tw_sb.x, label='x (SplineBoris)')
ax_co.plot(tw_sb.s, tw_sb.y, label='y (SplineBoris)')
ax_co.plot(tw_mult.s, tw_mult.x, '-.', label='x (Multipole)')
ax_co.plot(tw_mult.s, tw_mult.y, '-.', label='y (Multipole)')
ax_co.set_xlabel('s [m]')
ax_co.set_ylabel('closed orbit [m]')
ax_co.legend(fontsize=8)
ax_co.grid()
fig_co.savefig('splineboris_sls_closed_orbit.png', dpi=200, bbox_inches='tight')

#####################################################################
# Plot: betx2 and bety1 (coupling beta functions) in separate       #
# panels of the same figure, SplineBoris vs Multipole in each panel #
#####################################################################

fig_beta, (ax_betx2, ax_bety1) = plt.subplots(
    2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

ax_betx2.plot(tw_sb.s, tw_sb.betx2, label='SplineBoris')
ax_betx2.plot(tw_mult.s, tw_mult.betx2, '-.', label='Multipole')
ax_betx2.set_ylabel(r'$\beta_{x,2}$ [m]')
ax_betx2.legend()
ax_betx2.grid()

ax_bety1.plot(tw_sb.s, tw_sb.bety1, label='SplineBoris')
ax_bety1.plot(tw_mult.s, tw_mult.bety1, '-.', label='Multipole')
ax_bety1.set_ylabel(r'$\beta_{y,1}$ [m]')
ax_bety1.set_xlabel('s [m]')
ax_bety1.legend()
ax_bety1.grid()

fig_beta.suptitle('Coupling beta functions: SplineBoris vs Multipole undulators')
fig_beta.savefig('splineboris_sls_beta_coupling.png', dpi=200, bbox_inches='tight')

plt.show()
