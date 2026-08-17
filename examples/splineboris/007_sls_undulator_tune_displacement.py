import xtrack as xt
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from xtrack._temp.splineboris.tube_fitter import TubeFitter
from xtrack._temp.splineboris.splineboris_sequence import SplineBorisSequence


multipole_order = 5

E0 = 2.7e9

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)

# Load SLS MADX file
madx_file = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'sls.madx'
env = xt.load(str(madx_file))
line_sls = env.ring

# Configure bend model
line_sls.configure_bend_model(core='mat-kick-mat')

# Set particle reference
line_sls.particle_ref = p0.copy()

BASE_DIR = Path(__file__).resolve().parent

# Load the raw field map data from shared test_data
file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "simona_field_map.txt"
df_raw_data = pd.read_csv(
    file_path, sep="\t", header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
    dtype=float,
).set_index(["X", "Y", "Z"])

# Distance unit in meters (the dataset uses mm, so 1 mm = 0.001 m)
distance_unit = 0.001

n_frames = 4441

tube_fitter = TubeFitter(
    raw_data=df_raw_data,
    n_frames=n_frames,
    #residual_tol=1e-3,
    distance_unit=distance_unit,
    deg=multipole_order-1,
    field_tol=1e-4,
    #tube_radius=0.0005,
)

tube_fitter.fit()

# for der in range(0, multipole_order):
#     tube_fitter.plot_fields(der=der)

# plt.show()

# Build undulator using TubeFitter.to_line() -- creates one SplineBoris
# element per polynomial piece with n_steps based on the data point count.
# It lives in its own implicit Environment, so import it into its own
# `und_env` to add correctors and match it (same build path as
# 004b_undulators_in_sls_ring.py / 004a_build_undulator.py) before placing
# it into the SLS ring.
und_env = xt.Environment()
und_env.particle_ref = p0.copy()

undulator_line = tube_fitter.to_line(multipole_order=multipole_order)
undulator = und_env.import_line(undulator_line, line_name='undulator')

l_wig = undulator.get_length()

# Create env variables for corrector strengths (needed for matching)
und_env['k0l_corr1'] = 0.
und_env['k0l_corr2'] = 0.
und_env['k0l_corr3'] = 0.
und_env['k0l_corr4'] = 0.
und_env['k0sl_corr1'] = 0.
und_env['k0sl_corr2'] = 0.
und_env['k0sl_corr3'] = 0.
und_env['k0sl_corr4'] = 0.

# Create corrector elements with expressions referencing env variables
und_env.new('corr1', xt.Multipole, knl=['k0l_corr1'], ksl=['k0sl_corr1'])
und_env.new('corr2', xt.Multipole, knl=['k0l_corr2'], ksl=['k0sl_corr2'])
und_env.new('corr3', xt.Multipole, knl=['k0l_corr3'], ksl=['k0sl_corr3'])
und_env.new('corr4', xt.Multipole, knl=['k0l_corr4'], ksl=['k0sl_corr4'])

# Insert correctors at nearest element boundary (s_tol avoids slicing)
undulator.insert([
    und_env.place('corr1', at=0.02),
    und_env.place('corr2', at=0.1),
    und_env.place('corr3', at=l_wig - 0.1),
    und_env.place('corr4', at=l_wig - 0.02),
], s_tol=5e-3)

opt = undulator.match(
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
        xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.START),
        xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
        ],
)
opt.step(2)

# Uncomment which undulator you want to insert
wiggler_places = [
    #'ars11_uind_0210_1',
    'ars11_uind_0610_1',
]

# Twiss with no undulator inserted -- reused both as the tune baseline
# below and as the "no undulator" closed-orbit reference in the comparison
# plot further down.
tw_no_undulator = line_sls.twiss4d(include_collective=True)
qx_0 = tw_no_undulator.qx
qy_0 = tw_no_undulator.qy

tt = line_sls.get_table()
insertions = []
for i, wig_place in enumerate(wiggler_places):
    # Fresh import per placement -- an element repeated at multiple
    # positions in the same line confuses downstream analyses (same
    # rationale as 004b_undulators_in_sls_ring.py's _build_and_twiss).
    print(f"Inserting undulator at {wig_place} ({tt['s', wig_place]})")
    import_name = f'undulator_{i}'
    env.import_line(undulator, line_name=import_name)
    insertions.append(
        env.place(env[import_name], anchor='start', at=tt['s', wig_place]))
line_sls.insert(insertions)

spline_names = [
    nn for nn in line_sls.element_names
    if isinstance(line_sls[nn], xt.SplineBoris)
    ]

# s ranges (start, end) of the active undulators, for marking their
# location on the plots below. `tt` still reflects the pre-insertion
# element table, so its s values are the undulator start positions.
undulator_s_ranges = [
    (tt['s', wig_place], tt['s', wig_place] + l_wig)
    for wig_place in wiggler_places
    ]


def mark_undulator_bounds(ax):
    for s_start, s_end in undulator_s_ranges:
        ax.axvline(s_start, color='0.4', linestyle='--', linewidth=1)
        ax.axvline(s_end, color='0.4', linestyle='--', linewidth=1)

# Closed orbit comparison: no undulator vs. on-axis (shift_x=0) vs.
# off-axis (shift_x=0.5 mm) undulator.
for nn in spline_names:
    line_sls[nn].shift_x = 0.
tw_onaxis = line_sls.twiss4d(include_collective=True)

for nn in spline_names:
    line_sls[nn].shift_x = 0.5e-3
tw_offaxis = line_sls.twiss4d(include_collective=True)

fig_orbit, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for tw, label in [(tw_no_undulator, 'No undulator'),
                   (tw_onaxis, 'On-axis undulator'),
                   (tw_offaxis, 'Off-axis undulator (shift_x=0.5 mm)')]:
    ax_x.plot(tw.s, tw.x, label=label)
    ax_y.plot(tw.s, tw.y, label=label)
mark_undulator_bounds(ax_x)
mark_undulator_bounds(ax_y)
ax_x.set_ylabel('x [m]')
ax_x.set_title('Horizontal closed orbit around the SLS ring')
ax_x.grid(True, alpha=0.3)
ax_x.legend()
ax_y.set_xlabel('s [m]')
ax_y.set_ylabel('y [m]')
ax_y.set_title('Vertical closed orbit around the SLS ring')
ax_y.grid(True, alpha=0.3)
ax_y.legend()
fig_orbit.tight_layout()

deltaqx_list = []
deltaqy_list = []
orbit_scan_x = []
orbit_scan_y = []
orbit_scan_s = None
betx_scan = []
bety_scan = []

n_tunes = 30

hor_off_list = np.linspace(-0.5e-3, 0.5e-3, n_tunes)

# Correctors were only matched once, at shift_x=0 -- as the offset is
# scanned, the field (and hence the kick) seen by the SplineBoris elements
# changes, so those fixed corrector strengths stop restoring x=px=y=py=0 at
# the undulator exit. Left uncorrected, that residual orbit distortion leaks
# into the rest of the ring and contaminates the tune shift with an
# orbit-distortion effect on top of the field nonlinearity we're trying to
# isolate. Re-match the correctors (on the standalone `undulator` line, same
# targets as the initial match above) for every offset instead.
undulator_spline_names = [
    nn for nn in undulator.element_names
    if isinstance(undulator[nn], xt.SplineBoris)
    ]
corrector_vars = ['k0l_corr1', 'k0sl_corr1', 'k0l_corr2', 'k0sl_corr2',
                   'k0l_corr3', 'k0sl_corr3', 'k0l_corr4', 'k0sl_corr4']

for dx in hor_off_list:   # dx in meters
    # apply horizontal offset to all undulator slices (standalone line and
    # the copy inserted in the ring)
    for nn in undulator_spline_names:
        undulator[nn].shift_x = dx
    for nn in spline_names:
        line_sls[nn].shift_x = dx

    # re-correct the orbit for this offset
    opt_dx = undulator.match(
        solve=False,
        betx=0, bety=0,
        only_orbit=True,
        include_collective=True,
        vary=xt.VaryList(corrector_vars, step=1e-6),
        targets=[
            xt.TargetSet(x=0, y=0, at=xt.START),
            xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
            ],
    )
    opt_dx.step(2)

    # push the re-matched corrector strengths to the copy inserted in the ring
    for kk in corrector_vars:
        env[kk] = und_env[kk]

    # then compute tune for this offset
    tw = line_sls.twiss4d(include_collective=True)
    deltaqx_list.append(tw.qx - qx_0)
    deltaqy_list.append(tw.qy - qy_0)
    orbit_scan_x.append(tw.x)
    orbit_scan_y.append(tw.y)
    betx_scan.append(tw.betx)
    bety_scan.append(tw.bety)
    if orbit_scan_s is None:
        orbit_scan_s = tw.s

# Closed orbit at each offset of the tune scan above -- same panel layout as
# the no-undulator/on-axis/off-axis comparison plot, but colored by offset
# (a per-curve legend would be unreadable with n_tunes=30 curves).
norm = plt.Normalize(vmin=hor_off_list.min(), vmax=hor_off_list.max())
cmap = plt.cm.viridis

fig_orbit_scan, (ax_x_scan, ax_y_scan) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for dx, x_orbit, y_orbit in zip(hor_off_list, orbit_scan_x, orbit_scan_y):
    color = cmap(norm(dx))
    ax_x_scan.plot(orbit_scan_s, x_orbit, color=color)
    ax_y_scan.plot(orbit_scan_s, y_orbit, color=color)
mark_undulator_bounds(ax_x_scan)
mark_undulator_bounds(ax_y_scan)
ax_x_scan.set_ylabel('x [m]')
ax_x_scan.set_title('Horizontal closed orbit across the tune scan')
ax_x_scan.grid(True, alpha=0.3)
ax_y_scan.set_xlabel('s [m]')
ax_y_scan.set_ylabel('y [m]')
ax_y_scan.set_title('Vertical closed orbit across the tune scan')
ax_y_scan.grid(True, alpha=0.3)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig_orbit_scan.colorbar(sm, ax=[ax_x_scan, ax_y_scan], label='Horizontal offset [m]')

# Beta functions at each offset of the tune scan -- same colour-coded-by-
# offset layout as the orbit scan above. tw_no_undulator carries a different
# element grid (no undulator slices), so interpolate it onto the scan's
# (shared, undulator-including) s grid to overlay it as a reference.
betx_no_und_i = np.interp(orbit_scan_s, tw_no_undulator.s, tw_no_undulator.betx)
bety_no_und_i = np.interp(orbit_scan_s, tw_no_undulator.s, tw_no_undulator.bety)

fig_beta_scan, (ax_betx_scan, ax_bety_scan) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for dx, betx, bety in zip(hor_off_list, betx_scan, bety_scan):
    color = cmap(norm(dx))
    ax_betx_scan.plot(orbit_scan_s, betx, color=color)
    ax_bety_scan.plot(orbit_scan_s, bety, color=color)
ax_betx_scan.plot(orbit_scan_s, betx_no_und_i, color='k', linestyle='--',
                   linewidth=1, label='No undulator')
ax_bety_scan.plot(orbit_scan_s, bety_no_und_i, color='k', linestyle='--',
                   linewidth=1, label='No undulator')
mark_undulator_bounds(ax_betx_scan)
mark_undulator_bounds(ax_bety_scan)
ax_betx_scan.set_ylabel(r'$\beta_x$ [m]')
ax_betx_scan.set_title('Beta functions across the tune scan')
ax_betx_scan.grid(True, alpha=0.3)
ax_betx_scan.legend()
ax_bety_scan.set_xlabel('s [m]')
ax_bety_scan.set_ylabel(r'$\beta_y$ [m]')
ax_bety_scan.grid(True, alpha=0.3)
ax_bety_scan.legend()

sm_beta_abs = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm_beta_abs.set_array([])
fig_beta_scan.colorbar(sm_beta_abs, ax=[ax_betx_scan, ax_bety_scan], label='Horizontal offset [m]')

# Beta beat at each offset of the tune scan, relative to the no-undulator
# baseline -- same colour-coded-by-offset layout as the orbit scan above.
fig_beta_diff, (ax_dbetx, ax_dbety) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for dx, betx, bety in zip(hor_off_list, betx_scan, bety_scan):
    color = cmap(norm(dx))
    ax_dbetx.plot(orbit_scan_s, betx - betx_no_und_i, color=color)
    ax_dbety.plot(orbit_scan_s, bety - bety_no_und_i, color=color)
mark_undulator_bounds(ax_dbetx)
mark_undulator_bounds(ax_dbety)
ax_dbetx.set_ylabel(r'$\Delta\beta_x$ [m]')
ax_dbetx.set_title('Beta beat across the tune scan (relative to no undulator)')
ax_dbetx.grid(True, alpha=0.3)
ax_dbety.set_xlabel('s [m]')
ax_dbety.set_ylabel(r'$\Delta\beta_y$ [m]')
ax_dbety.grid(True, alpha=0.3)

sm_beta = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm_beta.set_array([])
fig_beta_diff.colorbar(sm_beta, ax=[ax_dbetx, ax_dbety], label='Horizontal offset [m]')

coef_qx = np.polyfit(hor_off_list, deltaqx_list, 2)
coef_qy = np.polyfit(hor_off_list, deltaqy_list, 2)

poly_qx = np.poly1d(coef_qx)
poly_qy = np.poly1d(coef_qy)

print(f"d²B_x/dx² = {coef_qx[0]}")
print(f"dB_x/dx   = {coef_qx[1]}")
print(f"B_x(0)    = {coef_qx[2]}")
print(f"d²B_y/dx² = {coef_qy[0]}")
print(f"dB_y/dx   = {coef_qy[1]}")
print(f"B_y(0)    = {coef_qy[2]}")

text_box_kwargs = dict(va='top', ha='left', fontsize=8, linespacing=1.4,
                        bbox=dict(boxstyle='round', fc='white', alpha=0.85, edgecolor='0.7'))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
ax1.plot(hor_off_list, deltaqx_list, marker='o', color='tab:blue')
ax1.plot(hor_off_list, poly_qx(hor_off_list), linestyle='--', color='k', label='Quadratic fit')
ax1.set_ylabel('Delta Qx')
ax1.set_title('Tune shift vs undulator horizontal offset')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.text(0.02, 0.95,
          f'$d^2B_x/dx^2$ = {coef_qx[0]:.4e}\n'
          f'$dB_x/dx$   = {coef_qx[1]:.4e}\n'
          f'$B_x(0)$    = {coef_qx[2]:.4e}',
          transform=ax1.transAxes, **text_box_kwargs)

ax2.plot(hor_off_list, deltaqy_list, marker='s', color='tab:orange')
ax2.plot(hor_off_list, poly_qy(hor_off_list), linestyle='--', color='k', label='Quadratic fit')
ax2.set_xlabel('Horizontal offset [m]')
ax2.set_ylabel('Delta Qy')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.text(0.02, 0.95,
          f'$d^2B_y/dx^2$ = {coef_qy[0]:.4e}\n'
          f'$dB_y/dx$   = {coef_qy[1]:.4e}\n'
          f'$B_y(0)$    = {coef_qy[2]:.4e}',
          transform=ax2.transAxes, **text_box_kwargs)

plt.tight_layout()
plt.show()

