"""
Spin tracking with undulators and radiation.

This script loads the SLS MADX file, builds undulator using SplineBorisSequence,
computes twiss with spin tracking and radiation, and displays results.
"""

import xtrack as xt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xtrack._temp.splineboris.field_fitter import FieldFitter
from xtrack._temp.splineboris.splineboris_sequence import SplineBorisSequence


multipole_order = 5

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)

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
field_map_path = BASE_DIR.parent.parent / "test_data" / "sls" / "simona_field_map.txt"
df_raw_data = pd.read_csv(
    field_map_path,
    sep=r"\s+",
    header=None,
    names=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
).set_index(["X", "Y", "Z"])

# Distance unit in meters (the dataset uses mm, so 1 mm = 0.001 m)
distance_unit = 0.001

field_fitter = FieldFitter(
    raw_data=df_raw_data,
    xy_point=(0, 0),
    distance_unit=distance_unit,
    min_region_size=10,
    deg=multipole_order-1,
    field_tol=1e-4,
)

# Save fit parameters if needed
# field_fitter.save_fit_pars(
#     BASE_DIR
#     / "test_data" / "sls"
#     / "undulator_fit_pars.csv"
# )

# Build undulator using SplineBorisSequence - automatically creates one SplineBoris
# element per polynomial piece with n_steps based on the data point count
seq = SplineBorisSequence(
    df_fit_pars=field_fitter.df_fit_pars,
    multipole_order=multipole_order,
    steps_per_point=1,
)

# Get the Line of SplineBoris elements (pass env for insert support)
piecewise_undulator = seq.to_line(env=env)
l_wig = seq.length

piecewise_undulator.build_tracker()

piecewise_undulator.particle_ref = p0.copy()

# The issue: When you use betx=1, bety=1, twiss4d treats the line as OPEN (non-periodic).
# For an open line, the orbit is computed from initial conditions in particle_on_co.
# If particle_on_co has zero initial conditions (x=0, px=0, y=0, py=0), the orbit will
# be zero unless there are kicks from the wiggler.
#
# Solution: Use only_orbit=True to explicitly compute the orbit, which should properly
# propagate through the wiggler and show any kicks/deviations.

# Create env variables for corrector strengths (needed for matching)
env['k0l_corr1'] = 0.
env['k0l_corr2'] = 0.
env['k0l_corr3'] = 0.
env['k0l_corr4'] = 0.
env['k0sl_corr1'] = 0.
env['k0sl_corr2'] = 0.
env['k0sl_corr3'] = 0.
env['k0sl_corr4'] = 0.

# Create corrector elements with expressions referencing env variables
env.new('corr1', xt.Multipole, knl=['k0l_corr1'], ksl=['k0sl_corr1'])
env.new('corr2', xt.Multipole, knl=['k0l_corr2'], ksl=['k0sl_corr2'])
env.new('corr3', xt.Multipole, knl=['k0l_corr3'], ksl=['k0sl_corr3'])
env.new('corr4', xt.Multipole, knl=['k0l_corr4'], ksl=['k0sl_corr4'])

# Insert correctors at nearest element boundary (s_tol avoids slicing)
piecewise_undulator.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
], s_tol=5e-3)

# Match correctors such that the average orbit over the undulator is centered.
# Keep end-orbit closure to avoid a drifting trajectory.
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

tw_undulator_orbit = piecewise_undulator.twiss4d(
    betx=1, bety=1,
    only_orbit=True,
    include_collective=True,
)
print(f"Average orbit in undulator: <x> = {np.mean(tw_undulator_orbit.x):.3e} m, "
      f"<y> = {np.mean(tw_undulator_orbit.y):.3e} m")


piecewise_undulator.particle_ref.anomalous_magnetic_moment = 0.00115965218128

tw_undulator_corr_spin = piecewise_undulator.twiss4d(
    betx=1, bety=1, 
    include_collective=True, 
    spin=True,
    spin_x=0.5, spin_y=0.25, spin_z=0.25
    )
tw_undulator_corr_spin.plot('x y')
tw_undulator_corr_spin.plot('betx bety', 'dx dy')
tw_undulator_corr_spin.plot('spin_x')
tw_undulator_corr_spin.plot('spin_y')
tw_undulator_corr_spin.plot('spin_z')

# 3D orbit through the undulator: longitudinal coordinate s, horizontal x, vertical y
fig3d = plt.figure(figsize=(9, 6))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.plot(
    tw_undulator_corr_spin.s,
    1e6 * tw_undulator_corr_spin.x,
    1e6 * tw_undulator_corr_spin.y,
    lw=1.8,
    label='orbit (no radiation)',
)
ax3d.set_xlabel('s [m]')
ax3d.set_ylabel('x [um]')
ax3d.set_zlabel('y [um]')
ax3d.set_title('3D orbit through the undulator')
ax3d.legend()
ax3d.grid(True)

# Keep transverse scales equal (x/y), while leaving s-axis independent.
x_um = 1e6 * tw_undulator_corr_spin.x
y_um = 1e6 * tw_undulator_corr_spin.y
xy_half_span = 0.5 * max(np.ptp(x_um), np.ptp(y_um), 1e-12)
x_um_mid = 0.5 * (np.max(x_um) + np.min(x_um))
y_um_mid = 0.5 * (np.max(y_um) + np.min(y_um))
ax3d.set_ylim(x_um_mid - xy_half_span, x_um_mid + xy_half_span)
ax3d.set_zlim(y_um_mid - xy_half_span, y_um_mid + xy_half_span)

plt.show()

# Enable radiation on all elements (this sets radiation_flag on elements)
piecewise_undulator.configure_radiation(model='mean')

# Verify radiation is enabled on SplineBoris elements
tt = piecewise_undulator.get_table()
spline_boris_elements = tt.rows[tt.element_type == 'SplineBoris']
if len(spline_boris_elements) > 0:
    first_elem = piecewise_undulator[spline_boris_elements.name[0]]
    print(f"Radiation flag on SplineBoris element: {first_elem.radiation_flag}")

# Run twiss4d including radiation effects (average energy loss)
tw_undulator_corr_spin_rad = piecewise_undulator.twiss4d(
    betx=1, bety=1,
    include_collective=True,
    spin=True,
    spin_x=0.5, spin_y=0.25, spin_z=0.25,
    radiation_method='full'  # Use 'kick_as_co' for average energy loss, or 'full' for full computation
)

# Check if there's any energy loss (delta change)
delta_no_rad = tw_undulator_corr_spin.delta
delta_with_rad = tw_undulator_corr_spin_rad.delta
delta_diff = delta_with_rad - delta_no_rad
print(f"\nEnergy loss check:")
print(f"  Max |delta| without radiation: {np.max(np.abs(delta_no_rad)):.2e}")
print(f"  Max |delta| with radiation: {np.max(np.abs(delta_with_rad)):.2e}")
print(f"  Max |delta difference|: {np.max(np.abs(delta_diff)):.2e}")
if np.max(np.abs(delta_diff)) < 1e-10:
    print("  WARNING: No measurable energy loss detected! Radiation may not be working.")
else:
    print(f"  Energy loss detected: {np.max(np.abs(delta_diff)) * piecewise_undulator.particle_ref.energy0[0] / 1e6:.4f} MeV")

# Plot results to compare with/without radiation
fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

# Plot closed orbit x
axs[0].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.x, label='no rad')
axs[0].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.x, label='with rad')
axs[0].set_ylabel('x [m]')
axs[0].legend()
axs[0].grid(True)

# Plot energy loss (delta)
axs[1].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.delta, label='delta (no rad)', linestyle='-')
axs[1].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.delta, label='delta (with rad)', linestyle='--')
axs[1].set_ylabel('delta (relative energy deviation)')
axs[1].legend()
axs[1].grid(True)

# Plot spin components
axs[2].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.spin_x, label='spin_x (no rad)', linestyle='-')
axs[2].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.spin_y, label='spin_y (no rad)', linestyle='-')
axs[2].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.spin_z, label='spin_z (no rad)', linestyle='-')
axs[2].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.spin_x, label='spin_x (with rad)', linestyle='--')
axs[2].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.spin_y, label='spin_y (with rad)', linestyle='--')
axs[2].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.spin_z, label='spin_z (with rad)', linestyle='--')
axs[2].set_ylabel('spin')
axs[2].set_xlabel('s [m]')
axs[2].legend()
axs[2].grid(True)

# Optional: 3D comparison with and without radiation
fig3d_rad = plt.figure(figsize=(9, 6))
ax3d_rad = fig3d_rad.add_subplot(111, projection='3d')
ax3d_rad.plot(
    tw_undulator_corr_spin.s,
    1e6 * tw_undulator_corr_spin.x,
    1e6 * tw_undulator_corr_spin.y,
    lw=1.8,
    label='no radiation',
)
ax3d_rad.plot(
    tw_undulator_corr_spin_rad.s,
    1e6 * tw_undulator_corr_spin_rad.x,
    1e6 * tw_undulator_corr_spin_rad.y,
    lw=1.6,
    linestyle='--',
    label='with radiation',
)
ax3d_rad.set_xlabel('s [m]')
ax3d_rad.set_ylabel('x [um]')
ax3d_rad.set_zlabel('y [um]')
ax3d_rad.set_title('3D orbit comparison in undulator')
ax3d_rad.legend()
ax3d_rad.grid(True)
# Keep transverse scales equal (x/y) also in radiation comparison.
x_rad_um = 1e6 * np.r_[tw_undulator_corr_spin.x, tw_undulator_corr_spin_rad.x]
y_rad_um = 1e6 * np.r_[tw_undulator_corr_spin.y, tw_undulator_corr_spin_rad.y]
xy_half_span_rad = 0.5 * max(np.ptp(x_rad_um), np.ptp(y_rad_um), 1e-12)
x_rad_um_mid = 0.5 * (np.max(x_rad_um) + np.min(x_rad_um))
y_rad_um_mid = 0.5 * (np.max(y_rad_um) + np.min(y_rad_um))
ax3d_rad.set_ylim(x_rad_um_mid - xy_half_span_rad, x_rad_um_mid + xy_half_span_rad)
ax3d_rad.set_zlim(y_rad_um_mid - xy_half_span_rad, y_rad_um_mid + xy_half_span_rad)

plt.tight_layout()
plt.show()