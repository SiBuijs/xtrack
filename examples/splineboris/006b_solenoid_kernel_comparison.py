"""
TubeFitter kernel comparison on the same solenoid field map used in
006a_solenoid.py -- fits with "tent", "quadratic", and "cubic" longitudinal
B-spline kernels and compares each against the BorisSpatialIntegrator
reference (the gold-standard full 3D field integration), alongside
SplineBoris (FieldFitter) and VariableSolenoid for context.

See examples/splineboris/claude_notes/ for background on TubeFitter and its
kernel choices (tube_approach_kernel_simplification.md in particular).
"""
import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import pandas as pd
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack._temp.splineboris.field_fitter import FieldFitter
from xtrack._temp.splineboris.splineboris_sequence import SplineBorisSequence
from xtrack._temp.splineboris.tube_fitter import TubeFitter
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

# Set basic parameters (identical to 006a_solenoid.py, so results are
# directly comparable).
interval = 30
dx = 0.001
dy = 0.001
multipole_order = 2
n_steps = 5000

KERNELS = ("tent", "quadratic", "cubic")

# Make initial particles
delta = np.array([0, 4])
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                energy0=45.6e6,  # 45.6 GeV (e.g. FCC-ee Z-pole)
                x=1e-3,  # Start slightly off-axis
                px=-1e-3*(1+delta),
                y=1e-3,
                delta=delta)

# Make solenoid field instance
sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)


def get_field(x, y, z):
    return sf.get_field(x, y, z)


z_point_count = n_steps + 1
x_axis = np.linspace(-multipole_order * dx / 2, multipole_order * dx / 2, multipole_order + 1)
y_axis = np.linspace(-multipole_order * dy / 2, multipole_order * dy / 2, multipole_order + 1)
z_axis = np.linspace(0, interval, z_point_count)
x_grid, y_grid, z_grid = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
bx, by, bz = sf.get_field(x_grid.ravel(), y_grid.ravel(), z_grid.ravel())

df_raw_data = pd.DataFrame(
    np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), bx, by, bz]),
    columns=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
).set_index(["X", "Y", "Z"])

# TubeFitter expects raw Bx/By column names rather than FieldFitter's Bskew/Bnorm.
df_raw_data_tube = pd.DataFrame(
    np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), bx, by, bz]),
    columns=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])

# --- SplineBoris (FieldFitter) reference, for context ---
fitter = FieldFitter(
    raw_data=df_raw_data,
    xy_point=(0, 0),
    distance_unit=1,
    min_region_size=10,
    deg=multipole_order - 1,
    field_tol=1e-4,
)
fitter.fit()
seq = SplineBorisSequence(
    df_fit_pars=fitter.df_fit_pars,
    multipole_order=multipole_order,
    steps_per_point=1,
)
line_splineboris = seq.to_line()
line_splineboris.build_tracker()

# --- TubeFitter, once per kernel ---
lines_tubefitter = {}
for kernel in KERNELS:
    tube_fitter = TubeFitter(
        raw_data=df_raw_data_tube,
        n_frames=500,
        distance_unit=1,
        deg=multipole_order - 1,
        kernel=kernel,
        field_tol=1e-4,
    )
    tube_fitter.fit()
    line = tube_fitter.to_line(multipole_order=multipole_order, steps_per_point=1)
    line.build_tracker()
    lines_tubefitter[kernel] = line

# --- TRUE REFERENCE: BorisSpatialIntegrator with the same analytical field ---
boris_integrator = xt.BorisSpatialIntegrator(
    fieldmap_callable=get_field,
    s_start=0,
    s_end=interval,
    n_steps=n_steps,
)

# --- VariableSolenoid reference (paraxial approximation, on-axis Bz only) ---
n_ref_steps = n_steps
z_axis_ref = np.linspace(0, interval, n_ref_steps)
Bz_axis = sf.get_field(0 * z_axis_ref, 0 * z_axis_ref, z_axis_ref)[2]
P0_J = p0.p0c[0] * qe / clight
brho = P0_J / qe / p0.q0
ks = Bz_axis / brho
ks_entry = ks[:-1]
ks_exit = ks[1:]
dz = z_axis_ref[1] - z_axis_ref[0]
line_varsol = xt.Line(elements=[
    xt.VariableSolenoid(length=dz, ks_profile=[ks_entry[ii], ks_exit[ii]])
    for ii in range(len(z_axis_ref) - 1)
])
line_varsol.build_tracker()

# --- Track everything ---
p_splineboris = p0.copy()
line_splineboris.track(p_splineboris, turn_by_turn_monitor='ONE_TURN_EBE')
mon_splineboris = line_splineboris.record_last_track

boris_integrator.log_trajectories = True
p_boris = p0.copy()
boris_integrator.track(p_boris)
x_boris = np.array(boris_integrator.x_log)
y_boris = np.array(boris_integrator.y_log)
z_boris = np.array(boris_integrator.z_log)

p_varsol = p0.copy()
line_varsol.track(p_varsol, turn_by_turn_monitor='ONE_TURN_EBE')
mon_varsol = line_varsol.record_last_track

mons_tubefitter = {}
for kernel, line in lines_tubefitter.items():
    p = p0.copy()
    line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
    mons_tubefitter[kernel] = line.record_last_track

n_part = mon_splineboris.x.shape[0]


# --- Quantitative comparison against the BorisSpatialIntegrator reference ---
def _rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


print("\n=== Trajectory deviation vs BorisSpatialIntegrator (reference) ===")
for i in range(n_part):
    s_ref, x_ref, y_ref = z_boris[:, i], x_boris[:, i], y_boris[:, i]
    print(f"--- particle {i} (delta={delta[i]}) ---")
    rows = [("SplineBoris (FieldFitter)", mon_splineboris)]
    rows += [(f"TubeFitter ({kernel})", mons_tubefitter[kernel]) for kernel in KERNELS]
    rows += [("VariableSolenoid", mon_varsol)]
    for label, mon in rows:
        x_i = np.interp(s_ref, mon.s[i, :], mon.x[i, :])
        y_i = np.interp(s_ref, mon.s[i, :], mon.y[i, :])
        print(
            f"  {label:<28s} x_rms={_rms(x_i, x_ref)*1e6:8.3f} um  "
            f"y_rms={_rms(y_i, y_ref)*1e6:8.3f} um  "
            f"x_end_err={abs(x_i[-1]-x_ref[-1])*1e6:8.3f} um  "
            f"y_end_err={abs(y_i[-1]-y_ref[-1])*1e6:8.3f} um"
        )

# --- Plot: x vs s and y vs s, one column per particle, one curve per kernel ---
kernel_colors = {"tent": "C0", "quadratic": "C1", "cubic": "C2"}
fig, axes = plt.subplots(2, n_part, figsize=(14, 10))

for i in range(n_part):
    for row, (comp, comp_boris) in enumerate((("x", x_boris), ("y", y_boris))):
        ax = axes[row, i]
        ax.plot(z_boris[:, i], comp_boris[:, i] * 1e3, 'k:', label='Boris (ref)', linewidth=2)
        ax.plot(mon_splineboris.s[i, :], getattr(mon_splineboris, comp)[i, :] * 1e3,
                 '--', color='gray', label='SplineBoris', alpha=0.7)
        for kernel in KERNELS:
            mon = mons_tubefitter[kernel]
            ax.plot(mon.s[i, :], getattr(mon, comp)[i, :] * 1e3,
                     '-', color=kernel_colors[kernel], label=f'TubeFitter ({kernel})',
                     linewidth=1.5, alpha=0.8)
        ax.set_xlabel('s [m]')
        ax.set_ylabel(f'{comp} [mm]')
        ax.set_title(f'Particle {i} (delta={delta[i]}): {comp} vs s')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

fig.suptitle('TubeFitter kernel comparison (tent / quadratic / cubic)')
fig.tight_layout()
plt.show()
