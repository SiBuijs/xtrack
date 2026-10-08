import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import pandas as pd
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack._temp.splineboris import TubeFitter, LongitudinalFitter
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})
# Set basic parameters
interval = 30
dx = 0.001
dy = 0.001
multipole_order = 2
n_steps = 5000

# Make initial particles
delta = np.array([0, 4])
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                energy0=45.6e6,  # 45.6 GeV (e.g. FCC-ee Z-pole)
                x=1e-3,  # Start slightly off-axis
                px=-1e-3*(1+delta),
                y=1e-3,
                delta=delta)
p0.spin_x = 1.0
p0.spin_y = 0.0
p0.spin_z = 0.0
p0.anomalous_magnetic_moment = 0.00115965218128

# Make solenoid field instance
sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)

# Small wrapper, used to use it for x and y offsets, but kept it for simplicity.
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
    columns=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])

# Stage 1: on-axis multipoles at the tube frames.
tube_fitter = TubeFitter(
    raw_data=df_raw_data,
    n_frames=2000,
    distance_unit=1,
    deg=multipole_order - 1,
)
tube_fitter.fit()
z, F, names = tube_fitter.on_axis_multipoles()
z_bs, bs = tube_fitter.on_axis_bs()

# Stage 2, twice: coarse elements (default, ~5 frames per element) and fine
# elements (2 frames per element), to see how the element length affects
# tracking. The field is negligible at both map ends ("zero" end condition).
N_ELEMENTS_COARSE = 400
N_ELEMENTS_FINE = 1000


def build_line(n_elements):
    lf = LongitudinalFitter(0, interval, n_elements=n_elements)
    lf.fit(z, F, names)
    lf.fit(z_bs, bs, [("Bs", 0)])
    line = lf.to_line(multipole_order=multipole_order, steps_per_point=1)
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False  # enable spin tracking
    line.build_tracker()
    return line


line_splineboris = build_line(N_ELEMENTS_COARSE)
line_fine = build_line(N_ELEMENTS_FINE)
LABEL_COARSE = f"SplineBoris E={N_ELEMENTS_COARSE}"
LABEL_FINE = f"SplineBoris E={N_ELEMENTS_FINE}"

# --- TRUE REFERENCE: BorisSpatialIntegrator with same analytical field ---
# This is the gold standard - uses the same full 3D field directly
boris_integrator = xt.BorisSpatialIntegrator(
    fieldmap_callable=get_field,  # Same field function as used for fitting
    s_start=0,
    s_end=interval,
    n_steps=n_steps,
)
boris_integrator.log_trajectories = False

# --- VariableSolenoid reference (paraxial approximation, on-axis Bz only) ---
n_ref_steps = n_steps
z_axis_ref = np.linspace(0, interval, n_ref_steps)
# Get on-axis Bz
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

# Produce monitored trajectories for diagnostics/plots.
p_splineboris = p0.copy()
line_splineboris.track(p_splineboris, turn_by_turn_monitor='ONE_TURN_EBE')
mon_splineboris = line_splineboris.record_last_track

boris_integrator.log_trajectories = True
p_boris = p0.copy()
boris_integrator.track(p_boris)

p_varsol = p0.copy()
line_varsol.track(p_varsol, turn_by_turn_monitor='ONE_TURN_EBE')
mon_varsol = line_varsol.record_last_track

p_fine = p0.copy()
line_fine.track(p_fine, turn_by_turn_monitor='ONE_TURN_EBE')
mon_fine = line_fine.record_last_track

# Use mon_varsol as mon_ref for plotting
mon_ref = mon_varsol

n_part = mon_splineboris.x.shape[0]

# Boris integrator logs
x_boris = np.array(boris_integrator.x_log)
y_boris = np.array(boris_integrator.y_log)
z_boris = np.array(boris_integrator.z_log)

# --- Quantitative comparison against the BorisSpatialIntegrator reference ---
# Interpolate each method's trajectory onto the (denser) Boris s-grid, so RMS
# and end-point errors are directly comparable across methods.
def _rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

print("\n=== Trajectory deviation vs BorisSpatialIntegrator (reference) ===")
for i in range(n_part):
    s_ref, x_ref, y_ref = z_boris[:, i], x_boris[:, i], y_boris[:, i]
    print(f"--- particle {i} (delta={delta[i]}) ---")
    for label, mon in (
        (LABEL_COARSE, mon_splineboris),
        (LABEL_FINE, mon_fine),
        ("VariableSolenoid", mon_varsol),
    ):
        x_i = np.interp(s_ref, mon.s[i, :], mon.x[i, :])
        y_i = np.interp(s_ref, mon.s[i, :], mon.y[i, :])
        print(
            f"  {label:<28s} x_rms={_rms(x_i, x_ref)*1e6:8.3f} um  "
            f"y_rms={_rms(y_i, y_ref)*1e6:8.3f} um  "
            f"x_end_err={abs(x_i[-1]-x_ref[-1])*1e6:8.3f} um  "
            f"y_end_err={abs(y_i[-1]-y_ref[-1])*1e6:8.3f} um"
        )

# Plot particle tracks in 3D: x horizontal, y vertical, s longitudinal
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab10.colors
for i in range(n_part):
    # SplineBoris tracks (solid lines)
    ax.plot(mon_splineboris.s[i, :], 
            mon_splineboris.x[i, :] * 1e3, 
            mon_splineboris.y[i, :] * 1e3, 
            '-', color=colors[i], linewidth=2, alpha=0.7,
            label=f'{LABEL_COARSE} p{i}')
    # Boris integrator (dotted - this is the TRUE reference)
    ax.plot(z_boris[:, i],
            x_boris[:, i] * 1e3,
            y_boris[:, i] * 1e3,
            ':', color=colors[i], linewidth=2,
            label=f'Boris p{i}')
    # VariableSolenoid tracks (dashed lines)
    ax.plot(mon_ref.s[i, :],
            mon_ref.x[i, :] * 1e3,
            mon_ref.y[i, :] * 1e3,
            '--', color=colors[i], alpha=0.5, linewidth=1.5,
            label=f'VarSol p{i}')
    # Fine-element SplineBoris tracks (dash-dot lines)
    ax.plot(mon_fine.s[i, :],
            mon_fine.x[i, :] * 1e3,
            mon_fine.y[i, :] * 1e3,
            '-.', color=colors[i], alpha=0.8, linewidth=1.5,
            label=f'{LABEL_FINE} p{i}')

ax.set_xlabel('s [m]')
ax.set_ylabel('x [mm]')
ax.set_zlabel('y [mm]')
ax.set_title(f'{LABEL_COARSE} (solid) vs Boris (dotted) vs VarSol (dashed) vs {LABEL_FINE} (dash-dot)')
ax.legend(loc='upper left')
ax.view_init(elev=20, azim=-60)
fig.tight_layout()
plt.show()

# Also plot 2D comparison in x and y vs s (easier to see differences)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i in range(n_part):
    # x vs s
    axes[0, i].plot(mon_splineboris.s[i, :], mon_splineboris.x[i, :] * 1e3, '-',
                    label=LABEL_COARSE, linewidth=2, alpha=0.7)
    axes[0, i].plot(z_boris[:, i], x_boris[:, i] * 1e3, ':',
                    label='Boris (ref)', linewidth=2)
    axes[0, i].plot(mon_ref.s[i, :], mon_ref.x[i, :] * 1e3, '--',
                    label='VarSol', alpha=0.7)
    axes[0, i].plot(mon_fine.s[i, :], mon_fine.x[i, :] * 1e3, '-.',
                    label=LABEL_FINE, alpha=0.7)
    axes[0, i].set_xlabel('s [m]')
    axes[0, i].set_ylabel('x [mm]')
    axes[0, i].set_title(f'Particle {i} (delta={delta[i]}): x vs s')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)

    # y vs s
    axes[1, i].plot(mon_splineboris.s[i, :], mon_splineboris.y[i, :] * 1e3, '-',
                    label=LABEL_COARSE, linewidth=2, alpha=0.7)
    axes[1, i].plot(z_boris[:, i], y_boris[:, i] * 1e3, ':',
                    label='Boris (ref)', linewidth=2)
    axes[1, i].plot(mon_ref.s[i, :], mon_ref.y[i, :] * 1e3, '--',
                    label='VarSol', alpha=0.7)
    axes[1, i].plot(mon_fine.s[i, :], mon_fine.y[i, :] * 1e3, '-.',
                    label=LABEL_FINE, alpha=0.7)
    axes[1, i].set_xlabel('s [m]')
    axes[1, i].set_ylabel('y [mm]')
    axes[1, i].set_title(f'Particle {i} (delta={delta[i]}): y vs s')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot spin evolution (all three components in one graph).
i_spin = 0
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mon_splineboris.s[i_spin, :], mon_splineboris.spin_x[i_spin, :], label=r"$S_x$")
ax.plot(mon_splineboris.s[i_spin, :], mon_splineboris.spin_y[i_spin, :], label=r"$S_y$")
ax.plot(mon_splineboris.s[i_spin, :], mon_splineboris.spin_z[i_spin, :], label=r"$S_z$")
ax.set_xlabel("s [m]")
ax.set_ylabel("Spin component")
ax.set_title(f"Spin tracking (SplineBoris)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()

# Plot spin vector as 3D arrows along the trajectory.
n_arrows = 1000
i_part = 0
n_pts = mon_splineboris.s.shape[1]
idx = np.linspace(0, n_pts - 1, n_arrows, dtype=int)

s_arrow = mon_splineboris.s[i_part, idx]
x_arrow = mon_splineboris.x[i_part, idx] * 1e3  # mm
y_arrow = mon_splineboris.y[i_part, idx] * 1e3  # mm
sx = mon_splineboris.spin_x[i_part, idx]
sy = mon_splineboris.spin_y[i_part, idx]
sz = mon_splineboris.spin_z[i_part, idx]

arrow_len = 2.0  # length in data coords (mixed s [m], x/y [mm])
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(s_arrow, x_arrow, y_arrow, '-', color='gray', alpha=0.7, label='Trajectory')
ax.quiver(
    s_arrow, x_arrow, y_arrow,
    sx, sy, sz,
    length=arrow_len, normalize=True, color='C0', alpha=0.8,
    arrow_length_ratio=0.15,
)
ax.set_xlabel('s [m]')
ax.set_ylabel('x [mm]')
ax.set_zlabel('y [mm]')
ax.set_title(f'Spin vector along trajectory (particle {i_part}, {n_arrows} points)')
ax.legend()
ax.view_init(elev=20, azim=-60)
fig.tight_layout()
plt.show()