import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import pandas as pd
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack._temp.field_fitter import FieldFitter
import matplotlib.pyplot as plt
import time
# Set basic parameters
interval = 30
dx = 0.001
dy = 0.001
multipole_order = 6
n_steps = 5000
S_PLOT_MIN_M = 0#16.0
S_PLOT_MAX_M = 30#24.0
TICK_FONTSIZE = 16
LABEL_FONTSIZE = 17
TITLE_FONTSIZE = 18

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


def benchmark_track(track_callable, n_warmup=1, n_repeats=7):
    # First-call timing includes one-time effects.
    t0 = time.perf_counter()
    track_callable()
    first_call_s = time.perf_counter() - t0

    # Warm-up before steady-state timing.
    for _ in range(n_warmup):
        track_callable()

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        track_callable()
        times.append(time.perf_counter() - t0)

    return {
        "first_call_s": float(first_call_s),
        "median_s": float(np.median(times)),
    }


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

fitter = FieldFitter(
    raw_data=df_raw_data,
    xy_point=(0, 0),
    distance_unit=1,
    min_region_size=10,
    deg=multipole_order - 1,
    field_tol=1e-8,
)
fitter.fit()
df_fit_pars = fitter.df_fit_pars

# Plot only the longitudinal component (Bs), similar to FieldFitter plotting.
s_fit = fitter.s_full
bs_raw = fitter.df_on_axis_raw[("Bs", 0)].to_numpy()
bs_fit = fitter.df_on_axis_fit[("Bs", 0)].to_numpy()

fig, ax = plt.subplots(figsize=(10, 4), dpi=140, constrained_layout=True)
ax.plot(s_fit, bs_raw, label=r"$B_s$ raw", linewidth=2)
ax.plot(s_fit, bs_fit, "--", label=r"$B_s$ fit", linewidth=2)

# Draw piece boundaries for Bs, derivative_x=0 fit segments.
idx = fitter.df_fit_pars.index
comp = np.asarray(idx.get_level_values("field_component"))
der_x = np.asarray(idx.get_level_values("derivative_x")).astype(int)
mask = (comp == "Bs") & (der_x == 0)
if np.any(mask):
    s_start = np.asarray(idx.get_level_values("s_start"))[mask].astype(float)
    s_end = np.asarray(idx.get_level_values("s_end"))[mask].astype(float)
    for sb in np.unique(np.concatenate((s_start, s_end))):
        ax.axvline(sb, color="k", linestyle="--", linewidth=1, alpha=0.2)

ax.set_title("Longitudinal field fit (central axis)", fontsize=TITLE_FONTSIZE)
ax.set_xlabel("s [m]", fontsize=LABEL_FONTSIZE)
ax.set_ylabel(r"$B_s$ [T]", fontsize=LABEL_FONTSIZE)
ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
plt.show()
# Build solenoid using SplineBorisSequence - automatically creates one SplineBoris
# element per polynomial piece with n_steps based on the data point count
seq = xt.SplineBorisSequence(
    df_fit_pars=df_fit_pars,
    multipole_order=multipole_order,
    steps_per_point=1,  # one integration step per data point
)

# # Evaluate the reconstructed field along the on-axis longitudinal direction
# x0, y0 = fitter.xy_point
# Bx_eval = np.array([seq.evaluate_field(x0, y0, s)[0] for s in fitter.s_full])
# By_eval = np.array([seq.evaluate_field(x0, y0, s)[1] for s in fitter.s_full])
# Bs_eval = np.array([seq.evaluate_field(x0, y0, s)[2] for s in fitter.s_full])

# # Compare the SplineBorisSequence field evaluation against the analytical solenoid field
# Bx_ref, By_ref, Bs_ref = sf.get_field(
#     x0 * np.ones_like(fitter.s_full),
#     y0 * np.ones_like(fitter.s_full),
#     fitter.s_full,
# )

# fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6), constrained_layout=True)
# for ax, comp_eval, comp_ref, label in zip(
#     [ax1, ax2, ax3],
#     [Bx_eval, By_eval, Bs_eval],
#     [Bx_ref, By_ref, Bs_ref],
#     [r"$B_x$", r"$B_y$", r"$B_s$"],
# ):
#     ax.plot(fitter.s_full, comp_ref, label="Analytical")
#     ax.plot(fitter.s_full, comp_eval, "--", label="SplineBorisSequence")
#     ax.set_ylabel(f"{label} [T]")
#     ax.legend()
#     ax.grid(True, alpha=0.3)

# ax1.set_title(f"On-axis field at (x, y) = ({x0}, {y0})")
# ax3.set_xlabel("s [m]")
# plt.show()

# Get the Line of SplineBoris elements
line_splineboris = seq.to_line()
# Needed for spin transport in line elements (default is True -> spin off).
line_splineboris.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
line_splineboris.build_tracker()

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
line_varsol.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
line_varsol.build_tracker()

# n_warmup = 1
# n_repeats = 7
# print(f"Benchmark settings: n_warmup={n_warmup}, n_repeats={n_repeats}")

# t_spline = benchmark_track(
#     lambda: line_splineboris.track(p0.copy()),
#     n_warmup=n_warmup,
#     n_repeats=n_repeats,
# )
# t_boris = benchmark_track(
#     lambda: boris_integrator.track(p0.copy()),
#     n_warmup=n_warmup,
#     n_repeats=n_repeats,
# )
# t_varsol = benchmark_track(
#     lambda: line_varsol.track(p0.copy()),
#     n_warmup=n_warmup,
#     n_repeats=n_repeats,
# )

# print(f"SplineBoris first call:          {t_spline['first_call_s']:.6f} s")
# print(f"BorisSpatialIntegrator first call: {t_boris['first_call_s']:.6f} s")
# print(f"VariableSolenoid first call:     {t_varsol['first_call_s']:.6f} s")
# print("")
# print(f"SplineBoris steady-state:          {t_spline['median_s']:.6f} s (median)")
# print(f"BorisSpatialIntegrator steady-state: {t_boris['median_s']:.6f} s (median)")
# print(f"VariableSolenoid steady-state:     {t_varsol['median_s']:.6f} s (median)")
# print("")
# print(f"Speedup spline vs boris (steady): {t_boris['median_s'] / t_spline['median_s']:.2f}x")
# print(f"Speedup spline vs varsol (steady): {t_varsol['median_s'] / t_spline['median_s']:.2f}x")

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

# Use mon_varsol as mon_ref for plotting
mon_ref = mon_varsol

for name, mon in [("SplineBoris", mon_splineboris), ("VariableSolenoid", mon_varsol)]:
    sx0, sy0, sz0 = mon.spin_x[0, 0], mon.spin_y[0, 0], mon.spin_z[0, 0]
    sxf, syf, szf = mon.spin_x[0, -1], mon.spin_y[0, -1], mon.spin_z[0, -1]
    print(
        f"{name} spin p0: "
        f"start=({sx0:+.6f},{sy0:+.6f},{sz0:+.6f}) "
        f"end=({sxf:+.6f},{syf:+.6f},{szf:+.6f})"
    )

# Plot particle tracks in 3D: x horizontal, y vertical, s longitudinal
n_part = mon_splineboris.x.shape[0]
fig = plt.figure(figsize=(12, 8), dpi=140)
ax = fig.add_subplot(111, projection='3d')

# Boris integrator logs
x_boris = np.array(boris_integrator.x_log)
y_boris = np.array(boris_integrator.y_log)
z_boris = np.array(boris_integrator.z_log)

colors = plt.cm.tab10.colors
for i in range(n_part):
    # SplineBoris tracks (solid lines)
    ax.plot(mon_splineboris.s[i, :], 
            mon_splineboris.x[i, :] * 1e3, 
            mon_splineboris.y[i, :] * 1e3, 
            '-', color=colors[i], alpha=0.5, linewidth=2,
            label=f'SplineBoris p{i}')
    # Boris integrator (dotted - this is the TRUE reference)
    ax.plot(z_boris[:, i], 
            x_boris[:, i] * 1e3, 
            y_boris[:, i] * 1e3, 
            ':', color=colors[i], linewidth=2,
            label=f'Boris p{i}')
    # # VariableSolenoid tracks (dashed lines)
    # ax.plot(mon_ref.s[i, :], 
    #         mon_ref.x[i, :] * 1e3, 
    #         mon_ref.y[i, :] * 1e3, 
    #         '--', color=colors[i], alpha=0.5, linewidth=1.5,
    #         label=f'VarSol p{i}')

ax.set_xlabel('s [m]', fontsize=13, labelpad=8)
ax.set_ylabel('x [mm]', fontsize=13, labelpad=8)
ax.set_zlabel('y [mm]', fontsize=13, labelpad=8)
ax.set_xlim(S_PLOT_MIN_M, S_PLOT_MAX_M)
ax.set_title(
    'SplineBoris (solid) vs Boris ref (dotted) vs VariableSolenoid (dashed)',
    fontsize=14,
    pad=12,
)
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=11)
ax.tick_params(axis='z', labelsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.view_init(elev=20, azim=-60)
fig.tight_layout()
plt.show()

# Also plot 2D comparison in x and y vs s (easier to see differences)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=140)

for i in range(n_part):
    # x vs s
    axes[0, i].plot(mon_splineboris.s[i, :], mon_splineboris.x[i, :] * 1e3, '-', 
                    label='SplineBoris', linewidth=2)
    axes[0, i].plot(z_boris[:, i], x_boris[:, i] * 1e3, ':', 
                    label='Boris (ref)', linewidth=2)
    axes[0, i].plot(mon_ref.s[i, :], mon_ref.x[i, :] * 1e3, '--', 
                    label='VarSol', alpha=0.7)
    axes[0, i].set_xlabel('s [m]')
    axes[0, i].set_ylabel('x [mm]')
    axes[0, i].set_xlim(S_PLOT_MIN_M, S_PLOT_MAX_M)
    axes[0, i].set_title(f'Particle {i} (delta={delta[i]}): x vs s', fontsize=13)
    axes[0, i].tick_params(axis='x', labelsize=11)
    axes[0, i].tick_params(axis='y', labelsize=11)
    axes[0, i].legend(fontsize=10)
    axes[0, i].grid(True, alpha=0.3)
    
    # y vs s
    axes[1, i].plot(mon_splineboris.s[i, :], mon_splineboris.y[i, :] * 1e3, '-', 
                    label='SplineBoris', linewidth=2)
    axes[1, i].plot(z_boris[:, i], y_boris[:, i] * 1e3, ':', 
                    label='Boris (ref)', linewidth=2)
    axes[1, i].plot(mon_ref.s[i, :], mon_ref.y[i, :] * 1e3, '--', 
                    label='VarSol', alpha=0.7)
    axes[1, i].set_xlabel('s [m]')
    axes[1, i].set_ylabel('y [mm]')
    axes[1, i].set_xlim(S_PLOT_MIN_M, S_PLOT_MAX_M)
    axes[1, i].set_title(f'Particle {i} (delta={delta[i]}): y vs s', fontsize=13)
    axes[1, i].tick_params(axis='x', labelsize=11)
    axes[1, i].tick_params(axis='y', labelsize=11)
    axes[1, i].legend(fontsize=10)
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot spin evolution for first particle.
i_spin = 0
fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
ax.plot(mon_splineboris.s[i_spin, :], mon_splineboris.spin_x[i_spin, :], label=r"$S_x$")
ax.plot(mon_splineboris.s[i_spin, :], mon_splineboris.spin_y[i_spin, :], label=r"$S_y$")
ax.plot(mon_splineboris.s[i_spin, :], mon_splineboris.spin_z[i_spin, :], label=r"$S_z$")
ax.set_xlabel("s [m]", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Spin component", fontsize=LABEL_FONTSIZE)
ax.set_xlim(S_PLOT_MIN_M, S_PLOT_MAX_M)
ax.set_title("Spin tracking (SplineBoris)", fontsize=TITLE_FONTSIZE)
ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
fig.tight_layout()
plt.show()

# If available, overlay VariableSolenoid spin components for comparison.
if hasattr(mon_ref, "spin_x") and mon_ref.spin_x is not None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    ax.plot(mon_splineboris.s[i_spin, :], mon_splineboris.spin_x[i_spin, :], "-", linewidth=2, label=r"SplineBoris $S_x$")
    ax.plot(mon_splineboris.s[i_spin, :], mon_splineboris.spin_y[i_spin, :], "-", linewidth=2, label=r"SplineBoris $S_y$")
    ax.plot(mon_splineboris.s[i_spin, :], mon_splineboris.spin_z[i_spin, :], "-", linewidth=2, label=r"SplineBoris $S_z$")
    ax.plot(mon_ref.s[i_spin, :], mon_ref.spin_x[i_spin, :], "--", alpha=0.8, label=r"VarSol $S_x$")
    ax.plot(mon_ref.s[i_spin, :], mon_ref.spin_y[i_spin, :], "--", alpha=0.8, label=r"VarSol $S_y$")
    ax.plot(mon_ref.s[i_spin, :], mon_ref.spin_z[i_spin, :], "--", alpha=0.8, label=r"VarSol $S_z$")
    ax.set_xlabel("s [m]", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Spin component", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(S_PLOT_MIN_M, S_PLOT_MAX_M)
    ax.set_title("Spin tracking: SplineBoris vs VariableSolenoid", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=11)
    fig.tight_layout()
    plt.show()

# Plot spin vector as 3D arrows along trajectory.
n_arrows = 500
i_part = 0
n_pts = mon_splineboris.s.shape[1]
idx = np.linspace(0, n_pts - 1, n_arrows, dtype=int)

s_arrow = mon_splineboris.s[i_part, idx]
x_arrow = mon_splineboris.x[i_part, idx] * 1e3  # mm
y_arrow = mon_splineboris.y[i_part, idx] * 1e3  # mm
sx = mon_splineboris.spin_x[i_part, idx]
sy = mon_splineboris.spin_y[i_part, idx]
sz = mon_splineboris.spin_z[i_part, idx]

arrow_len = 2.0  # mixed axis units (s in m, x/y in mm)
fig = plt.figure(figsize=(10, 8), dpi=140)
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
ax.set_xlim(S_PLOT_MIN_M, S_PLOT_MAX_M)
ax.set_title(f'Spin vector along trajectory (particle {i_part}, {n_arrows} points)')
ax.legend()
ax.view_init(elev=20, azim=-60)
fig.tight_layout()
plt.show()
