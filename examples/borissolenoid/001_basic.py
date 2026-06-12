"""
Minimal introduction to the ``BorisSolenoid`` element.

The element integrates the Lorentz force with a field-aligned helical
exponential map in the analytical magnetic field of a finite-length
circular solenoid (Hampton et al., elliptic-integral closed form). The
coil parameters are stored directly on the element — no field fitting
required.

This example:

- builds a ``BorisSolenoid`` and checks ``get_field`` against the Python
  reference ``SolenoidField``,
- tracks particles through a single-element line,
- benchmarks the result against the slow Python ``BorisSpatialIntegrator``.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

plt.rcParams.update({"font.size": 14})

# ---------------------------------------------------------------
# Solenoid and element parameters
# ---------------------------------------------------------------

L_coil = 4.0      # coil length [m]
a = 0.3           # coil radius [m]
B0 = 1.5          # on-axis peak field at coil centre [T]
z0 = 20.0         # coil centre in global s [m]
length = 30.0     # Boris integration extent [m]
n_steps = 40000

solenoid = xt.BorisSolenoid(
    L_coil=L_coil,
    a=a,
    B0=B0,
    z0=z0,
    length=length,
    n_steps=n_steps,
)

sf = SolenoidField(L=L_coil, a=a, B0=B0, z0=z0)

# ---------------------------------------------------------------
# Field check on axis (element placed at s = 0)
# ---------------------------------------------------------------

s_local = np.linspace(0, length, 301)
bx_el, by_el, bz_el = solenoid.get_field(0.0, 0.0, s_local, s_at_element=0.0)
bx_ref, by_ref, bz_ref = sf.get_field(
    np.zeros_like(s_local),
    np.zeros_like(s_local),
    s_local,
)

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.plot(s_local, bz_ref, label=r"$B_z$ reference (SolenoidField)")
ax.plot(s_local, bz_el, "k--", label=r"$B_z$ BorisSolenoid.get_field")
ax.axvline(z0 - L_coil / 2, color="k", ls=":", alpha=0.4, label="coil ends")
ax.axvline(z0 + L_coil / 2, color="k", ls=":", alpha=0.4)
ax.set_xlabel("s [m]")
ax.set_ylabel(r"$B_z$ on axis [T]")
ax.set_title("On-axis field")
ax.grid(True, alpha=0.3)
ax.legend()
plt.savefig("borissolenoid_on_axis_field.png", dpi=150)
plt.show()

print("On-axis Bz max relative error:",
      np.max(np.abs(bz_el - bz_ref) / np.maximum(np.abs(bz_ref), 1e-30)))

# ---------------------------------------------------------------
# Particle tracking
# ---------------------------------------------------------------

delta = np.array([0.0, 4.0])
p0 = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV,
    q0=1,
    energy0=45.6e9 / 1000,
    x=[-1e-3, -1e-3],
    px=-1e-3 * (1 + delta),
    y=1e-3,
    delta=delta,
)

line = xt.Line(elements=[solenoid])

t0 = time.perf_counter()
line.build_tracker()
t_build = time.perf_counter() - t0

p_xt = p0.copy()
t0 = time.perf_counter()
line.track(p_xt, turn_by_turn_monitor="ONE_TURN_EBE")
t_borissolenoid = time.perf_counter() - t0
mon = line.record_last_track

# Slow Python reference integrator (same analytical field)
boris_ref = xt.BorisSpatialIntegrator(
    fieldmap_callable=sf.get_field,
    s_start=0,
    s_end=length,
    n_steps=n_steps,
)
boris_ref.log_trajectories = True
p_ref = p0.copy()
t0 = time.perf_counter()
boris_ref.track(p_ref)
t_boris_spatial = time.perf_counter() - t0

print("Tracking time comparison:")
print(f"  BorisSolenoid build_tracker:     {t_build:.3f} s  (one-time kernel compile)")
print(f"  BorisSolenoid track:             {t_borissolenoid:.3f} s  "
      f"({n_steps} steps, {p0.x.shape[0]} particles)")
print(f"  BorisSpatialIntegrator track:    {t_boris_spatial:.3f} s  "
      f"({n_steps} steps, {p0.x.shape[0]} particles)")
print(f"  Speedup (track only):            {t_boris_spatial / t_borissolenoid:.1f}x")
print(f"  Speedup incl. build_tracker:     "
      f"{t_boris_spatial / (t_build + t_borissolenoid):.1f}x")

x_boris = np.array(boris_ref.x_log)
y_boris = np.array(boris_ref.y_log)
s_boris = np.array(boris_ref.z_log)

print("End-state difference vs BorisSpatialIntegrator:")
for name in ("x", "y", "px", "py", "zeta"):
    diff = getattr(p_xt, name) - getattr(p_ref, name)
    print(f"  {name}: max |diff| = {np.max(np.abs(diff)):.3e}")

# ---------------------------------------------------------------
# Trajectory comparison plots
# ---------------------------------------------------------------

n_part = p0.x.shape[0]
colors = plt.cm.tab10.colors

fig = plt.figure(figsize=(12, 6))
ax3d = fig.add_subplot(121, projection="3d")
ax_xy = fig.add_subplot(122)

for i in range(n_part):
    ax3d.plot(
        mon.s[i, :],
        mon.x[i, :] * 1e3,
        mon.y[i, :] * 1e3,
        "-",
        color=colors[i],
        linewidth=2,
        alpha=0.8,
        label=f"BorisSolenoid p{i}",
    )
    ax3d.plot(
        s_boris[:, i],
        x_boris[:, i] * 1e3,
        y_boris[:, i] * 1e3,
        ":",
        color=colors[i],
        linewidth=2,
        label=f"BorisSpatial p{i}",
    )
    ax_xy.plot(
        mon.x[i, :] * 1e3,
        mon.y[i, :] * 1e3,
        "-",
        color=colors[i],
        linewidth=2,
        alpha=0.8,
        label=f"BorisSolenoid p{i}",
    )
    ax_xy.plot(
        x_boris[:, i] * 1e3,
        y_boris[:, i] * 1e3,
        ":",
        color=colors[i],
        linewidth=2,
        label=f"BorisSpatial p{i}",
    )

ax3d.set_xlabel("s [m]")
ax3d.set_ylabel("x [mm]")
ax3d.set_zlabel("y [mm]")
ax3d.set_title("BorisSolenoid (solid) vs BorisSpatialIntegrator (dotted)")
ax3d.legend(loc="upper left", fontsize=10)
ax3d.view_init(elev=20, azim=-60)

ax_xy.set_xlabel("x [mm]")
ax_xy.set_ylabel("y [mm]")
ax_xy.set_title("Transverse plane")
ax_xy.grid(True, alpha=0.3)
ax_xy.axis("equal")
ax_xy.legend(fontsize=10)

fig.tight_layout()
plt.savefig("borissolenoid_trajectories.png", dpi=150)
plt.show()

# x(s) and y(s) for each particle
fig, axes = plt.subplots(2, n_part, figsize=(5 * n_part, 8), squeeze=False)

for i in range(n_part):
    axes[0, i].plot(mon.s[i, :], mon.x[i, :] * 1e3, "-", label="BorisSolenoid", linewidth=2)
    axes[0, i].plot(s_boris[:, i], x_boris[:, i] * 1e3, ":", label="BorisSpatial", linewidth=2)
    axes[0, i].set_xlabel("s [m]")
    axes[0, i].set_ylabel("x [mm]")
    axes[0, i].set_title(f"Particle {i} (delta={delta[i]}): x vs s")
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)

    axes[1, i].plot(mon.s[i, :], mon.y[i, :] * 1e3, "-", label="BorisSolenoid", linewidth=2)
    axes[1, i].plot(s_boris[:, i], y_boris[:, i] * 1e3, ":", label="BorisSpatial", linewidth=2)
    axes[1, i].set_xlabel("s [m]")
    axes[1, i].set_ylabel("y [mm]")
    axes[1, i].set_title(f"Particle {i} (delta={delta[i]}): y vs s")
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("borissolenoid_trajectory_comparison.png", dpi=150)
plt.show()
