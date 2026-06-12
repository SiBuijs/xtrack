"""
BorisSolenoid line slicing for trajectory monitoring.

A single unsliced ``BorisSolenoid`` records only entry and exit states in a
turn-by-turn monitor, so trajectory plots look like straight chords. Slicing the
element into many shorter ``BorisSolenoid`` segments (same total length and step
count) adds monitor points along the integration interval without changing the
physics.
"""

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

plt.rcParams.update({"font.size": 14})

# Same solenoid as 001_basic.py
L_coil = 4.0
a = 0.3
B0 = 1.5
z0 = 20.0
length = 30.0
n_steps = 3000
n_segments = 1000

solenoid = xt.BorisSolenoid(
    L_coil=L_coil,
    a=a,
    B0=B0,
    z0=z0,
    length=length,
    n_steps=n_steps,
)

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

# --- Unsliced line: monitor has only entry + exit ---
line_unsliced = xt.Line(elements=[solenoid])
line_unsliced.build_tracker()
p_unsliced = p0.copy()
line_unsliced.track(p_unsliced, turn_by_turn_monitor="ONE_TURN_EBE")
mon_unsliced = line_unsliced.record_last_track

# --- Sliced line: N segments, N+1 monitor points along s ---
line_sliced = xt.Line(elements=[solenoid.copy()])
line_sliced.slice_thick_elements([
    xt.Strategy(
        slicing=xt.Uniform(n_segments, mode="thick"),
        element_type=xt.BorisSolenoid,
    ),
])
line_sliced.build_tracker()
p_sliced = p0.copy()
line_sliced.track(p_sliced, turn_by_turn_monitor="ONE_TURN_EBE")
mon_sliced = line_sliced.record_last_track

print(f"Unsliced monitor points per particle: {mon_unsliced.s.shape[1]}")
print(f"Sliced monitor points per particle:   {mon_sliced.s.shape[1]}")

for name in ("x", "y", "px", "py", "zeta"):
    diff = getattr(p_unsliced, name) - getattr(p_sliced, name)
    print(f"Final {name} max |diff|: {np.max(np.abs(diff)):.3e}")

atol = 1e-5
for name in ("x", "y", "px", "py", "zeta"):
    assert np.allclose(
        getattr(p_unsliced, name), getattr(p_sliced, name), atol=atol, rtol=0
    ), name

# --- BorisSpatialIntegrator reference (same field, same n_steps) ---
sf = SolenoidField(L=L_coil, a=a, B0=B0, z0=z0)
boris_spatial = xt.BorisSpatialIntegrator(
    fieldmap_callable=sf.get_field,
    s_start=0,
    s_end=length,
    n_steps=n_steps,
)
boris_spatial.log_trajectories = True
p_spatial = p0.copy()
boris_spatial.track(p_spatial)

x_spatial = np.array(boris_spatial.x_log)
y_spatial = np.array(boris_spatial.y_log)
s_spatial = np.array(boris_spatial.z_log)

print("End-state difference vs BorisSpatialIntegrator:")
for name in ("x", "y", "px", "py", "zeta"):
    for label, p in (("unsliced", p_unsliced), ("sliced", p_sliced)):
        diff = getattr(p, name) - getattr(p_spatial, name)
        print(f"  {label} {name}: max |diff| = {np.max(np.abs(diff)):.3e}")

# --- Trajectory plots: chord vs curved polyline vs Boris spatial ---
n_part = p0.x.shape[0]
colors = plt.cm.tab10.colors
# Lower alpha on the dense sliced trace so chord and reference stay visible.
STYLE = {
    "unsliced": {"ls": ":", "lw": 2.5, "alpha": 0.95},
    "sliced": {"ls": "-", "lw": 1.2, "alpha": 0.55},
    "spatial": {"ls": "--", "lw": 2.0, "alpha": 0.95},
}

fig = plt.figure(figsize=(16, 5), constrained_layout=True)
ax3d = fig.add_subplot(131, projection="3d")
ax_xs = fig.add_subplot(132)
ax_xy = fig.add_subplot(133)

for i in range(n_part):
    ax3d.plot(
        mon_unsliced.s[i, :],
        mon_unsliced.x[i, :] * 1e3,
        mon_unsliced.y[i, :] * 1e3,
        color=colors[i],
        label="unsliced (chord)" if i == 0 else None,
        **STYLE["unsliced"],
    )
    ax3d.plot(
        mon_sliced.s[i, :],
        mon_sliced.x[i, :] * 1e3,
        mon_sliced.y[i, :] * 1e3,
        color=colors[i],
        label="sliced (monitor)" if i == 0 else None,
        **STYLE["sliced"],
    )
    ax3d.plot(
        s_spatial[:, i],
        x_spatial[:, i] * 1e3,
        y_spatial[:, i] * 1e3,
        color=colors[i],
        label="BorisSpatialIntegrator" if i == 0 else None,
        **STYLE["spatial"],
    )
    ax_xs.plot(
        mon_unsliced.s[i, :],
        mon_unsliced.x[i, :] * 1e3,
        color=colors[i],
        **STYLE["unsliced"],
    )
    ax_xs.plot(
        mon_sliced.s[i, :],
        mon_sliced.x[i, :] * 1e3,
        color=colors[i],
        **STYLE["sliced"],
    )
    ax_xs.plot(
        s_spatial[:, i],
        x_spatial[:, i] * 1e3,
        color=colors[i],
        **STYLE["spatial"],
    )
    ax_xy.plot(
        mon_unsliced.x[i, :] * 1e3,
        mon_unsliced.y[i, :] * 1e3,
        color=colors[i],
        **STYLE["unsliced"],
    )
    ax_xy.plot(
        mon_sliced.x[i, :] * 1e3,
        mon_sliced.y[i, :] * 1e3,
        color=colors[i],
        **STYLE["sliced"],
    )
    ax_xy.plot(
        x_spatial[:, i] * 1e3,
        y_spatial[:, i] * 1e3,
        color=colors[i],
        **STYLE["spatial"],
    )

ax3d.set_xlabel("s [m]")
ax3d.set_ylabel("x [mm]")
ax3d.set_zlabel("y [mm]")
ax3d.set_title("3D trajectory")
ax3d.view_init(elev=20, azim=-60)
ax3d.legend(fontsize=10)

ax_xs.set_xlabel("s [m]")
ax_xs.set_ylabel("x [mm]")
ax_xs.set_title("x(s)")
ax_xs.grid(True, alpha=0.3)

ax_xy.set_xlabel("x [mm]")
ax_xy.set_ylabel("y [mm]")
ax_xy.set_title("Transverse plane")
ax_xy.grid(True, alpha=0.3)
ax_xy.axis("equal")

plt.savefig("borissolenoid_sliced_trajectories.png", dpi=150)
plt.show()
