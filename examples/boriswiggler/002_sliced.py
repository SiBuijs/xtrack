"""
BorisWiggler line slicing for trajectory monitoring.

A single unsliced ``BorisWiggler`` records only entry and exit states in a
turn-by-turn monitor. Slicing into shorter segments (same total length and step
count) adds monitor points along the integration interval without changing the
physics.
"""

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt

length = 2.0
n_periods = 4
g = 0.02
B_r = 1.0
n_steps = 200
n_segments = 40

wiggler = xt.BorisWiggler(
    length=length,
    g=g,
    B_r=B_r,
    n_periods=n_periods,
    n_steps=n_steps,
)

p0 = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV,
    q0=1.0,
    energy0=1e9,
    x=1e-4,
    y=2e-4,
    px=1e-4,
    py=-5e-5,
)

line_unsliced = xt.Line(elements=[wiggler])
line_unsliced.build_tracker(use_prebuilt_kernels=False)
p_unsliced = p0.copy()
line_unsliced.track(p_unsliced, turn_by_turn_monitor="ONE_TURN_EBE")
mon_unsliced = line_unsliced.record_last_track

line_sliced = xt.Line(elements=[wiggler.copy()])
line_sliced.slice_thick_elements([
    xt.Strategy(
        slicing=xt.Uniform(n_segments, mode="thick"),
        element_type=xt.BorisWiggler,
    ),
])
line_sliced.build_tracker(use_prebuilt_kernels=False)
p_sliced = p0.copy()
line_sliced.track(p_sliced, turn_by_turn_monitor="ONE_TURN_EBE")
mon_sliced = line_sliced.record_last_track

print(f"Unsliced monitor points: {mon_unsliced.s.shape[1]}")
print(f"Sliced monitor points:   {mon_sliced.s.shape[1]}")

for name in ("x", "y", "px", "py", "zeta"):
    diff = getattr(p_unsliced, name) - getattr(p_sliced, name)
    print(f"Final {name} max |diff|: {np.max(np.abs(diff)):.3e}")

fig = plt.figure(figsize=(12, 5))

ax3d = fig.add_subplot(121, projection="3d")
ax3d.plot(
    mon_unsliced.s[0, :],
    mon_unsliced.x[0, :] * 1e3,
    mon_unsliced.y[0, :] * 1e3,
    ":",
    lw=2.5,
    label="unsliced (chord)",
)
ax3d.plot(
    mon_sliced.s[0, :],
    mon_sliced.x[0, :] * 1e3,
    mon_sliced.y[0, :] * 1e3,
    "-",
    lw=1.2,
    alpha=0.8,
    label="sliced (monitor)",
)
ax3d.set_xlabel("s [m]")
ax3d.set_ylabel("x [mm]")
ax3d.set_zlabel("y [mm]")
ax3d.set_title("3D trajectory")
ax3d.view_init(elev=20, azim=-60)
ax3d.legend()

ax_xy = fig.add_subplot(122)
ax_xy.plot(
    mon_unsliced.x[0, :] * 1e3,
    mon_unsliced.y[0, :] * 1e3,
    ":",
    lw=2.5,
    label="unsliced",
)
ax_xy.plot(
    mon_sliced.x[0, :] * 1e3,
    mon_sliced.y[0, :] * 1e3,
    "-",
    lw=1.2,
    alpha=0.8,
    label="sliced",
)
ax_xy.set_xlabel("x [mm]")
ax_xy.set_ylabel("y [mm]")
ax_xy.set_title("Transverse plane")
ax_xy.axis("equal")
ax_xy.grid(True, alpha=0.3)
ax_xy.legend()

fig.tight_layout()
plt.show()
