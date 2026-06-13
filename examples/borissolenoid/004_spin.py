"""
Spin tracking through ``BorisSolenoid`` with vertical initial polarization.

An electron with ``spin_y = 1`` is tracked through the analytical solenoid
field. The element is sliced so the turn-by-turn monitor records spin
components along the integration path. Spin precession follows the
Thomas–BMT equation implemented in ``track_borissolenoid.h``.
"""

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt

plt.rcParams.update({"font.size": 14})

# ---------------------------------------------------------------
# Solenoid and tracking parameters
# ---------------------------------------------------------------

L_coil = 4.0
a = 0.3
B0 = 1.5
z0 = 20.0
length = 30.0
n_steps = 5000
n_segments = 50

G_spin = 0.00115965218128  # electron anomalous magnetic moment

solenoid = xt.BorisSolenoid(
    L_coil=L_coil,
    a=a,
    B0=B0,
    z0=z0,
    length=length,
    n_steps=n_steps,
)

p0 = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV,
    q0=1,
    p0c=700e9,
    anomalous_magnetic_moment=G_spin,
    x=0.0,
    px=0.0,
    y=0.0,
    py=0.0,
    delta=0.0,
    spin_x=0.0,
    spin_y=1.0,
    spin_z=0.0,
)

# ---------------------------------------------------------------
# Track with spin and monitor along sliced element
# ---------------------------------------------------------------

line = xt.Line(elements=[solenoid])
line.slice_thick_elements([
    xt.Strategy(
        slicing=xt.Uniform(n_segments, mode="thick"),
        element_type=xt.BorisSolenoid,
    ),
])
line.build_tracker()
line.configure_spin(spin_model="auto")

p = p0.copy()
line.track(p, turn_by_turn_monitor="ONE_TURN_EBE")
mon = line.record_last_track

spin_norm = np.sqrt(mon.spin_x**2 + mon.spin_y**2 + mon.spin_z**2)
print("Initial spin:", p0.spin_x[0], p0.spin_y[0], p0.spin_z[0])
print("Final spin:  ", p.spin_x[0], p.spin_y[0], p.spin_z[0])
print(f"Spin norm along track: min = {spin_norm.min():.6f}, max = {spin_norm.max():.6f}")

# ---------------------------------------------------------------
# Spin components vs s (one panel per component)
# ---------------------------------------------------------------

s = mon.s[0, :]
spin = {
    "x": mon.spin_x[0, :],
    "y": mon.spin_y[0, :],
    "z": mon.spin_z[0, :],
}

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)

panel_info = (
    ("x", r"$S_x$", "tab:blue"),
    ("y", r"$1 - S_y$", "tab:orange"),
    ("z", r"$S_z$", "tab:green"),
)
for ax, (key, ylabel, color) in zip(axes, panel_info):
    data = 1.0 - spin[key] if key == "y" else spin[key]
    ax.plot(s, data, color=color, linewidth=2)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs s")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.4)

axes[-1].set_xlabel("s [m]")
fig.suptitle(r"Spin precession through solenoid, initial $S_y = 1$", fontsize=15)

plt.savefig("borissolenoid_spin_y_precession.png", dpi=150)
plt.show()
