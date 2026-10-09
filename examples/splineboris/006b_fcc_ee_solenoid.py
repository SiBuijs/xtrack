from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import xtrack as xt
from xtrack._temp.splineboris import TubeFitter, LongitudinalFitter

"""
SplineBoris fit of the FCC-ee detector solenoid (BNL 2 T field map), tracked
at the Z-pole energy (45.6 GeV) and compared to a Boris integration through
the raw map.

The map covers one side of the IP, from the IP (z = 0) to z = 6 m, on an
11 x 11 transverse grid of +-10 mm (2 mm spacing) with 1 cm plane spacing.
Coordinates and field components are in the beam frame: the solenoid axis is
tilted by the 15 mrad half crossing angle, hence Bx ~ 0.015 * Bz on axis.
"""

plt.rcParams.update({"font.size": 14})

file_path = (Path(__file__).resolve().parent.parent.parent / "test_data"
             / "fcc_ee_solenoids" / "Field_map_2T_BNL_August_14_2026.csv")
df_raw_data = (
    pd.read_csv(file_path, usecols=["Xb", "Yb", "Zb", "Bx", "By", "Bz"])
    .rename(columns={"Xb": "X", "Yb": "Y", "Zb": "Z", "Bz": "Bs"})
    .set_index(["X", "Y", "Z"])
)

deg = 4
multipole_order = deg + 1

# Stage 1: on-axis multipoles, one frame per map plane (601 planes).
fitter = TubeFitter(raw_data=df_raw_data, n_frames=601, distance_unit=1, deg=deg)
fitter.fit()

# Stage 2: 3 frames per element (200 elements of 3 cm). The field is not zero
# at either end of the map (2 T at the IP), so the ends are free.
z, F, names = fitter.on_axis_multipoles()
lf = LongitudinalFitter(z[0], z[-1], points_per_element=3, end_condition="free")
lf.fit(z, F, names)
lf.fit(*fitter.on_axis_bs(), [("Bs", 0)])

for der in range(deg):
    lf.plot_fields(der=der)

# The map is symmetric in y, so all By_n vanish; field_tol drops them (and
# any other component below 1e-4 of the largest field at r_ref = 10 mm).
line = lf.to_line(multipole_order=multipole_order, steps_per_point=10,
                  field_tol=1e-4, r_ref=0.01)
line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False  # enable spin tracking
line.build_tracker()

# Reference: Boris integration through the raw map (cubic interpolation).
x_ax, y_ax, z_ax = (np.unique(df_raw_data.index.get_level_values(c)) for c in "XYZ")
shape = (len(x_ax), len(y_ax), len(z_ax))
df_sorted = df_raw_data.sort_index()
interps = [RegularGridInterpolator((x_ax, y_ax, z_ax), df_sorted[c].to_numpy().reshape(shape),
                                   method="cubic")
           for c in ("Bx", "By", "Bs")]


def get_field(x, y, z):
    pts = np.column_stack(np.broadcast_arrays(x, y, z))
    return tuple(f(pts) for f in interps)


boris_integrator = xt.BorisSpatialIntegrator(
    fieldmap_callable=get_field, s_start=z[0], s_end=z[-1], n_steps=6000)
boris_integrator.log_trajectories = True

# Particles at FCC-ee Z-pole energy, starting off axis at the IP.
delta = np.array([0, 1e-2])
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                  energy0=45.6e9,
                  x=1e-3, px=-1e-4 * (1 + delta),
                  y=1e-3, py=2e-4,
                  delta=delta)
p0.spin_x = 1.0
p0.spin_y = 0.0
p0.spin_z = 0.0
p0.anomalous_magnetic_moment = 0.00115965218128

p_sb = p0.copy()
line.track(p_sb, turn_by_turn_monitor="ONE_TURN_EBE")
mon = line.record_last_track

p_boris = p0.copy()
boris_integrator.track(p_boris)
s_boris = np.array(boris_integrator.z_log)
x_boris = np.array(boris_integrator.x_log)
y_boris = np.array(boris_integrator.y_log)

for coord in ("x", "px", "y", "py"):
    diff = getattr(p_sb, coord) - getattr(p_boris, coord)
    print(f"{coord:>2} at s = {z[-1]:g} m: SplineBoris - Boris = {diff}")

# Trajectories, and their difference interpolated on the element boundaries.
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
for i in range(len(delta)):
    for row, (coord, ref) in enumerate((("x", x_boris), ("y", y_boris))):
        sb = getattr(mon, coord)[i]
        axes[0, row].plot(mon.s[i], sb * 1e3, "-", color=f"C{i}", lw=2, alpha=0.7,
                          label=f"SplineBoris, delta={delta[i]:g}")
        axes[0, row].plot(s_boris[:, i], ref[:, i] * 1e3, ":", color=f"C{i}", lw=2,
                          label=f"Boris raw map, delta={delta[i]:g}")
        axes[1, row].plot(mon.s[i], (sb - np.interp(mon.s[i], s_boris[:, i], ref[:, i])) * 1e6,
                          color=f"C{i}", label=f"delta={delta[i]:g}")
        axes[0, row].set_ylabel(f"{coord} [mm]")
        axes[1, row].set_ylabel(f"{coord} SplineBoris - Boris [um]")
for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
for ax in axes[1]:
    ax.set_xlabel("s [m]")
fig.suptitle("FCC-ee solenoid, 45.6 GeV")
fig.tight_layout()

# Spin of the on-momentum particle.
fig, ax = plt.subplots(figsize=(10, 5))
for comp in ("x", "y", "z"):
    ax.plot(mon.s[0], getattr(mon, f"spin_{comp}")[0], label=rf"$S_{comp}$")
ax.set_xlabel("s [m]")
ax.set_ylabel("Spin component")
ax.set_title("Spin tracking (SplineBoris), 45.6 GeV")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
