"""2D depiction of the main detector solenoid's field in the y-s plane.

Uses the closed-form elliptic-integral field of SolenoidField (Hampton et
al., see xtrack/_temp/boris_and_solenoid_map/solenoid_field.py) evaluated at
x=0, i.e. the plane containing the beam axis (s) and the vertical (y). Draws
the (By, Bz) field vectors as unit-length arrows, coloured by local field
magnitude |B| = sqrt(By^2 + Bz^2). Geometry is the main-solenoid 2T case
from solenoid_params.py (untilted -- THETA is negligible for this picture).
"""
import matplotlib.pyplot as plt
import numpy as np

from solenoid_params import MAIN_SOLENOID_A, half_length_for_b0
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

B0 = 2.0
a = MAIN_SOLENOID_A
L = 2 * half_length_for_b0(B0)  # full length [m]

field_model = SolenoidField(L=L, a=a, B0=B0, z0=0.0)

# Plot area: s spans 1.5x the solenoid length, y spans 2x the solenoid
# diameter, both centred on the solenoid midpoint/axis.
half_extent_s = 0.75 * L
half_extent_y = 2 * a
n_points_s = 33
n_points_y = 23
s_axis = np.linspace(-half_extent_s, half_extent_s, n_points_s)
y_axis = np.linspace(-half_extent_y, half_extent_y, n_points_y)
S, Y = np.meshgrid(s_axis, y_axis)
X = np.zeros_like(S)

_, By, Bz = field_model.get_field(X, Y, S)
B_mag = np.sqrt(By**2 + Bz**2)

# Unit-length arrows: direction only, colour carries magnitude.
norm = np.where(B_mag > 0, B_mag, 1.0)
U = Bz / norm
V = By / norm

# Two panels stacked on a shared s-axis: field-vector map on top, on-axis
# Bz(s) below. Both live in gridspec column 0 (the colorbar gets its own
# column, next to the top panel only) so the two panels keep exactly the
# same width and their s-axes line up.
fig = plt.figure(figsize=(7, 6.5))
gs = fig.add_gridspec(
    2, 2, height_ratios=[2, 1], width_ratios=[30, 1],
    hspace=0.08, wspace=0.05)
ax = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax)

quiv = ax.quiver(
    S, Y, U, V, B_mag, cmap='viridis', pivot='mid', scale=32, width=0.004,
    clim=(0, B0))
cbar = fig.colorbar(quiv, cax=cax)
cbar.set_label('|B| [T]')

# Outline of the solenoid coil (radius a, length L) for reference.
ax.plot(
    [-L / 2, L / 2, L / 2, -L / 2, -L / 2],
    [a, a, -a, -a, a],
    color='0.3', linestyle='--', linewidth=1, label=f'Solenoid (a={a} m)')

ax.set_ylabel('y [m]')
ax.set_title(f'FCC-ee main detector solenoid field, {B0:g} T, x=0 plane')
ax.set_ylim(-half_extent_y, half_extent_y)
ax.set_aspect('auto')
ax.legend(loc='upper right')
ax.tick_params(labelbottom=False)

# On-axis (x=0, y=0) longitudinal field Bz(s), aligned to the s-axis above.
s_line = np.linspace(-half_extent_s, half_extent_s, 400)
zero_line = np.zeros_like(s_line)
_, _, Bz_axis = field_model.get_field(zero_line, zero_line, s_line)

ax2.plot(s_line, Bz_axis, color='tab:blue', label=r'$B_z$ [T]')
ax2.axhline(B0, color='0.6', linestyle=':', linewidth=1, label=fr'$B_0 = {B0:g}$ T')
ax2.axvline(-L / 2, color='0.3', linestyle='--', linewidth=1)
ax2.axvline(L / 2, color='0.3', linestyle='--', linewidth=1, label='Solenoid ends')
ax2.set_xlabel('s [m]')
ax2.set_ylabel(r'$B_z$ [T]')
ax2.set_xlim(-half_extent_s, half_extent_s)
ax2.legend(loc='lower center', ncol=3, fontsize='small')

plt.show()
