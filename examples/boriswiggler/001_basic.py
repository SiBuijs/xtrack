"""
Minimal introduction to the `BorisWiggler` element.

We build a single-period wiggler with the analytic field model from
equations (8.9)-(8.11) and show:

- how to specify length, gap, peak field, and period count,
- basic tracking through the element,
- field evaluation with `BorisWiggler.get_field(...)`.
"""

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt


# ---------------------------------------------------------------
# BorisWiggler definition
# ---------------------------------------------------------------

length = 2.0       # [m] total element length
n_periods = 4      # number of wiggler periods
lambda_u = length / n_periods
g = 0.02           # [m] magnet gap
B_r = 1.0          # [T] peak field parameter B_0

wiggler = xt.BorisWiggler(
    length=length,
    g=g,
    B_r=B_r,
    n_periods=n_periods,
)

print("Wiggler parameters:")
print(f"  length    = {wiggler.length:.4f} m")
print(f"  lambda_u  = {lambda_u:.4f} m")
print(f"  n_periods = {wiggler.n_periods}")
print(f"  k_u       = {wiggler.k_u:.4f} 1/m")
print(f"  b_tilde   = {wiggler.b_tilde:.4f} T")
print(f"  n_steps   = {wiggler.n_steps}")


# ---------------------------------------------------------------
# Basic tracking example
# ---------------------------------------------------------------

line = xt.Line(elements=[wiggler])
line.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV,
    q0=1.0,
    energy0=1e9,
)

part = line.particle_ref.copy()
part.x = 0.0
part.y = 1e-3   # 1 mm vertical offset
part.px = 1e-3
part.py = 0.0

line.build_tracker(use_prebuilt_kernels=False)
line.track(part)

print("\nFinal coordinates after BorisWiggler element:")
print(f"  x  = {part.x[0]:+.6e} m")
print(f"  y  = {part.y[0]:+.6e} m")
print(f"  px = {part.kin_px[0]:+.6e}")
print(f"  py = {part.kin_py[0]:+.6e}")


# ---------------------------------------------------------------
# Field evaluation
# ---------------------------------------------------------------

"""Plot B(s) along the axis and By(y) at s = lambda_u / 4."""

s_grid = np.linspace(0.0, wiggler.length, 300)
Bx_s, By_s, Bs_s = wiggler.get_field(0.0, 0.0, s_grid)

s_plot = 0.25 * lambda_u
y_grid = np.linspace(-3e-3, 3e-3, 200)
_, By_y, Bs_y = wiggler.get_field(0.0, y_grid, s_plot)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(s_grid, By_s, label="By(s)")
ax.plot(s_grid, Bs_s, label="Bs(s)")
ax.set_xlabel("s_local [m]")
ax.set_ylabel("B [T]")
ax.set_title("Field along s at x = y = 0")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
ax.plot(y_grid * 1e3, By_y, label="By(y)")
ax.plot(y_grid * 1e3, Bs_y, label="Bs(y)")
ax.set_xlabel("y [mm]")
ax.set_ylabel("B [T]")
ax.set_title(f"Field vs y at s_local = {s_plot:.4f} m")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
plt.show()
