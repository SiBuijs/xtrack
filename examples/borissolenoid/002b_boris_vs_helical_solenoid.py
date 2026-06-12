"""
Compare ``BorisSolenoid`` (helical exponential map) against the Python
``BorisSpatialIntegrator`` for increasing step counts in the analytical
solenoid field.

Uses the same solenoid setup as ``001_basic.py`` (delta = 0 only).
See also ``002a_boris_vs_helical_uniform.py`` for a uniform longitudinal field.
"""

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

plt.rcParams.update({"font.size": 14})

# ---------------------------------------------------------------
# Solenoid parameters (same as 001_basic.py)
# ---------------------------------------------------------------

L_coil = 4.0
a = 0.3
B0 = 1.5
z0 = 20.0
length = 30.0

sf = SolenoidField(L=L_coil, a=a, B0=B0, z0=z0)

p0 = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV,
    q0=1,
    energy0=45.6e9 / 1000,
    x=-1e-3,
    px=-1e-3,
    y=1e-3,
    delta=0.0,
)

n_steps_list = np.array([
    500, 1000, 2000, 5000, 10000, 20000, 40000, 80000, 160000, 320000,
], dtype=int)


def _scalar(value):
    """Particle quantity as float (works for scalar or length-1 array)."""
    return float(np.asarray(value).flat[0])


plot_coords = ("x", "y", "px", "py")
diffs = {name: np.zeros(n_steps_list.size) for name in plot_coords}
helical_end = {name: np.zeros(n_steps_list.size) for name in plot_coords}
boris_end = {name: np.zeros(n_steps_list.size) for name in plot_coords}

print("Helical (BorisSolenoid) vs BorisSpatialIntegrator, solenoid field, delta = 0")
print(f"{'n_steps':>8s}  {'ds [mm]':>8s}  {'|dx| [um]':>10s}  {'|dy| [um]':>10s}  "
      f"{'|dpx|':>10s}  {'|dpy|':>10s}")

for i, n_steps in enumerate(n_steps_list):
    ds = length / n_steps

    solenoid = xt.BorisSolenoid(
        L_coil=L_coil,
        a=a,
        B0=B0,
        z0=z0,
        length=length,
        n_steps=n_steps,
    )
    line = xt.Line(elements=[solenoid])
    line.build_tracker()

    p_helical = p0.copy()
    line.track(p_helical)

    boris = xt.BorisSpatialIntegrator(
        fieldmap_callable=sf.get_field,
        s_start=0,
        s_end=length,
        n_steps=n_steps,
    )
    p_boris = p0.copy()
    boris.track(p_boris)

    for name in plot_coords:
        helical_end[name][i] = _scalar(getattr(p_helical, name))
        boris_end[name][i] = _scalar(getattr(p_boris, name))
        diffs[name][i] = abs(helical_end[name][i] - boris_end[name][i])

    print(f"{n_steps:8d}  {ds * 1e3:8.3f}  {diffs['x'][i] * 1e6:10.2f}  "
          f"{diffs['y'][i] * 1e6:10.2f}  {diffs['px'][i]:10.3e}  {diffs['py'][i]:10.3e}")

# ---------------------------------------------------------------
# Difference vs n_steps
# ---------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
panel_info = (
    ("x", r"$|\Delta x|$ [µm]", 1e6),
    ("y", r"$|\Delta y|$ [µm]", 1e6),
    ("px", r"$|\Delta p_x|$", 1.0),
    ("py", r"$|\Delta p_y|$", 1.0),
)
for ax, (name, ylabel, scale) in zip(axes.flat, panel_info):
    ax.loglog(
        n_steps_list,
        np.maximum(diffs[name] * scale, 1e-30),
        "o-",
        linewidth=2,
    )
    ax.set_xlabel(r"$n_\mathrm{steps}$")
    ax.set_ylabel(ylabel)
    ax.set_title(rf"$|\Delta {name}|$, solenoid field")
    ax.grid(True, which="both", alpha=0.3)

plt.savefig("borissolenoid_boris_vs_helical_solenoid_diff.png", dpi=150)
plt.show()

# ---------------------------------------------------------------
# End states: helical vs Boris separately
# ---------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
end_panel_info = (
    ("x", r"$x$ [mm]", 1e3),
    ("y", r"$y$ [mm]", 1e3),
    ("px", r"$p_x$", 1.0),
    ("py", r"$p_y$", 1.0),
)
for ax, (name, ylabel, scale) in zip(axes.flat, end_panel_info):
    ax.semilogx(
        n_steps_list,
        helical_end[name] * scale,
        "o-",
        linewidth=2,
        label="Helical (BorisSolenoid)",
    )
    ax.semilogx(
        n_steps_list,
        boris_end[name] * scale,
        "s--",
        linewidth=2,
        label="Boris spatial",
    )
    ax.set_xlabel(r"$n_\mathrm{steps}$")
    ax.set_ylabel(ylabel)
    ax.set_title(rf"Final ${name}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

fig.suptitle(r"End state vs $n_\mathrm{steps}$, solenoid field ($\delta = 0$)")
plt.savefig("borissolenoid_boris_vs_helical_solenoid_endstate.png", dpi=150)
plt.show()
