import numpy as np
import xtrack as xt
import matplotlib.pyplot as plt


def construct_wiggler_field(g, B_r, length, n_periods, pole_width):
    """
    Build a magnetic field callable for `xt.BorisSpatialIntegrator`.

    The model uses a pole-width-related transverse wavenumber and satisfies
    k_y^2 = k_x^2 + k_s^2:

        lambda_u = length / n_periods
        k_s = 2*pi / lambda_u
        k_x = 2*pi / w_p
        k_y^2 = k_x^2 + k_s^2

        B_x = -B_0 * (k_x / k_y) * sin(k_x x) * sinh(k_y y) * sin(k_s s)
        B_y =  B_0 *             cos(k_x x) * cosh(k_y y) * sin(k_s s)
        B_s =  B_0 * (k_s / k_y) * cos(k_x x) * sinh(k_y y) * cos(k_s s)

    with

        B_0 = B_r / cosh(pi * g / lambda_u)
    """
    lambda_u = length / n_periods
    k_s = 2.0 * np.pi / lambda_u

    k_x = 2.0 * np.pi / pole_width
    k_y = np.sqrt(k_x**2 + k_s**2)

    B_0 = B_r / np.cosh(np.pi * g / lambda_u)

    def wiggler_field_callable(x, y, s):
        x = np.asarray(x)
        y = np.asarray(y)
        s = np.asarray(s)

        Bx = (
            -B_0
            * (k_x / k_y)
            * np.sin(k_x * x)
            * np.sinh(k_y * y)
            * np.sin(k_s * s)
        )
        By = B_0 * np.cos(k_x * x) * np.cosh(k_y * y) * np.sin(k_s * s)
        Bs = (
            B_0
            * (k_s / k_y)
            * np.cos(k_x * x)
            * np.sinh(k_y * y)
            * np.cos(k_s * s)
        )
        return Bx, By, Bs

    return wiggler_field_callable


# Field parameters
g = 12e-3
B_r = 1.2
length = 2.0
n_periods = 100
pole_width = 5e-2
wiggler_field = construct_wiggler_field(
    g=g,
    B_r=B_r,
    length=length,
    n_periods=n_periods,
    pole_width=pole_width,
)

# One BorisSpatial element over the full insertion length
boris = xt.BorisSpatialIntegrator(
    fieldmap_callable=wiggler_field,
    s_start=0.0,
    s_end=length,
    n_steps=4000,
)

line = xt.Line(elements=[boris], element_names=["undulator_boris"])
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)

# Simple single-particle track
p = line.particle_ref.copy()
p.x = 0
p.px = 1.0e-3
p.y = 1.0e-4
p.py = 0.0
boris.log_trajectories = True
boris.track(p)

# Simple Twiss through the Boris element
line.build_tracker(use_prebuilt_kernels=False)
tw = line.twiss(
    betx=10.0,
    bety=10.0,
    include_collective=False,
)

print("Track end coordinates:")
print(f"  x  = {p.x[0]:+.6e} m")
print(f"  px = {p.px[0]:+.6e}")
print(f"  y  = {p.y[0]:+.6e} m")
print(f"  py = {p.py[0]:+.6e}")
print(f"  zeta = {p.zeta[0]:+.6e} m")
print(f"  delta = {p.delta[0]:+.6e}")

print("\nTwiss at line end:")
print(f"  x   = {tw.x[-1]:+.6e} m")
print(f"  y   = {tw.y[-1]:+.6e} m")
print(f"  betx = {tw.betx[-1]:+.6e} m")
print(f"  bety = {tw.bety[-1]:+.6e} m")

# Track plot from Boris internal trajectory log.
s_log = np.array(boris.z_log)[:, 0]
x_log = np.array(boris.x_log)[:, 0]
y_log = np.array(boris.y_log)[:, 0]
_, by_log, bs_log = wiggler_field(x_log, y_log, s_log)

plt.close("all")
fig_track, (ax_x, ax_y, ax_by, ax_bs) = plt.subplots(
    4, 1, figsize=(10, 9), sharex=True
)
ax_x.plot(s_log, x_log, label="x(s)", color="C0")
ax_y.plot(s_log, y_log, label="y(s)", color="C1")
ax_by.plot(s_log, by_log, label="By(s) seen by particle", color="C2")
ax_bs.plot(s_log, bs_log, label="Bs(s) seen by particle", color="C3")
ax_x.set_ylabel("x [m]")
ax_y.set_ylabel("y [m]")
ax_by.set_ylabel("By [T]")
ax_bs.set_ylabel("Bs [T]")
ax_bs.set_xlabel("s [m]")
for ax in (ax_x, ax_y, ax_by, ax_bs):
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
fig_track.suptitle("BorisSpatial tracking in wiggler field")
fig_track.tight_layout()

# Twiss plot along the line.
fig_twiss, ax_twiss = plt.subplots(figsize=(10, 4))
ax_twiss.plot(tw.s, tw.betx, label="betx", color="C0")
ax_twiss.plot(tw.s, tw.bety, label="bety", color="C1")
ax_twiss.set_xlabel("s [m]")
ax_twiss.set_ylabel("beta [m]")
ax_twiss.set_title("Twiss through BorisSpatial wiggler")
ax_twiss.grid(True, alpha=0.3)
ax_twiss.legend(loc="best")
fig_twiss.tight_layout()

plt.show()
