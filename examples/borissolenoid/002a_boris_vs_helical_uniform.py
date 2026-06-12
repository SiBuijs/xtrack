"""
Compare helical exponential-map stepping against ``BorisSpatialIntegrator``
for increasing step counts in a uniform magnetic field.

Mirrors the layout of ``002b_boris_vs_helical_solenoid.py`` (delta = 0 only).
The helical integrator follows the same step logic as ``track_borissolenoid.h``
(Python reference).

Set ``Bx`` (or ``By``) to a small nonzero value to test tilted-field stepping.
With ``Bx = By = 0`` the helical map is exact. When ``B`` has a transverse
component, the step length ``h`` is found by Newton iteration so the particle
lands on the next longitudinal plane (same fix as ``track_borissolenoid.h``).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as clight
from scipy.constants import e as qe

import xtrack as xt

plt.rcParams.update({"font.size": 14})

# ---------------------------------------------------------------
# Uniform field and tracking parameters
# ---------------------------------------------------------------

Bz = 1.5          # uniform longitudinal field [T]
Bx = 0.0          # uniform transverse tilt [T]; try 0.05 to test tilted-B convergence
By = 0.0
length = 30.0     # integration extent along s [m]

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


def _sinc(theta):
    if abs(theta) < 1e-14:
        return 1.0
    return np.sin(theta) / theta


def _vers_over_theta(theta):
    if abs(theta) < 1e-14:
        return 0.0
    return (1.0 - np.cos(theta)) / theta


def _cos_minus_one_over_theta(theta):
    if abs(theta) < 1e-14:
        return 0.0
    return (np.cos(theta) - 1.0) / theta


def _vec_to_zeta(Bx_f, By_f, Bz_f, B_mag, B_perp, vx, vy, vz):
    if B_mag < 1e-30:
        return vx, vy, vz
    if B_perp < 1e-30:
        p_par = Bx_f * vx + By_f * vy + Bz_f * vz
        return vx, vy, p_par / B_mag
    inv_B_mag = 1.0 / B_mag
    inv_B_perp = 1.0 / B_perp
    Bz_over_B = Bz_f * inv_B_mag
    Bz_over_B_perp = Bz_over_B * inv_B_perp
    r00, r01 = -By_f * inv_B_perp, Bx_f * inv_B_perp
    r10 = -Bx_f * Bz_over_B_perp
    r11 = -By_f * Bz_over_B_perp
    r12 = B_perp * inv_B_mag
    r20, r21, r22 = Bx_f * inv_B_mag, By_f * inv_B_mag, Bz_f * inv_B_mag
    ox = r00 * vx + r01 * vy
    oy = r10 * vx + r11 * vy + r12 * vz
    oz = r20 * vx + r21 * vy + r22 * vz
    return ox, oy, oz


def _vec_to_lab(Bx_f, By_f, Bz_f, B_mag, B_perp, vx, vy, vz):
    if B_mag < 1e-30:
        return vx, vy, vz
    if B_perp < 1e-30:
        return vx, vy, Bz_f * vz / B_mag
    inv_B_mag = 1.0 / B_mag
    inv_B_perp = 1.0 / B_perp
    Bz_over_B = Bz_f * inv_B_mag
    Bz_over_B_perp = Bz_over_B * inv_B_perp
    r00, r10, r20 = -By_f * inv_B_perp, -Bx_f * Bz_over_B_perp, Bx_f * inv_B_mag
    r01, r11, r21 = Bx_f * inv_B_perp, -By_f * Bz_over_B_perp, By_f * inv_B_mag
    r02, r12, r22 = 0.0, B_perp * inv_B_mag, Bz_f * inv_B_mag
    ox = r00 * vx + r10 * vy + r20 * vz
    oy = r01 * vx + r11 * vy + r21 * vz
    oz = r02 * vx + r12 * vy + r22 * vz
    return ox, oy, oz


def _helical_F_step(x, y, px, py, B_mag, P_z, q_coulomb, h):
    """Pure helical map in the B-aligned frame (same as helical_map.h)."""
    theta = q_coulomb * B_mag * h / P_z
    ct, st = np.cos(theta), np.sin(theta)
    h_over_Pz = h / P_z
    x_out = x + h_over_Pz * _sinc(theta) * px + h_over_Pz * _vers_over_theta(theta) * py
    y_out = y + h_over_Pz * _cos_minus_one_over_theta(theta) * px + h_over_Pz * _sinc(theta) * py
    px_out = ct * px + st * py
    py_out = -st * px + ct * py
    return x_out, y_out, px_out, py_out


def _helical_step_z_lab(x_z, y_z, z_z, px_z, py_z, Bx_f, By_f, Bz_f, B_mag, B_perp, P_z, q_coulomb, h):
    xz, yz, pxz, pyz = _helical_F_step(x_z, y_z, px_z, py_z, B_mag, P_z, q_coulomb, h)
    _, _, z_lab = _vec_to_lab(Bx_f, By_f, Bz_f, B_mag, B_perp, xz, yz, z_z + h)
    return z_lab


def _helical_step_dz_lab_dh(x_z, y_z, z_z, px_z, py_z, Bx_f, By_f, Bz_f, B_mag, B_perp, P_z, q_coulomb, h):
    if B_perp < 1e-30:
        return Bz_f / B_mag
    r12 = B_perp / B_mag
    r22 = Bz_f / B_mag
    theta = q_coulomb * B_mag * h / P_z
    dtheta_dh = q_coulomb * B_mag / P_z
    sinc_t = _sinc(theta)
    vers_t = _vers_over_theta(theta)
    cosm1_t = _cos_minus_one_over_theta(theta)
    if abs(theta) < 1e-14:
        dsinc_dh = dvers_dh = dcosm1_dh = 0.0
    else:
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        dsinc_dh = (theta * cos_t - sin_t) / theta**2 * dtheta_dh
        dvers_dh = (sin_t * theta - (1.0 - cos_t)) / theta**2 * dtheta_dh
        dcosm1_dh = (-sin_t * theta - (cos_t - 1.0)) / theta**2 * dtheta_dh
    inv_Pz = 1.0 / P_z
    Bcoef = cosm1_t * px_z + sinc_t * py_z
    dBcoef_dh = px_z * dcosm1_dh + py_z * dsinc_dh
    dy_dh = Bcoef * inv_Pz + (h * inv_Pz) * dBcoef_dh
    return r12 * dy_dh + r22


def _solve_helical_h_for_z_plane(
    x_z, y_z, z_z, px_z, py_z, Bx_f, By_f, Bz_f, B_mag, B_perp, P_z, q_coulomb, z_target, h_init,
):
    if B_perp < 1e-30:
        return h_init
    h = h_init
    for _ in range(3):
        z_lab = _helical_step_z_lab(
            x_z, y_z, z_z, px_z, py_z, Bx_f, By_f, Bz_f, B_mag, B_perp, P_z, q_coulomb, h
        )
        f = z_lab - z_target
        if abs(f) < 1e-15:
            break
        dz_dh = _helical_step_dz_lab_dh(
            x_z, y_z, z_z, px_z, py_z, Bx_f, By_f, Bz_f, B_mag, B_perp, P_z, q_coulomb, h
        )
        if abs(dz_dh) < 1e-30:
            break
        h -= f / dz_dh
    return h


def track_helical_uniform(p, n_steps, length, Bx_f, By_f, Bz_f):
    """Python mirror of track_borissolenoid.h for uniform B = (Bx, By, Bz)."""
    ds = length / n_steps
    half_ds = 0.5 * ds

    mass0 = _scalar(p.mass0)
    p0c = _scalar(p.p0c)
    beta0 = _scalar(p.beta0)
    energy0 = _scalar(p.energy0)
    q0 = _scalar(p.q0)
    charge_ratio = _scalar(p.charge_ratio)
    delta = _scalar(p.delta)

    P0 = p0c * qe / clight
    px = _scalar(p.px) * P0
    py = _scalar(p.py) * P0
    x = _scalar(p.x)
    y = _scalar(p.y)
    zeta = _scalar(p.zeta)

    q_coulomb = q0 * charge_ratio * qe
    mass_kg = mass0 * qe / clight**2
    P = P0 * (1.0 + delta)
    gamma = energy0 * (1.0 + delta) / mass0

    B_mag = np.sqrt(Bx_f**2 + By_f**2 + Bz_f**2)
    B_perp = np.sqrt(Bx_f**2 + By_f**2)
    s_local = 0.0

    for _ in range(n_steps):
        ps = np.sqrt(max(P**2 - px**2 - py**2, 0.0))
        if ps == 0.0:
            break
        inv_ps = 1.0 / ps

        # Half-drift predictor (field is constant; position unused for B).
        _ = x + px * inv_ps * half_ds
        _ = y + py * inv_ps * half_ds

        if B_mag < 1e-30 or abs(Bz_f) < 1e-30:
            x += px * inv_ps * ds
            y += py * inv_ps * ds
            dt = ds / ps * gamma * mass_kg
        else:
            P_z = (Bx_f * px + By_f * py + Bz_f * ps) / B_mag
            if abs(P_z) < 1e-30:
                break

            s_current = s_local
            z_target = s_current + ds
            h_init = ds * B_mag / abs(Bz_f)

            x_z, y_z, z_z = _vec_to_zeta(Bx_f, By_f, Bz_f, B_mag, B_perp, x, y, s_current)
            px_z, py_z, _ = _vec_to_zeta(Bx_f, By_f, Bz_f, B_mag, B_perp, px, py, ps)

            h = _solve_helical_h_for_z_plane(
                x_z, y_z, z_z, px_z, py_z,
                Bx_f, By_f, Bz_f, B_mag, B_perp, P_z, q_coulomb, z_target, h_init,
            )

            x_z, y_z, px_z, py_z = _helical_F_step(
                x_z, y_z, px_z, py_z, B_mag, P_z, q_coulomb, h
            )
            z_z += h

            x, y, _ = _vec_to_lab(Bx_f, By_f, Bz_f, B_mag, B_perp, x_z, y_z, z_z)
            px, py, _ = _vec_to_lab(
                Bx_f, By_f, Bz_f, B_mag, B_perp, px_z, py_z, P_z
            )
            dt = h * gamma * mass_kg / P_z

        ps = np.sqrt(max(P**2 - px**2 - py**2, 0.0))
        if ps == 0.0:
            break
        s_local += ds
        zeta += ds - dt * clight * beta0

    return {
        "x": x,
        "y": y,
        "px": px / P0,
        "py": py / P0,
        "zeta": zeta,
    }


def uniform_field(x, y, z):
    return Bx * np.ones_like(x), By * np.ones_like(x), Bz * np.ones_like(x)


def _field_label():
    if abs(Bx) < 1e-30 and abs(By) < 1e-30:
        return rf"uniform $B_z = {Bz}$ T"
    theta_deg = np.degrees(np.arctan2(np.hypot(Bx, By), abs(Bz)))
    return rf"uniform $\mathbf{{B}} = ({Bx:.3g}, {By:.3g}, {Bz})$ T ($\theta_B \approx {theta_deg:.2f}^\circ$)"


_field_tag = "tilted" if abs(Bx) > 1e-30 or abs(By) > 1e-30 else "uniform"


plot_coords = ("x", "y", "px", "py")
diffs = {name: np.zeros(n_steps_list.size) for name in plot_coords}
helical_end = {name: np.zeros(n_steps_list.size) for name in plot_coords}
boris_end = {name: np.zeros(n_steps_list.size) for name in plot_coords}

print(f"Helical vs BorisSpatialIntegrator, {_field_label()}, delta = 0")
print(f"{'n_steps':>8s}  {'ds [mm]':>8s}  {'|dx| [um]':>10s}  {'|dy| [um]':>10s}  "
      f"{'|dpx|':>10s}  {'|dpy|':>10s}")

for i, n_steps in enumerate(n_steps_list):
    ds = length / n_steps

    helical = track_helical_uniform(p0, n_steps, length, Bx, By, Bz)

    boris = xt.BorisSpatialIntegrator(
        fieldmap_callable=uniform_field,
        s_start=0,
        s_end=length,
        n_steps=n_steps,
    )
    p_boris = p0.copy()
    boris.track(p_boris)

    for name in plot_coords:
        helical_end[name][i] = helical[name]
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
    ax.set_title(rf"$|\Delta {name}|$, {_field_label()}")
    ax.grid(True, which="both", alpha=0.3)

plt.savefig(f"borissolenoid_boris_vs_helical_{_field_tag}_diff.png", dpi=150)
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
        label="Helical",
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
    ax.set_title(rf"Final ${name}$" if name in ("x", "y") else rf"Final ${name}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

fig.suptitle(rf"End state vs $n_\mathrm{{steps}}$, {_field_label()} ($\delta = 0$)")
plt.savefig(f"borissolenoid_boris_vs_helical_{_field_tag}_endstate.png", dpi=150)
plt.show()
