import numpy as np
import xtrack as xt
import matplotlib.pyplot as plt
from scipy.constants import c as clight
from scipy.constants import e as qe


def make_wiggler_field(g, B_r, length, n_periods):
    """
    Build a magnetic field callable compatible with `xt.BorisSpatialIntegrator`.

    The image formulas are:
        Bz(s, z) = B_r / cosh(pi g / lambda_u) * cosh(k_u z) * cos(k_u s)
        Bs(s, z) = B_r / cosh(pi g / lambda_u) * sinh(k_u z) * sin(k_u s)

    Here we use coordinates (x, y, s), mapping:
      - image z -> y
      - image Bz -> By
      - image Bs -> Bs
    and we set Bx = 0.
    """
    lambda_u = length / n_periods
    k_u = 2.0 * np.pi / lambda_u
    scale = B_r / np.cosh(np.pi * g / lambda_u)

    def field(x, y, s):
        x = np.asarray(x)
        y = np.asarray(y)
        s = np.asarray(s)
        Bx = np.zeros_like(x)
        By = scale * np.cosh(k_u * y) * np.cos(k_u * s)
        Bs = scale * np.sinh(k_u * y) * np.sin(k_u * s)
        return Bx, By, Bs

    return field


def track_spin_tbmt_from_orbit(s, x, y, field, particle, sx0=0.0, sy0=1.0, sz0=0.0):
    """
    Post-process spin along a tracked orbit with the T-BMT equation in B fields.
    """
    s = np.asarray(s, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx_ds = np.gradient(x, s, edge_order=2)
    dy_ds = np.gradient(y, s, edge_order=2)

    tangent = np.stack([dx_ds, dy_ds, np.ones_like(s)], axis=1)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]

    bx, by, bs = field(x, y, s)
    bvec = np.stack([bx, by, bs], axis=1)

    gamma_rel = float(particle.energy[0] / particle.mass0)
    beta_rel = np.sqrt(1.0 - 1.0 / gamma_rel**2)
    q_over_m = float(particle.charge[0] * qe) / float(particle.mass0 * qe / clight**2)
    a = float(particle.anomalous_magnetic_moment[0])

    spin = np.zeros((len(s), 3), dtype=float)
    spin[0, :] = [sx0, sy0, sz0]

    for ii in range(len(s) - 1):
        ds_step = s[ii + 1] - s[ii]
        n_hat = tangent[ii]
        beta_vec = beta_rel * n_hat

        b_here = bvec[ii]
        beta_dot_b = np.dot(beta_vec, b_here)

        omega = -q_over_m * (
            (a + 1.0 / gamma_rel) * b_here
            - (a * gamma_rel / (gamma_rel + 1.0)) * beta_dot_b * beta_vec
        )

        v_s = beta_rel * clight * n_hat[2]
        omega_per_m = omega / v_s

        theta = np.linalg.norm(omega_per_m) * ds_step
        if theta > 0.0:
            axis = omega_per_m / np.linalg.norm(omega_per_m)
            s_old = spin[ii]
            spin[ii + 1] = (
                s_old * np.cos(theta)
                + np.cross(axis, s_old) * np.sin(theta)
                + axis * np.dot(axis, s_old) * (1.0 - np.cos(theta))
            )
        else:
            spin[ii + 1] = spin[ii]

    return spin[:, 0], spin[:, 1], spin[:, 2]


if __name__ == "__main__":
    # Field parameters
    g = 12e-3
    B_r = 1.2
    length = 2.0
    n_periods = 100
    field = make_wiggler_field(g=g, B_r=B_r, length=length, n_periods=n_periods)

    # One BorisSpatial element over the full insertion length
    boris = xt.BorisSpatialIntegrator(
        fieldmap_callable=field,
        s_start=0.0,
        s_end=length,
        n_steps=4000,
    )

    line = xt.Line(elements=[boris], element_names=["undulator_boris"])
    line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)
    line.particle_ref.anomalous_magnetic_moment = 0.00115965218128

    # Simple single-particle track
    p = line.particle_ref.copy()
    p.x = 0.0
    p.px = 0#1e-4
    p.y = 0.5e-3
    p.py = 0.0
    boris.log_trajectories = True
    boris.track(p)

    # Simple Twiss through the Boris element
    tw = None
    try:
        # This can fail on local setups where collective helper kernels are not
        # available in prebuilt form. Keep plotting functional regardless.
        line.build_tracker(use_prebuilt_kernels=False)
        tw = line.twiss(
            betx=10.0,
            bety=10.0,
            include_collective=False,
        )
    except Exception as exc:
        print("\nTwiss skipped on this setup:")
        print(f"  {type(exc).__name__}: {exc}")

    print("Track end coordinates:")
    print(f"  x  = {p.x[0]:+.6e} m")
    print(f"  px = {p.px[0]:+.6e}")
    print(f"  y  = {p.y[0]:+.6e} m")
    print(f"  py = {p.py[0]:+.6e}")
    print(f"  zeta = {p.zeta[0]:+.6e} m")
    print(f"  delta = {p.delta[0]:+.6e}")

    if tw is not None:
        print("\nTwiss at line end:")
        print(f"  x   = {tw.x[-1]:+.6e} m")
        print(f"  y   = {tw.y[-1]:+.6e} m")
        print(f"  betx = {tw.betx[-1]:+.6e} m")
        print(f"  bety = {tw.bety[-1]:+.6e} m")

    # Track plot from Boris internal trajectory log.
    s_log = np.array(boris.z_log)[:, 0]
    x_log = np.array(boris.x_log)[:, 0]
    y_log = np.array(boris.y_log)[:, 0]
    _, by_log, bs_log = field(x_log, y_log, s_log)
    spin_x_log, spin_y_log, spin_z_log = track_spin_tbmt_from_orbit(
        s=s_log, x=x_log, y=y_log, field=field, particle=p,
        sx0=0.0, sy0=1.0, sz0=0.0,
    )

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

    fig_spin, (ax_sx, ax_sy, ax_sz) = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True
    )
    ax_sx.plot(s_log, spin_x_log, color="C4", label="s_x")
    ax_sy.plot(s_log, 1.0 - spin_y_log, color="C5", label="1 - s_y")
    ax_sz.plot(s_log, spin_z_log, color="C6", label="s_z")
    ax_sx.set_ylabel("s_x")
    ax_sy.set_ylabel("1 - s_y")
    ax_sz.set_ylabel("s_z")
    ax_sz.set_xlabel("s [m]")
    for ax in (ax_sx, ax_sy, ax_sz):
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig_spin.suptitle("Spin tracking (T-BMT), initial vertical polarization")
    fig_spin.tight_layout()

    if tw is not None:
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
