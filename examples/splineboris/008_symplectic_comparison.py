import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c_light, e as qe, m_e


def gamma_from_momentum(P, m):
    return np.sqrt(1 + (P / (m * c_light))**2)


def on_shell_momentum(p, P):
    """Project momentum onto the relativistic mass shell |p| = P."""
    p = np.asarray(p, dtype=float)
    p_norm = np.linalg.norm(p)
    if p_norm == 0:
        return np.array([0.0, 0.0, P])
    return p * (P / p_norm)


def _transverse_kick_momentum(px, py, P):
    """Return (px, py, pz) on the mass shell after transverse kicks."""
    p_perp_sq = px**2 + py**2
    if p_perp_sq >= P**2:
        scale = (P * (1 - 1e-15)) / np.sqrt(p_perp_sq)
        px *= scale
        py *= scale
        p_perp_sq = px**2 + py**2
    return px, py, np.sqrt(P**2 - p_perp_sq)


def verlet_step(B, q, m, x_0, v_0, dt):
    x_0 = np.asarray(x_0, dtype=float)
    v_0 = np.asarray(v_0, dtype=float)

    abs_v = np.linalg.norm(v_0)
    beta_sq = min((abs_v / c_light)**2, 1 - 1e-30)
    gamma = 1 / np.sqrt(1 - beta_sq)
    p_0 = gamma * m * v_0
    P = np.linalg.norm(p_0)
    gamma = gamma_from_momentum(P, m)
    v_0 = p_0 / (gamma * m)

    a_0 = (q / (gamma * m)) * np.cross(v_0, B(x_0))

    x_h = x_0 + 0.5 * v_0 * dt
    v_h = v_0 + 0.5 * a_0 * dt

    a_h = (q / (gamma * m)) * np.cross(v_h, B(x_h))

    x_1 = x_0 + v_0 * dt + 0.5 * a_0 * dt**2
    v_1 = v_0 + a_h * dt

    p_1 = on_shell_momentum(gamma * m * v_1, P)
    v_1 = p_1 / (gamma * m)

    return x_1, v_1, p_1

def boris_step(B, q, m, x_0, v_0, dt):
    """
    Time-domain Boris step for magnetic fields only (E = 0),
    conserving relativistic momentum magnitude.

    Time translation of step_spatial_boris_B: spatial step dz is replaced
    by dz = vz * dt, and drifts use v * dt instead of (p/pz) * dz.

    Parameters
    ----------
    B : callable(x) -> (Bx, By, Bz) or ndarray
        Magnetic field [T] at position x [m].
    q : float
        Particle charge [C].
    m : float
        Particle rest mass [kg].
    x_0 : (3,) array
        Initial position [m].
    v_0 : (3,) array
        Initial velocity [m/s].
    dt : float
        Time step [s].

    Returns
    -------
    x_1, v_1, p_1 : (3,) arrays
        Updated position, velocity, and momentum [m, m/s, kg m/s].
    """
    x_0 = np.asarray(x_0, dtype=float)
    v_0 = np.asarray(v_0, dtype=float)

    abs_v = np.linalg.norm(v_0)
    beta_sq = min((abs_v / c_light)**2, 1 - 1e-30)
    gamma = 1 / np.sqrt(1 - beta_sq)
    px, py, pz = gamma * m * v_0
    P = np.sqrt(px**2 + py**2 + pz**2)
    gamma = gamma_from_momentum(P, m)

    vz = pz / (gamma * m)
    dz = vz * dt

    # --- Half drift
    x_h = x_0 + v_0 * dt * 0.5

    # --- Evaluate magnetic field at mid-step
    B_h = np.asarray(B(x_h))
    Bx, By, Bz = B_h[0], B_h[1], B_h[2]

    # ============================================
    # (1) FIRST HALF-KICK from (Bx, By)
    # ============================================
    pxm = px - 0.5 * q * dz * By
    pym = py + 0.5 * q * dz * Bx

    # --- Recompute pz to maintain |p| = P
    pxm, pym, pz = _transverse_kick_momentum(pxm, pym, P)

    # ============================================
    # (2) ROTATION due to Bz
    # ============================================
    pz_safe = max(pz, P * 1e-15)
    t = 0.5 * q * Bz * dz / pz_safe
    t2 = t * t
    s = 2.0 * t / (1.0 + t2)
    c0 = (1.0 - t2) / (1.0 + t2)

    pxp = c0 * pxm + s * pym
    pyp = -s * pxm + c0 * pym

    # ============================================
    # (3) SECOND HALF-KICK from (Bx, By)
    # ============================================
    px1 = pxp - 0.5 * q * dz * By
    py1 = pyp + 0.5 * q * dz * Bx

    # --- Recompute pz after full kick
    px1, py1, pz1 = _transverse_kick_momentum(px1, py1, P)

    # ============================================
    # (4) SECOND HALF-DRIFT
    # ============================================
    v_1 = np.array([px1, py1, pz1]) / (gamma * m)
    x_1 = x_h + v_1 * dt * 0.5
    p_1 = np.array([px1, py1, pz1])

    return x_1, v_1, p_1

def B(x, k_quad, k_sext, P_0, q):
    k_x = k_quad * x[1] + k_sext * x[0] * x[1]
    k_y = k_quad * x[0] + 1/2 * k_sext * ( x[0]**2 - x[1]**2)

    B_x = P_0 / q * k_x
    B_y = P_0 / q * k_y

    return np.array([B_x, B_y, 0])


def track_particle(step_fn, B_fn, q, m, x0, v0, dt, n_steps):
    """Integrate one particle and record x and px history."""
    x_hist = np.zeros(n_steps + 1)
    px_hist = np.zeros(n_steps + 1)

    x = np.asarray(x0, dtype=float)
    v = np.asarray(v0, dtype=float)
    gamma = 1 / np.sqrt(1 - (np.linalg.norm(v) / c_light)**2)
    p = gamma * m * v

    x_hist[0] = x[0]
    px_hist[0] = p[0]

    for i in range(n_steps):
        x, v, p = step_fn(B_fn, q, m, x, v, dt)
        x_hist[i + 1] = x[0]
        px_hist[i + 1] = p[0]

    return x_hist, px_hist


def compare_symplectic_integrators():
    """
    Track an electron through a nonlinear quadrupole field and compare
    Verlet vs Boris phase-space trajectories (p_x vs x) for several
    initial transverse momenta.
    """
    n_steps = 100_000

    q = qe
    m = m_e
    B_quad = 10   # T/m
    B_sext = 450   # T/m^2

    x0 = np.array([1.0e-6, 0.5e-6, 0.0])
    vz = 0.99 * c_light
    v0_ref = np.array([0.001 * vz, 0.0, vz])
    gamma0 = 1 / np.sqrt(1 - (np.linalg.norm(v0_ref) / c_light)**2)
    P = gamma0 * m * np.linalg.norm(v0_ref)

    B_fn = lambda x: B(x, B_quad, B_sext, P, q)

    cdt = 0.05
    dt = cdt / c_light

    px0_values = np.array([0.03, 0.0425, 0.052]) * P
    to_mev_c = lambda px: px * c_light / qe / 1e6
    cmap = plt.cm.viridis

    fig, (ax_verlet, ax_boris) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, px0 in enumerate(px0_values):
        pz0 = np.sqrt(P**2 - px0**2)
        v0 = np.array([px0 / (gamma0 * m), 0.0, pz0 / (gamma0 * m)])
        color = cmap(i / (len(px0_values) - 1))

        x_verlet, px_verlet = track_particle(
            verlet_step, B_fn, q, m, x0, v0, dt, n_steps)
        x_boris, px_boris = track_particle(
            boris_step, B_fn, q, m, x0, v0, dt, n_steps)

        ax_verlet.plot(
            x_verlet * 1e3, to_mev_c(px_verlet),
            linestyle='none', marker='.', markersize=1, alpha=0.6,
            color=color)
        ax_boris.plot(
            x_boris * 1e3, to_mev_c(px_boris),
            linestyle='none', marker='.', markersize=1, alpha=0.6,
            color=color)

    for ax, name in zip((ax_verlet, ax_boris), ('Verlet', 'Boris')):
        ax.set_xlabel(r'$x$ [mm]')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    ax_verlet.set_ylabel(r'$p_x$ [MeV/$c$]')
    fig.suptitle(
        f'Symplectic comparison: $p_x$ vs $x$ ({n_steps} steps, '
        f'{len(px0_values)} initial $p_x$)',
        y=1.02)
    fig.tight_layout()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        to_mev_c(px0_values[0]), to_mev_c(px0_values[-1])))
    sm.set_array([])
    pos = ax_boris.get_position()
    cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.012, pos.height])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r'initial $p_x$ [MeV/$c$]')
    plt.show()


if __name__ == '__main__':
    compare_symplectic_integrators()
