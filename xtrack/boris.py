import numpy as np

from scipy.constants import c as clight
from scipy.constants import e as qe

class BorisSpatialIntegrator:

    isthick = True
    _step_fn = None  # set below, after step_spatial_boris_B is defined

    def __init__(self, fieldmap_callable, s_start, s_end, n_steps,
                 vector_potential_callable=None, verbose=False):
        self.fieldmap_callable = fieldmap_callable
        self.vector_potential_callable = vector_potential_callable
        self.s_start = s_start
        self.s_end = s_end
        self.ds = (s_end - s_start) / n_steps
        self.n_steps = n_steps
        self.verbose = verbose
        self.length = s_end - s_start
        self.log_trajectories = False

    def _shift_canonical_kinetic(self, p, mask_alive, sign):
        # canonical_p = kinetic_p + q*A/P0 (xtrack convention, see
        # track_magnet.template.h: kin_px = px - ax). sign=-1 converts
        # canonical (as stored in p.px, p.py) to kinetic; sign=+1 converts back.
        Ax, Ay, _ = self.vector_potential_callable(
            p.x[mask_alive], p.y[mask_alive], p.s[mask_alive])
        charge_coulomb = p.charge[mask_alive] * qe
        P0 = p.p0c[mask_alive] * qe / clight
        p.px[mask_alive] += sign * charge_coulomb * Ax / P0
        p.py[mask_alive] += sign * charge_coulomb * Ay / P0

    def track(self, p):

        mask_alive = p.state > 0

        if self.log_trajectories:
            x_log = []
            y_log = []
            z_log = []

        s_in = p.s[mask_alive].copy()
        p.s[mask_alive] = self.s_start

        if self.vector_potential_callable is not None:
            self._shift_canonical_kinetic(p, mask_alive, sign=-1)

        for ii in range(self.n_steps):

            if self.verbose:
                print(f's_in = {s_in[0]:.3f} s_in_map = {p.s[0]:.3f}', end='\r', flush=True)

            x = p.x[mask_alive].copy()
            y = p.y[mask_alive].copy()
            z = p.s[mask_alive].copy()
            px = p.px[mask_alive].copy()
            py = p.py[mask_alive].copy()
            delta = p.delta[mask_alive].copy()
            energy = p.energy[mask_alive].copy()
            p0c = p.p0c[mask_alive].copy()
            beta0 = p.beta0[mask_alive].copy()
            charge = p.charge[mask_alive].copy()
            mass = p.mass[mask_alive].copy()

            charge_coulomb = charge * qe
            mass_kg = mass * qe / clight**2
            gamma = energy / mass

            P0 = p0c * qe / clight # in kg m/s
            P = P0 * (1 + delta)

            Px = px * P0
            Py = py * P0

            w = np.zeros((Px.shape[0], 3), dtype=Px.dtype)
            w[:, 0] = Px
            w[:, 1] = Py
            w[:, 2] = P

            x_new, y_new, z_new, w_new, dt = self._step_fn(x, y, z, w,
                charge_coulomb, self.ds,
                field_fn=self.fieldmap_callable, m_kg=mass_kg, gamma=gamma)
            p.x[mask_alive] = x_new
            p.y[mask_alive] = y_new
            p.s[mask_alive] = z_new
            p.px[mask_alive] = w_new[:, 0] / P0
            p.py[mask_alive] = w_new[:, 1] / P0
            p.zeta[mask_alive] += (self.ds - dt * clight * beta0)

            if self.log_trajectories:
                x_log.append(p.x.copy())
                y_log.append(p.y.copy())
                z_log.append(p.s.copy())

        if self.vector_potential_callable is not None:
            self._shift_canonical_kinetic(p, mask_alive, sign=+1)

        p.s[mask_alive] = s_in + self.length

        if self.log_trajectories:
            self.x_log = np.array(x_log)
            self.y_log = np.array(y_log)
            self.z_log = np.array(z_log)


import numpy as np
c = 299_792_458.0  # m/s

def step_spatial_boris_B(x, y, z, w, q, dz, field_fn, m_kg, gamma):
    """
    Spatial Boris step for magnetic fields only (E = 0),
    using constant total momentum magnitude P = w[:,2].

    Parameters
    ----------
    x, y, z : (N,) arrays
        Particle positions [m].
    w : (N, 3) array
        Momentum-like state: (px, py, P), where
        P = |p| = sqrt(px^2 + py^2 + pz^2) is constant.
    q : float
        Particle charge [C].
    dz : float
        Spatial step along z [m].
    field_fn : callable(x, y, z) -> (Bx, By, Bz)
        Function returning magnetic-field components [T].

    Returns
    -------
    x1, y1, z1 : (N,) arrays
        Updated positions [m].
    w1 : (N, 3) array
        Updated momentum state (px, py, P).
    """

    dt = 0.

    # --- Unpack and ensure arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    w = np.asarray(w)
    px, py, P = w[:, 0], w[:, 1], w[:, 2]

    # --- Compute longitudinal momentum from constant magnitude
    pz = np.sqrt(np.maximum(P**2 - px**2 - py**2, 0.0))

    # --- Half drift in (x, y)
    xh = x + (px / pz) * (dz * 0.5)
    yh = y + (py / pz) * (dz * 0.5)
    zh = z + dz * 0.5
    dt += (dz * 0.5) / pz * gamma * m_kg #  s

    # --- Evaluate magnetic field at mid-step
    Bx, By, Bz = field_fn(xh, yh, zh)

    # ============================================
    # (1) FIRST HALF-KICK from (Bx, By)
    # ============================================
    pxm = px - 0.5 * q * dz * By
    pym = py + 0.5 * q * dz * Bx

    # --- Recompute pz to maintain |p| = P
    pz = np.sqrt(np.maximum(P**2 - pxm**2 - pym**2, 0.0))

    # ============================================
    # (2) ROTATION due to Bz
    # ============================================
    t = 0.5 * q * Bz * dz / pz
    t2 = t * t
    s = 2.0 * t / (1.0 + t2)
    c0 = (1.0 - t2) / (1.0 + t2)

    # pxp = c0 * pxm - s * pym
    # pyp = s * pxm + c0 * pym
    pxp = c0 * pxm + s * pym
    pyp = -s * pxm + c0 * pym

    # ============================================
    # (3) SECOND HALF-KICK from (Bx, By)
    # ============================================
    px1 = pxp - 0.5 * q * dz * By
    py1 = pyp + 0.5 * q * dz * Bx

    # --- Recompute pz after full step
    pz1 = np.sqrt(np.maximum(P**2 - px1**2 - py1**2, 0.0))

    # ============================================
    # (4) SECOND HALF-DRIFT (x, y)
    # ============================================
    x1 = xh + (px1 / pz1) * (dz * 0.5)
    y1 = yh + (py1 / pz1) * (dz * 0.5)
    z1 = z + dz
    dt += (dz * 0.5) / pz1 * gamma * m_kg #  s

    # --- Pack updated (px, py, P)
    w1 = np.stack([px1, py1, P], axis=1)

    return x1, y1, z1, w1, dt


BorisSpatialIntegrator._step_fn = staticmethod(step_spatial_boris_B)


def _g_theta(theta):
    """g(theta) = (theta/2)*cot(theta/2).

    Scale factor from the BCH analysis of the drift-kick-rotate-kick-drift
    (D-K-R-K-D) splitting used in `step_spatial_boris_B`: replacing the
    D, K generators by g(theta)*D, g(theta)*K (theta = q*Bz*dz/pz, the
    Bz-rotation angle over the full step) makes them appear in the merged
    BCH exponent with coefficient 1 instead of 1 - theta^2/12 + O(theta^4),
    following the derivation in claude_notes/. g(0) = 1; series-expand near
    theta = 0 to avoid the removable 0/0 in cot(0).
    """
    theta = np.asarray(theta, dtype=float)
    g = np.empty_like(theta)
    small = np.abs(theta) < 1e-6
    th_small = theta[small]
    g[small] = 1.0 - th_small**2 / 12.0
    th = theta[~small] / 2.0
    g[~small] = th / np.tan(th)
    return g


def _pT_coeff(theta, g):
    """g(1-g)/theta, the theta-dependent factor in p_T = (h^2/P_z)*g(1-g)/theta.

    Series-expanded near theta=0 (using g = 1 - theta^2/12 + ...) to avoid
    the removable 0/0: g(1-g)/theta -> theta/12 + O(theta^3).
    """
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(g, dtype=float)
    small = np.abs(theta) < 1e-6
    out = np.empty_like(theta)
    out[small] = theta[small] / 12.0
    out[~small] = g[~small] * (1.0 - g[~small]) / theta[~small]
    return out


def step_spatial_boris_B_corrected(x, y, z, w, q, dz, field_fn, m_kg, gamma):
    """
    Spatial Boris step for magnetic fields only (E = 0), like
    `step_spatial_boris_B`, but with the drift and kick sub-step
    amplitudes rescaled by 1/g(theta), g(theta) = (theta/2)*cot(theta/2)
    -- see `_g_theta`.

    The BCH exponent of the (uncorrected) D-K-R-K-D splitting is

        exp(h/2 D) exp(h Z_KR) exp(h/2 D) = exp(h R + h g(theta) (D + K) + ...).

    Since theta (fixed by the rotation R alone) is unaffected by the
    amplitude of the D, K sub-steps, redoing the same composition with
    D, K scaled by a prefactor c gives a merged (D+K) coefficient of
    c*g(theta) (linear in c, at fixed theta). To recover coefficient 1
    (matching the true D + K + R generator) requires c = 1/g(theta) --
    *dividing* the sub-step amplitude by g(theta), not multiplying by it
    (multiplying, as an earlier version of this function did, gives
    coefficient g(theta)^2, moving away from the target instead of
    towards it).

    The residual T = q*Bx*d_x + q*By*d_y term from the full BCH exponent
    has coefficient p_T = (h^2/P_z)*g(1-g)/theta for c = 1 (see
    `_pT_coeff`). Since T's coefficient must also scale with c (the same
    way the (D+K) coefficient picked up a factor of c above), and p_T
    itself already carries one net factor of c from the BCH bookkeeping
    at c = 1, the c = 1/g version needs an extra 1/g^2 relative to the
    c = 1 formula: cT = p_T / g^2 = (h^2/P_z) * g(1-g)/theta / g^2. This
    is applied as a position shift x1 -= cT*q*Bx0, y1 -= cT*q*By0 using
    the predictor-pass field/P_z (Bx0, By0, pz_kick0), since cT is itself
    O(h^3) and the exact evaluation point only affects higher orders. The
    sign has not been independently verified (only checked empirically
    against the symplectic-error metric, see claude_notes/); flip it if
    that check disagrees.

    Per step:
      1. Predictor half-drift, evaluate the field at the (uncorrected)
         midpoint.
      2. Predictor half-kick from that field, to get P_z at the kick point
         -- together with the predictor Bz this gives theta and g(theta).
      3. Real drift-kick-rotate-kick-drift step, with the drift and kick
         (but not the rotation) sub-steps scaled by 1/g(theta).
      4. T-correction: shift x, y by -cT*q*Bx0, -cT*q*By0.

    Parameters / returns: same as `step_spatial_boris_B`.
    """

    dt = 0.

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    w = np.asarray(w)
    px, py, P = w[:, 0], w[:, 1], w[:, 2]

    pz = np.sqrt(np.maximum(P**2 - px**2 - py**2, 0.0))

    # ============================================
    # (0) PREDICTOR PASS: half-drift, evaluate field, half-kick, get P_z
    # ============================================
    xh0 = x + (px / pz) * (dz * 0.5)
    yh0 = y + (py / pz) * (dz * 0.5)
    zh0 = z + dz * 0.5

    Bx0, By0, Bz0 = field_fn(xh0, yh0, zh0)

    pxm0 = px - 0.5 * q * dz * By0
    pym0 = py + 0.5 * q * dz * Bx0
    pz_kick0 = np.sqrt(np.maximum(P**2 - pxm0**2 - pym0**2, 0.0))

    theta = q * Bz0 * dz / pz_kick0
    g = _g_theta(theta)
    ginv = 1.0 / g

    # --- Corrected half drift in (x, y) ---
    xh = x + ginv * (px / pz) * (dz * 0.5)
    yh = y + ginv * (py / pz) * (dz * 0.5)
    zh = z + dz * 0.5
    dt += (dz * 0.5) / pz * gamma * m_kg #  s

    # --- Evaluate magnetic field at the corrected mid-step ---
    Bx, By, Bz = field_fn(xh, yh, zh)

    # ============================================
    # (1) FIRST HALF-KICK from (Bx, By), corrected
    # ============================================
    pxm = px - ginv * 0.5 * q * dz * By
    pym = py + ginv * 0.5 * q * dz * Bx

    # --- Recompute pz to maintain |p| = P
    pz = np.sqrt(np.maximum(P**2 - pxm**2 - pym**2, 0.0))

    # ============================================
    # (2) ROTATION due to Bz -- unscaled
    # ============================================
    t = 0.5 * q * Bz * dz / pz
    t2 = t * t
    s = 2.0 * t / (1.0 + t2)
    c0 = (1.0 - t2) / (1.0 + t2)

    pxp = c0 * pxm + s * pym
    pyp = -s * pxm + c0 * pym

    # ============================================
    # (3) SECOND HALF-KICK from (Bx, By), corrected
    # ============================================
    px1 = pxp - ginv * 0.5 * q * dz * By
    py1 = pyp + ginv * 0.5 * q * dz * Bx

    # --- Recompute pz after full step
    pz1 = np.sqrt(np.maximum(P**2 - px1**2 - py1**2, 0.0))

    # ============================================
    # (4) SECOND HALF-DRIFT (x, y), corrected
    # ============================================
    x1 = xh + ginv * (px1 / pz1) * (dz * 0.5)
    y1 = yh + ginv * (py1 / pz1) * (dz * 0.5)
    z1 = z + dz
    dt += (dz * 0.5) / pz1 * gamma * m_kg #  s

    # ============================================
    # (5) T-CORRECTION: cancel residual p_T*T term, rederived for c=1/g
    # (sign TBD empirically -- see step_spatial_boris_B_corrected docstring)
    # ============================================
    cT = (dz**2 / pz_kick0) * _pT_coeff(theta, g) / g**2
    x1 = x1 - cT * q * Bx0
    y1 = y1 - cT * q * By0

    # --- Pack updated (px, py, P)
    w1 = np.stack([px1, py1, P], axis=1)

    return x1, y1, z1, w1, dt


class BorisSpatialCorrected(BorisSpatialIntegrator):
    """`BorisSpatialIntegrator`, but each step's drift and kick sub-steps
    are rescaled by 1/g(theta), g(theta) = (theta/2)*cot(theta/2),
    theta = q*Bz*dz/pz -- see `step_spatial_boris_B_corrected`. Also
    applies a rederived T-term correction (p_T/g^2, see the function
    docstring) to cancel the residual T = q*Bx*d_x + q*By*d_y generator
    from the BCH exponent.
    Experimental: tests the hypothesis (claude_notes/) that this rescaling
    reduces the integrator's symplectic error.
    """
    _step_fn = staticmethod(step_spatial_boris_B_corrected)
