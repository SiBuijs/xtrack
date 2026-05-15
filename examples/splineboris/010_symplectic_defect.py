"""
Validate the leading-order symplectic-defect formula

    Delta M = (h^3 / 12) * [
        [0, 0, -kappa^2/Pz,  0,            q*kappa*Bx/Pz],
        [0, 0,  0,           -kappa^2/Pz,  q*kappa*By/Pz],
        [0, 0,  0,            0,           q*kappa^2*By ],
        [0, 0,  0,            0,          -q*kappa^2*Bx ],
        [0, 0,  0,            0,            0           ],
    ] + O(h^4)

against the measured drift between ``SplineBoris`` (Strang Boris) and the
**legacy** closed-form helical F-map walked in Python.

The C ``HelicalBoris`` kernel in ``track_helicalboris.h`` now uses a
field-aligned exact constant-``B`` substep (midpoint ``B``, Newton for arc
length, Rodrigues).  For a quantitative cross-check on that kernel, this
script uses ``track_helical_fa_python`` (below), which mirrors the same
half-drift sampling and FA map (including the legacy F-map fallback path).

``predict_per_step_defect`` still walks the **legacy analytic F-map** per
substep (the pre–field-aligned ``HelicalBoris`` integrator).  It is kept
because:

    * it accumulates the leading-order ``Delta M . u`` at the same midpoint
      ``B`` samples as before, for comparison with Strang-split Boris;
    * the integrated ``Delta M`` prediction is aligned with the textbook
      split-operator helical map, not with the newer FA substeps.

We run two scenarios:

    A) Bs only:   the Strang split coincides with the textbook D K R K D
                  sequence the BCH analysis was built on.
                  The leading-order Delta M predicts the drift between
                  Boris and the **legacy** helical F-map with global O(h^2)
                  convergence.
    B) full B:    with B_x, B_y also nonzero, the legacy F-map divides by
                  ``P_z`` in the rotation step.  That introduces an extra
                  O(h^2) per-step error which is NOT part of the textbook
                  Delta M, so the predicted curve underestimates the drift
                  and shows global slope 2 vs observed slope 1.
                  This is informative, not a bug.
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import e as qe, c as clight

import xtrack as xt
from xtrack.beam_elements.splineboris import Spline4


# ===========================================================================
# Python mirror of field-aligned substeps in track_helicalboris.h
# ===========================================================================
def _helical_fa_try_step(
    x0: float,
    y0: float,
    px0: float,
    py0: float,
    ps0: float,
    P: float,
    q_coulomb: float,
    Bx: float,
    By: float,
    Bs: float,
    dz: float,
):
    """Return (ok, x, y, px, py, ds_arc).  Third axis = element s (Bs)."""
    B_mag = math.sqrt(Bx * Bx + By * By + Bs * Bs)
    if B_mag < 1.0e-30:
        inv_ps0 = 1.0 / ps0
        ds_arc = dz * P * inv_ps0
        return (
            True,
            x0 + px0 * inv_ps0 * dz,
            y0 + py0 * inv_ps0 * dz,
            px0,
            py0,
            ds_arc,
        )

    inv_B = 1.0 / B_mag
    bx, by, bz = Bx * inv_B, By * inv_B, Bs * inv_B
    P_par = px0 * bx + py0 * by + ps0 * bz
    px_perp = px0 - P_par * bx
    py_perp = py0 - P_par * by
    pz_perp = ps0 - P_par * bz
    c1x = by * pz_perp - bz * py_perp
    c1y = bz * px_perp - bx * pz_perp
    c1z = bx * py_perp - by * px_perp

    kappa = q_coulomb * B_mag / P
    inv_qB = 1.0 / (q_coulomb * B_mag)
    A = (P_par / P) * bz
    C1 = pz_perp * inv_qB
    C2 = c1z * inv_qB
    amp = math.sqrt(C1 * C1 + C2 * C2)
    kabs = abs(kappa)
    sv_rhs = amp * kabs
    sv_marg = 1.0e-14 * (abs(A) if abs(A) > 1.0 else 1.0)
    if not (A > sv_rhs + sv_marg):
        return (False, x0, y0, px0, py0, dz)

    tol = 1.0e-13 * (abs(dz) if abs(dz) > 0.0 else 1.0)
    if abs(A) > 1.0e-14 * max(abs(dz), 1.0):
        s_arc = dz / A
    else:
        s_arc = dz * P / max(abs(P_par), 1.0e-30 * P)

    converged = False
    for _ in range(20):
        u = kappa * s_arc
        ca, sa = math.cos(u), math.sin(u)
        f = A * s_arc + C1 * sa + C2 * (1.0 - ca) - dz
        if abs(f) < tol:
            converged = True
            break
        fp = A + C1 * kappa * ca + C2 * kappa * sa
        if abs(fp) < 1.0e-30 * P:
            break
        s_arc -= f / fp

    if not converged:
        u = kappa * s_arc
        ca, sa = math.cos(u), math.sin(u)
        f = A * s_arc + C1 * sa + C2 * (1.0 - ca) - dz
        if abs(f) > 100.0 * tol:
            return (False, x0, y0, px0, py0, dz)

    angle = kappa * s_arc
    c, sn = math.cos(angle), math.sin(angle)
    ppxe = c * px_perp - sn * c1x
    ppye = c * py_perp - sn * c1y
    px_e = P_par * bx + ppxe
    py_e = P_par * by + ppye

    drpx = inv_qB * (sn * px_perp + (1.0 - c) * c1x)
    drpy = inv_qB * (sn * py_perp + (1.0 - c) * c1y)
    fac = (P_par / P) * s_arc
    x_out = x0 + drpx + fac * bx
    y_out = y0 + drpy + fac * by
    return (True, x_out, y_out, px_e, py_e, s_arc)


def _helical_legacy_f_step(x, y, px, py, ps, inv_ps, q_coulomb, ds, Bx, By, Bs):
    theta = q_coulomb * Bs * ds * inv_ps
    if abs(theta) < 1.0e-3:
        t2 = theta * theta
        t4 = t2 * t2
        sinc_th = 1.0 - t2 / 6.0 + t4 / 120.0
        vers_th = 0.5 - t2 / 24.0 + t4 / 720.0
        cosc_th = 1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0
        S = theta * sinc_th
        C = 1.0 - t2 * vers_th
    else:
        S = math.sin(theta)
        C = math.cos(theta)
        sinc_th = S / theta
        vers_th = (1.0 - C) / (theta * theta)
        cosc_th = (theta - S) / (theta ** 3)

    ds_over_ps = ds * inv_ps
    ds2_over_ps = ds * ds_over_ps
    a_xx = ds_over_ps * sinc_th
    a_xy = ds_over_ps * theta * vers_th
    pos_kick_v = q_coulomb * ds2_over_ps * vers_th
    pos_kick_c = q_coulomb * ds2_over_ps * theta * cosc_th
    mom_kick_S = q_coulomb * ds * sinc_th
    mom_kick_V = q_coulomb * ds * theta * vers_th

    x_new = (
        x
        + a_xx * px
        + a_xy * py
        + pos_kick_v * (-By)
        + pos_kick_c * Bx
    )
    y_new = (
        y
        - a_xy * px
        + a_xx * py
        + pos_kick_c * By
        + pos_kick_v * Bx
    )
    px_new = C * px + S * py + mom_kick_S * (-By) + mom_kick_V * Bx
    py_new = -S * px + C * py + mom_kick_V * (-By) + mom_kick_S * Bx
    return x_new, y_new, px_new, py_new


def track_helical_fa_python(elem, particle, q_coulomb, P0):
    """Full element track mirroring C ``HelicalBoris`` (FA + legacy fallback)."""
    L = float(elem.length)
    n_steps = int(elem.n_steps)
    ds = L / n_steps
    half_ds = 0.5 * ds

    delta = float(np.atleast_1d(particle.delta)[0])
    P_tot = P0 * (1.0 + delta)
    x = float(np.atleast_1d(particle.x)[0])
    y = float(np.atleast_1d(particle.y)[0])
    px = float(np.atleast_1d(particle.px)[0]) * P0
    py = float(np.atleast_1d(particle.py)[0]) * P0
    s_local = 0.0

    for _ in range(n_steps):
        Pz = math.sqrt(max(P_tot * P_tot - px * px - py * py, 0.0))
        if Pz == 0.0:
            break
        inv_ps = 1.0 / Pz
        xh = x + (px / Pz) * half_ds
        yh = y + (py / Pz) * half_ds
        sh = s_local + half_ds
        Bx, By, Bs = elem.get_field(xh, yh, sh)
        Bx, By, Bs = float(Bx), float(By), float(Bs)

        ok, xa, ya, pxa, pya, _ds_arc = _helical_fa_try_step(
            x, y, px, py, Pz, P_tot, q_coulomb, Bx, By, Bs, ds,
        )
        if ok:
            tmp = P_tot * P_tot - pxa * pxa - pya * pya
            ps_chk = math.sqrt(max(tmp, 0.0))
            if ps_chk == 0.0:
                ok = False
        if not ok:
            xa, ya, pxa, pya = _helical_legacy_f_step(
                x, y, px, py, Pz, inv_ps, q_coulomb, ds, Bx, By, Bs,
            )
        x, y, px, py = xa, ya, pxa, pya
        s_local += ds

    return x, y, px / P0, py / P0


# ===========================================================================
# Python walk: legacy F-map + Delta M (symplectic-defect benchmark)
# ===========================================================================
def predict_per_step_defect(elem, particle, q_coulomb, P0):
    """Walk the legacy closed-form helical F-map in Python (not current C FA).

    For each substep this function does the same half-drift predictor,
    the same midpoint B evaluation through ``elem.get_field``, and the
    **legacy** F-map state update. In addition it accumulates the
    leading-order ``Delta M . u`` contribution at every substep.

    Parameters
    ----------
    elem : xt.SplineBoris
    particle : xt.Particles  (initial state, single particle)
    q_coulomb : float        signed particle charge [C]
    P0 : float               reference momentum [kg m / s]

    Returns
    -------
    s, x, y, px, py : ndarray
        Local s and the state after each substep. ``px``, ``py`` are
        dimensionless (divided by P0), matching the xobject convention.
    cum_dx, cum_dy, cum_dpx, cum_dpy : ndarray
        Cumulative predicted defect (signed: Boris - legacy helical F-map),
        in the same units as the state arrays.
    """
    L = float(elem.length)
    n_steps = int(elem.n_steps)
    ds = L / n_steps
    half_ds = 0.5 * ds
    coef = ds ** 3 / 12.0

    delta = float(np.atleast_1d(particle.delta)[0])
    P_tot = P0 * (1.0 + delta)

    x = float(np.atleast_1d(particle.x)[0])
    y = float(np.atleast_1d(particle.y)[0])
    Px = float(np.atleast_1d(particle.px)[0]) * P0
    Py = float(np.atleast_1d(particle.py)[0]) * P0
    s_local = 0.0

    s_log = [s_local]
    x_log = [x]
    y_log = [y]
    px_log = [Px / P0]
    py_log = [Py / P0]
    dx_log = [0.0]
    dy_log = [0.0]
    dpx_log = [0.0]
    dpy_log = [0.0]

    cum_dx = cum_dy = cum_dPx = cum_dPy = 0.0

    for _ in range(n_steps):
        Pz = np.sqrt(max(P_tot * P_tot - Px * Px - Py * Py, 0.0))
        if Pz == 0.0:
            break

        xh = x + (Px / Pz) * half_ds
        yh = y + (Py / Pz) * half_ds
        s_h = s_local + half_ds

        Bx, By, Bs = elem.get_field(xh, yh, s_h)
        Bx = float(Bx)
        By = float(By)
        Bs = float(Bs)

        kappa = q_coulomb * Bs / Pz
        theta = kappa * ds
        if abs(theta) < 1.0e-3:
            t2 = theta * theta
            t4 = t2 * t2
            sinc_th = 1.0 - t2 / 6.0 + t4 / 120.0
            vers_th = 0.5 - t2 / 24.0 + t4 / 720.0
            cosc_th = 1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0
            S = theta * sinc_th
            C = 1.0 - t2 * vers_th
        else:
            S = np.sin(theta)
            C = np.cos(theta)
            sinc_th = S / theta
            vers_th = (1.0 - C) / (theta * theta)
            cosc_th = (theta - S) / (theta ** 3)

        # Predicted leading-order Delta M . u using the state BEFORE this
        # substep. Position rows use (1,3)+(1,5) entries; momentum rows
        # come from (3,5)+(4,5).
        dx_step = coef * (-(kappa ** 2) / Pz * Px + q_coulomb * kappa * Bx / Pz)
        dy_step = coef * (-(kappa ** 2) / Pz * Py + q_coulomb * kappa * By / Pz)
        dPx_step = coef * (q_coulomb * kappa ** 2 * By)
        dPy_step = coef * (-q_coulomb * kappa ** 2 * Bx)

        cum_dx += dx_step
        cum_dy += dy_step
        cum_dPx += dPx_step
        cum_dPy += dPy_step

        # Legacy F-map state advance (symplectic-defect benchmark; not current C FA)
        ds_over_Pz = ds / Pz
        ds2_over_Pz = ds * ds_over_Pz
        a_xx = ds_over_Pz * sinc_th
        a_xy = ds_over_Pz * theta * vers_th
        pos_kick_v = q_coulomb * ds2_over_Pz * vers_th
        pos_kick_c = q_coulomb * ds2_over_Pz * theta * cosc_th
        mom_kick_S = q_coulomb * ds * sinc_th
        mom_kick_V = q_coulomb * ds * theta * vers_th

        x_new = x + a_xx * Px + a_xy * Py + pos_kick_v * (-By) + pos_kick_c * Bx
        y_new = y - a_xy * Px + a_xx * Py + pos_kick_c * By + pos_kick_v * Bx
        Px_new = C * Px + S * Py + mom_kick_S * (-By) + mom_kick_V * Bx
        Py_new = -S * Px + C * Py + mom_kick_V * (-By) + mom_kick_S * Bx

        x, y, Px, Py = x_new, y_new, Px_new, Py_new
        s_local += ds

        s_log.append(s_local)
        x_log.append(x)
        y_log.append(y)
        px_log.append(Px / P0)
        py_log.append(Py / P0)
        dx_log.append(cum_dx)
        dy_log.append(cum_dy)
        dpx_log.append(cum_dPx / P0)
        dpy_log.append(cum_dPy / P0)

    return (
        np.array(s_log),
        np.array(x_log), np.array(y_log),
        np.array(px_log), np.array(py_log),
        np.array(dx_log), np.array(dy_log),
        np.array(dpx_log), np.array(dpy_log),
    )


# ===========================================================================
# Particle + reference momentum (shared across scenarios)
# ===========================================================================
p0c_eV = 2.7e9
p_init = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV,
    q0=1,
    p0c=p0c_eV,
    x=1.0e-3,
    y=-0.5e-3,
    px=2.0e-4,
    py=-1.0e-4,
)
q_coulomb = float(np.atleast_1d(p_init.q0)[0]) * qe       # [C]
P0 = p0c_eV * qe / clight                                 # [kg m / s]


def run_scenario(name, Bx0, By0, Bs0, L=1.0, N_main=200,
                 N_list=(25, 50, 100, 200, 400, 800, 1600)):
    """Run one constant-B benchmark and compare measured drift to Delta M.

    Returns a dict with the convergence arrays so the caller can compose
    cross-scenario plots if desired.
    """
    bs_spl = Spline4(Bs0, 0.0, Bs0, 0.0, Bs0 * L)
    bx_spl = Spline4(Bx0, 0.0, Bx0, 0.0, Bx0 * L)
    by_spl = Spline4(By0, 0.0, By0, 0.0, By0 * L)

    def make_elem(use_helical: bool) -> xt.SplineBoris:
        return xt.SplineBoris(
            bs=bs_spl, bx=(bx_spl,), by=(by_spl,),
            length=L, n_steps=N_main, use_helical_boris=use_helical,
        )

    elem_boris = make_elem(False)
    elem_helic = make_elem(True)
    line_b = xt.Line(elements=[elem_boris]); line_b.build_tracker()
    line_h = xt.Line(elements=[elem_helic]); line_h.build_tracker()

    p_b = p_init.copy(); line_b.track(p_b)
    p_h = p_init.copy(); line_h.track(p_h)

    xf, yf, pxf, pyf = track_helical_fa_python(elem_helic, p_init, q_coulomb, P0)

    (s_pred, _, _, _, _, cum_dx, cum_dy, cum_dpx, cum_dpy) = predict_per_step_defect(
        elem_helic, p_init, q_coulomb, P0,
    )

    print(f"\n========================================================")
    print(f"Scenario [{name}]: L = {L} m, N_main = {N_main}, h = {L/N_main:.3e} m")
    print(f"  Bx, By, Bs = {Bx0}, {By0}, {Bs0} T")
    print(f"  q B_s / P_z (kappa) ~ {q_coulomb * Bs0 / P0:.4e} rad/m,  "
          f"total bend ~ {q_coulomb * Bs0 / P0 * L:.4e} rad")
    print(f"--------------------------------------------------------")

    print("Cross-check: Python field-aligned mirror vs C HelicalBoris (end state)")
    print(f"  |dx_FA|  = {abs(xf - float(p_h.x[0])):.3e}")
    print(f"  |dy_FA|  = {abs(yf - float(p_h.y[0])):.3e}")
    print(f"  |dpx_FA| = {abs(pxf - float(p_h.px[0])):.3e}")
    print(f"  |dpy_FA| = {abs(pyf - float(p_h.py[0])):.3e}")

    obs_dx = float(p_b.x[0]  - p_h.x[0])
    obs_dy = float(p_b.y[0]  - p_h.y[0])
    obs_dpx = float(p_b.px[0] - p_h.px[0])
    obs_dpy = float(p_b.py[0] - p_h.py[0])
    pred_end = dict(dx=cum_dx[-1], dy=cum_dy[-1], dpx=cum_dpx[-1], dpy=cum_dpy[-1])
    obs_end = dict(dx=obs_dx, dy=obs_dy, dpx=obs_dpx, dpy=obs_dpy)

    rows = []
    for coord in ('dx', 'dy', 'dpx', 'dpy'):
        p = pred_end[coord]
        o = obs_end[coord]
        rows.append({
            'coord': coord,
            'predicted (Delta M, legacy F helix)': p,
            'observed (Boris - C FA helical)': o,
            'residual': o - p,
            'ratio pred/obs': (p / o) if o != 0 else float('nan'),
        })
    df = pd.DataFrame(rows).set_index('coord')
    fmt = {c: '{:+.3e}'.format for c in df.columns if c != 'ratio pred/obs'}
    fmt['ratio pred/obs'] = '{:.4f}'.format
    print('\nDelta M (legacy split helical) vs measured Boris - field-aligned C')
    print(df.to_string(formatters=fmt))

    # Convergence sweep
    h_arr = np.array([L / N_ for N_ in N_list], dtype=float)
    obs_x = []; obs_y = []; obs_px = []; obs_py = []
    prd_x = []; prd_y = []; prd_px = []; prd_py = []
    for N_test in N_list:
        elem_b_t = xt.SplineBoris(bs=bs_spl, bx=(bx_spl,), by=(by_spl,),
                                  length=L, n_steps=N_test, use_helical_boris=False)
        elem_h_t = xt.SplineBoris(bs=bs_spl, bx=(bx_spl,), by=(by_spl,),
                                  length=L, n_steps=N_test, use_helical_boris=True)
        lb = xt.Line(elements=[elem_b_t]); lb.build_tracker()
        lh = xt.Line(elements=[elem_h_t]); lh.build_tracker()
        pb = p_init.copy(); lb.track(pb)
        ph = p_init.copy(); lh.track(ph)
        (_, _, _, _, _, cdx, cdy, cdpx, cdpy) = predict_per_step_defect(
            elem_h_t, p_init, q_coulomb, P0,
        )
        obs_x.append(float(pb.x[0]  - ph.x[0]));   prd_x.append(float(cdx[-1]))
        obs_y.append(float(pb.y[0]  - ph.y[0]));   prd_y.append(float(cdy[-1]))
        obs_px.append(float(pb.px[0] - ph.px[0])); prd_px.append(float(cdpx[-1]))
        obs_py.append(float(pb.py[0] - ph.py[0])); prd_py.append(float(cdpy[-1]))

    obs_x = np.array(obs_x); obs_y = np.array(obs_y)
    obs_px = np.array(obs_px); obs_py = np.array(obs_py)
    prd_x = np.array(prd_x); prd_y = np.array(prd_y)
    prd_px = np.array(prd_px); prd_py = np.array(prd_py)

    def _slope(arr):
        arr = np.abs(arr)
        mask = arr > 0
        if mask.sum() < 2:
            return float('nan')
        p = np.polyfit(np.log(h_arr[mask]), np.log(arr[mask]), 1)
        return p[0]

    print("\nConvergence: |Boris - C (FA helical)|  vs  cumulative |Delta M| (legacy F helix)")
    hdr = (
        f"{'N':>6} {'h':>10}  "
        f"{'|obs dx|':>10} {'|prd dx|':>10}  "
        f"{'|obs dy|':>10} {'|prd dy|':>10}  "
        f"{'|obs dpx|':>10} {'|prd dpx|':>10}  "
        f"{'|obs dpy|':>10} {'|prd dpy|':>10}"
    )
    print(hdr)
    for N_, h_, ox, px_, oy, py_, opx, ppx, opy, ppy in zip(
            N_list, h_arr, obs_x, prd_x, obs_y, prd_y, obs_px, prd_px, obs_py, prd_py):
        print(f"{N_:>6d} {h_:>10.3e}  "
              f"{abs(ox):>10.3e} {abs(px_):>10.3e}  "
              f"{abs(oy):>10.3e} {abs(py_):>10.3e}  "
              f"{abs(opx):>10.3e} {abs(ppx):>10.3e}  "
              f"{abs(opy):>10.3e} {abs(ppy):>10.3e}")

    print(f"\nLog-log slopes vs h (Delta M predicts 2):")
    print(f"  observed:  x = {_slope(obs_x):.3f}, y = {_slope(obs_y):.3f}, "
          f"px = {_slope(obs_px):.3f}, py = {_slope(obs_py):.3f}")
    print(f"  predicted: x = {_slope(prd_x):.3f}, y = {_slope(prd_y):.3f}, "
          f"px = {_slope(prd_px):.3f}, py = {_slope(prd_py):.3f}")

    # Per-substep curve plot for the main h
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (lbl, cum, obs) in zip(
            axs.flat,
            [('dx [m]',          cum_dx,  obs_dx),
             ('dy [m]',          cum_dy,  obs_dy),
             (r'$dp_x / p_0$',   cum_dpx, obs_dpx),
             (r'$dp_y / p_0$',   cum_dpy, obs_dpy)],
    ):
        ax.plot(s_pred, cum, '-', lw=2, label='predicted (Delta M)')
        ax.axhline(obs, color='C1', ls='--', label='observed (end)')
        ax.set_xlabel('s [m]')
        ax.set_ylabel(lbl)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'[{name}] per-substep predicted vs observed (h = {L/N_main:.2e} m)')

    # Convergence plot
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (lbl, obs, pred) in zip(
            axs.flat,
            [('dx',  obs_x,  prd_x),
             ('dy',  obs_y,  prd_y),
             ('dpx', obs_px, prd_px),
             ('dpy', obs_py, prd_py)],
    ):
        ax.loglog(h_arr, np.abs(obs), 'o-', label='observed')
        ax.loglog(h_arr, np.abs(pred), 's--', label='predicted (Delta M)')
        ref = np.abs(obs[0]) * (h_arr / h_arr[0]) ** 2
        ax.loglog(h_arr, ref, 'k:', alpha=0.5, label=r'$\propto h^2$')
        ax.set_xlabel('h [m]')
        ax.set_ylabel(f'end |{lbl}|')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
    fig.suptitle(f'[{name}] convergence of the symplectic defect with step size')

    return dict(h=h_arr, obs_x=obs_x, obs_y=obs_y, obs_px=obs_px, obs_py=obs_py,
                prd_x=prd_x, prd_y=prd_y, prd_px=prd_px, prd_py=prd_py)


# ===========================================================================
# Two scenarios
# ===========================================================================
#
# Scenario A: only B_z (= B_s).  The Strang split here is *exactly* the
# textbook  D K R K D  sequence on which the user's BCH analysis was built;
# in particular the Boris rotation uses 1/P_z that has not been perturbed by
# any transverse kick (B_x = B_y = 0 => no kick).  Delta M should predict the
# Boris-Helical drift to leading order, with h^2 global convergence.
#
# Scenario B: full B = (B_x, B_y, B_s).  The kernel's rotation step uses
# ``ps_mid = sqrt(P^2 - pxm^2 - pym^2)`` *after* the half-kick, i.e. it
# divides by the longitudinal momentum at midpoint, not at start-of-step.
# That introduces an extra O(h^2) per-step error which is NOT part of the
# textbook Delta M.  Globally it dominates the BCH O(h^3) contribution and
# shows up as h^1 convergence in the measured drift, while the predicted
# curve still scales as h^2.  This is informative, not a bug: it tells us
# which approximation in the kernel is the bottleneck.
# ===========================================================================

print("\n" + "#" * 60)
print("# Symplectic-defect validation")
print("#" * 60)

scen_A = run_scenario(name="Bs only", Bx0=0.0, By0=0.0, Bs0=1.0)
scen_B = run_scenario(name="Bx, By, Bs nontrivial", Bx0=0.05, By0=0.10, Bs0=1.0)

plt.show()
