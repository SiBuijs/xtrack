// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_HELICALBORIS_H
#define XTRACK_TRACK_HELICALBORIS_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/splineboris_src/spline_B_field_eval.h" // evaluate_B for Bx, By, Bs (scalar version)
#ifndef XTRACK_MULTIPOLE_NO_SYNRAD
// Forward declarations for random functions needed by synrad_spectrum.h
// (These are normally declared in random headers but we avoid including them
//  directly to prevent issues with generated data types)
GPUFUN uint32_t RandomUniformUInt32_generate(LocalParticle* part);
GPUFUN double RandomUniform_generate(LocalParticle* part);
GPUFUN double RandomExponential_generate(LocalParticle* part);
#include "xtrack/beam_elements/elements_src/track_magnet_radiation.h"
#endif

// ----------------------------------------------------------------------
//  Helical Boris kernel (field-aligned exact constant-B substep)
//
//  Each substep:
//     1. Predictor half-drift in (x, y) to the midpoint (not committed);
//        same s_mid = s_local + ds/2 as before for evaluate_B.
//     2. Evaluate B at the midpoint.
//     3. Field-aligned exact helix: decompose p along bhat = B/|B|,
//        solve Newton for arc length ds_arc so longitudinal advance
//        along the element s-axis equals ds, then Rodrigues + closed-form
//        dr_perp / dr_par.  |p| is preserved to roundoff in the ideal map.
//
//  If the FA step fails (non-single-valued advance in s, Newton stall,
//  singular derivative), fall back to the legacy closed-form F-map
//  (sinc/vers/cosc) for that substep only — same void API as before.
//
//  Legacy F-map reference (used as fallback): matrix F = exp(h F) with
//  theta = q Bs ds / ps in the pure-Bs dominant parametrisation; see
//  helical_legacy_F_step below.
// ----------------------------------------------------------------------

GPUFUN double helical_dot3(double ax, double ay, double az,
                                  double bx, double by, double bz) {
    return ax * bx + ay * by + az * bz;
}

GPUFUN void helical_cross3(double ax, double ay, double az,
                                    double bx, double by, double bz,
                                    double* ox, double* oy, double* oz) {
    *ox = ay * bz - az * by;
    *oy = az * bx - ax * bz;
    *oz = ax * by - ay * bx;
}

GPUFUN double helical_norm3(double vx, double vy, double vz) {
    return sqrt(vx * vx + vy * vy + vz * vz);
}

/* Legacy analytic F-map (constant Bs-dominant form); used as fallback. */
GPUFUN void helical_legacy_F_step(
    double x, double y,
    double px, double py,
    double ps, double inv_ps,
    double P,
    double q_coulomb,
    double ds,
    double Bx_loc, double By_loc, double Bs_loc,
    double* x_new, double* y_new,
    double* px_new, double* py_new
) {
    const double theta = q_coulomb * Bs_loc * ds * inv_ps;
    double S, C, sinc_th, vers_th, cosc_th;
    const double abs_theta = fabs(theta);
    if (abs_theta < 1.0e-3) {
        const double t2 = theta * theta;
        const double t4 = t2 * t2;
        sinc_th = 1.0 - t2 / 6.0 + t4 / 120.0;
        vers_th = 0.5 - t2 / 24.0 + t4 / 720.0;
        cosc_th = 1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0;
        S = theta * sinc_th;
        C = 1.0 - t2 * vers_th;
    } else {
        S = sin(theta);
        C = cos(theta);
        sinc_th = S / theta;
        vers_th = (1.0 - C) / (theta * theta);
        cosc_th = (theta - S) / (theta * theta * theta);
    }

    const double ds_over_ps  = ds * inv_ps;
    const double ds2_over_ps = ds * ds_over_ps;

    const double a_xx = ds_over_ps * sinc_th;
    const double a_xy = ds_over_ps * theta * vers_th;

    const double pos_kick_v = q_coulomb * ds2_over_ps * vers_th;
    const double pos_kick_c = q_coulomb * ds2_over_ps * theta * cosc_th;

    const double mom_kick_S = q_coulomb * ds * sinc_th;
    const double mom_kick_V = q_coulomb * ds * theta * vers_th;

    *x_new =
          x
        + a_xx * px
        + a_xy * py
        + pos_kick_v * (-By_loc)
        + pos_kick_c *   Bx_loc;

    *y_new =
          y
        - a_xy * px
        + a_xx * py
        + pos_kick_c *   By_loc
        + pos_kick_v *   Bx_loc;

    *px_new =
          C * px
        + S * py
        + mom_kick_S * (-By_loc)
        + mom_kick_V *   Bx_loc;

    *py_new =
         -S * px
        + C * py
        + mom_kick_V * (-By_loc)
        + mom_kick_S *   Bx_loc;

    (void)P;
}

/*
 * Field-aligned exact step: advance (x,y) and (px,py,ps) so longitudinal
 * displacement along the element s-axis equals dz (= ds). Returns 1 on
 * success; 0 triggers legacy fallback. On success, *ds_arc_out is physical
 * arc length along the helix.
 */
GPUFUN int helical_fa_try_step(
    double x0, double y0,
    double px0, double py0, double ps0,
    double P,
    double q_coulomb,
    double Bx, double By, double Bs,
    double dz,
    double* x_out, double* y_out,
    double* px_out, double* py_out, double* ps_out,
    double* ds_arc_out
) {
    const double B_mag = helical_norm3(Bx, By, Bs);
    const double B_tol = 1.0e-30;

    /* Degenerate field: straight drift in (x,y) at fixed ps */
    if (B_mag < B_tol) {
        const double inv_ps0 = 1.0 / ps0;
        *x_out = x0 + px0 * inv_ps0 * dz;
        *y_out = y0 + py0 * inv_ps0 * dz;
        *px_out = px0;
        *py_out = py0;
        *ps_out = ps0;
        *ds_arc_out = dz * P * inv_ps0;
        return 1;
    }

    const double inv_B = 1.0 / B_mag;
    const double bx = Bx * inv_B;
    const double by = By * inv_B;
    const double bz = Bs * inv_B;

    const double P_par = helical_dot3(px0, py0, ps0, bx, by, bz);
    const double px_perp = px0 - P_par * bx;
    const double py_perp = py0 - P_par * by;
    const double pz_perp = ps0 - P_par * bz;

    double c1x, c1y, c1z;
    helical_cross3(bx, by, bz, px_perp, py_perp, pz_perp, &c1x, &c1y, &c1z);

    const double kappa = q_coulomb * B_mag / P;
    const double inv_qB = 1.0 / (q_coulomb * B_mag);

    const double A = (P_par / P) * bz;
    const double C1 = pz_perp * inv_qB;
    const double C2 = c1z * inv_qB;

    const double amp = sqrt(C1 * C1 + C2 * C2);
    const double kabs = fabs(kappa);
    const double sv_rhs = amp * kabs;
    const double sv_marg = 1.0e-14 * (fabs(A) > 1.0 ? fabs(A) : 1.0);
    if (!(A > sv_rhs + sv_marg)) {
        return 0;
    }

    /* Newton: f(s) = A*s + C1*sin(kappa*s) + C2*(1-cos(kappa*s)) - dz */
    const double tol = 1.0e-13 * (fabs(dz) > 0.0 ? fabs(dz) : 1.0);
    double s_arc;
    if (fabs(A) > 1.0e-14 * fmax(fabs(dz), 1.0)) {
        s_arc = dz / A;
    } else {
        s_arc = dz * P / fmax(fabs(P_par), 1.0e-30 * P);
    }

    int converged = 0;
    for (int it = 0; it < 20; ++it) {
        const double u = kappa * s_arc;
        const double ca = cos(u);
        const double sa = sin(u);
        const double f = A * s_arc + C1 * sa + C2 * (1.0 - ca) - dz;
        if (fabs(f) < tol) {
            converged = 1;
            break;
        }
        const double fp = A + C1 * kappa * ca + C2 * kappa * sa;
        if (fabs(fp) < 1.0e-30 * P) {
            break;
        }
        s_arc -= f / fp;
    }

    if (!converged) {
        const double u = kappa * s_arc;
        const double ca = cos(u);
        const double sa = sin(u);
        const double f = A * s_arc + C1 * sa + C2 * (1.0 - ca) - dz;
        if (fabs(f) > 100.0 * tol) {
            return 0;
        }
    }

    const double angle = kappa * s_arc;
    const double c = cos(angle);
    const double sn = sin(angle);

    /* Pperp_end = c * Pperp - sn * (bhat x Pperp);  bxp = bhat x Pperp */
    const double ppxe = c * px_perp - sn * c1x;
    const double ppye = c * py_perp - sn * c1y;
    const double ppze = c * pz_perp - sn * c1z;

    const double px_e = P_par * bx + ppxe;
    const double py_e = P_par * by + ppye;
    const double ps_e = P_par * bz + ppze;

    const double drpx = inv_qB * (sn * px_perp + (1.0 - c) * c1x);
    const double drpy = inv_qB * (sn * py_perp + (1.0 - c) * c1y);
    const double drpz = inv_qB * (sn * pz_perp + (1.0 - c) * c1z);

    const double fac = (P_par / P) * s_arc;
    const double drsx = fac * bx;
    const double drsy = fac * by;
    const double drsz = fac * bz;

    *x_out = x0 + drpx + drsx;
    *y_out = y0 + drpy + drsy;
    *px_out = px_e;
    *py_out = py_e;
    *ps_out = ps_e;
    *ds_arc_out = s_arc;

    return 1;
}

GPUFUN
void HelicalBoris_single_particle(
    LocalParticle* part,
    const double  bs[5],
    const double* const *by,
    const double* const *bx,
    const int      multipole_order,
    const double   length,
    const int      n_steps,
    const double   shift_x,
    const double   shift_y,
    const int64_t  radiation_flag,
    SynchrotronRadiationRecordData radiation_record
){

    // Skip dead particles (state <= 0)
    if (LocalParticle_get_state(part) <= 0){
        return;
    }

    // ----------------------------------------------------------------------
    //  Extract particle/reference parameters
    // ----------------------------------------------------------------------
    const double c      = C_LIGHT;   // [m/s]
    const double qe     = QELEM;     // [C], elementary charge (absolute value)

    const double q0     = LocalParticle_get_q0(part);      // charge in units of e
    const double mass0  = LocalParticle_get_mass0(part);   // [eV]
    const double p0c_ev = LocalParticle_get_p0c(part);     // reference p0 c [eV]
    const double beta0  = LocalParticle_get_beta0(part);   // reference beta

    // Positions and momenta (dimensionless px, py)
    double x    = LocalParticle_get_x(part);   // [m]
    double y    = LocalParticle_get_y(part);   // [m]
    double px_r = LocalParticle_get_px(part);  // dimensionless px / p0
    double py_r = LocalParticle_get_py(part);  // dimensionless py / p0
    double zeta = LocalParticle_get_zeta(part);

    // ----------------------------------------------------------------------
    //  Convert to physical units (kg, m, s)
    // ----------------------------------------------------------------------
    const double energy0 = LocalParticle_get_energy0(part);  // [eV]
    const double charge_ratio = LocalParticle_get_charge_ratio(part);
    const double chi = LocalParticle_get_chi(part);
    const double mass_ratio = charge_ratio / chi;

    const double mass = mass_ratio * mass0;  // [eV]
    const double mass_kg = mass * qe / (c * c);  // [kg]

    // Reference momentum P0 in SI units
    const double P0 = p0c_ev * qe / c;  // [kg m / s]

    double px = px_r * P0;  // [kg m / s]
    double py = py_r * P0;  // [kg m / s]

    const double q_coulomb = q0 * qe;  // [C]

    // ----------------------------------------------------------------------
    //  Set up longitudinal stepping in local s in [0, length]
    // ----------------------------------------------------------------------
    const double L    = length;
    const double ds   = L / (double) n_steps;
    const double half_ds = 0.5 * ds;

    double s_local = 0.0;

    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
    double old_kin_px = 0.0, old_kin_py = 0.0;
    double dp_record_exit = 0.0, dpx_record_exit = 0.0, dpy_record_exit = 0.0;
    #endif

    // ----------------------------------------------------------------------
    //  Loop over substeps
    // ----------------------------------------------------------------------
    for (int istep = 0; istep < n_steps; ++istep) {

        // Recalculate energy, gamma and total momentum each step since
        // ptau/delta may change (e.g., from radiation).
        const double ptau = LocalParticle_get_ptau(part);
        const double delta = LocalParticle_get_delta(part);
        const double energy = (energy0 + ptau * p0c_ev) * mass_ratio;  // [eV]
        const double gamma = energy / mass;

        const double P = P0 * (1.0 + delta);  // [kg m / s]

        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (radiation_flag && ds > 0) {
            double ax_old = LocalParticle_get_ax(part);
            double ay_old = LocalParticle_get_ay(part);
            old_kin_px = (px / P0) - ax_old;
            old_kin_py = (py / P0) - ay_old;
        }
        #endif

        // --------------------------------------------------------------
        //  (0) Longitudinal momentum from |p| = P
        // --------------------------------------------------------------
        double tmp = P * P - px * px - py * py;
        if (tmp < 0.0) tmp = 0.0;
        const double ps = sqrt(tmp);     // [kg m / s]
        if (ps == 0.0) {
            break;
        }
        const double inv_ps = 1.0 / ps;

        // --------------------------------------------------------------
        //  (1) PREDICTOR HALF-DRIFT (midpoint only; not committed)
        // --------------------------------------------------------------
        const double xh = x + (px * inv_ps) * half_ds;
        const double yh = y + (py * inv_ps) * half_ds;
        const double s_local_h = s_local + half_ds;

        // --------------------------------------------------------------
        //  (2) Evaluate B at the midpoint
        // --------------------------------------------------------------
        double Bx_loc;
        double By_loc;
        double Bs_loc;

        evaluate_B(
            xh - shift_x,
            yh - shift_y,
            s_local_h,
            bs,
            by,
            bx,
            L,
            multipole_order,
            &Bx_loc,
            &By_loc,
            &Bs_loc
        );

        // --------------------------------------------------------------
        //  (3) Field-aligned exact step (fallback: legacy F-map)
        // --------------------------------------------------------------
        double x_new, y_new, px_new, py_new, ps_new;
        double ds_arc = ds;
        double dt;

        const int fa_ok = helical_fa_try_step(
            x, y, px, py, ps, P, q_coulomb,
            Bx_loc, By_loc, Bs_loc, ds,
            &x_new, &y_new, &px_new, &py_new, &ps_new, &ds_arc
        );

        if (fa_ok) {
            tmp = P * P - px_new * px_new - py_new * py_new;
            if (tmp < 0.0) {
                tmp = 0.0;
            }
            const double ps_chk = sqrt(tmp);
            if (ps_chk == 0.0) {
                helical_legacy_F_step(
                    x, y, px, py, ps, inv_ps, P, q_coulomb, ds,
                    Bx_loc, By_loc, Bs_loc,
                    &x_new, &y_new, &px_new, &py_new
                );
                ds_arc = ds;
                tmp = P * P - px_new * px_new - py_new * py_new;
                if (tmp < 0.0) tmp = 0.0;
                ps_new = sqrt(tmp);
            }
            dt = gamma * mass_kg * ds_arc / P;
        } else {
            helical_legacy_F_step(
                x, y, px, py, ps, inv_ps, P, q_coulomb, ds,
                Bx_loc, By_loc, Bs_loc,
                &x_new, &y_new, &px_new, &py_new
            );
            ds_arc = ds;
            tmp = P * P - px_new * px_new - py_new * py_new;
            if (tmp < 0.0) tmp = 0.0;
            ps_new = sqrt(tmp);
            dt = half_ds * inv_ps * gamma * mass_kg;
            const double inv_ps_new = (ps_new > 0.0) ? (1.0 / ps_new) : 0.0;
            dt += half_ds * inv_ps_new * gamma * mass_kg;
        }

        x = x_new;
        y = y_new;
        px = px_new;
        py = py_new;
        s_local += ds;

        zeta += (ds - dt * c * beta0);

        // --------------------------------------------------------------
        //  (5) Spin and radiation (identical hooks to SplineBoris)
        // --------------------------------------------------------------
        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        double const rvv = LocalParticle_get_rvv(part);
        double const l_path = rvv * dt * c * beta0;

        magnet_spin(part, Bx_loc, By_loc, Bs_loc, 0.0, ds_arc, l_path);

        if (radiation_flag && ds > 0) {
            double ax_new = LocalParticle_get_ax(part);
            double ay_new = LocalParticle_get_ay(part);
            double new_kin_px = (px / P0) - ax_new;
            double new_kin_py = (py / P0) - ay_new;

            double mean_kin_px = 0.5 * (old_kin_px + new_kin_px);
            double mean_kin_py = 0.5 * (old_kin_py + new_kin_py);

            double const B_perp_T = compute_b_perp_mod(
                mean_kin_px,
                mean_kin_py,
                LocalParticle_get_delta(part),
                Bx_loc, By_loc, Bs_loc
            );

            magnet_radiation(
                part,
                B_perp_T,
                ds_arc,  // physical path length along trajectory
                l_path,
                radiation_flag,
                radiation_record,
                &dp_record_exit, &dpx_record_exit, &dpy_record_exit
            );
        }
        #endif
    }

    // ------------------------------------------------------------------
    //  Write back to LocalParticle
    // ------------------------------------------------------------------
    LocalParticle_set_x(part, x);
    LocalParticle_set_y(part, y);

    LocalParticle_add_to_s(part, L);

    LocalParticle_set_px(part, px / P0);
    LocalParticle_set_py(part, py / P0);

    LocalParticle_set_zeta(part, zeta);
}

#endif // XTRACK_TRACK_HELICALBORIS_H
