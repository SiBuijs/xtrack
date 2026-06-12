// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_TRACK_BORISSOLENOID_H
#define XTRACK_TRACK_BORISSOLENOID_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/borissolenoid_src/solenoid_B_field_eval.h"
#include "xtrack/beam_elements/borissolenoid_src/helical_map.h"
#ifndef XTRACK_MULTIPOLE_NO_SYNRAD
// Forward declarations for random functions needed by synrad_spectrum.h
GPUFUN uint32_t RandomUniformUInt32_generate(LocalParticle* part);
GPUFUN double RandomUniform_generate(LocalParticle* part);
GPUFUN double RandomExponential_generate(LocalParticle* part);
#include "xtrack/beam_elements/elements_src/track_magnet_radiation.h"
#endif

GPUFUN
void BorisSolenoid_single_particle(
    LocalParticle* part,
    const double L_coil,
    const double a,
    const double B0,
    const double z0,
    const double length,
    const int n_steps,
    const double shift_x,
    const double shift_y,
    const int64_t radiation_flag,
    SynchrotronRadiationRecordData radiation_record
) {
    if (LocalParticle_get_state(part) <= 0) {
        return;
    }

    const double c = C_LIGHT;
    const double qe = QELEM;

    const double q0 = LocalParticle_get_q0(part);
    const double mass0 = LocalParticle_get_mass0(part);
    const double p0c_ev = LocalParticle_get_p0c(part);
    const double beta0 = LocalParticle_get_beta0(part);

    double x = LocalParticle_get_x(part);
    double y = LocalParticle_get_y(part);
    double px_r = LocalParticle_get_px(part);
    double py_r = LocalParticle_get_py(part);
    double zeta = LocalParticle_get_zeta(part);

    const double energy0 = LocalParticle_get_energy0(part);
    const double charge_ratio = LocalParticle_get_charge_ratio(part);
    const double chi = LocalParticle_get_chi(part);
    const double mass_ratio = charge_ratio / chi;

    const double mass = mass_ratio * mass0;
    const double mass_kg = mass * qe / (c * c);
    const double P0 = p0c_ev * qe / c;

    double px = px_r * P0;
    double py = py_r * P0;

    const double q_coulomb = q0 * charge_ratio * qe;

    const double L = length;
    const double ds = L / (double)n_steps;
    const double half_ds = 0.5 * ds;

    const double s_entry = LocalParticle_get_s(part);
    double s_local = 0.0;

    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
    double old_kin_px = 0.0;
    double old_kin_py = 0.0;
    double dp_record_exit = 0.0;
    double dpx_record_exit = 0.0;
    double dpy_record_exit = 0.0;
    #endif

    for (int istep = 0; istep < n_steps; ++istep) {
        const double ptau = LocalParticle_get_ptau(part);
        const double delta = LocalParticle_get_delta(part);
        const double energy = (energy0 + ptau * p0c_ev) * mass_ratio;
        const double gamma = energy / mass;
        const double P = P0 * (1.0 + delta);

        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (radiation_flag && ds > 0) {
            const double ax_old = LocalParticle_get_ax(part);
            const double ay_old = LocalParticle_get_ay(part);
            old_kin_px = (px / P0) - ax_old;
            old_kin_py = (py / P0) - ay_old;
        }
        #endif

        double tmp = P * P - px * px - py * py;
        if (tmp < 0.0) tmp = 0.0;
        double ps = sqrt(tmp);
        if (ps == 0.0) {
            break;
        }

        const double inv_ps = 1.0 / ps;

        // Half-drift predictor for field evaluation only.
        const double xh = x + (px * inv_ps) * half_ds;
        const double yh = y + (py * inv_ps) * half_ds;
        const double s_eval = s_entry + s_local + half_ds;

        double Bx;
        double By;
        double Bz;

        evaluate_solenoid_B(
            xh - shift_x,
            yh - shift_y,
            s_eval,
            L_coil,
            a,
            B0,
            z0,
            &Bx,
            &By,
            &Bz
        );

        const double B_mag = sqrt(Bx * Bx + By * By + Bz * Bz);
        const double B_perp = sqrt(Bx * Bx + By * By);

        double dt = 0.0;
        int step_completed = 0;

        if (B_mag < BORISSOLENOID_HELICAL_EPS) {
            x += (px * inv_ps) * ds;
            y += (py * inv_ps) * ds;
            s_local += ds;
            dt = ds * inv_ps * gamma * mass_kg;
            step_completed = 1;
        } else if (fabs(Bz) < BORISSOLENOID_HELICAL_EPS) {
            break;
        } else {
            const double P_z = (Bx * px + By * py + Bz * ps) / B_mag;
            if (fabs(P_z) < BORISSOLENOID_HELICAL_EPS) {
                break;
            }

            const double s_current = s_entry + s_local;
            const double z_target = s_current + ds;
            const double h_init = ds * B_mag / fabs(Bz);

            double x_z;
            double y_z;
            double z_z;
            double px_z;
            double py_z;
            double pz_z;

            borissolenoid_vec_to_zeta(
                Bx, By, Bz, B_mag, B_perp,
                x, y, s_current,
                &x_z, &y_z, &z_z
            );
            borissolenoid_vec_to_zeta(
                Bx, By, Bz, B_mag, B_perp,
                px, py, ps,
                &px_z, &py_z, &pz_z
            );

            const double h = borissolenoid_solve_helical_h_for_z_plane(
                x_z, y_z, z_z, px_z, py_z,
                Bx, By, Bz, B_mag, B_perp,
                P_z, q_coulomb, z_target, h_init
            );

            // Pure helical map in zeta frame (B_x = B_y = 0, K = 0).
            borissolenoid_helical_F_step(
                &x_z,
                &y_z,
                &px_z,
                &py_z,
                B_mag,
                P_z,
                q_coulomb,
                h
            );
            z_z += h;

            double x_new;
            double y_new;
            double z_unused;
            double px_new;
            double py_new;
            double ps_new;

            borissolenoid_vec_to_lab(
                Bx, By, Bz, B_mag, B_perp,
                x_z, y_z, z_z,
                &x_new, &y_new, &z_unused
            );
            borissolenoid_vec_to_lab(
                Bx, By, Bz, B_mag, B_perp,
                px_z, py_z, P_z,
                &px_new, &py_new, &ps_new
            );

            x = x_new;
            y = y_new;
            px = px_new;
            py = py_new;

            tmp = P * P - px * px - py * py;
            if (tmp < 0.0) tmp = 0.0;
            ps = sqrt(tmp);
            if (ps == 0.0) {
                break;
            }

            s_local += ds;

            // Exact time along the frozen-field helix (P_z conserved over the step).
            dt = h * gamma * mass_kg / P_z;
            step_completed = 1;
        }

        if (!step_completed) {
            break;
        }

        zeta += ds - dt * c * beta0;

        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        const double rvv = LocalParticle_get_rvv(part);
        const double l_path = rvv * dt * c * beta0;

        // Field evaluated at midpoint (xh, yh); track spin over step length ds.
        magnet_spin(part, Bx, By, Bz, 0.0, ds, l_path);

        if (radiation_flag && ds > 0) {
            const double ax_new = LocalParticle_get_ax(part);
            const double ay_new = LocalParticle_get_ay(part);
            const double new_kin_px = (px / P0) - ax_new;
            const double new_kin_py = (py / P0) - ay_new;

            const double mean_kin_px = 0.5 * (old_kin_px + new_kin_px);
            const double mean_kin_py = 0.5 * (old_kin_py + new_kin_py);

            const double B_perp_T = compute_b_perp_mod(
                mean_kin_px,
                mean_kin_py,
                LocalParticle_get_delta(part),
                Bx, By, Bz
            );

            magnet_radiation(
                part,
                B_perp_T,
                ds,
                l_path,
                radiation_flag,
                radiation_record,
                &dp_record_exit, &dpx_record_exit, &dpy_record_exit
            );
        }
        #endif
    }

    LocalParticle_set_x(part, x);
    LocalParticle_set_y(part, y);
    LocalParticle_add_to_s(part, L);
    LocalParticle_set_px(part, px / P0);
    LocalParticle_set_py(part, py / P0);
    LocalParticle_set_zeta(part, zeta);
}

#endif // XTRACK_TRACK_BORISSOLENOID_H
