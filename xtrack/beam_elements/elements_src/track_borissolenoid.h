// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_TRACK_BORISSOLENOID_H
#define XTRACK_TRACK_BORISSOLENOID_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/borissolenoid_src/solenoid_B_field_eval.h"

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
    const double shift_y
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

    const double q_coulomb = q0 * qe;

    const double L = length;
    const double ds = L / (double)n_steps;
    const double half_ds = 0.5 * ds;

    const double s_entry = LocalParticle_get_s(part);
    double s_local = 0.0;

    for (int istep = 0; istep < n_steps; ++istep) {
        const double ptau = LocalParticle_get_ptau(part);
        const double delta = LocalParticle_get_delta(part);
        const double energy = (energy0 + ptau * p0c_ev) * mass_ratio;
        const double gamma = energy / mass;
        const double P = P0 * (1.0 + delta);

        double tmp = P * P - px * px - py * py;
        if (tmp < 0.0) tmp = 0.0;
        double ps = sqrt(tmp);
        if (ps == 0.0) {
            break;
        }

        const double inv_ps = 1.0 / ps;

        const double xh = x + (px * inv_ps) * half_ds;
        const double yh = y + (py * inv_ps) * half_ds;
        const double s_local_h = s_local + half_ds;
        const double z_eval = s_entry + s_local_h;

        double dt = half_ds * inv_ps * gamma * mass_kg;

        double Bx;
        double By;
        double Bz;

        evaluate_solenoid_B(
            xh - shift_x,
            yh - shift_y,
            z_eval,
            L_coil,
            a,
            B0,
            z0,
            &Bx,
            &By,
            &Bz
        );

        const double half_qds = q_coulomb * half_ds;

        double pxm = px - half_qds * By;
        double pym = py + half_qds * Bx;

        tmp = P * P - pxm * pxm - pym * pym;
        if (tmp < 0.0) tmp = 0.0;
        double ps_mid = sqrt(tmp);
        if (ps_mid == 0.0) {
            break;
        }

        double t = q_coulomb * Bz * half_ds / ps_mid;
        double t2 = t * t;
        double inv_den = 1.0 / (1.0 + t2);

        double sR = 2.0 * t * inv_den;
        double c0 = (1.0 - t2) * inv_den;

        double pxp = c0 * pxm + sR * pym;
        double pyp = -sR * pxm + c0 * pym;

        double px1 = pxp - half_qds * By;
        double py1 = pyp + half_qds * Bx;

        tmp = P * P - px1 * px1 - py1 * py1;
        if (tmp < 0.0) tmp = 0.0;
        double ps1 = sqrt(tmp);
        if (ps1 == 0.0) {
            break;
        }
        double inv_ps1 = 1.0 / ps1;

        x = xh + (px1 * inv_ps1) * half_ds;
        y = yh + (py1 * inv_ps1) * half_ds;
        s_local += ds;

        dt += half_ds * inv_ps1 * gamma * mass_kg;

        px = px1;
        py = py1;

        zeta += (ds - dt * c * beta0);
    }

    LocalParticle_set_x(part, x);
    LocalParticle_set_y(part, y);
    LocalParticle_add_to_s(part, L);
    LocalParticle_set_px(part, px / P0);
    LocalParticle_set_py(part, py / P0);
    LocalParticle_set_zeta(part, zeta);
}

#endif // XTRACK_TRACK_BORISSOLENOID_H
