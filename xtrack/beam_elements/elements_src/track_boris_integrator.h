// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_TRACK_BORIS_INTEGRATOR_H
#define XTRACK_TRACK_BORIS_INTEGRATOR_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/splineboris_src/spline_B_field_eval.h"
#include "xtrack/beam_elements/elements_src/wiggler_B_field_eval.h"
#include "xtrack/beam_elements/elements_src/track_magnet_kick.h"
#ifndef XTRACK_MULTIPOLE_NO_SYNRAD
GPUFUN uint32_t RandomUniformUInt32_generate(LocalParticle* part);
GPUFUN double RandomUniform_generate(LocalParticle* part);
GPUFUN double RandomExponential_generate(LocalParticle* part);
#include "xtrack/beam_elements/elements_src/track_magnet_radiation.h"
#endif

#define BORIS_FIELD_SPLINE  0
#define BORIS_FIELD_WIGGLER 1

GPUFUN
void boris_track_single_particle(
    LocalParticle* part,
    const int field_type,
    const double length,
    const int n_steps,
    // Spline field parameters (field_type == BORIS_FIELD_SPLINE)
    const double bs[5],
    const double* const *by,
    const double* const *bx,
    const int multipole_order,
    const double shift_x,
    const double shift_y,
    const double scale_b,
    const int64_t order,
    const double inv_factorial_order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    const int64_t radiation_flag,
    void* radiation_record,
    // Wiggler field parameters (field_type == BORIS_FIELD_WIGGLER)
    const double k_u,
    const double b_tilde,
    const double s_offset
){

    if (LocalParticle_get_state(part) <= 0){
        return;
    }

    const double c      = C_LIGHT;
    const double qe     = QELEM;

    const double q0     = LocalParticle_get_q0(part);
    const double mass0  = LocalParticle_get_mass0(part);
    const double p0c_ev = LocalParticle_get_p0c(part);
    const double beta0  = LocalParticle_get_beta0(part);

    double x    = LocalParticle_get_x(part);
    double y    = LocalParticle_get_y(part);
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

    const double L    = length;
    const int8_t backtrack = LocalParticle_check_track_flag(part, XS_FLAG_BACKTRACK);
    const double ds   = (backtrack ? -L : L) / (double) n_steps;
    const double half_ds = 0.5 * ds;

    double s_local = backtrack ? L : 0.0;

    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
    double old_kin_px = 0.0, old_kin_py = 0.0;
    double dp_record_exit = 0.0, dpx_record_exit = 0.0, dpy_record_exit = 0.0;
    #endif

    for (int istep = 0; istep < n_steps; ++istep) {

        const double ptau = LocalParticle_get_ptau(part);
        const double delta = LocalParticle_get_delta(part);
        const double energy = (energy0 + ptau * p0c_ev) * mass_ratio;
        const double gamma = energy / mass;

        const double P = P0 * (1.0 + delta);

        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (field_type == BORIS_FIELD_SPLINE && radiation_flag && ds > 0) {
            double ax_old = LocalParticle_get_ax(part);
            double ay_old = LocalParticle_get_ay(part);
            old_kin_px = (px / P0) - ax_old;
            old_kin_py = (py / P0) - ay_old;
        }
        #endif

        double tmp  = P * P - px * px - py * py;
        if (tmp < 0.0) tmp = 0.0;
        double ps   = sqrt(tmp);
        if (ps == 0.0) {
            break;
        }

        const double inv_ps  = 1.0 / ps;

        const double xh = x + (px * inv_ps) * half_ds;
        const double yh = y + (py * inv_ps) * half_ds;
        const double s_local_h = s_local + half_ds;

        double dt = half_ds * inv_ps * gamma * mass_kg;

        double Bx;
        double By;
        double Bs;

        if (field_type == BORIS_FIELD_WIGGLER) {
            evaluate_wiggler_B(
                xh,
                yh,
                s_offset + s_local_h,
                k_u,
                b_tilde,
                &Bx,
                &By,
                &Bs
            );
        } else {
            evaluate_B(
                xh - shift_x,
                yh - shift_y,
                s_local_h,
                bs,
                by,
                bx,
                L,
                multipole_order,
                &Bx,
                &By,
                &Bs
            );

            Bx *= scale_b;
            By *= scale_b;
            Bs *= scale_b;

            double Bx_mp = 0.0;
            double By_mp = 0.0;
            double Bs_mp = 0.0;

            if (order >= 0 && L != 0.0) {
                evaluate_field_from_strengths(
                    p0c_ev,
                    q0,
                    xh - shift_x,
                    yh - shift_y,
                    L,
                    order,
                    inv_factorial_order,
                    knl,
                    ksl,
                    -1,
                    1.0,
                    NULL,
                    NULL,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    &Bx_mp,
                    &By_mp,
                    &Bs_mp);

                Bx += Bx_mp;
                By += By_mp;
                Bs += Bs_mp;
            }
        }

        const double half_qds = q_coulomb * half_ds;

        double pxm = px - half_qds * By;
        double pym = py + half_qds * Bx;

        tmp = P * P - pxm * pxm - pym * pym;
        if (tmp < 0.0) tmp = 0.0;
        double ps_mid = sqrt(tmp);
        if (ps_mid == 0.0) {
            break;
        }

        double t  = q_coulomb * Bs * half_ds / ps_mid;
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

        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (field_type == BORIS_FIELD_SPLINE) {
            double const rvv = LocalParticle_get_rvv(part);
            double const l_path = rvv * dt * c * beta0;

            magnet_spin(part, Bx, By, Bs, 0.0, ds, l_path);

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
                    Bx, By, Bs
                );

                magnet_radiation(
                    part,
                    B_perp_T,
                    ds,
                    l_path,
                    radiation_flag,
                    (SynchrotronRadiationRecordData) radiation_record,
                    &dp_record_exit, &dpx_record_exit, &dpy_record_exit
                );
            }
        }
        #endif
    }

    LocalParticle_set_x(part, x);
    LocalParticle_set_y(part, y);

    LocalParticle_add_to_s(part, backtrack ? -L : L);

    LocalParticle_set_px(part, px / P0);
    LocalParticle_set_py(part, py / P0);

    LocalParticle_set_zeta(part, zeta);
}

#endif // XTRACK_TRACK_BORIS_INTEGRATOR_H
