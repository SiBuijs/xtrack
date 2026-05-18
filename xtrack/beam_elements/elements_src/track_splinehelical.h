// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_TRACK_SPLINEHELICAL_H
#define XTRACK_TRACK_SPLINEHELICAL_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/splineboris_src/spline_B_field_eval.h"
#ifndef XTRACK_MULTIPOLE_NO_SYNRAD
GPUFUN uint32_t RandomUniformUInt32_generate(LocalParticle* part);
GPUFUN double RandomUniform_generate(LocalParticle* part);
GPUFUN double RandomExponential_generate(LocalParticle* part);
#include "xtrack/beam_elements/elements_src/track_magnet_radiation.h"
#endif

GPUFUN
void SplineHelical_single_particle(
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
    if (LocalParticle_get_state(part) <= 0){
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
    const double q_coulomb = q0 * qe;

    double px = px_r * P0;
    double py = py_r * P0;

    const double L = length;
    const double ds = L / (double)n_steps;
    const double half_ds = 0.5 * ds;
    double s_local = 0.0;

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
        if (radiation_flag && ds > 0) {
            const double ax_old = LocalParticle_get_ax(part);
            const double ay_old = LocalParticle_get_ay(part);
            old_kin_px = (px / P0) - ax_old;
            old_kin_py = (py / P0) - ay_old;
        }
        #endif

        double tmp = P * P - px * px - py * py;
        if (tmp < 0.0) tmp = 0.0;
        const double ps = sqrt(tmp);
        if (ps == 0.0) break;

        const double inv_ps = 1.0 / ps;
        const double xh = x + (px * inv_ps) * half_ds;
        const double yh = y + (py * inv_ps) * half_ds;
        const double s_local_h = s_local + half_ds;

        double Bx, By, Bs;
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

        const double px_in = px;
        const double py_in = py;
        const double x_in = x;
        const double y_in = y;
        const double pz_in = ps;

        // Apply the explicit helical map in (x, y, px, py) using B at mid-step.
        // Coefficients are written with phi=q*Bz*ds/pz to avoid catastrophic
        // scaling from direct 1/(q*Bz^2) factors.
        if (fabs(Bs) > 1e-16 && fabs(q_coulomb * Bs) > 1e-24) {
            const double phi = q_coulomb * Bs * ds / pz_in;
            const double cph = cos(phi);
            const double sph = sin(phi);
            const double phi2 = phi * phi;

            double sin_over_phi;
            double omc_over_phi;
            double phi_minus_sin_over_phi2;
            double omc_over_phi2;
            if (fabs(phi) < 1e-8) {
                const double phi3 = phi2 * phi;
                const double phi4 = phi2 * phi2;
                sin_over_phi = 1.0 - phi2 / 6.0 + phi4 / 120.0;
                omc_over_phi = 0.5 * phi - phi3 / 24.0;
                phi_minus_sin_over_phi2 = phi / 6.0 - phi3 / 120.0;
                omc_over_phi2 = 0.5 - phi2 / 24.0;
            } else {
                sin_over_phi = sph / phi;
                omc_over_phi = (1.0 - cph) / phi;
                phi_minus_sin_over_phi2 = (phi - sph) / phi2;
                omc_over_phi2 = (1.0 - cph) / phi2;
            }

            const double cos_minus_one_over_phi = -omc_over_phi;
            const double ds_over_pz = ds / pz_in;
            const double qds2_over_pz = q_coulomb * ds * ds / pz_in;
            const double qds = q_coulomb * ds;

            x = x_in
                + ds_over_pz * (sin_over_phi * px_in + omc_over_phi * py_in)
                + qds2_over_pz * (-By * omc_over_phi2 + Bx * phi_minus_sin_over_phi2);

            y = y_in
                + ds_over_pz * (cos_minus_one_over_phi * px_in + sin_over_phi * py_in)
                + qds2_over_pz * (By * phi_minus_sin_over_phi2 + Bx * omc_over_phi2);

            px = cph * px_in
                + sph * py_in
                + qds * (-By * sin_over_phi + Bx * omc_over_phi);

            py = -sph * px_in
                + cph * py_in
                + qds * (By * cos_minus_one_over_phi + Bx * sin_over_phi);
        } else {
            // Bz -> 0 fallback: first-order transverse kick plus trapezoidal drift.
            const double px_k = px_in - q_coulomb * By * ds;
            const double py_k = py_in + q_coulomb * Bx * ds;
            double tmp_out = P * P - px_k * px_k - py_k * py_k;
            if (tmp_out < 0.0) tmp_out = 0.0;
            const double ps_out = sqrt(tmp_out);
            if (ps_out == 0.0) break;
            const double inv_ps_out = 1.0 / ps_out;

            x = x_in + 0.5 * ds * (px_in * inv_ps + px_k * inv_ps_out);
            y = y_in + 0.5 * ds * (py_in * inv_ps + py_k * inv_ps_out);
            px = px_k;
            py = py_k;
        }

        tmp = P * P - px * px - py * py;
        if (tmp < 0.0) tmp = 0.0;
        const double ps1 = sqrt(tmp);
        if (ps1 == 0.0) break;

        s_local += ds;
        const double dt = 0.5 * ds * (inv_ps + (1.0 / ps1)) * gamma * mass_kg;
        zeta += (ds - dt * c * beta0);

        #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        const double rvv = LocalParticle_get_rvv(part);
        const double l_path = rvv * dt * c * beta0;
        magnet_spin(part, Bx, By, Bs, 0.0, ds, l_path);

        if (radiation_flag && ds > 0) {
            const double ax_new = LocalParticle_get_ax(part);
            const double ay_new = LocalParticle_get_ay(part);
            const double new_kin_px = (px / P0) - ax_new;
            const double new_kin_py = (py / P0) - ay_new;
            const double mean_kin_px = 0.5 * (old_kin_px + new_kin_px);
            const double mean_kin_py = 0.5 * (old_kin_py + new_kin_py);
            const double B_perp_T = compute_b_perp_mod(
                mean_kin_px, mean_kin_py, LocalParticle_get_delta(part), Bx, By, Bs
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

#endif // XTRACK_TRACK_SPLINEHELICAL_H
