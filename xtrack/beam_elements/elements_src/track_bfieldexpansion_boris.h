#ifndef XTRACK_TRACK_BFIELDEXPANSION_BORIS_H
#define XTRACK_TRACK_BFIELDEXPANSION_BORIS_H

#include "track_bfieldexpansion_helpers.h"
#include "track_bfieldexpansion_straight.h"
#if !defined(XTRACK_MULTIPOLE_NO_SYNRAD) && !defined(XTRACK_TPSA_TRACK)
    #define BFIELDEXPANSION_BORIS_WITH_SYNRAD_AND_SPIN
    #include "xtrack/beam_elements/elements_src/track_magnet_radiation.h"
#endif

/* Spatial Boris integrator for straight BFieldExpansion elements.
 *
 * The particle is advanced in s using the kinetic transverse momenta
 * kx = px - chi*Ax, ky = py - chi*Ay (in units of the reference momentum)
 * and the normalized magnetic field B/(B rho)_0 of the expansion. The vector
 * potential is only used at the element boundaries, with the same boundary
 * conventions (pkin_const) as the RK4 integrator.
 *
 * In a straight frame the equations of motion are
 *     dx/ds  = kx/ks,                         dy/ds  = ky/ks,
 *     dkx/ds = chi*(ky*Bs/ks - By),           dky/ds = chi*(Bx - kx*Bs/ks),
 *     dzeta/ds = 1 - (1 + delta)/(rvv*ks),
 * with ks = sqrt((1 + delta)^2 - kx^2 - ky^2). Each step is
 *     half drift - half transverse kick - rotation about Bs
 *     - half transverse kick - half drift,
 * with the field evaluated once at the step midpoint. The scheme is second
 * order, time-reversible (backtracking with -ds retraces the forward step)
 * and keeps kx^2 + ky^2 unchanged in the rotation.
 *
 * Spin and synchrotron radiation (unless compiled out) are applied after each
 * step, as in Magnet: the shared helpers magnet_spin and magnet_radiation get
 * the midpoint field in tesla and the path length of the step, and act on the
 * particle's kinetic momenta. The integrator state is written to the particle
 * before and read back after them, so that the radiation kick on px, py and
 * the energy loss both reach the next step. Like Magnet, neither is applied
 * when backtracking.
 */

/* Returns 0 on success, -1 if the longitudinal momentum vanishes. */
GPUFUN
int bfieldexpansion_boris_step(
    Expansion *f, const double chi, const double delta1, const double rvv,
    const double s, const double ds,
    double *x, double *kx, double *y, double *ky, double *zeta,
    FieldValue *v_mid)
{
    const double hds = 0.5 * ds;
    const double delta1_sq = delta1 * delta1;

    /* First half drift */
    double rad = delta1_sq - (*kx) * (*kx) - (*ky) * (*ky);
    if (!(rad > 0.0)) return -1;
    double ks = sqrt(rad);
    const double xh = *x + (*kx / ks) * hds;
    const double yh = *y + (*ky / ks) * hds;
    double dzeta = hds * (1.0 - delta1 / (rvv * ks));

    /* Field at the midpoint */
    if (evaluate_expansion_straight(f, xh, yh, s + hds, v_mid) != 0) return -1;
    const double Bx = v_mid->Bx;
    const double By = v_mid->By;
    const double Bs = v_mid->Bs;

    /* First half transverse kick */
    double kxm = *kx - chi * hds * By;
    double kym = *ky + chi * hds * Bx;

    /* Rotation about Bs (Cayley form of the rotation by 2*atan(t)) */
    rad = delta1_sq - kxm * kxm - kym * kym;
    if (!(rad > 0.0)) return -1;
    const double t = chi * Bs * hds / sqrt(rad);
    const double inv_den = 1.0 / (1.0 + t * t);
    const double sn = 2.0 * t * inv_den;
    const double cs = (1.0 - t * t) * inv_den;
    const double kxp =  cs * kxm + sn * kym;
    const double kyp = -sn * kxm + cs * kym;

    /* Second half transverse kick */
    const double kx1 = kxp - chi * hds * By;
    const double ky1 = kyp + chi * hds * Bx;

    /* Second half drift */
    rad = delta1_sq - kx1 * kx1 - ky1 * ky1;
    if (!(rad > 0.0)) return -1;
    ks = sqrt(rad);
    *x = xh + (kx1 / ks) * hds;
    *y = yh + (ky1 / ks) * hds;
    dzeta += hds * (1.0 - delta1 / (rvv * ks));

    *kx = kx1;
    *ky = ky1;
    *zeta += dzeta;
    return 0;
}

GPUFUN
void BFieldExpansion_track_boris_straight(
    BFieldExpansionData el,
    LocalParticle* part0,
    const double length,
    double s_start,
    const int64_t nstep,
    const int64_t radiation_flag,
    SynchrotronRadiationRecordData radiation_record)
{
    double ds = length / nstep;

    Expansion f;
    BFieldExpansionData_init_expansion(el, &f);

    int64_t const backtrack = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);
    if (backtrack) {s_start += length; ds = -ds;}

    const int pkin_const = BFieldExpansionData_get_pkin_const(el);

    START_PER_PARTICLE_BLOCK(part0, part);
        FieldValue v;
        int valid = 1;
        const double chi    = LocalParticle_get_chi(part);
        double delta1 = 1.0 + LocalParticle_get_delta(part);
        double rvv    = LocalParticle_get_rvv(part);

        double x    = LocalParticle_get_x(part);
        double y    = LocalParticle_get_y(part);
        double zeta = LocalParticle_get_zeta(part);
        double kx, ky;

        /* Entry: kinetic momenta, consistent with the RK4 boundary handling.
         * pkin_const keeps the kinetic momentum carried by the particle;
         * otherwise the canonical momentum is kept and the element's own
         * vector potential at the entrance defines the kinetic momentum. */
        valid = evaluate_expansion_straight(&f, x, y, s_start, &v) == 0;
        if (pkin_const) {
            kx = LocalParticle_get_px(part) - LocalParticle_get_ax(part);
            ky = LocalParticle_get_py(part) - LocalParticle_get_ay(part);
        }
        else {
            kx = LocalParticle_get_px(part) - chi * v.Ax;
            ky = LocalParticle_get_py(part) - chi * v.Ay;
        }

        #ifdef BFIELDEXPANSION_BORIS_WITH_SYNRAD_AND_SPIN
        /* Normalized field -> tesla; spin needs the field even without radiation. */
        const double brho0 = LocalParticle_get_p0c(part) / C_LIGHT
                             / LocalParticle_get_q0(part);
        const int with_spin = (LocalParticle_get_spin_x(part) != 0.
                               || LocalParticle_get_spin_y(part) != 0.
                               || LocalParticle_get_spin_z(part) != 0.);
        const int apply_hook = (radiation_flag || with_spin) && ds > 0;
        /* Only used for the average radiation model's "same as first" mode. */
        double dp_record_exit = 0., dpx_record_exit = 0., dpy_record_exit = 0.;
        #endif

        double s = s_start;
        for (int64_t step = 0; valid && step < nstep; ++step) {
            FieldValue v_mid;
            #ifdef BFIELDEXPANSION_BORIS_WITH_SYNRAD_AND_SPIN
            const double kx_old = kx, ky_old = ky, zeta_old = zeta;
            #endif
            if (bfieldexpansion_boris_step(&f, chi, delta1, rvv, s, ds,
                    &x, &kx, &y, &ky, &zeta, &v_mid) != 0) {
                valid = 0;
                break;
            }
            s += ds;

            #ifdef BFIELDEXPANSION_BORIS_WITH_SYNRAD_AND_SPIN
            if (apply_hook) {
                const double Bx_T = v_mid.Bx * brho0;
                const double By_T = v_mid.By * brho0;
                const double Bs_T = v_mid.Bs * brho0;
                const double l_path = rvv * (ds - (zeta - zeta_old));

                /* Hand the state over with kinetic momenta (ax = ay = 0);
                 * the exit convention is applied after the loop. */
                LocalParticle_set_x(part, x);
                LocalParticle_set_y(part, y);
                LocalParticle_set_zeta(part, zeta);
                LocalParticle_set_px(part, kx);
                LocalParticle_set_py(part, ky);
                LocalParticle_set_ax(part, 0.);
                LocalParticle_set_ay(part, 0.);

                if (with_spin) {
                    magnet_spin(part, Bx_T, By_T, Bs_T, 0., ds, l_path);
                }
                if (radiation_flag) {
                    const double B_perp_T = compute_b_perp_mod(
                        0.5 * (kx_old + kx), 0.5 * (ky_old + ky),
                        delta1 - 1.0, Bx_T, By_T, Bs_T);
                    magnet_radiation(part, B_perp_T, ds, l_path,
                        radiation_flag, radiation_record,
                        &dp_record_exit, &dpx_record_exit, &dpy_record_exit);
                    kx = LocalParticle_get_px(part);
                    ky = LocalParticle_get_py(part);
                    delta1 = 1.0 + LocalParticle_get_delta(part);
                    rvv = LocalParticle_get_rvv(part);
                }
            }
            #endif
        }

        if (valid) valid = evaluate_expansion_straight(&f, x, y, s, &v) == 0;
        if (!valid) {
            /* Transverse kinetic momentum reached the total momentum.
             * Leave the last committed particle coordinates intact. */
            LocalParticle_set_state(part, XT_INVALID_BFIELD_EXPANSION);
        }
        else {
            if (pkin_const) {
                LocalParticle_set_px(part, kx);
                LocalParticle_set_py(part, ky);
                LocalParticle_set_ax(part, 0);
                LocalParticle_set_ay(part, 0);
            }
            else {
                LocalParticle_set_px(part, kx + chi * v.Ax);
                LocalParticle_set_py(part, ky + chi * v.Ay);
                LocalParticle_set_ax(part, chi * v.Ax);
                LocalParticle_set_ay(part, chi * v.Ay);
            }
            LocalParticle_set_x(part, x);
            LocalParticle_set_y(part, y);
            LocalParticle_set_zeta(part, zeta);
            LocalParticle_add_to_s(part, ds * nstep);
        }
    END_PER_PARTICLE_BLOCK
}

#endif
