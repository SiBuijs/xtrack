#ifndef XTRACK_TRACK_BFIELDEXPANSION_BORIS_H
#define XTRACK_TRACK_BFIELDEXPANSION_BORIS_H

#include "track_bfieldexpansion_helpers.h"
#include "track_bfieldexpansion_straight.h"
#include "track_bfieldexpansion_bent.h"
#include "track_bfieldexpansion_radiation.h"

/* Spatial Boris integrator for BFieldExpansion elements (straight frame;
 * the curved-frame step is described below bfieldexpansion_boris_step).
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
 * step by bfieldexpansion_radiation_and_spin (track_bfieldexpansion_radiation.h),
 * with the field of the step midpoint. The Boris state already holds the
 * kinetic momenta, so it is handed over with ax = ay = 0; kx, ky, delta and
 * rvv are read back afterwards. Like Magnet, neither is applied when
 * backtracking.
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

/* Curved frame (constant h, x pointing outward, the reference bends towards
 * -x for h > 0). The equations of motion become
 *     dx/ds  = (1 + h x) kx/ks,               dy/ds  = (1 + h x) ky/ks,
 *     dkx/ds = chi (1 + h x) (ky Bs/ks - By) + h ks,
 *     dky/ds = chi (1 + h x) (Bx - kx Bs/ks),
 * with B in curved-frame components. The step has the same structure as in
 * the straight frame, but the half drifts are exact sector bends in the
 * design field b0 = h, i.e. B0 = h P0/q, which keeps the reference particle
 * on the frame. The half kicks only apply By - b0 (and Bx), and the factor
 * 1 + h x (the lab path per unit s) multiplies the kicks and the rotation.
 * The sector bend is exact for any delta and chi; only the field that differs
 * from the design dipole is split off. */

GPUFUN
double bfieldexpansion_sinc(const double u) {
    if (fabs(u) < 1e-4) return 1.0 - u * u / 6.0;
    return sin(u) / u;
}

/* Exact field-free motion over ell in the curved frame (polar drift), on
 * kinetic momenta. Writes the signed lab path length to *l_path. */
GPUFUN
int bfieldexpansion_polar_drift(
    const double h, const double ell, const double delta1,
    double *x, double *y, double *kx, const double ky, double *l_path)
{
    double rad = delta1 * delta1 - (*kx) * (*kx) - ky * ky;
    if (!(rad > 0.0)) return -1;
    const double ks = sqrt(rad);
    const double phi = h * ell;
    const double kx_new = (*kx) * cos(phi) + ks * sin(phi);
    const double ks_new = ks * cos(phi) - (*kx) * sin(phi);
    if (!(ks_new > 0.0)) return -1;
    const double sin_over_h = ell * bfieldexpansion_sinc(phi);
    const double half_sinc = bfieldexpansion_sinc(0.5 * phi);
    const double one_m_cos_over_h = 0.5 * ell * phi * half_sinc * half_sinc;
    const double q = 1.0 + h * (*x);
    *x = ((*x) * ks + (*kx) * sin_over_h + ks * one_m_cos_over_h) / ks_new;
    *y += q * sin_over_h * ky / ks_new;
    *l_path = q * sin_over_h * delta1 / ks_new;
    *kx = kx_new;
    return 0;
}

/* Exact motion over ell in the uniform vertical field k0_chi = chi*b0 in the
 * curved frame (sector bend), on kinetic momenta. Same formulas as
 * track_curved_exact_bend_single_particle (track_magnet_drift.h). Writes the
 * signed lab path length to *l_path. */
GPUFUN
int bfieldexpansion_sector_bend(
    const double h, const double k0_chi, const double ell, const double delta1,
    double *x, double *y, double *kx, const double ky, double *l_path)
{
    if (fabs(k0_chi) < 1e-8) {
        return bfieldexpansion_polar_drift(h, ell, delta1, x, y, kx, ky, l_path);
    }
    const double hs = h * ell;
    const double sin_hs = sin(hs);
    const double cos_hs = cos(hs);
    const double sin_hs_2 = sin(0.5 * hs);

    double rad = delta1 * delta1 - (*kx) * (*kx) - ky * ky;
    if (!(rad > 0.0)) return -1;
    const double pz0 = sqrt(rad);
    const double px0 = *kx;
    const double C = pz0 - k0_chi * (1.0 / h + (*x));
    const double pxs = px0 * cos_hs + C * sin_hs;
    rad = delta1 * delta1 - pxs * pxs - ky * ky;
    if (!(rad > 0.0)) return -1;
    const double pzs = sqrt(rad);

    const double delta_pz = (px0 - pxs) * (px0 + pxs) / (pz0 + pzs);
    const double delta_D = -2.0 * C * sin_hs_2 * sin_hs_2 - px0 * sin_hs;
    const double delta_px = -2.0 * px0 * sin_hs_2 * sin_hs_2 + C * sin_hs;
    const double delta_a = atan2(px0 * delta_pz - pz0 * delta_px,
                                 pz0 * pzs + px0 * pxs);
    const double integ = (hs + delta_a) / k0_chi;

    *x += (delta_pz - delta_D) / k0_chi;
    *y += ky * integ;
    *kx = pxs;
    *l_path = delta1 * integ;
    return 0;
}

/* Returns 0 on success, -1 if the longitudinal momentum vanishes. */
GPUFUN
int bfieldexpansion_boris_step_bent(
    Expansion *f, const double chi, const double delta1, const double rvv,
    const double s, const double ds,
    double *x, double *kx, double *y, double *ky, double *zeta,
    FieldValue *v_mid)
{
    const double h = f->h;
    const double b0 = h;  /* design field, normalized to (B rho)_0 */
    const double k0_chi = chi * b0;
    const double hds = 0.5 * ds;
    const double delta1_sq = delta1 * delta1;
    double l1, l2;

    /* First half sector bend */
    if (bfieldexpansion_sector_bend(h, k0_chi, hds, delta1,
                                    x, y, kx, *ky, &l1) != 0) return -1;

    /* Field at the midpoint */
    if (evaluate_expansion_bent(f, *x, *y, s + hds, v_mid) != 0) return -1;
    const double q = 1.0 + h * (*x);
    const double dBy = v_mid->By - b0;
    const double Bx = v_mid->Bx;
    const double Bs = v_mid->Bs;

    /* First half transverse kick, design field removed */
    double kxm = *kx - chi * q * hds * dBy;
    double kym = *ky + chi * q * hds * Bx;

    /* Rotation about Bs (Cayley form of the rotation by 2*atan(t)) */
    const double rad = delta1_sq - kxm * kxm - kym * kym;
    if (!(rad > 0.0)) return -1;
    const double t = chi * q * Bs * hds / sqrt(rad);
    const double inv_den = 1.0 / (1.0 + t * t);
    const double sn = 2.0 * t * inv_den;
    const double cs = (1.0 - t * t) * inv_den;
    const double kxp =  cs * kxm + sn * kym;
    const double kyp = -sn * kxm + cs * kym;

    /* Second half transverse kick */
    *kx = kxp - chi * q * hds * dBy;
    *ky = kyp + chi * q * hds * Bx;

    /* Second half sector bend */
    if (bfieldexpansion_sector_bend(h, k0_chi, hds, delta1,
                                    x, y, kx, *ky, &l2) != 0) return -1;

    *zeta += ds - (l1 + l2) / rvv;
    return 0;
}

GPUFUN
int bfieldexpansion_boris_evaluate(Expansion *f, double x, double y, double s,
                                   FieldValue *out) {
    if (f->straight) return evaluate_expansion_straight(f, x, y, s, out);
    return evaluate_expansion_bent(f, x, y, s, out);
}

GPUFUN
void BFieldExpansion_track_boris(
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
        valid = bfieldexpansion_boris_evaluate(&f, x, y, s_start, &v) == 0;
        if (pkin_const) {
            kx = LocalParticle_get_px(part) - LocalParticle_get_ax(part);
            ky = LocalParticle_get_py(part) - LocalParticle_get_ay(part);
        }
        else {
            kx = LocalParticle_get_px(part) - chi * v.Ax;
            ky = LocalParticle_get_py(part) - chi * v.Ay;
        }

        #ifdef BFIELDEXPANSION_WITH_SYNRAD_AND_SPIN
        const int apply_hook = bfieldexpansion_radiation_or_spin_active(
            part, radiation_flag, ds);
        const double h_frame = f.straight ? 0. : f.h;
        BFieldExpansionRadiationExit rad_exit = {0., 0., 0.};
        #endif

        double s = s_start;
        for (int64_t step = 0; valid && step < nstep; ++step) {
            FieldValue v_mid;
            #ifdef BFIELDEXPANSION_WITH_SYNRAD_AND_SPIN
            const double kx_old = kx, ky_old = ky, zeta_old = zeta;
            #endif
            const int status = f.straight
                ? bfieldexpansion_boris_step(&f, chi, delta1, rvv, s, ds,
                      &x, &kx, &y, &ky, &zeta, &v_mid)
                : bfieldexpansion_boris_step_bent(&f, chi, delta1, rvv, s, ds,
                      &x, &kx, &y, &ky, &zeta, &v_mid);
            if (status != 0) {
                valid = 0;
                break;
            }
            s += ds;

            #ifdef BFIELDEXPANSION_WITH_SYNRAD_AND_SPIN
            if (apply_hook) {
                /* Kinetic momenta, so ax = ay = 0; the exit convention is
                 * applied after the loop. */
                bfieldexpansion_radiation_and_spin(part, &v_mid, h_frame, ds,
                    zeta - zeta_old, x, y, zeta, kx, ky, 0., 0.,
                    0.5 * (kx_old + kx), 0.5 * (ky_old + ky), delta1 - 1.0,
                    radiation_flag, radiation_record, &rad_exit);
                if (radiation_flag) {
                    kx = LocalParticle_get_px(part);
                    ky = LocalParticle_get_py(part);
                    delta1 = 1.0 + LocalParticle_get_delta(part);
                    rvv = LocalParticle_get_rvv(part);
                }
            }
            #endif
        }

        if (valid) valid = bfieldexpansion_boris_evaluate(&f, x, y, s, &v) == 0;
        if (!valid) {
            /* Transverse kinetic momentum reached the total momentum, or
             * the curved chart is singular (1 + h x = 0). Leave the last
             * committed particle coordinates intact. */
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
