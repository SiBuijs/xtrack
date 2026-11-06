#ifndef XTRACK_TRACK_BFIELDEXPANSION_RADIATION_H
#define XTRACK_TRACK_BFIELDEXPANSION_RADIATION_H

/* Spin and synchrotron radiation for the BFieldExpansion integrators (RK4 and
 * Boris), unless compiled out. Both integrators call
 * bfieldexpansion_radiation_and_spin after each integration step, as Magnet
 * does after each of its kicks. The physics is the one of Magnet
 * (magnet_spin, magnet_radiation); only the field and the path length of the
 * step come from the integrator. */

#include "track_bfieldexpansion_helpers.h"

#if !defined(XTRACK_MULTIPOLE_NO_SYNRAD) && !defined(XTRACK_TPSA_TRACK)
#define BFIELDEXPANSION_WITH_SYNRAD_AND_SPIN
#include "xtrack/beam_elements/elements_src/track_magnet_radiation.h"

/* Accumulators for the average radiation model ("same as first" mode), kept
 * over the steps of one element. */
typedef struct {
    double dp;
    double dpx;
    double dpy;
} BFieldExpansionRadiationExit;

/* Whether the step hook has anything to do for this particle. Spin is tracked
 * when the particle carries a spin; neither spin nor radiation is applied when
 * backtracking (ds < 0), as in Magnet. */
GPUFUN
int bfieldexpansion_radiation_or_spin_active(LocalParticle* part,
                                             const int64_t radiation_flag,
                                             const double ds) {
    const int with_spin = (LocalParticle_get_spin_x(part) != 0.
                           || LocalParticle_get_spin_y(part) != 0.
                           || LocalParticle_get_spin_z(part) != 0.);
    return (radiation_flag || with_spin) && ds > 0;
}

/* Apply spin and synchrotron radiation for one integration step.
 *
 * The integrator state at the end of the step (x, y, zeta, px, py, with the
 * vector potential ax, ay the momenta px, py include) is written to the
 * particle, so that magnet_spin and magnet_radiation act on the kinetic
 * momenta px - ax, py - ay. They get
 *   - v_mid: the normalized field at the step midpoint (converted to tesla),
 *   - the path length of the step, rvv * (ds - dzeta),
 *   - h_frame: the frame curvature (0 in a straight frame), for the rotation
 *     of the spin components with the frame,
 *   - kx_mid, ky_mid, delta_mid: the kinetic momenta and delta at the step
 *     midpoint, for the field component perpendicular to the motion.
 * Afterwards the caller reads px, py and the energy variables back from the
 * particle, so that the radiation kick and the energy loss reach the next
 * step. */
GPUFUN
void bfieldexpansion_radiation_and_spin(
    LocalParticle* part,
    const FieldValue* v_mid,
    const double h_frame,
    const double ds,
    const double dzeta,
    const double x, const double y, const double zeta,
    const double px, const double py, const double ax, const double ay,
    const double kx_mid, const double ky_mid, const double delta_mid,
    const int64_t radiation_flag,
    SynchrotronRadiationRecordData radiation_record,
    BFieldExpansionRadiationExit* rad_exit)
{
    /* Normalized field -> tesla */
    const double brho0 = LocalParticle_get_p0c(part) / C_LIGHT
                         / LocalParticle_get_q0(part);
    const double Bx_T = v_mid->Bx * brho0;
    const double By_T = v_mid->By * brho0;
    const double Bs_T = v_mid->Bs * brho0;
    /* delta is constant within a step (it only changes here), so the
     * particle's rvv is the one of the step. */
    const double l_path = LocalParticle_get_rvv(part) * (ds - dzeta);

    LocalParticle_set_x(part, x);
    LocalParticle_set_y(part, y);
    LocalParticle_set_zeta(part, zeta);
    LocalParticle_set_px(part, px);
    LocalParticle_set_py(part, py);
    LocalParticle_set_ax(part, ax);
    LocalParticle_set_ay(part, ay);

    /* magnet_spin does nothing for a particle without spin */
    magnet_spin(part, Bx_T, By_T, Bs_T, h_frame, ds, l_path);

    if (radiation_flag) {
        const double B_perp_T = compute_b_perp_mod(
            kx_mid, ky_mid, delta_mid, Bx_T, By_T, Bs_T);
        magnet_radiation(part, B_perp_T, ds, l_path,
            radiation_flag, radiation_record,
            &rad_exit->dp, &rad_exit->dpx, &rad_exit->dpy);
    }
}

#endif /* !XTRACK_MULTIPOLE_NO_SYNRAD && !XTRACK_TPSA_TRACK */

#endif /* XTRACK_TRACK_BFIELDEXPANSION_RADIATION_H */
