// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_BORISSOLENOID_HELICAL_MAP_H
#define XTRACK_BORISSOLENOID_HELICAL_MAP_H

#include <math.h>

#ifndef GPUFUN
#define GPUFUN
#endif

#ifndef BORISSOLENOID_HELICAL_EPS
#define BORISSOLENOID_HELICAL_EPS 1e-30
#endif

#ifndef BORISSOLENOID_HELICAL_SINC_EPS
#define BORISSOLENOID_HELICAL_SINC_EPS 1e-14
#endif

// sin(theta)/theta, 1 at theta=0
GPUFUN
double borissolenoid_sinc(const double theta) {
    if (fabs(theta) < BORISSOLENOID_HELICAL_SINC_EPS) {
        return 1.0;
    }
    return sin(theta) / theta;
}

// (1 - cos(theta))/theta, 0 at theta=0
GPUFUN
double borissolenoid_vers_over_theta(const double theta) {
    if (fabs(theta) < BORISSOLENOID_HELICAL_SINC_EPS) {
        return 0.0;
    }
    return (1.0 - cos(theta)) / theta;
}

// (cos(theta) - 1)/theta, 0 at theta=0
GPUFUN
double borissolenoid_cos_minus_one_over_theta(const double theta) {
    if (fabs(theta) < BORISSOLENOID_HELICAL_SINC_EPS) {
        return 0.0;
    }
    return (cos(theta) - 1.0) / theta;
}

// Field-aligned rotation: v_zeta = R * v_lab.
// Returns 1 on success, 0 when B_perp is too small (caller uses identity).
GPUFUN
int borissolenoid_vec_to_zeta(
    const double Bx,
    const double By,
    const double Bz,
    const double B_mag,
    const double B_perp,
    const double vx,
    const double vy,
    const double vz,
    double* ox,
    double* oy,
    double* oz
) {
    if (B_mag < BORISSOLENOID_HELICAL_EPS) {
        *ox = vx;
        *oy = vy;
        *oz = vz;
        return 0;
    }

    const double inv_B_mag = 1.0 / B_mag;

    if (B_perp < BORISSOLENOID_HELICAL_EPS) {
        *ox = vx;
        *oy = vy;
        *oz = (Bx * vx + By * vy + Bz * vz) * inv_B_mag;
        return 0;
    }

    const double inv_B_perp = 1.0 / B_perp;
    const double Bz_over_B = Bz * inv_B_mag;
    const double Bz_over_B_perp = Bz_over_B * inv_B_perp;

    const double r00 = -By * inv_B_perp;
    const double r01 = Bx * inv_B_perp;
    const double r10 = -Bx * Bz_over_B_perp;
    const double r11 = -By * Bz_over_B_perp;
    const double r12 = B_perp * inv_B_mag;
    const double r20 = Bx * inv_B_mag;
    const double r21 = By * inv_B_mag;
    const double r22 = Bz * inv_B_mag;

    *ox = r00 * vx + r01 * vy;
    *oy = r10 * vx + r11 * vy + r12 * vz;
    *oz = r20 * vx + r21 * vy + r22 * vz;
    return 1;
}

// Inverse rotation: v_lab = R^T * v_zeta.
GPUFUN
int borissolenoid_vec_to_lab(
    const double Bx,
    const double By,
    const double Bz,
    const double B_mag,
    const double B_perp,
    const double vx,
    const double vy,
    const double vz,
    double* ox,
    double* oy,
    double* oz
) {
    if (B_mag < BORISSOLENOID_HELICAL_EPS) {
        *ox = vx;
        *oy = vy;
        *oz = vz;
        return 0;
    }

    const double inv_B_mag = 1.0 / B_mag;

    if (B_perp < BORISSOLENOID_HELICAL_EPS) {
        const double p_par = vz;
        *ox = vx;
        *oy = vy;
        *oz = Bz * p_par * inv_B_mag;
        return 0;
    }

    const double inv_B_perp = 1.0 / B_perp;
    const double Bz_over_B = Bz * inv_B_mag;
    const double Bz_over_B_perp = Bz_over_B * inv_B_perp;

    const double r00 = -By * inv_B_perp;
    const double r10 = -Bx * Bz_over_B_perp;
    const double r20 = Bx * inv_B_mag;
    const double r01 = Bx * inv_B_perp;
    const double r11 = -By * Bz_over_B_perp;
    const double r21 = By * inv_B_mag;
    const double r02 = 0.0;
    const double r12 = B_perp * inv_B_mag;
    const double r22 = Bz * inv_B_mag;

    *ox = r00 * vx + r10 * vy + r20 * vz;
    *oy = r01 * vx + r11 * vy + r21 * vz;
    *oz = r02 * vx + r12 * vy + r22 * vz;
    return 1;
}

// Pure helical exponential map in the B-aligned (zeta) frame: B_x = B_y = 0,
// B_z = B_mag constant, P_z conserved. Arc-length step h along B.
// Uses kappa = q B / P_z and stable sin(theta)/(qB) = (h/P_z) sinc(theta).
GPUFUN
void borissolenoid_helical_F_step(
    double* x,
    double* y,
    double* px,
    double* py,
    const double B_mag,
    const double P_z,
    const double q_coulomb,
    const double h
) {
    const double theta = q_coulomb * B_mag * h / P_z;
    const double st = sin(theta);
    const double ct = cos(theta);
    const double h_over_Pz = h / P_z;

    const double sinc_t = borissolenoid_sinc(theta);
    const double vers_over_t = borissolenoid_vers_over_theta(theta);
    const double cosm1_over_t = borissolenoid_cos_minus_one_over_theta(theta);

    const double x_in = *x;
    const double y_in = *y;
    const double px_in = *px;
    const double py_in = *py;

    *x = x_in
        + h_over_Pz * sinc_t * px_in
        + h_over_Pz * vers_over_t * py_in;

    *y = y_in
        + h_over_Pz * cosm1_over_t * px_in
        + h_over_Pz * sinc_t * py_in;

    *px = ct * px_in + st * py_in;
    *py = -st * px_in + ct * py_in;
}

#endif // XTRACK_BORISSOLENOID_HELICAL_MAP_H
