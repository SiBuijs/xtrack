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
// Caller must advance the B-parallel position coordinate by h after this step.
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

// d/dtheta of borissolenoid_sinc(theta)
GPUFUN
double borissolenoid_dsinc_dtheta(const double theta) {
    if (fabs(theta) < BORISSOLENOID_HELICAL_SINC_EPS) {
        return 0.0;
    }
    const double sin_th = sin(theta);
    const double cos_th = cos(theta);
    return (theta * cos_th - sin_th) / (theta * theta);
}

// d/dtheta of borissolenoid_vers_over_theta(theta)
GPUFUN
double borissolenoid_dvers_dtheta(const double theta) {
    if (fabs(theta) < BORISSOLENOID_HELICAL_SINC_EPS) {
        return 0.0;
    }
    const double sin_th = sin(theta);
    const double cos_th = cos(theta);
    return (sin_th * theta - (1.0 - cos_th)) / (theta * theta);
}

// d/dtheta of borissolenoid_cos_minus_one_over_theta(theta)
GPUFUN
double borissolenoid_dcosm1_dtheta(const double theta) {
    if (fabs(theta) < BORISSOLENOID_HELICAL_SINC_EPS) {
        return 0.0;
    }
    const double sin_th = sin(theta);
    const double cos_th = cos(theta);
    return (-sin_th * theta - (cos_th - 1.0)) / (theta * theta);
}

// Lab-z after a helical step of arc length h (zeta-frame inputs, non-mutating).
GPUFUN
double borissolenoid_helical_step_z_lab(
    const double x_z0,
    const double y_z0,
    const double z_z0,
    const double px_z0,
    const double py_z0,
    const double Bx,
    const double By,
    const double Bz,
    const double B_mag,
    const double B_perp,
    const double P_z,
    const double q_coulomb,
    const double h
) {
    double x_z = x_z0;
    double y_z = y_z0;
    double px_z = px_z0;
    double py_z = py_z0;

    borissolenoid_helical_F_step(
        &x_z, &y_z, &px_z, &py_z, B_mag, P_z, q_coulomb, h
    );

    const double z_z = z_z0 + h;
    double x_lab;
    double y_lab;
    double z_lab;

    borissolenoid_vec_to_lab(
        Bx, By, Bz, B_mag, B_perp,
        x_z, y_z, z_z,
        &x_lab, &y_lab, &z_lab
    );
    return z_lab;
}

// dz_lab/dh for borissolenoid_helical_step_z_lab.
GPUFUN
double borissolenoid_helical_step_dz_lab_dh(
    const double x_z0,
    const double y_z0,
    const double z_z0,
    const double px_z0,
    const double py_z0,
    const double Bx,
    const double By,
    const double Bz,
    const double B_mag,
    const double B_perp,
    const double P_z,
    const double q_coulomb,
    const double h
) {
    (void)x_z0;
    (void)z_z0;

    if (B_perp < BORISSOLENOID_HELICAL_EPS) {
        return Bz / B_mag;
    }

    const double inv_B_mag = 1.0 / B_mag;
    const double r12 = B_perp * inv_B_mag;
    const double r22 = Bz * inv_B_mag;

    const double theta = q_coulomb * B_mag * h / P_z;
    const double dtheta_dh = q_coulomb * B_mag / P_z;

    const double sinc_t = borissolenoid_sinc(theta);
    const double vers_t = borissolenoid_vers_over_theta(theta);
    const double cosm1_t = borissolenoid_cos_minus_one_over_theta(theta);

    const double dsinc_dh = borissolenoid_dsinc_dtheta(theta) * dtheta_dh;
    const double dvers_dh = borissolenoid_dvers_dtheta(theta) * dtheta_dh;
    const double dcosm1_dh = borissolenoid_dcosm1_dtheta(theta) * dtheta_dh;

    const double inv_Pz = 1.0 / P_z;
    const double A = sinc_t * px_z0 + vers_t * py_z0;
    const double Bcoef = cosm1_t * px_z0 + sinc_t * py_z0;

    const double dA_dh = px_z0 * dsinc_dh + py_z0 * dvers_dh;
    const double dBcoef_dh = px_z0 * dcosm1_dh + py_z0 * dsinc_dh;

    const double dy_dh = Bcoef * inv_Pz + (h * inv_Pz) * dBcoef_dh;
    (void)A;
    (void)dA_dh;

    return r12 * dy_dh + r22;
}

// Find h so the particle lands on lab-z = z_target after the helical step.
// Uses Newton iterations (typically 1-2) with initial guess h_init.
GPUFUN
double borissolenoid_solve_helical_h_for_z_plane(
    const double x_z0,
    const double y_z0,
    const double z_z0,
    const double px_z0,
    const double py_z0,
    const double Bx,
    const double By,
    const double Bz,
    const double B_mag,
    const double B_perp,
    const double P_z,
    const double q_coulomb,
    const double z_target,
    const double h_init
) {
    if (B_perp < BORISSOLENOID_HELICAL_EPS) {
        return h_init;
    }

    const double tol = 1e-15;
    const int max_iter = 3;

    double h = h_init;
    for (int iter = 0; iter < max_iter; ++iter) {
        const double z_lab = borissolenoid_helical_step_z_lab(
            x_z0, y_z0, z_z0, px_z0, py_z0,
            Bx, By, Bz, B_mag, B_perp, P_z, q_coulomb, h
        );
        const double f = z_lab - z_target;
        if (fabs(f) < tol) {
            break;
        }
        const double dz_dh = borissolenoid_helical_step_dz_lab_dh(
            x_z0, y_z0, z_z0, px_z0, py_z0,
            Bx, By, Bz, B_mag, B_perp, P_z, q_coulomb, h
        );
        if (fabs(dz_dh) < BORISSOLENOID_HELICAL_EPS) {
            break;
        }
        h -= f / dz_dh;
    }
    return h;
}

#endif // XTRACK_BORISSOLENOID_HELICAL_MAP_H
