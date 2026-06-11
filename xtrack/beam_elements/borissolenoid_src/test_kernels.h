// Test helpers exposing borissolenoid C functions to Python via xobjects kernels.

#ifndef XTRACK_BORISSOLENOID_TEST_KERNELS_H
#define XTRACK_BORISSOLENOID_TEST_KERNELS_H

#include "xtrack/beam_elements/borissolenoid_src/elliptic_integrals.h"
#include "xtrack/beam_elements/borissolenoid_src/helical_map.h"
#include "xtrack/beam_elements/borissolenoid_src/solenoid_B_field_eval.h"

static void borissolenoid_test_elliptic(
    double m, double n,
    double* k_out, double* e_out, double* pi_out
) {
    *k_out = ellip_k(m);
    *e_out = ellip_e(m);
    *pi_out = ellip_pi(n, m);
}

static void borissolenoid_test_field(
    double x, double y, double z,
    double L, double a, double B0, double z0,
    double* bx_out, double* by_out, double* bz_out
) {
    evaluate_solenoid_B(x, y, z, L, a, B0, z0, bx_out, by_out, bz_out);
}

static void borissolenoid_test_helical_step(
    double x, double y, double px, double py, double ps,
    double Bx, double By, double Bz,
    double q_coulomb, double h,
    double* x_out, double* y_out, double* px_out, double* py_out
) {
    const double B_mag = sqrt(Bx * Bx + By * By + Bz * Bz);
    const double B_perp = sqrt(Bx * Bx + By * By);
    const double P_z = (Bx * px + By * py + Bz * ps) / B_mag;

    double x_z;
    double y_z;
    double z_z;
    double px_z;
    double py_z;
    double pz_z;

    borissolenoid_vec_to_zeta(
        Bx, By, Bz, B_mag, B_perp,
        x, y, 0.0,
        &x_z, &y_z, &z_z
    );
    borissolenoid_vec_to_zeta(
        Bx, By, Bz, B_mag, B_perp,
        px, py, ps,
        &px_z, &py_z, &pz_z
    );

    borissolenoid_helical_F_step(
        &x_z, &y_z, &px_z, &py_z,
        B_mag, P_z, q_coulomb, h
    );

    double x_lab;
    double y_lab;
    double z_unused;
    double px_lab;
    double py_lab;
    double ps_lab;

    borissolenoid_vec_to_lab(
        Bx, By, Bz, B_mag, B_perp,
        x_z, y_z, z_z,
        &x_lab, &y_lab, &z_unused
    );
    borissolenoid_vec_to_lab(
        Bx, By, Bz, B_mag, B_perp,
        px_z, py_z, P_z,
        &px_lab, &py_lab, &ps_lab
    );

    *x_out = x_lab;
    *y_out = y_lab;
    *px_out = px_lab;
    *py_out = py_lab;
}

#endif
