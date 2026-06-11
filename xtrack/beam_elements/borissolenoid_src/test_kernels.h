// Test helpers exposing borissolenoid C functions to Python via xobjects kernels.

#ifndef XTRACK_BORISSOLENOID_TEST_KERNELS_H
#define XTRACK_BORISSOLENOID_TEST_KERNELS_H

#include "xtrack/beam_elements/borissolenoid_src/elliptic_integrals.h"
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

#endif
