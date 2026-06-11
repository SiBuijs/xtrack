// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_SOLENOID_B_FIELD_EVAL_H
#define XTRACK_SOLENOID_B_FIELD_EVAL_H

#ifndef GPUFUN
#define GPUFUN
#endif

// Hampton et al., Closed-form expressions for the magnetic fields of rectangular
// and circular finite-length solenoids and current loops
// https://pubs.aip.org/aip/adv/article/10/6/065320/997382/
//
// Port of xtrack/_temp/boris_and_solenoid_map/solenoid_field.py

#include <math.h>
#include "xtrack/beam_elements/borissolenoid_src/elliptic_integrals.h"

#ifndef XTRACK_SOLENOID_R_TOL
#define XTRACK_SOLENOID_R_TOL 1e-11
#endif

GPUFUN
void evaluate_solenoid_B(
    const double x,
    const double y,
    const double z,
    const double L,
    const double a,
    const double B0,
    const double z0,
    double *Bx,
    double *By,
    double *Bz
) {
    const double r = sqrt(x * x + y * y);
    const double zeta_plus = z - z0 + 0.5 * L;
    const double zeta_minus = z - z0 - 0.5 * L;

    const double a_plus_r = a + r;
    const double a_minus_r = a - r;

    const double u = (r > 0.0) ? (4.0 * a * r / (a_plus_r * a_plus_r)) : 0.0;

    const double denom_plus = a_plus_r * a_plus_r + zeta_plus * zeta_plus;
    const double denom_minus = a_plus_r * a_plus_r + zeta_minus * zeta_minus;

    const double m_plus = (r > 0.0) ? (4.0 * a * r / denom_plus) : 0.0;
    const double m_minus = (r > 0.0) ? (4.0 * a * r / denom_minus) : 0.0;

    const double sqrt_plus = sqrt(4.0 / denom_plus);
    const double sqrt_minus = sqrt(4.0 / denom_minus);

    const double kk_plus = ellip_k(m_plus);
    const double kk_minus = ellip_k(m_minus);
    const double pp_plus = ellip_pi(u, m_plus);
    const double pp_minus = ellip_pi(u, m_minus);

    const double pi = 3.141592653589793238462643383279502884;
    const double bz_plus = B0 * zeta_plus / (4.0 * pi) * sqrt_plus *
        (kk_plus + (a_minus_r / a_plus_r) * pp_plus);
    const double bz_minus = B0 * zeta_minus / (4.0 * pi) * sqrt_minus *
        (kk_minus + (a_minus_r / a_plus_r) * pp_minus);

    *Bz = bz_plus - bz_minus;

    if (r < XTRACK_SOLENOID_R_TOL) {
        *Bx = 0.0;
        *By = 0.0;
        return;
    }

    const double ee_plus = ellip_e(m_plus);
    const double ee_minus = ellip_e(m_minus);

    const double sqrt_r_plus = sqrt(a / (r * m_plus));
    const double sqrt_r_minus = sqrt(a / (r * m_minus));

    const double br_plus = B0 / pi * sqrt_r_plus *
        (ee_plus - (1.0 - 0.5 * m_plus) * kk_plus);
    const double br_minus = B0 / pi * sqrt_r_minus *
        (ee_minus - (1.0 - 0.5 * m_minus) * kk_minus);

    const double br = br_plus - br_minus;

    *Bx = br * x / r;
    *By = br * y / r;
}

#endif // XTRACK_SOLENOID_B_FIELD_EVAL_H
