// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_WIGGLER_B_FIELD_EVAL_H
#define XTRACK_WIGGLER_B_FIELD_EVAL_H

#include "xtrack/headers/track.h"

GPUFUN
void evaluate_wiggler_B(
    const double x,
    const double y,
    const double s,
    const double k_u,
    const double b_tilde,
    double *Bx_out,
    double *By_out,
    double *Bs_out
) {
    (void)x;
    *Bx_out = 0.0;
    *By_out = b_tilde * cosh(k_u * y) * cos(k_u * s);
    *Bs_out = b_tilde * sinh(k_u * y) * sin(k_u * s);
}

#endif // XTRACK_WIGGLER_B_FIELD_EVAL_H
