#include <stddef.h>
#include <stdio.h>
#include <string.h>
#define MAX_DEGREE 4

#ifndef SPLINE_A_FIELD_EVAL_H
#define SPLINE_A_FIELD_EVAL_H

// Auto-generated symbolic field expressions for A
// NOTE:
//   - 's' is the local coordinate within the element: s_local ∈ [0, L].
//   - Hermite coefficients are defined on s_local ∈ [0, L] and are converted
//     internally to polynomials in s_local via hermite_to_polynomial(0, L, ...).
//
// Hermite input layout
// --------------------
//   - bs        : one scalar Hermite polynomial (5 coeffs) for bs(s_local)
//   - by[i]     : Hermite coeffs (5) for polynomial group by_i_*(s_local)
//   - bx[i]     : Hermite coeffs (5) for polynomial group bx_i_*(s_local)
//
// For multipole_order = n (1 ≤ n ≤ 7):
//   - bs:       1 polynomial      → bs_0..bs_4 from bs
//   - by:       n polynomials     → by_i_0..by_i_4 from by[i], i=0..n-1
//   - bx:       n polynomials     → bx_i_0..bx_i_4 from bx[i], i=0..n-1
//
// The symbolic expressions below are unchanged; only the way the bs_*, by_*_*,
// and bx_*_* scalars are populated has been refactored to use Hermite data.
#include "xtrack/beam_elements/splineboris_src/spline_poly_helpers.h"

GPUFUN
void evaluate_A(const double x, const double y, const double s,
                const double *bs,
                const double *const *by,
                const double *const *bx,
                const double L,
                const int multipole_order,
                double *Ax_out, double *Ay_out, double *As_out){

	switch (multipole_order) {
	case 1: {
		// Hermite → polynomial coefficients (order 1)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = (1.0/5.0)*y*y*y*y*y;
		const double x1 = 4*by_0_4;
		const double x2 = 2*s;
		const double x3 = s*s;
		const double x4 = 3*x3;
		const double x5 = s*s*s;
		const double x6 = y*y*y;
		const double x7 = 4*bx_0_4;
		const double x8 = 3*s;
		const double x9 = 6*x3;
		const double x10 = s*s*s*s;

		// Reduced expressions
		*Ax_out = -bs_4*x0 - x6*(-1.0/3.0*bs_2 - 1.0/3.0*bs_3*x8 - 1.0/3.0*bs_4*x9 - x*(bx_0_3 + s*x7)) + (1.0/4.0)*y*y*y*y*(by_0_3 + s*x1) - 1.0/2.0*y*y*(by_0_1 + by_0_2*x2 + by_0_3*x4 + x1*x5) - y*(bs_0 + bs_1*s + bs_2*x3 + bs_3*x5 + bs_4*x10 + x*(bx_0_1 + bx_0_2*x2 + bx_0_3*x4 + x5*x7));
		*Ay_out = 0;
		*As_out = bx_0_4*x0 - x*(by_0_0 + by_0_1*s + by_0_2*x3 + by_0_3*x5 + by_0_4*x10) - 1.0/3.0*x6*(bx_0_2 + bx_0_3*x8 + bx_0_4*x9) + y*(bx_0_0 + bx_0_1*s + bx_0_2*x3 + bx_0_3*x5 + bx_0_4*x10);
		return;

	}
	case 2: {
		// Hermite → polynomial coefficients (order 2)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = y*y*y*y*y;
		const double x1 = 4*s;
		const double x2 = bx_1_3 + bx_1_4*x1;
		const double x3 = y*y*y*y;
		const double x4 = y*y;
		const double x5 = s*s;
		const double x6 = s*s*s;
		const double x7 = 2*s;
		const double x8 = 3*x5;
		const double x9 = 4*x6;
		const double x10 = y*y*y;
		const double x11 = (1.0/2.0)*x*x;
		const double x12 = 3*s;
		const double x13 = 6*x5;
		const double x14 = s*s*s*s;
		const double x15 = by_1_0 + by_1_1*s + by_1_2*x5 + by_1_3*x6 + by_1_4*x14;

		// Reduced expressions
		*Ax_out = -x0*((1.0/5.0)*bs_4 + (1.0/10.0)*x2) - x10*(-1.0/3.0*bs_2 - 1.0/3.0*bs_3*x12 - 1.0/3.0*bs_4*x13 - 1.0/6.0*bx_1_1 - 1.0/3.0*bx_1_2*s - 1.0/2.0*bx_1_3*x5 - 2.0/3.0*bx_1_4*x6 - x*(bx_0_3 + bx_0_4*x1) - x11*x2) - x3*(-1.0/4.0*by_0_3 - 1.0/4.0*by_0_4*x1 - 1.0/4.0*x*(by_1_3 + by_1_4*x1)) - x4*((1.0/2.0)*by_0_1 + by_0_2*s + (3.0/2.0)*by_0_3*x5 + 2*by_0_4*x6 + (1.0/2.0)*x*(by_1_1 + by_1_2*x7 + by_1_3*x8 + by_1_4*x9)) - y*(bs_0 + bs_1*s + bs_2*x5 + bs_3*x6 + bs_4*x14 + x*(bx_0_1 + bx_0_2*x7 + bx_0_3*x8 + bx_0_4*x9) + x11*(bx_1_1 + bx_1_2*x7 + bx_1_3*x8 + bx_1_4*x9));
		*Ay_out = 0;
		*As_out = (1.0/30.0)*by_1_4*y*y*y*y*y*y - x*(by_0_0 + by_0_1*s + by_0_2*x5 + by_0_3*x6 + by_0_4*x14) + x0*((1.0/5.0)*bx_0_4 + (1.0/5.0)*bx_1_4*x) + x10*(-1.0/3.0*bx_0_2 - 1.0/3.0*bx_0_3*x12 - 1.0/3.0*bx_0_4*x13 - 1.0/3.0*x*(bx_1_2 + bx_1_3*x12 + bx_1_4*x13)) - x11*x15 + (1.0/2.0)*x15*x4 - 1.0/12.0*x3*(by_1_2 + by_1_3*x12 + by_1_4*x13) + y*(bx_0_0 + bx_0_1*s + bx_0_2*x5 + bx_0_3*x6 + bx_0_4*x14 + x*(bx_1_0 + bx_1_1*s + bx_1_2*x5 + bx_1_3*x6 + bx_1_4*x14));
		return;

	}
	case 3: {
		// Hermite → polynomial coefficients (order 3)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly by2_poly = hermite_to_polynomial(0.0, L, by[2]);
		const double by_2_0 = by2_poly.coeffs[0];
		const double by_2_1 = by2_poly.coeffs[1];
		const double by_2_2 = by2_poly.coeffs[2];
		const double by_2_3 = by2_poly.coeffs[3];
		const double by_2_4 = by2_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		const Poly bx2_poly = hermite_to_polynomial(0.0, L, bx[2]);
		const double bx_2_0 = bx2_poly.coeffs[0];
		const double bx_2_1 = bx2_poly.coeffs[1];
		const double bx_2_2 = bx2_poly.coeffs[2];
		const double bx_2_3 = bx2_poly.coeffs[3];
		const double bx_2_4 = bx2_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = y*y*y*y*y*y;
		const double x1 = 4*s;
		const double x2 = by_2_3 + by_2_4*x1;
		const double x3 = y*y*y*y*y;
		const double x4 = bx_1_3 + bx_1_4*x1;
		const double x5 = bx_2_3 + bx_2_4*x1;
		const double x6 = y*y*y*y;
		const double x7 = s*s;
		const double x8 = s*s*s;
		const double x9 = x*x;
		const double x10 = y*y;
		const double x11 = 2*s;
		const double x12 = 3*x7;
		const double x13 = 4*x8;
		const double x14 = (1.0/2.0)*x;
		const double x15 = s*s*s*s;
		const double x16 = (1.0/2.0)*x9;
		const double x17 = x*x*x;
		const double x18 = (1.0/6.0)*bx_2_1 + (1.0/6.0)*bx_2_2*x11 + (1.0/6.0)*bx_2_3*x12 + (1.0/6.0)*bx_2_4*x13;
		const double x19 = y*y*y;
		const double x20 = (1.0/6.0)*x17;
		const double x21 = 3*s;
		const double x22 = 6*x7;
		const double x23 = by_1_1*s;
		const double x24 = by_1_2*x7;
		const double x25 = by_1_3*x8;
		const double x26 = by_1_4*x15;
		const double x27 = by_2_0 + by_2_1*s + by_2_2*x7 + by_2_3*x8 + by_2_4*x15;
		const double x28 = bx_2_2 + bx_2_3*x21 + bx_2_4*x22;
		const double x29 = bx_2_1*s;
		const double x30 = bx_2_2*x7;
		const double x31 = bx_2_3*x8;
		const double x32 = bx_2_4*x15;

		// Reduced expressions
		*Ax_out = -1.0/60.0*x0*x2 - x10*((1.0/2.0)*by_0_1 + by_0_2*s + (3.0/2.0)*by_0_3*x7 + 2*by_0_4*x8 + x14*(by_1_1 + by_1_2*x11 + by_1_3*x12 + by_1_4*x13) + (1.0/4.0)*x9*(by_2_1 + by_2_2*x11 + by_2_3*x12 + by_2_4*x13)) - x19*(-1.0/3.0*bs_2 - 1.0/3.0*bs_3*x21 - 1.0/3.0*bs_4*x22 - 1.0/6.0*bx_1_1 - 1.0/3.0*bx_1_2*s - 1.0/2.0*bx_1_3*x7 - 2.0/3.0*bx_1_4*x8 - x*x18 - x*(bx_0_3 + bx_0_4*x1) - x16*x4 - x20*x5) - x3*((1.0/5.0)*bs_4 + (1.0/10.0)*x*x5 + (1.0/10.0)*x4) - x6*(-1.0/4.0*by_0_3 - 1.0/4.0*by_0_4*x1 - 1.0/24.0*by_2_1 - 1.0/12.0*by_2_2*s - 1.0/8.0*by_2_3*x7 - 1.0/6.0*by_2_4*x8 - 1.0/4.0*x*(by_1_3 + by_1_4*x1) - 1.0/8.0*x2*x9) - y*(bs_0 + bs_1*s + bs_2*x7 + bs_3*x8 + bs_4*x15 + x*(bx_0_1 + bx_0_2*x11 + bx_0_3*x12 + bx_0_4*x13) + x16*(bx_1_1 + bx_1_2*x11 + bx_1_3*x12 + bx_1_4*x13) + x17*x18);
		*Ay_out = 0;
		*As_out = -1.0/70.0*bx_2_4*y*y*y*y*y*y*y - x*(by_0_0 + by_0_1*s + by_0_2*x7 + by_0_3*x8 + by_0_4*x15) + x0*((1.0/30.0)*by_1_4 + (1.0/30.0)*by_2_4*x) + x10*((1.0/2.0)*by_1_0 + x14*x27 + (1.0/2.0)*x23 + (1.0/2.0)*x24 + (1.0/2.0)*x25 + (1.0/2.0)*x26) - x16*(by_1_0 + x23 + x24 + x25 + x26) + x19*(-1.0/3.0*bx_0_2 - 1.0/3.0*bx_0_3*x21 - 1.0/3.0*bx_0_4*x22 - 1.0/6.0*bx_2_0 - 1.0/3.0*x*(bx_1_2 + bx_1_3*x21 + bx_1_4*x22) - 1.0/6.0*x28*x9 - 1.0/6.0*x29 - 1.0/6.0*x30 - 1.0/6.0*x31 - 1.0/6.0*x32) - x20*x27 + x3*((1.0/5.0)*bx_0_4 + (1.0/5.0)*bx_1_4*x + (1.0/10.0)*bx_2_4*x9 + (1.0/30.0)*x28) + x6*(-1.0/12.0*by_1_2 - 1.0/12.0*by_1_3*x21 - 1.0/12.0*by_1_4*x22 - 1.0/12.0*x*(by_2_2 + by_2_3*x21 + by_2_4*x22)) + y*(bx_0_0 + bx_0_1*s + bx_0_2*x7 + bx_0_3*x8 + bx_0_4*x15 + x*(bx_1_0 + bx_1_1*s + bx_1_2*x7 + bx_1_3*x8 + bx_1_4*x15) + x16*(bx_2_0 + x29 + x30 + x31 + x32));
		return;

	}
	case 4: {
		// Hermite → polynomial coefficients (order 4)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly by2_poly = hermite_to_polynomial(0.0, L, by[2]);
		const double by_2_0 = by2_poly.coeffs[0];
		const double by_2_1 = by2_poly.coeffs[1];
		const double by_2_2 = by2_poly.coeffs[2];
		const double by_2_3 = by2_poly.coeffs[3];
		const double by_2_4 = by2_poly.coeffs[4];

		const Poly by3_poly = hermite_to_polynomial(0.0, L, by[3]);
		const double by_3_0 = by3_poly.coeffs[0];
		const double by_3_1 = by3_poly.coeffs[1];
		const double by_3_2 = by3_poly.coeffs[2];
		const double by_3_3 = by3_poly.coeffs[3];
		const double by_3_4 = by3_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		const Poly bx2_poly = hermite_to_polynomial(0.0, L, bx[2]);
		const double bx_2_0 = bx2_poly.coeffs[0];
		const double bx_2_1 = bx2_poly.coeffs[1];
		const double bx_2_2 = bx2_poly.coeffs[2];
		const double bx_2_3 = bx2_poly.coeffs[3];
		const double bx_2_4 = bx2_poly.coeffs[4];

		const Poly bx3_poly = hermite_to_polynomial(0.0, L, bx[3]);
		const double bx_3_0 = bx3_poly.coeffs[0];
		const double bx_3_1 = bx3_poly.coeffs[1];
		const double bx_3_2 = bx3_poly.coeffs[2];
		const double bx_3_3 = bx3_poly.coeffs[3];
		const double bx_3_4 = bx3_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = y*y*y*y*y*y*y;
		const double x1 = 4*s;
		const double x2 = bx_3_3 + bx_3_4*x1;
		const double x3 = y*y*y*y*y*y;
		const double x4 = by_2_3 + by_2_4*x1;
		const double x5 = by_3_3 + by_3_4*x1;
		const double x6 = y*y*y*y*y;
		const double x7 = s*s;
		const double x8 = s*s*s;
		const double x9 = (1.0/30.0)*bx_3_4;
		const double x10 = bx_1_3 + bx_1_4*x1;
		const double x11 = bx_2_3 + bx_2_4*x1;
		const double x12 = x*x;
		const double x13 = y*y*y*y;
		const double x14 = x*x*x;
		const double x15 = 2*s;
		const double x16 = 3*x7;
		const double x17 = 4*x8;
		const double x18 = by_3_1 + by_3_2*x15 + by_3_3*x16 + by_3_4*x17;
		const double x19 = y*y;
		const double x20 = (1.0/2.0)*x;
		const double x21 = (1.0/4.0)*x12;
		const double x22 = s*s*s*s;
		const double x23 = (1.0/2.0)*x12;
		const double x24 = (1.0/6.0)*bx_2_1 + (1.0/6.0)*bx_2_2*x15 + (1.0/6.0)*bx_2_3*x16 + (1.0/6.0)*bx_2_4*x17;
		const double x25 = bx_3_1 + bx_3_2*x15 + bx_3_3*x16 + bx_3_4*x17;
		const double x26 = (1.0/24.0)*x*x*x*x;
		const double x27 = y*y*y;
		const double x28 = (1.0/6.0)*x14;
		const double x29 = 3*s;
		const double x30 = 6*x7;
		const double x31 = by_1_1*s;
		const double x32 = by_1_2*x7;
		const double x33 = by_1_3*x8;
		const double x34 = by_1_4*x22;
		const double x35 = by_2_0 + by_2_1*s + by_2_2*x7 + by_2_3*x8 + by_2_4*x22;
		const double x36 = by_3_1*s;
		const double x37 = by_3_2*x7;
		const double x38 = by_3_3*x8;
		const double x39 = by_3_4*x22;
		const double x40 = by_3_0 + x36 + x37 + x38 + x39;
		const double x41 = (1.0/30.0)*x;
		const double x42 = by_3_2 + by_3_3*x29 + by_3_4*x30;
		const double x43 = bx_2_2 + bx_2_3*x29 + bx_2_4*x30;
		const double x44 = bx_3_2 + bx_3_3*x29 + bx_3_4*x30;
		const double x45 = bx_2_1*s;
		const double x46 = bx_2_2*x7;
		const double x47 = bx_2_3*x8;
		const double x48 = bx_2_4*x22;
		const double x49 = bx_3_0 + bx_3_1*s + bx_3_2*x7 + bx_3_3*x8 + bx_3_4*x22;

		// Reduced expressions
		*Ax_out = (1.0/280.0)*x0*x2 - x13*(-1.0/4.0*by_0_3 - 1.0/4.0*by_0_4*x1 - 1.0/24.0*by_2_1 - 1.0/12.0*by_2_2*s - 1.0/8.0*by_2_3*x7 - 1.0/6.0*by_2_4*x8 - 1.0/24.0*x*x18 - 1.0/4.0*x*(by_1_3 + by_1_4*x1) - 1.0/8.0*x12*x4 - 1.0/24.0*x14*x5) - x19*((1.0/2.0)*by_0_1 + by_0_2*s + (3.0/2.0)*by_0_3*x7 + 2*by_0_4*x8 + (1.0/12.0)*x14*x18 + x20*(by_1_1 + by_1_2*x15 + by_1_3*x16 + by_1_4*x17) + x21*(by_2_1 + by_2_2*x15 + by_2_3*x16 + by_2_4*x17)) - x27*(-1.0/3.0*bs_2 - 1.0/3.0*bs_3*x29 - 1.0/3.0*bs_4*x30 - 1.0/6.0*bx_1_1 - 1.0/3.0*bx_1_2*s - 1.0/2.0*bx_1_3*x7 - 2.0/3.0*bx_1_4*x8 - x*x24 - x*(bx_0_3 + bx_0_4*x1) - x10*x23 - x11*x28 - 1.0/12.0*x12*x25 - x2*x26) - x3*((1.0/60.0)*x*x5 + (1.0/60.0)*x4) - x6*((1.0/5.0)*bs_4 + (1.0/120.0)*bx_3_1 + (1.0/60.0)*bx_3_2*s + (1.0/40.0)*bx_3_3*x7 + (1.0/10.0)*x*x11 + (1.0/10.0)*x10 + (1.0/20.0)*x12*x2 + x8*x9) - y*(bs_0 + bs_1*s + bs_2*x7 + bs_3*x8 + bs_4*x22 + x*(bx_0_1 + bx_0_2*x15 + bx_0_3*x16 + bx_0_4*x17) + x14*x24 + x23*(bx_1_1 + bx_1_2*x15 + bx_1_3*x16 + bx_1_4*x17) + x25*x26);
		*Ay_out = 0;
		*As_out = -1.0/560.0*by_3_4*y*y*y*y*y*y*y*y - x*(by_0_0 + by_0_1*s + by_0_2*x7 + by_0_3*x8 + by_0_4*x22) + x0*(-1.0/70.0*bx_2_4 - 1.0/70.0*bx_3_4*x) + x13*(-1.0/12.0*by_1_2 - 1.0/12.0*by_1_3*x29 - 1.0/12.0*by_1_4*x30 - 1.0/24.0*by_3_0 - 1.0/12.0*x*(by_2_2 + by_2_3*x29 + by_2_4*x30) - 1.0/24.0*x12*x42 - 1.0/24.0*x36 - 1.0/24.0*x37 - 1.0/24.0*x38 - 1.0/24.0*x39) + x19*((1.0/2.0)*by_1_0 + x20*x35 + x21*x40 + (1.0/2.0)*x31 + (1.0/2.0)*x32 + (1.0/2.0)*x33 + (1.0/2.0)*x34) - x23*(by_1_0 + x31 + x32 + x33 + x34) - x26*x40 + x27*(-1.0/3.0*bx_0_2 - 1.0/3.0*bx_0_3*x29 - 1.0/3.0*bx_0_4*x30 - 1.0/6.0*bx_2_0 - 1.0/6.0*x*x49 - 1.0/3.0*x*(bx_1_2 + bx_1_3*x29 + bx_1_4*x30) - 1.0/6.0*x12*x43 - 1.0/18.0*x14*x44 - 1.0/6.0*x45 - 1.0/6.0*x46 - 1.0/6.0*x47 - 1.0/6.0*x48) - x28*x35 + x3*((1.0/30.0)*by_1_4 + by_2_4*x41 + (1.0/60.0)*by_3_4*x12 + (1.0/180.0)*x42) + x6*((1.0/5.0)*bx_0_4 + (1.0/5.0)*bx_1_4*x + (1.0/10.0)*bx_2_4*x12 + x14*x9 + x41*x44 + (1.0/30.0)*x43) + y*(bx_0_0 + bx_0_1*s + bx_0_2*x7 + bx_0_3*x8 + bx_0_4*x22 + x*(bx_1_0 + bx_1_1*s + bx_1_2*x7 + bx_1_3*x8 + bx_1_4*x22) + x23*(bx_2_0 + x45 + x46 + x47 + x48) + x28*x49);
		return;

	}
	case 5: {
		// Hermite → polynomial coefficients (order 5)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly by2_poly = hermite_to_polynomial(0.0, L, by[2]);
		const double by_2_0 = by2_poly.coeffs[0];
		const double by_2_1 = by2_poly.coeffs[1];
		const double by_2_2 = by2_poly.coeffs[2];
		const double by_2_3 = by2_poly.coeffs[3];
		const double by_2_4 = by2_poly.coeffs[4];

		const Poly by3_poly = hermite_to_polynomial(0.0, L, by[3]);
		const double by_3_0 = by3_poly.coeffs[0];
		const double by_3_1 = by3_poly.coeffs[1];
		const double by_3_2 = by3_poly.coeffs[2];
		const double by_3_3 = by3_poly.coeffs[3];
		const double by_3_4 = by3_poly.coeffs[4];

		const Poly by4_poly = hermite_to_polynomial(0.0, L, by[4]);
		const double by_4_0 = by4_poly.coeffs[0];
		const double by_4_1 = by4_poly.coeffs[1];
		const double by_4_2 = by4_poly.coeffs[2];
		const double by_4_3 = by4_poly.coeffs[3];
		const double by_4_4 = by4_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		const Poly bx2_poly = hermite_to_polynomial(0.0, L, bx[2]);
		const double bx_2_0 = bx2_poly.coeffs[0];
		const double bx_2_1 = bx2_poly.coeffs[1];
		const double bx_2_2 = bx2_poly.coeffs[2];
		const double bx_2_3 = bx2_poly.coeffs[3];
		const double bx_2_4 = bx2_poly.coeffs[4];

		const Poly bx3_poly = hermite_to_polynomial(0.0, L, bx[3]);
		const double bx_3_0 = bx3_poly.coeffs[0];
		const double bx_3_1 = bx3_poly.coeffs[1];
		const double bx_3_2 = bx3_poly.coeffs[2];
		const double bx_3_3 = bx3_poly.coeffs[3];
		const double bx_3_4 = bx3_poly.coeffs[4];

		const Poly bx4_poly = hermite_to_polynomial(0.0, L, bx[4]);
		const double bx_4_0 = bx4_poly.coeffs[0];
		const double bx_4_1 = bx4_poly.coeffs[1];
		const double bx_4_2 = bx4_poly.coeffs[2];
		const double bx_4_3 = bx4_poly.coeffs[3];
		const double bx_4_4 = bx4_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = y*y*y*y*y*y*y*y;
		const double x1 = 4*s;
		const double x2 = by_4_3 + by_4_4*x1;
		const double x3 = y*y*y*y*y*y*y;
		const double x4 = bx_3_3 + bx_3_4*x1;
		const double x5 = bx_4_3 + bx_4_4*x1;
		const double x6 = y*y*y*y*y*y;
		const double x7 = s*s;
		const double x8 = s*s*s;
		const double x9 = (1.0/180.0)*by_4_4;
		const double x10 = by_2_3 + by_2_4*x1;
		const double x11 = by_3_3 + by_3_4*x1;
		const double x12 = x*x;
		const double x13 = y*y*y*y*y;
		const double x14 = (1.0/30.0)*bx_3_4;
		const double x15 = bx_1_3 + bx_1_4*x1;
		const double x16 = bx_2_3 + bx_2_4*x1;
		const double x17 = x*x*x;
		const double x18 = 2*s;
		const double x19 = 3*x7;
		const double x20 = 4*x8;
		const double x21 = bx_4_1 + bx_4_2*x18 + bx_4_3*x19 + bx_4_4*x20;
		const double x22 = y*y;
		const double x23 = (1.0/2.0)*x;
		const double x24 = (1.0/4.0)*x12;
		const double x25 = by_3_1 + by_3_2*x18 + by_3_3*x19 + by_3_4*x20;
		const double x26 = (1.0/12.0)*x17;
		const double x27 = x*x*x*x;
		const double x28 = (1.0/48.0)*by_4_1 + (1.0/48.0)*by_4_2*x18 + (1.0/48.0)*by_4_3*x19 + (1.0/48.0)*by_4_4*x20;
		const double x29 = y*y*y*y;
		const double x30 = (1.0/24.0)*x;
		const double x31 = s*s*s*s;
		const double x32 = (1.0/2.0)*x12;
		const double x33 = (1.0/6.0)*bx_2_1 + (1.0/6.0)*bx_2_2*x18 + (1.0/6.0)*bx_2_3*x19 + (1.0/6.0)*bx_2_4*x20;
		const double x34 = bx_3_1 + bx_3_2*x18 + bx_3_3*x19 + bx_3_4*x20;
		const double x35 = (1.0/24.0)*x27;
		const double x36 = (1.0/120.0)*x*x*x*x*x;
		const double x37 = y*y*y;
		const double x38 = (1.0/6.0)*x17;
		const double x39 = 3*s;
		const double x40 = 6*x7;
		const double x41 = (1.0/12.0)*x12;
		const double x42 = by_1_1*s;
		const double x43 = by_1_2*x7;
		const double x44 = by_1_3*x8;
		const double x45 = by_1_4*x31;
		const double x46 = by_2_0 + by_2_1*s + by_2_2*x7 + by_2_3*x8 + by_2_4*x31;
		const double x47 = by_3_1*s;
		const double x48 = by_3_2*x7;
		const double x49 = by_3_3*x8;
		const double x50 = by_3_4*x31;
		const double x51 = by_3_0 + x47 + x48 + x49 + x50;
		const double x52 = by_4_0 + by_4_1*s + by_4_2*x7 + by_4_3*x8 + by_4_4*x31;
		const double x53 = bx_4_2 + bx_4_3*x39 + bx_4_4*x40;
		const double x54 = (1.0/30.0)*x;
		const double x55 = (1.0/60.0)*x12;
		const double x56 = by_3_2 + by_3_3*x39 + by_3_4*x40;
		const double x57 = by_4_2 + by_4_3*x39 + by_4_4*x40;
		const double x58 = bx_4_1*s;
		const double x59 = bx_4_2*x7;
		const double x60 = bx_4_3*x8;
		const double x61 = bx_4_4*x31;
		const double x62 = bx_2_2 + bx_2_3*x39 + bx_2_4*x40;
		const double x63 = bx_3_2 + bx_3_3*x39 + bx_3_4*x40;
		const double x64 = bx_2_1*s;
		const double x65 = bx_2_2*x7;
		const double x66 = bx_2_3*x8;
		const double x67 = bx_2_4*x31;
		const double x68 = bx_3_0 + bx_3_1*s + bx_3_2*x7 + bx_3_3*x8 + bx_3_4*x31;
		const double x69 = bx_4_0 + x58 + x59 + x60 + x61;

		// Reduced expressions
		*Ax_out = (1.0/2240.0)*x0*x2 - x13*((1.0/5.0)*bs_4 + (1.0/120.0)*bx_3_1 + (1.0/60.0)*bx_3_2*s + (1.0/40.0)*bx_3_3*x7 + (1.0/10.0)*x*x16 + (1.0/120.0)*x*x21 + (1.0/20.0)*x12*x4 + x14*x8 + (1.0/10.0)*x15 + (1.0/60.0)*x17*x5) - x22*((1.0/2.0)*by_0_1 + by_0_2*s + (3.0/2.0)*by_0_3*x7 + 2*by_0_4*x8 + x23*(by_1_1 + by_1_2*x18 + by_1_3*x19 + by_1_4*x20) + x24*(by_2_1 + by_2_2*x18 + by_2_3*x19 + by_2_4*x20) + x25*x26 + x27*x28) - x29*(-1.0/4.0*by_0_3 - 1.0/4.0*by_0_4*x1 - 1.0/24.0*by_2_1 - 1.0/12.0*by_2_2*s - 1.0/8.0*by_2_3*x7 - 1.0/6.0*by_2_4*x8 - 1.0/4.0*x*(by_1_3 + by_1_4*x1) - 1.0/8.0*x10*x12 - 1.0/24.0*x11*x17 - x12*x28 - 1.0/96.0*x2*x27 - x25*x30) - x3*(-1.0/280.0*x*x5 - 1.0/280.0*x4) - x37*(-1.0/3.0*bs_2 - 1.0/3.0*bs_3*x39 - 1.0/3.0*bs_4*x40 - 1.0/6.0*bx_1_1 - 1.0/3.0*bx_1_2*s - 1.0/2.0*bx_1_3*x7 - 2.0/3.0*bx_1_4*x8 - x*x33 - x*(bx_0_3 + bx_0_4*x1) - x15*x32 - x16*x38 - 1.0/36.0*x17*x21 - x34*x41 - x35*x4 - x36*x5) - x6*((1.0/720.0)*by_4_1 + (1.0/360.0)*by_4_2*s + (1.0/240.0)*by_4_3*x7 + (1.0/60.0)*x*x11 + (1.0/60.0)*x10 + (1.0/120.0)*x12*x2 + x8*x9) - y*(bs_0 + bs_1*s + bs_2*x7 + bs_3*x8 + bs_4*x31 + x*(bx_0_1 + bx_0_2*x18 + bx_0_3*x19 + bx_0_4*x20) + x17*x33 + x21*x36 + x32*(bx_1_1 + bx_1_2*x18 + bx_1_3*x19 + bx_1_4*x20) + x34*x35);
		*Ay_out = 0;
		*As_out = (1.0/2520.0)*bx_4_4*y*y*y*y*y*y*y*y*y - x*(by_0_0 + by_0_1*s + by_0_2*x7 + by_0_3*x8 + by_0_4*x31) + x0*(-1.0/560.0*by_3_4 - 1.0/560.0*by_4_4*x) + x13*((1.0/5.0)*bx_0_4 + (1.0/5.0)*bx_1_4*x + (1.0/10.0)*bx_2_4*x12 + (1.0/120.0)*bx_4_0 + (1.0/120.0)*bx_4_4*x27 + x14*x17 + x53*x55 + x54*x63 + (1.0/120.0)*x58 + (1.0/120.0)*x59 + (1.0/120.0)*x60 + (1.0/120.0)*x61 + (1.0/30.0)*x62) + x22*((1.0/2.0)*by_1_0 + x23*x46 + x24*x51 + x26*x52 + (1.0/2.0)*x42 + (1.0/2.0)*x43 + (1.0/2.0)*x44 + (1.0/2.0)*x45) + x29*(-1.0/12.0*by_1_2 - 1.0/12.0*by_1_3*x39 - 1.0/12.0*by_1_4*x40 - 1.0/24.0*by_3_0 - 1.0/12.0*x*(by_2_2 + by_2_3*x39 + by_2_4*x40) - 1.0/24.0*x12*x56 - 1.0/72.0*x17*x57 - x30*x52 - 1.0/24.0*x47 - 1.0/24.0*x48 - 1.0/24.0*x49 - 1.0/24.0*x50) + x3*(-1.0/70.0*bx_2_4 - 1.0/70.0*bx_3_4*x - 1.0/140.0*bx_4_4*x12 - 1.0/840.0*x53) - x32*(by_1_0 + x42 + x43 + x44 + x45) - x35*x51 - x36*x52 + x37*(-1.0/3.0*bx_0_2 - 1.0/3.0*bx_0_3*x39 - 1.0/3.0*bx_0_4*x40 - 1.0/6.0*bx_2_0 - 1.0/6.0*x*x68 - 1.0/3.0*x*(bx_1_2 + bx_1_3*x39 + bx_1_4*x40) - 1.0/6.0*x12*x62 - 1.0/18.0*x17*x63 - 1.0/72.0*x27*x53 - x41*x69 - 1.0/6.0*x64 - 1.0/6.0*x65 - 1.0/6.0*x66 - 1.0/6.0*x67) - x38*x46 + x6*((1.0/30.0)*by_1_4 + by_2_4*x54 + by_3_4*x55 + (1.0/180.0)*x*x57 + x17*x9 + (1.0/180.0)*x56) + y*(bx_0_0 + bx_0_1*s + bx_0_2*x7 + bx_0_3*x8 + bx_0_4*x31 + x*(bx_1_0 + bx_1_1*s + bx_1_2*x7 + bx_1_3*x8 + bx_1_4*x31) + x32*(bx_2_0 + x64 + x65 + x66 + x67) + x35*x69 + x38*x68);
		return;

	}
	case 6: {
		// Hermite → polynomial coefficients (order 6)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly by2_poly = hermite_to_polynomial(0.0, L, by[2]);
		const double by_2_0 = by2_poly.coeffs[0];
		const double by_2_1 = by2_poly.coeffs[1];
		const double by_2_2 = by2_poly.coeffs[2];
		const double by_2_3 = by2_poly.coeffs[3];
		const double by_2_4 = by2_poly.coeffs[4];

		const Poly by3_poly = hermite_to_polynomial(0.0, L, by[3]);
		const double by_3_0 = by3_poly.coeffs[0];
		const double by_3_1 = by3_poly.coeffs[1];
		const double by_3_2 = by3_poly.coeffs[2];
		const double by_3_3 = by3_poly.coeffs[3];
		const double by_3_4 = by3_poly.coeffs[4];

		const Poly by4_poly = hermite_to_polynomial(0.0, L, by[4]);
		const double by_4_0 = by4_poly.coeffs[0];
		const double by_4_1 = by4_poly.coeffs[1];
		const double by_4_2 = by4_poly.coeffs[2];
		const double by_4_3 = by4_poly.coeffs[3];
		const double by_4_4 = by4_poly.coeffs[4];

		const Poly by5_poly = hermite_to_polynomial(0.0, L, by[5]);
		const double by_5_0 = by5_poly.coeffs[0];
		const double by_5_1 = by5_poly.coeffs[1];
		const double by_5_2 = by5_poly.coeffs[2];
		const double by_5_3 = by5_poly.coeffs[3];
		const double by_5_4 = by5_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		const Poly bx2_poly = hermite_to_polynomial(0.0, L, bx[2]);
		const double bx_2_0 = bx2_poly.coeffs[0];
		const double bx_2_1 = bx2_poly.coeffs[1];
		const double bx_2_2 = bx2_poly.coeffs[2];
		const double bx_2_3 = bx2_poly.coeffs[3];
		const double bx_2_4 = bx2_poly.coeffs[4];

		const Poly bx3_poly = hermite_to_polynomial(0.0, L, bx[3]);
		const double bx_3_0 = bx3_poly.coeffs[0];
		const double bx_3_1 = bx3_poly.coeffs[1];
		const double bx_3_2 = bx3_poly.coeffs[2];
		const double bx_3_3 = bx3_poly.coeffs[3];
		const double bx_3_4 = bx3_poly.coeffs[4];

		const Poly bx4_poly = hermite_to_polynomial(0.0, L, bx[4]);
		const double bx_4_0 = bx4_poly.coeffs[0];
		const double bx_4_1 = bx4_poly.coeffs[1];
		const double bx_4_2 = bx4_poly.coeffs[2];
		const double bx_4_3 = bx4_poly.coeffs[3];
		const double bx_4_4 = bx4_poly.coeffs[4];

		const Poly bx5_poly = hermite_to_polynomial(0.0, L, bx[5]);
		const double bx_5_0 = bx5_poly.coeffs[0];
		const double bx_5_1 = bx5_poly.coeffs[1];
		const double bx_5_2 = bx5_poly.coeffs[2];
		const double bx_5_3 = bx5_poly.coeffs[3];
		const double bx_5_4 = bx5_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = y*y*y*y*y*y*y*y*y;
		const double x1 = 4*s;
		const double x2 = bx_5_3 + bx_5_4*x1;
		const double x3 = y*y*y*y*y*y*y*y;
		const double x4 = by_4_3 + by_4_4*x1;
		const double x5 = by_5_3 + by_5_4*x1;
		const double x6 = y*y*y*y*y*y*y;
		const double x7 = s*s;
		const double x8 = s*s*s;
		const double x9 = bx_3_3 + bx_3_4*x1;
		const double x10 = bx_4_3 + bx_4_4*x1;
		const double x11 = x*x;
		const double x12 = y*y*y*y*y*y;
		const double x13 = (1.0/180.0)*by_4_4;
		const double x14 = by_2_3 + by_2_4*x1;
		const double x15 = by_3_3 + by_3_4*x1;
		const double x16 = x*x*x;
		const double x17 = 2*s;
		const double x18 = 3*x7;
		const double x19 = 4*x8;
		const double x20 = by_5_1 + by_5_2*x17 + by_5_3*x18 + by_5_4*x19;
		const double x21 = y*y*y*y*y;
		const double x22 = (1.0/30.0)*bx_3_4;
		const double x23 = bx_1_3 + bx_1_4*x1;
		const double x24 = bx_2_3 + bx_2_4*x1;
		const double x25 = x*x*x*x;
		const double x26 = bx_4_1 + bx_4_2*x17 + bx_4_3*x18 + bx_4_4*x19;
		const double x27 = (1.0/120.0)*x;
		const double x28 = bx_5_1 + bx_5_2*x17 + bx_5_3*x18 + bx_5_4*x19;
		const double x29 = y*y;
		const double x30 = (1.0/2.0)*x;
		const double x31 = (1.0/4.0)*x11;
		const double x32 = by_3_1 + by_3_2*x17 + by_3_3*x18 + by_3_4*x19;
		const double x33 = (1.0/12.0)*x16;
		const double x34 = (1.0/48.0)*by_4_1 + (1.0/48.0)*by_4_2*x17 + (1.0/48.0)*by_4_3*x18 + (1.0/48.0)*by_4_4*x19;
		const double x35 = x*x*x*x*x;
		const double x36 = y*y*y*y;
		const double x37 = (1.0/24.0)*x;
		const double x38 = s*s*s*s;
		const double x39 = (1.0/2.0)*x11;
		const double x40 = (1.0/6.0)*bx_2_1 + (1.0/6.0)*bx_2_2*x17 + (1.0/6.0)*bx_2_3*x18 + (1.0/6.0)*bx_2_4*x19;
		const double x41 = bx_3_1 + bx_3_2*x17 + bx_3_3*x18 + bx_3_4*x19;
		const double x42 = (1.0/24.0)*x25;
		const double x43 = (1.0/120.0)*x35;
		const double x44 = (1.0/720.0)*x*x*x*x*x*x;
		const double x45 = y*y*y;
		const double x46 = (1.0/6.0)*x16;
		const double x47 = 3*s;
		const double x48 = 6*x7;
		const double x49 = (1.0/12.0)*x11;
		const double x50 = (1.0/36.0)*x16;
		const double x51 = by_1_1*s;
		const double x52 = by_1_2*x7;
		const double x53 = by_1_3*x8;
		const double x54 = by_1_4*x38;
		const double x55 = by_2_0 + by_2_1*s + by_2_2*x7 + by_2_3*x8 + by_2_4*x38;
		const double x56 = by_3_1*s;
		const double x57 = by_3_2*x7;
		const double x58 = by_3_3*x8;
		const double x59 = by_3_4*x38;
		const double x60 = by_3_0 + x56 + x57 + x58 + x59;
		const double x61 = by_4_0 + by_4_1*s + by_4_2*x7 + by_4_3*x8 + by_4_4*x38;
		const double x62 = by_5_1*s;
		const double x63 = by_5_2*x7;
		const double x64 = by_5_3*x8;
		const double x65 = by_5_4*x38;
		const double x66 = by_5_0 + x62 + x63 + x64 + x65;
		const double x67 = by_5_2 + by_5_3*x47 + by_5_4*x48;
		const double x68 = bx_4_2 + bx_4_3*x47 + bx_4_4*x48;
		const double x69 = bx_5_2 + bx_5_3*x47 + bx_5_4*x48;
		const double x70 = (1.0/30.0)*x;
		const double x71 = (1.0/60.0)*x11;
		const double x72 = by_3_2 + by_3_3*x47 + by_3_4*x48;
		const double x73 = by_4_2 + by_4_3*x47 + by_4_4*x48;
		const double x74 = (1.0/48.0)*x66;
		const double x75 = bx_2_1*s;
		const double x76 = bx_2_2*x7;
		const double x77 = bx_2_3*x8;
		const double x78 = bx_2_4*x38;
		const double x79 = bx_3_0 + bx_3_1*s + bx_3_2*x7 + bx_3_3*x8 + bx_3_4*x38;
		const double x80 = bx_4_1*s;
		const double x81 = bx_4_2*x7;
		const double x82 = bx_4_3*x8;
		const double x83 = bx_4_4*x38;
		const double x84 = bx_4_0 + x80 + x81 + x82 + x83;
		const double x85 = bx_5_0 + bx_5_1*s + bx_5_2*x7 + bx_5_3*x8 + bx_5_4*x38;
		const double x86 = bx_2_2 + bx_2_3*x47 + bx_2_4*x48;
		const double x87 = bx_3_2 + bx_3_3*x47 + bx_3_4*x48;

		// Reduced expressions
		*Ax_out = -1.0/15120.0*x0*x2 - x12*((1.0/720.0)*by_4_1 + (1.0/360.0)*by_4_2*s + (1.0/240.0)*by_4_3*x7 + (1.0/60.0)*x*x15 + (1.0/720.0)*x*x20 + (1.0/120.0)*x11*x4 + x13*x8 + (1.0/60.0)*x14 + (1.0/360.0)*x16*x5) - x21*((1.0/5.0)*bs_4 + (1.0/120.0)*bx_3_1 + (1.0/60.0)*bx_3_2*s + (1.0/40.0)*bx_3_3*x7 + (1.0/10.0)*x*x24 + (1.0/60.0)*x10*x16 + (1.0/240.0)*x11*x28 + (1.0/20.0)*x11*x9 + (1.0/240.0)*x2*x25 + x22*x8 + (1.0/10.0)*x23 + x26*x27) - x29*((1.0/2.0)*by_0_1 + by_0_2*s + (3.0/2.0)*by_0_3*x7 + 2*by_0_4*x8 + (1.0/240.0)*x20*x35 + x25*x34 + x30*(by_1_1 + by_1_2*x17 + by_1_3*x18 + by_1_4*x19) + x31*(by_2_1 + by_2_2*x17 + by_2_3*x18 + by_2_4*x19) + x32*x33) - x3*(-1.0/2240.0*x*x5 - 1.0/2240.0*x4) - x36*(-1.0/4.0*by_0_3 - 1.0/4.0*by_0_4*x1 - 1.0/24.0*by_2_1 - 1.0/12.0*by_2_2*s - 1.0/8.0*by_2_3*x7 - 1.0/6.0*by_2_4*x8 - 1.0/4.0*x*(by_1_3 + by_1_4*x1) - 1.0/8.0*x11*x14 - x11*x34 - 1.0/24.0*x15*x16 - 1.0/144.0*x16*x20 - 1.0/96.0*x25*x4 - x32*x37 - 1.0/480.0*x35*x5) - x45*(-1.0/3.0*bs_2 - 1.0/3.0*bs_3*x47 - 1.0/3.0*bs_4*x48 - 1.0/6.0*bx_1_1 - 1.0/3.0*bx_1_2*s - 1.0/2.0*bx_1_3*x7 - 2.0/3.0*bx_1_4*x8 - x*x40 - x*(bx_0_3 + bx_0_4*x1) - x10*x43 - x2*x44 - x23*x39 - x24*x46 - 1.0/144.0*x25*x28 - x26*x50 - x41*x49 - x42*x9) - x6*(-1.0/5040.0*bx_5_1 - 1.0/2520.0*bx_5_2*s - 1.0/1680.0*bx_5_3*x7 - 1.0/1260.0*bx_5_4*x8 - 1.0/280.0*x*x10 - 1.0/560.0*x11*x2 - 1.0/280.0*x9) - y*(bs_0 + bs_1*s + bs_2*x7 + bs_3*x8 + bs_4*x38 + x*(bx_0_1 + bx_0_2*x17 + bx_0_3*x18 + bx_0_4*x19) + x16*x40 + x26*x43 + x28*x44 + x39*(bx_1_1 + bx_1_2*x17 + bx_1_3*x18 + bx_1_4*x19) + x41*x42);
		*Ay_out = 0;
		*As_out = (1.0/25200.0)*by_5_4*y*y*y*y*y*y*y*y*y*y - x*(by_0_0 + by_0_1*s + by_0_2*x7 + by_0_3*x8 + by_0_4*x38) + x0*((1.0/2520.0)*bx_4_4 + (1.0/2520.0)*bx_5_4*x) + x12*((1.0/30.0)*by_1_4 + by_2_4*x70 + by_3_4*x71 + (1.0/720.0)*by_5_0 + (1.0/720.0)*by_5_4*x25 + (1.0/180.0)*x*x73 + (1.0/360.0)*x11*x67 + x13*x16 + (1.0/720.0)*x62 + (1.0/720.0)*x63 + (1.0/720.0)*x64 + (1.0/720.0)*x65 + (1.0/180.0)*x72) + x21*((1.0/5.0)*bx_0_4 + (1.0/5.0)*bx_1_4*x + (1.0/10.0)*bx_2_4*x11 + (1.0/120.0)*bx_4_0 + (1.0/120.0)*bx_4_4*x25 + (1.0/600.0)*bx_5_4*x35 + x16*x22 + (1.0/180.0)*x16*x69 + x27*x85 + x68*x71 + x70*x87 + (1.0/120.0)*x80 + (1.0/120.0)*x81 + (1.0/120.0)*x82 + (1.0/120.0)*x83 + (1.0/30.0)*x86) + x29*((1.0/2.0)*by_1_0 + x25*x74 + x30*x55 + x31*x60 + x33*x61 + (1.0/2.0)*x51 + (1.0/2.0)*x52 + (1.0/2.0)*x53 + (1.0/2.0)*x54) + x3*(-1.0/560.0*by_3_4 - 1.0/560.0*by_4_4*x - 1.0/1120.0*by_5_4*x11 - 1.0/6720.0*x67) + x36*(-1.0/12.0*by_1_2 - 1.0/12.0*by_1_3*x47 - 1.0/12.0*by_1_4*x48 - 1.0/24.0*by_3_0 - 1.0/12.0*x*(by_2_2 + by_2_3*x47 + by_2_4*x48) - 1.0/24.0*x11*x72 - x11*x74 - 1.0/72.0*x16*x73 - 1.0/288.0*x25*x67 - x37*x61 - 1.0/24.0*x56 - 1.0/24.0*x57 - 1.0/24.0*x58 - 1.0/24.0*x59) - x39*(by_1_0 + x51 + x52 + x53 + x54) - x42*x60 - x43*x61 - x44*x66 + x45*(-1.0/3.0*bx_0_2 - 1.0/3.0*bx_0_3*x47 - 1.0/3.0*bx_0_4*x48 - 1.0/6.0*bx_2_0 - 1.0/6.0*x*x79 - 1.0/3.0*x*(bx_1_2 + bx_1_3*x47 + bx_1_4*x48) - 1.0/6.0*x11*x86 - 1.0/18.0*x16*x87 - 1.0/72.0*x25*x68 - 1.0/360.0*x35*x69 - x49*x84 - x50*x85 - 1.0/6.0*x75 - 1.0/6.0*x76 - 1.0/6.0*x77 - 1.0/6.0*x78) - x46*x55 + x6*(-1.0/70.0*bx_2_4 - 1.0/70.0*bx_3_4*x - 1.0/140.0*bx_4_4*x11 - 1.0/420.0*bx_5_4*x16 - 1.0/840.0*x*x69 - 1.0/840.0*x68) + y*(bx_0_0 + bx_0_1*s + bx_0_2*x7 + bx_0_3*x8 + bx_0_4*x38 + x*(bx_1_0 + bx_1_1*s + bx_1_2*x7 + bx_1_3*x8 + bx_1_4*x38) + x39*(bx_2_0 + x75 + x76 + x77 + x78) + x42*x84 + x43*x85 + x46*x79);
		return;

	}
	case 7: {
		// Hermite → polynomial coefficients (order 7)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly by2_poly = hermite_to_polynomial(0.0, L, by[2]);
		const double by_2_0 = by2_poly.coeffs[0];
		const double by_2_1 = by2_poly.coeffs[1];
		const double by_2_2 = by2_poly.coeffs[2];
		const double by_2_3 = by2_poly.coeffs[3];
		const double by_2_4 = by2_poly.coeffs[4];

		const Poly by3_poly = hermite_to_polynomial(0.0, L, by[3]);
		const double by_3_0 = by3_poly.coeffs[0];
		const double by_3_1 = by3_poly.coeffs[1];
		const double by_3_2 = by3_poly.coeffs[2];
		const double by_3_3 = by3_poly.coeffs[3];
		const double by_3_4 = by3_poly.coeffs[4];

		const Poly by4_poly = hermite_to_polynomial(0.0, L, by[4]);
		const double by_4_0 = by4_poly.coeffs[0];
		const double by_4_1 = by4_poly.coeffs[1];
		const double by_4_2 = by4_poly.coeffs[2];
		const double by_4_3 = by4_poly.coeffs[3];
		const double by_4_4 = by4_poly.coeffs[4];

		const Poly by5_poly = hermite_to_polynomial(0.0, L, by[5]);
		const double by_5_0 = by5_poly.coeffs[0];
		const double by_5_1 = by5_poly.coeffs[1];
		const double by_5_2 = by5_poly.coeffs[2];
		const double by_5_3 = by5_poly.coeffs[3];
		const double by_5_4 = by5_poly.coeffs[4];

		const Poly by6_poly = hermite_to_polynomial(0.0, L, by[6]);
		const double by_6_0 = by6_poly.coeffs[0];
		const double by_6_1 = by6_poly.coeffs[1];
		const double by_6_2 = by6_poly.coeffs[2];
		const double by_6_3 = by6_poly.coeffs[3];
		const double by_6_4 = by6_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		const Poly bx2_poly = hermite_to_polynomial(0.0, L, bx[2]);
		const double bx_2_0 = bx2_poly.coeffs[0];
		const double bx_2_1 = bx2_poly.coeffs[1];
		const double bx_2_2 = bx2_poly.coeffs[2];
		const double bx_2_3 = bx2_poly.coeffs[3];
		const double bx_2_4 = bx2_poly.coeffs[4];

		const Poly bx3_poly = hermite_to_polynomial(0.0, L, bx[3]);
		const double bx_3_0 = bx3_poly.coeffs[0];
		const double bx_3_1 = bx3_poly.coeffs[1];
		const double bx_3_2 = bx3_poly.coeffs[2];
		const double bx_3_3 = bx3_poly.coeffs[3];
		const double bx_3_4 = bx3_poly.coeffs[4];

		const Poly bx4_poly = hermite_to_polynomial(0.0, L, bx[4]);
		const double bx_4_0 = bx4_poly.coeffs[0];
		const double bx_4_1 = bx4_poly.coeffs[1];
		const double bx_4_2 = bx4_poly.coeffs[2];
		const double bx_4_3 = bx4_poly.coeffs[3];
		const double bx_4_4 = bx4_poly.coeffs[4];

		const Poly bx5_poly = hermite_to_polynomial(0.0, L, bx[5]);
		const double bx_5_0 = bx5_poly.coeffs[0];
		const double bx_5_1 = bx5_poly.coeffs[1];
		const double bx_5_2 = bx5_poly.coeffs[2];
		const double bx_5_3 = bx5_poly.coeffs[3];
		const double bx_5_4 = bx5_poly.coeffs[4];

		const Poly bx6_poly = hermite_to_polynomial(0.0, L, bx[6]);
		const double bx_6_0 = bx6_poly.coeffs[0];
		const double bx_6_1 = bx6_poly.coeffs[1];
		const double bx_6_2 = bx6_poly.coeffs[2];
		const double bx_6_3 = bx6_poly.coeffs[3];
		const double bx_6_4 = bx6_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = y*y*y*y*y*y*y*y*y*y;
		const double x1 = 4*s;
		const double x2 = by_6_3 + by_6_4*x1;
		const double x3 = y*y*y*y*y*y*y*y*y;
		const double x4 = bx_5_3 + bx_5_4*x1;
		const double x5 = bx_6_3 + bx_6_4*x1;
		const double x6 = y*y*y*y*y*y*y*y;
		const double x7 = s*s;
		const double x8 = s*s*s;
		const double x9 = by_4_3 + by_4_4*x1;
		const double x10 = by_5_3 + by_5_4*x1;
		const double x11 = x*x;
		const double x12 = y*y*y*y*y*y*y;
		const double x13 = bx_3_3 + bx_3_4*x1;
		const double x14 = bx_4_3 + bx_4_4*x1;
		const double x15 = x*x*x;
		const double x16 = 2*s;
		const double x17 = 3*x7;
		const double x18 = 4*x8;
		const double x19 = bx_6_1 + bx_6_2*x16 + bx_6_3*x17 + bx_6_4*x18;
		const double x20 = y*y*y*y*y*y;
		const double x21 = (1.0/180.0)*by_4_4;
		const double x22 = by_2_3 + by_2_4*x1;
		const double x23 = by_3_3 + by_3_4*x1;
		const double x24 = x*x*x*x;
		const double x25 = by_5_1 + by_5_2*x16 + by_5_3*x17 + by_5_4*x18;
		const double x26 = (1.0/720.0)*x;
		const double x27 = by_6_1 + by_6_2*x16 + by_6_3*x17 + by_6_4*x18;
		const double x28 = (1.0/1440.0)*x27;
		const double x29 = y*y*y*y*y;
		const double x30 = (1.0/30.0)*bx_3_4;
		const double x31 = bx_1_3 + bx_1_4*x1;
		const double x32 = bx_2_3 + bx_2_4*x1;
		const double x33 = x*x*x*x*x;
		const double x34 = bx_4_1 + bx_4_2*x16 + bx_4_3*x17 + bx_4_4*x18;
		const double x35 = (1.0/120.0)*x;
		const double x36 = bx_5_1 + bx_5_2*x16 + bx_5_3*x17 + bx_5_4*x18;
		const double x37 = (1.0/240.0)*x11;
		const double x38 = (1.0/720.0)*x19;
		const double x39 = y*y;
		const double x40 = (1.0/2.0)*x;
		const double x41 = (1.0/4.0)*x11;
		const double x42 = by_3_1 + by_3_2*x16 + by_3_3*x17 + by_3_4*x18;
		const double x43 = (1.0/12.0)*x15;
		const double x44 = (1.0/48.0)*by_4_1 + (1.0/48.0)*by_4_2*x16 + (1.0/48.0)*by_4_3*x17 + (1.0/48.0)*by_4_4*x18;
		const double x45 = (1.0/240.0)*x33;
		const double x46 = x*x*x*x*x*x;
		const double x47 = s*s*s*s;
		const double x48 = (1.0/2.0)*x11;
		const double x49 = (1.0/6.0)*bx_2_1 + (1.0/6.0)*bx_2_2*x16 + (1.0/6.0)*bx_2_3*x17 + (1.0/6.0)*bx_2_4*x18;
		const double x50 = bx_3_1 + bx_3_2*x16 + bx_3_3*x17 + bx_3_4*x18;
		const double x51 = (1.0/24.0)*x24;
		const double x52 = (1.0/120.0)*x33;
		const double x53 = (1.0/720.0)*x46;
		const double x54 = (1.0/5040.0)*x*x*x*x*x*x*x;
		const double x55 = y*y*y*y;
		const double x56 = (1.0/24.0)*x;
		const double x57 = (1.0/144.0)*x15;
		const double x58 = y*y*y;
		const double x59 = (1.0/6.0)*x15;
		const double x60 = 3*s;
		const double x61 = 6*x7;
		const double x62 = (1.0/12.0)*x11;
		const double x63 = (1.0/36.0)*x15;
		const double x64 = (1.0/144.0)*x24;
		const double x65 = by_1_1*s;
		const double x66 = by_1_2*x7;
		const double x67 = by_1_3*x8;
		const double x68 = by_1_4*x47;
		const double x69 = by_2_0 + by_2_1*s + by_2_2*x7 + by_2_3*x8 + by_2_4*x47;
		const double x70 = by_3_1*s;
		const double x71 = by_3_2*x7;
		const double x72 = by_3_3*x8;
		const double x73 = by_3_4*x47;
		const double x74 = by_3_0 + x70 + x71 + x72 + x73;
		const double x75 = by_4_0 + by_4_1*s + by_4_2*x7 + by_4_3*x8 + by_4_4*x47;
		const double x76 = by_5_1*s;
		const double x77 = by_5_2*x7;
		const double x78 = by_5_3*x8;
		const double x79 = by_5_4*x47;
		const double x80 = by_5_0 + x76 + x77 + x78 + x79;
		const double x81 = by_6_0 + by_6_1*s + by_6_2*x7 + by_6_3*x8 + by_6_4*x47;
		const double x82 = bx_6_2 + bx_6_3*x60 + bx_6_4*x61;
		const double x83 = by_5_2 + by_5_3*x60 + by_5_4*x61;
		const double x84 = by_6_2 + by_6_3*x60 + by_6_4*x61;
		const double x85 = bx_6_1*s;
		const double x86 = bx_6_2*x7;
		const double x87 = bx_6_3*x8;
		const double x88 = bx_6_4*x47;
		const double x89 = bx_4_2 + bx_4_3*x60 + bx_4_4*x61;
		const double x90 = bx_5_2 + bx_5_3*x60 + bx_5_4*x61;
		const double x91 = (1.0/30.0)*x;
		const double x92 = (1.0/60.0)*x11;
		const double x93 = (1.0/720.0)*x24;
		const double x94 = by_3_2 + by_3_3*x60 + by_3_4*x61;
		const double x95 = by_4_2 + by_4_3*x60 + by_4_4*x61;
		const double x96 = (1.0/48.0)*x80;
		const double x97 = bx_2_1*s;
		const double x98 = bx_2_2*x7;
		const double x99 = bx_2_3*x8;
		const double x100 = bx_2_4*x47;
		const double x101 = bx_3_0 + bx_3_1*s + bx_3_2*x7 + bx_3_3*x8 + bx_3_4*x47;
		const double x102 = bx_4_1*s;
		const double x103 = bx_4_2*x7;
		const double x104 = bx_4_3*x8;
		const double x105 = bx_4_4*x47;
		const double x106 = bx_4_0 + x102 + x103 + x104 + x105;
		const double x107 = bx_5_0 + bx_5_1*s + bx_5_2*x7 + bx_5_3*x8 + bx_5_4*x47;
		const double x108 = bx_6_0 + x85 + x86 + x87 + x88;
		const double x109 = bx_2_2 + bx_2_3*x60 + bx_2_4*x61;
		const double x110 = bx_3_2 + bx_3_3*x60 + bx_3_4*x61;

		// Reduced expressions
		*Ax_out = -1.0/151200.0*x0*x2 - x12*(-1.0/5040.0*bx_5_1 - 1.0/2520.0*bx_5_2*s - 1.0/1680.0*bx_5_3*x7 - 1.0/1260.0*bx_5_4*x8 - 1.0/280.0*x*x14 - 1.0/5040.0*x*x19 - 1.0/560.0*x11*x4 - 1.0/280.0*x13 - 1.0/1680.0*x15*x5) - x20*((1.0/720.0)*by_4_1 + (1.0/360.0)*by_4_2*s + (1.0/240.0)*by_4_3*x7 + (1.0/60.0)*x*x23 + (1.0/360.0)*x10*x15 + x11*x28 + (1.0/120.0)*x11*x9 + (1.0/1440.0)*x2*x24 + x21*x8 + (1.0/60.0)*x22 + x25*x26) - x29*((1.0/5.0)*bs_4 + (1.0/120.0)*bx_3_1 + (1.0/60.0)*bx_3_2*s + (1.0/40.0)*bx_3_3*x7 + (1.0/10.0)*x*x32 + (1.0/20.0)*x11*x13 + (1.0/60.0)*x14*x15 + x15*x38 + (1.0/240.0)*x24*x4 + x30*x8 + (1.0/10.0)*x31 + (1.0/1200.0)*x33*x5 + x34*x35 + x36*x37) - x3*((1.0/15120.0)*x*x5 + (1.0/15120.0)*x4) - x39*((1.0/2.0)*by_0_1 + by_0_2*s + (3.0/2.0)*by_0_3*x7 + 2*by_0_4*x8 + x24*x44 + x25*x45 + x28*x46 + x40*(by_1_1 + by_1_2*x16 + by_1_3*x17 + by_1_4*x18) + x41*(by_2_1 + by_2_2*x16 + by_2_3*x17 + by_2_4*x18) + x42*x43) - x55*(-1.0/4.0*by_0_3 - 1.0/4.0*by_0_4*x1 - 1.0/24.0*by_2_1 - 1.0/12.0*by_2_2*s - 1.0/8.0*by_2_3*x7 - 1.0/6.0*by_2_4*x8 - 1.0/4.0*x*(by_1_3 + by_1_4*x1) - 1.0/480.0*x10*x33 - 1.0/8.0*x11*x22 - x11*x44 - 1.0/24.0*x15*x23 - 1.0/2880.0*x2*x46 - 1.0/576.0*x24*x27 - 1.0/96.0*x24*x9 - x25*x57 - x42*x56) - x58*(-1.0/3.0*bs_2 - 1.0/3.0*bs_3*x60 - 1.0/3.0*bs_4*x61 - 1.0/6.0*bx_1_1 - 1.0/3.0*bx_1_2*s - 1.0/2.0*bx_1_3*x7 - 2.0/3.0*bx_1_4*x8 - x*x49 - x*(bx_0_3 + bx_0_4*x1) - x13*x51 - x14*x52 - x31*x48 - x32*x59 - x33*x38 - x34*x63 - x36*x64 - x4*x53 - x5*x54 - x50*x62) - x6*(-1.0/40320.0*by_6_1 - 1.0/20160.0*by_6_2*s - 1.0/13440.0*by_6_3*x7 - 1.0/10080.0*by_6_4*x8 - 1.0/2240.0*x*x10 - 1.0/4480.0*x11*x2 - 1.0/2240.0*x9) - y*(bs_0 + bs_1*s + bs_2*x7 + bs_3*x8 + bs_4*x47 + x*(bx_0_1 + bx_0_2*x16 + bx_0_3*x17 + bx_0_4*x18) + x15*x49 + x19*x54 + x34*x52 + x36*x53 + x48*(bx_1_1 + bx_1_2*x16 + bx_1_3*x17 + bx_1_4*x18) + x50*x51);
		*Ay_out = 0;
		*As_out = -1.0/166320.0*bx_6_4*y*y*y*y*y*y*y*y*y*y*y - x*(by_0_0 + by_0_1*s + by_0_2*x7 + by_0_3*x8 + by_0_4*x47) + x0*((1.0/25200.0)*by_5_4 + (1.0/25200.0)*by_6_4*x) + x12*(-1.0/70.0*bx_2_4 - 1.0/70.0*bx_3_4*x - 1.0/140.0*bx_4_4*x11 - 1.0/420.0*bx_5_4*x15 - 1.0/5040.0*bx_6_0 - 1.0/1680.0*bx_6_4*x24 - 1.0/840.0*x*x90 - 1.0/1680.0*x11*x82 - 1.0/5040.0*x85 - 1.0/5040.0*x86 - 1.0/5040.0*x87 - 1.0/5040.0*x88 - 1.0/840.0*x89) + x20*((1.0/30.0)*by_1_4 + by_2_4*x91 + by_3_4*x92 + (1.0/720.0)*by_5_0 + by_5_4*x93 + (1.0/3600.0)*by_6_4*x33 + (1.0/180.0)*x*x95 + (1.0/360.0)*x11*x83 + x15*x21 + (1.0/1080.0)*x15*x84 + x26*x81 + (1.0/720.0)*x76 + (1.0/720.0)*x77 + (1.0/720.0)*x78 + (1.0/720.0)*x79 + (1.0/180.0)*x94) + x29*((1.0/5.0)*bx_0_4 + (1.0/5.0)*bx_1_4*x + (1.0/10.0)*bx_2_4*x11 + (1.0/120.0)*bx_4_0 + (1.0/120.0)*bx_4_4*x24 + (1.0/600.0)*bx_5_4*x33 + (1.0/3600.0)*bx_6_4*x46 + (1.0/120.0)*x102 + (1.0/120.0)*x103 + (1.0/120.0)*x104 + (1.0/120.0)*x105 + x107*x35 + x108*x37 + (1.0/30.0)*x109 + x110*x91 + x15*x30 + (1.0/180.0)*x15*x90 + x82*x93 + x89*x92) + x3*((1.0/2520.0)*bx_4_4 + (1.0/2520.0)*bx_5_4*x + (1.0/5040.0)*bx_6_4*x11 + (1.0/45360.0)*x82) + x39*((1.0/2.0)*by_1_0 + x24*x96 + x40*x69 + x41*x74 + x43*x75 + x45*x81 + (1.0/2.0)*x65 + (1.0/2.0)*x66 + (1.0/2.0)*x67 + (1.0/2.0)*x68) - x48*(by_1_0 + x65 + x66 + x67 + x68) - x51*x74 - x52*x75 - x53*x80 - x54*x81 + x55*(-1.0/12.0*by_1_2 - 1.0/12.0*by_1_3*x60 - 1.0/12.0*by_1_4*x61 - 1.0/24.0*by_3_0 - 1.0/12.0*x*(by_2_2 + by_2_3*x60 + by_2_4*x61) - 1.0/24.0*x11*x94 - x11*x96 - 1.0/72.0*x15*x95 - 1.0/288.0*x24*x83 - 1.0/1440.0*x33*x84 - x56*x75 - x57*x81 - 1.0/24.0*x70 - 1.0/24.0*x71 - 1.0/24.0*x72 - 1.0/24.0*x73) + x58*(-1.0/3.0*bx_0_2 - 1.0/3.0*bx_0_3*x60 - 1.0/3.0*bx_0_4*x61 - 1.0/6.0*bx_2_0 - 1.0/6.0*x*x101 - 1.0/3.0*x*(bx_1_2 + bx_1_3*x60 + bx_1_4*x61) - 1.0/6.0*x100 - x106*x62 - x107*x63 - x108*x64 - 1.0/6.0*x109*x11 - 1.0/18.0*x110*x15 - 1.0/72.0*x24*x89 - 1.0/360.0*x33*x90 - 1.0/2160.0*x46*x82 - 1.0/6.0*x97 - 1.0/6.0*x98 - 1.0/6.0*x99) - x59*x69 + x6*(-1.0/560.0*by_3_4 - 1.0/560.0*by_4_4*x - 1.0/1120.0*by_5_4*x11 - 1.0/3360.0*by_6_4*x15 - 1.0/6720.0*x*x84 - 1.0/6720.0*x83) + y*(bx_0_0 + bx_0_1*s + bx_0_2*x7 + bx_0_3*x8 + bx_0_4*x47 + x*(bx_1_0 + bx_1_1*s + bx_1_2*x7 + bx_1_3*x8 + bx_1_4*x47) + x101*x59 + x106*x51 + x107*x52 + x108*x53 + x48*(bx_2_0 + x100 + x97 + x98 + x99));
		return;

	}
	default: {
		printf("Error: Unsupported multipole order %d\n", multipole_order);
		printf("Supported orders are 1 to 7\n");
		printf("Setting field values to zero.\n");
		// Reduced expressions
		*Ax_out = 0;
		*Ay_out = 0;
		*As_out = 0;
		return;
	}
	}
}

#endif // SPLINE_A_FIELD_EVAL_H
