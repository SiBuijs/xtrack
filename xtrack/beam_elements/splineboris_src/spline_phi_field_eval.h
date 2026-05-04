#include <stddef.h>
#include <stdio.h>
#include <string.h>
#define MAX_DEGREE 4

#ifndef SPLINE_PHI_FIELD_EVAL_H
#define SPLINE_PHI_FIELD_EVAL_H

// Auto-generated symbolic field expressions for phi
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
typedef struct {
	double coeffs[MAX_DEGREE + 1]; /* coeffs[i] = coefficient of x^i */
	int degree;
} Poly;

static inline Poly poly_scale(Poly p, double s) {
	for (int i = 0; i <= p.degree; i++) p.coeffs[i] *= s;
	return p;
}

static inline Poly poly_add(Poly a, Poly b) {
	Poly result = {0};
	result.degree = a.degree > b.degree ? a.degree : b.degree;
	for (int i = 0; i <= a.degree; i++) result.coeffs[i] += a.coeffs[i];
	for (int i = 0; i <= b.degree; i++) result.coeffs[i] += b.coeffs[i];
	return result;
}

static inline Poly poly_mul(Poly a, Poly b) {
	Poly result = {0};
	int deg = a.degree + b.degree;
	if (deg > MAX_DEGREE)
		deg = MAX_DEGREE;
	result.degree = deg;
	for (int i = 0; i <= a.degree; i++) {
		for (int j = 0; j <= b.degree; j++) {
			int k = i + j;
			if (k <= MAX_DEGREE)
				result.coeffs[k] += a.coeffs[i] * b.coeffs[j];
		}
	}
	return result;
}

/* Compose f(g(x)) via Horner's method:
   result = f[n] * g^n + ... + f[0]
		  = f[0] + g*(f[1] + g*(f[2] + ... + g*f[n]))  */
static inline Poly poly_compose(Poly f, Poly g) {
	Poly result = {0};
	result.coeffs[0] = f.coeffs[f.degree]; /* start with leading coeff */
	result.degree = 0;
	for (int i = f.degree - 1; i >= 0; i--) {
		result = poly_mul(result, g);       /* result = result * g      */
		if (result.degree < MAX_DEGREE) {
			result.degree++;
		}
		result.coeffs[0] += f.coeffs[i];   /* result = result * g + f[i] */
	}
	return result;
}

static inline Poly hermite_to_polynomial(double s_start, double s_end, const double coeffs[5]) {
	double c1 = coeffs[0], c2 = coeffs[1], c3 = coeffs[2];
	double c4 = coeffs[3], c5 = coeffs[4];
	double L = s_end - s_start;

	/* t(s_local) = s_local / L */
	Poly t = { .coeffs = {0.0, 1.0/L}, .degree = 1 };

	/* Hermite basis polynomials in t on [0,1] */
	Poly b1 = { .coeffs = { 1,  0,  -18,   32,  -15}, .degree = 4 };
	Poly b2 = { .coeffs = { 0,  1, -4.5,    6, -2.5}, .degree = 4 };
	Poly b3 = { .coeffs = { 0,  0,  -12,   28,  -15}, .degree = 4 };
	Poly b4 = { .coeffs = { 0,  0,  1.5,   -4,  2.5}, .degree = 4 };
	Poly b5 = { .coeffs = { 0,  0,   30,  -60,   30}, .degree = 4 };

	/* poly_t = c1*b1 + L*c2*b2 + c3*b3 + L*c4*b4 + c5*b5 */
	Poly poly_t = {0};
	poly_t = poly_add(poly_t, poly_scale(b1, c1));
	poly_t = poly_add(poly_t, poly_scale(b2, L * c2));
	poly_t = poly_add(poly_t, poly_scale(b3, c3));
	poly_t = poly_add(poly_t, poly_scale(b4, L * c4));
	poly_t = poly_add(poly_t, poly_scale(b5, c5));

	/* poly_s(s_local) = poly_t(t(s_local)) */
	return poly_compose(poly_t, t);
}

GPUFUN
void evaluate_phi(const double x, const double y, const double s,
                  const double *bs,
                  const double *const *by,
                  const double *const *bx,
                  const double L,
                  const int multipole_order,
                  double *phi_out){

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
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = 3*s;
		const double x4 = 6*x0;
		const double x5 = 4*bs_4;

		// Reduced expressions
		*phi_out = bs_0*s + (1.0/2.0)*bs_1*x0 + (1.0/3.0)*bs_2*x1 + (1.0/4.0)*bs_3*x2 + (1.0/5.0)*bs_4*s*s*s*s*s + (1.0/5.0)*by_0_4*y*y*y*y*y + x*(bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2) + (1.0/24.0)*y*y*y*y*(6*bs_3 + 24*bx_0_4*x + 6*s*x5) - 1.0/3.0*y*y*y*(by_0_2 + by_0_3*x3 + by_0_4*x4) - 1.0/2.0*y*y*(bs_1 + 2*bs_2*s + 3*bs_3*x0 + 2*x*(bx_0_2 + bx_0_3*x3 + bx_0_4*x4) + x1*x5) + y*(by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2);
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
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = x*x;
		const double x4 = bx_1_1*s;
		const double x5 = bx_1_2*x0;
		const double x6 = bx_1_3*x1;
		const double x7 = bx_1_4*x2;
		const double x8 = 3*s;
		const double x9 = 6*x0;
		const double x10 = 4*s;
		const double x11 = bx_1_2 + bx_1_3*x8 + bx_1_4*x9;

		// Reduced expressions
		*phi_out = bs_0*s + (1.0/2.0)*bs_1*x0 + (1.0/3.0)*bs_2*x1 + (1.0/4.0)*bs_3*x2 + (1.0/5.0)*bs_4*s*s*s*s*s - 1.0/10.0*bx_1_4*y*y*y*y*y*y + x*(bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2) + (1.0/2.0)*x3*(bx_1_0 + x4 + x5 + x6 + x7) + (1.0/120.0)*y*y*y*y*y*(24*by_0_4 + 24*by_1_4*x) + (1.0/48.0)*y*y*y*y*(12*bs_3 + 12*bs_4*x10 + 48*bx_0_4*x + 24*bx_1_4*x3 + 8*x11) - 1.0/6.0*y*y*y*(2*by_0_2 + 2*by_0_3*x8 + 2*by_0_4*x9 + 2*x*(by_1_2 + by_1_3*x8 + by_1_4*x9)) - 1.0/4.0*y*y*(2*bs_1 + bs_2*x10 + bs_3*x9 + 8*bs_4*x1 + 2*bx_1_0 + 4*x*(bx_0_2 + bx_0_3*x8 + bx_0_4*x9) + 2*x11*x3 + 2*x4 + 2*x5 + 2*x6 + 2*x7) + y*(by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + x*(by_1_0 + by_1_1*s + by_1_2*x0 + by_1_3*x1 + by_1_4*x2));
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
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = x*x;
		const double x4 = bx_1_1*s;
		const double x5 = bx_1_2*x0;
		const double x6 = bx_1_3*x1;
		const double x7 = bx_1_4*x2;
		const double x8 = x*x*x;
		const double x9 = bx_2_0 + bx_2_1*s + bx_2_2*x0 + bx_2_3*x1 + bx_2_4*x2;
		const double x10 = 3*s;
		const double x11 = 6*x0;
		const double x12 = by_2_2 + by_2_3*x10 + by_2_4*x11;
		const double x13 = bx_1_2 + bx_1_3*x10 + bx_1_4*x11;
		const double x14 = bx_2_2 + bx_2_3*x10 + bx_2_4*x11;
		const double x15 = by_2_1*s;
		const double x16 = by_2_2*x0;
		const double x17 = by_2_3*x1;
		const double x18 = by_2_4*x2;

		// Reduced expressions
		*phi_out = bs_0*s + (1.0/2.0)*bs_1*x0 + (1.0/3.0)*bs_2*x1 + (1.0/4.0)*bs_3*x2 + (1.0/5.0)*bs_4*s*s*s*s*s - 1.0/70.0*by_2_4*y*y*y*y*y*y*y + x*(bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2) + (1.0/2.0)*x3*(bx_1_0 + x4 + x5 + x6 + x7) + (1.0/6.0)*x8*x9 - 1.0/4320.0*y*y*y*y*y*y*(432*bx_1_4 + 432*bx_2_4*x) + (1.0/240.0)*y*y*y*y*y*(48*by_0_4 + 48*by_1_4*x + 24*by_2_4*x3 + 8*x12) + (1.0/144.0)*y*y*y*y*(144*bx_0_4*x + 72*bx_1_4*x3 + 24*bx_2_4*x8 + 24*x*x14 + 24*x13 + 36*(bs_3 + 4*bs_4*s)) - 1.0/12.0*y*y*y*(4*by_0_2 + 4*by_0_3*x10 + 4*by_0_4*x11 + 2*by_2_0 + 4*x*(by_1_2 + by_1_3*x10 + by_1_4*x11) + 2*x12*x3 + 2*x15 + 2*x16 + 2*x17 + 2*x18) - 1.0/12.0*y*y*(6*bs_1 + 12*bs_2*s + 18*bs_3*x0 + 24*bs_4*x1 + 6*bx_1_0 + 6*x*x9 + 12*x*(bx_0_2 + bx_0_3*x10 + bx_0_4*x11) + 6*x13*x3 + 2*x14*x8 + 6*x4 + 6*x5 + 6*x6 + 6*x7) + (1.0/2.0)*y*(2*by_0_0 + 2*by_0_1*s + 2*by_0_2*x0 + 2*by_0_3*x1 + 2*by_0_4*x2 + 2*x*(by_1_0 + by_1_1*s + by_1_2*x0 + by_1_3*x1 + by_1_4*x2) + x3*(by_2_0 + x15 + x16 + x17 + x18));
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
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = x*x;
		const double x4 = bx_1_1*s;
		const double x5 = bx_1_2*x0;
		const double x6 = bx_1_3*x1;
		const double x7 = bx_1_4*x2;
		const double x8 = x*x*x;
		const double x9 = bx_2_0 + bx_2_1*s + bx_2_2*x0 + bx_2_3*x1 + bx_2_4*x2;
		const double x10 = x*x*x*x;
		const double x11 = bx_3_1*s;
		const double x12 = bx_3_2*x0;
		const double x13 = bx_3_3*x1;
		const double x14 = bx_3_4*x2;
		const double x15 = bx_3_0 + x11 + x12 + x13 + x14;
		const double x16 = 3*s;
		const double x17 = 6*x0;
		const double x18 = bx_3_2 + bx_3_3*x16 + bx_3_4*x17;
		const double x19 = by_2_2 + by_2_3*x16 + by_2_4*x17;
		const double x20 = by_3_2 + by_3_3*x16 + by_3_4*x17;
		const double x21 = 24*x;
		const double x22 = 6*x;
		const double x23 = by_3_0 + by_3_1*s + by_3_2*x0 + by_3_3*x1 + by_3_4*x2;
		const double x24 = by_2_1*s;
		const double x25 = by_2_2*x0;
		const double x26 = by_2_3*x1;
		const double x27 = by_2_4*x2;
		const double x28 = bx_1_2 + bx_1_3*x16 + bx_1_4*x17;
		const double x29 = bx_2_2 + bx_2_3*x16 + bx_2_4*x17;

		// Reduced expressions
		*phi_out = bs_0*s + (1.0/2.0)*bs_1*x0 + (1.0/3.0)*bs_2*x1 + (1.0/4.0)*bs_3*x2 + (1.0/5.0)*bs_4*s*s*s*s*s + (1.0/280.0)*bx_3_4*y*y*y*y*y*y*y*y + x*(bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2) + (1.0/24.0)*x10*x15 + (1.0/2.0)*x3*(bx_1_0 + x4 + x5 + x6 + x7) + (1.0/6.0)*x8*x9 - 1.0/30240.0*y*y*y*y*y*y*y*(432*by_2_4 + 432*by_3_4*x) - 1.0/17280.0*y*y*y*y*y*y*(1728*bx_1_4 + 1728*bx_2_4*x + 864*bx_3_4*x3 + 144*x18) + (1.0/720.0)*y*y*y*y*y*(144*by_0_4 + 144*by_1_4*x + 72*by_2_4*x3 + 24*by_3_4*x8 + 24*x19 + x20*x21) + (1.0/576.0)*y*y*y*y*(576*bx_0_4*x + 288*bx_1_4*x3 + 96*bx_2_4*x8 + 24*bx_3_0 + 24*bx_3_4*x10 + 96*x*x29 + 24*x11 + 24*x12 + 24*x13 + 24*x14 + 48*x18*x3 + 96*x28 + 144*(bs_3 + 4*bs_4*s)) - 1.0/36.0*y*y*y*(12*by_0_2 + 12*by_0_3*x16 + 12*by_0_4*x17 + 6*by_2_0 + 12*x*(by_1_2 + by_1_3*x16 + by_1_4*x17) + 6*x19*x3 + 2*x20*x8 + x22*x23 + 6*x24 + 6*x25 + 6*x26 + 6*x27) - 1.0/48.0*y*y*(24*bs_1 + 48*bs_2*s + 72*bs_3*x0 + 96*bs_4*x1 + 24*bx_1_0 + 48*x*(bx_0_2 + bx_0_3*x16 + bx_0_4*x17) + 2*x10*x18 + 12*x15*x3 + x21*x9 + 24*x28*x3 + 8*x29*x8 + 24*x4 + 24*x5 + 24*x6 + 24*x7) + (1.0/6.0)*y*(6*by_0_0 + 6*by_0_1*s + by_0_2*x17 + 6*by_0_3*x1 + 6*by_0_4*x2 + x22*(by_1_0 + by_1_1*s + by_1_2*x0 + by_1_3*x1 + by_1_4*x2) + x23*x8 + 3*x3*(by_2_0 + x24 + x25 + x26 + x27));
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
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = x*x;
		const double x4 = bx_1_1*s;
		const double x5 = bx_1_2*x0;
		const double x6 = bx_1_3*x1;
		const double x7 = bx_1_4*x2;
		const double x8 = x*x*x;
		const double x9 = bx_2_0 + bx_2_1*s + bx_2_2*x0 + bx_2_3*x1 + bx_2_4*x2;
		const double x10 = x*x*x*x;
		const double x11 = bx_3_1*s;
		const double x12 = bx_3_2*x0;
		const double x13 = bx_3_3*x1;
		const double x14 = bx_3_4*x2;
		const double x15 = bx_3_0 + x11 + x12 + x13 + x14;
		const double x16 = x*x*x*x*x;
		const double x17 = bx_4_0 + bx_4_1*s + bx_4_2*x0 + bx_4_3*x1 + bx_4_4*x2;
		const double x18 = 3*s;
		const double x19 = 6*x0;
		const double x20 = by_4_2 + by_4_3*x18 + by_4_4*x19;
		const double x21 = bx_3_2 + bx_3_3*x18 + bx_3_4*x19;
		const double x22 = bx_4_2 + bx_4_3*x18 + bx_4_4*x19;
		const double x23 = by_4_1*s;
		const double x24 = by_4_2*x0;
		const double x25 = by_4_3*x1;
		const double x26 = by_4_4*x2;
		const double x27 = by_2_2 + by_2_3*x18 + by_2_4*x19;
		const double x28 = by_3_2 + by_3_3*x18 + by_3_4*x19;
		const double x29 = 24*x;
		const double x30 = by_4_0 + x23 + x24 + x25 + x26;
		const double x31 = by_2_1*s;
		const double x32 = by_2_2*x0;
		const double x33 = by_2_3*x1;
		const double x34 = by_2_4*x2;
		const double x35 = 12*x3;
		const double x36 = by_3_0 + by_3_1*s + by_3_2*x0 + by_3_3*x1 + by_3_4*x2;
		const double x37 = bx_1_2 + bx_1_3*x18 + bx_1_4*x19;
		const double x38 = bx_2_2 + bx_2_3*x18 + bx_2_4*x19;
		const double x39 = 120*x;

		// Reduced expressions
		*phi_out = bs_0*s + (1.0/2.0)*bs_1*x0 + (1.0/3.0)*bs_2*x1 + (1.0/4.0)*bs_3*x2 + (1.0/5.0)*bs_4*s*s*s*s*s + (1.0/2520.0)*by_4_4*y*y*y*y*y*y*y*y*y + x*(bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2) + (1.0/24.0)*x10*x15 + (1.0/120.0)*x16*x17 + (1.0/2.0)*x3*(bx_1_0 + x4 + x5 + x6 + x7) + (1.0/6.0)*x8*x9 + (1.0/4838400.0)*y*y*y*y*y*y*y*y*(17280*bx_3_4 + 17280*bx_4_4*x) - 1.0/120960.0*y*y*y*y*y*y*y*(1728*by_2_4 + 1728*by_3_4*x + 864*by_4_4*x3 + 144*x20) - 1.0/86400.0*y*y*y*y*y*y*(8640*bx_1_4 + 8640*bx_2_4*x + 4320*bx_3_4*x3 + 1440*bx_4_4*x8 + 720*x*x22 + 720*x21) + (1.0/2880.0)*y*y*y*y*y*(576*by_0_4 + 576*by_1_4*x + 288*by_2_4*x3 + 96*by_3_4*x8 + 24*by_4_0 + 24*by_4_4*x10 + 96*x*x28 + 48*x20*x3 + 24*x23 + 24*x24 + 24*x25 + 24*x26 + 96*x27) + (1.0/2880.0)*y*y*y*y*(2880*bx_0_4*x + 1440*bx_1_4*x3 + 480*bx_2_4*x8 + 120*bx_3_0 + 120*bx_3_4*x10 + 24*bx_4_4*x16 + 480*x*x38 + 120*x11 + 120*x12 + 120*x13 + 120*x14 + x17*x39 + 240*x21*x3 + 80*x22*x8 + 480*x37 + 720*(bs_3 + 4*bs_4*s)) - 1.0/144.0*y*y*y*(48*by_0_2 + 48*by_0_3*x18 + 48*by_0_4*x19 + 24*by_2_0 + 48*x*(by_1_2 + by_1_3*x18 + by_1_4*x19) + 2*x10*x20 + 24*x27*x3 + 8*x28*x8 + x29*x36 + x30*x35 + 24*x31 + 24*x32 + 24*x33 + 24*x34) - 1.0/240.0*y*y*(120*bs_1 + 240*bs_2*s + 360*bs_3*x0 + 480*bs_4*x1 + 120*bx_1_0 + 240*x*(bx_0_2 + bx_0_3*x18 + bx_0_4*x19) + 10*x10*x21 + 60*x15*x3 + 2*x16*x22 + 20*x17*x8 + 120*x3*x37 + 40*x38*x8 + x39*x9 + 120*x4 + 120*x5 + 120*x6 + 120*x7) + (1.0/24.0)*y*(24*by_0_0 + 24*by_0_1*s + 24*by_0_2*x0 + 24*by_0_3*x1 + 24*by_0_4*x2 + x10*x30 + x29*(by_1_0 + by_1_1*s + by_1_2*x0 + by_1_3*x1 + by_1_4*x2) + x35*(by_2_0 + x31 + x32 + x33 + x34) + 4*x36*x8);
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
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = 17280*x;
		const double x4 = x*x;
		const double x5 = bx_1_1*s;
		const double x6 = bx_1_2*x0;
		const double x7 = bx_1_3*x1;
		const double x8 = bx_1_4*x2;
		const double x9 = x*x*x;
		const double x10 = bx_2_0 + bx_2_1*s + bx_2_2*x0 + bx_2_3*x1 + bx_2_4*x2;
		const double x11 = x*x*x*x;
		const double x12 = bx_3_1*s;
		const double x13 = bx_3_2*x0;
		const double x14 = bx_3_3*x1;
		const double x15 = bx_3_4*x2;
		const double x16 = bx_3_0 + x12 + x13 + x14 + x15;
		const double x17 = x*x*x*x*x;
		const double x18 = bx_4_0 + bx_4_1*s + bx_4_2*x0 + bx_4_3*x1 + bx_4_4*x2;
		const double x19 = x*x*x*x*x*x;
		const double x20 = bx_5_1*s;
		const double x21 = bx_5_2*x0;
		const double x22 = bx_5_3*x1;
		const double x23 = bx_5_4*x2;
		const double x24 = bx_5_0 + x20 + x21 + x22 + x23;
		const double x25 = 3*s;
		const double x26 = 6*x0;
		const double x27 = bx_5_2 + bx_5_3*x25 + bx_5_4*x26;
		const double x28 = by_4_2 + by_4_3*x25 + by_4_4*x26;
		const double x29 = by_5_2 + by_5_3*x25 + by_5_4*x26;
		const double x30 = 720*x;
		const double x31 = bx_3_2 + bx_3_3*x25 + bx_3_4*x26;
		const double x32 = bx_4_2 + bx_4_3*x25 + bx_4_4*x26;
		const double x33 = 2880*x;
		const double x34 = by_4_1*s;
		const double x35 = 1440*x4;
		const double x36 = 480*x9;
		const double x37 = by_4_2*x0;
		const double x38 = by_4_3*x1;
		const double x39 = by_4_4*x2;
		const double x40 = 120*x11;
		const double x41 = by_2_2 + by_2_3*x25 + by_2_4*x26;
		const double x42 = by_3_2 + by_3_3*x25 + by_3_4*x26;
		const double x43 = by_5_0 + by_5_1*s + by_5_2*x0 + by_5_3*x1 + by_5_4*x2;
		const double x44 = 120*x;
		const double x45 = by_2_1*s;
		const double x46 = by_2_2*x0;
		const double x47 = by_2_3*x1;
		const double x48 = by_2_4*x2;
		const double x49 = 60*x4;
		const double x50 = by_3_0 + by_3_1*s + by_3_2*x0 + by_3_3*x1 + by_3_4*x2;
		const double x51 = 20*x9;
		const double x52 = by_4_0 + x34 + x37 + x38 + x39;
		const double x53 = bx_1_2 + bx_1_3*x25 + bx_1_4*x26;
		const double x54 = bx_2_2 + bx_2_3*x25 + bx_2_4*x26;
		const double x55 = 360*x4;

		// Reduced expressions
		*phi_out = bs_0*s + (1.0/2.0)*bs_1*x0 + (1.0/3.0)*bs_2*x1 + (1.0/4.0)*bs_3*x2 + (1.0/5.0)*bs_4*s*s*s*s*s - 1.0/15120.0*bx_5_4*y*y*y*y*y*y*y*y*y*y + x*(bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2) + (1.0/6.0)*x10*x9 + (1.0/24.0)*x11*x16 + (1.0/120.0)*x17*x18 + (1.0/720.0)*x19*x24 + (1.0/2.0)*x4*(bx_1_0 + x5 + x6 + x7 + x8) + (1.0/43545600.0)*y*y*y*y*y*y*y*y*y*(17280*by_4_4 + by_5_4*x3) + (1.0/29030400.0)*y*y*y*y*y*y*y*y*(103680*bx_3_4 + 103680*bx_4_4*x + 51840*bx_5_4*x4 + 5760*x27) - 1.0/604800.0*y*y*y*y*y*y*y*(8640*by_2_4 + 8640*by_3_4*x + 4320*by_4_4*x4 + 1440*by_5_4*x9 + 720*x28 + x29*x30) - 1.0/518400.0*y*y*y*y*y*y*(51840*bx_1_4 + 51840*bx_2_4*x + 25920*bx_3_4*x4 + 8640*bx_4_4*x9 + 720*bx_5_0 + 2160*bx_5_4*x11 + 4320*x*x32 + 720*x20 + 720*x21 + 720*x22 + 720*x23 + 2160*x27*x4 + 4320*x31) + (1.0/14400.0)*y*y*y*y*y*(2880*by_0_4 + by_1_4*x33 + by_2_4*x35 + by_3_4*x36 + 120*by_4_0 + by_4_4*x40 + 24*by_5_4*x17 + 480*x*x42 + 240*x28*x4 + 80*x29*x9 + 120*x34 + 120*x37 + 120*x38 + 120*x39 + 480*x41 + x43*x44) + (1.0/17280.0)*y*y*y*y*(bx_0_4*x3 + 8640*bx_1_4*x4 + 2880*bx_2_4*x9 + 720*bx_3_0 + 720*bx_3_4*x11 + 144*bx_4_4*x17 + 24*bx_5_4*x19 + 720*x12 + 720*x13 + 720*x14 + 720*x15 + x18*x30 + x24*x55 + x27*x40 + x31*x35 + x32*x36 + x33*x54 + 2880*x53 + 4320*(bs_3 + 4*bs_4*s)) - 1.0/720.0*y*y*y*(240*by_0_2 + 240*by_0_3*x25 + 240*by_0_4*x26 + 120*by_2_0 + 240*x*(by_1_2 + by_1_3*x25 + by_1_4*x26) + 10*x11*x28 + 2*x17*x29 + 120*x4*x41 + 40*x42*x9 + x43*x51 + x44*x50 + 120*x45 + 120*x46 + 120*x47 + 120*x48 + x49*x52) - 1.0/1440.0*y*y*(720*bs_1 + 1440*bs_2*s + 2160*bs_3*x0 + 2880*bs_4*x1 + 720*bx_1_0 + 1440*x*(bx_0_2 + bx_0_3*x25 + bx_0_4*x26) + x10*x30 + 30*x11*x24 + 60*x11*x31 + x16*x55 + 12*x17*x32 + 120*x18*x9 + 2*x19*x27 + 720*x4*x53 + 720*x5 + 240*x54*x9 + 720*x6 + 720*x7 + 720*x8) + (1.0/120.0)*y*(120*by_0_0 + 120*by_0_1*s + 120*by_0_2*x0 + 120*by_0_3*x1 + 120*by_0_4*x2 + 5*x11*x52 + x17*x43 + x44*(by_1_0 + by_1_1*s + by_1_2*x0 + by_1_3*x1 + by_1_4*x2) + x49*(by_2_0 + x45 + x46 + x47 + x48) + x50*x51);
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
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = x*x;
		const double x4 = bx_1_1*s;
		const double x5 = bx_1_2*x0;
		const double x6 = bx_1_3*x1;
		const double x7 = bx_1_4*x2;
		const double x8 = x*x*x;
		const double x9 = bx_2_0 + bx_2_1*s + bx_2_2*x0 + bx_2_3*x1 + bx_2_4*x2;
		const double x10 = x*x*x*x;
		const double x11 = bx_3_1*s;
		const double x12 = bx_3_2*x0;
		const double x13 = bx_3_3*x1;
		const double x14 = bx_3_4*x2;
		const double x15 = bx_3_0 + x11 + x12 + x13 + x14;
		const double x16 = x*x*x*x*x;
		const double x17 = bx_4_0 + bx_4_1*s + bx_4_2*x0 + bx_4_3*x1 + bx_4_4*x2;
		const double x18 = x*x*x*x*x*x;
		const double x19 = bx_5_1*s;
		const double x20 = bx_5_2*x0;
		const double x21 = bx_5_3*x1;
		const double x22 = bx_5_4*x2;
		const double x23 = bx_5_0 + x19 + x20 + x21 + x22;
		const double x24 = x*x*x*x*x*x*x;
		const double x25 = bx_6_0 + bx_6_1*s + bx_6_2*x0 + bx_6_3*x1 + bx_6_4*x2;
		const double x26 = 3*s;
		const double x27 = 6*x0;
		const double x28 = by_6_2 + by_6_3*x26 + by_6_4*x27;
		const double x29 = bx_5_2 + bx_5_3*x26 + bx_5_4*x27;
		const double x30 = bx_6_2 + bx_6_3*x26 + bx_6_4*x27;
		const double x31 = by_6_1*s;
		const double x32 = by_6_2*x0;
		const double x33 = by_6_3*x1;
		const double x34 = by_6_4*x2;
		const double x35 = by_4_2 + by_4_3*x26 + by_4_4*x27;
		const double x36 = by_5_2 + by_5_3*x26 + by_5_4*x27;
		const double x37 = bx_3_2 + bx_3_3*x26 + bx_3_4*x27;
		const double x38 = bx_4_2 + bx_4_3*x26 + bx_4_4*x27;
		const double x39 = 5040*x;
		const double x40 = 720*x;
		const double x41 = by_6_0 + x31 + x32 + x33 + x34;
		const double x42 = by_2_1*s;
		const double x43 = by_2_2*x0;
		const double x44 = by_2_3*x1;
		const double x45 = by_2_4*x2;
		const double x46 = 360*x3;
		const double x47 = by_3_0 + by_3_1*s + by_3_2*x0 + by_3_3*x1 + by_3_4*x2;
		const double x48 = 120*x8;
		const double x49 = by_4_1*s;
		const double x50 = by_4_2*x0;
		const double x51 = by_4_3*x1;
		const double x52 = by_4_4*x2;
		const double x53 = by_4_0 + x49 + x50 + x51 + x52;
		const double x54 = 30*x10;
		const double x55 = by_5_0 + by_5_1*s + by_5_2*x0 + by_5_3*x1 + by_5_4*x2;
		const double x56 = by_2_2 + by_2_3*x26 + by_2_4*x27;
		const double x57 = by_3_2 + by_3_3*x26 + by_3_4*x27;
		const double x58 = bx_1_2 + bx_1_3*x26 + bx_1_4*x27;
		const double x59 = bx_2_2 + bx_2_3*x26 + bx_2_4*x27;
		const double x60 = 2520*x3;
		const double x61 = 840*x8;

		// Reduced expressions
		*phi_out = bs_0*s + (1.0/2.0)*bs_1*x0 + (1.0/3.0)*bs_2*x1 + (1.0/4.0)*bs_3*x2 + (1.0/5.0)*bs_4*s*s*s*s*s - 1.0/166320.0*by_6_4*y*y*y*y*y*y*y*y*y*y*y + x*(bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2) + (1.0/24.0)*x10*x15 + (1.0/120.0)*x16*x17 + (1.0/720.0)*x18*x23 + (1.0/5040.0)*x24*x25 + (1.0/2.0)*x3*(bx_1_0 + x4 + x5 + x6 + x7) + (1.0/6.0)*x8*x9 - 1.0/18289152000.0*y*y*y*y*y*y*y*y*y*y*(1209600*bx_5_4 + 1209600*bx_6_4*x) + (1.0/261273600.0)*y*y*y*y*y*y*y*y*y*(103680*by_4_4 + 103680*by_5_4*x + 51840*by_6_4*x3 + 5760*x28) + (1.0/203212800.0)*y*y*y*y*y*y*y*y*(725760*bx_3_4 + 725760*bx_4_4*x + 362880*bx_5_4*x3 + 120960*bx_6_4*x8 + 40320*x*x30 + 40320*x29) - 1.0/3628800.0*y*y*y*y*y*y*y*(51840*by_2_4 + 51840*by_3_4*x + 25920*by_4_4*x3 + 8640*by_5_4*x8 + 720*by_6_0 + 2160*by_6_4*x10 + 4320*x*x36 + 2160*x28*x3 + 720*x31 + 720*x32 + 720*x33 + 720*x34 + 4320*x35) - 1.0/3628800.0*y*y*y*y*y*y*(362880*bx_1_4 + 362880*bx_2_4*x + 181440*bx_3_4*x3 + 60480*bx_4_4*x8 + 5040*bx_5_0 + 15120*bx_5_4*x10 + 3024*bx_6_4*x16 + 30240*x*x38 + 5040*x19 + 5040*x20 + 5040*x21 + 5040*x22 + x25*x39 + 15120*x29*x3 + 5040*x30*x8 + 30240*x37) + (1.0/86400.0)*y*y*y*y*y*(17280*by_0_4 + 17280*by_1_4*x + 8640*by_2_4*x3 + 2880*by_3_4*x8 + 720*by_4_0 + 720*by_4_4*x10 + 144*by_5_4*x16 + 24*by_6_4*x18 + 2880*x*x57 + 120*x10*x28 + 1440*x3*x35 + 480*x36*x8 + x40*x55 + x41*x46 + 720*x49 + 720*x50 + 720*x51 + 720*x52 + 2880*x56) + (1.0/120960.0)*y*y*y*y*(120960*bx_0_4*x + 60480*bx_1_4*x3 + 20160*bx_2_4*x8 + 5040*bx_3_0 + 5040*bx_3_4*x10 + 1008*bx_4_4*x16 + 168*bx_5_4*x18 + 24*bx_6_4*x24 + 20160*x*x59 + 840*x10*x29 + 5040*x11 + 5040*x12 + 5040*x13 + 5040*x14 + 168*x16*x30 + x17*x39 + x23*x60 + x25*x61 + 10080*x3*x37 + 3360*x38*x8 + 20160*x58 + 30240*(bs_3 + 4*bs_4*s)) - 1.0/4320.0*y*y*y*(1440*by_0_2 + 1440*by_0_3*x26 + 1440*by_0_4*x27 + 720*by_2_0 + 1440*x*(by_1_2 + by_1_3*x26 + by_1_4*x27) + 60*x10*x35 + 12*x16*x36 + 2*x18*x28 + 720*x3*x56 + x40*x47 + x41*x54 + 720*x42 + 720*x43 + 720*x44 + 720*x45 + x46*x53 + x48*x55 + 240*x57*x8) - 1.0/10080.0*y*y*(5040*bs_1 + 10080*bs_2*s + 15120*bs_3*x0 + 20160*bs_4*x1 + 5040*bx_1_0 + 10080*x*(bx_0_2 + bx_0_3*x26 + bx_0_4*x27) + 210*x10*x23 + 420*x10*x37 + x15*x60 + 42*x16*x25 + 84*x16*x38 + x17*x61 + 14*x18*x29 + 2*x24*x30 + 5040*x3*x58 + x39*x9 + 5040*x4 + 5040*x5 + 1680*x59*x8 + 5040*x6 + 5040*x7) + (1.0/720.0)*y*(720*by_0_0 + 720*by_0_1*s + 720*by_0_2*x0 + 720*by_0_3*x1 + 720*by_0_4*x2 + 6*x16*x55 + x18*x41 + x40*(by_1_0 + by_1_1*s + by_1_2*x0 + by_1_3*x1 + by_1_4*x2) + x46*(by_2_0 + x42 + x43 + x44 + x45) + x47*x48 + x53*x54);
		return;

	}
	default: {
		printf("Error: Unsupported multipole order %d\n", multipole_order);
		printf("Supported orders are 1 to 7\n");
		printf("Setting field values to zero.\n");
		// Reduced expressions
		*phi_out = 0;
		return;
	}
	}
}

#endif // SPLINE_PHI_FIELD_EVAL_H
