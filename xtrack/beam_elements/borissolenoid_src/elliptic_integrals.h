// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_ELLIPTIC_INTEGRALS_H
#define XTRACK_ELLIPTIC_INTEGRALS_H

#include <math.h>

#ifndef GPUFUN
#define GPUFUN
#endif

// Carlson symmetric elliptic integrals (TOMS algorithm 577, Carlson 1979).
// Complete elliptic integrals K(m), E(m), Pi(n,m) use scipy-compatible m.

#ifndef ELLIPTIC_INTEGRALS_ERRTOL
#define ELLIPTIC_INTEGRALS_ERRTOL 1e-11
#endif

#ifndef ELLIPTIC_INTEGRALS_MAX_ITER
#define ELLIPTIC_INTEGRALS_MAX_ITER 100
#endif

GPUFUN
double elliptic_max3(const double a, const double b, const double c) {
    double m = a;
    if (fabs(b) > fabs(m)) m = b;
    if (fabs(c) > fabs(m)) m = c;
    return m;
}

GPUFUN
double elliptic_max4(const double a, const double b, const double c, const double d) {
    double m = elliptic_max3(a, b, c);
    if (fabs(d) > fabs(m)) m = d;
    return m;
}

GPUFUN
double elliptic_carlson_rc(const double x, const double y) {
    double xn = x;
    double yn = y;
    const double errtol = ELLIPTIC_INTEGRALS_ERRTOL;

    for (int iter = 0; iter < ELLIPTIC_INTEGRALS_MAX_ITER; ++iter) {
        const double mu = (xn + yn + yn) / 3.0;
        const double sn = (yn + mu) / mu - 2.0;
        if (fabs(sn) < errtol) {
            const double c1 = 1.0 / 7.0;
            const double c2 = 9.0 / 22.0;
            const double s = sn * sn * (0.3 + sn * (c1 + sn * (0.375 + sn * c2)));
            return (1.0 + s) / sqrt(mu);
        }
        const double lamda = 2.0 * sqrt(xn) * sqrt(yn) + yn;
        xn = (xn + lamda) * 0.25;
        yn = (yn + lamda) * 0.25;
    }
    return 0.0;
}

GPUFUN
double elliptic_carlson_rf(const double x, const double y, const double z) {
    double xn = x;
    double yn = y;
    double zn = z;
    const double errtol = ELLIPTIC_INTEGRALS_ERRTOL;

    for (int iter = 0; iter < ELLIPTIC_INTEGRALS_MAX_ITER; ++iter) {
        const double mu = (xn + yn + zn) / 3.0;
        const double xndev = 2.0 - (mu + xn) / mu;
        const double yndev = 2.0 - (mu + yn) / mu;
        const double zndev = 2.0 - (mu + zn) / mu;
        const double epslon = elliptic_max3(fabs(xndev), fabs(yndev), fabs(zndev));

        if (epslon < errtol) {
            const double c1 = 1.0 / 24.0;
            const double c2 = 3.0 / 44.0;
            const double c3 = 1.0 / 14.0;
            const double e2 = xndev * yndev - zndev * zndev;
            const double e3 = xndev * yndev * zndev;
            const double s = 1.0 + (c1 * e2 - 0.1 - c2 * e3) * e2 + c3 * e3;
            return s / sqrt(mu);
        }

        const double xnroot = sqrt(xn);
        const double ynroot = sqrt(yn);
        const double znroot = sqrt(zn);
        const double lamda = xnroot * (ynroot + znroot) + ynroot * znroot;
        xn = (xn + lamda) * 0.25;
        yn = (yn + lamda) * 0.25;
        zn = (zn + lamda) * 0.25;
    }
    return 0.0;
}

GPUFUN
double elliptic_carlson_rd(const double x, const double y, const double z) {
    double xn = x;
    double yn = y;
    double zn = z;
    double sigma = 0.0;
    double power4 = 1.0;
    const double errtol = ELLIPTIC_INTEGRALS_ERRTOL;

    for (int iter = 0; iter < ELLIPTIC_INTEGRALS_MAX_ITER; ++iter) {
        const double mu = (xn + yn + 3.0 * zn) * 0.2;
        const double xndev = (mu - xn) / mu;
        const double yndev = (mu - yn) / mu;
        const double zndev = (mu - zn) / mu;
        const double epslon = elliptic_max3(fabs(xndev), fabs(yndev), fabs(zndev));

        if (epslon < errtol) {
            const double c1 = 3.0 / 14.0;
            const double c2 = 1.0 / 6.0;
            const double c3 = 9.0 / 22.0;
            const double c4 = 3.0 / 26.0;
            const double ea = xndev * yndev;
            const double eb = zndev * zndev;
            const double ec = ea - eb;
            const double ed = ea - 6.0 * eb;
            const double ef = ed + ec + ec;
            const double s1 = ed * (-c1 + 0.25 * c3 * ed - 1.5 * c4 * zndev * ef);
            const double s2 = zndev * (c2 * ef + zndev * (-c3 * ec + zndev * c4 * ea));
            return 3.0 * sigma + power4 * (1.0 + s1 + s2) / (mu * sqrt(mu));
        }

        const double xnroot = sqrt(xn);
        const double ynroot = sqrt(yn);
        const double znroot = sqrt(zn);
        const double lamda = xnroot * (ynroot + znroot) + ynroot * znroot;
        sigma += power4 / (znroot * (zn + lamda));
        power4 *= 0.25;
        xn = (xn + lamda) * 0.25;
        yn = (yn + lamda) * 0.25;
        zn = (zn + lamda) * 0.25;
    }
    return 0.0;
}

GPUFUN
double elliptic_carlson_rj(
    const double x, const double y, const double z, const double p
) {
    double xn = x;
    double yn = y;
    double zn = z;
    double pn = p;
    double sigma = 0.0;
    double power4 = 1.0;
    const double errtol = ELLIPTIC_INTEGRALS_ERRTOL;

    for (int iter = 0; iter < ELLIPTIC_INTEGRALS_MAX_ITER; ++iter) {
        const double mu = (xn + yn + zn + pn + pn) * 0.2;
        const double xndev = (mu - xn) / mu;
        const double yndev = (mu - yn) / mu;
        const double zndev = (mu - zn) / mu;
        const double pndev = (mu - pn) / mu;
        const double epslon = elliptic_max4(
            fabs(xndev), fabs(yndev), fabs(zndev), fabs(pndev));

        if (epslon < errtol) {
            const double c1 = 3.0 / 14.0;
            const double c2 = 1.0 / 3.0;
            const double c3 = 3.0 / 22.0;
            const double c4 = 3.0 / 26.0;
            const double ea = xndev * (yndev + zndev) + yndev * zndev;
            const double eb = xndev * yndev * zndev;
            const double ec = pndev * pndev;
            const double e2 = ea - 3.0 * ec;
            const double e3 = eb + 2.0 * pndev * (ea - ec);
            const double s1 = 1.0 + e2 * (-c1 + 0.75 * c3 * e2 - 1.5 * c4 * e3);
            const double s2 = eb * (0.5 * c2 + pndev * (-c3 - c3 + pndev * c4));
            const double s3 = pndev * ea * (c2 - pndev * c3) - c2 * pndev * ec;
            return 3.0 * sigma + power4 * (s1 + s2 + s3) / (mu * sqrt(mu));
        }

        const double xnroot = sqrt(xn);
        const double ynroot = sqrt(yn);
        const double znroot = sqrt(zn);
        const double lamda = xnroot * (ynroot + znroot) + ynroot * znroot;
        double alfa = pn * (xnroot + ynroot + znroot) + xnroot * ynroot * znroot;
        alfa = alfa * alfa;
        const double beta = pn * (pn + lamda) * (pn + lamda);
        sigma += power4 * elliptic_carlson_rc(alfa, beta);

        power4 *= 0.25;
        xn = (xn + lamda) * 0.25;
        yn = (yn + lamda) * 0.25;
        zn = (zn + lamda) * 0.25;
        pn = (pn + lamda) * 0.25;
    }
    return 0.0;
}

GPUFUN
double ellip_k(const double m) {
    if (m >= 1.0) {
        return INFINITY;
    }
    return elliptic_carlson_rf(0.0, 1.0 - m, 1.0);
}

GPUFUN
double ellip_e(const double m) {
    if (m >= 1.0) {
        return 1.0;
    }
    const double y = 1.0 - m;
    return elliptic_carlson_rf(0.0, y, 1.0) - (m / 3.0) * elliptic_carlson_rd(0.0, y, 1.0);
}

GPUFUN
double ellip_pi(const double n, const double m) {
    if (m >= 1.0) {
        return INFINITY;
    }
    const double y = 1.0 - m;
    const double p = 1.0 - n;
    if (p <= 0.0) {
        return INFINITY;
    }
    return elliptic_carlson_rf(0.0, y, 1.0) +
           elliptic_carlson_rj(0.0, y, 1.0, p) * n / 3.0;
}

#endif // XTRACK_ELLIPTIC_INTEGRALS_H
