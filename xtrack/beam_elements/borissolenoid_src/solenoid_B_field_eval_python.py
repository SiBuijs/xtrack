"""Python reference for solenoid_B_field_eval.h (uses scipy elliptic integrals)."""

import numpy as np
from scipy.special import ellipk, ellipe, elliprf, elliprj


def ellipp(n, m):
    y = 1.0 - m
    return elliprf(0.0, y, 1.0) + elliprj(0.0, y, 1.0, 1.0 - n) * n / 3.0


def evaluate_solenoid_B(x, y, z, L, a, B0, z0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r = np.sqrt(x**2 + y**2)
    zeta_plus = z - z0 + 0.5 * L
    zeta_minus = z - z0 - 0.5 * L

    a_plus_r = a + r
    a_minus_r = a - r

    u = np.where(r > 0.0, 4.0 * a * r / a_plus_r**2, 0.0)

    denom_plus = a_plus_r**2 + zeta_plus**2
    denom_minus = a_plus_r**2 + zeta_minus**2

    m_plus = np.where(r > 0.0, 4.0 * a * r / denom_plus, 0.0)
    m_minus = np.where(r > 0.0, 4.0 * a * r / denom_minus, 0.0)

    sqrt_plus = np.sqrt(4.0 / denom_plus)
    sqrt_minus = np.sqrt(4.0 / denom_minus)

    bz_plus = B0 * zeta_plus / (4.0 * np.pi) * sqrt_plus * (
        ellipk(m_plus) + a_minus_r / a_plus_r * ellipp(u, m_plus)
    )
    bz_minus = B0 * zeta_minus / (4.0 * np.pi) * sqrt_minus * (
        ellipk(m_minus) + a_minus_r / a_plus_r * ellipp(u, m_minus)
    )
    bz = bz_plus - bz_minus

    bx = np.zeros_like(z)
    by = np.zeros_like(z)

    mask = r > 1e-11
    sqrt_r_plus = np.sqrt(a / (r[mask] * m_plus[mask]))
    sqrt_r_minus = np.sqrt(a / (r[mask] * m_minus[mask]))

    br_plus = B0 / np.pi * sqrt_r_plus * (
        ellipe(m_plus[mask]) - (1.0 - 0.5 * m_plus[mask]) * ellipk(m_plus[mask])
    )
    br_minus = B0 / np.pi * sqrt_r_minus * (
        ellipe(m_minus[mask]) - (1.0 - 0.5 * m_minus[mask]) * ellipk(m_minus[mask])
    )
    br = br_plus - br_minus

    bx[mask] = br * x[mask] / r[mask]
    by[mask] = br * y[mask] / r[mask]

    if bx.shape == ():
        return float(bx), float(by), float(bz)
    return bx, by, bz
