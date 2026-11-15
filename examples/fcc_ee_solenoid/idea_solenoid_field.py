"""IDEA main (G), screening (S) and total (G - S) on-axis solenoid fields,
built from the digitized Opera curves; shared by 001e (edge scan, figures)
and 004a_build_idea_solenoid.py (BFieldExpansion line for the ring).

The method, its choices and the measured numbers are in 001e's docstring and
claude_notes/17. In short, for one screening edge z_e:
- the digitized curves, scaled to `main_field`, ripples removed (001d step 1);
- G fitted on 0.2 m elements and projected onto the 0.1 m grid of the edge;
- the screening fraction S/G split into a symmetric pair of edges rho;
- the target S = G [rho(z - z_e) + rho(-z - z_e)] k(z), with the
  over-screening factor k = 1 + eps w, w = 1 up to z_p (peak of S for
  eps = 0) and a C3 taper of `taper_length` to 0;
- S = G pinned from z_m (first node past z_p + taper_length), least squares
  inside, integral preserved.

Tilt convention. The beam-frame coefficients A_m use theta_b, the angle of
the solenoid axis from the beam axis (B_x = theta_b * B_s on the axis at the
IP). solenoid_params.THETA (tilted_solenoid.TiltedSolenoid, 004a, 010a) is
the beam's angle from the solenoid axis, B_x = -sin(THETA) B_s at the IP. So
theta_b = -THETA = +0.015 (checked 2026-11-15 against the 004a 2 T template:
B_x(IP) = +0.030 T, negative peak at the edge).
"""
from pathlib import Path
import json
import math

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BSpline

from solenoid_params import THETA
from xtrack._temp.splineboris.longitudinal_fitter import LongitudinalFitter

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / 'IDEA_solenoid_beam_axis_raw.json'

OPERA_FIELD = 2.0          # T, nominal main field of the digitized Opera curves

# Ripple removal and G fit: as in 001d.
DZ = 1e-3                  # m, common grid step (raw resolution ~3.4 mm)
Z_CUT = 2.05               # m, ripples in B start here
FIT_WINDOW = (1.70, Z_CUT) # m, exponential fit of the clean edge of B
BLEND = 0.05               # m, cosine blend raw -> fit, ending at Z_CUT
S_END = 6.0                # m, spline range +-S_END
MAIN_ELEMENT_LENGTH = 0.2  # m, G fit

ELEMENT_LENGTH = 0.1       # m, grid of the screening and total splines
INNER_TAIL_WINDOW = (0.5, 1.0)  # m, e-folding fit of one edge's inner tail
PEAK_WINDOW = 0.10         # m, half-width of the parabola fit locating z_p
MIN_TAPER_LENGTH = 0.4     # m, the 0.1 m spline cannot follow a shorter taper

TILT_ANGLE = -THETA        # rad, theta_b of the A_m formulas (see above)

DEG = 4                    # quartic B-splines, as LongitudinalFitter


def load_raw_curve(curves, key):
    """Raw points with repeated z (row-scanned steep parts) averaged."""
    z = np.array(curves[key]['z'])
    bz = np.array(curves[key]['Bz'])
    z_unique, inverse = np.unique(z, return_inverse=True)
    bz_mean = np.bincount(inverse, bz) / np.bincount(inverse)
    return z_unique, bz_mean


def fit_on_axis_bs(z_data, bs_data, element_length):
    """001d's fit: C3 quartic B-spline of ("Bs", 0) on +-S_END, data mirrored
    about the IP, field zero at both ends, integral preserved."""
    z_mirror = np.r_[-z_data[::-1], z_data]
    bs_mirror = np.r_[bs_data[::-1], bs_data]
    unique = np.r_[True, np.diff(z_mirror) > 0]      # z = 0 only once
    fitter = LongitudinalFitter(
        s_start=-S_END, s_end=S_END,
        n_elements=int(round(2 * S_END / element_length)),
        end_condition='zero', preserve_integral=True)
    fitter.fit(z_mirror[unique], bs_mirror[unique], ('Bs', 0))
    return fitter, fitter.splines[('Bs', 0)]


def beam_frame_from_derivatives(b0, b1, b2, b3, b4, s, theta):
    """A_m at x = y = 0 in the beam frame, to O(theta^2), as 001d step 3,
    from the on-axis profile b and its derivatives b1..b4 at the beam s
    (b_s^(5) term of A_4 dropped). A_0 = B_s, A_m = d^(m-1) B_x / dx^(m-1)
    for m >= 1; B_y = 0 on y = 0 (tilt in the x-s plane). Works for arrays
    and for numpy Polynomial objects alike.

    A_0 corrected 2026-11-15: 001d had theta^2 (s b/2 + s^2 b'/4), which is
    not even dimensionally consistent; rederived, theta^2 (b/2 + s b' +
    s^2 b''/4). Against TiltedSolenoid (analytic 2 T): 5.1 -> 0.06 mT max.
    The O(theta^3) remainder of A_1 is 0.5 mT there and ~2.5 mT at the sharp
    IDEA edge (r = s theta is not small against the edge width), so the
    elements use exact_beam_frame_polynomials() instead."""
    th2 = theta ** 2
    return {
        'A0': b0 - th2 * (b0 / 2 + s * b1 + s ** 2 / 4 * b2),
        'A1': theta * (b0 + s / 2 * b1),
        'A2': -0.5 * b1 + th2 * (1.5 * b1 + 1.25 * s * b2 + 3 / 16 * s ** 2 * b3),
        'A3': -theta * (1.5 * b2 + 0.375 * s * b3),
        'A4': 0.375 * b3 - th2 * (3.75 * b3 + 27 / 16 * s * b4),
    }


def beam_frame_coefficients(spline, s, theta):
    """A_m (see beam_frame_from_derivatives) of a spline profile at s."""
    b0, b1, b2, b3, b4 = (spline.derivative(k)(s) if k else spline(s)
                          for k in range(5))
    return beam_frame_from_derivatives(b0, b1, b2, b3, b4, s, theta)


def _poly2_mul(a, b):
    """Product of two bivariate polynomials, coefficient arrays c[i, j] of
    x**i * u**j."""
    out = np.zeros((a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1))
    for i, j in zip(*np.nonzero(a)):
        out[i:i + b.shape[0], j:j + b.shape[1]] += a[i, j] * b
    return out


def _poly2_add(*terms):
    """Sum of bivariate polynomials (coefficient arrays of any shapes)."""
    shape = tuple(max(t.shape[k] for t in terms) for k in range(2))
    out = np.zeros(shape)
    for t in terms:
        out[:t.shape[0], :t.shape[1]] += t
    return out


def _poly2_compose(coeffs, linear):
    """p(L(x, u)) for a univariate p (ascending coeffs) and a bivariate L,
    by Horner."""
    out = np.array([[coeffs[-1]]], dtype=float)
    for c in coeffs[-2::-1]:
        out = _poly2_add(_poly2_mul(out, linear), np.array([[c]]))
    return out


def exact_beam_frame_polynomials(profile, s_lo, s_hi, theta_b, n_rows=5):
    """Exact beam-frame A_0..A_n_rows on [s_lo, s_hi] (beam s, IP at 0) of
    the axisymmetric solenoid with on-axis profile `profile` (a quartic
    B-spline), tilted by theta_b (solenoid axis (sin, 0, cos) theta_b in the
    beam frame), as numpy Polynomials in u = s - s_lo. All orders in theta.

    Inside one spline piece the field is a polynomial in (x, y, s): the
    paraxial series ends there (Bz' = b - r^2 b''/4 + r^4 b''''/64, Br =
    -r b'/2 + r^3 b'''/16), and z' = x sin + s cos, x' = x cos - s sin are
    linear. A_0 = B_s(0, 0, s), A_m = d^(m-1) B_x/dx^(m-1) at x = y = 0;
    B_y = 0 on y = 0. The piece is the one at the element's midpoint: on the
    beam axis z' = s cos theta_b, so the knot sits up to 1.1e-4 s (0.26 mm
    at 2.3 m) inside the element, where the C3 piece is extrapolated (error
    ~ jump(b'''') d^4 / 24, below 1e-10 T)."""
    sn, cs = np.sin(theta_b), np.cos(theta_b)
    m_z = cs * 0.5 * (s_lo + s_hi)       # z' of the element midpoint on axis
    taylor = Polynomial([float(profile.derivative(k)(m_z) if k
                               else profile(m_z)) / math.factorial(k)
                         for k in range(5)])
    # w = z' - m_z and x' as bivariate (x, u) polynomials, s = s_lo + u
    w = np.array([[cs * s_lo - m_z, cs], [sn, 0.0]])
    x_prime = np.array([[-sn * s_lo, -sn], [cs, 0.0]])
    b = [_poly2_compose(taylor.deriv(k).coef if k else taylor.coef, w)
         for k in range(5)]
    r2 = _poly2_mul(x_prime, x_prime)
    bz_sol = _poly2_add(b[0], -_poly2_mul(r2, b[2]) / 4,
                        _poly2_mul(_poly2_mul(r2, r2), b[4]) / 64)
    radial = _poly2_add(-b[1] / 2, _poly2_mul(r2, b[3]) / 16)
    bx_sol = _poly2_mul(x_prime, radial)
    bx = _poly2_add(cs * bx_sol, sn * bz_sol)
    bs = _poly2_add(-sn * bx_sol, cs * bz_sol)
    out = {'A0': Polynomial(bs[0])}
    for m in range(1, n_rows + 1):
        row = bx[m - 1] if m - 1 < bx.shape[0] else np.zeros(1)
        out[f'A{m}'] = Polynomial(math.factorial(m - 1) * row)
    return out


def overshoot_factor(z, z_peak, eps, taper_length):
    """k(z) = 1 + eps w(z): w = 1 for |z| <= z_p, then a septic smoothstep
    (derivatives 1-3 zero at both ends, so C3) to 0 at |z| = z_p +
    taper_length."""
    t = np.clip((np.abs(z) - z_peak) / taper_length, 0.0, 1.0)
    w = 1.0 - t ** 4 * (35.0 - 84.0 * t + 70.0 * t ** 2 - 20.0 * t ** 3)
    return 1.0 + eps * w


class IdeaSolenoidData:
    """The digitized curves at `main_field`, ripples removed, G fitted and
    projected onto both 0.1 m grids, and the one-edge screening fraction
    rho. build_case() then builds S and the total for one edge."""

    def __init__(self, main_field, data_file=DATA_FILE):
        self.main_field = main_field
        # ------------------------------- data, ripple removal (001d step 1)
        with open(data_file) as fid:
            curves = json.load(fid)['curves']

        z_g, bz_g = load_raw_curve(curves, 'main_only')
        z_b, bz_b = load_raw_curve(curves, 'both')

        self.field_scale = main_field / OPERA_FIELD   # everything is linear
        bz_g = self.field_scale * bz_g
        bz_b = self.field_scale * bz_b

        z = np.round(np.arange(0.0, z_g.max() + 0.5 * DZ, DZ), 9)
        main = np.interp(z, z_g, bz_g)
        both_raw = np.interp(z, z_b, bz_b, right=np.nan)

        mask_fit = (z >= FIT_WINDOW[0]) & (z <= FIT_WINDOW[1])
        slope, intercept = np.polyfit(z[mask_fit], np.log(both_raw[mask_fit]),
                                      1)
        both_tail = np.exp(intercept + slope * z)
        weight = np.clip((z - (Z_CUT - BLEND)) / BLEND, 0.0, 1.0)
        weight = 0.5 * (1.0 - np.cos(np.pi * weight))
        both_raw_or_tail = np.where(np.isnan(both_raw), both_tail, both_raw)
        both_clean = (1.0 - weight) * both_raw_or_tail + weight * both_tail
        screening_clean = main - both_clean
        self.z, self.both_clean = z, both_clean

        # ------------------- G fit (001d step 2), on the two 0.1 m grids
        self.main_fitter, self.main_spline = fit_on_axis_bs(
            z_g, bz_g, MAIN_ELEMENT_LENGTH)
        # 001d's screening fit, the reference for the 1.40 m edge
        _, self.screening_001d = fit_on_axis_bs(z, screening_clean,
                                                ELEMENT_LENGTH)
        self._main_zero_beyond = BSpline(
            self.main_spline.t, self.main_spline.c, DEG,
            extrapolate=False)                       # NaN beyond +-S_END -> 0
        self.grids = {offset: self._make_grid(offset)
                      for offset in (0.0, 0.5 * ELEMENT_LENGTH)}

        # --------------------------- screening fraction, split into two edges
        z_profile = z[z <= 4.0]
        fraction = 1.0 - both_clean[z <= 4.0] / self.main_spline(z_profile)
        i_half = np.argmax(fraction >= 0.5)
        self.edge_data = np.interp(0.5, fraction[i_half - 1:i_half + 1],
                                   z_profile[i_half - 1:i_half + 1])
        self.half_at_ip = 0.5 * fraction[0]      # each edge's share at z = 0

        mask_tail = ((z_profile >= INNER_TAIL_WINDOW[0])
                     & (z_profile <= INNER_TAIL_WINDOW[1]))
        decay = 0.3
        for _ in range(200):
            one_edge = fraction - self.half_at_ip * np.exp(-z_profile / decay)
            decay_new = 1.0 / np.polyfit(z_profile[mask_tail],
                                         np.log(one_edge[mask_tail]), 1)[0]
            if abs(decay_new - decay) < 1e-9:
                break
            decay = decay_new
        else:
            raise RuntimeError('inner-tail e-folding length did not converge')
        self.z_profile, self.fraction = z_profile, fraction
        self.decay, self._one_edge = decay, one_edge
        self._u_data = z_profile - self.edge_data

    def _make_grid(self, offset):
        """ELEMENT_LENGTH grid on +-(S_END + offset), nodes at offset + k *
        ELEMENT_LENGTH, field zero at the ends; G projected onto it."""
        s_end = S_END + offset
        grid = LongitudinalFitter(
            s_start=-s_end, s_end=s_end,
            n_elements=int(round(2 * s_end / ELEMENT_LENGTH)),
            end_condition='zero')
        z_dense = np.linspace(-s_end, s_end, int(round(2 * s_end / 5e-4)) + 1)
        g_dense = np.nan_to_num(self._main_zero_beyond(z_dense))
        grid.fit(z_dense, g_dense, ('Bs', 0))
        main = grid.splines[('Bs', 0)]
        knots = grid.knots
        support_lo, support_hi = knots[:-DEG - 1], knots[DEG + 1:]
        return dict(offset=offset, s_end=s_end, knots=knots, nodes=grid.nodes,
                    main=main, main_coeffs=main.c.copy(),
                    support_lo=support_lo, support_hi=support_hi,
                    basis_integral=(support_hi - support_lo) / (DEG + 1),
                    projection_error=np.abs(main(z_dense) - g_dense).max())

    def one_edge_fraction(self, u):
        """rho(u), u = distance from the edge (negative towards the IP)."""
        u = np.asarray(u, dtype=float)
        beyond_ip = self.half_at_ip * np.exp(
            np.minimum(u + self.edge_data, 0.0) / self.decay)
        return np.where(u >= -self.edge_data,
                        np.interp(u, self._u_data, self._one_edge),
                        beyond_ip)

    def screening_peak(self, edge):
        """z_p: peak of the eps = 0 target G [rho(z - e) + rho(-z - e)], the
        vertex of a parabola fitted over +-PEAK_WINDOW around its maximum
        (the peak is flat, so the raw argmax follows the digitization)."""
        z = self.z
        z_near = z[(z >= edge) & (z <= edge + 1.0)]
        target = self.main_spline(z_near) * (
            self.one_edge_fraction(z_near - edge)
            + self.one_edge_fraction(-z_near - edge))
        i = np.argmax(target)
        mask = np.abs(z_near - z_near[i]) <= PEAK_WINDOW
        curvature, slope, _ = np.polyfit(z_near[mask] - z_near[i],
                                         target[mask], 2)
        if curvature >= 0:
            raise ValueError(f'edge {edge} m: no peak of S (eps = 0) found')
        return z_near[i] - slope / (2.0 * curvature)

    def grid_for_edge(self, edge):
        """The grid with a node on `edge`: nodes at k * ELEMENT_LENGTH, or
        offset by ELEMENT_LENGTH / 2."""
        half_steps = edge / (0.5 * ELEMENT_LENGTH)
        if abs(half_steps - round(half_steps)) > 1e-9:
            raise ValueError(f'edge {edge} m is not a multiple of '
                             f'{0.5 * ELEMENT_LENGTH} m')
        return self.grids[0.0 if round(half_steps) % 2 == 0
                          else 0.5 * ELEMENT_LENGTH]

    def build_case(self, edge, eps, taper_length):
        """S and the total for an edge at `edge` (S/G = (1 + eps)/2 there):
        S = G for |z| >= z_match (first node past z_p + taper_length), least
        squares to the moved (and over-screened) target inside, integral
        preserved. Returns a dict (see the keys at the end)."""
        if taper_length < MIN_TAPER_LENGTH - 1e-9:
            raise ValueError(
                f'taper_length {taper_length} m: needs >= {MIN_TAPER_LENGTH} '
                f'm (the {ELEMENT_LENGTH} m elements cannot follow less)')
        z = self.z
        grid = self.grid_for_edge(edge)
        knots, nodes = grid['knots'], grid['nodes']
        support_lo, support_hi = grid['support_lo'], grid['support_hi']
        basis_integral, main_coeffs = (grid['basis_integral'],
                                       grid['main_coeffs'])
        z_peak = self.screening_peak(edge)
        z_match = nodes[np.searchsorted(nodes, z_peak + taper_length - 1e-9)]
        if z_match > self.z_profile.max() - 0.5:
            raise ValueError(f'match point {z_match} m too far out')

        z_half = z[z <= z_match + 1e-9]
        target_half = (self.main_spline(z_half)
                       * (self.one_edge_fraction(z_half - edge)
                          + self.one_edge_fraction(-z_half - edge))
                       * overshoot_factor(z_half, z_peak, eps, taper_length))
        z_t = np.r_[-z_half[:0:-1], z_half]
        target = np.r_[target_half[:0:-1], target_half]

        free = ((support_lo >= -z_match - 1e-9)
                & (support_hi <= z_match + 1e-9))
        coeffs = main_coeffs.copy()                  # fixed ones: G's
        K = BSpline.design_matrix(z_t, knots, DEG).tocsc()
        K_free = K[:, np.flatnonzero(free)]
        rhs = target - K[:, np.flatnonzero(~free)] @ coeffs[~free]
        normal = (K_free.T @ K_free).toarray()
        c_lsq = np.linalg.solve(normal, K_free.T @ rhs)

        # integral over |z| <= z_m equal to the target's (S = G beyond)
        required = (np.trapezoid(target, z_t)
                    + 2 * grid['main'].integrate(z_match, grid['s_end'])
                    - basis_integral[~free] @ coeffs[~free])
        w = basis_integral[free]
        normal_inv_w = np.linalg.solve(normal, w)
        coeffs[free] = (c_lsq - normal_inv_w * (w @ c_lsq - required)
                        / (w @ normal_inv_w))

        screening = BSpline(knots, coeffs, DEG)
        total = BSpline(knots, main_coeffs - coeffs, DEG)
        # elements of the total on |z| <= z_m, with a boundary at the IP
        inside = nodes[np.abs(nodes) <= z_match + 1e-9]
        boundaries = np.unique(np.round(np.r_[inside, 0.0], 12))
        return dict(edge=edge, eps=eps, taper_length=taper_length,
                    main_field=self.main_field, z_peak=z_peak,
                    z_match=z_match, offset=grid['offset'],
                    main=grid['main'], screening=screening, total=total,
                    z_target=z_half, target=target_half,
                    element_boundaries=boundaries)
