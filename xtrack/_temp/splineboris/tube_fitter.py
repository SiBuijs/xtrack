"""
Global magnetic-field fitting via the tube approach (Riemann & Aiba, IPAC2021).

Fits a scalar potential
    Omega_tilde(x, y, z) = sum_{j,p,q} Psi[j,p,q] * x^p * y^q * beta_j(z)
with tent (degree-1 B-spline) longitudinal basis functions beta_j(z) -- i.e.
straight-line interpolation between frames -- then converts multipole
coefficients to the same Hermite-quartic ``df_fit_pars`` format as ``FieldFitter``.

Conventions (h = 0, straight frame, B = -grad Phi):
    - Minus sign is applied in the sparse system rows for Bx and By.
    - C_{p,q}(z) = sum_j Psi[j,p,q] * beta_j(z)
    - b_m(z) = -(m-1)! * C_{m-1,1}(z)  ->  ``Bnorm``, derivative_x = m-1
    - a_m(z) = -m! * C_{m,0}(z)  ->  ``Bskew``, derivative_x = m-1  (uses Psi[:, m, 0])
    - b_s(z): fitted independently (1D tent fit to on-axis Bs), exactly as
      ``FieldFitter`` does. Only ``q=0`` (skew) and ``q=1`` (norm) rows of
      ``Psi`` are ever read on export -- any ``q>=2`` content the fit picks
      up (only possible when ``y_symmetry=False``) is fit freely, purely to
      keep it from biasing the exported ``q=0``/``q=1`` columns, and is
      itself discarded. This mirrors the Van der Schueren potential's own
      Cauchy data (phi_0, phi_1 only): the downstream Table-1 field
      evaluator regenerates all q>=2 structure from (a_n, b_n, b_s) alone,
      Maxwell-consistent by construction, so the tube fit does not need to
      reproduce it or enforce div(B) = 0 itself -- see
      examples/splineboris/claude_notes/tube_schueren_integration.md. Use
      ``check_trace_consistency()`` (a diagnostic, not a correction) to see
      how well that assumption holds on a given dataset.
    - Default symmetry: only (p, q) with odd q; all (p, 0) skew terms if fit_skew=True
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sc
import xtrack as xt

from xtrack.beam_elements.splineboris import Spline4, SplineBoris
from xtrack.beam_elements.splineboris_src.spline_B_field_eval_python import (
    hermite_to_polynomial,
)

_REQUIRED_COLUMNS = ("Bx", "By", "Bs")
_INDEX_NAMES = ("X", "Y", "Z")

# Fallback n_frames used when neither n_frames nor residual_tol is given.
# Not tuned to any particular fit-quality target -- just a reasonable middle
# ground that avoids triggering the (potentially slow) residual_tol search.
DEFAULT_N_FRAMES = 200


def _generate_pq_pairs(M: int, y_symmetry: bool, fit_skew: bool) -> list[tuple[int, int]]:
    """(p, q) pairs with 0 < p+q <= M, filtered by symmetry options."""
    pairs: list[tuple[int, int]] = []
    for p in range(M + 1):
        for q in range(M + 1):
            s = p + q
            if s == 0 or s > M:
                continue
            if q % 2 == 1:
                pairs.append((p, q))
            elif q == 0 and p > 0:
                # Skew terms: (p, 0) gives multipole a_p via C_{p,0} (on-axis d^{p-1} B_x / dx^{p-1})
                if fit_skew or p == 1:
                    pairs.append((p, q))
            elif not y_symmetry:
                pairs.append((p, q))
    return pairs


def _frame_indices(s_full: np.ndarray, frames: np.ndarray) -> np.ndarray:
    return np.array([int(np.argmin(np.abs(s_full - f))) for f in frames], dtype=int)


def _tent_interval_index(z: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Interval ``k`` (``frames[k] <= z < frames[k + 1]``) each z falls in.
    A z exactly on an interior frame belongs to the interval on its right;
    ``z == frames[-1]`` belongs to the last interval."""
    return np.clip(np.searchsorted(frames, z, side="right") - 1, 0, len(frames) - 2)


def _tent_weights(
    z: np.ndarray, frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tent (degree-1 B-spline) basis at each z: only frames ``k`` and
    ``k + 1`` are non-zero, with weights ``b0 = 1 - t`` and ``b1 = t``,
    where ``t`` is the fractional position of z within interval ``k``.
    Returns ``(k, b0, b1)``."""
    k = _tent_interval_index(z, frames)
    t = (z - frames[k]) / (frames[k + 1] - frames[k])
    return k, 1.0 - t, t


def _tent_slope(z: np.ndarray, frames: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """d/dz of the tent interpolant ``np.interp(z, frames, coeffs)``:
    constant within each interval, taken from the interval on the right at
    an interior frame (see ``_tent_interval_index``)."""
    slopes = np.diff(coeffs) / np.diff(frames)
    return slopes[_tent_interval_index(z, frames)]


def _tent_intervals(
    z: np.ndarray, frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group raw points by the tent interval they fall in -- each point
    only touches frames ``k`` (weight ``b0``) and ``k + 1`` (weight
    ``b1``) of its own interval ``k``.

    Returns ``(order, bounds, b0, b1)``: ``order`` sorts the points by
    interval, after which the points of interval ``k`` are the contiguous
    slice ``bounds[k]:bounds[k + 1]``; ``b0``/``b1`` are already in sorted
    order.
    """
    k, b0, b1 = _tent_weights(z, frames)
    order = np.argsort(k, kind="stable")
    bounds = np.searchsorted(k[order], np.arange(len(frames)))
    return order, bounds, b0[order], b1[order]


def _transverse_gradients(
    x: np.ndarray, y: np.ndarray, pq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bx and By design rows, each ``(n_pts, n_pq)``: the (p, q) term of
    ``B = -grad(x^p y^q)``, i.e. ``-p x^(p-1) y^q`` and ``-q x^p y^(q-1)``
    (zero where p = 0 resp. q = 0). Powers come from a table built by
    repeated multiplication -- much faster than ``x ** p`` with array
    exponents."""
    p, q = pq[:, 0], pq[:, 1]
    n_pow = int(pq.max()) + 1
    xp = np.ones((len(x), n_pow))
    yp = np.ones((len(y), n_pow))
    for e in range(1, n_pow):
        xp[:, e] = xp[:, e - 1] * x
        yp[:, e] = yp[:, e - 1] * y
    g_bx = -p * xp[:, np.maximum(p - 1, 0)] * yp[:, q]
    g_by = -q * xp[:, p] * yp[:, np.maximum(q - 1, 0)]
    return g_bx, g_by


def _accumulate_block_tridiagonal(
    bounds, b0, b1, x, y, bx, by, pq, n_frames,
):
    """Build the tube fit's block-tridiagonal normal equations directly
    from the raw points -- no global design matrix ever gets built. Points
    must be grouped by tent interval (see ``_tent_intervals``). The points
    of interval ``k`` only touch frames ``k`` and ``k + 1``, so they
    contribute ``G0^T G0`` to diagonal block ``D[k]``, ``G1^T G1`` to
    ``D[k + 1]`` and ``G0^T G1`` to the coupling block ``E[k]``, where
    ``G0``/``G1`` are their transverse design rows weighted by ``b0``/``b1``
    -- a handful of small dense matrix products per interval. Memory stays
    at O(points per interval * n_pq), never O(n_pts * n_pq).
    """
    n_pq = len(pq)
    D = np.zeros((n_frames, n_pq, n_pq))
    E = np.zeros((n_frames - 1, n_pq, n_pq))
    r = np.zeros((n_frames, n_pq))
    for k in range(n_frames - 1):
        s = slice(bounds[k], bounds[k + 1])
        if s.start == s.stop:
            continue
        g_bx, g_by = _transverse_gradients(x[s], y[s], pq)
        w0, w1 = b0[s, None], b1[s, None]
        gx0, gy0 = w0 * g_bx, w0 * g_by
        gx1, gy1 = w1 * g_bx, w1 * g_by
        D[k] += gx0.T @ gx0 + gy0.T @ gy0
        D[k + 1] += gx1.T @ gx1 + gy1.T @ gy1
        E[k] += gx0.T @ gx1 + gy0.T @ gy1
        r[k] += gx0.T @ bx[s] + gy0.T @ by[s]
        r[k + 1] += gx1.T @ bx[s] + gy1.T @ by[s]
    return D, E, r


def _accumulate_residual_sumsq(bounds, b0, b1, x, y, bx, by, pq, x_sol):
    """Evaluate the fitted field at every raw point directly from the
    solved ``x_sol`` (shape ``(n_frames, n_pq)``) and reduce into
    sum-of-squares, interval by interval (same grouping as
    ``_accumulate_block_tridiagonal``) -- without ever materializing ``A``
    or a full per-point residual array. Returns ``(sumsq_res_bx,
    sumsq_res_by, sumsq_bx, sumsq_by)``."""
    n_frames = x_sol.shape[0]
    sums = np.zeros(4)
    for k in range(n_frames - 1):
        s = slice(bounds[k], bounds[k + 1])
        if s.start == s.stop:
            continue
        g_bx, g_by = _transverse_gradients(x[s], y[s], pq)
        coeffs = b0[s, None] * x_sol[k] + b1[s, None] * x_sol[k + 1]
        res_bx = bx[s] - np.einsum("ij,ij->i", g_bx, coeffs)
        res_by = by[s] - np.einsum("ij,ij->i", g_by, coeffs)
        sums += (res_bx @ res_bx, res_by @ res_by, bx[s] @ bx[s], by[s] @ by[s])
    return tuple(float(v) for v in sums)


def _hermite_from_tent(
    frames: np.ndarray, coeffs: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Hermite params ``(val_start, der_start, val_end, der_end, mean)`` of
    the tent interpolant on every region ``[frames[i], frames[i + 1]]``,
    each an ``(n_regions,)`` array. Within a region the interpolant is a
    straight line, so both end slopes are that region's own slope (the
    kink at each frame belongs to the neighbouring region's slope on the
    far side) and the mean is the average of the two end values."""
    c_left, c_right = coeffs[:-1], coeffs[1:]
    slope = (c_right - c_left) / np.diff(frames)
    return c_left, slope, c_right, slope, 0.5 * (c_left + c_right)


class TubeFitter:
    """
    Fit 3D magnetic field maps using the tube approach with B-splines in z.

    Parameters
    ----------
    raw_data :
        ``pd.DataFrame`` with MultiIndex ``('X', 'Y', 'Z')`` and columns
        ``('Bx', 'By', 'Bs')``.
    n_frames :
        Number of uniformly spaced longitudinal frames (B-spline control points).
        Number of Hermite regions is ``n_frames - 1``. Mutually exclusive with
        ``residual_tol`` (specifying both raises ``ValueError``). If neither
        is given, defaults to ``DEFAULT_N_FRAMES`` (clamped to the valid
        ``[2, dof_ceiling]`` range) -- a reasonable middle ground, not tuned
        to any particular fit-quality target.
    residual_tol :
        If given (and ``n_frames`` is not), search for the smallest
        ``n_frames`` whose worst-case relative tube-fit residual (Bskew/Bnorm,
        the relative version of ``tube_fit_residual_rms``) is <= this value,
        and use that. The search fits the tube system repeatedly (geometric
        doubling to bracket the transition, then integer bisection: ~log2 of
        the search range), so it can take a while for large datasets -- once
        you know a good value, prefer passing a fixed ``n_frames`` instead.
        Every evaluated ``n_frames`` is recorded in
        ``self.n_frames_search_trace`` (``{n_frames: (rel_bskew, rel_bnorm)}``)
        and plotted automatically via ``plot_n_frames_search()`` once the
        search finishes (call it again yourself to re-plot). If the target
        isn't reachable even at the DOF ceiling ``n_frames`` (a genuine
        residual floor -- e.g. limited transverse data or an unmet
        ``y_symmetry`` assumption; see
        ``examples/splineboris/claude_notes/fieldfitter_vs_tubefitter_comparison.md``),
        this does *not* raise -- it prints a warning and falls back to the
        DOF ceiling (the best achievable), so construction always succeeds.
    distance_unit :
        Scale factor applied to X, Y, Z index levels to convert to metres.
    deg :
        Maximum transverse derivative order (multipole order minus one).
    tube_radius :
        If set, only use data points with sqrt(x^2 + y^2) <= tube_radius [m].
    fit_skew :
        If True (default), include all skew (p, 0) pairs with 1 <= p <= M in the tube
        basis. If False, only (1, 0) is used (on-axis B_x dipole only).
    y_symmetry :
        If True, assume machine-plane (y-parity) symmetry: only use
        (p, q) with odd q for the normal (By) multipoles, so Bnorm terms are
        forced even in y and Bnorm/Bskew are decoupled accordingly. If False
        (default), also include even-q pairs, allowing a field with no
        assumed y-parity. Only the q=0 (skew) and q=1 (norm) rows are ever
        exported, regardless of this setting -- any q>=2 content fitted
        when ``y_symmetry=False`` is fit freely, purely to keep real even-q
        structure in the data from biasing the exported q=0/q=1 (a_n, b_n)
        columns, and is otherwise discarded (see
        examples/splineboris/claude_notes/tube_schueren_integration.md).
    field_tol :
        Relative tolerance for marking a field component as ``to_fit`` in
        ``df_fit_pars`` (same logic as ``FieldFitter``). ``Bs`` is always
        fit independently from on-axis data (``_fit_bs``, a plain 1D
        B-spline), exactly like ``FieldFitter`` -- the downstream Schueren/
        Table-1 field evaluator regenerates all q>=2 structure (including
        the y^2 term tying transverse curvature to ``Bs'(z)``) from
        (a_n, b_n, b_s) alone, Maxwell-consistent by construction, so
        ``TubeFitter`` itself never needs to enforce div(B) = 0 -- see
        ``check_trace_consistency()`` for diagnostics of how well that
        holds on a given dataset.

    The longitudinal basis is a fixed tent (hat-function, degree-1
    B-spline) basis: one tent per frame, peaking there and reaching zero at
    the neighbouring frames, so every ``C_pq(z)`` is the straight-line
    interpolation of its frame values ``Psi[:, p, q]``
    (``np.interp(z, frames, Psi[:, p, q])``). It is C^0: the slope jumps at
    every frame, and each region's Hermite export uses its own slope.
    """

    def __init__(
        self,
        raw_data: pd.DataFrame,
        n_frames: int | None = None,
        distance_unit: float = 1e-3,
        deg: int = 2,
        tube_radius: float | None = None,
        fit_skew: bool = True,
        y_symmetry: bool = False,
        field_tol: float = 1e-3,
        residual_tol: float | None = None,
    ):
        if n_frames is not None and residual_tol is not None:
            raise ValueError(
                "Specify at most one of n_frames and residual_tol -- pass "
                "n_frames for a fixed frame count, or residual_tol to search "
                "for the smallest n_frames that meets it, not both."
            )

        self.distance_unit = float(distance_unit)
        self.deg = int(deg)
        self.M = self.deg + 1
        self.tube_radius = tube_radius
        self.fit_skew = bool(fit_skew)
        self.y_symmetry = bool(y_symmetry)
        self.field_tol = float(field_tol)
        self.xy_point = (0.0, 0.0)
        self.component_to_fit: dict[tuple[str, int], bool] = {}

        self.frames: np.ndarray | None = None
        self.n_regions: int | None = None
        self.pq_pairs = _generate_pq_pairs(self.M, self.y_symmetry, self.fit_skew)
        self.pq_to_idx = {pq: i for i, pq in enumerate(self.pq_pairs)}

        self.s_full: np.ndarray | None = None
        self.Psi: np.ndarray | None = None
        self.Psi_bs: np.ndarray | None = None
        self._hermite: dict[tuple[str, int, int], tuple[float, ...]] | None = None

        self.df_raw_data: pd.DataFrame | None = None
        self.df_on_axis_raw: pd.DataFrame | None = None
        self.df_on_axis_fit: pd.DataFrame | None = None
        self.df_fit_pars: pd.DataFrame | None = None
        self.n_frames_search_trace: dict[int, tuple[float, float]] | None = None
        # Set by _search_n_frames (via _adopt_trial_system) when residual_tol
        # is given -- lets the first _build_linear_system() call adopt the
        # winning trial's already-built system instead of recomputing it.
        self._cached_system: tuple | None = None

        self._set_raw_data(raw_data)

        if n_frames is not None:
            resolved_n_frames = int(n_frames)
        elif residual_tol is not None:
            resolved_n_frames = self._search_n_frames(float(residual_tol))
        else:
            n_max = self._dof_ceiling()
            resolved_n_frames = int(np.clip(DEFAULT_N_FRAMES, 2, n_max))
            print(
                f"[TubeFitter] Neither n_frames nor residual_tol given -- "
                f"defaulting to n_frames={resolved_n_frames}."
            )

        if resolved_n_frames < 2:
            raise ValueError("n_frames must be at least 2")
        self.n_frames = resolved_n_frames

    def fit(self) -> None:
        """Run the full tube-approach fit and populate ``df_fit_pars``."""
        if self.df_raw_data is None:
            raise RuntimeError("Raw data must be provided before calling fit().")
        self._set_df_on_axis()
        self._setup_frames()
        self._build_linear_system()
        self._solve()
        self._populate_on_axis_from_psi()
        self._fit_bs()
        self._convert_to_hermite()
        self._assign_to_fit_flags()
        self._populate_df_fit_pars()
        self._fill_df_on_axis_fit()

    def save_fit_pars(self, file_path: str | Path) -> None:
        """Save ``df_fit_pars`` to CSV."""
        if self.df_fit_pars is None:
            raise RuntimeError("Call fit() before save_fit_pars().")
        self.df_fit_pars.to_csv(file_path, index=True)

    def to_line(
        self,
        multipole_order: int,
        steps_per_point: int = 1,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        radiation_flag: int = 0,
    ) -> xt.Line:
        """Build an ``xt.Line`` of ``SplineBoris`` elements from ``df_fit_pars``.

        Unlike ``FieldFitter`` (whose regions come from independent
        peak-finding per field/derivative and can straddle each other),
        every ``(field_component, derivative_x)`` in ``df_fit_pars`` here
        shares the same region grid (``self.frames``), so no boundary
        reconciliation across fields is needed -- each region's 5 stored
        Hermite params map directly onto a ``Spline4`` (``param_index``
        0..4 == val_start, der_start, val_end, der_end, mean), with no
        ``hermite_to_polynomial`` round-trip. See
        ``examples/splineboris/claude_notes/`` for the reasoning
        (``SplineBorisSequence`` remains the right tool for ``FieldFitter``
        output).

        Parameters
        ----------
        multipole_order :
            Number of multipole orders (``Bx``/``By`` derivative slots) per
            ``SplineBoris`` element. Orders at or beyond what was fit
            (``self.deg + 1``) are filled with a zero ``Spline4``.
        steps_per_point :
            Multiplier for integration steps per raw data point.
        shift_x, shift_y :
            Transverse shift [m] passed through to every element.
        radiation_flag :
            Radiation flag passed through to every element.
        """
        if self.df_fit_pars is None:
            raise RuntimeError("Call fit() before to_line().")
        if multipole_order <= 0:
            raise ValueError("multipole_order must be a positive integer")

        zero_spline = Spline4(
            val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, mean=0.0
        )

        df = self.df_fit_pars.reset_index()
        regions: dict[tuple[float, float], dict] = {}
        for (s_start, s_end, fc, der), grp in df.groupby(
            ["s_start", "s_end", "field_component", "derivative_x"], sort=True
        ):
            region = regions.setdefault((s_start, s_end), {
                "idx_start": int(grp["idx_start"].iloc[0]),
                "idx_end": int(grp["idx_end"].iloc[0]),
                "by": {}, "bx": {}, "bs": zero_spline,
            })
            grp_sorted = grp.sort_values("param_index")
            spline = Spline4(*grp_sorted["param_value"].to_numpy(dtype=float))
            der = int(der)
            if fc == "Bs":
                region["bs"] = spline
            elif fc == "Bnorm":
                region["by"][der] = spline
            elif fc == "Bskew":
                region["bx"][der] = spline

        elements = []
        names = []
        name_width = len(str(self.n_regions - 1)) if self.n_regions > 1 else 1
        for i_reg, (s_start, s_end) in enumerate(sorted(regions)):
            region = regions[(s_start, s_end)]
            by_tuple = tuple(region["by"].get(o, zero_spline) for o in range(multipole_order))
            bx_tuple = tuple(region["bx"].get(o, zero_spline) for o in range(multipole_order))
            n_steps = max(1, (region["idx_end"] - region["idx_start"]) * steps_per_point)

            elements.append(SplineBoris(
                bs=region["bs"],
                by=by_tuple,
                bx=bx_tuple,
                length=float(s_end - s_start),
                n_steps=n_steps,
                shift_x=shift_x,
                shift_y=shift_y,
                radiation_flag=radiation_flag,
            ))
            names.append(f"tubefitter_{i_reg:0{name_width}d}")

        return xt.Line(elements=elements, element_names=names)

    def to_multipole_line(
        self,
        multipole_order: int,
        p0c: float,
        q0: float = 1.0,
        field_at: str = "mean",
        shift_x: float = 0.0,
        shift_y: float = 0.0,
    ) -> xt.Line:
        """Build an ``xt.Line`` of thick ``Multipole`` elements from ``df_fit_pars``.

        Uses the same region grid as ``to_line()``, but replaces each
        region's ``SplineBoris`` (full spatial field integration) with a
        single ``Multipole`` -- a rigidity-normalized multipole kick over
        that region's length.

        Each ``knl``/``ksl`` order is derived from the corresponding
        ``Bnorm``/``Bskew`` Hermite field value (see ``field_at``),
        ``Bnorm``/``Bskew`` already being the on-axis multipole coefficients
        ``d^n By/dx^n`` and ``d^n Bx/dx^n``, via the same relation
        ``xt.Multipole`` itself uses (see
        ``track_magnet_kick.h::evaluate_field_from_strengths``):
        ``knl[n] = length / brho0 * field(d^n By/dx^n)``,
        ``ksl[n] = length / brho0 * field(d^n Bx/dx^n)``, with
        ``brho0 = p0c / (clight * q0)``.

        This discards everything ``to_line()`` keeps beyond a single field
        value per region: longitudinal field variation within a region
        (except at the sampled point), the remaining boundary-derivative
        (Hermite) terms, and the on-axis solenoid field ``Bs`` (no multipole
        equivalent, dropped with a warning if significant) -- so it is a
        coarse approximation, useful as a quick comparison baseline against
        the full ``SplineBoris`` line rather than a faithful field model.

        Parameters
        ----------
        multipole_order :
            Number of multipole orders (``knl``/``ksl`` length) per element.
            Orders at or beyond what was fit (``self.deg + 1``) are zero.
        p0c :
            Reference momentum times c [eV], used to compute the reference
            rigidity ``brho0``.
        q0 :
            Reference charge [elementary charges]. Default 1.0.
        field_at :
            Which field value along each region is used to derive
            ``knl``/``ksl``. ``"mean"`` (default) -- the region-averaged
            field (Hermite ``param_index=4``, a true integral over the
            region divided by its length). ``"midpoint"`` -- the field
            value at the region's longitudinal midpoint, reconstructed from
            the full Hermite quartic via ``hermite_to_polynomial`` (the same
            reconstruction ``to_line()``'s ``SplineBoris`` elements use
            internally to evaluate the field at any point within a region).
        shift_x, shift_y :
            Transverse shift [m] passed through to every ``Multipole``
            element (same meaning as ``to_line()``'s ``shift_x``/``shift_y``).
        """
        if self.df_fit_pars is None:
            raise RuntimeError("Call fit() before to_multipole_line().")
        if multipole_order <= 0:
            raise ValueError("multipole_order must be a positive integer")
        if field_at not in ("mean", "midpoint"):
            raise ValueError(f"field_at must be 'mean' or 'midpoint', got {field_at!r}")

        brho0 = p0c / (sc.constants.c * q0)

        if self.component_to_fit.get(("Bs", 0), False):
            print(
                "[TubeFitter] to_multipole_line(): the fitted Bs (solenoid) "
                "field component has no Multipole equivalent and is dropped."
            )

        df = self.df_fit_pars.reset_index()
        df = df[df["field_component"] != "Bs"]

        regions: dict[tuple[float, float], dict] = {}
        for (s_start, s_end, fc, der), grp in df.groupby(
            ["s_start", "s_end", "field_component", "derivative_x"], sort=True
        ):
            region = regions.setdefault((s_start, s_end), {"knl": {}, "ksl": {}})
            if field_at == "mean":
                value = float(grp.loc[grp["param_index"] == 4, "param_value"].iloc[0])
            else:
                coeffs = grp.sort_values("param_index")["param_value"].to_numpy(dtype=float)
                length = float(s_end - s_start)
                poly = hermite_to_polynomial(0.0, length, coeffs)
                value = float(poly(0.5 * length))
            der = int(der)
            if fc == "Bnorm":
                region["knl"][der] = value
            elif fc == "Bskew":
                region["ksl"][der] = value

        elements = []
        names = []
        name_width = len(str(self.n_regions - 1)) if self.n_regions > 1 else 1
        for i_reg, (s_start, s_end) in enumerate(sorted(regions)):
            region = regions[(s_start, s_end)]
            length = float(s_end - s_start)
            knl = [region["knl"].get(o, 0.0) * length / brho0 for o in range(multipole_order)]
            ksl = [region["ksl"].get(o, 0.0) * length / brho0 for o in range(multipole_order)]

            elements.append(xt.Multipole(
                knl=knl,
                ksl=ksl,
                length=length,
                isthick=True,
                shift_x=shift_x,
                shift_y=shift_y,
            ))
            names.append(f"tubefitter_mult_{i_reg:0{name_width}d}")

        return xt.Line(elements=elements, element_names=names)

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------

    def _set_raw_data(self, raw_data: pd.DataFrame) -> None:
        if not isinstance(raw_data, pd.DataFrame):
            raise TypeError(
                f"raw_data must be a pd.DataFrame with MultiIndex "
                f"('X', 'Y', 'Z'), got {type(raw_data).__name__}"
            )
        missing = set(_REQUIRED_COLUMNS) - set(raw_data.columns)
        if missing:
            raise ValueError(f"raw_data must have columns {_REQUIRED_COLUMNS}, missing {sorted(missing)}")

        idx = raw_data.index
        if list(idx.names) != list(_INDEX_NAMES):
            raise ValueError(f"raw_data index must be {_INDEX_NAMES}, got {idx.names}")

        scaled_index = pd.MultiIndex.from_arrays(
            [idx.get_level_values(lvl).astype(float) * self.distance_unit for lvl in idx.names],
            names=idx.names,
        )
        self.df_raw_data = raw_data.set_axis(scaled_index, axis=0)
        self.s_full = np.sort(self.df_raw_data.index.get_level_values("Z").unique()).astype(float)

    # ------------------------------------------------------------------
    # n_frames selection (residual_tol search)
    # ------------------------------------------------------------------

    def _filtered_z_values(self) -> np.ndarray:
        """Sorted unique z-values actually available to the tube fit -- i.e.
        after applying ``tube_radius`` (if set), same filter
        ``_build_linear_system`` applies. Falls back to ``self.s_full``
        (already sorted+unique) when there's no tube_radius to filter by."""
        assert self.df_raw_data is not None
        if self.tube_radius is None:
            assert self.s_full is not None
            return self.s_full
        idx = self.df_raw_data.index
        x = idx.get_level_values("X").to_numpy(dtype=float)
        y = idx.get_level_values("Y").to_numpy(dtype=float)
        z = idx.get_level_values("Z").to_numpy(dtype=float)
        mask = (x * x + y * y) <= self.tube_radius ** 2
        return np.unique(z[mask])

    def _filtered_n_pts_and_n_z(self) -> tuple[int, int]:
        """(n_pts, n_unique_z) actually available to the tube fit -- i.e.
        after applying ``tube_radius``, if set, same as ``_build_linear_system``
        does. ``_dof_ceiling``/``_z_resolution_ceiling`` need to bound
        n_frames against what the fit will really see, not the full
        (unfiltered) raw_data row count -- a tight tube_radius can drop
        both the point count and, in principle, whole z-slices entirely."""
        assert self.df_raw_data is not None
        idx = self.df_raw_data.index
        if self.tube_radius is None:
            return len(idx), int(idx.get_level_values("Z").nunique())
        x = idx.get_level_values("X").to_numpy(dtype=float)
        y = idx.get_level_values("Y").to_numpy(dtype=float)
        z = idx.get_level_values("Z").to_numpy(dtype=float)
        mask = (x * x + y * y) <= self.tube_radius ** 2
        return int(mask.sum()), int(np.unique(z[mask]).size)

    def _dof_ceiling(self) -> int:
        """
        Max usable n_frames: limited by both (a) the tube system's degrees
        of freedom -- 2*n_pts equations (Bx, By at every grid point) vs
        n_frames*n_pq unknowns -- and (b) the data's own z resolution (see
        ``_z_resolution_ceiling``). Beyond either, the system is
        underdetermined -- or a normal-equations block goes singular --
        regardless of how much data exists overall.
        """
        n_pts, _ = self._filtered_n_pts_and_n_z()
        eq_ceiling = (2 * n_pts) // len(self.pq_pairs)
        return min(eq_ceiling, self._z_resolution_ceiling())

    def _z_resolution_ceiling(self) -> int:
        """
        Max usable n_frames set by the data's own z resolution: beyond the
        number of distinct z-slices actually available (post-tube_radius),
        frames sit closer together than the data itself is sampled,
        guaranteeing some frame's normal-equations block goes entirely
        unconstrained (tent's local support means a raw-data row only ever
        contributes to the elementary interval its z falls in).
        """
        _, n_z = self._filtered_n_pts_and_n_z()
        return n_z

    def _trial_relative_residual(self, n_frames: int) -> tuple[float, float, "TubeFitter"]:
        """Fit a throwaway TubeFitter at n_frames, return its (Bskew, Bnorm)
        relative tube-fit residuals plus the fitted trial itself -- so the
        winning trial's expensive work (``_build_linear_system``) can be
        adopted by ``_search_n_frames`` instead of being thrown away and
        redone by the caller's subsequent explicit ``fit()`` call."""
        assert self.df_raw_data is not None
        with contextlib.redirect_stdout(io.StringIO()):
            trial = TubeFitter(
                raw_data=self.df_raw_data,  # already scaled by distance_unit
                n_frames=n_frames,
                distance_unit=1.0,
                deg=self.deg,
                tube_radius=self.tube_radius,
                fit_skew=self.fit_skew,
                y_symmetry=self.y_symmetry,
                field_tol=self.field_tol,
            )
            trial.fit()
        b = trial._b_vec
        n_bxby = trial._n_bx_by_rows
        rels = {}
        for field, rows in (("Bskew", slice(0, n_bxby, 2)), ("Bnorm", slice(1, n_bxby, 2))):
            rms = trial.tube_fit_residual_rms[field]
            field_rms = float(np.sqrt(np.mean(b[rows] ** 2)))
            rels[field] = rms / field_rms if field_rms > 0 else 0.0
        return rels["Bskew"], rels["Bnorm"], trial

    def _search_n_frames(self, residual_tol: float) -> int:
        """
        Smallest n_frames whose worst-case relative residual is <=
        residual_tol, found via geometric-doubling bracket + bisection (see
        examples/splineboris/003d_tube_fitter_nframes_scan.py, folded in here).
        Records every evaluated point in ``self.n_frames_search_trace``.
        """
        n_min = 2
        n_max = self._dof_ceiling()
        cache: dict[int, tuple[float, float]] = {}
        # Retains only the single smallest-n-so-far trial that still could
        # end up being the answer -- NOT a dict keyed by every evaluated n.
        # The search can probe ~15-20 different n_frames before converging,
        # and each trial retains O(n_pts) arrays (the design matrix, x/y,
        # b_vec) for the *full* dataset -- holding all of them alive at once
        # is what caused the ~20GB+ OOM kills fixed here, since every other
        # trial's system is genuinely never needed again once a smaller
        # passing n_frames is found (bisection's hi only ever shrinks).
        best_trial: tuple[int, "TubeFitter"] | None = None

        def meets(n: int) -> bool:
            nonlocal best_trial
            if n not in cache:
                rel_bskew, rel_bnorm, trial = self._trial_relative_residual(n)
                cache[n] = (rel_bskew, rel_bnorm)
                print(f"[TubeFitter]   n_frames={n:5d}  Bskew={rel_bskew * 100:6.2f}%  Bnorm={rel_bnorm * 100:6.2f}%")
                passes = max(cache[n]) <= residual_tol
                if passes and (best_trial is None or n < best_trial[0]):
                    best_trial = (n, trial)
                elif n == n_max and not passes:
                    # Special case: about to fall back to n_max anyway (see
                    # below) despite it not meeting the tolerance -- keep it
                    # so that fallback can still adopt it.
                    best_trial = (n, trial)
                return passes
            return max(cache[n]) <= residual_tol

        print(
            f"[TubeFitter] Searching for smallest n_frames with relative "
            f"residual <= {residual_tol * 100:.3g}% in [{n_min}, {n_max}]..."
        )
        if not meets(n_max):
            self.n_frames_search_trace = dict(cache)
            print(
                f"[TubeFitter] WARNING: residual_tol={residual_tol} not reachable "
                f"even at the DOF ceiling n_frames={n_max} (relative residual="
                f"{max(cache[n_max]) * 100:.2f}%). Falling back to "
                f"n_frames={n_max} (the best achievable)."
            )
            self.plot_n_frames_search(target=residual_tol, selected=n_max)
            assert best_trial is not None
            self._adopt_trial_system(best_trial[1])
            return n_max

        lo, hi = n_min, n_max
        n_probe = n_min
        while n_probe < hi and not meets(n_probe):
            lo = n_probe
            n_probe = min(n_probe * 2, hi)
        hi = n_probe

        while hi - lo > 1:
            mid = (lo + hi) // 2
            if meets(mid):
                hi = mid
            else:
                lo = mid

        self.n_frames_search_trace = dict(cache)
        print(f"[TubeFitter] Selected n_frames={hi} (relative residual={max(cache[hi]) * 100:.3f}%)")
        self.plot_n_frames_search(target=residual_tol, selected=hi)
        assert best_trial is not None and best_trial[0] == hi
        self._adopt_trial_system(best_trial[1])
        return hi

    def _adopt_trial_system(self, trial: "TubeFitter") -> None:
        """Stash the winning search trial's already-built (expensive) block-
        tridiagonal system, so the subsequent (always-required) explicit
        ``fit()`` call can adopt it in ``_build_linear_system`` instead of
        redoing that ~O(n_pts) accumulation pass from scratch. Everything
        downstream of it in ``fit()`` (the solve, Hermite conversion,
        to_fit flags, ...) is cheap and still runs normally, so it keeps
        printing its usual diagnostics."""
        self._cached_system = (
            trial._D, trial._E, trial._r,
            trial._residual_design, trial._residual_xy, trial._residual_pq,
            trial._b_vec, trial._n_bx_by_rows,
        )

    def plot_n_frames_search(self, target: float | None = None, selected: int | None = None) -> None:
        """
        Plot the residual_tol search trace (``self.n_frames_search_trace``):
        Bskew/Bnorm relative residual vs every ``n_frames`` evaluated during
        the search. Called automatically at the end of a ``residual_tol``
        search (whether or not the target was reached); call it again
        yourself if you want to re-plot it later.
        """
        import matplotlib.pyplot as plt

        if not self.n_frames_search_trace:
            raise RuntimeError(
                "No n_frames search trace available -- construct TubeFitter "
                "with residual_tol to populate it."
            )

        ns = sorted(self.n_frames_search_trace)
        bskew = [self.n_frames_search_trace[n][0] * 100 for n in ns]
        bnorm = [self.n_frames_search_trace[n][1] * 100 for n in ns]

        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        ax.plot(ns, bskew, "o-", color="tab:blue", label="Bskew")
        ax.plot(ns, bnorm, "o-", color="tab:orange", label="Bnorm")
        if target is not None:
            ax.axhline(target * 100, color="k", linestyle="--", linewidth=1,
                       label=f"target ({target * 100:.2g}%)")
        if selected is not None:
            ax.axvline(selected, color="tab:green", linestyle=":", linewidth=1.5,
                       label=f"n_frames = {selected}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_frames")
        ax.set_ylabel("Tube fit residual (% of field RMS)")
        ax.set_title("TubeFitter n_frames search (residual_tol)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.show()

    def _set_df_on_axis(self) -> None:
        x0, y0 = self.xy_point
        df_on = self.df_raw_data.xs((x0, y0), level=["X", "Y"]).sort_index().copy(deep=True)
        df_on.columns = pd.MultiIndex.from_tuples([(col, 0) for col in df_on.columns])
        self.df_on_axis_raw = df_on
        self.df_on_axis_fit = self.df_on_axis_raw.copy(deep=True)
        self.df_on_axis_fit.loc[:, :] = 0.0

    def _on_axis_multipole_from_psi(self, field: str, der: int) -> np.ndarray | None:
        """On-axis multipole series b_m or a_m evaluated from ``Psi`` at all ``s_full``."""
        assert self.frames is not None and self.Psi is not None and self.s_full is not None
        assert self.pq_to_idx is not None
        if field == "By":
            if (der, 1) not in self.pq_to_idx:
                return None
            coeffs = -math.factorial(der) * self.Psi[:, der, 1]
        elif field == "Bx":
            m = der + 1
            if (m, 0) not in self.pq_to_idx:
                return None
            coeffs = -math.factorial(m) * self.Psi[:, m, 0]
        else:
            raise ValueError(f"field must be 'Bx' or 'By', got {field!r}")
        return np.interp(self.s_full, self.frames, coeffs)

    def _populate_on_axis_from_psi(self) -> None:
        """Fill higher-order on-axis Bx/By columns from tube multipoles (post-``_solve``)."""
        assert self.df_on_axis_raw is not None
        for der in range(1, self.deg + 1):
            for field in ("Bx", "By"):
                series = self._on_axis_multipole_from_psi(field, der)
                if series is not None:
                    self.df_on_axis_raw[(field, der)] = series

    def _setup_frames(self) -> None:
        assert self.s_full is not None
        z_min, z_max = float(self.s_full[0]), float(self.s_full[-1])
        self.frames = np.linspace(z_min, z_max, self.n_frames)
        self.n_regions = self.n_frames - 1
        self._check_frame_z_resolution()

    def _check_frame_z_resolution(self) -> None:
        """
        Fail fast, before the expensive assembly + block-tridiagonal solve,
        if ``n_frames`` puts frames closer together than the data's own z
        resolution can support. Each raw-data row only ever contributes to
        the elementary interval ``[frames[k], frames[k+1])`` its z falls in
        (tent's local support), so a frame whose neighbouring interval(s) are
        completely empty of data z-slices gets an all-zero normal-equations
        block -- exactly the singular-block failure ``_solve_block_tridiagonal``
        raises, just caught here immediately instead of after building the
        full system. (Necessary, not sufficient: a nonempty interval can
        still leave a block singular if its transverse x/y spread doesn't
        constrain every (p, q) term -- this only catches the "no data at
        all" case.)
        """
        assert self.frames is not None
        z_filtered = self._filtered_z_values()
        bin_idx = _tent_interval_index(z_filtered, self.frames)
        counts = np.bincount(bin_idx, minlength=self.n_frames - 1)
        empty = np.flatnonzero(counts == 0)
        if len(empty) == 0:
            return
        k = int(empty[0])
        raise RuntimeError(
            f"n_frames={self.n_frames} puts frames closer together than the "
            f"data's z resolution supports: interval [{self.frames[k]:.6g}, "
            f"{self.frames[k + 1]:.6g}] (and {len(empty) - 1} others) "
            f"contain no raw-data z-slices at all -- try n_frames <= "
            f"{self._z_resolution_ceiling()} (the number of distinct "
            f"z-slices in the data)."
        )

    # ------------------------------------------------------------------
    # Tube fit (Bx, By)
    # ------------------------------------------------------------------

    def _build_linear_system(self) -> None:
        if self._cached_system is not None:
            (
                self._D, self._E, self._r,
                self._residual_design, self._residual_xy, self._residual_pq,
                self._b_vec, self._n_bx_by_rows,
            ) = self._cached_system
            self._cached_system = None
            return

        assert self.df_raw_data is not None
        assert self.frames is not None
        assert self.pq_pairs is not None

        idx = self.df_raw_data.index
        x = idx.get_level_values("X").to_numpy(dtype=float)
        y = idx.get_level_values("Y").to_numpy(dtype=float)
        z = idx.get_level_values("Z").to_numpy(dtype=float)
        bx = self.df_raw_data["Bx"].to_numpy(dtype=float)
        by = self.df_raw_data["By"].to_numpy(dtype=float)

        if self.tube_radius is not None:
            r2 = x * x + y * y
            mask = r2 <= self.tube_radius ** 2
            x, y, z, bx, by = x[mask], y[mask], z[mask], bx[mask], by[mask]

        n_pts = len(x)

        # Group the points by tent interval (2 active frames per point), so
        # each interval's contribution is a contiguous slice -- the only
        # per-point bookkeeping here is O(n_pts), not O(n_pts * n_pq).
        order, bounds, b0, b1 = _tent_intervals(z, self.frames)
        x, y, bx, by = x[order], y[order], bx[order], by[order]

        b_vec = np.empty(2 * n_pts, dtype=float)
        b_vec[0::2] = bx
        b_vec[1::2] = by

        pq = np.array(self.pq_pairs, dtype=int)

        # Block-tridiagonal normal-equations accumulators, built interval by
        # interval directly from the raw points (see
        # _accumulate_block_tridiagonal) -- no global A matrix (of size
        # O(n_pts * n_pq)) ever gets built.
        D, E, r = _accumulate_block_tridiagonal(
            bounds, b0, b1, x, y, bx, by, pq, self.n_frames,
        )

        self._D = D
        self._E = E
        self._r = r
        self._b_vec = b_vec
        self._n_bx_by_rows = 2 * n_pts
        # Retained for _report_tube_fit_residual's second pass (direct
        # per-point evaluation of the fitted field, no A matrix needed there
        # either) -- all O(n_pts) or smaller, not O(n_pts * n_pq).
        self._residual_design = (bounds, b0, b1)
        self._residual_xy = (x, y)
        self._residual_pq = pq

    def _solve(self) -> None:
        n_pq = len(self.pq_pairs)

        x_sol = self._solve_block_tridiagonal(self._D, self._E, self._r).reshape(-1)

        self.Psi = np.zeros((self.n_frames, self.M + 1, self.M + 1), dtype=float)
        flat = x_sol.reshape(self.n_frames, n_pq)
        for pq_idx, (p, q) in enumerate(self.pq_pairs):
            self.Psi[:, p, q] = flat[:, pq_idx]

        self._report_tube_fit_residual(flat)

    def _solve_block_tridiagonal(
        self, D: np.ndarray, E: np.ndarray, r: np.ndarray
    ) -> np.ndarray:
        """Block Thomas algorithm for a symmetric block-tridiagonal system:
        diagonal blocks ``D[j]``, super-diagonal blocks ``E[j]`` (coupling
        frame ``j`` to ``j + 1``; sub-diagonal is ``E[j].T`` by symmetry of
        ``A^T A``), right-hand-side blocks ``r[j]``. Returns the solution as
        an ``(n_frames, n_pq)`` array.

        A singular diagonal block means the data local to that frame doesn't
        constrain all its (p, q) parameters -- raised as an error rather than
        patched with regularization, so it surfaces as an actionable
        "increase data density / lower n_frames here" signal.
        """
        n_frames = D.shape[0]
        Dp = D.copy()
        rp = r.copy()
        for j in range(1, n_frames):
            try:
                L = np.linalg.solve(Dp[j - 1], E[j - 1]).T
            except np.linalg.LinAlgError as exc:
                z_lo, z_hi = self.frames[j - 1], self.frames[j]
                raise RuntimeError(
                    f"Tube fit normal-equations block for frame {j - 1} "
                    f"(z in [{z_lo:.6g}, {z_hi:.6g}]) is singular -- the "
                    f"data local to this frame doesn't constrain all "
                    f"(p, q) parameters. Try a smaller n_frames or a wider "
                    f"tube_radius."
                ) from exc
            Dp[j] = D[j] - L @ E[j - 1]
            rp[j] = r[j] - L @ rp[j - 1]

        x = np.empty_like(r)
        try:
            x[-1] = np.linalg.solve(Dp[-1], rp[-1])
        except np.linalg.LinAlgError as exc:
            z_lo, z_hi = self.frames[-2], self.frames[-1]
            raise RuntimeError(
                f"Tube fit normal-equations block for frame {n_frames - 1} "
                f"(z in [{z_lo:.6g}, {z_hi:.6g}]) is singular -- the data "
                f"local to this frame doesn't constrain all (p, q) "
                f"parameters. Try a smaller n_frames or a wider tube_radius."
            ) from exc
        for j in range(n_frames - 2, -1, -1):
            try:
                x[j] = np.linalg.solve(Dp[j], rp[j] - E[j] @ x[j + 1])
            except np.linalg.LinAlgError as exc:
                z_lo, z_hi = self.frames[max(j - 1, 0)], self.frames[j]
                raise RuntimeError(
                    f"Tube fit normal-equations block for frame {j} "
                    f"(z in [{z_lo:.6g}, {z_hi:.6g}]) is singular -- the "
                    f"data local to this frame doesn't constrain all "
                    f"(p, q) parameters. Try a smaller n_frames or a wider "
                    f"tube_radius."
                ) from exc
        return x

    def _report_tube_fit_residual(self, x_sol_flat: np.ndarray) -> None:
        """
        Report the residual of the tube linear system against the raw Bx/By
        values it was regressed against (der=0). Higher-order multipoles
        are read off other components of this same fitted Psi rather than
        being fit independently, so this single residual is the analogue of
        FieldFitter's per-field der=0 transverse-fit residual. (Bs is fit
        completely separately -- see ``_fit_bs`` -- so it has no residual
        here.)

        ``x_sol_flat`` is ``(n_frames, n_pq)`` -- evaluated directly at every
        raw point via ``_accumulate_residual_sumsq`` (same direct-evaluation
        approach ``_build_linear_system`` uses to avoid ever building a
        global ``A`` matrix), rather than a stored-matrix matvec.
        """
        bounds, b0, b1 = self._residual_design
        x, y = self._residual_xy
        bx = self._b_vec[0::2]
        by = self._b_vec[1::2]

        sumsq_res_bx, sumsq_res_by, sumsq_bx, sumsq_by = _accumulate_residual_sumsq(
            bounds, b0, b1, x, y, bx, by, self._residual_pq, x_sol_flat,
        )

        n_pts = len(bx)
        self.tube_fit_residual_rms = {}
        for field, sumsq_res, sumsq_sig in (
            ("Bskew", sumsq_res_bx, sumsq_bx), ("Bnorm", sumsq_res_by, sumsq_by),
        ):
            rms = float(np.sqrt(sumsq_res / n_pts))
            field_rms = float(np.sqrt(sumsq_sig / n_pts))
            rel = rms / field_rms if field_rms > 0 else 0.0
            self.tube_fit_residual_rms[field] = rms
            print(f"[TubeFitter] {field} tube fit residual (der=0): "
                  f"RMS = {rms:.3e} T ({rel * 100:.2f}% of field RMS)")

    def _fit_bs(self) -> None:
        assert self.frames is not None
        assert self.s_full is not None
        assert self.df_on_axis_raw is not None

        bs_on_axis = self.df_on_axis_raw[("Bs", 0)].to_numpy(dtype=float)
        n = self.n_frames

        # Same tent normal equations as the tube fit, with one unknown per
        # frame instead of n_pq: a point in interval k adds b0^2 / b1^2 to
        # diagonals k / k+1 and b0*b1 to the coupling between them.
        k, b0, b1 = _tent_weights(self.s_full, self.frames)
        D = np.bincount(k, b0 * b0, n) + np.bincount(k + 1, b1 * b1, n)
        E = np.bincount(k, b0 * b1, n - 1)
        r = np.bincount(k, b0 * bs_on_axis, n) + np.bincount(k + 1, b1 * bs_on_axis, n)

        self.Psi_bs = self._solve_block_tridiagonal(
            D[:, None, None], E[:, None, None], r[:, None]
        )[:, 0]

    def check_trace_consistency(self) -> dict[str, np.ndarray | float]:
        """
        Fit-quality diagnostic: compare the freely-fit ``Psi[0,2](z)`` trace
        term against the div(B) = 0 prediction ``-Psi[2,0](z) + Bs'(z)/2``.

        Not needed for correct SplineBoris export -- only ``q=0``/``q=1``
        rows are ever exported, and the downstream Table-1 field evaluator
        regenerates ``q=2`` content from ``(a_n, b_n, b_s)`` alone,
        Maxwell-consistent by construction. This is purely a sanity check
        that the tube's own (unused) ``q=2`` fit -- and by extension its
        resolution of the field's z-structure -- is consistent with the
        physical field: a large gap means the Bx/By data isn't finely
        resolved enough in z (raise ``n_frames``, shrink ``tube_radius``,
        or check grid sampling), independent of whether that shows up in
        the exported (a_n, b_n, b_s) themselves.

        Requires ``y_symmetry=False``, so ``(0, 2)`` -- fit freely from
        Bx/By data, like every other coefficient -- is in the basis.
        """
        if self.y_symmetry:
            raise RuntimeError(
                "check_trace_consistency needs y_symmetry=False: Psi[0,2] "
                "is excluded from the basis entirely when y_symmetry=True."
            )
        assert self.Psi is not None and self.frames is not None
        assert self.pq_to_idx is not None
        if (0, 2) not in self.pq_to_idx or (2, 0) not in self.pq_to_idx:
            raise RuntimeError(
                "check_trace_consistency needs both (0,2) and (2,0) in the "
                "fitted basis -- construct with deg>=1."
            )
        if self.Psi_bs is None:
            self._fit_bs()

        # At the frames, the tent interpolant is just the frame values.
        psi_02 = self.Psi[:, 0, 2].copy()
        psi_20 = self.Psi[:, 2, 0].copy()
        bs_prime = _tent_slope(self.frames, self.frames, self.Psi_bs)
        predicted_psi_02 = -psi_20 + 0.5 * bs_prime

        gap = psi_02 - predicted_psi_02
        scale = max(np.max(np.abs(psi_02)), np.max(np.abs(predicted_psi_02)), 1e-30)
        relative_rms = float(np.sqrt(np.mean(gap ** 2)) / scale)
        print(
            f"[TubeFitter] trace consistency check: Psi[0,2] vs "
            f"-Psi[2,0]+Bs'/2 gap RMS = {relative_rms * 100:.3f}% of scale"
        )
        return {
            "z": self.frames.copy(),
            "psi_02_tube": psi_02,
            "psi_02_predicted": predicted_psi_02,
            "gap": gap,
            "relative_rms": relative_rms,
        }

    # ------------------------------------------------------------------
    # Hermite conversion and output tables
    # ------------------------------------------------------------------

    def _convert_to_hermite(self) -> None:
        assert self.frames is not None
        assert self.Psi_bs is not None

        series = {}
        for der in range(self.deg + 1):
            m = der + 1
            # b_m = -(m-1)! * C_{m-1,1}  ->  Psi[:, der, 1]
            series[("Bnorm", der)] = -math.factorial(der) * self.Psi[:, der, 1]
            if (m, 0) in self.pq_to_idx:
                # a_m = -m! * C_{m,0}  (from d^{m-1} B_x / dx^{m-1} |_{x=y=0})
                series[("Bskew", der)] = -math.factorial(m) * self.Psi[:, m, 0]
        series[("Bs", 0)] = self.Psi_bs

        self._hermite = {}
        for (field, der), coeffs in series.items():
            params = np.stack(_hermite_from_tent(self.frames, coeffs), axis=1)
            for i_reg in range(self.n_regions):
                self._hermite[(field, der, i_reg)] = tuple(params[i_reg].tolist())

    def _assign_to_fit_flags(self) -> None:
        """Set ``component_to_fit`` using the same relative scale test as FieldFitter."""
        assert self.df_on_axis_raw is not None
        assert self.df_raw_data is not None
        assert self.pq_to_idx is not None

        col_map = {"Bskew": "Bx", "Bnorm": "By", "Bs": "Bs"}
        abs_max = 0.0
        for col in ("Bx", "By", "Bs"):
            try:
                abs_max = max(abs_max, float(np.max(np.abs(self.df_on_axis_raw[(col, 0)].values))))
            except KeyError:
                pass
        if abs_max == 0.0:
            abs_max = 1.0

        x_max = float(np.max(np.abs(self.df_raw_data.index.get_level_values("X"))))
        self.component_to_fit = {}

        for field in ("Bskew", "Bnorm", "Bs"):
            ders = [0] if field == "Bs" else list(range(self.deg + 1))
            for der in ders:
                in_basis = True
                if field == "Bskew":
                    in_basis = (der + 1, 0) in self.pq_to_idx
                elif field == "Bnorm":
                    in_basis = (der, 1) in self.pq_to_idx

                try:
                    series = self.df_on_axis_raw[(col_map[field], der)].values
                except KeyError:
                    self.component_to_fit[(field, der)] = False
                    continue

                field_der_max = float(np.max(np.abs(series)))
                relative_max = field_der_max / math.factorial(der) * (x_max ** der)
                significant = relative_max >= self.field_tol * abs_max
                self.component_to_fit[(field, der)] = bool(in_basis and significant)
                print(
                    f"{field} der={der} -> to_fit={str(self.component_to_fit[(field, der)]):<5} "
                    f"(rel_max={relative_max:.3e}, tol={self.field_tol * abs_max:.3e})"
                )

    def _populate_df_fit_pars(self) -> None:
        assert self.s_full is not None
        assert self.frames is not None
        assert self.n_regions is not None
        assert self._hermite is not None

        idx_extrema = _frame_indices(self.s_full, self.frames)
        index_width = len(str(self.n_regions - 1)) if self.n_regions > 1 else 1
        rows: list[dict] = []

        for field in ("Bskew", "Bnorm", "Bs"):
            ders = [0] if field == "Bs" else list(range(self.deg + 1))

            for der in ders:
                to_fit = self.component_to_fit.get((field, der), False)
                if field == "Bskew":
                    prefix = f"Bskew_{der}"
                elif field == "Bnorm":
                    prefix = f"Bnorm_{der}"
                else:
                    prefix = "Bs"
                pars = [f"{prefix}_{s}" for s in xt.SplineBoris._SB_HERMITE_SUFFIXES]

                for i_reg in range(self.n_regions):
                    idx_start = int(idx_extrema[i_reg])
                    idx_end = int(idx_extrema[i_reg + 1])
                    # s_start/s_end must be the TRUE frame positions -- the
                    # same ones used to compute self._hermite via
                    # _hermite_from_tent(frames, ...) in _convert_to_hermite. Using the nearest-raw-sample
                    # snapped positions (s_full[idx_start/idx_end]) instead
                    # silently mismatches the L used to derive the Hermite
                    # derivative terms (c2, c4) from the L used to reconstruct
                    # them downstream (hermite_to_polynomial scales those
                    # terms by L) -- see
                    # examples/splineboris/claude_notes/tube_hermite_export_boundary_bug.md.
                    s_start = float(self.frames[i_reg])
                    s_end = float(self.frames[i_reg + 1])
                    region_name = f"Poly_{i_reg:0{index_width}d}"

                    if to_fit and (field, der, i_reg) in self._hermite:
                        hermite = self._hermite[(field, der, i_reg)]
                    else:
                        hermite = (0.0, 0.0, 0.0, 0.0, 0.0)

                    for param_index, (name, val) in enumerate(zip(pars, hermite)):
                        rows.append({
                            "field_component": field,
                            "derivative_x": der,
                            "region_name": region_name,
                            "s_start": s_start,
                            "s_end": s_end,
                            "idx_start": idx_start,
                            "idx_end": idx_end,
                            "param_index": param_index,
                            "param_name": name,
                            "param_value": val,
                            "to_fit": to_fit,
                        })

        self.df_fit_pars = pd.DataFrame(rows)
        self.df_fit_pars.set_index(
            [
                "field_component",
                "derivative_x",
                "region_name",
                "s_start",
                "s_end",
                "idx_start",
                "idx_end",
                "param_index",
            ],
            inplace=True,
        )
        self.df_fit_pars.sort_index(inplace=True)

    def _fill_df_on_axis_fit(self) -> None:
        assert self.frames is not None
        assert self.Psi is not None
        assert self.s_full is not None
        assert self.df_on_axis_fit is not None

        n_z = len(self.s_full)
        for der in range(self.deg + 1):
            if self.component_to_fit.get(("Bnorm", der), False):
                series = self._on_axis_multipole_from_psi("By", der)
                self.df_on_axis_fit[("By", der)] = series if series is not None else np.zeros(n_z)
            else:
                self.df_on_axis_fit[("By", der)] = np.zeros(n_z)

            if self.component_to_fit.get(("Bskew", der), False):
                series = self._on_axis_multipole_from_psi("Bx", der)
                self.df_on_axis_fit[("Bx", der)] = series if series is not None else np.zeros(n_z)
            else:
                self.df_on_axis_fit[("Bx", der)] = np.zeros(n_z)

        if self.component_to_fit.get(("Bs", 0), False):
            assert self.Psi_bs is not None
            self.df_on_axis_fit[("Bs", 0)] = np.interp(self.s_full, self.frames, self.Psi_bs)
        else:
            self.df_on_axis_fit[("Bs", 0)] = np.zeros(n_z)

    # ------------------------------------------------------------------
    # Plotting (Bx / By / Bs on-axis columns)
    # ------------------------------------------------------------------

    def plot_fields(self, der: int = 0) -> None:
        import matplotlib.pyplot as plt

        if self.df_on_axis_raw is None or self.df_on_axis_fit is None:
            raise RuntimeError("`df_on_axis_raw` and `df_on_axis_fit` must be set before plotting.")

        s = self.s_full

        def get_series(df, field, d):
            try:
                return df[(field, d)].to_numpy()
            except KeyError:
                ref = df.iloc[:, 0].to_numpy()
                return np.zeros_like(ref)

        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)
        raw_label = "Measured on axis" if der == 0 else "Tube multipoles"
        fit_label = "Fit" if der == 0 else "Exported (to_fit)"
        ax1.plot(s, get_series(self.df_on_axis_raw, "Bx", der), label=raw_label)
        ax1.plot(s, get_series(self.df_on_axis_fit, "Bx", der), label=fit_label, linestyle="--")
        ax2.plot(s, get_series(self.df_on_axis_raw, "By", der), label=raw_label)
        ax2.plot(s, get_series(self.df_on_axis_fit, "By", der), label=fit_label, linestyle="--")
        ax3.plot(s, get_series(self.df_on_axis_raw, "Bs", der), label=raw_label)
        ax3.plot(s, get_series(self.df_on_axis_fit, "Bs", der), label=fit_label, linestyle="--")

        def _borders_for_field(field_component: str):
            if self.df_fit_pars is None:
                return []
            try:
                lvl_field = np.asarray(self.df_fit_pars.index.get_level_values("field_component"))
                lvl_der = np.asarray(self.df_fit_pars.index.get_level_values("derivative_x")).astype(int)
                mask = (lvl_field == field_component) & (lvl_der == int(der))
                if not np.any(mask):
                    return []
                s_start_vals = np.asarray(self.df_fit_pars.index.get_level_values("s_start"))[mask].astype(float)
                s_end_vals = np.asarray(self.df_fit_pars.index.get_level_values("s_end"))[mask].astype(float)
                return np.unique(np.concatenate((s_start_vals, s_end_vals)))
            except Exception:
                return []

        for field_ax, ax, fc in [("Bx", ax1, "Bskew"), ("By", ax2, "Bnorm"), ("Bs", ax3, "Bs")]:
            for s_border in _borders_for_field(fc):
                ax.axvline(x=s_border, color="k", linestyle="--", linewidth=1, alpha=0.3)

        if der == 2:
            x_label = r"$\frac{d^2 B_x}{d x^2}$"
            y_label = r"$\frac{d^2 B_y}{d x^2}$"
            s_label = r"$\frac{d^2 B_s}{d x^2}$"
        elif der == 1:
            x_label = r"$\frac{d B_x}{d x}$"
            y_label = r"$\frac{d B_y}{d x}$"
            s_label = r"$\frac{d B_s}{d x}$"
        else:
            x_label = r"$B_x$"
            y_label = r"$B_y$"
            s_label = r"$B_s$"

        ax1.set_title(f"Magnetic Field at (X, Y) = {self.xy_point}")
        ax1.set_ylabel(f"Horizontal Field, {x_label} [T]")
        ax2.set_ylabel(f"Vertical Field, {y_label} [T]")
        ax3.set_ylabel(f"Longitudinal Field, {s_label} [T]")
        ax3.set_xlabel(r"Longitudinal Position, $s$ [m]")
        ax1.legend(loc="lower right")
        ax2.legend(loc="lower right")
        ax3.legend(loc="upper right")
        ax1.grid()
        ax2.grid()
        ax3.grid()
        plt.show()

    def plot_integrated_fields(self) -> None:
        import matplotlib.pyplot as plt

        if self.df_on_axis_raw is None or self.df_on_axis_fit is None:
            raise RuntimeError("`df_on_axis_raw` and `df_on_axis_fit` must be set before plotting.")

        s = self.s_full
        Bx_raw = self.df_on_axis_raw[("Bx", 0)].to_numpy()
        By_raw = self.df_on_axis_raw[("By", 0)].to_numpy()
        try:
            Bs_raw = self.df_on_axis_raw[("Bs", 0)].to_numpy()
        except KeyError:
            Bs_raw = np.zeros_like(Bx_raw)

        Bx_fit = self.df_on_axis_fit[("Bx", 0)].to_numpy()
        By_fit = self.df_on_axis_fit[("By", 0)].to_numpy()
        try:
            Bs_fit = self.df_on_axis_fit[("Bs", 0)].to_numpy()
        except KeyError:
            Bs_fit = np.zeros_like(Bx_fit)

        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)
        ax1.plot(s, sc.integrate.cumulative_trapezoid(Bx_raw, x=s, initial=0), label="Raw Data")
        ax1.plot(s, sc.integrate.cumulative_trapezoid(Bx_fit, x=s, initial=0), label="Fit", linestyle="--")
        ax2.plot(s, sc.integrate.cumulative_trapezoid(By_raw, x=s, initial=0), label="Raw Data")
        ax2.plot(s, sc.integrate.cumulative_trapezoid(By_fit, x=s, initial=0), label="Fit", linestyle="--")
        ax3.plot(s, sc.integrate.cumulative_trapezoid(Bs_raw, x=s, initial=0), label="Raw Data")
        ax3.plot(s, sc.integrate.cumulative_trapezoid(Bs_fit, x=s, initial=0), label="Fit", linestyle="--")

        ax1.set_title(f"Integrated Magnetic Field at (X, Y) = {self.xy_point}")
        ax1.set_ylabel(r"Integrated Horizontal Field, $\int B_x \, ds$ [T·m]")
        ax2.set_ylabel(r"Integrated Vertical Field, $\int B_y \, ds$ [T·m]")
        ax3.set_ylabel(r"Integrated Longitudinal Field, $\int B_s \, ds$ [T·m]")
        ax3.set_xlabel(r"Longitudinal Position, $s$ [m]")
        ax1.legend(loc="lower right")
        ax2.legend(loc="lower right")
        ax3.legend(loc="upper right")
        ax1.grid()
        ax2.grid()
        ax3.grid()
        plt.show()


def _constant_dipole_sign_check() -> None:
    """Verify Bnorm der=0 endpoints match a uniform By = B0 field."""
    B0 = 0.5
    xs = np.linspace(-0.002, 0.002, 3)
    ys = np.linspace(-0.002, 0.002, 3)
    zs = np.linspace(0.0, 1.0, 21)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    df = pd.DataFrame(
        {
            "X": xg.ravel(),
            "Y": yg.ravel(),
            "Z": zg.ravel(),
            "Bx": np.zeros(xg.size),
            "By": np.full(xg.size, B0),
            "Bs": np.zeros(xg.size),
        }
    ).set_index(["X", "Y", "Z"])

    fitter = TubeFitter(df, n_frames=8, distance_unit=1.0, deg=2)
    fitter.fit()

    sub = fitter.df_fit_pars.loc[("Bnorm", 0)].reset_index()
    c1 = sub.loc[sub["param_index"] == 0, "param_value"].iloc[0]
    c3 = sub.loc[sub["param_index"] == 2, "param_value"].iloc[0]
    if not (np.isclose(c1, B0, rtol=1e-2) and np.isclose(c3, B0, rtol=1e-2)):
        raise AssertionError(
            f"Constant dipole sign check failed: "
            f"expected val_start/val_end ~ {B0}, got c1={c1}, c3={c3}"
        )
    print(
        f"Constant dipole sign check passed: "
        f"Bnorm_0 val_start={c1:.6f}, val_end={c3:.6f} (B0={B0})"
    )


if __name__ == "__main__":
    _constant_dipole_sign_check()

    dz = 0.001
    file_path = Path(__file__).resolve().parents[2] / "test_data" / "sls" / "simona_field_map.txt"
    df_raw = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["X", "Y", "Z", "Bx", "By", "Bs"],
        dtype=float,
    ).set_index(["X", "Y", "Z"])

    deg = 2
    fitter = TubeFitter(
        raw_data=df_raw,
        n_frames=550,
        distance_unit=dz,
        deg=deg,
        tube_radius=0.001,
    )
    print("\n=== Fitting ===")
    fitter.fit()
    print(
        f"Fit complete: {fitter.n_regions} regions, "
        f"{len(fitter.pq_pairs)} (p,q) pairs per frame"
    )
    for der in range(deg + 1):
        fitter.plot_fields(der=der)
