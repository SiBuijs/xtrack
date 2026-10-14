"""
Global magnetic-field fitting via the tube approach (Riemann & Aiba, IPAC2021).

Stage 1 of the SplineBoris field-map pipeline (the multipole finder). Fits a
scalar potential
    Omega_tilde(x, y, z) = sum_{j,p,q} Psi[j,p,q] * x^p * y^q * beta_j(z)
with tent (degree-1 B-spline) longitudinal basis functions beta_j(z) -- i.e.
straight-line interpolation between frames -- and returns the on-axis
components at the frame positions z_j, as data points for the longitudinal
fit (stage 2, ``LongitudinalFitter``).

Conventions (h = 0, straight frame, B = -grad Phi):
    - Minus sign is applied in the sparse system rows for Bx and By.
    - ("By", n)(z_j) = d^n B_y/dx^n (0, 0, z_j) = -n! * Psi[j, n, 1]
    - ("Bx", n)(z_j) = d^n B_x/dx^n (0, 0, z_j) = -(n+1)! * Psi[j, n+1, 0]
    - ("Bs", 0): not part of the tube fit; ``on_axis_bs()`` returns the
      map's own on-axis B_s on its planes. Only ``q=0`` and ``q=1`` rows of
      ``Psi`` are passed on -- any ``q>=2`` content the fit picks up (only
      possible when ``y_symmetry=False``) is fit freely, purely to keep it
      from biasing the ``q=0``/``q=1`` columns, and is itself discarded.
      This mirrors the Van der Schueren potential's own Cauchy data (phi_0,
      phi_1 only): the downstream Table-1 field evaluator regenerates all
      q>=2 structure from the on-axis components alone, Maxwell-consistent
      by construction -- see
      examples/splineboris/claude_notes/tube_schueren_integration.md. Use
      ``check_trace_consistency()`` (a diagnostic, not a correction) to see
      how well that assumption holds on a given dataset.
    - Default symmetry: only (p, q) with odd q; all (p, 0) skew terms if fit_skew=True
"""

from __future__ import annotations

import contextlib
import io
import math

import numpy as np
import pandas as pd

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


class TubeFitter:
    """
    Fit 3D magnetic field maps using the tube approach with B-splines in z.

    Parameters
    ----------
    raw_data :
        ``pd.DataFrame`` with MultiIndex ``('X', 'Y', 'Z')`` and columns
        ``('Bx', 'By', 'Bs')``.
    n_frames :
        Number of uniformly spaced longitudinal frames (tent peaks). One
        frame per map plane avoids the slight smoothing that coarser tent
        frames apply to the on-axis components. Mutually exclusive with
        ``residual_tol`` (specifying both raises ``ValueError``). If neither
        is given, defaults to ``DEFAULT_N_FRAMES`` (clamped to the valid
        ``[2, dof_ceiling]`` range) -- a reasonable middle ground, not tuned
        to any particular fit-quality target.
    residual_tol :
        If given (and ``n_frames`` is not), search for the smallest
        ``n_frames`` whose worst-case relative tube-fit residual (Bx/By,
        the relative version of ``tube_fit_residual_rms``) is <= this value,
        and use that. The search fits the tube system repeatedly (geometric
        doubling to bracket the transition, then integer bisection: ~log2 of
        the search range), so it can take a while for large datasets -- once
        you know a good value, prefer passing a fixed ``n_frames`` instead.
        Every evaluated ``n_frames`` is recorded in
        ``self.n_frames_search_trace`` (``{n_frames: (rel_bx, rel_by)}``)
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
        (p, q) with odd q (plus the (p, 0) terms), so the ("By", n) and
        ("Bx", n) components are decoupled accordingly. If False (default),
        also include even-q pairs, allowing a field with no assumed
        y-parity. Only the q=0 and q=1 rows are ever passed on, regardless
        of this setting -- any q>=2 content fitted when ``y_symmetry=False``
        is fit freely, purely to keep real even-q structure in the data
        from biasing the q=0/q=1 columns, and is otherwise discarded (see
        examples/splineboris/claude_notes/tube_schueren_integration.md).

    The longitudinal basis is a fixed tent (hat-function, degree-1
    B-spline) basis: one tent per frame, peaking there and reaching zero at
    the neighbouring frames, so every ``C_pq(z)`` is the straight-line
    interpolation of its frame values ``Psi[:, p, q]``
    (``np.interp(z, frames, Psi[:, p, q])``). Only the frame values are
    passed on (``on_axis_multipoles()``); the longitudinal shape of the
    exported field is fitted separately by ``LongitudinalFitter``.
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

        self.frames: np.ndarray | None = None
        self.pq_pairs = _generate_pq_pairs(self.M, self.y_symmetry, self.fit_skew)
        self.pq_to_idx = {pq: i for i, pq in enumerate(self.pq_pairs)}

        self.s_full: np.ndarray | None = None
        self.Psi: np.ndarray | None = None

        self.df_raw_data: pd.DataFrame | None = None
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
        """Run the tube fit, populating ``Psi``."""
        if self.df_raw_data is None:
            raise RuntimeError("Raw data must be provided before calling fit().")
        self._setup_frames()
        self._build_linear_system()
        self._solve()

    def on_axis_multipoles(self) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
        """Stage-1 output: ``(z, F, names)`` with ``z`` the frame positions,
        ``F[j, i]`` the value of component ``names[i]`` (``("By", n)`` or
        ``("Bx", n)``, see module docstring) at ``z[j]``. Components whose
        (p, q) pair is not in the basis are left out."""
        if self.Psi is None:
            raise RuntimeError("Call fit() before on_axis_multipoles().")
        names, columns = [], []
        for n in range(self.deg + 1):
            if (n, 1) in self.pq_to_idx:
                names.append(("By", n))
                columns.append(-math.factorial(n) * self.Psi[:, n, 1])
            if (n + 1, 0) in self.pq_to_idx:
                names.append(("Bx", n))
                columns.append(-math.factorial(n + 1) * self.Psi[:, n + 1, 0])
        return self.frames.copy(), np.column_stack(columns), names

    def on_axis_bs(self) -> tuple[np.ndarray, np.ndarray]:
        """``(z, Bs)``: the map's own on-axis B_s on its planes (not part of
        the tube fit)."""
        df_on = self.df_raw_data.xs((0.0, 0.0), level=["X", "Y"]).sort_index()
        return df_on.index.to_numpy(dtype=float), df_on["Bs"].to_numpy(dtype=float)

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
        """Fit a throwaway TubeFitter at n_frames, return its (Bx, By)
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
            )
            trial.fit()
        b = trial._b_vec
        n_bxby = trial._n_bx_by_rows
        rels = {}
        for field, rows in (("Bx", slice(0, n_bxby, 2)), ("By", slice(1, n_bxby, 2))):
            rms = trial.tube_fit_residual_rms[field]
            field_rms = float(np.sqrt(np.mean(b[rows] ** 2)))
            rels[field] = rms / field_rms if field_rms > 0 else 0.0
        return rels["Bx"], rels["By"], trial

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
                rel_bx, rel_by, trial = self._trial_relative_residual(n)
                cache[n] = (rel_bx, rel_by)
                print(f"[TubeFitter]   n_frames={n:5d}  Bx={rel_bx:.2e}  By={rel_by:.2e}")
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
            f"residual <= {residual_tol:.2e} in [{n_min}, {n_max}]..."
        )
        if not meets(n_max):
            self.n_frames_search_trace = dict(cache)
            print(
                f"[TubeFitter] WARNING: residual_tol={residual_tol} not reachable "
                f"even at the DOF ceiling n_frames={n_max} (relative residual="
                f"{max(cache[n_max]):.2e}). Falling back to "
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
        print(f"[TubeFitter] Selected n_frames={hi} (relative residual={max(cache[hi]):.2e})")
        self.plot_n_frames_search(target=residual_tol, selected=hi)
        assert best_trial is not None and best_trial[0] == hi
        self._adopt_trial_system(best_trial[1])
        return hi

    def _adopt_trial_system(self, trial: "TubeFitter") -> None:
        """Stash the winning search trial's already-built (expensive) block-
        tridiagonal system, so the subsequent (always-required) explicit
        ``fit()`` call can adopt it in ``_build_linear_system`` instead of
        redoing that ~O(n_pts) accumulation pass from scratch. Everything
        downstream of it in ``fit()`` (the solve and residual report) is
        cheap and still runs normally, so it keeps printing its usual
        diagnostics."""
        self._cached_system = (
            trial._D, trial._E, trial._r,
            trial._residual_design, trial._residual_xy, trial._residual_pq,
            trial._b_vec, trial._n_bx_by_rows,
        )

    def plot_n_frames_search(self, target: float | None = None, selected: int | None = None) -> None:
        """
        Plot the residual_tol search trace (``self.n_frames_search_trace``):
        Bx/By relative residual vs every ``n_frames`` evaluated during
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
        bx_rel = [self.n_frames_search_trace[n][0] for n in ns]
        by_rel = [self.n_frames_search_trace[n][1] for n in ns]

        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        ax.plot(ns, bx_rel, "o-", color="tab:blue", label="Bx")
        ax.plot(ns, by_rel, "o-", color="tab:orange", label="By")
        if target is not None:
            ax.axhline(target, color="k", linestyle="--", linewidth=1,
                       label=f"target ({target:.2e})")
        if selected is not None:
            ax.axvline(selected, color="tab:green", linestyle=":", linewidth=1.5,
                       label=f"n_frames = {selected}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_frames")
        ax.set_ylabel("Tube fit residual / field RMS")
        ax.set_title("TubeFitter n_frames search (residual_tol)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.show()

    def _setup_frames(self) -> None:
        assert self.s_full is not None
        z_min, z_max = float(self.s_full[0]), float(self.s_full[-1])
        frames = np.linspace(z_min, z_max, self.n_frames)
        # Frames that coincide with a map plane up to rounding take the
        # plane's exact z -- otherwise a plane a few ulp left of its frame
        # lands in the interval before it (e.g. one frame per plane).
        i = np.clip(np.searchsorted(self.s_full, frames), 1, len(self.s_full) - 1)
        nearest = np.where(frames - self.s_full[i - 1] < self.s_full[i] - frames,
                           self.s_full[i - 1], self.s_full[i])
        snap = np.abs(nearest - frames) <= 1e-9 * (z_max - z_min)
        frames[snap] = nearest[snap]
        self.frames = frames
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
            ("Bx", sumsq_res_bx, sumsq_bx), ("By", sumsq_res_by, sumsq_by),
        ):
            rms = float(np.sqrt(sumsq_res / n_pts))
            field_rms = float(np.sqrt(sumsq_sig / n_pts))
            rel = rms / field_rms if field_rms > 0 else 0.0
            self.tube_fit_residual_rms[field] = rms
            print(f"[TubeFitter] {field} tube fit residual (der=0): "
                  f"RMS = {rms:.3e} T ({rel:.2e} of field RMS)")

    def check_trace_consistency(self, bs_prime=None) -> dict[str, np.ndarray | float]:
        """
        Fit-quality diagnostic: compare the freely-fit ``Psi[0,2](z)`` trace
        term against the div(B) = 0 prediction ``-Psi[2,0](z) + Bs'(z)/2``.

        Not needed for correct SplineBoris export -- only ``q=0``/``q=1``
        rows are ever exported, and the downstream Table-1 field evaluator
        regenerates ``q=2`` content from the on-axis components alone,
        Maxwell-consistent by construction. This is purely a sanity check
        that the tube's own (unused) ``q=2`` fit -- and by extension its
        resolution of the field's z-structure -- is consistent with the
        physical field: a large gap means the Bx/By data isn't finely
        resolved enough in z (raise ``n_frames``, shrink ``tube_radius``,
        or check grid sampling), independent of whether that shows up in
        the on-axis components themselves.

        Requires ``y_symmetry=False``, so ``(0, 2)`` -- fit freely from
        Bx/By data, like every other coefficient -- is in the basis.

        ``bs_prime`` is a callable giving dBs/ds at given z (e.g. the
        derivative of a ``LongitudinalFitter`` spline); by default it is the
        finite-difference slope of the map's on-axis Bs.
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
        if bs_prime is None:
            z_bs, bs = self.on_axis_bs()
            bs_prime_frames = np.interp(self.frames, z_bs, np.gradient(bs, z_bs))
        else:
            bs_prime_frames = np.asarray(bs_prime(self.frames), dtype=float)

        # At the frames, the tent interpolant is just the frame values.
        psi_02 = self.Psi[:, 0, 2].copy()
        psi_20 = self.Psi[:, 2, 0].copy()
        predicted_psi_02 = -psi_20 + 0.5 * bs_prime_frames

        gap = psi_02 - predicted_psi_02
        scale = max(np.max(np.abs(psi_02)), np.max(np.abs(predicted_psi_02)), 1e-30)
        relative_rms = float(np.sqrt(np.mean(gap ** 2)) / scale)
        print(
            f"[TubeFitter] trace consistency check: Psi[0,2] vs "
            f"-Psi[2,0]+Bs'/2 gap RMS = {relative_rms:.2e} of scale"
        )
        return {
            "z": self.frames.copy(),
            "psi_02_tube": psi_02,
            "psi_02_predicted": predicted_psi_02,
            "gap": gap,
            "relative_rms": relative_rms,
        }
