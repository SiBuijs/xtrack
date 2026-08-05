"""
Global magnetic-field fitting via the tube approach (Riemann & Aiba, IPAC2021).

Fits a scalar potential
    Omega_tilde(x, y, z) = sum_{j,p,q} Psi[j,p,q] * x^p * y^q * beta_j(z)
with B-spline longitudinal basis functions beta_j(z), then converts multipole
coefficients to the same Hermite-quartic ``df_fit_pars`` format as ``FieldFitter``.

Conventions (h = 0, straight frame, B = -grad Phi):
    - Minus sign is applied in the sparse system rows for Bx and By.
    - C_{p,q}(z) = sum_j Psi[j,p,q] * beta_j(z)
    - b_m(z) = -(m-1)! * C_{m-1,1}(z)  ->  ``Bnorm``, derivative_x = m-1
    - a_m(z) = -m! * C_{m,0}(z)  ->  ``Bskew``, derivative_x = m-1  (uses Psi[:, m, 0])
    - b_s(z) fitted independently (1D B-spline on on-axis Bs)
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
from scipy.interpolate import BSpline
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsmr

_REQUIRED_COLUMNS = ("Bx", "By", "Bs")
_INDEX_NAMES = ("X", "Y", "Z")

KERNEL_TO_DEGREE = {
    "tent": 1,       # B_1, C^0 — narrowest support, derivative discontinuous at frames
    "quadratic": 2,  # B_2, C^1 — narrower than cubic, derivatives continuous
    "cubic": 3,      # B_3, C^2 — widest support but smoothest (default)
}

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


def _safe_derivative_at(f: BSpline, s: float, eps: float) -> float:
    """
    Evaluate f' at s, averaging left/right limits.

    For C^1+ splines this just returns f'(s). For tent (C^0), it averages
    the two piecewise-constant limits, recovering a sensible derivative value.
    """
    fp = f.derivative()
    return 0.5 * (float(fp(s - eps)) + float(fp(s + eps)))


def _hermite_from_bspline(f: BSpline, s_left: float, s_right: float) -> tuple[float, ...]:
    L = s_right - s_left
    eps = 1e-6 * L
    return (
        float(f(s_left)),
        _safe_derivative_at(f, s_left, eps) * L,
        float(f(s_right)),
        _safe_derivative_at(f, s_right, eps) * L,
        float(f.integrate(s_left, s_right)) / L,
    )


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
        ``[basis_order + 1, dof_ceiling]`` range) -- a reasonable middle
        ground, not tuned to any particular fit-quality target.
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
    kernel :
        Longitudinal B-spline basis: ``"tent"``, ``"quadratic"``, or ``"cubic"``
        (default). See **Kernel choice** below.
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
        assumed y-parity.
    field_tol :
        Relative tolerance for marking a field component as ``to_fit`` in
        ``df_fit_pars`` (same logic as ``FieldFitter``).

    Kernel choice
    -------------
    ``"tent"``:
        Narrowest basis, 2 frames wide. Best preserves sharp features in the
        data, but derivatives are discontinuous at frames (the conversion
        averages left/right limits to recover sensible values). Use when you
        care about peak values and the field has sub-frame structure.
    ``"quadratic"``:
        3 frames wide, C^1 continuous. Good balance: narrower than cubic so
        finer features survive, but derivatives are continuous so no numerical
        tricks needed. Recommended general-purpose choice when ``"cubic"``
        over-smooths your data.
    ``"cubic"`` (default):
        4 frames wide, C^2 continuous. Smoothest output, most averaging of fine
        features. Best when your data is already smooth and you want maximally
        well-behaved derivatives.
    """

    def __init__(
        self,
        raw_data: pd.DataFrame,
        n_frames: int | None = None,
        distance_unit: float = 1e-3,
        deg: int = 2,
        kernel: str = "cubic",
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
        if kernel not in KERNEL_TO_DEGREE:
            valid = ", ".join(sorted(KERNEL_TO_DEGREE))
            raise ValueError(f"kernel must be one of {{{valid}}}, got {kernel!r}")

        self.kernel = kernel
        self.basis_order = KERNEL_TO_DEGREE[kernel]

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
        self.knots: np.ndarray | None = None
        self.n_regions: int | None = None
        self.pq_pairs: list[tuple[int, int]] | None = None
        self.pq_to_idx: dict[tuple[int, int], int] | None = None

        self.s_full: np.ndarray | None = None
        self.Psi: np.ndarray | None = None
        self.Psi_bs: np.ndarray | None = None
        self._hermite: dict[tuple[str, int, int], tuple[float, ...]] | None = None

        self.df_raw_data: pd.DataFrame | None = None
        self.df_on_axis_raw: pd.DataFrame | None = None
        self.df_on_axis_fit: pd.DataFrame | None = None
        self.df_fit_pars: pd.DataFrame | None = None
        self.n_frames_search_trace: dict[int, tuple[float, float]] | None = None

        self._set_raw_data(raw_data)

        if n_frames is not None:
            resolved_n_frames = int(n_frames)
        elif residual_tol is not None:
            resolved_n_frames = self._search_n_frames(float(residual_tol))
        else:
            n_min = self.basis_order + 1
            n_max = self._dof_ceiling()
            resolved_n_frames = int(np.clip(DEFAULT_N_FRAMES, n_min, n_max))
            print(
                f"[TubeFitter] Neither n_frames nor residual_tol given -- "
                f"defaulting to n_frames={resolved_n_frames}."
            )

        if resolved_n_frames < 2:
            raise ValueError("n_frames must be at least 2")
        if resolved_n_frames < self.basis_order + 1:
            raise ValueError(
                f"n_frames ({resolved_n_frames}) must be >= basis_order + 1 "
                f"({self.basis_order + 1}) for kernel={kernel!r}"
            )
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

        self.df_raw_data = raw_data.copy()
        idx = self.df_raw_data.index
        if list(idx.names) != list(_INDEX_NAMES):
            raise ValueError(f"raw_data index must be {_INDEX_NAMES}, got {idx.names}")

        self.df_raw_data.index = pd.MultiIndex.from_arrays(
            [idx.get_level_values(lvl).astype(float) * self.distance_unit for lvl in idx.names],
            names=idx.names,
        )
        self.s_full = np.sort(self.df_raw_data.index.get_level_values("Z").unique()).astype(float)

    # ------------------------------------------------------------------
    # n_frames selection (residual_tol search)
    # ------------------------------------------------------------------

    def _dof_ceiling(self) -> int:
        """
        Max usable n_frames: the tube system has 2*N_x*N_y*N_z equations (Bx,
        By at every grid point) and n_frames*n_pq unknowns (n_pq = size of
        the (p, q) multipole basis for this fitter's deg/y_symmetry/fit_skew).
        Beyond this the system is underdetermined regardless of how much z
        data exists.
        """
        assert self.df_raw_data is not None
        idx = self.df_raw_data.index
        n_x = idx.get_level_values("X").nunique()
        n_y = idx.get_level_values("Y").nunique()
        n_z = idx.get_level_values("Z").nunique()
        n_pq = len(_generate_pq_pairs(self.M, self.y_symmetry, self.fit_skew))
        return (2 * n_x * n_y * n_z) // n_pq

    def _trial_relative_residual(self, n_frames: int) -> tuple[float, float]:
        """Fit a throwaway TubeFitter at n_frames, return its (Bskew, Bnorm)
        relative tube-fit residuals."""
        assert self.df_raw_data is not None
        with contextlib.redirect_stdout(io.StringIO()):
            trial = TubeFitter(
                raw_data=self.df_raw_data,  # already scaled by distance_unit
                n_frames=n_frames,
                distance_unit=1.0,
                deg=self.deg,
                kernel=self.kernel,
                tube_radius=self.tube_radius,
                fit_skew=self.fit_skew,
                y_symmetry=self.y_symmetry,
                field_tol=self.field_tol,
            )
            trial.fit()
        b = trial._b_vec
        rels = {}
        for field, rows in (("Bskew", slice(0, None, 2)), ("Bnorm", slice(1, None, 2))):
            rms = trial.tube_fit_residual_rms[field]
            field_rms = float(np.sqrt(np.mean(b[rows] ** 2)))
            rels[field] = rms / field_rms if field_rms > 0 else 0.0
        return rels["Bskew"], rels["Bnorm"]

    def _search_n_frames(self, residual_tol: float) -> int:
        """
        Smallest n_frames whose worst-case relative residual is <=
        residual_tol, found via geometric-doubling bracket + bisection (see
        examples/splineboris/003d_tube_fitter_nframes_scan.py, folded in here).
        Records every evaluated point in ``self.n_frames_search_trace``.
        """
        n_min = self.basis_order + 1
        n_max = self._dof_ceiling()
        cache: dict[int, tuple[float, float]] = {}

        def meets(n: int) -> bool:
            if n not in cache:
                cache[n] = self._trial_relative_residual(n)
                rel_bskew, rel_bnorm = cache[n]
                print(f"[TubeFitter]   n_frames={n:5d}  Bskew={rel_bskew * 100:6.2f}%  Bnorm={rel_bnorm * 100:6.2f}%")
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
                f"{max(cache[n_max]) * 100:.2f}%). This is a genuine residual "
                f"floor, not a frame-count problem -- falling back to "
                f"n_frames={n_max} (the best achievable). Consider raising "
                f"residual_tol, providing more/denser data, or checking whether "
                f"y_symmetry matches the data."
            )
            self.plot_n_frames_search(target=residual_tol, selected=n_max)
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
        return hi

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
        assert self.knots is not None and self.Psi is not None and self.s_full is not None
        assert self.pq_to_idx is not None
        k = self.basis_order
        if field == "By":
            if (der, 1) not in self.pq_to_idx:
                return None
            f = BSpline(self.knots, -math.factorial(der) * self.Psi[:, der, 1], k)
        elif field == "Bx":
            m = der + 1
            if (m, 0) not in self.pq_to_idx:
                return None
            f = BSpline(self.knots, -math.factorial(m) * self.Psi[:, m, 0], k)
        else:
            raise ValueError(f"field must be 'Bx' or 'By', got {field!r}")
        return f(self.s_full)

    def evaluate_transverse_field(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the fitted (Bx, By) field at arbitrary (x, y, z) points from
        the fitted multipole potential. Call after ``fit()``.

        x, y, z are broadcast together (each may be a scalar or an array of
        matching shape) and must already be in metres, i.e. pre-scaled the
        same way ``raw_data`` is scaled by ``distance_unit`` in ``fit()``.
        """
        if self.Psi is None or self.knots is None or self.pq_pairs is None:
            raise RuntimeError("Call fit() before evaluate_transverse_field().")

        x, y, z = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
        )
        k = self.basis_order
        design = BSpline.design_matrix(z, self.knots, k)

        bx = np.zeros(z.shape, dtype=float)
        by = np.zeros(z.shape, dtype=float)
        for p, q in self.pq_pairs:
            psi_pq_z = design @ self.Psi[:, p, q]
            if p > 0:
                bx += -psi_pq_z * p * x ** (p - 1) * y ** q
            if q > 0:
                by += -psi_pq_z * q * x ** p * y ** (q - 1)
        return bx, by

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
        k = self.basis_order
        # Clamped knots for n_frames B-spline coefficients: len(t) = n_frames + k + 1,
        # with (k+1) repeats at each end and (n_frames - k - 1) interior knots.
        if self.n_frames > k + 1:
            interior = np.linspace(z_min, z_max, self.n_frames - k + 1)[1:-1]
        else:
            interior = np.array([], dtype=float)
        self.knots = np.r_[
            np.full(k + 1, z_min),
            interior,
            np.full(k + 1, z_max),
        ]
        if len(self.knots) != self.n_frames + k + 1:
            raise RuntimeError(
                f"Unexpected knot vector length {len(self.knots)}, "
                f"expected {self.n_frames + k + 1}"
            )
        self.pq_pairs = _generate_pq_pairs(self.M, self.y_symmetry, self.fit_skew)
        self.pq_to_idx = {pq: i for i, pq in enumerate(self.pq_pairs)}

    # ------------------------------------------------------------------
    # Tube fit (Bx, By)
    # ------------------------------------------------------------------

    def _build_linear_system(self) -> None:
        assert self.df_raw_data is not None
        assert self.knots is not None
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
        n_pq = len(self.pq_pairs)
        k = self.basis_order
        n_cols = self.n_frames * n_pq
        n_rows = 2 * n_pts

        A = lil_matrix((n_rows, n_cols), dtype=float)
        b_vec = np.zeros(n_rows, dtype=float)

        x_pow = [np.ones(n_pts, dtype=float)]
        y_pow = [np.ones(n_pts, dtype=float)]
        for _ in range(1, self.M + 1):
            x_pow.append(x_pow[-1] * x)
            y_pow.append(y_pow[-1] * y)

        for i_pt in range(n_pts):
            row_bx = 2 * i_pt
            row_by = row_bx + 1
            b_vec[row_bx] = bx[i_pt]
            b_vec[row_by] = by[i_pt]

            design = BSpline.design_matrix(
                np.array([z[i_pt]]), self.knots, k
            ).toarray()[0]

            for j in np.nonzero(design)[0]:
                beta_j = design[j]
                col_base = j * n_pq
                for pq_idx, (p, q) in enumerate(self.pq_pairs):
                    col = col_base + pq_idx
                    if p > 0:
                        A[row_bx, col] = -p * x_pow[p - 1][i_pt] * y_pow[q][i_pt] * beta_j
                    if q > 0:
                        A[row_by, col] = -q * x_pow[p][i_pt] * y_pow[q - 1][i_pt] * beta_j

        self._A_csr = A.tocsr()
        self._b_vec = b_vec

    def _solve(self) -> None:
        result = lsmr(
            self._A_csr,
            self._b_vec,
            atol=1e-10,
            btol=1e-10,
            maxiter=10000,
        )
        x_sol = result[0]
        n_pq = len(self.pq_pairs)
        flat = x_sol.reshape(self.n_frames, n_pq)
        self.Psi = np.zeros((self.n_frames, self.M + 1, self.M + 1), dtype=float)
        for pq_idx, (p, q) in enumerate(self.pq_pairs):
            self.Psi[:, p, q] = flat[:, pq_idx]

        self._report_tube_fit_residual(x_sol)

    def _report_tube_fit_residual(self, x_sol: np.ndarray) -> None:
        """
        Report the residual of the tube linear system against the raw Bx/By
        values it was regressed against (der=0). Higher-order multipoles are
        read off other components of this same fitted Psi rather than being
        fit independently, so this single residual is the analogue of
        FieldFitter's per-field der=0 transverse-fit residual.
        """
        residual = self._b_vec - self._A_csr @ x_sol
        self.tube_fit_residual_rms = {}
        # rows alternate Bx (even), By (odd); see _build_linear_system.
        for field, rows in (("Bskew", slice(0, None, 2)), ("Bnorm", slice(1, None, 2))):
            res = residual[rows]
            sig = self._b_vec[rows]
            rms = float(np.sqrt(np.mean(res ** 2)))
            field_rms = float(np.sqrt(np.mean(sig ** 2)))
            rel = rms / field_rms if field_rms > 0 else 0.0
            self.tube_fit_residual_rms[field] = rms
            print(f"[TubeFitter] {field} tube fit residual (der=0): "
                  f"RMS = {rms:.3e} T ({rel * 100:.2f}% of field RMS)")

    def _fit_bs(self) -> None:
        assert self.knots is not None
        assert self.s_full is not None
        assert self.df_on_axis_raw is not None

        bs_on_axis = self.df_on_axis_raw[("Bs", 0)].to_numpy(dtype=float)
        k = self.basis_order
        n_pts = len(self.s_full)
        n_cols = self.n_frames

        A = lil_matrix((n_pts, n_cols), dtype=float)
        for i_pt, z_i in enumerate(self.s_full):
            design = BSpline.design_matrix(np.array([z_i]), self.knots, k).toarray()[0]
            for j in np.nonzero(design)[0]:
                A[i_pt, j] = design[j]

        result = lsmr(A.tocsr(), bs_on_axis, atol=1e-10, btol=1e-10, maxiter=10000)
        self.Psi_bs = result[0]

    # ------------------------------------------------------------------
    # Hermite conversion and output tables
    # ------------------------------------------------------------------

    def _convert_to_hermite(self) -> None:
        assert self.frames is not None
        assert self.knots is not None
        assert self.Psi_bs is not None

        self._hermite = {}
        k = self.basis_order

        for i_reg in range(self.n_regions):
            s_left = float(self.frames[i_reg])
            s_right = float(self.frames[i_reg + 1])

            for der in range(self.deg + 1):
                m = der + 1
                # b_m = -(m-1)! * C_{m-1,1}  ->  Psi[:, der, 1]
                f_norm = BSpline(
                    self.knots, -math.factorial(der) * self.Psi[:, der, 1], k
                )
                self._hermite[("Bnorm", der, i_reg)] = _hermite_from_bspline(
                    f_norm, s_left, s_right
                )

                if (m, 0) in self.pq_to_idx:
                    # a_m = -m! * C_{m,0}  (from d^{m-1} B_x / dx^{m-1} |_{x=y=0})
                    f_skew = BSpline(
                        self.knots, -math.factorial(m) * self.Psi[:, m, 0], k
                    )
                    self._hermite[("Bskew", der, i_reg)] = _hermite_from_bspline(
                        f_skew, s_left, s_right
                    )

            f_bs = BSpline(self.knots, self.Psi_bs, k)
            self._hermite[("Bs", 0, i_reg)] = _hermite_from_bspline(f_bs, s_left, s_right)

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
                pars = [f"{prefix}_{s}" for s in xt.SplineBoris._HERMITE_SUFFIXES]

                for i_reg in range(self.n_regions):
                    idx_start = int(idx_extrema[i_reg])
                    idx_end = int(idx_extrema[i_reg + 1])
                    s_start = float(self.s_full[idx_start])
                    s_end = float(self.s_full[idx_end])
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
        assert self.knots is not None
        assert self.Psi is not None
        assert self.Psi_bs is not None
        assert self.s_full is not None
        assert self.df_on_axis_fit is not None

        k = self.basis_order
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
            f_bs = BSpline(self.knots, self.Psi_bs, k)
            self.df_on_axis_fit[("Bs", 0)] = f_bs(self.s_full)
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
                s_borders = np.unique(np.concatenate((s_start_vals, s_end_vals)))
                s_arr = np.asarray(s)
                return sorted({int(np.argmin(np.abs(s_arr - float(sb)))) for sb in s_borders})
            except Exception:
                return []

        for field_ax, ax, fc in [("Bx", ax1, "Bskew"), ("By", ax2, "Bnorm"), ("Bs", ax3, "Bs")]:
            for idx in _borders_for_field(fc):
                if 0 <= idx < len(s):
                    ax.axvline(x=s[idx], color="k", linestyle="--", linewidth=1, alpha=0.3)

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


def _constant_dipole_sign_check(kernel: str) -> None:
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

    fitter = TubeFitter(df, n_frames=8, distance_unit=1.0, deg=2, kernel=kernel)
    fitter.fit()

    sub = fitter.df_fit_pars.loc[("Bnorm", 0)].reset_index()
    c1 = sub.loc[sub["param_index"] == 0, "param_value"].iloc[0]
    c3 = sub.loc[sub["param_index"] == 2, "param_value"].iloc[0]
    if not (np.isclose(c1, B0, rtol=1e-2) and np.isclose(c3, B0, rtol=1e-2)):
        raise AssertionError(
            f"Constant dipole sign check failed (kernel={kernel!r}): "
            f"expected val_start/val_end ~ {B0}, got c1={c1}, c3={c3}"
        )
    print(
        f"Constant dipole sign check passed (kernel={kernel}): "
        f"Bnorm_0 val_start={c1:.6f}, val_end={c3:.6f} (B0={B0})"
    )


if __name__ == "__main__":
    for kernel in KERNEL_TO_DEGREE:
        _constant_dipole_sign_check(kernel)

    dz = 0.001
    file_path = Path(__file__).resolve().parents[2] / "test_data" / "sls" / "simona_field_map.txt"
    df_raw = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["X", "Y", "Z", "Bx", "By", "Bs"],
    ).set_index(["X", "Y", "Z"])

    deg = 2
    for kernel in ("tent",):#, "quadratic", "cubic"):
        fitter = TubeFitter(
            raw_data=df_raw,
            n_frames=550,
            distance_unit=dz,
            deg=deg,
            kernel=kernel,
            tube_radius=0.001,
        )
        print(f"\n=== Fitting with kernel={kernel} ===")
        fitter.fit()
        print(
            f"Fit complete: {fitter.n_regions} regions, "
            f"{len(fitter.pq_pairs)} (p,q) pairs per frame"
        )
        for der in range(deg + 1):
            fitter.plot_fields(der=der)
