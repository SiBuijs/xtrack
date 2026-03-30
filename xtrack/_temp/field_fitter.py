from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy as sc
import math

from scipy.signal import find_peaks

import xtrack as xt


@dataclass
class FitSegment:
    """One longitudinal Hermite piece for a given component/derivative."""

    field_component: str  # "Bx", "By", "Bs"
    derivative_x: int
    region_index: int  # canonical region id shared across components
    s_start: float
    s_end: float
    idx_start: int
    idx_end: int
    hermite_params: np.ndarray  # (f_left, df_left, f_right, df_right, average)
    to_fit: bool


@dataclass
class FieldFitResult:
    """Canonical, pandas-free representation of a full field fit."""

    s_full: np.ndarray
    segments: List[FitSegment]
    multipole_order: int  # max derivative_x across By/Bx plus 1


def _hermite_csv_param_names(field_component, derivative_x):
    """Labels for the five Hermite scalars in ``df_fit_pars`` (CSV ``param_name`` column).

    Field-map columns map to :class:`xtrack.SplineBoris` inputs as ``Bx`` → ``Bskew``,
    ``By`` → ``Bnorm``, ``Bs`` → ``Bs`` (same naming family as :meth:`SplineBoris._get_param_names`
    but with Hermite suffixes instead of polynomial indices).
    """
    if field_component not in ("Bx", "By", "Bs"):
        raise ValueError(
            "field_component must be 'Bx', 'By', or 'Bs', "
            f"got {field_component!r}"
        )
    if field_component == "Bs":
        prefix = "Bs"
    elif field_component == "By":
        prefix = f"Bnorm_{derivative_x}"
    else:
        prefix = f"Bskew_{derivative_x}"
    return [f"{prefix}_{suf}" for suf in xt.SplineBoris._HERMITE_SUFFIXES]


class FieldFitter:
    '''
    Fit on-axis field data and transverse derivatives using piecewise polynomials.

    The fitting pipeline extracts on-axis data, identifies longitudinal regions,
    and fits per-region polynomials to produce spline parameters stored in
    ``df_fit_pars``.

    Parameters
    ----------
    raw_data :
        A numpy-based field grid, e.g. a dict with 1D arrays
        ``{'x', 'y', 'z', 'Bx', 'By', 'Bs'}`` of equal length.
    xy_point :
        On-axis transverse point ``(X, Y)`` in meters, used to select the
        longitudinal series for fitting. Because the input coordinates are
        multiplied by ``distance_unit`` at import time, ``xy_point`` must be
        given in post-scaling (meter) coordinates.
    distance_unit :
        Coordinate scale factor applied to the X, Y, and Z index levels of
        the input data to convert them to meters.  For example, set
        ``distance_unit=0.001`` when the input coordinates are in millimetres.
    min_region_size :
        Minimum number of points per longitudinal fitting region.
        Ignored when ``n_pieces`` is set.
    deg :
        Maximum transverse derivative order to compute and fit.
    n_pieces : int, optional
        If set, override automatic region detection and use this many
        equally-spaced polynomial pieces for all field components.
        The ``min_region_size`` parameter is ignored in this case.

    Notes
    -----
    All internal storage is numpy-based; no pandas dependency is required.
    '''

    def __init__(
            self,
            raw_data,
            xy_point=(0, 0),
            distance_unit=0.001,
            min_region_size=10,
            deg=2,
            field_tol=1e-3,
            n_pieces=None,
    ):
        # Parameters
        self.xy_point = xy_point
        self.distance_unit = distance_unit
        self.poly_order = 4  # fixed at 4 for now (5 coefficients)
        self.min_region_size = min_region_size
        self.s_full = None
        self.length = None
        self.deg = deg
        self.field_tol = field_tol
        self.n_pieces = n_pieces

        # Raw field grid (flattened arrays)
        self._x = None
        self._y = None
        self._z = None
        self._Bx = None
        self._By = None
        self._Bs = None

        # On-axis raw and fitted series: (field, der) -> np.ndarray over s_full
        self._on_axis_raw: Dict[Tuple[str, int], np.ndarray] = {}
        self._on_axis_fit: Dict[Tuple[str, int], np.ndarray] = {}
        # Numpy-native view of the final fit, filled during region setup and fitting.
        self._fit_segments: List[FitSegment] = []
        # Map (field, der, region_name, s_start, s_end, idx_start, idx_end) -> index in _fit_segments
        self._segment_index = {}
        self._set_raw_data(raw_data)

    # PUBLIC
    # Method that calls all the other methods to arrive at a fit.
    def fit(self) -> FieldFitResult:
        if self._x is None:
            raise RuntimeError("Raw data must be provided before calling fit().")

        # Reset any previous fit so that fit() is idempotent.
        self._on_axis_raw = {}
        self._on_axis_fit = {}
        self._fit_segments = []
        self._segment_index = {}

        self._set_df_on_axis()
        self._find_regions()
        self._fit_slices()

        if not self._fit_segments:
            raise RuntimeError("No fit segments were produced by FieldFitter.")

        # Multipole order is driven by transverse derivatives (Bx, By).
        max_der = -1
        for seg in self._fit_segments:
            if seg.field_component in ("Bx", "By"):
                if seg.derivative_x > max_der:
                    max_der = seg.derivative_x
        multipole_order = max_der + 1 if max_der >= 0 else 1

        # Renumber region_index canonically by unique geometry to keep grouping stable.
        geom_to_region = {}
        next_region = 0
        for seg in self._fit_segments:
            key = (seg.s_start, seg.s_end, seg.idx_start, seg.idx_end)
            if key not in geom_to_region:
                geom_to_region[key] = next_region
                next_region += 1
            seg.region_index = geom_to_region[key]

        return FieldFitResult(
            s_full=np.asarray(self.s_full, dtype=float).copy(),
            segments=list(self._fit_segments),
            multipole_order=multipole_order,
        )



    @staticmethod
    def _poly(s0, s1, coeffs):
        """
        Build a 4th-order polynomial over [s0, s1] from Hermite parameters.

        Convenience wrapper around ``xt.SplineBoris.hermite_to_polynomial``.
        """
        return xt.SplineBoris.hermite_to_polynomial(s0, s1, coeffs)



    # PRIVATE
    # This method stores raw data and extracts the global s grid.
    def _set_raw_data(self, raw_data):
        """
        Set the raw data arrays, scale coordinates to meters, and compute ``s_full``.
        """

        try:
            x = np.asarray(raw_data["x"], dtype=float)
            y = np.asarray(raw_data["y"], dtype=float)
            z = np.asarray(raw_data["z"], dtype=float)
            Bx = np.asarray(raw_data["Bx"], dtype=float)
            By = np.asarray(raw_data["By"], dtype=float)
            Bs = np.asarray(raw_data["Bs"], dtype=float)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                "raw_data must be a mapping with keys 'x', 'y', 'z', 'Bx', 'By', 'Bs' "
                f"containing 1D numpy-compatible arrays; got {type(raw_data).__name__}"
            ) from exc

        if not (x.shape == y.shape == z.shape == Bx.shape == By.shape == Bs.shape):
            raise ValueError("All raw_data arrays must have the same shape.")

        # Apply distance scaling to coordinates (field values are left untouched).
        self._x = x.astype(float) * float(self.distance_unit)
        self._y = y.astype(float) * float(self.distance_unit)
        self._z = z.astype(float) * float(self.distance_unit)
        self._Bx = Bx.astype(float)
        self._By = By.astype(float)
        self._Bs = Bs.astype(float)

        # Build the global s grid from points on the chosen transverse axis.
        x0, y0 = self.xy_point
        mask_axis = np.isclose(self._x, x0) & np.isclose(self._y, y0)
        if not np.any(mask_axis):
            raise ValueError(
                f"No data points found on axis at xy_point={self.xy_point!r}; "
                "check the raw grid coordinates and distance_unit."
            )
        s_on_axis = self._z[mask_axis]
        self.s_full = np.sort(np.unique(s_on_axis))
        if self.s_full.size < 2:
            raise ValueError("Need at least two distinct s samples on axis to build a fit.")
        self.length = float(self.s_full[-1] - self.s_full[0])

    # PRIVATE
    # This method extracts on-axis data into numpy arrays and fits transverse polynomials.
    def _set_df_on_axis(self):
        """
        Extract on-axis data and compute transverse derivatives.

        On-axis data and all transverse derivatives are stored in
        ``self._on_axis_raw[(field, der)]`` and corresponding fit values in
        ``self._on_axis_fit[(field, der)]``.
        """

        if self.s_full is None:
            raise RuntimeError("s_full must be set before extracting on-axis data.")

        x0, y0 = self.xy_point
        n_s = self.s_full.size

        # Base on-axis series for Bx, By, Bs (derivative 0).
        for field, arr in (("Bx", self._Bx), ("By", self._By), ("Bs", self._Bs)):
            vals = np.zeros(n_s, dtype=float)
            for i_s, s in enumerate(self.s_full):
                mask = np.isclose(self._x, x0) & np.isclose(self._y, y0) & np.isclose(self._z, s)
                field_vals = arr[mask]
                if field_vals.size == 0:
                    vals[i_s] = 0.0
                else:
                    vals[i_s] = float(np.mean(field_vals))
            self._on_axis_raw[(field, 0)] = vals
            self._on_axis_fit[(field, 0)] = np.zeros_like(vals)

        # Compute transverse derivatives (Bx, By only).
        self._fit_transverse_polynomials()

    # PRIVATE
    # This method loops over all fields and derivatives.
    # It first checks if a field/derivative needs fitting based on its maximum value compared to the maximum of the main field.
    # It finds peaks and valleys in the data within the peak_window, with specified width and prominence.
    # It uses these extrema to define regions for polynomial fitting.
    # Then, it cuts regions if they span too wide a range.
    # Finally, it stores the regions in the df_fit_pars DataFrame.
    def _find_regions(self):
        """
        Identify regions for polynomial fitting.

        This method first checks if a field/derivative needs fitting based on its maximum value compared to the maximum of the main field.
        It finds peaks and valleys in the data within the peak_window, with specified width and prominence.
        It uses these extrema to define regions for polynomial fitting.
        Then, it cuts regions if they span too wide a range.
        Finally, it stores the regions in the df_fit_pars DataFrame.
        """

        fields = ["Bx", "By", "Bs"]

        abs_max = 0.0
        for field in fields:
            series = self._on_axis_raw.get((field, 0))
            if series is None:
                continue
            field_max = float(np.max(np.abs(series)))
            if field_max > abs_max:
                abs_max = field_max

        # Estimate transverse scale from the raw x-array.
        x_max = float(np.max(np.abs(self._x)))

        n_data = len(self.s_full)

        for field in fields:
            # Bs only has der = 0; other fields range 0..deg
            ders = [0] if field == "Bs" else range(0, self.deg + 1)

            for der in ders:
                series = self._on_axis_raw.get((field, der))
                if series is None:
                    continue

                # FIELD TOLERANCE AREA: check if this field/derivative needs fitting
                field_der_max = np.max(np.abs(series))
                relative_max = 1 / math.factorial(der) * field_der_max * (x_max ** der)
                if relative_max < self.field_tol * abs_max:
                    # set to single region with zero parameters and skip expensive processing
                    field_extrema = np.array([0, len(series) - 1], dtype=int)
                    to_fit = False
                elif self.n_pieces is not None:
                    # Manual override: equally-spaced regions
                    field_extrema = np.round(
                        np.linspace(0, n_data - 1, self.n_pieces + 1)
                    ).astype(int)
                    to_fit = True
                else:
                    # SPLIT REGIONS AREA
                    # choose prominence: more permissive for Bs
                    std_series = np.std(series)
                    prominence = 0.5 * std_series if field == "Bs" else std_series

                    field_peaks = find_peaks(series, width=15, prominence=prominence)[0]
                    field_valleys = find_peaks(-series, width=15, prominence=prominence)[0]
                    field_extrema = np.sort(np.concatenate((field_peaks, field_valleys)))

                    # include endpoints
                    field_extrema = np.insert(field_extrema, 0, 0)
                    field_extrema = np.append(field_extrema, len(series) - 1)

                    # split long regions while ensuring each part has at least `min_region_size` points
                    this_min_region_size = self.min_region_size
                    new_extrema = [int(field_extrema[0])]
                    for left, right in zip(field_extrema[:-1], field_extrema[1:]):
                        length = int(right - left)
                        if length < 2 * this_min_region_size:
                            new_extrema.append(int(right))
                            continue
                        n_parts = int(np.floor(length / this_min_region_size))
                        if n_parts <= 1:
                            new_extrema.append(int(right))
                            continue
                        splits = np.round(np.linspace(left, right, n_parts + 1)).astype(int)
                        for s_split in splits[1:]:
                            if s_split > new_extrema[-1]:
                                new_extrema.append(int(s_split))

                    field_extrema = np.unique(np.asarray(new_extrema, dtype=int))
                    to_fit = True

                # number of pieces is number of extrema - 1 (ensure at least 1)
                actual_n_pieces = max(1, len(field_extrema) - 1)
                print(f"{field} der={der} -> n_pieces={actual_n_pieces}")
                self._set_df_fit_pars(der, actual_n_pieces, field, field_extrema, to_fit)

    # PRIVATE
    # This method initializes and appends rows to the df_fit_pars DataFrame.
    # Each row corresponds to a polynomial piece for a specific field and derivative.
    # It stores metadata about the piece, including parameter names and initial values.
    # This method is called by _find_regions to populate the DataFrame.
    # In case the set consists of only one piece, the parameters are initialized to 0.
    def _set_df_fit_pars(self, der_order, n_pieces, field, idx_extrema, to_fit=True):
        """
        Initialize and append rows to the df_fit_pars DataFrame.

        Each row corresponds to a polynomial piece for a specific field and derivative.
        It stores metadata about the piece, including parameter names and initial values.
        This method is called by _find_regions to populate the DataFrame.
        In case the set consists of only one piece, the parameters are initialized to 0.
        """

        # Zero-pad index so alphabetical sort matches numerical sort
        index_width = len(str(n_pieces - 1)) if n_pieces > 1 else 1
        for i in range(n_pieces):
            idx_start = idx_extrema[i]
            idx_end = idx_extrema[i+1]
            s_start = self.s_full[idx_start]
            s_end = self.s_full[idx_end]
            region_name = f"Poly_{i:0{index_width}d}"

            # Create or register a numpy-native FitSegment placeholder for this piece.
            key = (field, int(der_order), region_name, float(s_start), float(s_end), int(idx_start), int(idx_end))
            if key not in self._segment_index:
                seg_idx = len(self._fit_segments)
                self._segment_index[key] = seg_idx
                self._fit_segments.append(
                    FitSegment(
                        field_component=str(field),
                        derivative_x=int(der_order),
                        region_index=seg_idx,  # temporary; will be renumbered in fit()
                        s_start=float(s_start),
                        s_end=float(s_end),
                        idx_start=int(idx_start),
                        idx_end=int(idx_end),
                        hermite_params=np.zeros(len(xt.SplineBoris._HERMITE_SUFFIXES), dtype=float),
                        to_fit=bool(to_fit),
                    )
                )
            # No DataFrame construction; segments are tracked in self._fit_segments.


    
    def _boundary_from_poly(self, sL, poly):
        """
        Compute the boundary conditions from a previously fitted polynomial.

        Because the fitting is done from left to right, this method is used to compute the boundary conditions from the previous polynomial.
        Accepts the polynomial and the position sL where to evaluate it (we fit from left to right, so always the leftmost point).
        """

        dp = poly.deriv()
        return np.array([poly(sL), dp(sL)], dtype=float)

    def _boundary_from_finite_differences(self, b_region, s_region, get_right_point=True):
        """
        Compute the boundary conditions from finite differences in the specified region.

        Because the fitting is done from left to right, this method is used to compute the boundary conditions from the data on the rightmost point of the region.
        Accepts the field values in the region and the longitudinal spacing.
        """

        if get_right_point:
            h = s_region[-1] - s_region[-2]
            dbR = (3 * b_region[-1] - 4 * b_region[-2] + b_region[-3]) / (2 * h)
            return np.array([b_region[-1], dbR], dtype=float)
        else:
            h = s_region[1] - s_region[0]
            dbL = (-3 * b_region[0] + 4 * b_region[1] - b_region[2]) / (2 * h)
            return np.array([b_region[0], dbL], dtype=float)



    def _fit_single_poly(self, field, der_order, segment: FitSegment, prev_segment: FitSegment | None = None):
        """
        Fit a single polynomial piece and store its Hermite parameters.

        The five stored parameters are
        ``(f_left, df_left, f_right, df_right, average)``.
        Continuity is enforced by reading ``f_right`` / ``df_right`` from the
        previous piece when available, avoiding polynomial reconstruction.
        """

        idx_left = int(segment.idx_start)
        idx_right = int(segment.idx_end)
        s_left = float(segment.s_start)
        s_right = float(segment.s_end)

        s_region = self.s_full[idx_left:idx_right + 1]
        b_region = self._on_axis_raw[(field, der_order)][idx_left:idx_right + 1]
        L = s_right - s_left
        average = sc.integrate.trapezoid(b_region, s_region) / L

        if prev_segment is not None and prev_segment.to_fit and np.any(prev_segment.hermite_params):
            prev_vals = prev_segment.hermite_params
            left_bounds = np.array(
                [float(prev_vals[2]), float(prev_vals[3])],  # f_right, df_right of previous
                dtype=float,
            )
        else:
            left_bounds = self._boundary_from_finite_differences(b_region, s_region, get_right_point=False)

        right_bounds = self._boundary_from_finite_differences(b_region, s_region, get_right_point=True)
        hermite_params = (left_bounds[0], left_bounds[1],
                          right_bounds[0], right_bounds[1], average)

        poly = self._poly(s_left, s_right, hermite_params)

        # Store Hermite parameters on the segment.
        segment.hermite_params = np.asarray(hermite_params, dtype=float)

        # Update fitted on-axis series.
        idx_slice = slice(idx_left, idx_right + 1)
        self._on_axis_fit[(field, der_order)][idx_slice] = poly(s_region - s_left)

    # PRIVATE
    # This method loops over all fields and derivatives and fits polynomials to each region.
    def _fit_slices(self):
        """
        Fit polynomials to each region for all fields and derivatives.

        This method loops over all fields and derivatives and fits polynomials to each region.
        It skips the derivatives of Bs and regions that do not need fitting.
        """

        for field in ["Bx", "By", "Bs"]:
            for der in range(0, self.deg + 1):
                if field == "Bs" and der > 0:
                    continue

                print(f"Fitting field {field} derivative {der}")
                # Collect segments for this (field, der) and sort by s_start.
                segs = [
                    seg
                    for seg in self._fit_segments
                    if seg.field_component == field and seg.derivative_x == int(der)
                ]
                if not segs:
                    continue
                segs.sort(key=lambda s: (s.s_start, s.idx_start))

                prev_seg = None
                for seg in segs:
                    if not seg.to_fit:
                        prev_seg = seg
                        continue
                    self._fit_single_poly(field, der, seg, prev_seg)
                    prev_seg = seg



    def _fit_transverse_polynomials(self):
        """
        Fit transverse polynomials and compute all derivatives at ``self.xy_point``.

        This method fits a polynomial of degree ``self.deg`` to the transverse
        field variation at each longitudinal position, using every X value
        present in the input data at the Y coordinate of ``self.xy_point``.
        It then evaluates all derivatives from order 1 to ``self.deg`` at
        the X coordinate of ``self.xy_point`` and stores them in
        ``df_on_axis_raw``.
        """
        x_point, y_point = self.xy_point

        if self.s_full is None:
            raise RuntimeError("s_full must be set before computing transverse polynomials.")

        n_s = self.s_full.size

        for field, arr in (("Bx", self._Bx), ("By", self._By)):
            # Prepare derivative arrays for this field.
            derivs = {der: np.zeros(n_s, dtype=float) for der in range(1, self.deg + 1)}

            for i_s, s in enumerate(self.s_full):
                # All points at this longitudinal position and y = y_point.
                mask_plane = np.isclose(self._y, y_point) & np.isclose(self._z, s)
                xs = self._x[mask_plane]
                Bs = arr[mask_plane]
                if xs.size == 0:
                    # No data at this s-slice; leave derivatives at zero.
                    continue
                # Fit polynomial of degree self.deg: B(x) ~ poly(x).
                coeffs = np.polyfit(xs, Bs, self.deg)
                for der in range(1, self.deg + 1):
                    d_coeffs = np.polyder(coeffs, m=der)
                    derivs[der][i_s] = float(np.polyval(d_coeffs, x_point))

            for der in range(1, self.deg + 1):
                self._on_axis_raw[(field, der)] = derivs[der]
                self._on_axis_fit[(field, der)] = np.zeros_like(derivs[der])



    def plot_integrated_fields(self):
        """
        Plot the integrated fields for the raw and fit data.

        This method plots the integrated fields for the raw and fit data.
        It accepts the derivative order.
        It computes the derivatives of the polynomials and stores them in the df_on_axis_raw DataFrame.
        """
        import matplotlib.pyplot as plt

        s = self.s_full

        Bx_raw = self._on_axis_raw.get(("Bx", 0), np.zeros_like(s, dtype=float))
        By_raw = self._on_axis_raw.get(("By", 0), np.zeros_like(s, dtype=float))
        Bs_raw = self._on_axis_raw.get(("Bs", 0), np.zeros_like(s, dtype=float))

        Bx_fit = self._on_axis_fit.get(("Bx", 0), np.zeros_like(s, dtype=float))
        By_fit = self._on_axis_fit.get(("By", 0), np.zeros_like(s, dtype=float))
        Bs_fit = self._on_axis_fit.get(("Bs", 0), np.zeros_like(s, dtype=float))

        fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)

        Bx_int_raw = sc.integrate.cumulative_trapezoid(Bx_raw, x=s, initial=0)
        By_int_raw = sc.integrate.cumulative_trapezoid(By_raw, x=s, initial=0)
        Bs_int_raw = sc.integrate.cumulative_trapezoid(Bs_raw, x=s, initial=0)

        Bx_int_fit = sc.integrate.cumulative_trapezoid(Bx_fit, x=s, initial=0)
        By_int_fit = sc.integrate.cumulative_trapezoid(By_fit, x=s, initial=0)
        Bs_int_fit = sc.integrate.cumulative_trapezoid(Bs_fit, x=s, initial=0)

        ax1.plot(s, Bx_int_raw, label='Raw Data')
        ax1.plot(s, Bx_int_fit, label='Fit', linestyle='--')
        ax2.plot(s, By_int_raw, label='Raw Data')
        ax2.plot(s, By_int_fit, label='Fit', linestyle='--')
        ax3.plot(s, Bs_int_raw, label='Raw Data')
        ax3.plot(s, Bs_int_fit, label='Fit', linestyle='--')

        # Vertical border lines removed

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

    def plot_fields(self, der=0):
        """
        Plot the data against the fit.

        This method plots the data against the fit.
        It accepts the derivative order.
        It computes the derivatives of the polynomials and stores them in the df_on_axis_raw DataFrame.
        """
        import matplotlib.pyplot as plt

        s = self.s_full
        fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)

        def get_series_raw(field, der):
            return self._on_axis_raw.get((field, der), np.zeros_like(s, dtype=float))

        def get_series_fit(field, der):
            return self._on_axis_fit.get((field, der), np.zeros_like(s, dtype=float))

        ax1.plot(s, get_series_raw("Bx", der), label='Raw Data')
        ax1.plot(s, get_series_fit("Bx", der), label='Fit', linestyle='--')
        ax2.plot(s, get_series_raw("By", der), label='Raw Data')
        ax2.plot(s, get_series_fit("By", der), label='Fit', linestyle='--')
        ax3.plot(s, get_series_raw("Bs", der), label='Raw Data')
        ax3.plot(s, get_series_fit("Bs", der), label='Fit', linestyle='--')

        # compute border indices per field/derivative from FitSegments
        def _borders_for_field(field_ax):
            idxs = []
            for seg in self._fit_segments:
                if seg.field_component != field_ax or int(seg.derivative_x) != int(der):
                    continue
                for sb in (seg.s_start, seg.s_end):
                    idx = int(np.argmin(np.abs(s - float(sb))))
                    idxs.append(idx)
            return sorted(set(i for i in idxs if 0 <= i < len(s)))

        for field_ax in ["Bx", "By", "Bs"]:
            ax = {"Bx": ax1, "By": ax2, "Bs": ax3}[field_ax]
            borders_idx_field = _borders_for_field(field_ax)
            for idx in borders_idx_field or []:
                if 0 <= idx < len(s):
                    ax.axvline(x=s[idx], color='k', linestyle='--', linewidth=1, alpha=0.3)

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

        ax1.legend([f"{x_label} Data", f"{x_label} Fit"], loc="lower right")
        ax2.legend([f"{y_label} Data", f"{y_label} Fit"], loc="lower right")
        ax3.legend([f"{s_label} Data", f"{s_label} Fit"], loc="upper right")

        ax1.grid()
        ax2.grid()
        ax3.grid()

        plt.show()