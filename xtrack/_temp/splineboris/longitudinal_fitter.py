"""
Longitudinal fit of on-axis field components with a C3 quartic B-spline.

Stage 2 of the SplineBoris field-map pipeline: it takes data points
``(z, f(z))`` for any set of on-axis components, from any source (e.g. the
frames of ``TubeFitter.on_axis_multipoles()`` or the map's own on-axis B_s),
and least-squares fits each with a degree-4 B-spline on uniform nodes
``s_k = s_start + k * Delta``, ``k = 0..E``. The export stores, per element
``[s_k, s_{k+1}]`` and component, the 5 numbers of ``Spline4``:
``(f(s_k), f'(s_k), f(s_{k+1}), f'(s_{k+1}), mean_k)``.

Component names:
    ("By", n) = d^n B_y / dx^n at (0, 0, s)
    ("Bx", n) = d^n B_x / dx^n at (0, 0, s)
    ("Bs", 0) = B_s(0, 0, s)

End conditions:
    "zero" -- f = f' = f'' = f''' = 0 at s_start and s_end, imposed by
              dropping the first and last 4 basis functions. Requires the
              field to be negligible at both ends.
    "free" -- all E + 4 basis functions are kept.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import scipy as sc
import xtrack as xt

from xtrack.beam_elements.splineboris import Spline4, SplineBoris

_DEG = 4


class LongitudinalFitter:
    """
    Fit on-axis field components with a quartic B-spline on uniform nodes.

    Parameters
    ----------
    s_start, s_end :
        Range covered by the elements [m].
    n_elements :
        Number of elements E. If None, it is set on the first ``fit()`` call
        to ``round(n_data / points_per_element)``.
    points_per_element :
        Used only when ``n_elements`` is None. More elements give higher
        accuracy on clean data (error ~ Delta^5) but amplify noise in the
        s-derivatives the field evaluator uses (up to 4th order).
    end_condition :
        ``"zero"`` (default) or ``"free"``, see the module docstring.
    preserve_integral :
        If True, constrain each component's total integral to the
        trapezoid integral of its data.
    period :
        Optional undulator period [m]; warns if there are fewer than 12
        elements per period.
    """

    def __init__(
        self,
        s_start: float,
        s_end: float,
        n_elements: int | None = None,
        points_per_element: float = 5,
        end_condition: str = "zero",
        preserve_integral: bool = False,
        period: float | None = None,
    ):
        if end_condition not in ("zero", "free"):
            raise ValueError(f"end_condition must be 'zero' or 'free', got {end_condition!r}")
        if not s_end > s_start:
            raise ValueError("s_end must be larger than s_start")
        self.s_start = float(s_start)
        self.s_end = float(s_end)
        self.points_per_element = points_per_element
        self.end_condition = end_condition
        self.preserve_integral = bool(preserve_integral)
        self.period = period
        self.splines: dict[tuple[str, int], sc.interpolate.BSpline] = {}
        self.data: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._data_spacing = np.inf
        self.n_elements = None
        if n_elements is not None:
            self._set_grid(int(n_elements))

    def _set_grid(self, n_elements: int) -> None:
        min_elements = 5 if self.end_condition == "zero" else 1
        if n_elements < min_elements:
            raise ValueError(
                f"n_elements must be at least {min_elements} for "
                f"end_condition={self.end_condition!r}, got {n_elements}"
            )
        self.n_elements = n_elements
        self.nodes = np.linspace(self.s_start, self.s_end, n_elements + 1)
        self.delta = (self.s_end - self.s_start) / n_elements
        self.knots = np.r_[[self.s_start] * _DEG, self.nodes, [self.s_end] * _DEG]
        if self.period is not None and self.period / self.delta < 12:
            warnings.warn(
                f"Only {self.period / self.delta:.1f} elements per period "
                f"(fewer than 12); increase n_elements."
            )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, z, F, names) -> None:
        """Fit the components ``names`` to data ``F`` (``(n_data,)`` or
        ``(n_data, n_components)``) at positions ``z``. All components in one
        call share ``z`` and a single factorisation; components with other
        positions (e.g. B_s) are fitted in a separate call."""
        z = np.asarray(z, dtype=float)
        F = np.asarray(F, dtype=float).reshape(len(z), -1)
        names = [names] if isinstance(names, tuple) else list(names)
        if F.shape[1] != len(names):
            raise ValueError(f"F has {F.shape[1]} columns but {len(names)} names were given")

        outside = (z < self.s_start) | (z > self.s_end)
        if outside.any():
            raise ValueError(
                f"{int(outside.sum())} data points lie outside "
                f"[s_start, s_end] = [{self.s_start}, {self.s_end}]; drop them first."
            )
        order = np.argsort(z, kind="stable")
        z, F = z[order], F[order]

        if self.n_elements is None:
            self._set_grid(max(1, round(len(z) / self.points_per_element)))
        self._check_data_per_element(z)
        self._data_spacing = min(self._data_spacing, float(np.median(np.diff(z))))

        K = sc.interpolate.BSpline.design_matrix(z, self.knots, _DEG).tocsc()
        n_basis = K.shape[1]
        kept = slice(_DEG, n_basis - _DEG) if self.end_condition == "zero" else slice(0, n_basis)
        K = K[:, kept]

        M = (K.T @ K).todia()
        ab = np.zeros((_DEG + 1, K.shape[1]))
        for k in range(_DEG + 1):
            ab[_DEG - k, k:] = M.diagonal(k)
        try:
            cb = sc.linalg.cholesky_banded(ab)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Normal equations are singular: {K.shape[1]} unknowns for "
                f"{len(z)} data points, or too few data points near the ends "
                f"(end_condition='free' has 4 more unknowns than elements); "
                f"lower n_elements."
            ) from exc
        coeffs = sc.linalg.cho_solve_banded((cb, False), K.T @ F)

        if self.preserve_integral:
            t = self.knots
            w = ((t[_DEG + 1:] - t[:-_DEG - 1]) / (_DEG + 1))[kept]
            Minv_w = sc.linalg.cho_solve_banded((cb, False), w)
            target = sc.integrate.trapezoid(F, z, axis=0)
            coeffs -= np.outer(Minv_w, (w @ coeffs - target) / (w @ Minv_w))

        full = np.zeros((n_basis, F.shape[1]))
        full[kept] = coeffs
        for i, name in enumerate(names):
            self.splines[name] = sc.interpolate.BSpline(self.knots, full[:, i], _DEG)
            self.data[name] = (z, F[:, i])

    def _check_data_per_element(self, z: np.ndarray) -> None:
        k = np.clip(np.searchsorted(self.nodes, z, side="right") - 1, 0, self.n_elements - 1)
        empty = np.flatnonzero(np.bincount(k, minlength=self.n_elements) == 0)
        if len(empty):
            j = int(empty[0])
            raise ValueError(
                f"Element [{self.nodes[j]:.6g}, {self.nodes[j + 1]:.6g}] (and "
                f"{len(empty) - 1} others) contains no data point; lower "
                f"n_elements (at most one element per data point)."
            )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def element_params(self, name: tuple[str, int]) -> np.ndarray:
        """``(n_elements, 5)`` array of ``(f(s_k), f'(s_k), f(s_{k+1}),
        f'(s_{k+1}), mean_k)`` for component ``name``."""
        spl = self.splines[name]
        f = spl(self.nodes)
        df = spl.derivative()(self.nodes)
        mean = np.diff(spl.antiderivative()(self.nodes)) / self.delta
        return np.column_stack([f[:-1], df[:-1], f[1:], df[1:], mean])

    def _kept_names(self, field_tol: float | None, r_ref: float | None) -> list:
        """Components to export. With ``field_tol``, drop those whose scale
        ``max|f| * r_ref^n / n!`` is below ``field_tol`` times the largest
        der-0 field."""
        names = list(self.splines)
        if field_tol is None:
            return names
        if r_ref is None:
            raise ValueError("field_tol needs r_ref (reference radius [m])")
        scale = {nm: float(np.max(np.abs(self.splines[nm](self.nodes))))
                 * r_ref ** nm[1] / math.factorial(nm[1]) for nm in names}
        ref = max((v for nm, v in scale.items() if nm[1] == 0), default=0.0) or 1.0
        return [nm for nm in names if scale[nm] >= field_tol * ref]

    def to_line(
        self,
        multipole_order: int | None = None,
        steps_per_point: int = 1,
        field_tol: float | None = None,
        r_ref: float | None = None,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        radiation_flag: int = 0,
    ) -> xt.Line:
        """Line of ``SplineBoris`` elements, one per knot interval.

        ``multipole_order`` defaults to the highest fitted Bx/By order + 1;
        components not fitted are zero. Each element gets
        ``ceil(Delta / ds_data * steps_per_point)`` integration steps, with
        ``ds_data`` the finest data spacing seen in ``fit()``."""
        if not self.splines:
            raise RuntimeError("Call fit() before to_line().")
        if multipole_order is None:
            multipole_order = 1 + max(n for (fc, n) in self.splines if fc in ("Bx", "By"))
        params = {nm: self.element_params(nm) for nm in self._kept_names(field_tol, r_ref)}
        zero = np.zeros((self.n_elements, 5))

        def splines(fc, n, i):
            return Spline4(*params.get((fc, n), zero)[i])

        n_steps = max(1, math.ceil(self.delta / self._data_spacing * steps_per_point - 1e-6))
        name_width = len(str(self.n_elements - 1))
        elements, names = [], []
        for i in range(self.n_elements):
            elements.append(SplineBoris(
                bs=splines("Bs", 0, i),
                by=tuple(splines("By", n, i) for n in range(multipole_order)),
                bx=tuple(splines("Bx", n, i) for n in range(multipole_order)),
                length=self.delta,
                n_steps=n_steps,
                shift_x=shift_x,
                shift_y=shift_y,
                radiation_flag=radiation_flag,
            ))
            names.append(f"splineboris_{i:0{name_width}d}")
        return xt.Line(elements=elements, element_names=names)

    def to_multipole_line(
        self,
        p0c: float,
        multipole_order: int | None = None,
        q0: float = 1.0,
        field_at: str = "mean",
        field_tol: float | None = None,
        r_ref: float | None = None,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
    ) -> xt.Line:
        """Line of thick ``Multipole`` elements on the same nodes, with
        ``knl[n] = Delta / brho0 * (By, n)`` and ``ksl[n] = Delta / brho0 *
        (Bx, n)``, the field taken as the element mean (``field_at="mean"``)
        or the value at the element centre (``"midpoint"``). A coarse
        comparison baseline: B_s has no Multipole equivalent and is dropped."""
        if not self.splines:
            raise RuntimeError("Call fit() before to_multipole_line().")
        if field_at not in ("mean", "midpoint"):
            raise ValueError(f"field_at must be 'mean' or 'midpoint', got {field_at!r}")
        if multipole_order is None:
            multipole_order = 1 + max(n for (fc, n) in self.splines if fc in ("Bx", "By"))
        brho0 = p0c / (sc.constants.c * q0)
        kept = self._kept_names(field_tol, r_ref)
        if ("Bs", 0) in kept:
            print("[LongitudinalFitter] to_multipole_line(): Bs (solenoid) "
                  "has no Multipole equivalent and is dropped.")

        centres = 0.5 * (self.nodes[:-1] + self.nodes[1:])
        values = {}
        for nm in kept:
            if nm[0] in ("Bx", "By") and nm[1] < multipole_order:
                values[nm] = (self.element_params(nm)[:, 4] if field_at == "mean"
                              else self.splines[nm](centres))
        zero = np.zeros(self.n_elements)
        k = self.delta / brho0

        name_width = len(str(self.n_elements - 1))
        elements, names = [], []
        for i in range(self.n_elements):
            elements.append(xt.Multipole(
                knl=[k * values.get(("By", n), zero)[i] for n in range(multipole_order)],
                ksl=[k * values.get(("Bx", n), zero)[i] for n in range(multipole_order)],
                length=self.delta,
                isthick=True,
                shift_x=shift_x,
                shift_y=shift_y,
            ))
            names.append(f"multipole_{i:0{name_width}d}")
        return xt.Line(elements=elements, element_names=names)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_fields(self, der: int = 0, integrated: bool = False) -> None:
        """Data and fit of ``(Bx, der)``, ``(By, der)`` and (for der=0)
        ``(Bs, 0)``; with ``integrated=True`` their running integrals."""
        import matplotlib.pyplot as plt

        fcs = ["Bx", "By"] + (["Bs"] if der == 0 else [])
        fig, axes = plt.subplots(len(fcs), sharex=True, figsize=(10, 2.5 * len(fcs)),
                                 constrained_layout=True)
        s = np.linspace(self.s_start, self.s_end, 20 * self.n_elements + 1)
        for ax, fc in zip(axes, fcs):
            name = (fc, der)
            if name in self.splines:
                z, f = self.data[name]
                spl = self.splines[name]
                if integrated:
                    ax.plot(z, sc.integrate.cumulative_trapezoid(f, z, initial=0), label="Data")
                    ax.plot(s, spl.antiderivative()(s), "--", label="Fit")
                else:
                    ax.plot(z, f, ".", ms=2, label="Data")
                    ax.plot(s, spl(s), "-", lw=1, label="Fit")
            label = f"d^{der} {fc}/dx^{der}" if der else fc
            unit = "T" + (f"/m^{der}" if der else "")
            ax.set_ylabel(f"int {label} ds [{unit} m]" if integrated else f"{label} [{unit}]")
            ax.grid()
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("s [m]")
        plt.show()
