from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

import xtrack as xt
from .field_fitter import FieldFitResult, FitSegment


@dataclass
class SplineBorisSequence:
    """Builder-only sequence of ``SplineBoris`` elements from a ``FieldFitResult``.

    This is a thin, experimental helper that turns canonical fit output into a
    list of :class:`xtrack.SplineBoris` elements, suitable for constructing a
    line. It intentionally does *not* expose any DataFrame-based interfaces.
    """

    elements: List[xt.SplineBoris]

    @classmethod
    def from_fit_result(
        cls,
        fit: FieldFitResult,
        *,
        steps_per_point: Optional[int] = None,
        steps_per_meter: Optional[float] = None,
        radiation_flag: int = 0,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        hx: float = 0.0,
    ) -> "SplineBorisSequence":
        if (steps_per_point is None) == (steps_per_meter is None):
            raise ValueError(
                "Exactly one of steps_per_point or steps_per_meter must be provided."
            )

        segments = fit.segments
        if not segments:
            return cls(elements=[])

        s_full = np.asarray(fit.s_full, dtype=float)
        if s_full.ndim != 1 or s_full.size < 2:
            raise ValueError("fit.s_full must be a 1D array with at least two points.")
        if np.any(~np.isfinite(s_full)):
            raise ValueError("fit.s_full contains non-finite values.")

        zero_spline = xt.Spline4(
            val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, integral=0.0
        )

        def _split_segment_on_interval(seg: FitSegment, s0: float, s1: float) -> xt.Spline4:
            h = np.asarray(seg.hermite_params, dtype=float)
            if h.size != len(xt.SplineBoris._HERMITE_SUFFIXES):
                raise ValueError(
                    f"Expected {len(xt.SplineBoris._HERMITE_SUFFIXES)} Hermite parameters, "
                    f"got {h.size}"
                )

            poly = xt.SplineBoris.hermite_to_polynomial(seg.s_start, seg.s_end, h)
            dpoly = poly.deriv()
            ipoly = poly.integ()

            l0 = float(s0 - seg.s_start)
            l1 = float(s1 - seg.s_start)
            ds = l1 - l0
            if ds <= 0.0:
                raise ValueError(
                    "Invalid canonical interval while splitting segment: "
                    f"s0={s0}, s1={s1}, segment=({seg.s_start}, {seg.s_end})"
                )

            return xt.Spline4(
                val_start=float(poly(l0)),
                der_start=float(dpoly(l0)),
                val_end=float(poly(l1)),
                der_end=float(dpoly(l1)),
                integral=float((ipoly(l1) - ipoly(l0)) / ds),
            )

        by_segments: dict[int, List[FitSegment]] = {}
        bx_segments: dict[int, List[FitSegment]] = {}
        bs_segments: List[FitSegment] = []
        present_channels: set[tuple[str, int]] = set()

        boundary_indices = {0, int(s_full.size - 1)}
        for seg in segments:
            if seg.field_component not in ("Bx", "By", "Bs"):
                raise ValueError(f"Unknown field_component={seg.field_component!r}")
            if seg.field_component == "Bs" and seg.derivative_x != 0:
                raise ValueError("Bs segments must have derivative_x == 0")
            if seg.derivative_x < 0:
                raise ValueError(f"derivative_x must be >= 0, got {seg.derivative_x}")
            if seg.idx_start < 0 or seg.idx_end >= s_full.size or seg.idx_start >= seg.idx_end:
                raise ValueError(
                    "Invalid idx bounds in fit segment: "
                    f"idx_start={seg.idx_start}, idx_end={seg.idx_end}, n_s={s_full.size}"
                )

            boundary_indices.add(int(seg.idx_start))
            boundary_indices.add(int(seg.idx_end))
            present_channels.add((seg.field_component, int(seg.derivative_x)))

            if seg.field_component == "By":
                by_segments.setdefault(int(seg.derivative_x), []).append(seg)
            elif seg.field_component == "Bx":
                bx_segments.setdefault(int(seg.derivative_x), []).append(seg)
            else:
                bs_segments.append(seg)

        idx_to_s_values: dict[int, list[float]] = {}
        for seg in segments:
            idx_to_s_values.setdefault(int(seg.idx_start), []).append(float(seg.s_start))
            idx_to_s_values.setdefault(int(seg.idx_end), []).append(float(seg.s_end))

        sorted_boundaries = sorted(boundary_indices)
        if len(sorted_boundaries) < 2:
            return cls(elements=[])

        boundary_s: dict[int, float] = {}
        for ii in sorted_boundaries:
            vals = idx_to_s_values.get(int(ii))
            if vals:
                boundary_s[int(ii)] = float(np.mean(vals))
            else:
                boundary_s[int(ii)] = float(s_full[ii])

        def _find_covering_segment(seg_list: List[FitSegment], i0: int, i1: int) -> FitSegment:
            covering = [ss for ss in seg_list if ss.idx_start <= i0 and ss.idx_end >= i1]
            if not covering:
                raise ValueError(
                    f"No segment covers interval idx [{i0}, {i1}] for requested component."
                )
            if len(covering) > 1:
                raise ValueError(
                    f"Multiple segments cover interval idx [{i0}, {i1}] for requested component."
                )
            return covering[0]

        multipole_order = max(int(fit.multipole_order), 1)

        elements: List[xt.SplineBoris] = []
        for i_left, i_right in zip(sorted_boundaries[:-1], sorted_boundaries[1:]):
            if i_right <= i_left:
                raise ValueError(
                    f"Canonical boundaries are not strictly increasing: {i_left}, {i_right}"
                )
            s_start = float(boundary_s[i_left])
            s_end = float(boundary_s[i_right])
            length = float(s_end - s_start)
            if length <= 0.0 or not np.isfinite(length):
                raise ValueError(
                    "Invalid canonical interval length: "
                    f"idx=({i_left}, {i_right}), s=({s_start}, {s_end})"
                )

            # steps_per_point refers to intervals (sample-to-sample), not point count.
            if steps_per_point is not None:
                n_intervals = int(i_right - i_left)
                n_steps = max(n_intervals * int(steps_per_point), 1)
            else:
                n_steps = max(int(length * float(steps_per_meter)), 1)

            bs_seg = _find_covering_segment(bs_segments, i_left, i_right)
            bs_spline = _split_segment_on_interval(bs_seg, s_start, s_end)

            by_tuple: tuple[xt.Spline4, ...] = tuple(
                _split_segment_on_interval(
                    _find_covering_segment(by_segments[der], i_left, i_right), s_start, s_end
                )
                if ("By", der) in present_channels
                else zero_spline
                for der in range(multipole_order)
            )
            bx_tuple: tuple[xt.Spline4, ...] = tuple(
                _split_segment_on_interval(
                    _find_covering_segment(bx_segments[der], i_left, i_right), s_start, s_end
                )
                if ("Bx", der) in present_channels
                else zero_spline
                for der in range(multipole_order)
            )

            elem = xt.SplineBoris(
                bs=bs_spline,
                by=by_tuple,
                bx=bx_tuple,
                s_start=float(s_start),
                length=length,
                n_steps=int(n_steps),
                radiation_flag=int(radiation_flag),
                shift_x=float(shift_x),
                shift_y=float(shift_y),
                hx=float(hx),
            )
            elements.append(elem)

        for prev, nxt in zip(elements[:-1], elements[1:]):
            if not np.isclose(prev.s_end, nxt.s_start):
                raise ValueError(
                    "Canonical intervals are not contiguous: "
                    f"{prev.s_start}->{prev.s_end} then {nxt.s_start}->{nxt.s_end}"
                )

        return cls(elements=elements)

    @property
    def length(self) -> float:
        if not self.elements:
            return 0.0
        first = self.elements[0]
        last = self.elements[-1]
        return float(last.s_end - first.s_start)

    def to_line(self) -> xt.Line:
        """Return a new ``xt.Line`` built from the underlying ``SplineBoris`` elements."""
        return xt.Line(elements=list(self.elements))

