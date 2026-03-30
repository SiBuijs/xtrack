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

        # Group segments by canonical region_index / geometry.
        # Within each region we expect:
        #   - one Bs segment with derivative_x == 0 (longitudinal)
        #   - zero or more By/Bx segments for derivative_x in [0, multipole_order-1]
        by_components: dict[int, List[Optional[xt.Spline4]]] = {}
        bx_components: dict[int, List[Optional[xt.Spline4]]] = {}
        bs_components: dict[int, Optional[xt.Spline4]] = {}
        region_geom: dict[int, tuple[float, float, int, int]] = {}

        max_region_index = -1
        for seg in segments:
            rid = seg.region_index
            max_region_index = max(max_region_index, rid)
            # Record geometry (s_start, s_end, idx_start, idx_end), checking consistency.
            geom = (seg.s_start, seg.s_end, seg.idx_start, seg.idx_end)
            prev_geom = region_geom.get(rid)
            if prev_geom is None:
                region_geom[rid] = geom
            else:
                if any(abs(a - b) > 0 for a, b in zip(prev_geom[:2], geom[:2])) or prev_geom[2:] != geom[2:]:
                    raise ValueError(
                        f"Inconsistent geometry for region_index={rid}: "
                        f"{prev_geom} vs {geom}"
                    )

            # Build Spline4 objects from Hermite parameters.
            h = np.asarray(seg.hermite_params, dtype=float)
            if h.size != len(xt.SplineBoris._HERMITE_SUFFIXES):
                raise ValueError(
                    f"Expected {len(xt.SplineBoris._HERMITE_SUFFIXES)} Hermite parameters, "
                    f"got {h.size}"
                )
            val_start, der_start, val_end, der_end, integral = h
            spline = xt.Spline4(
                val_start=float(val_start),
                der_start=float(der_start),
                val_end=float(val_end),
                der_end=float(der_end),
                integral=float(integral),
            )

            if seg.field_component == "Bs":
                if seg.derivative_x != 0:
                    raise ValueError("Bs segments must have derivative_x == 0")
                prev_bs = bs_components.get(rid)
                if prev_bs is not None:
                    raise ValueError(f"Multiple Bs segments for region_index={rid}")
                bs_components[rid] = spline

            elif seg.field_component == "By":
                comp = by_components.setdefault(rid, [])
                # Grow list to accommodate this derivative index.
                while len(comp) <= seg.derivative_x:
                    comp.append(None)
                if comp[seg.derivative_x] is not None:
                    raise ValueError(
                        f"Duplicate By segment for region_index={rid}, der={seg.derivative_x}"
                    )
                comp[seg.derivative_x] = spline

            elif seg.field_component == "Bx":
                comp = bx_components.setdefault(rid, [])
                while len(comp) <= seg.derivative_x:
                    comp.append(None)
                if comp[seg.derivative_x] is not None:
                    raise ValueError(
                        f"Duplicate Bx segment for region_index={rid}, der={seg.derivative_x}"
                    )
                comp[seg.derivative_x] = spline

            else:
                raise ValueError(f"Unknown field_component={seg.field_component!r}")

        elements: List[xt.SplineBoris] = []
        for rid in range(max_region_index + 1):
            if rid not in region_geom:
                continue
            s_start, s_end, idx_start, idx_end = region_geom[rid]
            length = float(s_end - s_start)
            if length <= 0.0 or not np.isfinite(length):
                raise ValueError(f"Invalid region length for region_index={rid}: {length}")

            # Determine n_steps from either data-point indices or physical length.
            if steps_per_point is not None:
                n_points = max(int(idx_end - idx_start), 1)
                n_steps = max(n_points * int(steps_per_point), 1)
            else:
                n_steps = max(int(length * float(steps_per_meter)), 1)

            # Longitudinal component: if absent, pass None so SplineBoris can infer.
            bs_spline = bs_components.get(rid)

            # Transverse multipoles: ensure tuples span up to fit.multipole_order - 1
            # using None for missing derivative orders.
            def _pad_component(comp_dict: dict[int, List[Optional[xt.Spline4]]]) -> tuple:
                if rid not in comp_dict:
                    return ()
                comp_list = list(comp_dict[rid])
                # Trim trailing None to keep representation compact while preserving
                # derivative ordering (SplineBoris will handle sparse tuples).
                while comp_list and comp_list[-1] is None:
                    comp_list.pop()
                return tuple(comp_list)

            by_tuple = _pad_component(by_components)
            bx_tuple = _pad_component(bx_components)

            elem = xt.SplineBoris(
                bs=bs_spline,
                by=by_tuple if by_tuple else None,
                bx=bx_tuple if bx_tuple else None,
                s_start=float(s_start),
                length=length,
                n_steps=int(n_steps),
                radiation_flag=int(radiation_flag),
                shift_x=float(shift_x),
                shift_y=float(shift_y),
                hx=float(hx),
            )
            elements.append(elem)

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

