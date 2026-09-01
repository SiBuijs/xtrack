"""Shared lattice knob helpers for FCC-ee solenoid study scripts."""

from __future__ import annotations

import numpy as np
import xtrack as xt
from xtrack.twiss import ClosedOrbitSearchError

IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]

# Local-s grid (distance from the IP; matches MAIN_SOLENOID_S_AXIS in
# 004a_build_and_check_solenoids.py / spline_boris_setup.build_splineboris_line)
# that every IP's 200-slice main solenoid (sol_slice_{ip}_*) is built on.
# 004b_install_solenoids_in_fcc_ring.py installs the main-solenoid template
# anchor='center' on the IP, so slice ii's local interval
# [MAIN_SOLENOID_S_AXIS[ii], MAIN_SOLENOID_S_AXIS[ii + 1]] equals its ring-s
# offset from that IP exactly -- used by zero_sextupole_window below.
MAIN_SOLENOID_S_AXIS = np.linspace(-2.399, 2.399, 201)

# Index of the main-solenoid SplineBoris slice immediately downstream of the
# insertion point: the extra sextupole is placed between slice
# (EXTRA_SEXTUPOLE_SLICE_INDEX - 1) and slice EXTRA_SEXTUPOLE_SLICE_INDEX of
# each IP's 200-slice main solenoid (built from MAIN_SOLENOID_S_AXIS =
# np.linspace(-2.399, 2.399, 201) in 004a_build_and_check_solenoids.py). Index
# 49 puts the boundary at s ~= -1.223 m from the IP -- the closest slice edge
# to the requested s = -1.23 m (chosen at user request, 2026-07-15).
EXTRA_SEXTUPOLE_SLICE_INDEX = 49


def set_lattice_knobs(
    line,
    *,
    with_solenoids: bool,
    with_correctors: bool,
    sext_amp: float = 1.0,
    comp_b_scale: float = 1.0,
    main_b_scale: float = 1.0,
) -> None:
    for ip_name in IP_NAMES:
        if f"on_sol_{ip_name}" in line.vars:
            line[f"on_sol_{ip_name}"] = float(with_solenoids)

        for corr_knob in (
            f"on_sol_corr_{ip_name}",
            f"on_comp_sol_{ip_name}",
            f"on_rot_doublet_left_{ip_name}",
            f"on_rot_doublet_right_{ip_name}",
            f"on_sol_orbit_corr_{ip_name}",
            f"on_sol_optics_corr_{ip_name}",
            f"on_sol_coupling_corr_{ip_name}",
        ):
            if corr_knob in line.vars:
                line[corr_knob] = float(with_correctors)

    if "sext_amp" in line.vars:
        line["sext_amp"] = float(sext_amp)

    if "comp_b_scale" in line.vars:
        line["comp_b_scale"] = float(comp_b_scale)

    if "main_b_scale" in line.vars:
        line["main_b_scale"] = float(main_b_scale)


def set_solenoid_offset(
    line,
    *,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    ip_names: list[str] = IP_NAMES,
) -> None:
    """Rigidly displace each IP's main detector solenoid by (x_offset, y_offset).

    Only the main solenoid slices (element names `sol_slice_{ip}_*`) are
    shifted; the compensation solenoids (ring hardware, not part of the
    detector) are left at their nominal position.
    """
    if x_offset == 0.0 and y_offset == 0.0:
        return

    prefixes = tuple(f"sol_slice_{ip_name}_" for ip_name in ip_names)
    for name in line.element_names:
        if name.startswith(prefixes):
            element = line[name]
            element.allow_rot_and_shift = True
            element.shift_x = x_offset
            element.shift_y = y_offset


def install_extra_sextupole(
    line,
    *,
    k2l: float,
    ip_names: list[str] = IP_NAMES,
    slice_index: int = EXTRA_SEXTUPOLE_SLICE_INDEX,
) -> None:
    """Insert a thin extra sextupole into each IP's main detector solenoid.

    Off by default: k2l=0.0 is a no-op, so the lattice is left untouched
    unless a nonzero integrated strength is explicitly requested. When
    active, one zero-length `xt.Sextupole` named `extra_sext_{ip_name}` is
    inserted per IP in `ip_names`, exactly at the boundary between slice
    (slice_index - 1) and slice slice_index of that IP's main solenoid (i.e.
    between two existing SplineBoris slices, at s ~= -1.223 m from the IP
    for the default slice_index -- see EXTRA_SEXTUPOLE_SLICE_INDEX above).
    The element has length=0 and carries its whole strength as an integrated
    `knl[2] = k2l` kick (rather than via `k2`, which is only meaningful for
    nonzero length): this is required for the insertion to land exactly on
    the existing slice boundary without splitting the neighbouring
    SplineBoris elements, which do not support slicing.
    """
    if k2l == 0.0:
        return

    for ip_name in ip_names:
        anchor_name = f"sol_slice_{ip_name}_{slice_index:03d}"
        if anchor_name not in line.element_names:
            raise ValueError(
                f"Could not find {anchor_name!r} in the line -- the main "
                "solenoid slicing may have changed; update "
                "EXTRA_SEXTUPOLE_SLICE_INDEX in lattice_knobs.py."
            )
        line.insert(
            f"extra_sext_{ip_name}",
            xt.Sextupole(length=0.0, knl=[0.0, 0.0, k2l]),
            at=0,
            from_=anchor_name,
            anchor="start",
            from_anchor="start",
        )


def zero_sextupole_window(
    line,
    *,
    s_min: float,
    s_max: float,
    order: int = 2,
    ip_names: list[str] = IP_NAMES,
    s_axis=MAIN_SOLENOID_S_AXIS,
) -> tuple[int, float, float]:
    """Zero the order-`order` bx/by Hermite row of every main-solenoid
    SplineBoris slice (`sol_slice_{ip}_*`) whose local-s interval (distance
    from that IP, negative=upstream/positive=downstream) overlaps
    [s_min, s_max], for every IP in `ip_names`.

    Ring-installed counterpart of
    004e_sextupole_window_study.py's zero_transverse_order_in_window, which
    probes the same idea on an isolated, not-yet-installed main-solenoid
    line. Here the target is the already-installed, knob-driven lattice: for
    order 2, the installed bx[2, k]/by[2, k] entries are normally tied to the
    sext_amp knob as a deferred expression (see
    004b_install_solenoids_in_fcc_ring.py), so a plain attribute assignment
    on the element would get silently overwritten the next time sext_amp
    changes (e.g. via set_lattice_knobs(), or robust_twiss()'s closed-orbit
    continuation fallback). Assigning through `line.element_refs[...]`
    instead replaces that dependency with a constant 0.0, so it survives
    later sext_amp changes (verified: a slice zeroed this way stays 0.0
    across a subsequent set_lattice_knobs(..., sext_amp=2.0) call, while an
    untouched slice still scales with sext_amp as expected).

    Call this AFTER set_lattice_knobs() (so sext_amp has already been
    applied to the elements this will override) and before the tracker is
    (re)built -- same ordering as install_extra_sextupole().

    Raises ValueError if no slice overlaps [s_min, s_max], or if the
    expected sol_slice_{ip}_* naming isn't found (main-solenoid slicing
    changed since s_axis was last checked against 004a's build).

    Returns (n_zeroed_total, covered_s_start, covered_s_end): the total
    slice count zeroed across all IPs, and the [start, end] window actually
    covered once s_min/s_max are snapped out to slice boundaries (the same
    for every IP, since they all share `s_axis`).
    """
    if s_min >= s_max:
        raise ValueError(f"s_min ({s_min}) must be < s_max ({s_max}).")

    name_width = len(str(len(s_axis) - 2))
    matching_indices = [
        ii for ii in range(len(s_axis) - 1)
        if s_axis[ii + 1] > s_min and s_axis[ii] < s_max
    ]
    if not matching_indices:
        raise ValueError(
            f"No main-solenoid slices overlap [{s_min}, {s_max}] -- window "
            "too narrow relative to the slice grid, or outside the "
            f"s-axis range [{s_axis[0]:g}, {s_axis[-1]:g}]."
        )
    covered_s_start = float(s_axis[matching_indices[0]])
    covered_s_end = float(s_axis[matching_indices[-1] + 1])

    n_zeroed = 0
    for ip_name in ip_names:
        for ii in matching_indices:
            element_name = f"sol_slice_{ip_name}_{ii:0{name_width}d}"
            if element_name not in line.element_names:
                raise ValueError(
                    f"Could not find {element_name!r} in the line -- the "
                    "main solenoid slicing may have changed; update "
                    "zero_sextupole_window's s_axis in lattice_knobs.py."
                )
            ref = line.element_refs[element_name]
            for k in range(5):
                ref.bx[order, k] = 0.0
                ref.by[order, k] = 0.0
            n_zeroed += 1

    return n_zeroed, covered_s_start, covered_s_end


def _continue_closed_orbit(
    line,
    *,
    var_name: str,
    target_value: float,
    baseline_value: float,
    growth_factor: float = 2.0,
    shrink_factor: float = 4.0,
    min_step: float | None = None,
    verbose: bool = True,
):
    """Ramp ``line.vars[var_name]`` from `baseline_value` to `target_value`.

    Same trick as `radial_steering.py`'s `df_hz` continuation chain: each
    step's converged closed orbit is used as the `co_guess` for the next
    step, which reaches closed orbits a single Newton search from the
    default (zero) guess cannot find. Unlike that script's fixed 1-unit
    step, the step size here adapts -- shrinking on a failed step and
    growing again after successes -- since the distance from `baseline_value`
    to an arbitrary requested `target_value` (e.g. sext_amp=3000) is not
    known in advance to be reachable in one step, or how many steps it needs.

    Leaves `var_name` set to `target_value` on return. Raises
    `ClosedOrbitSearchError` if the step shrinks below `min_step` before
    reaching the target.
    """
    total_span = target_value - baseline_value
    if min_step is None:
        min_step = max(abs(total_span) * 1e-4, 1e-9)
    direction = 1.0 if total_span >= 0 else -1.0

    line[var_name] = baseline_value
    value = baseline_value
    p_co = line.find_closed_orbit()

    step = abs(total_span)
    while value != target_value:
        next_value = value + direction * min(step, abs(target_value - value))
        line[var_name] = next_value
        try:
            p_co_next = line.find_closed_orbit(co_guess=p_co)
        except ClosedOrbitSearchError:
            step /= shrink_factor
            if step < min_step:
                line[var_name] = value
                raise ClosedOrbitSearchError(
                    f"Closed-orbit continuation on {var_name!r} stalled at "
                    f"{value:g} (target {target_value:g}, baseline "
                    f"{baseline_value:g}): step shrank below "
                    f"min_step={min_step:g}."
                ) from None
            continue
        p_co = p_co_next
        value = next_value
        if verbose:
            print(f"    co continuation: {var_name}={value:g} OK (step={step:g})")
        step = min(step * growth_factor, abs(target_value - value))

    return p_co


def robust_twiss(
    line,
    *,
    twiss_method: str = "twiss",
    var_name: str = "sext_amp",
    baseline_value: float = 1.0,
    verbose: bool = True,
    **twiss_kwargs,
):
    """`line.twiss(**twiss_kwargs)` with a closed-orbit continuation fallback.

    Tries the requested twiss (or `twiss6d` etc., via `twiss_method`)
    directly first, so the common case (`sext_amp` at or near its default of
    1.0) pays no extra cost. If that raises `ClosedOrbitSearchError` (as it
    does for large `sext_amp`, e.g. --sexamp 3000, where the true closed
    orbit is too far from the zero-guess starting point for a direct Newton
    search to converge), falls back to ramping `var_name` from
    `baseline_value` up to its current value on the line via
    `_continue_closed_orbit`, then retries the twiss using the converged
    orbit as `co_guess`.

    Only ramps `var_name` if it is an actual knob on `line` -- e.g. the
    VariableSolenoid lattice has no `sext_amp` knob at all, in which case
    the original error is re-raised unchanged.
    """
    twiss_fn = getattr(line, twiss_method)
    try:
        return twiss_fn(**twiss_kwargs)
    except ClosedOrbitSearchError:
        if var_name not in line.vars:
            raise
        target_value = float(line.vars[var_name]._value)
        if verbose:
            print(
                f"  twiss() failed to find closed orbit at "
                f"{var_name}={target_value:g}; retrying via continuation "
                f"from {var_name}={baseline_value:g} ..."
            )
        p_co = _continue_closed_orbit(
            line,
            var_name=var_name,
            target_value=target_value,
            baseline_value=baseline_value,
            verbose=verbose,
        )
        return twiss_fn(co_guess=p_co, **twiss_kwargs)
