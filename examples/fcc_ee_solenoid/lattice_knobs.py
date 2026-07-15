"""Shared lattice knob helpers for FCC-ee solenoid study scripts."""

from __future__ import annotations

import xtrack as xt

IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]

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
