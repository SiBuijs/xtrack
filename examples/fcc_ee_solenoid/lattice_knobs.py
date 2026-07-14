"""Shared lattice knob helpers for FCC-ee solenoid study scripts."""

from __future__ import annotations

IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]


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
