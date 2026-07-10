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
