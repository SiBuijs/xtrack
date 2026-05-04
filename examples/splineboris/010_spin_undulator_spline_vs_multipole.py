"""
Quick spin comparison: SplineBoris vs multipole undulator.

Tracks one particle with initial spin (sx, sy, sz) = (0, 1, 0) through:
- a SplineBoris undulator line
- an equivalent thick-multipole undulator line

and overlays spin components along s using dashed style for multipole.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xtrack as xt

from _undulator_multipole_builder import build_multipole_kick_undulator
from xtrack._temp.field_fitter import FieldFitter
from xtrack._temp.splineboris_sequence import SplineBorisSequence


FIT_PARS_INDEX_COLS = [
    "field_component",
    "derivative_x",
    "region_name",
    "s_start",
    "s_end",
    "idx_start",
    "idx_end",
    "param_index",
]

MULTIPOLE_ORDER = 3
OUTPUT_DIRNAME = "_010_results"
OUTPUT_FIG_NAME = "spin_components_spline_vs_multipole.png"


def _load_fit_pars():
    test_data = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls"
    df_raw_data = pd.read_csv(
        test_data / "undulator_field_map.txt",
        sep=r"\s+",
        header=None,
        names=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
    ).set_index(["X", "Y", "Z"])

    fitter = FieldFitter(
        raw_data=df_raw_data,
        xy_point=(0.0, 0.0),
        distance_unit=1e-3,
        min_region_size=5,
        deg=MULTIPOLE_ORDER - 1,
        field_tol=1e-3,
    )
    fitter.fit()
    return fitter.df_fit_pars


def _add_correctors_and_match_orbit(env, line, name_prefix, match_orbit=True):
    corr_var_names = [
        f"{name_prefix}_k0l_corr1",
        f"{name_prefix}_k0l_corr2",
        f"{name_prefix}_k0l_corr3",
        f"{name_prefix}_k0l_corr4",
        f"{name_prefix}_k0sl_corr1",
        f"{name_prefix}_k0sl_corr2",
        f"{name_prefix}_k0sl_corr3",
        f"{name_prefix}_k0sl_corr4",
    ]
    for vv in corr_var_names:
        if vv not in env.vars:
            env[vv] = 0.0

    corr_names = [f"{name_prefix}_corr{i}" for i in range(1, 5)]
    for ii, corr_name in enumerate(corr_names, start=1):
        if corr_name not in env.element_dict:
            env.new(
                corr_name,
                xt.Multipole,
                knl=[f"{name_prefix}_k0l_corr{ii}"],
                ksl=[f"{name_prefix}_k0sl_corr{ii}"],
            )

    tt = line.get_table()
    s_vals = np.asarray(tt.s[:-1], dtype=float)
    element_names = list(line.element_names)
    line_length = float(tt.s[-1])
    desired_positions = [0.02, 0.1, line_length - 0.1, line_length - 0.02]
    desired_positions = [float(np.clip(ss, 0.0, line_length)) for ss in desired_positions]

    insertions = {}
    for corr_name, s_target in zip(corr_names, desired_positions):
        idx = int(np.argmin(np.abs(s_vals - s_target)))
        insertions.setdefault(idx, []).append(corr_name)

    element_names_with_corr = []
    for ii, ee_name in enumerate(element_names):
        if ii in insertions:
            element_names_with_corr.extend(insertions[ii])
        element_names_with_corr.append(ee_name)

    corrected_line = xt.Line(env=env, element_names=element_names_with_corr)
    corrected_line.particle_ref = line.particle_ref.copy()
    if match_orbit:
        corrected_line.build_tracker()
        opt = corrected_line.match(
            solve=False,
            betx=2.59727482,
            bety=1.44807519,
            only_orbit=True,
            include_collective=True,
            vary=xt.VaryList(corr_var_names, step=1e-6),
            targets=[
                xt.TargetSet(x=0, px=0, y=0, py=0, at=xt.END),
                xt.TargetSet(x=0, y=0, at=corr_names[1]),
                xt.TargetSet(x=0, y=0, at=corr_names[2]),
            ],
        )
        opt.step(2)
        corrected_line.discard_tracker()
    return corrected_line


def _build_lines(df_fit_pars, p_ref):
    env = xt.Environment()

    x_off = 0*1e-3
    y_off = 0*1e-3

    seq = SplineBorisSequence(
        df_fit_pars=df_fit_pars,
        multipole_order=MULTIPOLE_ORDER,
        steps_per_point=1,
        shift_x=x_off,
        shift_y=y_off,
    )
    line_spline = seq.to_line(env=env)
    line_spline.particle_ref = p_ref.copy()

    line_multipole, _ = build_multipole_kick_undulator(
        env=env,
        p_ref=p_ref,
        df_fit_pars=df_fit_pars,
        multipole_order=MULTIPOLE_ORDER,
        name_prefix="spin_cmp",
        multipole_isthick=True,
        shift_x=x_off,
        shift_y=y_off,
    )
    line_multipole.particle_ref = p_ref.copy()

    line_spline = _add_correctors_and_match_orbit(
        env=env, line=line_spline, name_prefix="sp", match_orbit=True
    )
    line_multipole = _add_correctors_and_match_orbit(
        env=env, line=line_multipole, name_prefix="sp", match_orbit=False
    )

    return line_spline, line_multipole


def _track_spin_ebe(line, p_init):
    p = p_init.copy()
    line.configure_spin(spin_model="auto")
    line.track(p, turn_by_turn_monitor="ONE_TURN_EBE")
    mon = line.record_last_track
    return (
        np.asarray(mon.s[0], dtype=float),
        np.asarray(mon.spin_x[0], dtype=float),
        np.asarray(mon.spin_y[0], dtype=float),
        np.asarray(mon.spin_z[0], dtype=float),
    )


def _track_orbit_ebe(line, p_init):
    p = p_init.copy()
    line.track(p, turn_by_turn_monitor="ONE_TURN_EBE")
    mon = line.record_last_track
    return (
        np.asarray(mon.s[0], dtype=float),
        np.asarray(mon.x[0], dtype=float),
        np.asarray(mon.y[0], dtype=float),
    )


def _plot_spin_components(spline_data, multipole_data):
    s_sp, sx_sp, sy_sp, sz_sp = spline_data
    s_mp, sx_mp, sy_mp, sz_mp = multipole_data

    # Requested palette:
    # sy uses blue/orange; remaining series continue with default C2..C5.
    colors = {
        "sy_sp": "C0",
        "sy_mp": "C1",
        "sx_sp": "C2",
        "sx_mp": "C3",
        "sz_sp": "C4",
        "sz_mp": "C5",
    }

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(s_sp, sy_sp, "-", color=colors["sy_sp"], lw=1.8, label="SplineBoris")
    axs[0].plot(s_mp, sy_mp, "--", color=colors["sy_mp"], lw=1.8, label="Multipole")
    axs[0].set_ylabel("spin_y")
    axs[0].grid(True, alpha=0.35)
    axs[0].legend(loc="best")

    axs[1].plot(s_sp, sx_sp, "-", color=colors["sx_sp"], lw=1.8, label="SplineBoris")
    axs[1].plot(s_mp, sx_mp, "--", color=colors["sx_mp"], lw=1.8, label="Multipole")
    axs[1].set_ylabel("spin_x")
    axs[1].grid(True, alpha=0.35)
    axs[1].legend(loc="best")

    axs[2].plot(s_sp, sz_sp, "-", color=colors["sz_sp"], lw=1.8, label="SplineBoris")
    axs[2].plot(s_mp, sz_mp, "--", color=colors["sz_mp"], lw=1.8, label="Multipole")
    axs[2].set_ylabel("spin_z")
    axs[2].set_xlabel("s [m]")
    axs[2].grid(True, alpha=0.35)
    axs[2].legend(loc="best")

    fig.suptitle("Spin through undulator: SplineBoris (solid) vs Multipole (dashed)")
    fig.tight_layout()
    plt.show()


def _plot_spin_x(spline_data, multipole_data):
    """Spin x-component only: SplineBoris vs multipole in one figure."""
    s_sp, sx_sp, _sy_sp, _sz_sp = spline_data
    s_mp, sx_mp, _sy_mp, _sz_mp = multipole_data

    label_fs, title_fs, tick_fs, legend_fs = 14, 15, 12, 12

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(s_sp, sx_sp, "-", color="C0", lw=1.8, label="SplineBoris")
    ax.plot(s_mp, sx_mp, "--", color="C1", lw=1.8, label="Multipole")
    ax.set_xlabel(r"Longitudinal position, $s$ [m]", fontsize=label_fs)
    ax.set_ylabel(r"Hor. spin component, $s_x$", fontsize=label_fs)
    ax.set_title(
        r"$s_x$ through undulator: SB vs MK (1 mm horizontal offset)",
        fontsize=title_fs,
    )
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=legend_fs)
    fig.tight_layout()
    plt.show()


def _interp_orbit_xy_onto_s(s_src, x_src, y_src, s_tgt):
    x_tgt = np.interp(s_tgt, s_src, x_src)
    y_tgt = np.interp(s_tgt, s_src, y_src)
    return x_tgt, y_tgt


def _print_orbit_xy_difference_meters(spline_orbit, multipole_orbit):
    """Print max (and RMS) transverse orbit difference; multipole is interpolated to spline s."""
    s_sp, x_sp, y_sp = spline_orbit
    s_mp, x_mp, y_mp = multipole_orbit
    x_mp_i, y_mp_i = _interp_orbit_xy_onto_s(s_mp, x_mp, y_mp, s_sp)
    dx = x_sp - x_mp_i
    dy = y_sp - y_mp_i
    dr = np.hypot(dx, dy)
    print(
        "Orbit difference (SplineBoris minus Multipole Kick), multipole interpolated to spline s:"
    )
    print(f"  max |Δx| = {np.max(np.abs(dx)):.6e} m")
    print(f"  max |Δy| = {np.max(np.abs(dy)):.6e} m")
    print(f"  max √(Δx²+Δy²) = {np.max(dr):.6e} m")
    rms_dx = float(np.sqrt(np.mean(dx * dx)))
    rms_dy = float(np.sqrt(np.mean(dy * dy)))
    print(f"  RMS Δx = {rms_dx:.6e} m")
    print(f"  RMS Δy = {rms_dy:.6e} m")

    # Relative to SplineBoris (same s samples): peak amplitude and RMS orbit amplitude.
    max_abs_x_sb = float(np.max(np.abs(x_sp)))
    max_abs_y_sb = float(np.max(np.abs(y_sp)))
    r_sb = np.hypot(x_sp, y_sp)
    max_r_sb = float(np.max(r_sb))
    rms_x_sb = float(np.sqrt(np.mean(x_sp * x_sp)))
    rms_y_sb = float(np.sqrt(np.mean(y_sp * y_sp)))
    rms_r_sb = float(np.sqrt(np.mean(r_sb * r_sb)))

    print("Relative to SplineBoris (SB) reference on the same s grid:")
    eps = 1e-30
    if max_abs_x_sb > eps:
        print(f"  max |Δx| / max|x_SB| = {np.max(np.abs(dx)) / max_abs_x_sb:.6e}")
    else:
        print("  max |Δx| / max|x_SB| = (n/a, max|x_SB| ~ 0)")
    if max_abs_y_sb > eps:
        print(f"  max |Δy| / max|y_SB| = {np.max(np.abs(dy)) / max_abs_y_sb:.6e}")
    else:
        print("  max |Δy| / max|y_SB| = (n/a, max|y_SB| ~ 0)")
    if max_r_sb > eps:
        print(f"  max √(Δx²+Δy²) / max|√(x²+y²)_SB| = {np.max(dr) / max_r_sb:.6e}")
    else:
        print("  max √(Δx²+Δy²) / max|√(x²+y²)_SB| = (n/a, transverse amplitude ~ 0)")
    if rms_x_sb > eps:
        print(f"  RMS Δx / RMS|x_SB| = {rms_dx / rms_x_sb:.6e}")
    else:
        print("  RMS Δx / RMS|x_SB| = (n/a, RMS|x_SB| ~ 0)")
    if rms_y_sb > eps:
        print(f"  RMS Δy / RMS|y_SB| = {rms_dy / rms_y_sb:.6e}")
    else:
        print("  RMS Δy / RMS|y_SB| = (n/a, RMS|y_SB| ~ 0)")
    if rms_r_sb > eps:
        rms_dr = float(np.sqrt(np.mean(dr * dr)))
        print(f"  RMS √(Δx²+Δy²) / RMS|√(x²+y²)_SB| = {rms_dr / rms_r_sb:.6e}")
    else:
        print("  RMS transverse / RMS|√(x²+y²)_SB| = (n/a, RMS transverse SB ~ 0)")


def _plot_orbit_xy(spline_orbit, multipole_orbit):
    """x(s) and y(s) for SplineBoris vs multipole: one figure, two stacked subplots."""
    s_sp, x_sp, y_sp = spline_orbit
    s_mp, x_mp, y_mp = multipole_orbit

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axs[0].plot(s_sp, x_sp, "-", color="C0", lw=1.8, label="SplineBoris")
    axs[0].plot(s_mp, x_mp, "--", color="C1", lw=1.8, label="Multipole Kick")
    axs[0].set_ylabel("x [m]")
    axs[0].grid(True, alpha=0.35)
    axs[0].legend(loc="best")

    axs[1].plot(s_sp, y_sp, "-", color="C0", lw=1.8, label="SplineBoris")
    axs[1].plot(s_mp, y_mp, "--", color="C1", lw=1.8, label="Multipole Kick")
    axs[1].set_ylabel("y [m]")
    axs[1].set_xlabel("s [m]")
    axs[1].grid(True, alpha=0.35)
    axs[1].legend(loc="best")

    fig.suptitle("Transverse orbit: SplineBoris (solid) vs Multipole Kick (dashed)")
    fig.tight_layout()
    plt.show()


def _plot_orbit_3d(spline_orbit, multipole_orbit):
    s_sp, x_sp, y_sp = spline_orbit
    s_mp, x_mp, y_mp = multipole_orbit

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot transverse orbit in mm for easier visual comparison.
    ax.plot(s_sp, x_sp * 1e3, y_sp * 1e3, "-", lw=1.8, color="C0", label="SplineBoris")
    ax.plot(
        s_mp,
        x_mp * 1e3,
        y_mp * 1e3,
        "--",
        lw=1.8,
        color="C1",
        label="Multipole",
    )

    ax.set_xlabel("s [m]")
    ax.set_ylabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_title("3D orbit through undulator")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare spin tracking in undulator: SplineBoris vs multipole."
    )
    parser.parse_args()

    df_fit_pars = _load_fit_pars()
    p_ref = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1,
        p0c=2.7e9,
        anomalous_magnetic_moment=0.00115965218128,
    )
    line_spline, line_multipole = _build_lines(df_fit_pars=df_fit_pars, p_ref=p_ref)

    p_init = p_ref.copy()
    p_init.x = 0.0
    p_init.px = 0.0
    p_init.y = 0.0
    p_init.py = 0.0
    p_init.zeta = 0.0
    p_init.delta = 0.0
    p_init.spin_x = 0.0
    p_init.spin_y = 1.0
    p_init.spin_z = 0.0

    spline_data = _track_spin_ebe(line_spline, p_init)
    multipole_data = _track_spin_ebe(line_multipole, p_init)
    spline_orbit = _track_orbit_ebe(line_spline, p_init)
    multipole_orbit = _track_orbit_ebe(line_multipole, p_init)

    _print_orbit_xy_difference_meters(spline_orbit, multipole_orbit)
    _plot_orbit_xy(spline_orbit, multipole_orbit)
    _plot_orbit_3d(spline_orbit, multipole_orbit)

    _plot_spin_components(spline_data, multipole_data)
    _plot_spin_x(spline_data, multipole_data)

if __name__ == "__main__":
    main()