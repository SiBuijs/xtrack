"""
Undulator-only comparison between SplineBoris and multipole-kick models.

This script:
1) Builds standalone undulator lines for both models.
2) Scans initial x from 0 to 1e-2 m (10 points), with y=px=py=0.
3) Tracks one particle per model and stores final-state signed differences
   (SplineBoris - Multipole) for x, y, px, py.
4) Runs twiss4d(betx=1, bety=1) on each line and compares betx2/bety2.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xtrack as xt
from xtrack._temp.field_fitter import FieldFitter
from xtrack._temp.splineboris_sequence import SplineBorisSequence

from _undulator_multipole_builder import build_multipole_kick_undulator


E0_EV = 2.7e9
MULTIPOLE_ORDER = 3
DISTANCE_UNIT_M = 0.001
SHOW_PLOTS = True
SAVE_PLOTS = False
SAVE_RESULTS = True
RESULTS_DIRNAME = "_008_results"
CORRECTOR_RELATIVE_POSITIONS_M = (0.02, 0.1, -0.1, -0.02)
TWISS_INITIAL_CONDITIONS = {
    "ars02_uind_0500_1": dict(betx=2.59727482, bety=1.44807519, alfx=0.19591103, alfy=0.38979385, dx=-7.43560007e-08, dy=0.0),
    "ars03_uind_0380_1": dict(betx=4.39717902, bety=3.75565786, alfx=0.54261554, alfy=0.77946628, dx=-1.51077915e-07, dy=0.0),
    "ars04_uind_0500_1": dict(betx=2.78336681, bety=1.81837908, alfx=0.33584091, alfy=0.66821784, dx=3.83840914e-08, dy=0.0),
    "ars05_uind_0650_1": dict(betx=11.45026796, bety=11.97554896, alfx=0.83777879, alfy=1.40409169, dx=1.23912248e-08, dy=0.0),
    "ars06_uind_0500_1": dict(betx=3.41525195, bety=3.07569734, alfx=0.60450229, alfy=1.20279189, dx=-1.1604547e-07, dy=0.0),
    "ars07_uind_0200_1": dict(betx=4.51579259, bety=3.92604491, alfx=0.57388924, alfy=0.82492427, dx=1.99498907e-07, dy=0.0),
    "ars08_uind_0500_1": dict(betx=3.41526298, bety=3.07569706, alfx=0.60449373, alfy=1.20279147, dx=-5.8337148e-08, dy=0.0),
    "ars09_uind_0790_1": dict(betx=10.08357702, bety=8.01748989, alfx=0.52981076, alfy=1.07857512, dx=-1.70706185e-07, dy=0.0),
    "ars11_uind_0210_1": dict(betx=4.44118089, bety=3.83947657, alfx=0.56150852, alfy=0.82022044, dx=-1.40673725e-07, dy=0.0),
    "ars11_uind_0610_1": dict(betx=3.4990302, bety=2.48298121, alfx=-0.19043697, alfy=-0.28595681, dx=-1.43041448e-07, dy=0.0),
    "ars12_uind_0500_1": dict(betx=3.41529512, bety=3.07569724, alfx=0.6044942, alfy=1.20279132, dx=-5.53357827e-08, dy=0.0),
}
TWISS_SEED_NAME = "ars02_uind_0500_1"


def _load_raw_field_data():
    base_dir = Path(__file__).resolve().parent
    field_map_path = base_dir.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
    return pd.read_csv(
        field_map_path,
        sep="\t",
        header=None,
        names=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
    ).set_index(["X", "Y", "Z"])


def _fit_field_data(df_raw):
    fitter = FieldFitter(
        raw_data=df_raw,
        xy_point=(0, 0),
        distance_unit=DISTANCE_UNIT_M,
        min_region_size=5,
        deg=MULTIPOLE_ORDER - 1,
    )
    fitter.fit()
    return fitter.df_fit_pars


def _insert_corrected_line(env, line, name_prefix, relative_positions_m, strengths=None):
    env[f"k0l_{name_prefix}1"] = 0.0
    env[f"k0l_{name_prefix}2"] = 0.0
    env[f"k0l_{name_prefix}3"] = 0.0
    env[f"k0l_{name_prefix}4"] = 0.0
    env[f"k0sl_{name_prefix}1"] = 0.0
    env[f"k0sl_{name_prefix}2"] = 0.0
    env[f"k0sl_{name_prefix}3"] = 0.0
    env[f"k0sl_{name_prefix}4"] = 0.0

    corr_names = [f"{name_prefix}_corr{i}" for i in range(1, 5)]
    env.new(corr_names[0], xt.Multipole, knl=[f"k0l_{name_prefix}1"], ksl=[f"k0sl_{name_prefix}1"])
    env.new(corr_names[1], xt.Multipole, knl=[f"k0l_{name_prefix}2"], ksl=[f"k0sl_{name_prefix}2"])
    env.new(corr_names[2], xt.Multipole, knl=[f"k0l_{name_prefix}3"], ksl=[f"k0sl_{name_prefix}3"])
    env.new(corr_names[3], xt.Multipole, knl=[f"k0l_{name_prefix}4"], ksl=[f"k0sl_{name_prefix}4"])

    tt = line.get_table()
    element_boundaries = [0.0]
    for ss in np.asarray(tt.s[:-1]):
        element_boundaries.append(float(ss))

    length = float(element_boundaries[-1])
    desired_positions = [
        relative_positions_m[0],
        relative_positions_m[1],
        length + relative_positions_m[2],
        length + relative_positions_m[3],
    ]

    insertions = {}
    for corr_name, s_target in zip(corr_names, desired_positions):
        idx = int(np.argmin(np.abs(np.asarray(element_boundaries) - s_target)))
        insertions.setdefault(idx, []).append(corr_name)

    names_with_correctors = []
    for ii, nn in enumerate(line.element_names):
        if ii in insertions:
            names_with_correctors.extend(insertions[ii])
        names_with_correctors.append(nn)
    if len(line.element_names) in insertions:
        names_with_correctors.extend(insertions[len(line.element_names)])

    corrected_line = xt.Line(env=env, element_names=names_with_correctors)
    corrected_line.particle_ref = line.particle_ref.copy()

    if strengths is not None:
        for key, value in strengths.items():
            env[key] = float(value)

    return corrected_line, corr_names


def _solve_orbit_correctors(line, corr_names, knob_prefix):
    vary_names = []
    for ii in range(1, 5):
        vary_names.extend([f"k0l_{knob_prefix}{ii}", f"k0sl_{knob_prefix}{ii}"])

    opt = line.match(
        solve=False,
        betx=1,
        bety=1,
        only_orbit=True,
        vary=xt.VaryList(vary_names, step=1e-6),
        targets=[
            xt.TargetSet(x=0, px=0, y=0, py=0, at=xt.END),
            xt.TargetSet(x=0, y=0, at=corr_names[1]),
            xt.TargetSet(x=0, y=0, at=corr_names[2]),
        ],
    )
    opt.step(2)

    strengths = {name: float(line.env[name]) for name in vary_names}
    return strengths


def _build_lines(df_fit_pars, p0, shift_x=0.0, shift_y=0.0, strengths_spline=None, strengths_multipole=None):
    env = xt.Environment()
    seq = SplineBorisSequence(
        df_fit_pars=df_fit_pars,
        multipole_order=MULTIPOLE_ORDER,
        steps_per_point=1,
        shift_x=shift_x,
        shift_y=shift_y,
    )
    line_spline_raw = seq.to_line(env=env)
    line_spline_raw.particle_ref = p0.copy()
    line_spline, _ = _insert_corrected_line(
        env=env,
        line=line_spline_raw,
        name_prefix="sp",
        relative_positions_m=CORRECTOR_RELATIVE_POSITIONS_M,
        strengths=strengths_spline,
    )

    line_multipole_raw, _ = build_multipole_kick_undulator(
        env=env,
        p_ref=p0,
        df_fit_pars=df_fit_pars,
        multipole_order=MULTIPOLE_ORDER,
        shift_x=shift_x,
        shift_y=shift_y,
        name_prefix="offset_scan_und",
        multipole_isthick=True,
    )
    line_multipole_raw.particle_ref = p0.copy()
    line_multipole, _ = _insert_corrected_line(
        env=env,
        line=line_multipole_raw,
        name_prefix="mp",
        relative_positions_m=CORRECTOR_RELATIVE_POSITIONS_M,
        strengths=strengths_multipole,
    )
    return line_spline, line_multipole


def _compute_reference_corrector_strengths(df_fit_pars, p0):
    line_spline, line_multipole = _build_lines(df_fit_pars, p0, shift_x=0.0, shift_y=0.0)
    corr_names_spline = [f"sp_corr{i}" for i in range(1, 5)]
    corr_names_multipole = [f"mp_corr{i}" for i in range(1, 5)]

    strengths_spline = _solve_orbit_correctors(line_spline, corr_names_spline, "sp")
    strengths_multipole = _solve_orbit_correctors(line_multipole, corr_names_multipole, "mp")
    return strengths_spline, strengths_multipole


def _track_final_state(line, p0, x0):
    part = p0.copy()
    part.x = float(x0)
    part.y = 0.0
    part.px = 0.0
    part.py = 0.0
    part.zeta = 0.0
    part.delta = 0.0
    line.track(part)
    return {
        "x": float(part.x[0]),
        "y": float(part.y[0]),
        "px": float(part.px[0]),
        "py": float(part.py[0]),
    }


def _run_offset_scan(line_spline, line_multipole, p0):
    offsets_m = 1e-3 * np.arange(11, dtype=float)
    diffs = {key: [] for key in ("x", "y", "px", "py")}

    for x0 in offsets_m:
        final_spline = _track_final_state(line_spline, p0, x0)
        final_multipole = _track_final_state(line_multipole, p0, x0)
        for key in diffs:
            diffs[key].append(final_spline[key] - final_multipole[key])

    for key in diffs:
        diffs[key] = np.asarray(diffs[key])
    return offsets_m, diffs


def _plot_tracking_differences(offsets_m, diffs):
    offsets_mm = 1e3 * offsets_m
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    comp_order = [("x", "dx [m]"), ("y", "dy [m]"), ("px", "dpx [rad]"), ("py", "dpy [rad]")]

    for ax, (comp, ylabel) in zip(axs.ravel(), comp_order):
        ax.plot(offsets_mm, diffs[comp], "o-", lw=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
    axs[1, 0].set_xlabel("Initial x [mm]")
    axs[1, 1].set_xlabel("Initial x [mm]")
    fig.suptitle("Final-state differences: SplineBoris - Multipole")
    fig.tight_layout()
    return fig


def _plot_twiss_bet2(tw_spline, tw_multipole):
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    s_sp = np.asarray(tw_spline.s)
    s_mp = np.asarray(tw_multipole.s)

    axs[0].plot(s_sp, np.asarray(tw_spline.betx2), "-", lw=1.5, label="SplineBoris")
    axs[0].plot(s_mp, np.asarray(tw_multipole.betx2), "--", lw=1.5, label="Multipole")
    axs[0].set_ylabel("betx2 [m]")
    axs[0].grid(True, alpha=0.35)
    axs[0].legend()

    axs[1].plot(s_sp, np.asarray(tw_spline.bety2), "-", lw=1.5, label="SplineBoris")
    axs[1].plot(s_mp, np.asarray(tw_multipole.bety2), "--", lw=1.5, label="Multipole")
    axs[1].set_ylabel("bety2 [m]")
    axs[1].set_xlabel("s [m]")
    axs[1].grid(True, alpha=0.35)
    axs[1].legend()

    fig.suptitle("Twiss4d mode-2 beta comparison")
    fig.tight_layout()
    return fig


def _run_twiss_and_cminus_displacement_scan(
    df_fit_pars, p0, offsets_m, strengths_spline, strengths_multipole, twiss_init_kwargs
):
    twiss_pairs = []
    cminus_spline = []
    cminus_multipole = []
    n_offsets = len(offsets_m)
    print(f"Running Twiss + c_minus scan ({n_offsets} offsets)...")
    for ii, x_shift in enumerate(offsets_m, start=1):
        print(f"  [scan {ii:>2}/{n_offsets}] horizontal offset = {1e3 * x_shift:7.3f} mm")
        line_spline, line_multipole = _build_lines(
            df_fit_pars,
            p0,
            shift_x=float(x_shift),
            shift_y=0.0,
            strengths_spline=strengths_spline,
            strengths_multipole=strengths_multipole,
        )
        line_spline.build_tracker()
        line_multipole.build_tracker()

        tw_spline = line_spline.twiss4d(**twiss_init_kwargs, skip_global_quantities=False)
        tw_multipole = line_multipole.twiss4d(**twiss_init_kwargs, skip_global_quantities=False)
        twiss_pairs.append((float(x_shift), tw_spline, tw_multipole))
        cminus_spline.append(_get_cminus(tw_spline))
        cminus_multipole.append(_get_cminus(tw_multipole))

    return twiss_pairs, np.asarray(cminus_spline), np.asarray(cminus_multipole)


def _plot_twiss_bet2_all_offsets(twiss_pairs):
    offsets_mm = np.asarray([1e3 * x_shift for x_shift, _, _ in twiss_pairs], dtype=float)
    n_offsets = len(twiss_pairs)
    spline_colors = plt.cm.Blues(np.linspace(0.35, 0.95, n_offsets))
    multipole_colors = plt.cm.Oranges(np.linspace(0.35, 0.95, n_offsets))

    fig_betx, ax_betx = plt.subplots(1, 1, figsize=(10, 5))
    fig_bety, ax_bety = plt.subplots(1, 1, figsize=(10, 5))

    for ii, (x_shift, tw_spline, tw_multipole) in enumerate(twiss_pairs):
        offset_mm = 1e3 * x_shift
        label = f"{offset_mm:.3f} mm"
        s_sp = np.asarray(tw_spline.s)
        s_mp = np.asarray(tw_multipole.s)

        ax_betx.plot(
            s_sp,
            np.asarray(tw_spline.betx2),
            "-",
            lw=1.4,
            color=spline_colors[ii],
            label=f"SplineBoris ({label})",
        )
        ax_betx.plot(
            s_mp,
            np.asarray(tw_multipole.betx2),
            "--",
            lw=1.4,
            color=multipole_colors[ii],
            label=f"Multipole ({label})",
        )

        ax_bety.plot(
            s_sp,
            np.asarray(tw_spline.bety2),
            "-",
            lw=1.4,
            color=spline_colors[ii],
            label=f"SplineBoris ({label})",
        )
        ax_bety.plot(
            s_mp,
            np.asarray(tw_multipole.bety2),
            "--",
            lw=1.4,
            color=multipole_colors[ii],
            label=f"Multipole ({label})",
        )

    for ax in (ax_betx, ax_bety):
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("s [m]")

    ax_betx.set_ylabel("betx2 [m]")
    ax_bety.set_ylabel("bety2 [m]")

    if len(offsets_mm) > 0:
        offset_min = np.min(offsets_mm)
        offset_max = np.max(offsets_mm)
        subtitle = (
            f"Horizontal displacement: {offset_min:.3f} to {offset_max:.3f} mm "
            "(light to dark = increasing displacement)"
        )
    else:
        subtitle = "No displacement points available"

    fig_betx.suptitle(f"Twiss4d mode-2 beta_x comparison\n{subtitle}")
    fig_bety.suptitle(f"Twiss4d mode-2 beta_y comparison\n{subtitle}")
    fig_betx.tight_layout()
    fig_bety.tight_layout()
    return fig_betx, fig_bety


def _plot_cminus_vs_displacement(offsets_m, cminus_spline, cminus_multipole):
    offsets_mm = 1e3 * offsets_m
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axs[0].plot(offsets_mm, cminus_spline, "o-", lw=1.5, label="SplineBoris")
    axs[0].plot(offsets_mm, cminus_multipole, "s--", lw=1.5, label="Multipole")
    axs[0].set_ylabel("C^-")
    axs[0].grid(True, alpha=0.35)
    axs[0].legend()

    axs[1].plot(offsets_mm, cminus_spline - cminus_multipole, "o-", lw=1.5)
    axs[1].set_xlabel("Displacement x [mm]")
    axs[1].set_ylabel("dC^- (Spline - Multipole)")
    axs[1].grid(True, alpha=0.35)

    fig.suptitle("Coupling constant C^- vs displacement")
    fig.tight_layout()
    return fig


def _plot_3d_orbit_zero_offset(twiss_pairs):
    if len(twiss_pairs) == 0:
        raise ValueError("Cannot build 3D orbit plot: twiss_pairs is empty.")

    # Use the point closest to zero displacement (nominal case).
    idx0 = int(np.argmin([abs(x_shift) for x_shift, _, _ in twiss_pairs]))
    x_shift, tw_spline, tw_multipole = twiss_pairs[idx0]
    offset_mm = 1e3 * x_shift

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        np.asarray(tw_spline.s),
        1e3 * np.asarray(tw_spline.x),
        1e3 * np.asarray(tw_spline.y),
        "-",
        lw=1.6,
        color="C0",
        label="SplineBoris",
    )
    ax.plot(
        np.asarray(tw_multipole.s),
        1e3 * np.asarray(tw_multipole.x),
        1e3 * np.asarray(tw_multipole.y),
        "--",
        lw=1.6,
        color="C1",
        label="Multipole",
    )
    ax.set_xlabel("s [m]")
    ax.set_ylabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_title(f"3D closed orbit from Twiss (offset={offset_mm:.3f} mm)")
    ax.legend()
    return fig


def _get_cminus(tw):

    s_vect = np.asarray(tw.s)
    mux = np.asarray(tw.mux)
    muy = np.asarray(tw.muy)
    w_matrix = np.asarray(tw.W_matrix)

    w11 = w_matrix[:, 0, 0]
    w13 = w_matrix[:, 0, 2]
    w14 = w_matrix[:, 0, 3]
    w31 = w_matrix[:, 2, 0]
    w32 = w_matrix[:, 2, 1]
    w33 = w_matrix[:, 2, 2]

    c_r1 = np.sqrt(w31**2 + w32**2) / w11
    c_r2 = np.sqrt(w13**2 + w14**2) / w33
    cmin_arr = (
        2
        * np.sqrt(c_r1 * c_r2)
        * np.abs(np.mod(mux[-1], 1) - np.mod(muy[-1], 1))
        / (1 + c_r1 * c_r2)
    )

    circumference = float(np.max(s_vect) - np.min(s_vect))
    if circumference > 0:
        ds = np.diff(s_vect)
        integ = np.sum(0.5 * (cmin_arr[1:] + cmin_arr[:-1]) * ds)
        c_minus = integ / circumference
    else:
        c_minus = np.mean(cmin_arr)

    return float(c_minus)


def _save_results_json(
    output_path,
    offsets_m,
    diffs,
    twiss_pairs,
    cminus_spline,
    cminus_multipole,
    twiss_seed_name,
    twiss_init_kwargs,
    strengths_spline,
    strengths_multipole,
):
    serialized_twiss_scan = []
    for x_shift, tw_spline, tw_multipole in twiss_pairs:
        serialized_twiss_scan.append(
            {
                "offset_m": float(x_shift),
                "offset_mm": float(1e3 * x_shift),
                "splineboris": {
                    "s_m": np.asarray(tw_spline.s, dtype=float).tolist(),
                    "betx2_m": np.asarray(tw_spline.betx2, dtype=float).tolist(),
                    "bety2_m": np.asarray(tw_spline.bety2, dtype=float).tolist(),
                },
                "multipole": {
                    "s_m": np.asarray(tw_multipole.s, dtype=float).tolist(),
                    "betx2_m": np.asarray(tw_multipole.betx2, dtype=float).tolist(),
                    "bety2_m": np.asarray(tw_multipole.bety2, dtype=float).tolist(),
                },
            }
        )

    results = {
        "metadata": {
            "twiss_seed_name": twiss_seed_name,
            "twiss_initial_conditions": {kk: float(vv) for kk, vv in twiss_init_kwargs.items()},
            "multipole_order": int(MULTIPOLE_ORDER),
            "distance_unit_m": float(DISTANCE_UNIT_M),
        },
        "corrector_strengths": {
            "splineboris": {kk: float(vv) for kk, vv in strengths_spline.items()},
            "multipole": {kk: float(vv) for kk, vv in strengths_multipole.items()},
        },
        "tracking_offset_scan": {
            "offsets_m": np.asarray(offsets_m, dtype=float).tolist(),
            "offsets_mm": np.asarray(1e3 * offsets_m, dtype=float).tolist(),
            "diff_spline_minus_multipole": {
                key: np.asarray(values, dtype=float).tolist() for key, values in diffs.items()
            },
        },
        "twiss_scan": serialized_twiss_scan,
        "cminus_scan": {
            "offsets_m": np.asarray(offsets_m, dtype=float).tolist(),
            "splineboris": np.asarray(cminus_spline, dtype=float).tolist(),
            "multipole": np.asarray(cminus_multipole, dtype=float).tolist(),
            "diff_spline_minus_multipole": np.asarray(
                cminus_spline - cminus_multipole, dtype=float
            ).tolist(),
        },
    }

    with open(output_path, "w") as fid:
        json.dump(results, fid, indent=2)


def main():
    p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0_EV)
    df_fit_pars = _fit_field_data(_load_raw_field_data())
    strengths_spline, strengths_multipole = _compute_reference_corrector_strengths(df_fit_pars, p0)
    print("Using fixed corrector strengths from reference (shift_x=0).")
    line_spline, line_multipole = _build_lines(
        df_fit_pars,
        p0,
        shift_x=0.0,
        shift_y=0.0,
        strengths_spline=strengths_spline,
        strengths_multipole=strengths_multipole,
    )

    offsets_m, diffs = _run_offset_scan(line_spline, line_multipole, p0)
    print("Offset scan coverage [m]:")
    print(offsets_m)
    print(f"Number of scan points: {len(offsets_m)}")

    twiss_init_kwargs = TWISS_INITIAL_CONDITIONS[TWISS_SEED_NAME]
    print(f"Using Twiss initial conditions: {TWISS_SEED_NAME}")
    twiss_pairs, cminus_spline, cminus_multipole = _run_twiss_and_cminus_displacement_scan(
        df_fit_pars, p0, offsets_m, strengths_spline, strengths_multipole, twiss_init_kwargs
    )

    fig_diff = _plot_tracking_differences(offsets_m, diffs)
    fig_betx2, fig_bety2 = _plot_twiss_bet2_all_offsets(twiss_pairs)
    fig_cminus = _plot_cminus_vs_displacement(offsets_m, cminus_spline, cminus_multipole)
    fig_orbit_3d = _plot_3d_orbit_zero_offset(twiss_pairs)

    fig_diff.canvas.draw()
    fig_betx2.canvas.draw()
    fig_bety2.canvas.draw()
    fig_cminus.canvas.draw()
    fig_orbit_3d.canvas.draw()

    if SAVE_PLOTS:
        out_dir = Path(__file__).resolve().parent / "_008_plots"
        out_dir.mkdir(exist_ok=True)
        fig_diff.savefig(out_dir / "offset_scan_diffs.png", dpi=150)
        fig_betx2.savefig(out_dir / "twiss_betx2_all_offsets_compare.png", dpi=150)
        fig_bety2.savefig(out_dir / "twiss_bety2_all_offsets_compare.png", dpi=150)
        fig_cminus.savefig(out_dir / "cminus_vs_displacement.png", dpi=150)
        fig_orbit_3d.savefig(out_dir / "twiss_orbit3d_zero_offset.png", dpi=150)
        print(f"Saved plots in {out_dir}")

    if SAVE_RESULTS:
        results_dir = Path(__file__).resolve().parent / RESULTS_DIRNAME
        results_dir.mkdir(exist_ok=True)
        results_file = results_dir / f"offset_scan_seed_{TWISS_SEED_NAME}.json"
        _save_results_json(
            output_path=results_file,
            offsets_m=offsets_m,
            diffs=diffs,
            twiss_pairs=twiss_pairs,
            cminus_spline=cminus_spline,
            cminus_multipole=cminus_multipole,
            twiss_seed_name=TWISS_SEED_NAME,
            twiss_init_kwargs=twiss_init_kwargs,
            strengths_spline=strengths_spline,
            strengths_multipole=strengths_multipole,
        )
        print(f"Saved scan data to {results_file}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_diff)
        plt.close(fig_betx2)
        plt.close(fig_bety2)
        plt.close(fig_cminus)
        plt.close(fig_orbit_3d)


if __name__ == "__main__":
    main()
