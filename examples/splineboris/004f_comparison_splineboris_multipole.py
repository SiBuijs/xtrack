"""
Compare SplineBoris and multipole-kick undulator models in SLS.

This script builds four machine variants:
1) On-axis SplineBoris
2) Off-axis SplineBoris
3) On-axis multipole kicks
4) Off-axis multipole kicks

For each case it computes full-ring twiss/radiation/spin metrics, prints a compact
summary table and LaTeX table, and generates two custom comparison figures:
- bet2 comparison from twiss arrays (on-axis vs off-axis panels)
- standalone corrected-undulator tracking comparison (orbit + spin overlays)
- standalone corrected-undulator 3D orbit comparison
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xtrack as xt
import scipy as sc
from xtrack._temp.field_fitter import FieldFitter
from xtrack._temp.splineboris_sequence import SplineBorisSequence

from _undulator_multipole_builder import build_multipole_kick_undulator


# -----------------------------
# Top-level configuration
# -----------------------------
E0_EV = 2.7e9
MULTIPOLE_ORDER = 3
DISTANCE_UNIT_M = 0.001
ELECTRON_ANOMALOUS_MAGNETIC_MOMENT = 0.00115965218059
OFFSET_X_M = 5e-4
OFFSET_Y_M = 0
USE_RANDOM_MISALIGNMENT_DISTRIBUTION = True
MISALIGNMENT_SEED = 12345
MISALIGNMENT_X_RANGE_M = (1e-3, 2e-3)
MISALIGNMENT_Y_RANGE_M = (0.0, 0.0)
PLOT_STRIDE = 5
SHOW_PLOTS = True
TRACK_INIT = dict(x=2e-6, px=0.0, y=0.0, py=0.0, zeta=0.0, delta=0.0)
TRACK_INIT_SPIN = dict(
    spin_x=0.50,
    spin_y=0.25,
    spin_z=float(np.sqrt(1.0 - 0.50**2 - 0.25**2)),
)
USE_CACHE_IF_AVAILABLE = True
WRITE_CACHE = True
FORCE_RERUN = False
USE_SHARED_CORRECTOR_SETTINGS = True
USE_SHARED_CORRECTOR_POSITIONS = True
USE_THICK_MULTIPOLE_SLICES = True
CACHE_DIR = Path(__file__).resolve().parent / "_004f_cache"
CACHE_METRICS_CSV = CACHE_DIR / "summary_metrics.csv"
CACHE_PLOT_JSON = CACHE_DIR / "plot_payload.json"
CACHE_VERSION = 28
TICK_FONTSIZE = 16
LABEL_FONTSIZE = 17
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 12

WIGGLER_PLACES = [
    "ars02_uind_0500_1",
    "ars03_uind_0380_1",
    "ars04_uind_0500_1",
    "ars05_uind_0650_1",
    "ars06_uind_0500_1",
    "ars07_uind_0200_1",
    "ars08_uind_0500_1",
    "ars09_uind_0790_1",
    "ars11_uind_0210_1",
    "ars11_uind_0610_1",
    "ars12_uind_0500_1",
]

CORRECTOR_TARGET_POSITIONS = {
    "corr1": 0.02,
    "corr2": 0.10,
    "corr3": "l_wig-0.10",
    "corr4": "l_wig-0.02",
}
CORRECTOR_KEYS = ("corr1", "corr2", "corr3", "corr4")


def _cache_context():
    return {
        "offset_x_m": OFFSET_X_M,
        "offset_y_m": OFFSET_Y_M,
        "use_random_misalignment_distribution": USE_RANDOM_MISALIGNMENT_DISTRIBUTION,
        "misalignment_seed": MISALIGNMENT_SEED,
        "misalignment_x_range_m": list(MISALIGNMENT_X_RANGE_M),
        "misalignment_y_range_m": list(MISALIGNMENT_Y_RANGE_M),
        "track_init": TRACK_INIT,
        "track_init_spin": TRACK_INIT_SPIN,
        "use_shared_corrector_settings": USE_SHARED_CORRECTOR_SETTINGS,
        "use_shared_corrector_positions": USE_SHARED_CORRECTOR_POSITIONS,
        "use_thick_multipole_slices": USE_THICK_MULTIPOLE_SLICES,
        "multipole_order": MULTIPOLE_ORDER,
        "distance_unit_m": DISTANCE_UNIT_M,
    }


def feed_down_sextupole(m_normal, m_skew, shift_x):
    k_fd_normal =   2 * m_normal * shift_x# + m_skew * shift_x
    k_fd_skew   = - 2 * m_skew * shift_x# + 2 * m_normal * shift_x
    return k_fd_normal, k_fd_skew

def theoretical_chromaticity(m_normal, m_skew, k_normal, k_skew, shift_x, beta_x, beta_y, disp_x, disp_y, z):
    k_fd_normal, k_fd_skew = feed_down_sextupole(m_normal, m_skew, shift_x)
    k_total_normal = k_normal + k_fd_normal
    k_total_skew = k_skew + k_fd_skew
    
    term_11 = - k_total_normal + m_normal * disp_x - 2 * m_skew * disp_y
    term_12 = - k_total_skew - 2 * m_normal * disp_y
    term_21 = - k_total_skew + 2 * m_skew * disp_x
    term_22 =   k_total_normal + m_skew * disp_y + 2 * m_normal * disp_x

    xi_x = 1/(4*np.pi) * sc.integrate.trapezoid(term_11 * beta_x + term_21 * beta_y, x=z)
    xi_y = 1/(4*np.pi) * sc.integrate.trapezoid(term_12 * beta_x + term_22 * beta_y, x=z)
    return xi_x, xi_y


def sum_undulator_multipole_chromaticity(
    line,
    twiss,
    undulator_ranges,
    shift_x=0.0,
    shift_x_by_range=None,
    exclude_name_pattern="corr",
):
    """
    Sum the chromaticity contributions from multipole elements inside the undulator
    regions only, using the theoretical formula with Twiss beta_x, beta_y, disp_x, disp_y.

    Uses s as the longitudinal coordinate (z). Only elements whose s falls within
    `undulator_ranges` are included. Excludes elements whose names contain
    `exclude_name_pattern` (e.g. correctors).

    Parameters
    ----------
    line : xtrack.Line
        Line containing the multipole undulator elements (full ring with undulators).
    twiss : Twiss
        Twiss table with s, betx, bety, dx, dy.
    undulator_ranges : list of (s_start, s_end)
        s-ranges of the undulator insertions. Only multipoles within these ranges
        are summed (to isolate the undulator contribution for comparison with
        ring-without-undulators vs ring-with-undulators chromaticity difference).
    shift_x : float, optional
        Fallback transverse shift for feed-down when `shift_x_by_range` is not provided.
    shift_x_by_range : list[float], optional
        Per-undulator transverse shifts in the same order as `undulator_ranges`.
    exclude_name_pattern : str, optional
        Exclude elements whose name contains this string (default "corr").
    """
    s_twiss = np.asarray(twiss.s)
    beta_x = np.asarray(twiss.betx)
    beta_y = np.asarray(twiss.bety)
    disp_x = np.asarray(twiss.dx)
    disp_y = np.asarray(twiss.dy)

    tt = line.get_table()
    xi_x_total = 0.0
    xi_y_total = 0.0
    n_contrib = 0

    for name in line.element_names:
        if exclude_name_pattern and exclude_name_pattern in name:
            continue
        elem = line[name]
        if not isinstance(elem, xt.Multipole):
            continue
        knl = np.atleast_1d(np.asarray(elem.knl))
        ksl = np.atleast_1d(np.asarray(elem.ksl))
        length = float(elem.length)
        if length <= 0:
            continue

        s_start = float(tt["s", name])
        s_center = s_start + length / 2

        i_range = None
        for ii, (ss, se) in enumerate(undulator_ranges):
            if ss <= s_center <= se:
                i_range = ii
                break
        if i_range is None:
            continue
        if shift_x_by_range is not None and i_range < len(shift_x_by_range):
            shift_x_local = float(shift_x_by_range[i_range])
        else:
            shift_x_local = float(shift_x)

        k_normal = knl[1] / length if len(knl) > 1 else 0.0
        k_skew = ksl[1] / length if len(ksl) > 1 else 0.0
        m_normal = knl[2] / length if len(knl) > 2 else 0.0
        m_skew = ksl[2] / length if len(ksl) > 2 else 0.0

        z_elem = np.array([s_start, s_start + length], dtype=float)
        betx_elem = np.interp(z_elem, s_twiss, beta_x)
        bety_elem = np.interp(z_elem, s_twiss, beta_y)
        dx_elem = np.interp(z_elem, s_twiss, disp_x)
        dy_elem = np.interp(z_elem, s_twiss, disp_y)
        xi_x_elem, xi_y_elem = theoretical_chromaticity(
            m_normal=np.full_like(z_elem, m_normal),
            m_skew=np.full_like(z_elem, m_skew),
            k_normal=np.full_like(z_elem, k_normal),
            k_skew=np.full_like(z_elem, k_skew),
            shift_x=shift_x_local,
            beta_x=betx_elem,
            beta_y=bety_elem,
            disp_x=dx_elem,
            disp_y=dy_elem,
            z=z_elem,
        )

        xi_x_total += xi_x_elem
        xi_y_total += xi_y_elem
        n_contrib += 1

    print(
        f"\nSum of chromaticities from {n_contrib} multipole undulator slices:\n"
        f"  xi_x = {xi_x_total:.6g}\n"
        f"  xi_y = {xi_y_total:.6g}"
    )
    return xi_x_total, xi_y_total


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


def _new_env_and_ring(p0):
    madx_file = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "sls.madx"
    env = xt.load(str(madx_file))
    line = env.ring
    line.configure_bend_model(core="mat-kick-mat")
    line.particle_ref = p0.copy()
    return env, line


def _add_corrector_knobs(env, corr_names):
    for name in corr_names.values():
        env[f"k0l_{name}"] = 0.0
        env[f"k0sl_{name}"] = 0.0
        if name not in env:
            env.new(name, xt.Multipole, knl=[f"k0l_{name}"], ksl=[f"k0sl_{name}"])


def _resolve_target_position(expr, l_wig):
    if isinstance(expr, str) and expr.startswith("l_wig-"):
        return l_wig - float(expr.split("-")[1])
    return float(expr)


def _insert_correctors_spline(piecewise_undulator, env, l_wig, corr_names):
    corr_places = []
    for corr_key, target_expr in CORRECTOR_TARGET_POSITIONS.items():
        corr_places.append(env.place(corr_names[corr_key], at=_resolve_target_position(target_expr, l_wig)))
    piecewise_undulator.insert(corr_places, s_tol=5e-3)
    tt = piecewise_undulator.get_table()
    return {cc: float(tt["s", corr_names[cc]]) for cc in CORRECTOR_KEYS}


def _insert_correctors_multipole(piecewise_undulator, env, l_wig, corr_names, ref_positions=None):
    # Optionally enforce exact insertion positions obtained from SplineBoris.
    corr_places = []
    for corr_key, target_expr in CORRECTOR_TARGET_POSITIONS.items():
        if ref_positions is not None:
            s_pos = float(ref_positions[corr_key])
        else:
            s_pos = _resolve_target_position(target_expr, l_wig)
        corr_places.append(env.place(corr_names[corr_key], at=s_pos))
    piecewise_undulator.insert(corr_places, s_tol=5e-3)
    tt = piecewise_undulator.get_table()
    final_positions = {cc: float(tt["s", corr_names[cc]]) for cc in CORRECTOR_KEYS}
    if ref_positions is not None:
        for cc in CORRECTOR_KEYS:
            print(
                f"{cc}: spline s={float(ref_positions[cc]):.6f}, "
                f"multipole s={final_positions[cc]:.6f}"
            )
    return piecewise_undulator


def _corrector_knob_map(corr_names):
    out = {}
    for cc in CORRECTOR_KEYS:
        out[f"k0l_{cc}"] = f"k0l_{corr_names[cc]}"
        out[f"k0sl_{cc}"] = f"k0sl_{corr_names[cc]}"
    return out


def _match_or_apply_correctors(piecewise_undulator, env, corr_names, corrector_settings=None):
    knob_map = _corrector_knob_map(corr_names)
    knob_names = list(knob_map.values())
    if corrector_settings is not None:
        for canonical_name, actual_name in knob_map.items():
            env[actual_name] = float(corrector_settings[canonical_name])
        return {canonical_name: float(env[actual_name])
                for canonical_name, actual_name in knob_map.items()}

    opt = piecewise_undulator.match(
        solve=False,
        betx=0,
        bety=0,
        only_orbit=True,
        include_collective=True,
        vary=xt.VaryList(knob_names, step=1e-6),
        targets=[
            # Enforce closed bump at undulator entrance and exit;
            # corrector locations are free to take whatever values help the edges.
            xt.TargetSet(x=0.0, px=0.0, y=0.0, py=0.0, at=xt.START),
            xt.TargetSet(x=0.0, px=0.0, y=0.0, py=0.0, at=xt.END),
        ],
    )
    opt.step(2)
    return {canonical_name: float(env[actual_name])
            for canonical_name, actual_name in knob_map.items()}


def _build_corrected_undulator(
    env,
    p0,
    df_fit_pars,
    model,
    shift_x,
    shift_y,
    name_prefix,
    corrector_settings=None,
    corrector_positions=None,
):
    corr_names = {cc: f"{name_prefix}_{cc}" for cc in CORRECTOR_KEYS}

    if model == "splineboris":
        seq = SplineBorisSequence(
            df_fit_pars=df_fit_pars,
            multipole_order=MULTIPOLE_ORDER,
            steps_per_point=1,
            shift_x=shift_x,
            shift_y=shift_y,
        )
        seq.element_names = tuple(
            f"{name_prefix}_sb_{ii:03d}" for ii in range(len(seq.element_names))
        )
        und = seq.to_line(env=env)
        l_wig = seq.length
    elif model == "multipole":
        und, l_wig = build_multipole_kick_undulator(
            env=env,
            p_ref=p0,
            df_fit_pars=df_fit_pars,
            multipole_order=MULTIPOLE_ORDER,
            shift_x=shift_x,
            shift_y=shift_y,
            name_prefix=name_prefix,
            multipole_isthick=USE_THICK_MULTIPOLE_SLICES,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    und.particle_ref = p0.copy()
    _add_corrector_knobs(env, corr_names)

    if model == "splineboris":
        used_corrector_positions = _insert_correctors_spline(und, env, l_wig, corr_names)
    else:
        und = _insert_correctors_multipole(
            und, env, l_wig, corr_names, ref_positions=corrector_positions
        )
        tt = und.get_table()
        used_corrector_positions = {cc: float(tt["s", corr_names[cc]]) for cc in CORRECTOR_KEYS}

    used_corrector_settings = _match_or_apply_correctors(
        und, env, corr_names, corrector_settings=corrector_settings
    )
    return und, l_wig, used_corrector_settings, used_corrector_positions


def _insert_wigglers_in_ring(line, corrected_undulator, undulator_length):
    tt = line.get_table()
    undulator_ranges = []
    for ii, wig_place in enumerate(WIGGLER_PLACES):
        und_this = corrected_undulator.replicate(suffix=f"ins{ii}")
        print(f"Inserting corrected undulator {wig_place} at {tt['s', wig_place]}")
        s_start = float(tt["s", wig_place])
        undulator_ranges.append((s_start, s_start + float(undulator_length)))
        line.insert(und_this, anchor="start", at=s_start)
    return undulator_ranges


def _build_misalignment_map(shift_x, shift_y, rng=None):
    shift_map = {}
    if rng is None:
        for wig_place in WIGGLER_PLACES:
            shift_map[wig_place] = (float(shift_x), float(shift_y))
        return shift_map

    x0, x1 = MISALIGNMENT_X_RANGE_M
    y0, y1 = MISALIGNMENT_Y_RANGE_M
    sampled_x = rng.uniform(x0, x1, size=len(WIGGLER_PLACES))
    sampled_y = rng.uniform(y0, y1, size=len(WIGGLER_PLACES))
    for wig_place, sx, sy in zip(WIGGLER_PLACES, sampled_x, sampled_y):
        shift_map[wig_place] = (float(sx), float(sy))
    return shift_map


def _insert_wigglers_with_shift_map(
    line,
    env,
    p0,
    df_fit_pars,
    model,
    case_name,
    shift_map,
    corrector_settings_by_wig=None,
    corrector_positions_by_wig=None,
):
    tt = line.get_table()
    undulator_ranges = []
    first_template = None
    corrector_settings_out_by_wig = {}
    corrector_positions_out_by_wig = {}

    for ii, wig_place in enumerate(WIGGLER_PLACES):
        shift_x, shift_y = shift_map[wig_place]
        settings_in = None
        positions_in = None
        if corrector_settings_by_wig is not None:
            settings_in = corrector_settings_by_wig.get(wig_place)
        if corrector_positions_by_wig is not None:
            positions_in = corrector_positions_by_wig.get(wig_place)

        und_this, l_wig, used_settings, used_positions = _build_corrected_undulator(
            env=env,
            p0=p0,
            df_fit_pars=df_fit_pars,
            model=model,
            shift_x=shift_x,
            shift_y=shift_y,
            name_prefix=f"und_{case_name.lower().replace(' ', '_').replace('-', '_')}_{ii:03d}",
            corrector_settings=settings_in,
            corrector_positions=positions_in,
        )
        if model == "multipole" and settings_in is not None:
            max_abs_delta = max(
                abs(float(used_settings[kk]) - float(settings_in[kk]))
                for kk in settings_in
            )
            print(
                f"  shared-corrector check at {wig_place}: "
                f"max |applied - spline| = {max_abs_delta:.3e}"
            )

        if first_template is None:
            first_template = und_this

        s_start = float(tt["s", wig_place])
        print(
            f"Inserting corrected undulator {wig_place} at {s_start:.6f} "
            f"with shifts x={shift_x:+.3e} m, y={shift_y:+.3e} m"
        )
        undulator_ranges.append((s_start, s_start + float(l_wig)))
        line.insert(und_this, anchor="start", at=s_start)
        corrector_settings_out_by_wig[wig_place] = used_settings
        corrector_positions_out_by_wig[wig_place] = used_positions

    return (
        undulator_ranges,
        first_template,
        corrector_settings_out_by_wig,
        corrector_positions_out_by_wig,
    )


def _run_full_ring_case(
    case_name,
    model,
    shift_x,
    shift_y,
    p0,
    df_fit_pars,
    shift_map,
    corrector_settings_by_wig=None,
    corrector_positions_by_wig=None,
):
    print(f"\n=== Building case: {case_name} ===")
    env, ring = _new_env_and_ring(p0)
    (
        undulator_ranges,
        corrected_und,
        used_corrector_settings_by_shift,
        used_corrector_positions_by_shift,
    ) = _insert_wigglers_with_shift_map(
        line=ring,
        env=env,
        p0=p0,
        df_fit_pars=df_fit_pars,
        model=model,
        case_name=case_name,
        shift_map=shift_map,
        corrector_settings_by_wig=corrector_settings_by_wig,
        corrector_positions_by_wig=corrector_positions_by_wig,
    )

    #ring.configure_radiation(model="mean")
    ring.build_tracker()
    tw = ring.twiss4d(
        radiation_integrals=True,
        spin=True,
        polarization=True,
        radiation_method="full",
    )
    if model == "multipole":
        shift_x_by_range = [float(shift_map[wig][0]) for wig in WIGGLER_PLACES]
        sum_undulator_multipole_chromaticity(
            ring, tw,
            undulator_ranges=undulator_ranges,
            shift_x_by_range=shift_x_by_range,
            exclude_name_pattern=None,
        )
    return {
        "case": case_name,
        "model": model,
        "shift_x": shift_x,
        "shift_y": shift_y,
        "shift_map": shift_map,
        "twiss": tw,
        "corrected_undulator": corrected_und,
        "corrector_settings_by_wig": used_corrector_settings_by_shift,
        "corrector_positions_by_wig": used_corrector_positions_by_shift,
        "undulator_ranges": undulator_ranges,
    }


def _run_reference_ring_case(p0):
    print("\n=== Building case: Reference ring (no inserted undulators) ===")
    _, ring = _new_env_and_ring(p0)
    #ring.configure_radiation(model="mean")
    ring.build_tracker()
    tw = ring.twiss4d(
        radiation_integrals=True,
        spin=True,
        polarization=True,
        radiation_method="full",
    )
    return {
        "case": "Reference (no undulators)",
        "model": "reference",
        "shift_x": 0.0,
        "shift_y": 0.0,
        "shift_map": {},
        "twiss": tw,
        "corrected_undulator": None,
        "corrector_settings_by_wig": {},
        "corrector_positions_by_wig": {},
        "undulator_ranges": [],
    }


def _extract_metrics(case_data):
    tw = case_data["twiss"]
    return {
        "case": case_data["case"],
        "qx": float(tw.qx),
        "qy": float(tw.qy),
        "qs": float(tw.qs),
        "dqx": float(tw.dqx),
        "dqy": float(tw.dqy),
        "J_x": float(tw.rad_int_partition_number_x),
        "J_y": float(tw.rad_int_partition_number_y),
        "J_zeta": float(tw.rad_int_partition_number_zeta),
        "C^-": float(tw.c_minus),
        "spin_polarization_eq": float(tw.spin_polarization_eq),
    }


def _print_metrics_table(rows):
    df = pd.DataFrame(rows)
    tune_cols = {"qx", "qy", "qs"}
    fmt_df = df.copy()
    for col in fmt_df.columns:
        if col == "case":
            continue
        if col in tune_cols:
            fmt_df[col] = fmt_df[col].map(lambda v: _format_sig_with_trailing_zeros(v, sig=5))
        else:
            fmt_df[col] = fmt_df[col].map(lambda v: f"{float(v):.4g}")

    print("\n" + "=" * 100)
    print(f"{len(rows)}-CASE SUMMARY TABLE")
    print("=" * 100)
    print(fmt_df.to_string(index=False))

    latex_df = df.rename(
        columns={
            "case": r"Case",
            "qx": r"$q_x$",
            "qy": r"$q_y$",
            "qs": r"$q_s$",
            "dqx": r"$d q_x$",
            "dqy": r"$d q_y$",
            "J_x": r"$J_x$",
            "J_y": r"$J_y$",
            "J_zeta": r"$J_\zeta$",
            "C^-": r"$C^-$",
            "spin_polarization_eq": r"$P_{\mathrm{eq}}$",
        }
    )
    latex_tune_cols = {r"$q_x$", r"$q_y$", r"$q_s$"}
    latex_table = _dataframe_to_latex_without_deps(latex_df, tune_cols=latex_tune_cols)
    print("\nLaTeX table:\n")
    print(latex_table)


def _format_sig_with_trailing_zeros(value, sig):
    vv = float(value)
    if vv == 0:
        return "0." + "0" * (sig - 1)
    exponent = int(np.floor(np.log10(abs(vv))))
    decimals = max(sig - exponent - 1, 0)
    return f"{vv:.{decimals}f}"


def _dataframe_to_latex_without_deps(df, tune_cols=None):
    tune_cols = tune_cols or set()
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    header = " & ".join(cols) + r" \\"
    body_lines = []
    for _, row in df.iterrows():
        row_cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, str):
                row_cells.append(val)
            else:
                if col in tune_cols:
                    row_cells.append(_format_sig_with_trailing_zeros(val, sig=5))
                else:
                    row_cells.append(f"{float(val):.4g}")
        body_lines.append(" & ".join(row_cells) + r" \\")

    out_lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\hline",
        header,
        r"\hline",
        *body_lines,
        r"\hline",
        r"\end{tabular}",
    ]
    return "\n".join(out_lines)


def _make_twiss_bet2_figure(case_map, component):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

    comparisons = [
        ("on-axis", "On-axis"),
        ("off-axis", "Off-axis"),
    ]
    for ax, (key, title) in zip(axs, comparisons):
        tw_spline = case_map[f"{key}-spline"]
        tw_multipole = case_map[f"{key}-multipole"]
        undulator_ranges = tw_spline.get("undulator_ranges", [])

        s1 = np.asarray(tw_spline["s"])[::PLOT_STRIDE]
        b1 = np.asarray(tw_spline[component])[::PLOT_STRIDE]
        s2 = np.asarray(tw_multipole["s"])[::PLOT_STRIDE]
        b2 = np.asarray(tw_multipole[component])[::PLOT_STRIDE]

        if component == "betx2":
            label_comp = r"\beta_{x,2}"
        else:
            label_comp = r"\beta_{y,2}"

        ax.plot(s1, b1, "-", lw=1.4, label=rf"SplineBoris ${label_comp}$")
        ax.plot(s2, b2, "--", lw=1.4, label=rf"Multipole ${label_comp}$")
        for ii, (ss0, ss1) in enumerate(undulator_ranges):
            lbl = "Undulator region" if ii == 0 else None
            ax.axvspan(ss0, ss1, color="gray", alpha=0.3, lw=0, zorder=0, label=lbl)
        ax.set_title(f"{title}: {component}", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel(r"$s$ [m]", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelleft=True, labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=LEGEND_FONTSIZE)

    if component == "betx2":
        ylabel = r"$\beta_{x,2}$ [m]"
    else:
        ylabel = r"$\beta_{y,2}$ [m]"
    axs[0].set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    axs[1].set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    axs[1].yaxis.set_label_position("right")
    fig.suptitle(f"Twiss mode-2 beta comparison: {component}", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    return fig


def _make_twiss_betx2_on_axis_figure(case_map):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    tw_spline = case_map["on-axis-spline"]
    tw_multipole = case_map["on-axis-multipole"]
    undulator_ranges = tw_spline.get("undulator_ranges", [])

    s1 = np.asarray(tw_spline["s"])[::PLOT_STRIDE]
    b1 = np.asarray(tw_spline["betx2"])[::PLOT_STRIDE]
    s2 = np.asarray(tw_multipole["s"])[::PLOT_STRIDE]
    b2 = np.asarray(tw_multipole["betx2"])[::PLOT_STRIDE]

    ax.plot(s1, b1, "-", lw=1.4, label=r"SplineBoris $\beta_{x,2}$")
    ax.plot(s2, b2, "--", lw=1.4, label=r"Multipole $\beta_{x,2}$")
    for ii, (ss0, ss1) in enumerate(undulator_ranges):
        lbl = "Undulator region" if ii == 0 else None
        ax.axvspan(ss0, ss1, color="gray", alpha=0.3, lw=0, zorder=0, label=lbl)

    ax.set_title(r"On-axis: $\beta_{x,2}$", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(r"$s$ [m]", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"$\beta_{x,2}$ [m]", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    return fig


def _twiss_standalone_spin(corrected_undulator):
    line = corrected_undulator.copy()
    tw = line.twiss4d(
        betx=1,
        bety=1,
        include_collective=True,
        spin=True,
        spin_x=TRACK_INIT_SPIN["spin_x"],
        spin_y=TRACK_INIT_SPIN["spin_y"],
        spin_z=TRACK_INIT_SPIN["spin_z"],
    )
    return {
        "s": np.asarray(tw.s),
        "x": np.asarray(tw.x),
        "y": np.asarray(tw.y),
        "px": np.asarray(tw.px),
        "py": np.asarray(tw.py),
        "spin_x": np.asarray(tw.spin_x),
        "spin_y": np.asarray(tw.spin_y),
        "spin_z": np.asarray(tw.spin_z),
    }


def _make_tracking_orbit_figure(case_map):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    panel_data = [("on-axis", "On-axis"), ("off-axis", "Off-axis")]

    for ax, (key, title) in zip(axs, panel_data):
        tw_spline = case_map[f"{key}-spline"]
        tw_multipole = case_map[f"{key}-multipole"]

        s_sp = np.asarray(tw_spline["s"])
        s_mp = np.asarray(tw_multipole["s"])
        x_sp = np.asarray(tw_spline["x"])
        x_mp = np.asarray(tw_multipole["x"])
        y_sp = np.asarray(tw_spline["y"])
        y_mp = np.asarray(tw_multipole["y"])

        ax.plot(s_sp, x_sp, "-", lw=1.3, label=r"SplineBoris $x$")
        ax.plot(s_mp, x_mp, "--", lw=1.3, label=r"Multipole $x$")
        ax.plot(s_sp, y_sp, "-", lw=1.3, label=r"SplineBoris $y$")
        ax.plot(s_mp, y_mp, "--", lw=1.3, label=r"Multipole $y$")
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel(r"$s$ [m]", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelleft=True, labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), fontsize=LEGEND_FONTSIZE, ncol=2)

    axs[0].set_ylabel(r"$x, y$ [m]", fontsize=LABEL_FONTSIZE)
    axs[1].set_ylabel(r"$x, y$ [m]", fontsize=LABEL_FONTSIZE)
    axs[1].yaxis.set_label_position("right")
    fig.suptitle("Standalone corrected-undulator tracking: orbit comparison", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    return fig


def _make_tracking_spin_component_figure(case_map):
    fig, axs = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    panel_data = [("on-axis", "On-axis"), ("off-axis", "Off-axis")]
    component_data = [
        ("spin_x", r"$S_x$"),
        ("spin_y", r"$S_y$"),
        ("spin_z", r"$S_z$"),
    ]

    for col, (key, title) in enumerate(panel_data):
        tw_spline = case_map[f"{key}-spline"]
        tw_multipole = case_map[f"{key}-multipole"]
        s_sp = np.asarray(tw_spline["s"])
        s_mp = np.asarray(tw_multipole["s"])

        for row, (comp, comp_label) in enumerate(component_data):
            ax = axs[row, col]
            spin_sp = np.asarray(tw_spline[comp])
            spin_mp = np.asarray(tw_multipole[comp])
            ax.plot(s_sp, spin_sp, "-", lw=1.2, label="SplineBoris")
            ax.plot(s_mp, spin_mp, "--", lw=1.2, label="Multipole")
            ax.grid(True, alpha=0.35)
            ax.set_ylabel(comp_label, fontsize=LABEL_FONTSIZE)
            ax.set_xlabel(r"$s$ [m]", fontsize=LABEL_FONTSIZE)
            ax.tick_params(axis="x", labelbottom=True, labelsize=TICK_FONTSIZE)
            ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
            if row == 0:
                ax.set_title(title, fontsize=TITLE_FONTSIZE)
            if col == 1:
                ax.yaxis.set_label_position("right")

    axs[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), fontsize=LEGEND_FONTSIZE, ncol=2)
    axs[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), fontsize=LEGEND_FONTSIZE, ncol=2)
    fig.suptitle("Standalone corrected-undulator twiss: spin component comparison", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    return fig


def _make_tracking_orbit_3d_figure(case_map):
    fig = plt.figure(figsize=(13, 5))
    axs = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    panel_data = [("on-axis", "On-axis"), ("off-axis", "Off-axis")]

    for ax, (key, title) in zip(axs, panel_data):
        tw_spline = case_map[f"{key}-spline"]
        tw_multipole = case_map[f"{key}-multipole"]
        s_sp = np.asarray(tw_spline["s"])
        s_mp = np.asarray(tw_multipole["s"])
        x_sp = np.asarray(tw_spline["x"])
        x_mp = np.asarray(tw_multipole["x"])
        y_sp = np.asarray(tw_spline["y"])
        y_mp = np.asarray(tw_multipole["y"])

        ax.plot(s_sp, x_sp, y_sp, "-", lw=1.2, label="SplineBoris")
        ax.plot(s_mp, x_mp, y_mp, "--", lw=1.2, label="Multipole")
        ax.set_title(f"{title} 3D orbit", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel(r"$s$ [m]", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(r"$x$ [m]", fontsize=LABEL_FONTSIZE)
        ax.set_zlabel(r"$y$ [m]", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="z", labelsize=TICK_FONTSIZE)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), fontsize=LEGEND_FONTSIZE)

    fig.suptitle("Standalone corrected-undulator 3D orbit", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    return fig


def _make_plot_payload(case_map):
    twiss_payload = {}
    standalone_spin_twiss_payload = {}
    for key, case_data in case_map.items():
        tw = case_data["twiss"]
        twiss_payload[key] = {
            "s": np.asarray(tw.s).tolist(),
            "betx2": np.asarray(tw.betx2).tolist(),
            "bety2": np.asarray(tw.bety2).tolist(),
            "undulator_ranges": [[float(a), float(b)] for a, b in case_data["undulator_ranges"]],
        }
        standalone_spin_twiss_payload[key] = {
            kk: np.asarray(vv).tolist()
            for kk, vv in _twiss_standalone_spin(case_data["corrected_undulator"]).items()
        }
    return {
        "twiss": twiss_payload,
        "standalone_spin_twiss": standalone_spin_twiss_payload,
    }


def _print_standalone_final_rel_differences(standalone_spin_twiss):
    print("\nStandalone corrected-undulator final-state relative differences")
    print("(Multipole vs SplineBoris)")
    eps = 1e-30
    for scenario in ("on-axis", "off-axis"):
        tw_spline = standalone_spin_twiss[f"{scenario}-spline"]
        tw_multipole = standalone_spin_twiss[f"{scenario}-multipole"]
        print(f"  {scenario}:")
        for comp in ("x", "y", "px", "py"):
            spline_val = float(np.asarray(tw_spline[comp])[-1])
            multipole_val = float(np.asarray(tw_multipole[comp])[-1])
            abs_diff = abs(multipole_val - spline_val)
            rel_diff = abs_diff / max(abs(spline_val), eps)
            print(
                f"    {comp:>2}: "
                f"spline={spline_val:+.6e}, multipole={multipole_val:+.6e}, "
                f"abs_diff={abs_diff:.3e}, rel_diff={rel_diff:.3e}"
            )


def _save_cache(metrics_rows, payload):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(CACHE_METRICS_CSV, index=False)
    payload_to_save = {
        "_cache_version": CACHE_VERSION,
        "context": _cache_context(),
        "data": payload,
    }
    with open(CACHE_PLOT_JSON, "w", encoding="utf-8") as fid:
        json.dump(payload_to_save, fid)
    print(f"Saved cache files in {CACHE_DIR}")


def _load_cache():
    metrics_rows = pd.read_csv(CACHE_METRICS_CSV).to_dict(orient="records")
    with open(CACHE_PLOT_JSON, "r", encoding="utf-8") as fid:
        payload_raw = json.load(fid)
    if payload_raw.get("_cache_version", -1) != CACHE_VERSION:
        raise ValueError("Cache version mismatch. Please rerun simulation to refresh cache.")
    if payload_raw.get("context") != _cache_context():
        raise ValueError("Cache context mismatch (configuration changed). Please rerun simulation.")
    payload = payload_raw["data"]
    print(f"Loaded cached results from {CACHE_DIR}")
    return metrics_rows, payload


def main():
    can_load_cache = CACHE_METRICS_CSV.exists() and CACHE_PLOT_JSON.exists()
    if USE_CACHE_IF_AVAILABLE and can_load_cache and not FORCE_RERUN:
        try:
            metrics_rows, payload = _load_cache()
        except Exception as err:
            print(f"Cache load skipped: {err}")
            metrics_rows = None
            payload = None
    else:
        metrics_rows = None
        payload = None

    if metrics_rows is None or payload is None:
        p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0_EV)
        p0.anomalous_magnetic_moment = ELECTRON_ANOMALOUS_MAGNETIC_MOMENT
        df_fit_pars = _fit_field_data(_load_raw_field_data())
        reference_case = _run_reference_ring_case(p0)

        if USE_RANDOM_MISALIGNMENT_DISTRIBUTION:
            rng = np.random.default_rng(MISALIGNMENT_SEED)
            off_axis_shift_map = _build_misalignment_map(OFFSET_X_M, OFFSET_Y_M, rng=rng)
            print("\nUsing per-undulator random misalignments (off-axis cases):")
            for wig_place in WIGGLER_PLACES:
                sx, sy = off_axis_shift_map[wig_place]
                print(f"  {wig_place}: x={sx:+.3e} m, y={sy:+.3e} m")
            if any(abs(off_axis_shift_map[ww][1]) > 0 for ww in WIGGLER_PLACES):
                print(
                    "Note: the multipole proxy is built from field samples on y=0 "
                    "(x-scan only). Non-zero y offsets are expected to increase "
                    "SplineBoris-vs-Multipole mismatch in y."
                )
        else:
            off_axis_shift_map = _build_misalignment_map(OFFSET_X_M, OFFSET_Y_M, rng=None)
        on_axis_shift_map = _build_misalignment_map(0.0, 0.0, rng=None)

        case_specs = [
            ("on-axis", "on-axis-spline", "On-axis SplineBoris", "splineboris", 0.0, 0.0, on_axis_shift_map),
            ("off-axis", "off-axis-spline", "Off-axis SplineBoris", "splineboris", OFFSET_X_M, OFFSET_Y_M, off_axis_shift_map),
            ("on-axis", "on-axis-multipole", "On-axis Multipole-kicks", "multipole", 0.0, 0.0, on_axis_shift_map),
            ("off-axis", "off-axis-multipole", "Off-axis Multipole-kicks", "multipole", OFFSET_X_M, OFFSET_Y_M, off_axis_shift_map),
        ]

        all_cases = []
        case_map = {}
        shared_correctors_by_case = {}
        shared_positions_by_case = {}
        for scenario_key, key, case_name, model, shift_x, shift_y, shift_map in case_specs:
            settings_for_case = None
            positions_for_case = None
            if model == "multipole":
                if USE_SHARED_CORRECTOR_SETTINGS:
                    settings_for_case = shared_correctors_by_case.get(scenario_key)
                if USE_SHARED_CORRECTOR_POSITIONS:
                    positions_for_case = shared_positions_by_case.get(scenario_key)
                if USE_SHARED_CORRECTOR_SETTINGS and settings_for_case is None:
                    raise RuntimeError(
                        f"Missing shared SplineBoris corrector settings for scenario `{scenario_key}`"
                    )
                if USE_SHARED_CORRECTOR_POSITIONS and positions_for_case is None:
                    raise RuntimeError(
                        f"Missing shared SplineBoris corrector positions for scenario `{scenario_key}`"
                    )
            case_data = _run_full_ring_case(
                case_name=case_name,
                model=model,
                shift_x=shift_x,
                shift_y=shift_y,
                p0=p0,
                df_fit_pars=df_fit_pars,
                shift_map=shift_map,
                corrector_settings_by_wig=settings_for_case,
                corrector_positions_by_wig=positions_for_case,
            )
            if model == "splineboris":
                if USE_SHARED_CORRECTOR_SETTINGS:
                    shared_correctors_by_case[scenario_key] = dict(
                        case_data["corrector_settings_by_wig"]
                    )
                if USE_SHARED_CORRECTOR_POSITIONS:
                    shared_positions_by_case[scenario_key] = dict(
                        case_data["corrector_positions_by_wig"]
                    )
            all_cases.append(case_data)
            case_map[key] = case_data

        metrics_rows = [_extract_metrics(reference_case)] + [_extract_metrics(cc) for cc in all_cases]
        payload = _make_plot_payload(case_map)
        if WRITE_CACHE:
            _save_cache(metrics_rows, payload)

    _print_metrics_table(metrics_rows)
    _print_standalone_final_rel_differences(payload["standalone_spin_twiss"])
    fig_twiss_betx2 = _make_twiss_bet2_figure(payload["twiss"], component="betx2")
    fig_twiss_bety2 = _make_twiss_bet2_figure(payload["twiss"], component="bety2")
    fig_twiss_betx2_on_axis = _make_twiss_betx2_on_axis_figure(payload["twiss"])
    fig_track_orbit = _make_tracking_orbit_figure(payload["standalone_spin_twiss"])
    fig_track_spin = _make_tracking_spin_component_figure(payload["standalone_spin_twiss"])
    fig_track_orbit_3d = _make_tracking_orbit_3d_figure(payload["standalone_spin_twiss"])

    # Render figures in all modes to validate plotting paths even when not showing GUI.
    fig_twiss_betx2.canvas.draw()
    fig_twiss_bety2.canvas.draw()
    fig_twiss_betx2_on_axis.canvas.draw()
    fig_track_orbit.canvas.draw()
    fig_track_spin.canvas.draw()
    fig_track_orbit_3d.canvas.draw()

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_twiss_betx2)
        plt.close(fig_twiss_bety2)
        plt.close(fig_twiss_betx2_on_axis)
        plt.close(fig_track_orbit)
        plt.close(fig_track_spin)
        plt.close(fig_track_orbit_3d)


if __name__ == "__main__":
    main()
