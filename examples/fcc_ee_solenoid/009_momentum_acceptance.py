from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt

from aperture_grid import initial_conditions_grid
from aperture_study_io import save_ma_study, variant_suffix
from lattice_knobs import (
    install_extra_sextupole,
    robust_twiss,
    set_lattice_knobs,
    set_solenoid_offset,
    zero_sextupole_window,
)
from solenoid_params import (
    MAIN_SOLENOID_B0,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    order_tag,
)

plt.close("all")

HERE = Path(__file__).resolve().parent


def _lattice_paths(tag, order_tag_str=""):
    # order_tag_str only applies to the SplineBoris path -- VariableSolenoid
    # is linear-only and has no transverse-order knob (see 00_overview.md).
    return (
        HERE / f"fccee_z_lcc_splineboris_solenoids_coupling_corrected_{tag}{order_tag_str}.json",
        HERE / f"fccee_z_lcc_varsol_solenoids_coupling_corrected_{tag}.json",
    )

NEMITT_X = 6.33e-5
NEMITT_Y = 1.69e-7
ENERGY_SPREAD = 3.9e-4
NN_Y_R = 25
MAX_Y_R = 35
# Same self-chosen (not sigma_x/sigma_y-derived) amplification used for
# 010_dynamic_aperture.py's y-axis: sigma_y is far smaller than sigma_x, so a
# y_only scan capped at the same amplitude (in sigma units) as x_only would
# under-explore where the real y aperture boundary sits. Applied only to the
# y_only direction's amplitude cap via MA_DIRECTIONS below; x_only is
# unaffected.
Y_AXIS_SCAN_FACTOR = 6.0
GLOBAL_XY_LIMIT = 5e-2
DELTA_INITIAL_VALUES = np.linspace(-35 * ENERGY_SPREAD, 35 * ENERGY_SPREAD, 51)
N_TURNS = 10_000

def _build_ma_cases(tag, order_tag_str=""):
    """Build the MA_CASES list for a given field_tag (e.g. '2T', '3T') and
    order_tag (e.g. '', '_o2' -- SplineBoris-only, see _lattice_paths)."""
    splineboris_json, varsol_json = _lattice_paths(tag, order_tag_str)
    sb_tag = f"{tag}{order_tag_str}"
    return [
        dict(
            name="sb_on",
            model="SB",
            lattice_json=splineboris_json,
            with_solenoids=True,
            with_correctors=True,
            title=f"SplineBoris ({sb_tag}): solenoids powered + correction scheme",
        ),
        dict(
            name="varsol_on",
            model="VarSol",
            lattice_json=varsol_json,
            with_solenoids=True,
            with_correctors=True,
            title=f"VariableSolenoid ({tag}): solenoids powered + correction scheme",
        ),
        dict(
            name="sb_off",
            model="SB",
            lattice_json=splineboris_json,
            with_solenoids=False,
            with_correctors=False,
            title=f"SplineBoris ({sb_tag}): solenoids unpowered",
        ),
    ]


# sb_off is the bare-machine baseline: it has already been run once and does
# not depend on solenoid/correction changes, so skip it by default (pass
# --cases sb_off explicitly to rerun it).
DEFAULT_MA_CASE_NAMES = ["sb_on", "varsol_on"]

# Pure-plane MA scans: theta=0 is on-axis x (y_normalized=0), theta=pi/2 is
# on-axis y (x_normalized=0). x_only runs by default; y_only must be opted
# into via --directions (see --directions).
MA_DIRECTIONS = {
    "x_only": dict(
        theta=0.0, axis_key="x_normalized", axis_symbol=r"\hat{x}", max_y_r_factor=1.0
    ),
    "y_only": dict(
        theta=np.pi / 2,
        axis_key="y_normalized",
        axis_symbol=r"\hat{y}",
        max_y_r_factor=Y_AXIS_SCAN_FACTOR,
    ),
}


def _configure_radiative_tracking(line, radiation="mean"):
    """Sets up the deterministic (mean) radiation model used for RF
    compensation and the closed-orbit twiss below -- twiss cannot run once
    a stochastic model is active. Callers that want quantum tracking switch
    to it explicitly afterwards, right before building particles/tracking
    (see _switch_to_quantum_if_requested below), same mean-then-quantum
    idiom as 014/015's radiative tracking."""
    line.particle_ref.anomalous_magnetic_moment = 0.00115965218128
    if radiation == "off":
        line.configure_radiation(model=None)
        return
    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss()


def _switch_to_quantum_if_requested(line, radiation):
    if radiation == "quantum":
        line.configure_radiation(model="quantum")


def _plot_momentum_acceptance_figure(out, tt_init, title, direction):
    info = MA_DIRECTIONS[direction]
    amplitude = getattr(tt_init, info["axis_key"])

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    lost = out["lost"]
    particles = out["particles"]
    ax.plot(
        tt_init.delta_init[~lost],
        amplitude[~lost],
        ".",
        ms=3,
        label="survived",
    )
    sc = ax.scatter(
        tt_init.delta_init[lost],
        amplitude[lost],
        c=particles.at_turn[lost],
        marker="o",
        s=18,
        label="lost",
    )
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(fr"${info['axis_symbol']}$")
    ax.set_title(
        f"{title} ({'y' if direction == 'y_only' else 'x'}-only)\n"
        f"frac_lost={out['frac_lost']:.4g}, "
        f"at_turn_mean={out['at_turn_mean']:.4g}"
    )
    fig.colorbar(sc, ax=ax, label="lost at turn")
    return fig


def _run_momentum_acceptance(
    case, direction, *, n_turns, with_progress, sexamp, x_offset, y_offset,
    extra_sext_strength=0.0, sext_window_s_min=None, sext_window_s_max=None,
    sext_window_order=2, radiation="mean", tag,
):
    lattice_json = case["lattice_json"]
    title = case["title"]
    theta = MA_DIRECTIONS[direction]["theta"]
    max_y_r = MAX_Y_R * MA_DIRECTIONS[direction]["max_y_r_factor"]

    if not lattice_json.exists():
        raise SystemExit(
            f"Missing lattice file: {lattice_json.name}\n"
            f"Build it first for tag {tag!r} via "
            "004a_build_and_check_solenoids.py -> "
            "004b_install_solenoids_in_fcc_ring.py / "
            "004b_install_varsol_solenoids_in_fcc_ring.py -> "
            "004c_correct_solenoids_in_fcc_ring.py (each accepts --b0 and, "
            "for the SplineBoris path, --max-transverse-order)."
        )

    print(f"\n=== {title} ({direction}) ===")
    print(f"Loading lattice: {lattice_json.name}")
    print(f"Amplitude range: max_y_r={max_y_r:g} (in the scanned plane's own sigma units)")
    env = xt.load(lattice_json)
    line = env.fccee_p_ring
    line.cycle("ipa")
    set_lattice_knobs(
        line,
        with_solenoids=case["with_solenoids"],
        with_correctors=case["with_correctors"],
        sext_amp=sexamp,
    )
    set_solenoid_offset(line, x_offset=x_offset, y_offset=y_offset)
    install_extra_sextupole(line, k2l=extra_sext_strength)
    if sext_window_s_min is not None:
        n_zeroed, covered_s_start, covered_s_end = zero_sextupole_window(
            line, s_min=sext_window_s_min, s_max=sext_window_s_max,
            order=sext_window_order)
        print(
            f"  Zeroed order-{sext_window_order} sextupole-like content in "
            f"{n_zeroed} slices across 4 IPs, covering s in "
            f"[{covered_s_start:.4f}, {covered_s_end:.4f}] m from each IP "
            f"(requested [{sext_window_s_min:g}, {sext_window_s_max:g}] m)."
        )

    line.discard_tracker()
    line.build_tracker()

    _configure_radiative_tracking(line, radiation=radiation)

    line.discard_tracker()
    line.build_tracker()
    # robust_twiss falls back to a co-guess continuation (ramping sext_amp
    # from 1.0) if the direct closed-orbit search fails, e.g. for large
    # --sexamp values -- see lattice_knobs.robust_twiss.
    tw_co = robust_twiss(line)

    tt_init = initial_conditions_grid(
        study="MA",
        nn_y_r=NN_Y_R,
        max_y_r=max_y_r,
        energy_spread=ENERGY_SPREAD,
        delta_initial_values=DELTA_INITIAL_VALUES,
        min_x_theta=theta,
        max_x_theta=theta,
    )

    particles = line.build_particles(
        method="6d",
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        delta=tt_init.delta_init,
        x_norm=tt_init.x_normalized,
        y_norm=tt_init.y_normalized,
        particle_on_co=tw_co.particle_on_co,
    )

    _switch_to_quantum_if_requested(line, radiation)

    line.discard_tracker()
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads="auto"))

    line.config.XTRACK_GLOBAL_XY_LIMIT = GLOBAL_XY_LIMIT
    line.track(particles, num_turns=n_turns, with_progress=with_progress)
    particles.sort(interleave_lost_particles=True)

    lost = particles.state <= 0
    frac_lost = lost.sum() / len(lost)
    at_turn_mean = particles.at_turn.mean()

    out = {
        "particles": particles,
        "lost": lost,
        "frac_lost": frac_lost,
        "at_turn_mean": at_turn_mean,
        "tt_init": tt_init,
    }
    fig = _plot_momentum_acceptance_figure(out, tt_init, title, direction)
    save_ma_study(
        out=out,
        tt_init=tt_init,
        fig=fig,
        model=case["model"],
        direction=direction,
        with_solenoids=case["with_solenoids"],
        with_correctors=case["with_correctors"],
        n_turns=n_turns,
        global_xy_limit=GLOBAL_XY_LIMIT,
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        nn_y_r=NN_Y_R,
        max_y_r=max_y_r,
        energy_spread=ENERGY_SPREAD,
        delta_initial_values=DELTA_INITIAL_VALUES,
        n_part=len(tt_init),
        radiation=radiation,
        variant=variant_suffix(
            sexamp=sexamp, x_offset=x_offset, y_offset=y_offset,
            extra_sext_strength=extra_sext_strength,
            sext_window_s_min=sext_window_s_min,
            sext_window_s_max=sext_window_s_max,
            sext_window_order=sext_window_order,
        ),
        sexamp=sexamp,
        x_offset=x_offset,
        y_offset=y_offset,
        extra_sext_strength=extra_sext_strength,
        sext_window_s_min=sext_window_s_min,
        sext_window_s_max=sext_window_s_max,
        sext_window_order=sext_window_order,
        field_tag=tag,
    )
    print(
        f"[{title} ({direction})] Momentum acceptance run complete:"
        f" frac_lost={frac_lost:.6g},"
        f" at_turn_mean={at_turn_mean:.6g}"
    )
    return out


def _select_cases(case_names, cases_by_name):
    if not case_names:
        return [cases_by_name[name] for name in DEFAULT_MA_CASE_NAMES]

    unknown = [name for name in case_names if name not in cases_by_name]
    if unknown:
        valid = ", ".join(cases_by_name)
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Choose from: {valid}")

    return [cases_by_name[name] for name in case_names]


def _select_directions(direction_names):
    if not direction_names:
        return ["x_only"]

    unknown = [name for name in direction_names if name not in MA_DIRECTIONS]
    if unknown:
        valid = ", ".join(MA_DIRECTIONS)
        raise SystemExit(
            f"Unknown direction(s): {', '.join(unknown)}. Choose from: {valid}"
        )

    return list(direction_names)


def main():
    parser = argparse.ArgumentParser(
        description="Run momentum-acceptance studies for FCC-ee solenoid lattices."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help=(
            "Cases to run (default: sb_on, varsol_on -- sb_off is skipped by "
            "default since the bare-machine baseline doesn't change; pass "
            "--cases sb_off explicitly to include it). "
            "Available: sb_on, varsol_on, sb_off"
        ),
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available cases and exit.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        metavar="DIRECTION",
        choices=list(MA_DIRECTIONS),
        help=(
            "Momentum-acceptance directions to run (default: x_only only -- "
            "pass --directions y_only or --directions x_only y_only to "
            "include the y_only scan). Available: x_only, y_only"
        ),
    )
    add_b0_argument(parser, default=MAIN_SOLENOID_B0)
    add_max_order_argument(parser)
    parser.add_argument(
        "--n-turns",
        type=int,
        default=N_TURNS,
        metavar="N",
        help=f"Number of turns to track (default: {N_TURNS}).",
    )
    parser.add_argument(
        "--sexamp",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Sextupole amplification knob (default: 1.0).",
    )
    parser.add_argument(
        "--x-offset",
        type=float,
        default=0.0,
        metavar="M",
        help="Main detector solenoid x-offset in meters (default: 0.0).",
    )
    parser.add_argument(
        "--y-offset",
        type=float,
        default=0.0,
        metavar="M",
        help="Main detector solenoid y-offset in meters (default: 0.0).",
    )
    parser.add_argument(
        "--extra-sext-strength",
        type=float,
        default=0.0,
        metavar="K2L",
        help=(
            "Integrated strength (k2*L, in m^-2) of an extra thin sextupole "
            "inserted into each IP's main detector solenoid, between two "
            "SplineBoris slices at s ~= -1.223 m from the IP (default: 0.0, "
            "i.e. off -- the lattice is unmodified unless this is nonzero)."
        ),
    )
    parser.add_argument(
        "--sext-window-s-min",
        type=float,
        default=None,
        metavar="M",
        help=(
            "Start (m, distance from the IP) of an s-window in which the "
            "sextupole-like SplineBoris content is zeroed in each IP's main "
            "detector solenoid, e.g. for probing which part of the solenoid "
            "dominates a beta_x-weighted d^2Bx/dx^2 driving-term integral "
            "(see 004e_sextupole_window_study.py). Must be given together "
            "with --sext-window-s-max. Default: None, i.e. off."
        ),
    )
    parser.add_argument(
        "--sext-window-s-max",
        type=float,
        default=None,
        metavar="M",
        help="End (m, distance from the IP) of the --sext-window-s-min window.",
    )
    parser.add_argument(
        "--sext-window-order",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Transverse-x Hermite order zeroed within the --sext-window-s-min"
            "/--sext-window-s-max window (default: 2, the sextupole-like term)."
        ),
    )
    parser.add_argument(
        "--radiation",
        choices=["off", "mean", "quantum"],
        default="mean",
        help=(
            "Radiation model used for tracking (default: mean, deterministic "
            "damping only). 'off' disables radiation and RF compensation "
            "entirely. 'quantum' adds stochastic photon-emission kicks on "
            "top of the mean-computed RF compensation, letting momentum "
            "acceptance be studied under quantum excitation."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive figure display (data and PDFs are still saved).",
    )
    args = parser.parse_args()

    if (args.sext_window_s_min is None) != (args.sext_window_s_max is None):
        raise SystemExit(
            "--sext-window-s-min and --sext-window-s-max must be given "
            "together."
        )

    tag = field_tag(args.b0)
    order_tag_str = order_tag(args.max_transverse_order)
    ma_cases = _build_ma_cases(tag, order_tag_str)
    ma_cases_by_name = {case["name"]: case for case in ma_cases}

    if args.list_cases:
        print(
            f"Field tag: {tag}{order_tag_str} (--b0 {args.b0:g}, "
            f"--max-transverse-order {args.max_transverse_order})")
        for case in ma_cases:
            print(f"{case['name']}: {case['title']}")
        return

    for case in _select_cases(args.cases, ma_cases_by_name):
        for direction in _select_directions(args.directions):
            _run_momentum_acceptance(
                case,
                direction,
                n_turns=args.n_turns,
                with_progress=1,
                sexamp=args.sexamp,
                x_offset=args.x_offset,
                y_offset=args.y_offset,
                extra_sext_strength=args.extra_sext_strength,
                sext_window_s_min=args.sext_window_s_min,
                sext_window_s_max=args.sext_window_s_max,
                sext_window_order=args.sext_window_order,
                radiation=args.radiation,
                tag=f"{tag}{order_tag_str}",
            )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
