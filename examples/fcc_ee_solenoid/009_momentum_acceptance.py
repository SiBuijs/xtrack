from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt

from aperture_grid import initial_conditions_grid
from aperture_study_io import save_ma_study, variant_suffix
from lattice_knobs import set_lattice_knobs

plt.close("all")

HERE = Path(__file__).resolve().parent
SPLINEBORIS_LATTICE_JSON = (
    HERE / "fccee_z_lcc_splineboris_solenoids_coupling_corrected.json"
)
VARSOL_LATTICE_JSON = (
    HERE / "fccee_z_lcc_varsol_solenoids_coupling_corrected.json"
)

IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]
NEMITT_X = 6.33e-5
NEMITT_Y = 1.69e-7
ENERGY_SPREAD = 3.9e-4
NN_Y_R = 25
MAX_Y_R = 25
GLOBAL_XY_LIMIT = 5e-2
DELTA_INITIAL_VALUES = np.linspace(-35 * ENERGY_SPREAD, 35 * ENERGY_SPREAD, 51)
N_TURNS = 10_000

MA_CASES = [
    dict(
        name="sb_on",
        model="SB",
        lattice_json=SPLINEBORIS_LATTICE_JSON,
        with_solenoids=True,
        with_correctors=True,
        title="SplineBoris: solenoids powered + correction scheme",
    ),
    dict(
        name="varsol_on",
        model="VarSol",
        lattice_json=VARSOL_LATTICE_JSON,
        with_solenoids=True,
        with_correctors=True,
        title="VariableSolenoid: solenoids powered + correction scheme",
    ),
    dict(
        name="sb_off",
        model="SB",
        lattice_json=SPLINEBORIS_LATTICE_JSON,
        with_solenoids=False,
        with_correctors=False,
        title="SplineBoris: solenoids unpowered",
    ),
]

MA_CASES_BY_NAME = {case["name"]: case for case in MA_CASES}


def _configure_radiative_tracking(line):
    line.particle_ref.anomalous_magnetic_moment = 0.00115965218128
    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss()


def _plot_momentum_acceptance_figure(out, tt_init, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    lost = out["lost"]
    particles = out["particles"]
    ax.plot(
        tt_init.delta_init[~lost],
        tt_init.x_normalized[~lost],
        ".",
        ms=3,
        label="survived",
    )
    sc = ax.scatter(
        tt_init.delta_init[lost],
        tt_init.x_normalized[lost],
        c=particles.at_turn[lost],
        marker="o",
        s=18,
        label="lost",
    )
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\hat{x}$")
    ax.set_title(
        f"{title}\n"
        f"frac_lost={out['frac_lost']:.4g}, "
        f"at_turn_mean={out['at_turn_mean']:.4g}"
    )
    fig.colorbar(sc, ax=ax, label="lost at turn")
    return fig


def _run_momentum_acceptance(case, *, n_turns, with_progress, sexamp):
    lattice_json = case["lattice_json"]
    title = case["title"]

    print(f"\n=== {title} ===")
    print(f"Loading lattice: {lattice_json.name}")
    env = xt.load(lattice_json)
    line = env.fccee_p_ring
    line.cycle("ipa")
    set_lattice_knobs(
        line,
        with_solenoids=case["with_solenoids"],
        with_correctors=case["with_correctors"],
        sext_amp=sexamp,
    )

    line.discard_tracker()
    line.build_tracker()

    _configure_radiative_tracking(line)

    line.discard_tracker()
    line.build_tracker()
    line.twiss()

    tt_init = initial_conditions_grid(
        study="MA",
        nn_y_r=NN_Y_R,
        max_y_r=MAX_Y_R,
        energy_spread=ENERGY_SPREAD,
        delta_initial_values=DELTA_INITIAL_VALUES,
    )

    particles = line.build_particles(
        method="6d",
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        delta=tt_init.delta_init,
        x_norm=tt_init.x_normalized,
        y_norm=tt_init.y_normalized,
    )

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
    fig = _plot_momentum_acceptance_figure(out, tt_init, title)
    save_ma_study(
        out=out,
        tt_init=tt_init,
        fig=fig,
        model=case["model"],
        with_solenoids=case["with_solenoids"],
        with_correctors=case["with_correctors"],
        n_turns=n_turns,
        global_xy_limit=GLOBAL_XY_LIMIT,
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        nn_y_r=NN_Y_R,
        max_y_r=MAX_Y_R,
        energy_spread=ENERGY_SPREAD,
        delta_initial_values=DELTA_INITIAL_VALUES,
        n_part=len(tt_init),
        variant=variant_suffix(sexamp=sexamp),
        sexamp=sexamp,
    )
    print(
        f"[{title}] Momentum acceptance run complete:"
        f" frac_lost={frac_lost:.6g},"
        f" at_turn_mean={at_turn_mean:.6g}"
    )
    return out


def _select_cases(case_names):
    if not case_names:
        return MA_CASES

    unknown = [name for name in case_names if name not in MA_CASES_BY_NAME]
    if unknown:
        valid = ", ".join(MA_CASES_BY_NAME)
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Choose from: {valid}")

    return [MA_CASES_BY_NAME[name] for name in case_names]


def main():
    parser = argparse.ArgumentParser(
        description="Run momentum-acceptance studies for FCC-ee solenoid lattices."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help=(
            "Cases to run (default: all). "
            "Available: sb_on, varsol_on, sb_off"
        ),
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available cases and exit.",
    )
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
        "--no-show",
        action="store_true",
        help="Skip interactive figure display (data and PDFs are still saved).",
    )
    args = parser.parse_args()

    if args.list_cases:
        for case in MA_CASES:
            print(f"{case['name']}: {case['title']}")
        return

    for case in _select_cases(args.cases):
        _run_momentum_acceptance(
            case, n_turns=args.n_turns, with_progress=1, sexamp=args.sexamp
        )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
