"""Run DA, MA, and emittance + spin-polarization studies in sequence.

The emittance and spin-polarization studies share a single tracking run per
case (018_emittance_and_polarization.py), so this driver covers all four
studies for the cost of three tracking passes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from solenoid_params import MAIN_SOLENOID_B0, add_b0_argument, add_max_order_argument

HERE = Path(__file__).resolve().parent
DA_SCRIPT = HERE / "010_dynamic_aperture.py"
MA_SCRIPT = HERE / "009_momentum_acceptance.py"
EMITT_POL_SCRIPT = HERE / "018_emittance_and_polarization.py"
# Mirrors 018_emittance_and_polarization.POL_FIT_START_TURN. Duplicated rather
# than imported so this driver stays import-light (018 pulls in xtrack), and
# used only to fail fast below instead of after DA and MA have already run.
EMITT_POL_FIT_START_TURN_DEFAULT = 2000


def _run_script(script: Path, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(script), *extra_args]
    print(f"\n{'=' * 72}")
    print("Running:", " ".join(cmd))
    print("=" * 72)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run DA (010_dynamic_aperture.py), MA (009_momentum_acceptance.py), "
            "and combined emittance-evolution + spin-polarization "
            "(018_emittance_and_polarization.py) studies in sequence."
        )
    )
    parser.add_argument(
        "--da-only",
        action="store_true",
        help="Run only dynamic-aperture studies.",
    )
    parser.add_argument(
        "--ma-only",
        action="store_true",
        help="Run only momentum-acceptance studies.",
    )
    parser.add_argument(
        "--emitt-only",
        action="store_true",
        help="Run only the emittance + spin-polarization studies.",
    )
    parser.add_argument(
        "--da-cases",
        nargs="+",
        metavar="CASE",
        help=(
            "DA cases for 010 (default: sb_on, varsol_on -- sb_off skipped by "
            "default). Available: sb_on, varsol_on, sb_off"
        ),
    )
    parser.add_argument(
        "--ma-cases",
        nargs="+",
        metavar="CASE",
        help=(
            "MA cases for 009 (default: sb_on, varsol_on -- sb_off skipped by "
            "default). Available: sb_on, varsol_on, sb_off"
        ),
    )
    parser.add_argument(
        "--ma-directions",
        nargs="+",
        metavar="DIRECTION",
        choices=["x_only", "y_only"],
        help=(
            "MA directions for 009 (default: x_only only -- pass "
            "--ma-directions y_only or --ma-directions x_only y_only to "
            "include the y_only scan)."
        ),
    )
    parser.add_argument(
        "--emitt-cases",
        nargs="+",
        metavar="CASE",
        help=(
            "Emittance + polarization cases for 018 (default: sb_on, "
            "varsol_on -- sb_off skipped by default). Available: sb_on, "
            "varsol_on, sb_off"
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help="Shortcut for --da-cases and --emitt-cases.",
    )
    add_b0_argument(parser, default=MAIN_SOLENOID_B0)
    add_max_order_argument(parser)
    parser.add_argument(
        "--n-turns",
        type=int,
        metavar="N",
        help="Number of turns for DA, MA, and emittance + polarization studies.",
    )
    parser.add_argument(
        "--da-n-turns",
        type=int,
        metavar="N",
        help="Number of turns for DA studies (overrides --n-turns).",
    )
    parser.add_argument(
        "--ma-n-turns",
        type=int,
        metavar="N",
        help="Number of turns for MA studies (overrides --n-turns).",
    )
    parser.add_argument(
        "--emitt-n-turns",
        type=int,
        metavar="N",
        help=(
            "Number of turns for the emittance + polarization studies "
            "(overrides --n-turns)."
        ),
    )
    parser.add_argument(
        "--n-part",
        type=int,
        metavar="N",
        help="Number of macroparticles for the emittance + polarization studies.",
    )
    parser.add_argument(
        "--no-emitt",
        action="store_true",
        help=(
            "Skip the emittance + polarization studies (run DA and/or MA only)."
        ),
    )
    parser.add_argument(
        "--sexamp",
        type=float,
        metavar="FACTOR",
        help=(
            "Sextupole amplification knob forwarded to the DA, MA and "
            "emittance+polarization scripts."
        ),
    )
    parser.add_argument(
        "--x-offset",
        type=float,
        metavar="M",
        help="Main detector solenoid x-offset in meters, forwarded to DA/MA scripts.",
    )
    parser.add_argument(
        "--y-offset",
        type=float,
        metavar="M",
        help="Main detector solenoid y-offset in meters, forwarded to DA/MA scripts.",
    )
    parser.add_argument(
        "--extra-sext-strength",
        type=float,
        metavar="K2L",
        help=(
            "Extra thin-sextupole integrated strength (k2*L, in m^-2), "
            "forwarded to DA/MA scripts' --extra-sext-strength (default: "
            "0.0, i.e. off)."
        ),
    )
    emitt_halves = parser.add_mutually_exclusive_group()
    emitt_halves.add_argument(
        "--emitt-no-pol",
        action="store_true",
        help=(
            "In the emittance + polarization stage, run only the emittance half "
            "(forwards --no-pol to 018)."
        ),
    )
    emitt_halves.add_argument(
        "--emitt-pol-only",
        action="store_true",
        help=(
            "In the emittance + polarization stage, run only the spin-polarization "
            "half (forwards --no-emitt to 018)."
        ),
    )
    parser.add_argument(
        "--emitt-pol-fit-start-turn",
        type=int,
        metavar="N",
        help=(
            "First turn included in 018's inline exponential depolarization fit "
            "(018 default: 2000, matching 016's --turn-start)."
        ),
    )
    parser.add_argument(
        "--emitt-bunch-divisor",
        type=float,
        metavar="F",
        help=(
            "Generate 018's bunch at eq_nemitt / F (018 default: 3, i.e. 014's "
            "initial condition; pass 1 for 015's equilibrium bunch)."
        ),
    )
    parser.add_argument(
        "--emitt-seed",
        type=int,
        metavar="N",
        help=(
            "RNG seed forwarded to 018, making its bunch and quantum-radiation "
            "stream reproducible. DA/MA have no seed flag."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after each script (default: save only).",
    )
    args = parser.parse_args()

    only_flags = sum([args.da_only, args.ma_only, args.emitt_only])
    if only_flags > 1:
        raise SystemExit("Choose at most one of --da-only, --ma-only, and --emitt-only.")
    if args.no_emitt and args.emitt_only:
        raise SystemExit("Choose at most one of --no-emitt and --emitt-only.")
    emitt_half_flags = (
        args.emitt_no_pol
        or args.emitt_pol_only
        or args.emitt_pol_fit_start_turn is not None
        or args.emitt_bunch_divisor is not None
        or args.emitt_seed is not None
    )
    if emitt_half_flags and (args.no_emitt or args.da_only or args.ma_only):
        raise SystemExit(
            "The --emitt-* tuning flags only apply when the emittance + "
            "polarization stage runs; drop --no-emitt/--da-only/--ma-only."
        )

    da_cases = args.da_cases or args.cases
    emitt_cases = args.emitt_cases or args.cases
    da_args: list[str] = []
    ma_args: list[str] = []
    emitt_args: list[str] = []
    if da_cases:
        da_args.extend(["--cases", *da_cases])
    if args.ma_cases:
        ma_args.extend(["--cases", *args.ma_cases])
    if args.ma_directions:
        ma_args.extend(["--directions", *args.ma_directions])
    if emitt_cases:
        emitt_args.extend(["--cases", *emitt_cases])
    da_args.extend(["--b0", str(args.b0)])
    ma_args.extend(["--b0", str(args.b0)])
    emitt_args.extend(["--b0", str(args.b0)])
    da_args.extend(["--max-transverse-order", str(args.max_transverse_order)])
    ma_args.extend(["--max-transverse-order", str(args.max_transverse_order)])
    emitt_args.extend(["--max-transverse-order", str(args.max_transverse_order)])
    da_n_turns = args.da_n_turns if args.da_n_turns is not None else args.n_turns
    ma_n_turns = args.ma_n_turns if args.ma_n_turns is not None else args.n_turns
    emitt_n_turns = args.emitt_n_turns if args.emitt_n_turns is not None else args.n_turns
    if da_n_turns is not None:
        da_args.extend(["--n-turns", str(da_n_turns)])
    if ma_n_turns is not None:
        ma_args.extend(["--n-turns", str(ma_n_turns)])
    if emitt_n_turns is not None:
        emitt_args.extend(["--n-turns", str(emitt_n_turns)])
    if args.n_part is not None:
        emitt_args.extend(["--n-part", str(args.n_part)])
    if args.sexamp is not None:
        da_args.extend(["--sexamp", str(args.sexamp)])
        ma_args.extend(["--sexamp", str(args.sexamp)])
        emitt_args.extend(["--sexamp", str(args.sexamp)])
    if args.emitt_no_pol:
        emitt_args.append("--no-pol")
    if args.emitt_pol_only:
        emitt_args.append("--no-emitt")
    if args.emitt_pol_fit_start_turn is not None:
        emitt_args.extend(
            ["--pol-fit-start-turn", str(args.emitt_pol_fit_start_turn)])
    if args.emitt_bunch_divisor is not None:
        emitt_args.extend(["--bunch-emitt-divisor", str(args.emitt_bunch_divisor)])
    if args.emitt_seed is not None:
        emitt_args.extend(["--seed", str(args.emitt_seed)])
    if args.x_offset is not None:
        da_args.extend(["--x-offset", str(args.x_offset)])
        ma_args.extend(["--x-offset", str(args.x_offset)])
    if args.y_offset is not None:
        da_args.extend(["--y-offset", str(args.y_offset)])
        ma_args.extend(["--y-offset", str(args.y_offset)])
    if args.extra_sext_strength is not None:
        da_args.extend(["--extra-sext-strength", str(args.extra_sext_strength)])
        ma_args.extend(["--extra-sext-strength", str(args.extra_sext_strength)])
    if not args.show:
        da_args.append("--no-show")
        ma_args.append("--no-show")
        emitt_args.append("--no-show")

    run_da = not args.ma_only and not args.emitt_only
    run_ma = not args.da_only and not args.emitt_only
    run_emitt = not args.da_only and not args.ma_only and not args.no_emitt

    # 018 rejects a polarization fit window that starts at or past the last tracked
    # turn. Catch that here rather than letting it abort the run after DA and MA
    # have already completed -- short smoke runs (--n-turns 100) hit it every time.
    if run_emitt and not args.emitt_no_pol and emitt_n_turns is not None:
        pol_fit_start = (
            args.emitt_pol_fit_start_turn
            if args.emitt_pol_fit_start_turn is not None
            else EMITT_POL_FIT_START_TURN_DEFAULT
        )
        if pol_fit_start >= emitt_n_turns:
            raise SystemExit(
                f"The emittance+polarization stage would track {emitt_n_turns} "
                f"turns but fit polarization from turn {pol_fit_start}, leaving "
                "nothing to fit. Pass --emitt-pol-fit-start-turn below "
                f"{emitt_n_turns} (short smoke runs typically want something like "
                "20), or --emitt-no-pol to skip the polarization half."
            )

    if run_da:
        _run_script(DA_SCRIPT, da_args)
    if run_ma:
        _run_script(MA_SCRIPT, ma_args)
    if run_emitt:
        _run_script(EMITT_POL_SCRIPT, emitt_args)

    print("\nAll requested studies finished.")


if __name__ == "__main__":
    main()
