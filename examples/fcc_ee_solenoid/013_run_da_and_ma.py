"""Run DA, MA, and emittance-evolution studies in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from solenoid_params import MAIN_SOLENOID_B0, add_b0_argument

HERE = Path(__file__).resolve().parent
DA_SCRIPT = HERE / "010_dynamic_aperture.py"
MA_SCRIPT = HERE / "009_momentum_acceptance.py"
EMITT_SCRIPT = HERE / "014_emittance_evolution.py"


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
            "and emittance-evolution (014_emittance_evolution.py) studies in sequence."
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
        help="Run only emittance-evolution studies.",
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
            "Emittance cases for 014 (default: sb_on, varsol_on -- sb_off "
            "skipped by default). Available: sb_on, varsol_on, sb_off"
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help="Shortcut for --da-cases (DA only).",
    )
    add_b0_argument(parser, default=MAIN_SOLENOID_B0)
    parser.add_argument(
        "--n-turns",
        type=int,
        metavar="N",
        help="Number of turns for DA, MA, and emittance-evolution studies.",
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
        help="Number of turns for emittance-evolution studies (overrides --n-turns).",
    )
    parser.add_argument(
        "--n-part",
        type=int,
        metavar="N",
        help="Number of macroparticles for emittance-evolution studies.",
    )
    parser.add_argument(
        "--no-emitt",
        action="store_true",
        help="Skip emittance-evolution studies (run DA and/or MA only).",
    )
    parser.add_argument(
        "--sexamp",
        type=float,
        metavar="FACTOR",
        help="Sextupole amplification knob forwarded to DA/MA/emittance scripts.",
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

    if run_da:
        _run_script(DA_SCRIPT, da_args)
    if run_ma:
        _run_script(MA_SCRIPT, ma_args)
    if run_emitt:
        _run_script(EMITT_SCRIPT, emitt_args)

    print("\nAll requested studies finished.")


if __name__ == "__main__":
    main()
