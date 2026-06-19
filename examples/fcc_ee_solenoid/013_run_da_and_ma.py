"""Run dynamic-aperture (010) and momentum-acceptance (009) studies in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DA_SCRIPT = HERE / "010_dynamic_aperture.py"
MA_SCRIPT = HERE / "009_momentum_acceptance.py"


def _run_script(script: Path, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(script), *extra_args]
    print(f"\n{'=' * 72}")
    print("Running:", " ".join(cmd))
    print("=" * 72)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run DA (010_dynamic_aperture.py) and MA (009_momentum_acceptance.py) "
            "studies in sequence."
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
        "--da-cases",
        nargs="+",
        metavar="CASE",
        help="DA cases for 010 (default: all). Available: sb_on, varsol_on, sb_off",
    )
    parser.add_argument(
        "--ma-cases",
        nargs="+",
        metavar="CASE",
        help="MA cases for 009 (default: all). Available: sb_on, varsol_on, sb_off",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help="Shortcut for --da-cases (DA only).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after each script (default: save only).",
    )
    args = parser.parse_args()

    if args.da_only and args.ma_only:
        raise SystemExit("Choose at most one of --da-only and --ma-only.")

    da_cases = args.da_cases or args.cases
    da_args: list[str] = []
    ma_args: list[str] = []
    if da_cases:
        da_args.extend(["--cases", *da_cases])
    if args.ma_cases:
        ma_args.extend(["--cases", *args.ma_cases])
    if not args.show:
        da_args.append("--no-show")
        ma_args.append("--no-show")

    run_da = not args.ma_only
    run_ma = not args.da_only

    if run_da:
        _run_script(DA_SCRIPT, da_args)
    if run_ma:
        _run_script(MA_SCRIPT, ma_args)

    print("\nAll requested studies finished.")


if __name__ == "__main__":
    main()
