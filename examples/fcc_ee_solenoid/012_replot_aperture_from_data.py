"""Regenerate PDF plots from saved DA / MA .npz files in examples/fcc_ee_solenoid/data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from aperture_study_io import DATA_DIR, replot_from_npz

plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="Replot DA/MA studies from saved .npz data."
    )
    parser.add_argument(
        "npz_files",
        nargs="*",
        help="One or more .npz paths. If omitted, replot every file in data/.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving PDFs.",
    )
    args = parser.parse_args()

    if args.npz_files:
        npz_paths = [Path(p) for p in args.npz_files]
    else:
        npz_paths = sorted(DATA_DIR.glob("*.npz"))

    if not npz_paths:
        raise SystemExit(f"No .npz files found in {DATA_DIR}")

    for npz_path in npz_paths:
        replot_from_npz(npz_path, show=args.show)


if __name__ == "__main__":
    main()
