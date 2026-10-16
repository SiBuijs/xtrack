"""Regenerate PDF plots from saved DA / MA / EMIT / POL / TUNE .npz files in
examples/fcc_ee_solenoid/data, without rerunning any tracking or twiss."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from aperture_study_io import DATA_DIR, REPLOTTABLE_STUDIES, replot_from_npz

plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="Replot DA/MA/EMIT/POL/TUNE studies from saved .npz data."
    )
    parser.add_argument(
        "npz_files",
        nargs="*",
        help=(
            "One or more .npz paths. If omitted, replot every DA/MA/EMIT/POL/"
            "TUNE file in data/."
        ),
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
        # Only study files: data/ also holds other scripts' caches (e.g. 004dd).
        npz_paths = sorted(
            path for study in REPLOTTABLE_STUDIES
            for path in DATA_DIR.glob(f"{study}_*.npz")
        )

    if not npz_paths:
        raise SystemExit(f"No .npz files found in {DATA_DIR}")

    for npz_path in npz_paths:
        replot_from_npz(npz_path, show=args.show)


if __name__ == "__main__":
    main()
