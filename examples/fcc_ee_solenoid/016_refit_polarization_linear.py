"""Refit saved spin-polarization decay (015_spin_polarization.py output) with
a straight-line model over a restricted turn range, without rerunning
tracking.

Fits the linear form P(n) = P0 + slope * n restricted to turns >= --turn-start
(default 2000), the same fit 015/018 now do inline, to redo it with another
start turn. The early-turn transient is dropped before the decay settles
into its (very slowly varying) asymptotic slope. tau_depol
is recovered as -1 / slope (turns), matching the convention used in the
project notes for the linear-regime fit.

Usage:
    python 016_refit_polarization_linear.py                     # all data/POL_*.npz
    python 016_refit_polarization_linear.py data/POL_Sol_On_SB_3T_1000p_10000t_xylim1m.npz
    python 016_refit_polarization_linear.py --turn-start 5000 --show
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from aperture_study_io import DATA_DIR, save_figure_pdf
from study_plots import (
    fit_linear_depolarization,
    p_eq_from_tau_depol,
    plot_polarization_figure,
)


def refit_pol_npz(npz_path: Path, *, turn_start: int, show: bool) -> Path:
    with np.load(npz_path, allow_pickle=True) as data:
        turns = data["turns"]
        polarization = data["polarization"]
        t_rev0 = float(data["t_rev0"])
        p_inf = float(data["p_inf"])
        tau_pol_s = float(data["tau_pol_s"])
        tau_depol_twiss_s = float(data["tau_depol_twiss_s"])
        p_eq_twiss = float(data["p_eq_twiss"])
        model = str(data["model"])
        with_solenoids = bool(data["with_solenoids"])
        with_correctors = bool(data["with_correctors"])

    if turn_start >= turns[-1]:
        raise SystemExit(
            f"--turn-start {turn_start} is >= the last tracked turn "
            f"({turns[-1]}) in {npz_path.name}; nothing left to fit."
        )

    fit_intercept, fit_slope, fit_tau_depol_turns, fit_tau_depol_s = (
        fit_linear_depolarization(turns, polarization, t_rev0, turn_start)
    )
    p_eq_derived = p_eq_from_tau_depol(p_inf, tau_pol_s, fit_tau_depol_s)

    model_name = "SplineBoris" if model == "SB" else "VariableSolenoid"
    sol_state = "solenoids on" if with_solenoids else "solenoids off"
    if with_solenoids and not with_correctors:
        sol_state += ", correctors off"
    title = f"{npz_path.stem}\n{model_name}: {sol_state} (linear fit, n>={turn_start})"

    print(f"\n=== {npz_path.name} ===")
    print(f"  fit range                   = turns [{turn_start}, {int(turns[-1])}]")
    print(f"  P0 (fit intercept at n=0)   = {fit_intercept:.6f}")
    print(f"  slope (fit)                 = {fit_slope:.6e} 1/turn")
    print(
        f"  tau_depol (fit, linear)     = {fit_tau_depol_s:.6e} s "
        f"({fit_tau_depol_turns:.6e} turns)   [Twiss: {tau_depol_twiss_s:.6e} s]"
    )
    print(
        f"  P_eq (derived, eq. 8.37)    = {p_eq_derived:.6e} "
        f"[Twiss direct: {p_eq_twiss:.6e}]"
    )

    fig = plot_polarization_figure(
        turns=turns,
        polarization=polarization,
        turn_start=turn_start,
        fit_p0=fit_intercept,
        fit_slope=fit_slope,
        fit_tau_depol_s=fit_tau_depol_s,
        p_inf=p_inf,
        tau_pol_s=tau_pol_s,
        tau_depol_twiss_s=tau_depol_twiss_s,
        p_eq_twiss=p_eq_twiss,
        p_eq_derived=p_eq_derived,
        title=title,
    )
    pdf_path = save_figure_pdf(fig, f"{npz_path.stem}_linfit_from{turn_start}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Refit saved POL_*.npz spin-polarization data with a straight-line "
            "(P0 + slope*n) model from a chosen start turn (the same fit "
            "015/018 do inline). Does not rerun tracking."
        )
    )
    parser.add_argument(
        "npz_files",
        nargs="*",
        type=Path,
        help="POL .npz files to refit (default: every data/POL_*.npz).",
    )
    parser.add_argument(
        "--turn-start",
        type=int,
        default=2000,
        metavar="N",
        help="Only fit turns >= N, to drop the early-turn transient (default: 2000).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure interactively (PDFs are always saved).",
    )
    args = parser.parse_args()

    npz_files = args.npz_files or sorted(DATA_DIR.glob("POL_*.npz"))
    if not npz_files:
        raise SystemExit(f"No POL_*.npz files found in {DATA_DIR}")

    for npz_path in npz_files:
        refit_pol_npz(npz_path, turn_start=args.turn_start, show=args.show)


if __name__ == "__main__":
    main()
