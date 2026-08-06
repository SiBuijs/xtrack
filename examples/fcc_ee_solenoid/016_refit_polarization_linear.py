"""Refit saved spin-polarization decay (015_spin_polarization.py output) with
a straight-line model over a restricted turn range, without rerunning
tracking.

015 fits P(n) = P0 * exp(-n / tau_depol) over the full tracked range. This
script instead fits the linear form P(n) = P0 + slope * n restricted to
turns >= --turn-start (default 2000), to drop the early-turn transient before
the decay settles into its (very slowly varying) asymptotic slope. tau_depol
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


def _fit_linear_depolarization(turns, polarization, t_rev0, turn_start):
    """P(n) ~= P0 + slope*n for n >= turn_start. tau_depol = -1/slope (turns),
    inf if the fitted slope isn't negative (no resolvable decay in range)."""
    mask = np.isfinite(polarization) & (turns >= turn_start)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan

    slope, intercept = np.polyfit(turns[mask], polarization[mask], deg=1)
    if slope < 0:
        tau_depol_turns = -1.0 / slope
        tau_depol_s = tau_depol_turns * t_rev0
    else:
        tau_depol_turns = np.inf
        tau_depol_s = np.inf
    return float(intercept), float(slope), float(tau_depol_turns), float(tau_depol_s)


def _plot_linear_refit_figure(
    *,
    turns,
    polarization,
    turn_start,
    fit_intercept,
    fit_slope,
    fit_tau_depol_s,
    p_inf,
    tau_pol_s,
    tau_depol_twiss_s,
    p_eq_twiss,
    p_eq_derived,
    title,
):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(turns, polarization, label="tracked")
    ax.axvline(turn_start, color="0.5", ls=":", lw=1, label=f"fit start (n={turn_start})")

    if np.isfinite(fit_slope):
        fit_turns = turns[turns >= turn_start]
        fit_curve = fit_intercept + fit_slope * fit_turns
        ax.plot(fit_turns, fit_curve, "--", color="C3", label="linear fit")

    ax.set_xlabel("turn")
    ax.set_ylabel(r"Polarization $P = |\langle \vec{s}\rangle|$")
    ax.legend(loc="lower right", fontsize=8)

    fit_lines = [f"fit (tracking, linear, n>={turn_start}):"]
    if np.isfinite(fit_slope):
        fit_lines += [
            fr"  $P_0\ \mathrm{{(intercept)}} = {fit_intercept:.6f}$",
            fr"  $\tau_\mathrm{{depol}} = {fit_tau_depol_s:.3e}\ \mathrm{{s}}$",
        ]
    else:
        fit_lines += ["  (no significant decay resolved)"]
    twiss_lines = [
        "Twiss:",
        fr"  $P_\infty = {p_inf:.6f}$",
        fr"  $\tau_\mathrm{{pol}} = {tau_pol_s:.3e}\ \mathrm{{s}}$",
        fr"  $\tau_\mathrm{{depol}} = {tau_depol_twiss_s:.3e}\ \mathrm{{s}}$",
        fr"  $P_\mathrm{{eq}} = {p_eq_twiss:.3e}$",
    ]
    derived_lines = [
        r"Derived ($P_\infty,\tau_\mathrm{pol}$ Twiss + $\tau_\mathrm{depol}$ fit):",
        fr"  $P_\mathrm{{eq}} = {p_eq_derived:.3e}$",
    ]
    info_text = "\n".join(fit_lines + twiss_lines + derived_lines)
    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        linespacing=1.4,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    ax.set_title(title)
    fig.tight_layout()
    return fig


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
        _fit_linear_depolarization(turns, polarization, t_rev0, turn_start)
    )

    if np.isfinite(fit_tau_depol_s) and fit_tau_depol_s > 0:
        p_eq_derived = p_inf / (1.0 + tau_pol_s / fit_tau_depol_s)
    else:
        p_eq_derived = np.nan

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

    fig = _plot_linear_refit_figure(
        turns=turns,
        polarization=polarization,
        turn_start=turn_start,
        fit_intercept=fit_intercept,
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
            "(P0 + slope*n) model over a restricted turn range, instead of the "
            "exponential fit 015_spin_polarization.py performs over the full "
            "range. Does not rerun tracking."
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
