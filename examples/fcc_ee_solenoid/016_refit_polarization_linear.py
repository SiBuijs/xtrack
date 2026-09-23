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
from matplotlib.patches import FancyBboxPatch

from aperture_study_io import DATA_DIR, save_figure_pdf

# Everything that happens in these runs happens in the fifth and sixth decimal
# of P: the tracked curve here spans P = 1 down to 0.99996. On that range
# matplotlib gives up on plain ticks and prints an offset ("+9.9999e-1") above
# the axis, which hides the scale and makes the numbers unreadable. So plot the
# departure from full polarization in units of 1e-6 instead: P = 1 sits at 0,
# the ticks are small integers, and no offset appears.
P_PLOT_SCALE = 1e-6
P_PLOT_UNIT = r"$10^{-6}$"


def _sci(value, unit=""):
    """Number as mathtext a x 10^b, for the summary table."""
    if value is None or not np.isfinite(value):
        return r"$\infty$" if value == np.inf else "n/a"
    mantissa, exponent = f"{value:.3e}".split("e")
    text = fr"${mantissa} \times 10^{{{int(exponent)}}}$"
    return f"{text} {unit}" if unit else text


def _framed_text_columns(ax, columns, x_right, y_top, fontsize=6.5,
                         linespacing=1.5, col_gap=0.02):
    """Lay out several left-aligned text blocks side by side as table columns
    and draw a single rounded frame around the lot. Returns the frame's bottom
    edge in axes coordinates, so a caller can stack the next frame under it.

    One text object per column, rather than one string for the whole table:
    keeping columns aligned inside a single string would need a monospace font,
    and monospace has no effect inside mathtext, so the tau_depol and P_eq rows
    would drift out of line the moment they carry symbols. Widths therefore have
    to be measured, which means a draw per column -- cheap here, but it does mean
    this has to run after the axes are otherwise final.
    """
    fig = ax.figure
    placed, x_cursor = [], 0.0
    for column in columns:
        text = ax.text(x_cursor, y_top, "\n".join(column),
                       transform=ax.transAxes, ha="left", va="top",
                       fontsize=fontsize, linespacing=linespacing, zorder=3)
        fig.canvas.draw()
        width = text.get_window_extent().transformed(
            ax.transAxes.inverted()).width
        placed.append((text, x_cursor))
        x_cursor += width + col_gap
    # Laid out from zero, then shifted as a block so the table's right edge
    # lands on x_right -- the total width is not known until it is measured.
    shift = x_right - (x_cursor - col_gap)
    for text, x0 in placed:
        text.set_x(x0 + shift)
    fig.canvas.draw()

    boxes = [t.get_window_extent().transformed(ax.transAxes.inverted())
             for t, _ in placed]
    x0 = min(b.x0 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    ax.add_patch(FancyBboxPatch(
        (x0, y0), max(b.x1 for b in boxes) - x0, max(b.y1 for b in boxes) - y0,
        boxstyle="round,pad=0.012", transform=ax.transAxes, zorder=2.5,
        facecolor="white", edgecolor="0.6", linewidth=0.8, alpha=0.9))
    return y0


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

    def to_plot(p):
        """Absolute P -> the plotted departure-from-1 in units of P_PLOT_SCALE."""
        return (np.asarray(p) - 1.0) / P_PLOT_SCALE

    ax.plot(turns, to_plot(polarization), label="tracked")
    ax.axvline(turn_start, color="0.5", ls=":", lw=1,
               label=f"fit start (n={turn_start})")

    if np.isfinite(fit_slope):
        fit_turns = turns[turns >= turn_start]
        fit_curve = fit_intercept + fit_slope * fit_turns
        ax.plot(fit_turns, to_plot(fit_curve), "--", color="C3",
                label="linear fit")

    ax.set_xlabel("turn")
    ax.set_ylabel(fr"$P - 1$  [{P_PLOT_UNIT}],   $P = |\langle \vec{{s}}\rangle|$")
    # Plain integer turn labels; the default would offset these too.
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title)
    # The frames are placed by measuring rendered text, so the axes have to be
    # final first -- tight_layout after them would move everything.
    fig.tight_layout()

    # Two frames, upper right. First the quantities that are not per-method:
    # the fitted line itself, and the two Twiss numbers that both P_eq columns
    # of the table below are built from.
    setup_lines = [f"tracking: linear fit, $n \\geq {turn_start}$"]
    if np.isfinite(fit_slope):
        setup_lines += [
            fr"   $P_0$ (intercept) $= {fit_intercept:.6f}$",
            fr"   slope $=$ {_sci(fit_slope)} / turn",
        ]
    else:
        setup_lines += ["   (no significant decay resolved)"]
    setup_lines += [
        fr"Twiss: $P_\infty = {p_inf:.6f}$,  "
        fr"$\tau_\mathrm{{pol}} =$ {_sci(tau_pol_s, 's')}",
    ]
    table_top = _framed_text_columns(
        ax, [setup_lines], x_right=0.98, y_top=0.97) - 0.035

    # Then the two quantities that Twiss and the fit each give a value for.
    caption_top = _framed_text_columns(
        ax,
        [
            ["", r"$\tau_\mathrm{depol}$", r"$P_\mathrm{eq}$"],
            ["Twiss", _sci(tau_depol_twiss_s, "s"), _sci(p_eq_twiss)],
            ["fit", _sci(fit_tau_depol_s, "s"), _sci(p_eq_derived)],
        ],
        x_right=0.98,
        y_top=table_top,
    )
    # Outside the frame: the fit column's P_eq is not an independent fit
    # result, it is the Twiss P_inf and tau_pol combined with the fitted
    # tau_depol, and the table would otherwise read as two rival measurements.
    ax.text(0.98, caption_top - 0.038,
            r"$P_\mathrm{eq}$ (fit) $= P_\infty / (1 + \tau_\mathrm{pol} /"
            r" \tau_\mathrm{depol}^\mathrm{\,fit})$, Twiss "
            r"$P_\infty, \tau_\mathrm{pol}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
            color="0.35")
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
