"""Figure-drawing functions shared by the emittance (014, 018), polarization
(015, 018) and tune-vs-delta (017) scripts and by 012's replot from saved
data, so a replotted figure is drawn by exactly the same code as the
original. They take only arrays and numbers that the save_*_study functions
in aperture_study_io store in the .npz.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def analytic_emittance(turns, eps_init, eps_eq, damp_turn):
    return (eps_init - eps_eq) * np.exp(-2.0 * damp_turn * turns) + eps_eq


def plot_emittance_evolution_figure(
    *,
    turns,
    gemitt_x,
    gemitt_y,
    gemitt_z,
    eq_x,
    eq_y,
    eq_z,
    damp_turns,
    fit_eps_eq,
    fit_alpha,
    twiss_eps_init,
    title,
):
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True)
    series = [
        (gemitt_x, eq_x, damp_turns[0], fit_eps_eq[0], fit_alpha[0], twiss_eps_init[0], "x", r"$\varepsilon_x$ [m·rad]"),
        (gemitt_y, eq_y, damp_turns[1], fit_eps_eq[1], fit_alpha[1], twiss_eps_init[1], "y", r"$\varepsilon_y$ [m·rad]"),
        (gemitt_z, eq_z, damp_turns[2], fit_eps_eq[2], fit_alpha[2], twiss_eps_init[2], r"\zeta", r"$\varepsilon_\zeta$ [m·rad]"),
    ]

    for ax, (gemitt, eq, damp, eps_eq_fit, alpha_fit, eps_init_tw, sym, ylabel) in zip(axes, series):
        eps_init = gemitt[0]
        ax.plot(turns, gemitt, label="tracked")
        ax.axhline(eq, linestyle="--", color="C2", label=r"$\varepsilon_\mathrm{eq}$ (Twiss)")

        if np.isfinite(alpha_fit):
            fit_curve = (eps_init - eps_eq_fit) * np.exp(-alpha_fit * turns) + eps_eq_fit
            ax.plot(turns, fit_curve, "--", color="C3", label="fit")

        analytic = analytic_emittance(turns, eps_init, eq, damp)
        ax.plot(turns, analytic, ":", color="C4", label=r"Twiss $-2\lambda t$")

        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=8)

        alpha_tw = 2.0 * damp
        tau_tw = 1.0 / alpha_tw
        fit_lines = ["fit:"]
        if np.isfinite(alpha_fit):
            tau_fit = 1.0 / alpha_fit
            fit_lines += [
                fr"  $\varepsilon_{{{sym},0}} = {eps_init:.3e}$",
                fr"  $\varepsilon_{{{sym},\mathrm{{eq}}}} = {eps_eq_fit:.3e}$",
                fr"  $\alpha_{{{sym}}} = {alpha_fit:.3e}\ \mathrm{{turn}}^{{-1}}$",
                fr"  $\tau_{{{sym}}} = {tau_fit:.3e}\ \mathrm{{turns}}$",
            ]
        else:
            fit_lines += ["  (failed)"]
        twiss_lines = [
            "Twiss:",
            fr"  $\varepsilon_{{{sym},0}} = {eps_init_tw:.3e}$",
            fr"  $\varepsilon_{{{sym},\mathrm{{eq}}}} = {eq:.3e}$",
            fr"  $\alpha_{{{sym}}} = {alpha_tw:.3e}\ \mathrm{{turn}}^{{-1}}$",
            fr"  $\tau_{{{sym}}} = {tau_tw:.3e}\ \mathrm{{turns}}$",
        ]
        info_text = "\n".join(fit_lines + twiss_lines)
        ax.text(
            0.98,
            0.03,
            info_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            linespacing=1.4,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )

    axes[-1].set_xlabel("turn")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# Everything that happens in the polarization runs happens in the fifth and
# sixth decimal of P (P = 1 down to ~0.99999). On that range matplotlib gives
# up on plain ticks and prints an offset ("+9.9999e-1"), which hides the scale
# and makes two figures look equally steep when they differ by 10x. So P is
# plotted as its departure from 1 in units of 1e-6.
P_PLOT_SCALE = 1e-6
P_PLOT_UNIT = r"$10^{-6}$"


def fit_linear_depolarization(turns, polarization, t_rev0, turn_start):
    """Straight-line fit P(n) ~= P0 + slope*n over n >= turn_start.

    tau_depol is far beyond any feasible turn count, so the tracked range is
    deep in the linear onset of the decay and tau_depol = -1/slope (P ~= 1).
    Returns (P0, slope [1/turn], tau_depol [turns], tau_depol [s]);
    tau_depol is inf if the slope isn't negative (no resolvable decay).
    """
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


def p_eq_from_tau_depol(p_inf, tau_pol_s, tau_depol_s):
    """Eq. 8.37: P_eq = P_inf / (1 + tau_pol / tau_depol); nan if no decay."""
    if np.isfinite(tau_depol_s) and tau_depol_s > 0:
        return p_inf / (1.0 + tau_pol_s / tau_depol_s)
    return np.nan


def _sci(value, unit=""):
    """Number as mathtext a x 10^b, for the summary table."""
    if value is None or not np.isfinite(value):
        return r"$\infty$" if value == np.inf else "n/a"
    mantissa, exponent = f"{value:.3e}".split("e")
    text = fr"${mantissa} \times 10^{{{int(exponent)}}}$"
    return f"{text} {unit}" if unit else text


def _framed_table(ax, columns, x_right, y_top, fontsize=7.5, row_pitch=0.055,
                  col_gap=0.025):
    """Small table in a rounded frame, right edge at x_right, top at y_top
    (axes coordinates). columns[i][j] is the cell in column i, row j.

    Every cell is its own text object at a fixed row height: mathtext cells
    are taller than plain ones, so a newline-joined column would drift out of
    line with its neighbours. Column widths are measured, so this has to run
    after the axes are otherwise final. Returns the frame's bottom edge.
    """
    fig = ax.figure
    placed, x_cursor = [], 0.0
    for column in columns:
        texts = [ax.text(x_cursor, y_top - j * row_pitch, cell,
                         transform=ax.transAxes, ha="left", va="top",
                         fontsize=fontsize, zorder=3)
                 for j, cell in enumerate(column)]
        fig.canvas.draw()
        width = max(t.get_window_extent().transformed(
            ax.transAxes.inverted()).width for t in texts)
        placed.append((texts, x_cursor))
        x_cursor += width + col_gap
    # Laid out from zero, then shifted as a block so the table's right edge
    # lands on x_right -- the total width is not known until it is measured.
    shift = x_right - (x_cursor - col_gap)
    for texts, x0 in placed:
        for t in texts:
            t.set_x(x0 + shift)
    fig.canvas.draw()

    boxes = [t.get_window_extent().transformed(ax.transAxes.inverted())
             for texts, _ in placed for t in texts]
    x0 = min(b.x0 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    ax.add_patch(FancyBboxPatch(
        (x0, y0), max(b.x1 for b in boxes) - x0, max(b.y1 for b in boxes) - y0,
        boxstyle="round,pad=0.015", transform=ax.transAxes, zorder=2.5,
        facecolor="white", edgecolor="0.6", linewidth=0.8, alpha=0.9))
    return y0


def _p_cell(value):
    """Polarization for the table: 4 decimals, scientific once that hides it."""
    if not np.isfinite(value):
        return "n/a"
    return f"${value:.4f}$" if abs(value) >= 1e-2 else _sci(value)


def plot_polarization_figure(
    *,
    turns,
    polarization,
    fit_p0,
    fit_slope,
    fit_tau_depol_s,
    p_inf,
    tau_pol_s,
    tau_depol_twiss_s,
    p_eq_twiss,
    p_eq_derived,
    title,
    turn_start,
):
    """Tracked P(n) with its linear fit, and a box with the spin quantities:
    P_inf and tau_pol (Twiss only -- tracking has no spin flip, so cannot
    measure them), tau_depol and P_eq (Twiss vs fit)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    def to_plot(p):
        return (np.asarray(p) - 1.0) / P_PLOT_SCALE

    ax.plot(turns, to_plot(polarization), label="tracked")
    if turn_start > 0:
        ax.axvline(turn_start, color="0.5", ls=":", lw=1,
                   label=f"fit start (n={turn_start})")
    if np.isfinite(fit_slope):
        # Drawn only over the fitted range, so the figure cannot suggest the
        # fit describes the excluded transient.
        fit_turns = turns[turns >= turn_start]
        ax.plot(fit_turns, to_plot(fit_p0 + fit_slope * fit_turns), "--",
                color="C3", label="linear fit")

    ax.set_xlabel("turn")
    ax.set_ylabel(fr"$P - 1$  [{P_PLOT_UNIT}],   $P = |\langle \vec{{s}}\rangle|$")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title)
    # The frame is placed by measuring rendered text, so the axes have to be
    # final first -- tight_layout after it would move everything.
    fig.tight_layout()

    caption_top = _framed_table(
        ax,
        [
            ["", "Twiss", "fit"],
            [r"$P_\infty$", _p_cell(p_inf), "–"],
            [r"$\tau_\mathrm{pol}$", _sci(tau_pol_s, "s"), "–"],
            [r"$\tau_\mathrm{depol}$", _sci(tau_depol_twiss_s, "s"),
             _sci(fit_tau_depol_s, "s")],
            [r"$P_\mathrm{eq}$", _p_cell(p_eq_twiss), _p_cell(p_eq_derived)],
        ],
        x_right=0.98,
        y_top=0.97,
    )
    # P_eq (fit) is not an independent measurement: it combines the fitted
    # tau_depol with the Twiss P_inf and tau_pol.
    ax.text(0.98, caption_top - 0.03,
            r"fit: $\tau_\mathrm{depol} = -1/\mathrm{slope}$,  "
            r"$P_\mathrm{eq} = P_\infty / (1 + \tau_\mathrm{pol} /"
            r" \tau_\mathrm{depol})$",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color="0.35")
    return fig


def _order_symbol(n: int, sym: str) -> str:
    """LaTeX label for the order-n delta-derivative of q_sym: dq_x, ddq_x,
    then d^3q_x, d^4q_x, ... beyond that."""
    if n == 1:
        return fr"dq_{sym}"
    if n == 2:
        return fr"ddq_{sym}"
    return fr"d^{n}q_{sym}"


def plot_tune_vs_delta_figure(
    *,
    delta_values,
    qx,
    qy,
    qx0,
    qy0,
    poly_qx,
    poly_qy,
    fit_order,
    coeffs_x_fit,
    coeffs_y_fit,
    dqx_twiss,
    ddqx_twiss,
    dqy_twiss,
    ddqy_twiss,
    title,
):
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(6.4, 7.6), sharex=True)
    delta_fine = np.linspace(delta_values[0], delta_values[-1], 400)

    # Plot relative to the on-momentum tune (qx0/qy0, from the Method-2
    # Twiss-direct call at delta0=0) -- this only shifts the curve vertically
    # by a constant, so it doesn't change dq/ddq/... at all, just strips the
    # large integer-tune offset (e.g. ~194) that otherwise dominates the axis.
    panels = (
        (ax_x, qx, qx0, poly_qx, coeffs_x_fit, dqx_twiss, ddqx_twiss, "x"),
        (ax_y, qy, qy0, poly_qy, coeffs_y_fit, dqy_twiss, ddqy_twiss, "y"),
    )
    for ax, q, q0, poly, coeffs_fit, dq_tw, ddq_tw, sym in panels:
        ax.plot(delta_values, q - q0, "o", ms=4, color="C0", label="Twiss scan")
        ax.plot(
            delta_fine, np.polyval(poly, delta_fine) - q0, "--", color="C3",
            label=f"order-{fit_order} fit",
        )
        ax.set_ylabel(fr"$q_{sym} - q_{sym}(\delta{{=}}0)$")
        ax.legend(loc="upper left", fontsize=8)

        fit_lines = [f"fit (order-{fit_order} scan):"]
        for n in range(1, fit_order + 1):
            fit_lines.append(fr"  ${_order_symbol(n, sym)} = {coeffs_fit[n]:.5g}$")
        twiss_lines = [
            "Twiss (direct):",
            fr"  ${_order_symbol(1, sym)} = {dq_tw:.5g}$",
            fr"  ${_order_symbol(2, sym)} = {ddq_tw:.5g}$",
        ]
        info_text = "\n".join(fit_lines + twiss_lines)
        ax.text(
            0.98, 0.03, info_text,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=7.5, linespacing=1.4,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )

    ax_y.set_xlabel(r"$\delta$")
    ax_y.tick_params(axis="x", labelrotation=30)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
