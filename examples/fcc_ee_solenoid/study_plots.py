"""Figure-drawing functions shared by the emittance (014, 018), polarization
(015, 018) and tune-vs-delta (017) scripts and by 012's replot from saved
data, so a replotted figure is drawn by exactly the same code as the
original. They take only arrays and numbers that the save_*_study functions
in aperture_study_io store in the .npz.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


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


def plot_polarization_figure(
    *,
    turns,
    polarization,
    fit_p0,
    fit_tau_depol_turns,
    fit_tau_depol_s,
    p_inf,
    tau_pol_s,
    tau_depol_twiss_s,
    p_eq_twiss,
    p_eq_derived,
    title,
    t_rev0,
    turn_start,
):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(turns, polarization, label="tracked")
    if turn_start > 0:
        ax.axvline(turn_start, color="0.5", ls=":", lw=1,
                   label=f"fit start (n={turn_start})")

    if np.isfinite(fit_tau_depol_turns):
        # Drawn only over the fitted range, so the figure cannot suggest the
        # fit describes the excluded transient.
        fit_turns = turns[turns >= turn_start]
        fit_curve = fit_p0 * np.exp(-fit_turns / fit_tau_depol_turns)
        ax.plot(fit_turns, fit_curve, "--", color="C3",
                label="exponential fit")

    ax.set_xlabel("turn")
    ax.set_ylabel(r"Polarization $P = |\langle \vec{s}\rangle|$")
    ax.legend(loc="lower right", fontsize=8)

    # Each figure autoscales to its own data, and P sits so close to 1 that
    # matplotlib falls back to offset notation ("1e-6 + 9.9999e-1"). Two
    # figures can then differ by 10x in y-scale while looking identically
    # steep -- a 2e-06 wobble fills the frame exactly like a 2e-05 one, so
    # comparing two of these figures by eye gives the wrong answer, and can
    # invert the ranking of which case depolarizes more. State the absolute
    # excursion in the box so the figures are comparable as numbers, and
    # state what Twiss's own tau_depol predicts over the same span so it is
    # obvious that the tracked curve is orders of magnitude flatter than the
    # analytic decay it is supposed to be measuring.
    span_lines = []
    if len(polarization) and np.isfinite(polarization[0]):
        finite = polarization[np.isfinite(polarization)]
        if finite.size:
            drop_total = float(polarization[0] - finite[-1])
            span_lines = [
                "span (tracked):",
                fr"  $\Delta P$ over run $= {drop_total:.3e}$",
            ]
            if np.isfinite(tau_depol_twiss_s) and tau_depol_twiss_s > 0:
                tau_twiss_turns = tau_depol_twiss_s / t_rev0
                n_span = float(turns[-1] - turns[0])
                drop_twiss = float(
                    polarization[0] * (1.0 - np.exp(-n_span / tau_twiss_turns)))
                span_lines.append(
                    fr"  $\Delta P$ if $\tau_\mathrm{{depol}}$(Twiss)"
                    fr"$ = {drop_twiss:.3e}$")
                if drop_total > 0:
                    span_lines.append(
                        fr"  ratio Twiss/tracked $= {drop_twiss / drop_total:.3g}$")

    fit_lines = [f"fit (tracking, exponential decay, n>={turn_start}):"]
    if np.isfinite(fit_tau_depol_turns):
        fit_lines += [
            fr"  $P_0 = {fit_p0:.6f}$",
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
    info_text = "\n".join(span_lines + fit_lines + twiss_lines + derived_lines)
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
