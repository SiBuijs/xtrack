"""Radiative spin-polarization buildup/depolarization (FCC-ee solenoid).

Initializes a bunch fully polarized along y (P(0) = 1) and tracks it under
quantum synchrotron radiation. Quantum energy-diffusion couples to spin
precession through dn/ddelta, so the ensemble-averaged polarization
P(t) = |<spin_vec>| slowly decoheres; over the trackable turn range this
decay is well inside the linear regime of the true exponential (tau_depol is
typically orders of magnitude larger than a feasible number of tracked
turns), so a straight-line fit gives tau_depol directly (slope = -P(0)/tau).

The polarization-buildup time (Sokolov-Ternov effect alone, no
depolarization) and the asymptotic polarization P_inf are not things this
script's tracking can measure directly (radiative spin-flip is not part of
the per-turn tracking model here) -- both come from the analytic radiation-
integral formalism in Twiss (`polarization_analysis=True`): P_inf ==
`tw.spin_polarization_inf_no_depol`, tau_pol == `tw.spin_t_pol_component_s`.
Combining the tracking-fit tau_depol with the Twiss P_inf/tau_pol via the
standard P_eq = P_inf / (1 + tau_pol/tau_depol) relation gives a
tracking-informed equilibrium-polarization estimate, compared against
Twiss's own (independently, analytically derived) `spin_polarization_eq`.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt

from aperture_study_io import save_pol_study, variant_suffix
from lattice_knobs import robust_twiss, set_lattice_knobs

plt.close("all")

HERE = Path(__file__).resolve().parent
SPLINEBORIS_LATTICE_JSON = (
    HERE / "fccee_z_lcc_splineboris_solenoids_coupling_corrected.json"
)
VARSOL_LATTICE_JSON = (
    HERE / "fccee_z_lcc_varsol_solenoids_coupling_corrected.json"
)

GLOBAL_XY_LIMIT = 1.0
N_TURNS = 10_000
N_PART = 1000

POL_CASES = [
    dict(
        name="sb_on",
        model="SB",
        lattice_json=SPLINEBORIS_LATTICE_JSON,
        with_solenoids=True,
        with_correctors=True,
        title="SplineBoris: solenoids powered + correction scheme",
    ),
    dict(
        name="varsol_on",
        model="VarSol",
        lattice_json=VARSOL_LATTICE_JSON,
        with_solenoids=True,
        with_correctors=True,
        title="VariableSolenoid: solenoids powered + correction scheme",
    ),
    dict(
        name="sb_off",
        model="SB",
        lattice_json=SPLINEBORIS_LATTICE_JSON,
        with_solenoids=False,
        with_correctors=False,
        title="SplineBoris: solenoids unpowered",
    ),
]

POL_CASES_BY_NAME = {case["name"]: case for case in POL_CASES}
# sb_off's depolarization time is many orders of magnitude longer than any
# feasible number of tracked turns (no solenoid coupling to drive dn/ddelta),
# so it is skipped by default -- pass --cases sb_off explicitly to run it.
DEFAULT_POL_CASE_NAMES = [name for name in POL_CASES_BY_NAME if name != "sb_off"]


def _configure_radiative_tracking(line):
    line.particle_ref.anomalous_magnetic_moment = 0.00115965218128
    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss()


def _compute_polarization(mon, n_turns):
    has_state = hasattr(mon, "state") and mon.state is not None

    spin_x_mean = np.empty(n_turns)
    spin_y_mean = np.empty(n_turns)
    spin_z_mean = np.empty(n_turns)

    for t in range(n_turns):
        if has_state:
            alive = mon.state[:, t] > 0
            if not np.any(alive):
                spin_x_mean[t] = np.nan
                spin_y_mean[t] = np.nan
                spin_z_mean[t] = np.nan
                continue
        else:
            alive = slice(None)

        spin_x_mean[t] = np.mean(mon.spin_x[alive, t])
        spin_y_mean[t] = np.mean(mon.spin_y[alive, t])
        spin_z_mean[t] = np.mean(mon.spin_z[alive, t])

    polarization = np.sqrt(spin_x_mean**2 + spin_y_mean**2 + spin_z_mean**2)
    return spin_x_mean, spin_y_mean, spin_z_mean, polarization


def _fit_linear_depolarization(turns, polarization, t_rev0):
    """Straight-line fit P(n) = P0 + slope * n, valid for n << tau_depol.

    Returns (P0, slope [1/turn], tau_depol [turns], tau_depol [s]). tau_depol
    is inf if the fitted slope is non-negative (no significant decay
    resolved above noise -- e.g. too few turns/particles, or a case like
    sb_off where the true depolarization is negligible on any feasible turn
    count).
    """
    mask = np.isfinite(polarization)
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


def _plot_polarization_figure(
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
):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(turns, polarization, label="tracked")

    if np.isfinite(fit_slope):
        fit_curve = fit_p0 + fit_slope * turns
        ax.plot(turns, fit_curve, "--", color="C3", label="linear fit")

    ax.set_xlabel("turn")
    ax.set_ylabel(r"Polarization $P = |\langle \vec{s}\rangle|$")
    ax.legend(loc="upper right", fontsize=8)

    fit_lines = ["fit (tracking, linear regime):"]
    if np.isfinite(fit_slope):
        fit_lines += [
            fr"  $P_0 = {fit_p0:.6f}$",
            fr"  $dP/dn = {fit_slope:.3e}\ \mathrm{{turn}}^{{-1}}$",
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
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        linespacing=1.4,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    ax.set_title(title)
    fig.tight_layout()
    return fig


def _run_polarization_evolution(
    case, *, n_turns, n_part, with_progress, sexamp,
):
    lattice_json = case["lattice_json"]
    title = case["title"]

    print(f"\n=== {title} ===")
    print(f"Loading lattice: {lattice_json.name}")
    env = xt.load(lattice_json)
    line = env.fccee_p_ring
    line.cycle("ipa")
    set_lattice_knobs(
        line,
        with_solenoids=case["with_solenoids"],
        with_correctors=case["with_correctors"],
        sext_amp=sexamp,
    )

    line.discard_tracker()
    line.build_tracker()
    _configure_radiative_tracking(line)

    line.discard_tracker()
    line.build_tracker()
    # robust_twiss falls back to a co-guess continuation for large --sexamp,
    # see lattice_knobs.robust_twiss. polarization_analysis=True implies
    # spin=True and radiation_integrals=True internally.
    tw = robust_twiss(
        line,
        twiss_method="twiss6d",
        radiation_analysis=True,
        polarization_analysis=True,
        strengths=True,
    )

    p_inf = float(tw.spin_polarization_inf_no_depol)
    p_eq_twiss = float(tw.spin_polarization_eq)
    tau_pol_s = float(tw.spin_t_pol_component_s)
    tau_depol_twiss_s = float(tw.spin_t_depol_component_s)
    spin_tune_fractional = float(tw.spin_tune_fractional)
    t_rev0 = float(tw.t_rev0)

    print(f"spin_tune_fractional        = {spin_tune_fractional:.6f}")
    print(f"P_inf (asymptotic, no depol)= {p_inf:.6f}")
    print(
        f"tau_pol   (Twiss, ST-only)  = {tau_pol_s:.6e} s "
        f"({tau_pol_s / t_rev0:.6e} turns)"
    )
    print(
        f"tau_depol (Twiss, analytic) = {tau_depol_twiss_s:.6e} s "
        f"({tau_depol_twiss_s / t_rev0:.6e} turns)"
    )
    print(f"P_eq (Twiss, analytic)      = {p_eq_twiss:.6e}")

    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        nemitt_x=tw.eq_nemitt_x,
        nemitt_y=tw.eq_nemitt_y,
        sigma_z=np.sqrt(tw.eq_gemitt_zeta * tw.bets0),
        line=line,
        particle_on_co=tw.particle_on_co,
        engine="single-rf-harmonic",
    )
    particles.zeta += tw.zeta[0]
    particles.delta += tw.delta[0]

    # Fully polarized along y at t=0 (all spins aligned -> P(0) = 1).
    particles.spin_x = 0.0
    particles.spin_y = 1.0
    particles.spin_z = 0.0

    line.configure_radiation(model="quantum")
    line.discard_tracker()
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads="auto"))

    line.config.XTRACK_GLOBAL_XY_LIMIT = GLOBAL_XY_LIMIT
    print(f"Tracking {n_part} particles for {n_turns} turns (quantum radiation)")
    line.track(
        particles,
        num_turns=n_turns,
        turn_by_turn_monitor=True,
        with_progress=with_progress,
    )
    mon = line.record_last_track

    turns = np.arange(n_turns)
    spin_x_mean, spin_y_mean, spin_z_mean, polarization = _compute_polarization(
        mon, n_turns
    )

    fit_p0, fit_slope, fit_tau_depol_turns, fit_tau_depol_s = (
        _fit_linear_depolarization(turns, polarization, t_rev0)
    )

    print(f"  P(0) tracked                = {polarization[0]:.6f}")
    print(f"  P0 (fit intercept)          = {fit_p0:.6f}")
    print(f"  dP/dturn (fit slope)        = {fit_slope:.6e} 1/turn")
    print(
        f"  tau_depol (fit, tracking)   = {fit_tau_depol_s:.6e} s "
        f"({fit_tau_depol_turns:.6e} turns)   [Twiss: {tau_depol_twiss_s:.6e} s]"
    )

    if np.isfinite(fit_tau_depol_s) and fit_tau_depol_s > 0:
        # Eq. 8.37: P_eq = P_inf / (1 + tau_pol / tau_depol).
        p_eq_derived = p_inf / (1.0 + tau_pol_s / fit_tau_depol_s)
    else:
        p_eq_derived = np.nan
    print(
        f"  P_eq (derived, eq. 8.37)    = {p_eq_derived:.6e} "
        f"[Twiss direct: {p_eq_twiss:.6e}]"
    )

    fig = _plot_polarization_figure(
        turns=turns,
        polarization=polarization,
        fit_p0=fit_p0,
        fit_slope=fit_slope,
        fit_tau_depol_s=fit_tau_depol_s,
        p_inf=p_inf,
        tau_pol_s=tau_pol_s,
        tau_depol_twiss_s=tau_depol_twiss_s,
        p_eq_twiss=p_eq_twiss,
        p_eq_derived=p_eq_derived,
        title=title,
    )
    save_pol_study(
        fig=fig,
        model=case["model"],
        with_solenoids=case["with_solenoids"],
        with_correctors=case["with_correctors"],
        n_turns=n_turns,
        global_xy_limit=GLOBAL_XY_LIMIT,
        n_part=n_part,
        turns=turns,
        spin_x_mean=spin_x_mean,
        spin_y_mean=spin_y_mean,
        spin_z_mean=spin_z_mean,
        polarization=polarization,
        p_inf=p_inf,
        p_eq_twiss=p_eq_twiss,
        tau_pol_s=tau_pol_s,
        tau_depol_twiss_s=tau_depol_twiss_s,
        spin_tune_fractional=spin_tune_fractional,
        t_rev0=t_rev0,
        fit_p0=fit_p0,
        fit_slope=fit_slope,
        fit_tau_depol_turns=fit_tau_depol_turns,
        fit_tau_depol_s=fit_tau_depol_s,
        p_eq_derived=p_eq_derived,
        variant=variant_suffix(
            sexamp=sexamp,
        ),
        sexamp=sexamp,
    )
    print(f"[{title}] Polarization evolution run complete.")
    return dict(
        turns=turns,
        polarization=polarization,
        fit_p0=fit_p0,
        fit_slope=fit_slope,
        fit_tau_depol_s=fit_tau_depol_s,
        p_eq_derived=p_eq_derived,
    )


def _select_cases(case_names):
    if not case_names:
        return [POL_CASES_BY_NAME[name] for name in DEFAULT_POL_CASE_NAMES]

    unknown = [name for name in case_names if name not in POL_CASES_BY_NAME]
    if unknown:
        valid = ", ".join(POL_CASES_BY_NAME)
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Choose from: {valid}")

    return [POL_CASES_BY_NAME[name] for name in case_names]


def main():
    parser = argparse.ArgumentParser(
        description="Run spin-polarization studies for FCC-ee solenoid lattices."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help=(
            "Cases to run (default: sb_on, varsol_on -- sb_off is skipped by "
            "default since its depolarization time is negligibly slow to "
            "resolve by tracking; pass --cases sb_off explicitly to include "
            "it). Available: sb_on, varsol_on, sb_off"
        ),
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available cases and exit.",
    )
    parser.add_argument(
        "--n-turns",
        type=int,
        default=N_TURNS,
        metavar="N",
        help=(
            f"Number of turns to track (default: {N_TURNS}). The "
            "depolarization time is typically orders of magnitude larger "
            "than any feasible turn count; increase this for a cleaner "
            "linear-fit slope, at the cost of runtime."
        ),
    )
    parser.add_argument(
        "--n-part",
        type=int,
        default=N_PART,
        metavar="N",
        help=f"Number of macroparticles (default: {N_PART}).",
    )
    parser.add_argument(
        "--sexamp",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Sextupole amplification knob (default: 1.0).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive figure display (data and PDFs are still saved).",
    )
    args = parser.parse_args()

    if args.list_cases:
        for case in POL_CASES:
            print(f"{case['name']}: {case['title']}")
        return

    for case in _select_cases(args.cases):
        _run_polarization_evolution(
            case,
            n_turns=args.n_turns,
            n_part=args.n_part,
            with_progress=1,
            sexamp=args.sexamp,
        )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
