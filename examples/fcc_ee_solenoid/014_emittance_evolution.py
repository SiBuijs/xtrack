"""Emittance evolution toward synchrotron-radiation equilibrium (FCC-ee solenoid).

Compares damping for the three lattice cases (sb_on, varsol_on, sb_off) using
mean-mode Twiss for optics and bunch generation, then quantum-mode tracking.

Coupling caveat: with powered solenoids, x-y coupling may make single-plane
Courant-Snyder emittances approximate. A follow-up could extract epsilon from
the 4x4 transverse covariance eigenvalues.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
from scipy.optimize import curve_fit

from aperture_study_io import save_emitt_study

plt.close("all")

HERE = Path(__file__).resolve().parent
SPLINEBORIS_LATTICE_JSON = (
    HERE / "fccee_z_lcc_splineboris_solenoids_coupling_corrected.json"
)
VARSOL_LATTICE_JSON = (
    HERE / "fccee_z_lcc_varsol_solenoids_coupling_corrected.json"
)

IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]
GLOBAL_XY_LIMIT = 1.0
N_TURNS = 10_000
N_PART = 1000

EMITT_CASES = [
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

EMITT_CASES_BY_NAME = {case["name"]: case for case in EMITT_CASES}


def _set_solenoid_knobs(line, *, with_solenoids, with_correctors):
    for ip_name in IP_NAMES:
        if f"on_sol_{ip_name}" in line.vars:
            line[f"on_sol_{ip_name}"] = float(with_solenoids)

        for corr_knob in (
            f"on_sol_corr_{ip_name}",
            f"on_comp_sol_{ip_name}",
            f"on_rot_doublet_left_{ip_name}",
            f"on_rot_doublet_right_{ip_name}",
            f"on_sol_orbit_corr_{ip_name}",
            f"on_sol_optics_corr_{ip_name}",
            f"on_sol_coupling_corr_{ip_name}",
        ):
            if corr_knob in line.vars:
                line[corr_knob] = float(with_correctors)


def _configure_radiative_tracking(line):
    line.particle_ref.anomalous_magnetic_moment = 0.00115965218128
    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss()


def _compute_geometric_emittances(mon, tw, n_turns):
    i0 = 0
    betx = float(tw["betx"][i0])
    alfx = float(tw["alfx"][i0])
    gamx = (1.0 + alfx**2) / betx
    bety = float(tw["bety"][i0])
    alfy = float(tw["alfy"][i0])
    gamy = (1.0 + alfy**2) / bety
    bets0 = float(tw["bets0"])

    has_state = hasattr(mon, "state") and mon.state is not None

    gemitt_x = np.empty(n_turns)
    gemitt_y = np.empty(n_turns)
    gemitt_z = np.empty(n_turns)

    for t in range(n_turns):
        if has_state:
            alive = mon.state[:, t] > 0
            if not np.any(alive):
                gemitt_x[t] = np.nan
                gemitt_y[t] = np.nan
                gemitt_z[t] = np.nan
                continue
        else:
            alive = slice(None)

        x = mon.x[alive, t]
        px = mon.px[alive, t]
        y = mon.y[alive, t]
        py = mon.py[alive, t]
        zeta = mon.zeta[alive, t]
        delta = mon.delta[alive, t]

        gemitt_x[t] = 0.5 * np.mean(gamx * x**2 + 2 * alfx * x * px + betx * px**2)
        gemitt_y[t] = 0.5 * np.mean(gamy * y**2 + 2 * alfy * y * py + bety * py**2)
        gemitt_z[t] = 0.5 * np.mean(zeta**2 / bets0 + bets0 * delta**2)

    return gemitt_x, gemitt_y, gemitt_z


def _fit_damping_rate(turns, gemitt, eps_eq, eps_init):
    def model(turns_arr, a):
        return (eps_init - eps_eq) * np.exp(a * turns_arr) + eps_eq

    mask = np.isfinite(gemitt)
    if mask.sum() < 3:
        return np.nan

    popt, _ = curve_fit(
        model,
        turns[mask],
        gemitt[mask],
        p0=[-2.0 * 1e-3],
        maxfev=10_000,
    )
    return float(popt[0])


def _analytic_emittance(turns, eps_init, eps_eq, damp_turn):
    return (eps_init - eps_eq) * np.exp(-2.0 * damp_turn * turns) + eps_eq


def _plot_emittance_evolution_figure(
    *,
    turns,
    gemitt_x,
    gemitt_y,
    gemitt_z,
    eq_x,
    eq_y,
    eq_z,
    damp_turns,
    fit_a,
    title,
):
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True)
    series = [
        (gemitt_x, eq_x, damp_turns[0], fit_a[0], r"$\varepsilon_x$ [m·rad]"),
        (gemitt_y, eq_y, damp_turns[1], fit_a[1], r"$\varepsilon_y$ [m·rad]"),
        (gemitt_z, eq_z, damp_turns[2], fit_a[2], r"$\varepsilon_\zeta$ [m·rad]"),
    ]

    for ax, (gemitt, eq, damp, a_fit, ylabel) in zip(axes, series):
        eps_init = gemitt[0]
        ax.plot(turns, gemitt, label="tracked")
        ax.axhline(eq, linestyle="--", color="C2", label=r"$\varepsilon_\mathrm{eq}$ (Twiss)")

        if np.isfinite(a_fit):
            fit_curve = (eps_init - eq) * np.exp(a_fit * turns) + eq
            ax.plot(turns, fit_curve, "--", color="C3", label="fit")

        analytic = _analytic_emittance(turns, eps_init, eq, damp)
        ax.plot(turns, analytic, ":", color="C4", label=r"Twiss $-2\lambda t$")

        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("turn")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _run_emittance_evolution(case, *, n_turns, n_part, with_progress):
    lattice_json = case["lattice_json"]
    title = case["title"]

    print(f"\n=== {title} ===")
    print(f"Loading lattice: {lattice_json.name}")
    env = xt.load(lattice_json)
    line = env.fccee_p_ring
    line.cycle("ipa")
    _set_solenoid_knobs(
        line,
        with_solenoids=case["with_solenoids"],
        with_correctors=case["with_correctors"],
    )

    line.discard_tracker()
    line.build_tracker()
    _configure_radiative_tracking(line)

    line.discard_tracker()
    line.build_tracker()
    tw = line.twiss6d(radiation_analysis=True, strengths=True)

    eq_x = float(tw.eq_gemitt_x)
    eq_y = float(tw.eq_gemitt_y)
    eq_z = float(tw.eq_gemitt_zeta)
    damp_turns = np.asarray(tw.damping_constants_turns, dtype=float)

    print(f"eq_gemitt_x   = {eq_x:.6e} m·rad")
    print(f"eq_gemitt_y   = {eq_y:.6e} m·rad")
    print(f"eq_gemitt_zeta= {eq_z:.6e} m·rad")
    print(f"damping_constants_turns = {damp_turns}")

    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        nemitt_x=tw.eq_nemitt_x / 3,
        nemitt_y=tw.eq_nemitt_y / 3,
        sigma_z=np.sqrt(tw.eq_gemitt_zeta / 3 * tw.bets0),
        line=line,
        particle_on_co=tw.particle_on_co,
        engine="single-rf-harmonic",
    )
    particles.zeta += tw.zeta[0]
    particles.delta += tw.delta[0]

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

    gemitt_x, gemitt_y, gemitt_z = _compute_geometric_emittances(mon, tw, n_turns)
    turns = np.arange(n_turns)

    fit_a = np.array(
        [
            _fit_damping_rate(turns, gemitt_x, eq_x, gemitt_x[0]),
            _fit_damping_rate(turns, gemitt_y, eq_y, gemitt_y[0]),
            _fit_damping_rate(turns, gemitt_z, eq_z, gemitt_z[0]),
        ]
    )
    for plane, a_fit, damp in zip(["x", "y", "zeta"], fit_a, damp_turns):
        print(
            f"  plane {plane}: fitted a = {a_fit:.6e}, "
            f"-2*damp_turn = {-2 * damp:.6e}"
        )

    fig = _plot_emittance_evolution_figure(
        turns=turns,
        gemitt_x=gemitt_x,
        gemitt_y=gemitt_y,
        gemitt_z=gemitt_z,
        eq_x=eq_x,
        eq_y=eq_y,
        eq_z=eq_z,
        damp_turns=damp_turns,
        fit_a=fit_a,
        title=title,
    )
    save_emitt_study(
        fig=fig,
        model=case["model"],
        with_solenoids=case["with_solenoids"],
        with_correctors=case["with_correctors"],
        n_turns=n_turns,
        global_xy_limit=GLOBAL_XY_LIMIT,
        n_part=n_part,
        turns=turns,
        gemitt_x=gemitt_x,
        gemitt_y=gemitt_y,
        gemitt_z=gemitt_z,
        eq_gemitt_x=eq_x,
        eq_gemitt_y=eq_y,
        eq_gemitt_zeta=eq_z,
        damping_constants_turns=damp_turns,
        fit_a=fit_a,
        fit_eps_init=np.array([gemitt_x[0], gemitt_y[0], gemitt_z[0]]),
    )
    print(f"[{title}] Emittance evolution run complete.")
    return dict(
        turns=turns,
        gemitt_x=gemitt_x,
        gemitt_y=gemitt_y,
        gemitt_z=gemitt_z,
        fit_a=fit_a,
    )


def _select_cases(case_names):
    if not case_names:
        return EMITT_CASES

    unknown = [name for name in case_names if name not in EMITT_CASES_BY_NAME]
    if unknown:
        valid = ", ".join(EMITT_CASES_BY_NAME)
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Choose from: {valid}")

    return [EMITT_CASES_BY_NAME[name] for name in case_names]


def main():
    parser = argparse.ArgumentParser(
        description="Run emittance-evolution studies for FCC-ee solenoid lattices."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help=(
            "Cases to run (default: all). "
            "Available: sb_on, varsol_on, sb_off"
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
        help=f"Number of turns to track (default: {N_TURNS}).",
    )
    parser.add_argument(
        "--n-part",
        type=int,
        default=N_PART,
        metavar="N",
        help=f"Number of macroparticles (default: {N_PART}).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive figure display (data and PDFs are still saved).",
    )
    args = parser.parse_args()

    if args.list_cases:
        for case in EMITT_CASES:
            print(f"{case['name']}: {case['title']}")
        return

    for case in _select_cases(args.cases):
        _run_emittance_evolution(
            case,
            n_turns=args.n_turns,
            n_part=args.n_part,
            with_progress=1,
        )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
