"""Emittance evolution + spin polarization from a SINGLE tracking run (FCC-ee solenoid).

Merges 014_emittance_evolution.py and 015_spin_polarization.py. Those two scripts
each performed their own full tracking run over the same lattice while differing in
only three places (twiss flavour, bunch initial emittance, spin initial condition),
so this script does one twiss, one bunch, one line.track() and one turn-by-turn
monitor, then runs both analyses off that monitor and writes both studies' npz+PDF
via the unchanged save_emitt_study/save_pol_study helpers.

Why spin is free: magnet_spin (xtrack/beam_elements/elements_src/track_magnet_radiation.h)
early-returns when all three spin components are zero, consumes no random numbers, and
never writes back to the orbit. So switching spin on costs CPU only, and at a fixed
--seed the emittance results here are bit-identical to a 014 run.

Bunch initial condition: 014's eq/3 (see --bunch-emitt-divisor), so the damping-rate
fit still has a transient to fit. The polarization fit therefore starts at
--pol-fit-start-turn (default 2000, matching 016's --turn-start) to skip both the
unmatched-spin-IC relaxation in the first turns and the bulk of the damping
transient. Note tau_z ~ 600 turns, so by turn 2000 the longitudinal plane -- which
drives dn/ddelta depolarization -- is within ~1% of equilibrium.

Coupling caveat (inherited from 014): with powered solenoids, x-y coupling may make
single-plane Courant-Snyder emittances approximate. A follow-up could extract epsilon
from the 4x4 transverse covariance eigenvalues.
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

from aperture_study_io import (
    format_tag_float,
    save_emitt_study,
    save_pol_study,
    variant_suffix,
)
from lattice_knobs import robust_twiss, set_lattice_knobs
from solenoid_params import (
    MAIN_SOLENOID_B0,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    order_tag,
)

plt.close("all")

HERE = Path(__file__).resolve().parent

GLOBAL_XY_LIMIT = 1.0
N_TURNS = 10_000
N_PART = 1000

# Bunch is generated at eq_nemitt / BUNCH_EMITT_DIVISOR (014's initial condition).
BUNCH_EMITT_DIVISOR = 3.0
# First turn included in the inline exponential depolarization fit; matches 016's
# --turn-start default so the inline fit and the 016 refit agree out of the box.
POL_FIT_START_TURN = 2000

# Divisors the two parent scripts used. A study's stem is left untagged when its
# bunch IC matches its own parent's, and tagged otherwise -- so an untagged file on
# disk always means "same physics as the legacy script would have written".
EMIT_REFERENCE_DIVISOR = 3.0   # 014
POL_REFERENCE_DIVISOR = 1.0    # 015


def _lattice_paths(tag, order_tag_str=""):
    # order_tag_str only applies to the SplineBoris path -- VariableSolenoid
    # is linear-only and has no transverse-order knob (see 00_overview.md).
    return (
        HERE / f"fccee_z_lcc_splineboris_solenoids_coupling_corrected_{tag}{order_tag_str}.json",
        HERE / f"fccee_z_lcc_varsol_solenoids_coupling_corrected_{tag}.json",
    )


def _build_cases(tag, order_tag_str=""):
    """Build the case list for a given field_tag (e.g. '2T', '3T') and
    order_tag (e.g. '', '_o2' -- SplineBoris-only, see _lattice_paths)."""
    splineboris_json, varsol_json = _lattice_paths(tag, order_tag_str)
    sb_tag = f"{tag}{order_tag_str}"
    return [
        dict(
            name="sb_on",
            model="SB",
            lattice_json=splineboris_json,
            with_solenoids=True,
            with_correctors=True,
            title=f"SplineBoris ({sb_tag}): solenoids powered + correction scheme",
        ),
        dict(
            name="varsol_on",
            model="VarSol",
            lattice_json=varsol_json,
            with_solenoids=True,
            with_correctors=True,
            title=f"VariableSolenoid ({tag}): solenoids powered + correction scheme",
        ),
        dict(
            name="sb_off",
            model="SB",
            lattice_json=splineboris_json,
            with_solenoids=False,
            with_correctors=False,
            title=f"SplineBoris ({sb_tag}): solenoids unpowered",
        ),
    ]


# sb_off is skipped by default on both counts: it is the bare-machine emittance
# baseline that does not depend on solenoid/correction changes, and its
# depolarization time is many orders of magnitude longer than any feasible turn
# count (no solenoid coupling to drive dn/ddelta). Pass --cases sb_off to include it.
DEFAULT_CASE_NAMES = ["sb_on", "varsol_on"]


def _configure_radiative_tracking(line):
    line.particle_ref.anomalous_magnetic_moment = 0.00115965218128
    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss()


def _bunch_variant_tag(divisor, *, reference):
    """'' when the bunch IC matches the reference script's, else '__bunchdiv<F>'.

    Filenames do not otherwise record the bunch initial condition, so without this
    a POL file produced here from an eq/3 bunch would be indistinguishable from one
    015 produced from an equilibrium bunch.
    """
    if divisor == reference:
        return ""
    return f"__bunchdiv{format_tag_float(divisor)}"


# --------------------------------------------------------------------------
# Emittance half (from 014_emittance_evolution.py)
# --------------------------------------------------------------------------


def _compute_geometric_emittances(mon, tw, n_turns):
    i0 = 0
    betx = float(tw["betx"][i0])
    alfx = float(tw["alfx"][i0])
    gamx = (1.0 + alfx**2) / betx
    bety = float(tw["bety"][i0])
    alfy = float(tw["alfy"][i0])
    gamy = (1.0 + alfy**2) / bety
    bets0 = float(tw["bets0"])
    x_co = float(tw["x"][i0])
    px_co = float(tw["px"][i0])
    y_co = float(tw["y"][i0])
    py_co = float(tw["py"][i0])
    zeta_co = float(tw["zeta"][i0])
    delta_co = float(tw["delta"][i0])

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

        # Compute second moments around the closed orbit, not around zero.
        x = mon.x[alive, t] - x_co
        px = mon.px[alive, t] - px_co
        y = mon.y[alive, t] - y_co
        py = mon.py[alive, t] - py_co
        zeta = mon.zeta[alive, t] - zeta_co
        delta = mon.delta[alive, t] - delta_co

        gemitt_x[t] = 0.5 * np.mean(gamx * x**2 + 2 * alfx * x * px + betx * px**2)
        gemitt_y[t] = 0.5 * np.mean(gamy * y**2 + 2 * alfy * y * py + bety * py**2)
        gemitt_z[t] = 0.5 * np.mean(zeta**2 / bets0 + bets0 * delta**2)

    return gemitt_x, gemitt_y, gemitt_z


def _fit_damping_rate(turns, gemitt, eps_eq_guess, eps_init):
    """Fit eps_eq and the damping rate alpha, with the initial emittance
    fixed to the value observed at the start of tracking.

    Model: eps(n) = (eps_init - eps_eq) * exp(-alpha * n) + eps_eq
    """

    def model(turns_arr, eps_eq, alpha):
        return (eps_init - eps_eq) * np.exp(-alpha * turns_arr) + eps_eq

    mask = np.isfinite(gemitt)
    if mask.sum() < 3:
        return np.nan, np.nan

    popt, _ = curve_fit(
        model,
        turns[mask],
        gemitt[mask],
        p0=[eps_eq_guess, 2.0 * 1e-3],
        maxfev=10_000,
    )
    return float(popt[0]), float(popt[1])


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

        analytic = _analytic_emittance(turns, eps_init, eq, damp)
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


# --------------------------------------------------------------------------
# Polarization half (from 015_spin_polarization.py)
# --------------------------------------------------------------------------


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


def _fit_exponential_depolarization(turns, polarization, t_rev0, *, turn_start=0):
    """Exponential fit P(n) = P0 * exp(-n / tau_depol), the actual functional
    form of radiative depolarization (a straight-line fit is only its
    linear-regime approximation for n << tau_depol).

    Fit by linear regression of ln(P) vs n (equivalent to a weighted
    nonlinear least-squares fit of P itself, and far more robust/cheap than
    a general nonlinear solve here). Only turns >= turn_start are fitted, to
    drop the early-turn transient (the initial all-along-y spin state relaxing
    onto the invariant spin field, plus the tail of the emittance damping).

    Returns (P0, slope [1/turn] -- initial dP/dn at n=0, for compatibility
    with the linear-regime annotation/npz schema --, tau_depol [turns],
    tau_depol [s]). P0 remains the n=0 extrapolation of the fitted exponential
    regardless of turn_start. tau_depol is inf if the fitted log-slope is
    non-negative (no significant decay resolved above noise -- e.g. too few
    turns/particles, or a case like sb_off where the true depolarization is
    negligible on any feasible turn count).
    """
    mask = np.isfinite(polarization) & (polarization > 0) & (turns >= turn_start)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan

    log_slope, log_intercept = np.polyfit(
        turns[mask], np.log(polarization[mask]), deg=1)
    p0 = float(np.exp(log_intercept))
    if log_slope < 0:
        tau_depol_turns = -1.0 / log_slope
        tau_depol_s = tau_depol_turns * t_rev0
    else:
        tau_depol_turns = np.inf
        tau_depol_s = np.inf
    slope = float(log_slope * p0)
    return p0, slope, float(tau_depol_turns), float(tau_depol_s)


def _plot_polarization_figure(
    *,
    turns,
    polarization,
    fit_p0,
    fit_tau_depol_turns,
    fit_tau_depol_s,
    fit_turn_start,
    p_inf,
    tau_pol_s,
    tau_depol_twiss_s,
    p_eq_twiss,
    p_eq_derived,
    title,
):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(turns, polarization, label="tracked")
    if fit_turn_start > 0:
        ax.axvline(fit_turn_start, color="0.5", ls=":", lw=1,
                   label=f"fit start (n={fit_turn_start})")

    if np.isfinite(fit_tau_depol_turns):
        fit_turns = turns[turns >= fit_turn_start]
        fit_curve = fit_p0 * np.exp(-fit_turns / fit_tau_depol_turns)
        ax.plot(fit_turns, fit_curve, "--", color="C3", label="exponential fit")

    ax.set_xlabel("turn")
    ax.set_ylabel(r"Polarization $P = |\langle \vec{s}\rangle|$")
    ax.legend(loc="lower right", fontsize=8)

    fit_lines = [f"fit (tracking, exponential decay, n>={fit_turn_start}):"]
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


# --------------------------------------------------------------------------
# Combined runner: one twiss, one bunch, one track, two analyses
# --------------------------------------------------------------------------


def _run_emittance_and_polarization(
    case,
    *,
    n_turns,
    n_part,
    with_progress,
    sexamp,
    tag,
    run_emitt=True,
    run_pol=True,
    bunch_emitt_divisor=BUNCH_EMITT_DIVISOR,
    pol_fit_start_turn=POL_FIT_START_TURN,
    seed=None,
):
    lattice_json = case["lattice_json"]
    title = case["title"]

    if not run_emitt and not run_pol:
        raise SystemExit("Nothing to do: both the emittance and polarization halves are disabled.")

    if not lattice_json.exists():
        raise SystemExit(
            f"Missing lattice file: {lattice_json.name}\n"
            f"Build it first for tag {tag!r} via "
            "004a_build_and_check_solenoids.py -> "
            "004b_install_solenoids_in_fcc_ring.py / "
            "004b_install_varsol_solenoids_in_fcc_ring.py -> "
            "004c_correct_solenoids_in_fcc_ring.py (each accepts --b0 and, "
            "for the SplineBoris path, --max-transverse-order)."
        )

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
    # robust_twiss falls back to a co-guess continuation (ramping sext_amp
    # from 1.0) if the direct closed-orbit search fails, e.g. for large
    # --sexamp values -- see lattice_knobs.robust_twiss.
    # polarization_analysis is requested only when the polarization half runs:
    # it implies spin=True and radiation_integrals=True, and pulls in a full
    # element-by-element R-matrix plus a spin-response track. Skipping it makes
    # --no-pol cost exactly what 014 costs and produce exactly what 014 produces.
    twiss_kwargs = dict(twiss_method="twiss6d", radiation_analysis=True, strengths=True)
    if run_pol:
        twiss_kwargs["polarization_analysis"] = True
    tw = robust_twiss(line, **twiss_kwargs)

    eq_x = float(tw.eq_gemitt_x)
    eq_y = float(tw.eq_gemitt_y)
    eq_z = float(tw.eq_gemitt_zeta)
    damp_turns = np.asarray(tw.damping_constants_turns, dtype=float)

    print(f"eq_gemitt_x   = {eq_x:.6e} m·rad")
    print(f"eq_gemitt_y   = {eq_y:.6e} m·rad")
    print(f"eq_gemitt_zeta= {eq_z:.6e} m·rad")
    print(f"damping_constants_turns = {damp_turns}")

    if run_pol:
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

    if seed is not None:
        # Seeds both the bunch generation and, via particles._init_random_number_generator
        # (called at the first track() because configure_radiation('quantum') sets
        # line._needs_rng), the whole quantum-radiation stream. This works only because
        # radiation is still 'mean' at twiss time, so the twiss's internal probe tracking
        # draws no np.random numbers -- moving configure_radiation('quantum') above the
        # twiss would silently break reproducibility here.
        np.random.seed(seed)
        print(f"RNG seeded with {seed} (bunch generation + quantum radiation)")

    print(f"Bunch initial emittance: eq / {bunch_emitt_divisor:g}")
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        nemitt_x=tw.eq_nemitt_x / bunch_emitt_divisor,
        nemitt_y=tw.eq_nemitt_y / bunch_emitt_divisor,
        sigma_z=np.sqrt(tw.eq_gemitt_zeta / bunch_emitt_divisor * tw.bets0),
        line=line,
        particle_on_co=tw.particle_on_co,
        engine="single-rf-harmonic",
    )
    particles.zeta += tw.zeta[0]
    particles.delta += tw.delta[0]

    # Spin IC, set explicitly in BOTH branches. twiss(spin=True) writes the invariant
    # spin field onto tw.particle_on_co, and build_particles copies particle_ref's spin
    # into every generated particle -- so after a polarization twiss the bunch is NOT
    # spin-zero by default, and --no-pol would silently do full spin tracking unless we
    # zero it here. Conversely spin_y=1 OVERWRITES the matched n0 direction, which is
    # why P drops by ~1e-5 between turn 0 and turn 1 before any real depolarization.
    particles.spin_x = 0.0
    particles.spin_y = 1.0 if run_pol else 0.0
    particles.spin_z = 0.0

    line.configure_radiation(model="quantum")
    line.discard_tracker()
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads="auto"))

    line.config.XTRACK_GLOBAL_XY_LIMIT = GLOBAL_XY_LIMIT
    halves = " + ".join(
        [n for n, on in (("emittance", run_emitt), ("polarization", run_pol)) if on])
    print(f"Tracking {n_part} particles for {n_turns} turns (quantum radiation) [{halves}]")
    line.track(
        particles,
        num_turns=n_turns,
        turn_by_turn_monitor=True,
        with_progress=with_progress,
    )
    mon = line.record_last_track
    turns = np.arange(n_turns)

    emitt_result = None
    pol_result = None

    if run_emitt:
        gemitt_x, gemitt_y, gemitt_z = _compute_geometric_emittances(mon, tw, n_turns)

        fit_eps_init = np.array([gemitt_x[0], gemitt_y[0], gemitt_z[0]])
        fit_results = [
            _fit_damping_rate(turns, gemitt_x, eq_x, fit_eps_init[0]),
            _fit_damping_rate(turns, gemitt_y, eq_y, fit_eps_init[1]),
            _fit_damping_rate(turns, gemitt_z, eq_z, fit_eps_init[2]),
        ]
        fit_eps_eq = np.array([r[0] for r in fit_results])
        fit_alpha = np.array([r[1] for r in fit_results])
        fit_tau = 1.0 / fit_alpha

        plane_symbols = ["x", "y", "ζ"]
        eq_twiss = [eq_x, eq_y, eq_z]
        for symbol, eps0, eps_eq_fit, alpha_fit, tau_fit, eq_tw, damp_tw in zip(
            plane_symbols, fit_eps_init, fit_eps_eq, fit_alpha, fit_tau, eq_twiss, damp_turns
        ):
            print(f"  Plane {symbol}:")
            print(f"    ε_{symbol},0  = {eps0:.6e} m·rad  (initial, from tracking)")
            print(
                f"    ε_{symbol},eq = {eps_eq_fit:.6e} m·rad  (fit)   "
                f"[Twiss: {eq_tw:.6e}]"
            )
            print(f"    α_{symbol}    = {alpha_fit:.6e} 1/turn  (fit)")
            print(
                f"    τ_{symbol}    = {tau_fit:.6e} turns  (fit)   "
                f"[Twiss damp_turns: {damp_tw:.6e}]"
            )

        # The bunch is generated with 1/bunch_emitt_divisor of the Twiss equilibrium
        # emittance (see generate_matched_gaussian_bunch above), so that is the
        # Twiss-predicted initial emittance for the fit comparison box.
        twiss_eps_init = np.array([eq_x, eq_y, eq_z]) / bunch_emitt_divisor

        fig_emitt = _plot_emittance_evolution_figure(
            turns=turns,
            gemitt_x=gemitt_x,
            gemitt_y=gemitt_y,
            gemitt_z=gemitt_z,
            eq_x=eq_x,
            eq_y=eq_y,
            eq_z=eq_z,
            damp_turns=damp_turns,
            fit_eps_eq=fit_eps_eq,
            fit_alpha=fit_alpha,
            twiss_eps_init=twiss_eps_init,
            title=title,
        )
        save_emitt_study(
            fig=fig_emitt,
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
            fit_eps_eq=fit_eps_eq,
            fit_alpha=fit_alpha,
            fit_tau=fit_tau,
            fit_eps_init=fit_eps_init,
            radiation="quantum",
            variant=(
                variant_suffix(sexamp=sexamp)
                + _bunch_variant_tag(bunch_emitt_divisor, reference=EMIT_REFERENCE_DIVISOR)
            ),
            sexamp=sexamp,
            field_tag=tag,
            bunch_emitt_divisor=bunch_emitt_divisor,
        )
        emitt_result = dict(
            turns=turns,
            gemitt_x=gemitt_x,
            gemitt_y=gemitt_y,
            gemitt_z=gemitt_z,
            fit_eps_eq=fit_eps_eq,
            fit_alpha=fit_alpha,
            fit_tau=fit_tau,
        )

    if run_pol:
        spin_x_mean, spin_y_mean, spin_z_mean, polarization = _compute_polarization(
            mon, n_turns
        )

        fit_p0, fit_slope, fit_tau_depol_turns, fit_tau_depol_s = (
            _fit_exponential_depolarization(
                turns, polarization, t_rev0, turn_start=pol_fit_start_turn)
        )

        print(f"  P(0) tracked                = {polarization[0]:.6f}")
        print(f"  P0 (fit intercept)          = {fit_p0:.6f}")
        print(
            f"  tau_depol (fit, tracking)   = {fit_tau_depol_s:.6e} s "
            f"({fit_tau_depol_turns:.6e} turns, fitted from turn "
            f"{pol_fit_start_turn})   [Twiss: {tau_depol_twiss_s:.6e} s]"
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

        fig_pol = _plot_polarization_figure(
            turns=turns,
            polarization=polarization,
            fit_p0=fit_p0,
            fit_tau_depol_turns=fit_tau_depol_turns,
            fit_tau_depol_s=fit_tau_depol_s,
            fit_turn_start=pol_fit_start_turn,
            p_inf=p_inf,
            tau_pol_s=tau_pol_s,
            tau_depol_twiss_s=tau_depol_twiss_s,
            p_eq_twiss=p_eq_twiss,
            p_eq_derived=p_eq_derived,
            title=title,
        )
        save_pol_study(
            fig=fig_pol,
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
            radiation="quantum",
            variant=(
                variant_suffix(sexamp=sexamp)
                + _bunch_variant_tag(bunch_emitt_divisor, reference=POL_REFERENCE_DIVISOR)
            ),
            sexamp=sexamp,
            field_tag=tag,
            fit_turn_start=pol_fit_start_turn,
            bunch_emitt_divisor=bunch_emitt_divisor,
        )
        pol_result = dict(
            turns=turns,
            polarization=polarization,
            fit_p0=fit_p0,
            fit_slope=fit_slope,
            fit_tau_depol_s=fit_tau_depol_s,
            p_eq_derived=p_eq_derived,
        )

    print(f"[{title}] Combined run complete ({halves}).")
    return dict(turns=turns, emitt=emitt_result, pol=pol_result)


def _select_cases(case_names, cases_by_name):
    if not case_names:
        return [cases_by_name[name] for name in DEFAULT_CASE_NAMES]

    unknown = [name for name in case_names if name not in cases_by_name]
    if unknown:
        valid = ", ".join(cases_by_name)
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Choose from: {valid}")

    return [cases_by_name[name] for name in case_names]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run combined emittance-evolution + spin-polarization studies for "
            "FCC-ee solenoid lattices from a single tracking run per case."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help=(
            "Cases to run (default: sb_on, varsol_on -- sb_off is skipped by "
            "default since the bare-machine baseline doesn't change and its "
            "depolarization is unresolvable by tracking; pass --cases sb_off "
            "explicitly to include it). Available: sb_on, varsol_on, sb_off"
        ),
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available cases and exit.",
    )
    add_b0_argument(parser, default=MAIN_SOLENOID_B0)
    add_max_order_argument(parser)
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
        "--sexamp",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Sextupole amplification knob (default: 1.0).",
    )
    halves = parser.add_mutually_exclusive_group()
    halves.add_argument(
        "--no-emitt",
        action="store_true",
        help=(
            "Skip the emittance half (track and analyse spin polarization only). "
            "The bunch initial condition is unchanged, so this stays a strict "
            "subset of the combined run."
        ),
    )
    halves.add_argument(
        "--no-pol",
        action="store_true",
        help=(
            "Skip the spin-polarization half (emittance only). Also drops "
            "polarization_analysis from the Twiss and leaves spin at zero, so "
            "this reproduces 014_emittance_evolution.py exactly."
        ),
    )
    parser.add_argument(
        "--pol-fit-start-turn",
        type=int,
        default=POL_FIT_START_TURN,
        metavar="N",
        help=(
            "First turn included in the inline exponential depolarization fit "
            f"(default: {POL_FIT_START_TURN}, matching 016's --turn-start). Drops "
            "the early-turn spin-IC transient and the bulk of the emittance damping."
        ),
    )
    parser.add_argument(
        "--bunch-emitt-divisor",
        type=float,
        default=BUNCH_EMITT_DIVISOR,
        metavar="F",
        help=(
            "Generate the bunch at eq_nemitt / F (default: "
            f"{BUNCH_EMITT_DIVISOR:g}, i.e. 014's initial condition, which leaves a "
            "damping transient to fit). Pass 1 for 015's equilibrium bunch. Studies "
            "whose bunch IC differs from their parent script's are tagged "
            "'__bunchdiv<F>' in the output filename."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Seed numpy's global RNG before bunch generation, making both the bunch "
            "and the quantum-radiation stream reproducible (default: unseeded)."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive figure display (data and PDFs are still saved).",
    )
    args = parser.parse_args()

    run_emitt = not args.no_emitt
    run_pol = not args.no_pol

    if args.bunch_emitt_divisor <= 0:
        raise SystemExit(
            f"--bunch-emitt-divisor must be positive, got {args.bunch_emitt_divisor:g}.")

    # Fail before tracking rather than returning silent NaN fits afterwards.
    if run_pol and args.pol_fit_start_turn >= args.n_turns:
        raise SystemExit(
            f"--pol-fit-start-turn {args.pol_fit_start_turn} is >= --n-turns "
            f"{args.n_turns}; nothing would be left to fit. Lower it (short smoke "
            "runs typically want something like --pol-fit-start-turn 20) or pass "
            "--no-pol."
        )

    tag = field_tag(args.b0)
    order_tag_str = order_tag(args.max_transverse_order)
    cases = _build_cases(tag, order_tag_str)
    cases_by_name = {case["name"]: case for case in cases}

    if args.list_cases:
        print(
            f"Field tag: {tag}{order_tag_str} (--b0 {args.b0:g}, "
            f"--max-transverse-order {args.max_transverse_order})")
        for case in cases:
            print(f"{case['name']}: {case['title']}")
        return

    for case in _select_cases(args.cases, cases_by_name):
        _run_emittance_and_polarization(
            case,
            n_turns=args.n_turns,
            n_part=args.n_part,
            with_progress=1,
            sexamp=args.sexamp,
            tag=f"{tag}{order_tag_str}",
            run_emitt=run_emitt,
            run_pol=run_pol,
            bunch_emitt_divisor=args.bunch_emitt_divisor,
            pol_fit_start_turn=args.pol_fit_start_turn,
            seed=args.seed,
        )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
