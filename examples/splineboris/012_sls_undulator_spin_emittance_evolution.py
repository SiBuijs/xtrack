"""SLS undulator spin & emittance evolution (SplineBoris vs Multipole undulator model).

Tracks quantum-radiation emittance damping and radiative spin depolarization
through the SLS ring for 7 cases: no undulator, a single SplineBoris-based
undulator at either of two straight sections (``ars11_uind_0210_1`` /
``ars11_uind_0610_1``), both together, and the same 3 configurations built
instead with ``TubeFitter.to_multipole_line()`` (a coarse thick-Multipole
undulator model, see ``xtrack/_temp/splineboris/tube_fitter.py``).

The undulator itself is fit once from the shared SLS field map via
``TubeFitter`` (the canonical fitter, see
``examples/splineboris/claude_notes/tubefitter_canonical_and_upstream_merge.md``)
and its orbit corrected with 4 thin dipole correctors, following the same
recipe as ``009_sls_undulators_closed_spin_radiation.py``.

``test_data/sls/sls.madx`` has no RF cavities, so a synthetic one is added
(same parameters as ``examples/radiation_thick/000_sls_radiation.py``) --
without it there is no restoring force for the energy lost to synchrotron
radiation and quantum tracking would not have any equilibrium to reach.

Equilibrium polarization is computed the same way as
``examples/fcc_ee_solenoid/015_spin_polarization.py`` on the ``fcc_da_ma``
branch: ``P_eq = P_inf / (1 + tau_pol / tau_depol)`` (eq. 8.37), with
``P_inf``/``tau_pol`` from Twiss (``spin_polarization_inf_no_depol`` /
``spin_t_pol_component_s``) and ``tau_depol`` from a straight-line fit of the
tracked polarization -- here restricted to turns >= ``FIT_START_TURN``
(default 2000) to skip the initial transient before the bunch/spin
distribution has settled.

An optional ``--offset-x`` applies a rigid transverse (horizontal)
misalignment to the undulator elements before the orbit is corrected --
same mechanism as ``007_sls_undulator_tune_displacement.py``'s
``line_sls[nn].shift_x = dx`` scan. With ``--corrector-strategy shared``
(default), the 4 correctors are matched once on the (misaligned)
SplineBoris line and the *same* corrector strengths are reused verbatim on
the Multipole line, rather than matching each model separately -- this
isolates the field-model difference (the actual subject of this study)
instead of letting each model's own correction re-absorb part of it. Pass
``--corrector-strategy separate`` to match each model's correctors
independently instead.

Figures are saved as PDF into ``PLOT_DIR`` (default
``~/cernbox/Pictures/SLS_Undulator_Studies``, override with
``SLS_UNDULATOR_PLOT_DIR`` -- same override mechanism as
``FCC_SOLENOID_PLOT_DIR`` in ``examples/fcc_ee_solenoid/aperture_study_io.py``
on ``fcc_da_ma``); raw per-turn arrays (plus everything needed to redraw the
figure) are saved as ``.npz`` next to this script. Pass ``--replot`` (with
the same ``--cases``/``--n-turns``/``--n-part``/``--offset-x`` used
originally) to regenerate the PDFs from that saved data without rerunning
the fit/build/track pipeline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xobjects as xo
import xpart as xp
import xtrack as xt

from xtrack._temp.splineboris.tube_fitter import TubeFitter

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
RESULTS_DIR = HERE / "results"
PLOT_DIR = Path(
    os.environ.get(
        "SLS_UNDULATOR_PLOT_DIR",
        Path.home() / "cernbox" / "Pictures" / "SLS_Undulator_Studies",
    )
)

E0 = 2.7e9  # eV
MULTIPOLE_ORDER = 3
ANOMALOUS_MAGNETIC_MOMENT = 0.00115965218128

# Synthetic RF cavity -- sls.madx has none, see module docstring.
VRF = 1.8e6
FRF = 499.6e6
PHASERF = np.pi

WIGGLER_PLACES = {
    "0210": "ars11_uind_0210_1",
    "0610": "ars11_uind_0610_1",
}

N_TURNS = 10_000
N_PART = 1000
FIT_START_TURN = 2000
GLOBAL_XY_LIMIT = 1.0
MIN_NEMITT = 1e-13  # floor for bunch generation when Twiss predicts ~0 (e.g. no coupling)
OFFSET_X = 0.0
CORRECTOR_STRATEGY = "shared"


def format_tag_float(value: float) -> str:
    """Format a float for filename tags, e.g. 0.0005 -> '0p0005' (matches
    examples/fcc_ee_solenoid/aperture_study_io.py on fcc_da_ma)."""
    if value == int(value):
        text = f"{int(value)}.0"
    else:
        text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def _case_stem(name, n_part, n_turns, offset_x):
    offset_tag = f"__xoff{format_tag_float(offset_x)}" if offset_x != 0.0 else ""
    return f"SLS_{name}_{n_part}p_{n_turns}t{offset_tag}"

CASES = [
    dict(name="no_undulators", kind=None, places=[], title="No undulators"),
    dict(name="sb_0210", kind="splineboris", places=["0210"],
         title="SplineBoris undulator @ ars11_uind_0210_1"),
    dict(name="sb_0610", kind="splineboris", places=["0610"],
         title="SplineBoris undulator @ ars11_uind_0610_1"),
    dict(name="sb_both", kind="splineboris", places=["0210", "0610"],
         title="SplineBoris undulators @ both locations"),
    dict(name="mult_0210", kind="multipole", places=["0210"],
         title="Multipole undulator @ ars11_uind_0210_1"),
    dict(name="mult_0610", kind="multipole", places=["0610"],
         title="Multipole undulator @ ars11_uind_0610_1"),
    dict(name="mult_both", kind="multipole", places=["0210", "0610"],
         title="Multipole undulators @ both locations"),
]
CASES_BY_NAME = {case["name"]: case for case in CASES}


# ------------------------------------------------------------------ #
# Undulator construction (fit once, reused across cases)
# ------------------------------------------------------------------ #

def _fit_undulator():
    field_map_path = REPO_ROOT / "test_data" / "sls" / "undulator_field_map.txt"
    df_raw_data = pd.read_csv(
        field_map_path, sep=r"\s+", header=None,
        names=["X", "Y", "Z", "Bx", "By", "Bs"],
    ).set_index(["X", "Y", "Z"])

    fitter = TubeFitter(
        raw_data=df_raw_data,
        n_frames=400,
        distance_unit=1e-3,
        deg=MULTIPOLE_ORDER - 1,
        field_tol=1e-3,
    )
    fitter.fit()
    return fitter


CORRECTOR_KNOBS = (
    "k0l_corr1", "k0l_corr2", "k0l_corr3", "k0l_corr4",
    "k0sl_corr1", "k0sl_corr2", "k0sl_corr3", "k0sl_corr4",
)


def _apply_shift_x(undulator_line, offset_x):
    """Rigid horizontal misalignment of every element in the line (xtrack
    elements have a built-in shift_x/shift_y alignment offset). Applied to
    the whole undulator assembly -- field elements *and* correctors, since
    physically the correctors are mounted on the same girder as the
    undulator and move with it -- same mechanism as
    007_sls_undulator_tune_displacement.py's ``line_sls[nn].shift_x = dx``.
    """
    if offset_x == 0.0:
        return
    for name in undulator_line.element_names:
        el = undulator_line.element_dict[name]
        if hasattr(el, "shift_x"):
            el.shift_x = offset_x


def _correct_undulator_orbit(undulator_line, offset_x=0.0, corrector_values=None):
    """Install 4 thin dipole correctors, following 009's recipe (corrector
    positions -- edges + 0.02/0.1 m insets -- are not tuned to this
    particular field map, just a reasonable spread to control position and
    angle independently), then apply the rigid misalignment (if any) to the
    whole assembly including the correctors, and finally correct the orbit.

    If ``corrector_values`` is None, match a closed (x=y=0) orbit through
    the undulator and return the resulting {knob: value} dict. Otherwise,
    skip matching and set the correctors directly to the given values --
    used to reuse one model's matched correctors on the other for a fair
    field-model comparison (see module docstring).
    """
    env = undulator_line.env
    l_und = undulator_line.get_length()

    for kk in CORRECTOR_KNOBS:
        env[kk] = 0.0
    env.new("corr1", xt.Multipole, knl=["k0l_corr1"], ksl=["k0sl_corr1"])
    env.new("corr2", xt.Multipole, knl=["k0l_corr2"], ksl=["k0sl_corr2"])
    env.new("corr3", xt.Multipole, knl=["k0l_corr3"], ksl=["k0sl_corr3"])
    env.new("corr4", xt.Multipole, knl=["k0l_corr4"], ksl=["k0sl_corr4"])

    undulator_line.insert([
        env.place("corr1", at=0.02),
        env.place("corr2", at=0.1),
        env.place("corr3", at=l_und - 0.1),
        env.place("corr4", at=l_und - 0.02),
    ], s_tol=5e-3)

    _apply_shift_x(undulator_line, offset_x)

    if corrector_values is None:
        opt = undulator_line.match(
            solve=False,
            betx=0, bety=0,
            only_orbit=True,
            include_collective=True,
            vary=xt.VaryList([
                "k0l_corr1", "k0sl_corr1", "k0l_corr2", "k0sl_corr2",
                "k0l_corr3", "k0sl_corr3", "k0l_corr4", "k0sl_corr4",
            ], step=1e-6),
            targets=[
                xt.TargetSet(x=0, px=0, y=0, py=0.0, at=xt.END),
                xt.TargetSet(x=0.0, y=0, at="corr2"),
                xt.TargetSet(x=0.0, y=0, at="corr3"),
            ],
        )
        opt.step(2)
        corrector_values = {kk: float(env[kk]) for kk in CORRECTOR_KNOBS}
    else:
        for kk in CORRECTOR_KNOBS:
            env[kk] = corrector_values[kk]

    undulator_line.discard_tracker()
    return corrector_values


def _build_undulator_lines(offset_x=OFFSET_X, corrector_strategy=CORRECTOR_STRATEGY):
    print("Fitting SLS undulator field map with TubeFitter...")
    fitter = _fit_undulator()

    print("Building + correcting SplineBoris-based undulator line...")
    line_sb = fitter.to_line(multipole_order=MULTIPOLE_ORDER, steps_per_point=1)
    line_sb.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)
    corrector_values = _correct_undulator_orbit(line_sb, offset_x=offset_x)

    print("Building + correcting Multipole-based undulator line...")
    line_mult = fitter.to_multipole_line(multipole_order=MULTIPOLE_ORDER, p0c=E0, q0=1.0)
    line_mult.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)
    if corrector_strategy == "shared":
        _correct_undulator_orbit(line_mult, offset_x=offset_x, corrector_values=corrector_values)
    elif corrector_strategy == "separate":
        _correct_undulator_orbit(line_mult, offset_x=offset_x)
    else:
        raise ValueError(
            f"corrector_strategy must be 'shared' or 'separate', got {corrector_strategy!r}")

    return {"splineboris": line_sb, "multipole": line_mult}


# ------------------------------------------------------------------ #
# SLS ring
# ------------------------------------------------------------------ #

def _load_sls_ring():
    madx_file = REPO_ROOT / "test_data" / "sls" / "sls.madx"
    env = xt.load(str(madx_file))
    line = env.ring
    line.configure_bend_model(core="mat-kick-mat")
    line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)

    # sls.madx has no RF cavities -- add a synthetic one so radiation losses
    # have a restoring force (same parameters as
    # examples/radiation_thick/000_sls_radiation.py).
    env["vrf"] = VRF
    env["frf"] = FRF
    env["phaserf"] = PHASERF
    line.insert(env.new("cav", "Cavity", voltage="vrf", frequency="frf",
                         phase="phaserf", at=0))

    line.particle_ref.anomalous_magnetic_moment = ANOMALOUS_MAGNETIC_MOMENT
    return env, line


def _insert_undulators(env, line, kind, places, undulator_lines):
    if not places:
        return
    # The undulator line's elements live in TubeFitter's own implicit
    # Environment (to_line()/to_multipole_line() don't take an `env`
    # argument) -- import them into the ring's env first so `line.insert()`
    # can resolve them, matching 004b_undulators_in_sls_ring.py's pattern.
    # A fresh import per placement gives each insertion its own element
    # instances -- compensate_radiation_energy_loss() refuses to run on a
    # line with elements repeated at multiple positions.
    undulator_line = undulator_lines[kind]
    tt = line.get_table()
    for key in places:
        wig_place = WIGGLER_PLACES[key]
        s_at = float(tt["s", wig_place])
        import_name = f"{kind}_undulator_{key}"
        env.import_line(undulator_line, line_name=import_name)
        print(f"  inserting {kind} undulator at {wig_place} (s={s_at:.3f} m)")
        line.insert(env[import_name], anchor="start", at=s_at)


# ------------------------------------------------------------------ #
# Per-turn observables (emittance from 014, polarization from 015 --
# fcc_ee_solenoid on fcc_da_ma)
# ------------------------------------------------------------------ #

def _compute_geometric_emittances(mon, tw, n_turns):
    i0 = 0
    betx = float(tw["betx"][i0]); alfx = float(tw["alfx"][i0])
    gamx = (1.0 + alfx**2) / betx
    bety = float(tw["bety"][i0]); alfy = float(tw["alfy"][i0])
    gamy = (1.0 + alfy**2) / bety
    bets0 = float(tw["bets0"])
    x_co = float(tw["x"][i0]); px_co = float(tw["px"][i0])
    y_co = float(tw["y"][i0]); py_co = float(tw["py"][i0])
    zeta_co = float(tw["zeta"][i0]); delta_co = float(tw["delta"][i0])

    has_state = hasattr(mon, "state") and mon.state is not None

    gemitt_x = np.empty(n_turns)
    gemitt_y = np.empty(n_turns)
    gemitt_z = np.empty(n_turns)

    for t in range(n_turns):
        if has_state:
            alive = mon.state[:, t] > 0
            if not np.any(alive):
                gemitt_x[t] = gemitt_y[t] = gemitt_z[t] = np.nan
                continue
        else:
            alive = slice(None)

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


def _compute_polarization(mon, n_turns):
    has_state = hasattr(mon, "state") and mon.state is not None

    spin_x_mean = np.empty(n_turns)
    spin_y_mean = np.empty(n_turns)
    spin_z_mean = np.empty(n_turns)

    for t in range(n_turns):
        if has_state:
            alive = mon.state[:, t] > 0
            if not np.any(alive):
                spin_x_mean[t] = spin_y_mean[t] = spin_z_mean[t] = np.nan
                continue
        else:
            alive = slice(None)

        spin_x_mean[t] = np.mean(mon.spin_x[alive, t])
        spin_y_mean[t] = np.mean(mon.spin_y[alive, t])
        spin_z_mean[t] = np.mean(mon.spin_z[alive, t])

    polarization = np.sqrt(spin_x_mean**2 + spin_y_mean**2 + spin_z_mean**2)
    return spin_x_mean, spin_y_mean, spin_z_mean, polarization


def _fit_linear_depolarization(turns, polarization, t_rev0, fit_start_turn):
    """Straight-line fit P(n) = P0 + slope*n over turns >= fit_start_turn,
    matching examples/fcc_ee_solenoid/015_spin_polarization.py's
    _fit_linear_depolarization (fcc_da_ma branch), restricted to turns
    beyond fit_start_turn to skip the initial transient.
    """
    mask = np.isfinite(polarization)
    mask[:fit_start_turn] = False
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


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #

def _plot_case_figure(*, turns, gemitt_x, gemitt_y, gemitt_z, eq_x, eq_y, eq_z,
                       polarization, fit_p0, fit_slope, fit_tau_depol_turns,
                       fit_start_turn, p_inf, tau_pol_s, p_eq_twiss,
                       p_eq_derived, title):
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 9.0), sharex=True)

    emitt_series = [
        (gemitt_x, eq_x, r"$\varepsilon_x$ [m$\cdot$rad]"),
        (gemitt_y, eq_y, r"$\varepsilon_y$ [m$\cdot$rad]"),
        (gemitt_z, eq_z, r"$\varepsilon_\zeta$ [m$\cdot$rad]"),
    ]
    for ax, (gemitt, eq, ylabel) in zip(axes[:3], emitt_series):
        ax.plot(turns, gemitt, label="tracked")
        ax.axhline(eq, linestyle="--", color="C2", label=r"$\varepsilon_\mathrm{eq}$ (Twiss)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=7)

    ax_pol = axes[3]
    ax_pol.plot(turns, polarization, label="tracked")
    if np.isfinite(fit_tau_depol_turns):
        fit_curve = fit_p0 + fit_slope * turns
        ax_pol.plot(turns[fit_start_turn:], fit_curve[fit_start_turn:], "--",
                    color="C3", label=f"linear fit (turn>={fit_start_turn})")
    ax_pol.set_ylabel(r"$P = |\langle \vec{s}\rangle|$")
    ax_pol.set_xlabel("turn")
    ax_pol.legend(loc="lower left", fontsize=7)

    info_lines = [
        f"P_inf (Twiss) = {p_inf:.6f}",
        f"tau_pol (Twiss) = {tau_pol_s:.3e} s",
        f"P_eq (Twiss direct) = {p_eq_twiss:.3e}",
        f"P_eq (derived, fit tau_depol) = {p_eq_derived:.3e}",
    ]
    ax_pol.text(0.98, 0.98, "\n".join(info_lines), transform=ax_pol.transAxes,
                ha="right", va="top", fontsize=6.5, linespacing=1.4,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"))

    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# Per-case run
# ------------------------------------------------------------------ #

def _run_case(case, undulator_lines, *, n_turns, n_part, fit_start_turn, offset_x,
              with_progress):
    name = case["name"]
    title = case["title"]
    print("\n" + "=" * 80)
    print(f"CASE: {name} -- {title}")
    print("=" * 80)

    env, line = _load_sls_ring()
    _insert_undulators(env, line, case["kind"], case["places"], undulator_lines)

    line.discard_tracker()
    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss()

    line.discard_tracker()
    tw = line.twiss6d(radiation_analysis=True, polarization_analysis=True, strengths=True)

    eq_x = float(tw.eq_gemitt_x)
    eq_y = float(tw.eq_gemitt_y)
    eq_z = float(tw.eq_gemitt_zeta)
    p_inf = float(tw.spin_polarization_inf_no_depol)
    tau_pol_s = float(tw.spin_t_pol_component_s)
    p_eq_twiss = float(tw.spin_polarization_eq)
    tau_depol_twiss_s = float(tw.spin_t_depol_component_s)
    t_rev0 = float(tw.t_rev0)

    print(f"eq_gemitt_x    = {eq_x:.6e} m*rad")
    print(f"eq_gemitt_y    = {eq_y:.6e} m*rad")
    print(f"eq_gemitt_zeta = {eq_z:.6e} m*rad")
    print(f"P_inf = {p_inf:.6f}   tau_pol = {tau_pol_s:.6e} s")
    print(f"tau_depol (Twiss) = {tau_depol_twiss_s:.6e} s   P_eq (Twiss) = {p_eq_twiss:.6e}")

    nemitt_x = max(float(tw.eq_nemitt_x), MIN_NEMITT)
    nemitt_y = max(float(tw.eq_nemitt_y), MIN_NEMITT)

    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=np.sqrt(eq_z * tw.bets0),
        line=line,
        particle_on_co=tw.particle_on_co,
        engine="single-rf-harmonic",
    )
    particles.zeta += tw.zeta[0]
    particles.delta += tw.delta[0]

    # Fully polarized along y at t=0.
    particles.spin_x = 0.0
    particles.spin_y = 1.0
    particles.spin_z = 0.0

    line.configure_radiation(model="quantum")
    line.discard_tracker()
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads="auto"))
    line.config.XTRACK_GLOBAL_XY_LIMIT = GLOBAL_XY_LIMIT

    print(f"Tracking {n_part} particles for {n_turns} turns (quantum radiation)")
    line.track(particles, num_turns=n_turns, turn_by_turn_monitor=True,
               with_progress=with_progress)
    mon = line.record_last_track

    turns = np.arange(n_turns)
    gemitt_x, gemitt_y, gemitt_z = _compute_geometric_emittances(mon, tw, n_turns)
    spin_x_mean, spin_y_mean, spin_z_mean, polarization = _compute_polarization(mon, n_turns)

    fit_p0, fit_slope, fit_tau_depol_turns, fit_tau_depol_s = _fit_linear_depolarization(
        turns, polarization, t_rev0, fit_start_turn)

    if np.isfinite(fit_tau_depol_s) and fit_tau_depol_s > 0:
        p_eq_derived = p_inf / (1.0 + tau_pol_s / fit_tau_depol_s)
    else:
        p_eq_derived = np.nan

    print(f"  P(0) tracked = {polarization[0]:.6f}")
    print(f"  tau_depol (fit, turn>={fit_start_turn}) = {fit_tau_depol_s:.6e} s "
          f"[Twiss: {tau_depol_twiss_s:.6e} s]")
    print(f"  P_eq (derived) = {p_eq_derived:.6e}   [Twiss direct: {p_eq_twiss:.6e}]")

    fig = _plot_case_figure(
        turns=turns, gemitt_x=gemitt_x, gemitt_y=gemitt_y, gemitt_z=gemitt_z,
        eq_x=eq_x, eq_y=eq_y, eq_z=eq_z, polarization=polarization,
        fit_p0=fit_p0, fit_slope=fit_slope, fit_tau_depol_turns=fit_tau_depol_turns,
        fit_start_turn=fit_start_turn, p_inf=p_inf, tau_pol_s=tau_pol_s,
        p_eq_twiss=p_eq_twiss, p_eq_derived=p_eq_derived, title=title,
    )
    stem = _case_stem(name, n_part, n_turns, offset_x)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PLOT_DIR / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")

    RESULTS_DIR.mkdir(exist_ok=True)
    npz_path = RESULTS_DIR / f"{stem}.npz"
    np.savez(
        npz_path,
        name=name, title=title,
        turns=turns, gemitt_x=gemitt_x, gemitt_y=gemitt_y, gemitt_z=gemitt_z,
        spin_x_mean=spin_x_mean, spin_y_mean=spin_y_mean, spin_z_mean=spin_z_mean,
        polarization=polarization,
        eq_gemitt_x=eq_x, eq_gemitt_y=eq_y, eq_gemitt_zeta=eq_z,
        p_inf=p_inf, tau_pol_s=tau_pol_s, tau_depol_twiss_s=tau_depol_twiss_s,
        p_eq_twiss=p_eq_twiss,
        fit_p0=fit_p0, fit_slope=fit_slope, fit_tau_depol_turns=fit_tau_depol_turns,
        fit_tau_depol_s=fit_tau_depol_s, p_eq_derived=p_eq_derived,
        t_rev0=t_rev0, fit_start_turn=fit_start_turn, offset_x=offset_x,
    )
    print(f"[{name}] done -- saved {pdf_path} / {npz_path.name}")

    return dict(
        name=name, title=title, eq_x=eq_x, eq_y=eq_y, eq_z=eq_z,
        p_inf=p_inf, tau_pol_s=tau_pol_s, tau_depol_twiss_s=tau_depol_twiss_s,
        p_eq_twiss=p_eq_twiss, fit_tau_depol_s=fit_tau_depol_s, p_eq_derived=p_eq_derived,
    )


def _replot_case(case, *, n_turns, n_part, offset_x):
    """Regenerate a case's PDF from its saved .npz, without rerunning the
    fit/build/track pipeline."""
    name = case["name"]
    stem = _case_stem(name, n_part, n_turns, offset_x)
    npz_path = RESULTS_DIR / f"{stem}.npz"
    if not npz_path.exists():
        raise SystemExit(
            f"No saved data for case {name!r} at {npz_path} -- run the same "
            f"command without --replot first (matching --n-turns/--n-part/"
            f"--offset-x)."
        )

    data = np.load(npz_path, allow_pickle=True)
    title = str(data["title"])
    eq_x = float(data["eq_gemitt_x"])
    eq_y = float(data["eq_gemitt_y"])
    eq_z = float(data["eq_gemitt_zeta"])
    p_inf = float(data["p_inf"])
    tau_pol_s = float(data["tau_pol_s"])
    tau_depol_twiss_s = float(data["tau_depol_twiss_s"])
    p_eq_twiss = float(data["p_eq_twiss"])
    fit_tau_depol_s = float(data["fit_tau_depol_s"])
    p_eq_derived = float(data["p_eq_derived"])
    fit_start_turn = int(data["fit_start_turn"])

    fig = _plot_case_figure(
        turns=data["turns"], gemitt_x=data["gemitt_x"], gemitt_y=data["gemitt_y"],
        gemitt_z=data["gemitt_z"], eq_x=eq_x, eq_y=eq_y, eq_z=eq_z,
        polarization=data["polarization"],
        fit_p0=float(data["fit_p0"]), fit_slope=float(data["fit_slope"]),
        fit_tau_depol_turns=float(data["fit_tau_depol_turns"]),
        fit_start_turn=fit_start_turn, p_inf=p_inf, tau_pol_s=tau_pol_s,
        p_eq_twiss=p_eq_twiss, p_eq_derived=p_eq_derived, title=title,
    )
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PLOT_DIR / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[{name}] replotted from {npz_path.name} -- saved {pdf_path}")

    return dict(
        name=name, title=title, eq_x=eq_x, eq_y=eq_y, eq_z=eq_z,
        p_inf=p_inf, tau_pol_s=tau_pol_s, tau_depol_twiss_s=tau_depol_twiss_s,
        p_eq_twiss=p_eq_twiss, fit_tau_depol_s=fit_tau_depol_s, p_eq_derived=p_eq_derived,
    )


def _print_summary(results):
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    header = (f"{'case':12s} {'eq_gemitt_x':>12s} {'eq_gemitt_y':>12s} "
              f"{'tau_depol[s]':>14s} {'P_eq(Twiss)':>12s} {'P_eq(derived)':>14s}")
    print(header)
    for r in results:
        print(f"{r['name']:12s} {r['eq_x']:12.4e} {r['eq_y']:12.4e} "
              f"{r['fit_tau_depol_s']:14.4e} {r['p_eq_twiss']:12.4e} "
              f"{r['p_eq_derived']:14.4e}")


def main():
    parser = argparse.ArgumentParser(
        description="SLS undulator spin & emittance evolution "
                     "(SplineBoris vs Multipole undulator model)."
    )
    parser.add_argument(
        "--cases", nargs="+", metavar="CASE",
        help=f"Cases to run (default: all 7). Available: {', '.join(CASES_BY_NAME)}",
    )
    parser.add_argument("--list-cases", action="store_true",
                         help="List available cases and exit.")
    parser.add_argument("--n-turns", type=int, default=N_TURNS, metavar="N",
                         help=f"Number of turns to track (default: {N_TURNS}).")
    parser.add_argument("--n-part", type=int, default=N_PART, metavar="N",
                         help=f"Number of macroparticles (default: {N_PART}).")
    parser.add_argument("--fit-start-turn", type=int, default=FIT_START_TURN, metavar="N",
                         help="First turn included in the linear depolarization "
                              f"fit (default: {FIT_START_TURN}).")
    parser.add_argument("--offset-x", type=float, default=OFFSET_X, metavar="METERS",
                         help="Rigid horizontal misalignment applied to the undulator "
                              "field elements before orbit correction, e.g. 0.5e-3 for "
                              f"0.5 mm (default: {OFFSET_X}).")
    parser.add_argument("--corrector-strategy", choices=["shared", "separate"],
                         default=CORRECTOR_STRATEGY,
                         help="'shared' (default): match correctors once on the "
                              "SplineBoris line and reuse them on the Multipole line. "
                              "'separate': match each model's correctors independently.")
    parser.add_argument("--replot", action="store_true",
                         help="Regenerate PDFs from previously saved .npz data "
                              "(same RESULTS_DIR/results/) instead of rerunning the "
                              "fit/build/track pipeline. Requires --cases/--n-turns/"
                              "--n-part/--offset-x to match the original run.")
    parser.add_argument("--no-show", action="store_true",
                         help="Skip interactive figure display (PDF/npz still saved).")
    args = parser.parse_args()

    if args.list_cases:
        for case in CASES:
            print(f"{case['name']}: {case['title']}")
        return

    if args.cases:
        unknown = [name for name in args.cases if name not in CASES_BY_NAME]
        if unknown:
            valid = ", ".join(CASES_BY_NAME)
            raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Choose from: {valid}")
        selected = [CASES_BY_NAME[name] for name in args.cases]
    else:
        selected = CASES

    if args.replot:
        results = [
            _replot_case(case, n_turns=args.n_turns, n_part=args.n_part,
                         offset_x=args.offset_x)
            for case in selected
        ]
        _print_summary(results)
        if not args.no_show:
            plt.show()
        return

    needs_sb = any(case["kind"] == "splineboris" for case in selected)
    needs_mult = any(case["kind"] == "multipole" for case in selected)
    undulator_lines = {}
    if needs_sb or needs_mult:
        built = _build_undulator_lines(
            offset_x=args.offset_x, corrector_strategy=args.corrector_strategy)
        if needs_sb:
            undulator_lines["splineboris"] = built["splineboris"]
        if needs_mult:
            undulator_lines["multipole"] = built["multipole"]

    results = []
    for case in selected:
        results.append(_run_case(
            case, undulator_lines,
            n_turns=args.n_turns, n_part=args.n_part,
            fit_start_turn=args.fit_start_turn, offset_x=args.offset_x,
            with_progress=1,
        ))

    _print_summary(results)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
