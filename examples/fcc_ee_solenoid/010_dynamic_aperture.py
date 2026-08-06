from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt

from aperture_study_io import save_da_study, variant_suffix
from lattice_knobs import (
    install_extra_sextupole,
    robust_twiss,
    set_lattice_knobs,
    set_solenoid_offset,
)
from solenoid_params import (
    MAIN_SOLENOID_B0,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    order_tag,
)

plt.close("all")

HERE = Path(__file__).resolve().parent


def _lattice_paths(tag, order_tag_str=""):
    # order_tag_str only applies to the SplineBoris path -- VariableSolenoid
    # is linear-only and has no transverse-order knob (see 00_overview.md).
    return (
        HERE / f"fccee_z_lcc_splineboris_solenoids_coupling_corrected_{tag}{order_tag_str}.json",
        HERE / f"fccee_z_lcc_varsol_solenoids_coupling_corrected_{tag}.json",
    )

NEMITT_X = 6.33e-5
NEMITT_Y = 1.69e-7  # flat beam: sigma_x/sigma_y ~ 220 at the IP
# Use the xtrack default (1 m), not the 5 cm limit from 009_momentum_acceptance.py:
# large beta functions amplify amplitudes well beyond 5 cm for sigma-levels ~ O(30).
GLOBAL_XY_LIMIT = 1.0
# Elliptical scan: y's amplitude cap is a self-chosen multiple of x's, NOT the
# true sigma_x/sigma_y (~219) physical ratio -- using the real ratio stretches
# the y-extent out to ~MAX_AMP_SIGMA_X*sigma_x/sigma_y ~ 8800 sigma_y, so far
# beyond where the DA boundary actually sits that the plotted shape collapses
# to an unreadable sliver near y=0. This factor is picked purely to keep the
# plot readable; the axis still reports the true (scaled) sigma_y value.
MAX_AMP_SIGMA_X = 40.0
Y_AXIS_SCAN_FACTOR = 6.0
MAX_AMP_SIGMA_Y = MAX_AMP_SIGMA_X * Y_AXIS_SCAN_FACTOR
N_TURNS = 10_000

# Each case can use its own grid resolution.
def _build_da_cases(tag, order_tag_str=""):
    """Build the DA_CASES list for a given field_tag (e.g. '2T', '3T') and
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
            nn_y_r=25,
            nn_x_theta=30,
        ),
        dict(
            name="varsol_on",
            model="VarSol",
            lattice_json=varsol_json,
            with_solenoids=True,
            with_correctors=True,
            title=f"VariableSolenoid ({tag}): solenoids powered + correction scheme",
            nn_y_r=30,
            nn_x_theta=50,
        ),
        dict(
            name="sb_off",
            model="SB",
            lattice_json=splineboris_json,
            with_solenoids=False,
            with_correctors=False,
            title=f"SplineBoris ({sb_tag}): solenoids unpowered",
            nn_y_r=25,
            nn_x_theta=30,
        ),
    ]


# sb_off is the bare-machine baseline: it has already been run once and does
# not depend on solenoid/correction changes, so skip it by default (pass
# --cases sb_off explicitly to rerun it).
DEFAULT_DA_CASE_NAMES = ["sb_on", "varsol_on"]


def _compute_beam_sizes(line):
    # robust_twiss falls back to a co-guess continuation (ramping sext_amp
    # from 1.0) if the direct closed-orbit search fails, e.g. for large
    # --sexamp values -- see lattice_knobs.robust_twiss.
    tw = robust_twiss(line)
    beta_x = tw["betx"][0]
    beta_y = tw["bety"][0]
    beta0, gamma0 = float(tw["beta0"]), float(tw["gamma0"])
    geom_emitt_x = NEMITT_X / (beta0 * gamma0)
    geom_emitt_y = NEMITT_Y / (beta0 * gamma0)
    sigma_x = np.sqrt(beta_x * geom_emitt_x)
    sigma_y = np.sqrt(beta_y * geom_emitt_y)
    return sigma_x, sigma_y, tw


def _build_da_initial_conditions(*, nn_y_r, nn_x_theta):
    # Unit polar grid (r in [0, 1]); on-momentum DA at delta=0.
    # Uses xpart directly (fcc_92 gen_grid has a column-length bug for single delta).
    x_unit, y_unit, _, _ = xp.generate_2D_polar_grid(
        r_range=(0, 1.0),
        theta_range=(0, np.pi / 2),
        nr=nn_y_r,
        ntheta=nn_x_theta,
    )
    # Scale each plane independently by its own sigma-amplitude cap, so the
    # scan is a genuine ellipse in normalized (sigma_x, sigma_y) units rather
    # than a physical-radius circle. See MAX_AMP_SIGMA_X/Y comment above.
    x_hat = x_unit * MAX_AMP_SIGMA_X
    y_hat = y_unit * MAX_AMP_SIGMA_Y
    return xt.Table(
        dict(
            id=np.arange(len(x_hat), dtype=int),
            x_hat=x_hat,
            y_hat=y_hat,
            x_normalized=x_hat,
            y_normalized=y_hat,
        ),
        index="id",
    )


def _configure_radiative_tracking(line):
    line.particle_ref.anomalous_magnetic_moment = 0.00115965218128
    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss()


def _plot_dynamic_aperture_figure(
    out, tt_init, title, *, sigma_x, sigma_y, nn_y_r, nn_x_theta
):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    particles = out["particles"]
    x_hat = tt_init.x_normalized
    y_hat = tt_init.y_normalized
    at_turn = particles.at_turn

    sc = axes[0].scatter(
        x_hat,
        y_hat,
        c=at_turn,
        s=8,
        marker="o",
    )
    axes[0].set_xlabel(r"$\hat{x}\,[\sigma_x]$")
    axes[0].set_ylabel(r"$\hat{y}\,[\sigma_y]$")
    axes[0].set_title("scatter")
    fig.colorbar(sc, ax=axes[0], label="lost at turn")

    pcm = axes[1].pcolormesh(
        x_hat.reshape(nn_y_r, nn_x_theta),
        y_hat.reshape(nn_y_r, nn_x_theta),
        at_turn.reshape(nn_y_r, nn_x_theta),
        shading="gouraud",
    )
    axes[1].set_xlabel(r"$\hat{x}\,[\sigma_x]$")
    axes[1].set_ylabel(r"$\hat{y}\,[\sigma_y]$")
    axes[1].set_title("pcolormesh")
    fig.colorbar(pcm, ax=axes[1], label="lost at turn")

    fig.suptitle(
        f"{title}\n"
        f"frac_lost={out['frac_lost']:.4g}, "
        f"at_turn_mean={out['at_turn_mean']:.4g}\n"
        f"$\\sigma_x={sigma_x*1e6:.2f}\\,\\mu$m, "
        f"$\\sigma_y={sigma_y*1e6:.2f}\\,\\mu$m "
        f"($\\sigma_x/\\sigma_y={sigma_x/sigma_y:.0f}$)"
    )
    fig.tight_layout()
    return fig


def _run_dynamic_aperture(
    case, *, n_turns, with_progress, sexamp, x_offset, y_offset,
    extra_sext_strength=0.0, tag,
):
    lattice_json = case["lattice_json"]
    title = case["title"]
    nn_y_r = case["nn_y_r"]
    nn_x_theta = case["nn_x_theta"]

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
    set_solenoid_offset(line, x_offset=x_offset, y_offset=y_offset)
    install_extra_sextupole(line, k2l=extra_sext_strength)

    line.discard_tracker()
    line.build_tracker()

    _configure_radiative_tracking(line)

    line.discard_tracker()
    line.build_tracker()
    sigma_x, sigma_y, tw_co = _compute_beam_sizes(line)
    print(
        f"Beam sizes at start: sigma_x={sigma_x*1e6:.3f} um, "
        f"sigma_y={sigma_y*1e6:.3f} um "
        f"(sigma_x/sigma_y={sigma_x/sigma_y:.1f})"
    )

    # On-momentum DA: scan (x_hat, y_hat), each in its own plane's sigma units,
    # at delta=0. For off-momentum DA, repeat with delta set to selected
    # momentum offsets.
    tt_init = _build_da_initial_conditions(
        nn_y_r=nn_y_r,
        nn_x_theta=nn_x_theta,
    )
    num_particles = len(tt_init)
    print(f"Tracking {num_particles} particles for {n_turns} turns")

    particles = line.build_particles(
        method="6d",
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        delta=0,
        x_norm=tt_init.x_normalized,
        y_norm=tt_init.y_normalized,
        px_norm=0,
        py_norm=0,
        particle_on_co=tw_co.particle_on_co,
    )

    line.discard_tracker()
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads="auto"))

    line.config.XTRACK_GLOBAL_XY_LIMIT = GLOBAL_XY_LIMIT
    line.track(particles, num_turns=n_turns, with_progress=with_progress)
    particles.sort(interleave_lost_particles=True)

    lost = particles.state <= 0
    frac_lost = lost.sum() / len(lost)
    at_turn_mean = particles.at_turn.mean()

    out = {
        "particles": particles,
        "lost": lost,
        "frac_lost": frac_lost,
        "at_turn_mean": at_turn_mean,
        "tt_init": tt_init,
    }
    fig = _plot_dynamic_aperture_figure(
        out,
        tt_init,
        title,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        nn_y_r=nn_y_r,
        nn_x_theta=nn_x_theta,
    )
    save_da_study(
        out=out,
        tt_init=tt_init,
        fig=fig,
        model=case["model"],
        with_solenoids=case["with_solenoids"],
        with_correctors=case["with_correctors"],
        n_turns=n_turns,
        global_xy_limit=GLOBAL_XY_LIMIT,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        nn_y_r=nn_y_r,
        nn_x_theta=nn_x_theta,
        max_amp_sigma_x=MAX_AMP_SIGMA_X,
        max_amp_sigma_y=MAX_AMP_SIGMA_Y,
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        n_part=num_particles,
        radiation="mean",
        variant=variant_suffix(
            sexamp=sexamp, x_offset=x_offset, y_offset=y_offset,
            extra_sext_strength=extra_sext_strength,
        ),
        sexamp=sexamp,
        x_offset=x_offset,
        y_offset=y_offset,
        extra_sext_strength=extra_sext_strength,
        field_tag=tag,
    )
    print(
        f"[{title}] Dynamic aperture run complete:"
        f" frac_lost={frac_lost:.6g},"
        f" at_turn_mean={at_turn_mean:.6g}"
    )
    return out


def _select_cases(case_names, cases_by_name):
    if not case_names:
        return [cases_by_name[name] for name in DEFAULT_DA_CASE_NAMES]

    unknown = [name for name in case_names if name not in cases_by_name]
    if unknown:
        valid = ", ".join(cases_by_name)
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Choose from: {valid}")

    return [cases_by_name[name] for name in case_names]


def main():
    parser = argparse.ArgumentParser(
        description="Run dynamic-aperture studies for FCC-ee solenoid lattices."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help=(
            "Cases to run (default: sb_on, varsol_on -- sb_off is skipped by "
            "default since the bare-machine baseline doesn't change; pass "
            "--cases sb_off explicitly to include it). "
            "Available: sb_on, varsol_on, sb_off"
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
        "--sexamp",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Sextupole amplification knob (default: 1.0).",
    )
    parser.add_argument(
        "--x-offset",
        type=float,
        default=0.0,
        metavar="M",
        help="Main detector solenoid x-offset in meters (default: 0.0).",
    )
    parser.add_argument(
        "--y-offset",
        type=float,
        default=0.0,
        metavar="M",
        help="Main detector solenoid y-offset in meters (default: 0.0).",
    )
    parser.add_argument(
        "--extra-sext-strength",
        type=float,
        default=0.0,
        metavar="K2L",
        help=(
            "Integrated strength (k2*L, in m^-2) of an extra thin sextupole "
            "inserted into each IP's main detector solenoid, between two "
            "SplineBoris slices at s ~= -1.223 m from the IP (default: 0.0, "
            "i.e. off -- the lattice is unmodified unless this is nonzero)."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive figure display (data and PDFs are still saved).",
    )
    args = parser.parse_args()

    tag = field_tag(args.b0)
    order_tag_str = order_tag(args.max_transverse_order)
    da_cases = _build_da_cases(tag, order_tag_str)
    da_cases_by_name = {case["name"]: case for case in da_cases}

    if args.list_cases:
        print(
            f"Field tag: {tag}{order_tag_str} (--b0 {args.b0:g}, "
            f"--max-transverse-order {args.max_transverse_order})")
        for case in da_cases:
            print(f"{case['name']}: {case['title']}")
        return

    for case in _select_cases(args.cases, da_cases_by_name):
        _run_dynamic_aperture(
            case,
            n_turns=args.n_turns,
            with_progress=1,
            sexamp=args.sexamp,
            x_offset=args.x_offset,
            y_offset=args.y_offset,
            extra_sext_strength=args.extra_sext_strength,
            tag=f"{tag}{order_tag_str}",
        )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
