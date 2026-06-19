from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt
import xobjects as xo

plt.close("all")

HERE = Path(__file__).resolve().parent
# Produced by 004b_install_solenoids_in_fcc_ring.py and
# 004c_correct_solenoids_in_fcc_ring.py.
INPUT_LATTICE_JSON = (
    HERE / "fccee_z_lcc_splineboris_solenoids_coupling_corrected.json"
)

IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]
N_TURNS = 10_000
NEMITT_X = 6.33e-5
NEMITT_Y = 1.69e-7  # flat beam: y_norm is scaled in _build_da_initial_conditions
ENERGY_SPREAD = 3.9e-4
# Use the xtrack default (1 m), not the 5 cm limit from 009_momentum_acceptance.py:
# large beta functions amplify amplitudes well beyond 5 cm for sigma-levels ~ O(30).
GLOBAL_XY_LIMIT = 1.0

# Asymmetric DA grid (flat beam): amplitudes on both axes are in horizontal-sigma
# units so x and y probe comparable physical excursions. Reduce NN_Y_R / NN_X_THETA
# if runtime is too long.
NN_Y_R = 25
MAX_AMP_SIGMA_X = 40.0
NN_X_THETA = 30.0


def _compute_beam_sizes(line):
    tw = line.twiss()
    beta_x = tw["betx"][0]
    beta_y = tw["bety"][0]
    beta0, gamma0 = float(tw["beta0"]), float(tw["gamma0"])
    geom_emitt_x = NEMITT_X / (beta0 * gamma0)
    geom_emitt_y = NEMITT_Y / (beta0 * gamma0)
    sigma_x = np.sqrt(beta_x * geom_emitt_x)
    sigma_y = np.sqrt(beta_y * geom_emitt_y)
    return sigma_x, sigma_y


def _build_da_initial_conditions(sigma_x_over_sigma_y):
    # Polar grid with radius in horizontal-sigma units; on-momentum DA at delta=0.
    # Uses xpart directly (fcc_92 gen_grid has a column-length bug for single delta).
    x_hat, y_hat, _, _ = xp.generate_2D_polar_grid(
        r_range=(0, MAX_AMP_SIGMA_X),
        theta_range=(0, np.pi / 2),
        nr=NN_Y_R,
        ntheta=NN_X_THETA,
    )
    # build_particles expects true vertical sigma units for y_norm.
    y_norm = y_hat * 5.0#sigma_x_over_sigma_y
    return xt.Table(
        dict(
            id=np.arange(len(x_hat), dtype=int),
            x_hat=x_hat,
            y_hat=y_hat,
            x_normalized=x_hat,
            y_normalized=y_norm,
        ),
        index="id",
    )


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


def _plot_dynamic_aperture_figure(out, tt_init, title, *, sigma_x, sigma_y):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    particles = out["particles"]
    x_hat = tt_init.x_hat
    y_hat = tt_init.y_hat
    at_turn = particles.at_turn

    sc = axes[0].scatter(
        x_hat,
        y_hat,
        c=at_turn,
        s=8,
        marker="o",
    )
    axes[0].set_xlabel(r"$\hat{x}\,[\sigma_x]$")
    axes[0].set_ylabel(r"$\hat{y}\,[\sigma_x]$")
    axes[0].set_title("scatter")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes[0], label="lost at turn")

    pcm = axes[1].pcolormesh(
        x_hat.reshape(NN_Y_R, NN_X_THETA),
        y_hat.reshape(NN_Y_R, NN_X_THETA),
        at_turn.reshape(NN_Y_R, NN_X_THETA),
        shading="gouraud",
    )
    axes[1].set_xlabel(r"$\hat{x}\,[\sigma_x]$")
    axes[1].set_ylabel(r"$\hat{y}\,[\sigma_x]$")
    axes[1].set_title("pcolormesh")
    axes[1].set_aspect("equal", adjustable="box")
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
    *,
    lattice_json,
    with_solenoids,
    with_correctors,
    title,
    with_progress,
):
    print(f"\n=== {title} ===")
    print(f"Loading lattice: {lattice_json.name}")
    env = xt.load(lattice_json)
    line = env.fccee_p_ring
    line.cycle("ipa")
    _set_solenoid_knobs(
        line,
        with_solenoids=with_solenoids,
        with_correctors=with_correctors,
    )

    line.discard_tracker()
    line.build_tracker()

    _configure_radiative_tracking(line)

    line.discard_tracker()
    line.build_tracker()
    sigma_x, sigma_y = _compute_beam_sizes(line)
    sigma_x_over_sigma_y = sigma_x / sigma_y
    print(
        f"Beam sizes at start: sigma_x={sigma_x*1e6:.3f} um, "
        f"sigma_y={sigma_y*1e6:.3f} um "
        f"(sigma_x/sigma_y={sigma_x_over_sigma_y:.1f})"
    )

    # On-momentum DA: scan (x_hat, y_hat) in horizontal-sigma units at delta=0.
    # For off-momentum DA, repeat with delta set to selected momentum offsets.
    tt_init = _build_da_initial_conditions(sigma_x_over_sigma_y)
    num_particles = len(tt_init)
    print(f"Tracking {num_particles} particles for {N_TURNS} turns")

    particles = line.build_particles(
        method="6d",
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        delta=0,
        x_norm=tt_init.x_normalized,
        y_norm=tt_init.y_normalized,
        px_norm=0,
        py_norm=0,
    )

    line.discard_tracker()
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads="auto"))

    line.config.XTRACK_GLOBAL_XY_LIMIT = GLOBAL_XY_LIMIT
    line.track(particles, num_turns=N_TURNS, with_progress=with_progress)
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
    _plot_dynamic_aperture_figure(
        out, tt_init, title, sigma_x=sigma_x, sigma_y=sigma_y
    )
    print(
        f"[{title}] Dynamic aperture run complete:"
        f" frac_lost={frac_lost:.6g},"
        f" at_turn_mean={at_turn_mean:.6g}"
    )
    return out


_run_dynamic_aperture(
    lattice_json=INPUT_LATTICE_JSON,
    with_solenoids=True,
    with_correctors=True,
    title="SplineBoris: solenoids powered + correction scheme",
    with_progress=1,
)
_run_dynamic_aperture(
    lattice_json=INPUT_LATTICE_JSON,
    with_solenoids=False,
    with_correctors=False,
    title="SplineBoris: solenoids unpowered",
    with_progress=1,
)
plt.show()
