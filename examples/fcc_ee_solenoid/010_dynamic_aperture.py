from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import xtrack as xt
import xobjects as xo

plt.close("all")

HERE = Path(__file__).resolve().parent
INPUT_LATTICE_SOL_JSON = (
    HERE / "fccee_z_lcc_splineboris_solenoids_coupling_corrected.json"
)
INPUT_LATTICE_NO_SOL_JSON = HERE / "fccee_z_lcc.json"

IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]
N_TURNS = 100
NEMITT_X = 6.33e-5
NEMITT_Y = 1.69e-7
ENERGY_SPREAD = 3.9e-4
GLOBAL_XY_LIMIT = 5e-2

# Polar DA grid: reduce NN_Y_R / NN_X_THETA if runtime is too long.
NN_Y_R = 30
MAX_Y_R = 40.0
NN_X_THETA = 50


def _build_da_initial_conditions():
    # Polar grid in normalized coordinates; on-momentum DA at delta=0.
    # Uses xpart directly (fcc_92 gen_grid has a column-length bug for single delta).
    x_normalized, y_normalized, _, _ = xp.generate_2D_polar_grid(
        r_range=(0, MAX_Y_R),
        theta_range=(0, np.pi / 2),
        nr=NN_Y_R,
        ntheta=NN_X_THETA,
    )
    return xt.Table(
        dict(
            id=np.arange(len(x_normalized), dtype=int),
            x_normalized=x_normalized,
            y_normalized=y_normalized,
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


def _plot_dynamic_aperture_figure(out, tt_init, title):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    particles = out["particles"]
    x_norm = tt_init.x_normalized
    y_norm = tt_init.y_normalized
    at_turn = particles.at_turn

    sc = axes[0].scatter(
        x_norm,
        y_norm,
        c=at_turn,
        s=8,
        marker="o",
    )
    axes[0].set_xlabel(r"$\hat{x}\,[\sigma]$")
    axes[0].set_ylabel(r"$\hat{y}\,[\sigma]$")
    axes[0].set_title("scatter")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes[0], label="lost at turn")

    pcm = axes[1].pcolormesh(
        x_norm.reshape(NN_Y_R, NN_X_THETA),
        y_norm.reshape(NN_Y_R, NN_X_THETA),
        at_turn.reshape(NN_Y_R, NN_X_THETA),
        shading="gouraud",
    )
    axes[1].set_xlabel(r"$\hat{x}\,[\sigma]$")
    axes[1].set_ylabel(r"$\hat{y}\,[\sigma]$")
    axes[1].set_title("pcolormesh")
    axes[1].set_aspect("equal", adjustable="box")
    fig.colorbar(pcm, ax=axes[1], label="lost at turn")

    fig.suptitle(
        f"{title}\n"
        f"frac_lost={out['frac_lost']:.4g}, "
        f"at_turn_mean={out['at_turn_mean']:.4g}"
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

    line.twiss()

    # On-momentum DA: scan (x_hat, y_hat) at fixed delta=0.
    # For off-momentum DA, repeat with delta set to selected momentum offsets.
    tt_init = _build_da_initial_conditions()
    num_particles = len(tt_init)
    print(f"Tracking {num_particles} particles for {N_TURNS} turns")

    particles = line.build_particles(
        method="4d",
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        delta=0,
        x_norm=tt_init.x_normalized,
        y_norm=tt_init.y_normalized,
        px_norm=0,
        py_norm=0,
    )

    # Optional: activate multi-core CPU parallelization
    line.discard_tracker()
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

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
    _plot_dynamic_aperture_figure(out, tt_init, title)
    print(
        f"[{title}] Dynamic aperture run complete:"
        f" frac_lost={frac_lost:.6g},"
        f" at_turn_mean={at_turn_mean:.6g}"
    )
    return out


if __name__ == "__main__":
    _run_dynamic_aperture(
        lattice_json=INPUT_LATTICE_SOL_JSON,
        with_solenoids=True,
        with_correctors=True,
        title="Ring with solenoids + correction scheme",
        with_progress=1,
    )
    _run_dynamic_aperture(
        lattice_json=INPUT_LATTICE_NO_SOL_JSON,
        with_solenoids=False,
        with_correctors=False,
        title="Ring without solenoids (no solenoid correctors)",
        with_progress=1,
    )
    plt.show()
