from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

plt.close("all")

HERE = Path(__file__).resolve().parent
XSUITE_ROOT = HERE.parent.parent.parent
INPUT_LATTICE_SOL_JSON = (
    HERE / "fccee_z_lcc_splineboris_solenoids_coupling_corrected.json"
)
INPUT_LATTICE_NO_SOL_JSON = HERE / "fccee_z_lcc.json"

FCC92_TUTORIAL_DIR = XSUITE_ROOT / "fcc_92" / "004_tutorial_cap_meeting"
if str(FCC92_TUTORIAL_DIR) not in sys.path:
    sys.path.append(str(FCC92_TUTORIAL_DIR))

try:
    from momentum_acceptance import ActionMomentumAcceptance
except ImportError as ee:
    raise ImportError(
        "Could not import ActionMomentumAcceptance from "
        f"{FCC92_TUTORIAL_DIR}. Please make sure fcc_92 is available "
        "next to xtrack in the xsuite directory."
    ) from ee


IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]
N_TURNS = 1000
NEMITT_X = 6.33e-5
NEMITT_Y = 1.69e-7
ENERGY_SPREAD = 3.9e-4
NN_Y_R = 15
MAX_Y_R = 15
GLOBAL_XY_LIMIT = 5e-2
DELTA_INITIAL_VALUES = np.linspace(-35 * ENERGY_SPREAD, 35 * ENERGY_SPREAD, 51)


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

def _plot_momentum_acceptance_figure(out, tt_init, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    lost = out["lost"]
    particles = out["particles"]
    ax.plot(
        tt_init.delta_init[~lost],
        tt_init.x_normalized[~lost],
        ".",
        ms=3,
        label="survived",
    )
    sc = ax.scatter(
        tt_init.delta_init[lost],
        tt_init.x_normalized[lost],
        c=particles.at_turn[lost],
        marker="o",
        s=18,
        label="lost",
    )
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\hat{x}$")
    ax.set_title(
        f"{title}\n"
        f"frac_lost={out['frac_lost']:.4g}, "
        f"at_turn_mean={out['at_turn_mean']:.4g}"
    )
    fig.colorbar(sc, ax=ax, label="lost at turn")
    return fig


def _run_momentum_acceptance(
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

    # Keep same spirit as fcc_92/001_characterize_lattice/001_momentum_acceptance.py
    # but on FCC-ee lattice variants.
    line.twiss()

    act = ActionMomentumAcceptance(
        line,
        NEMITT_X,
        NEMITT_Y,
        NN_Y_R,
        MAX_Y_R,
        ENERGY_SPREAD,
        num_turns=N_TURNS,
        delta_initial_values=DELTA_INITIAL_VALUES,
        global_xy_limit=GLOBAL_XY_LIMIT,
    )

    out = act.mom_acceptance(plot=False, with_progress=with_progress)
    _plot_momentum_acceptance_figure(out, act.tt_init, title)
    print(
        f"[{title}] Momentum acceptance run complete:"
        f" frac_lost={out['frac_lost']:.6g},"
        f" at_turn_mean={out['at_turn_mean']:.6g}"
    )
    return out


_run_momentum_acceptance(
    lattice_json=INPUT_LATTICE_SOL_JSON,
    with_solenoids=True,
    with_correctors=True,
    title="Ring with solenoids + correction scheme",
    with_progress=1,
)
_run_momentum_acceptance(
    lattice_json=INPUT_LATTICE_NO_SOL_JSON,
    with_solenoids=False,
    with_correctors=False,
    title="Ring without solenoids (no solenoid correctors)",
    with_progress=1,
)
plt.show()
