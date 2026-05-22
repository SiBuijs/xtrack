"""
SLS multi-turn spin polarization: bare ring vs SplineBoris vs MultipoleKick undulator.

Loads sls.madx with RF cavities (005a), optionally inserts a fitted undulator at
ars11_uind_0210_1, runs 6D twiss with spin/polarization, then tracks particles
with Gaussian offsets around the closed orbit and spin along y.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from _sls_ring_common import (
    WIGGLER_PLACE,
    build_multipole_undulator,
    build_splineboris_undulator,
    configure_tracking_spin,
    default_particle_ref,
    insert_undulator,
    load_field_fit,
    make_offset_bunch,
    new_ring,
    twiss_spin,
    twiss_summary_dict,
)

CASE_LABELS = {
    None: "bare",
    "splineboris": "SplineBoris",
    "multipole": "MultipoleKick",
}


def bunch_spin_means(mon):
    """Bunch-averaged spin components per turn (alive particles only)."""
    mask_alive = mon.state > 0
    n_alive = mask_alive.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sx = mon.spin_x.sum(axis=0) / n_alive
        sy = mon.spin_y.sum(axis=0) / n_alive
        sz = mon.spin_z.sum(axis=0) / n_alive
    pol = np.sqrt(sx**2 + sy**2 + sz**2)
    return sx, sy, sz, pol


def run_case(
    name,
    model,
    field_fitter,
    num_turns,
    radiation_model,
    with_progress,
    num_particles,
    bunch_seed,
    position_sigmas,
):
    p0 = default_particle_ref()
    env, ring = new_ring(p0)

    if model == "splineboris":
        und, _ = build_splineboris_undulator(env, p0, field_fitter)
        insert_undulator(ring, und, place=WIGGLER_PLACE)
    elif model == "multipole":
        und, _ = build_multipole_undulator(env, p0, field_fitter)
        insert_undulator(ring, und, place=WIGGLER_PLACE)
    elif model is not None:
        raise ValueError(f"Unknown model: {model!r}")

    configure_tracking_spin(ring)
    tw = twiss_spin(ring, radiation_model=radiation_model)

    particles = make_offset_bunch(
        ring,
        tw,
        num_particles=num_particles,
        seed=bunch_seed,
        **position_sigmas,
    )

    ring.discard_tracker()
    ring.build_tracker()
    ring.track(
        particles,
        num_turns=num_turns,
        turn_by_turn_monitor=True,
        with_progress=with_progress,
    )
    mon = ring.record_last_track

    summary = twiss_summary_dict(tw)
    summary["name"] = name
    summary["wiggler_place"] = WIGGLER_PLACE if model else None

    spin_norm = np.sqrt(mon.spin_x**2 + mon.spin_y**2 + mon.spin_z**2)
    summary["num_particles"] = num_particles
    summary["spin_norm_mean"] = float(np.mean(spin_norm[mon.state > 0]))
    summary["spin_norm_std"] = float(np.std(spin_norm[mon.state > 0]))
    _, _, _, pol = bunch_spin_means(mon)
    summary["pol_mean_final"] = float(pol[-1]) if len(pol) else np.nan

    return {"monitor": mon, "twiss": tw, "summary": summary}


def plot_spin_vs_turn(cases, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    components = ("spin_x", "spin_y", "spin_z")
    ylabels = (r"$s_x$", r"$s_y$", r"$s_z$")

    for ax, comp, ylab, idx in zip(axes, components, ylabels, range(3)):
        for case in cases:
            mon = case["monitor"]
            sx, sy, sz, _ = bunch_spin_means(mon)
            vals = (sx, sy, sz)[idx]
            n_turns = len(vals)
            turns = np.arange(n_turns)
            ax.plot(turns, vals, label=case["summary"]["name"], linewidth=1.2)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("Turn")
    n_part = cases[0]["summary"]["num_particles"] if cases else 0
    axes[0].set_title(
        f"Bunch-averaged spin vs turn ({n_part} particles, "
        r"Gaussian $\Delta$pos, initial $s_y=1$)"
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig


def print_summaries(cases):
    print("=" * 80)
    print("SLS MULTI-TURN SPIN POLARIZATION")
    print("=" * 80)
    for case in cases:
        s = case["summary"]
        print(f"\n--- {s['name']} ---")
        if s.get("wiggler_place"):
            print(f"  Undulator at: {s['wiggler_place']}")
        print(f"  qx = {s['qx']:.6f},  qy = {s['qy']:.6f},  qs = {s['qs']:.6f}")
        print(f"  spin_polarization_eq = {s['spin_polarization_eq']:.6e}")
        if "spin_t_pol_buildup_turns" in s:
            print(
                f"  spin_t_pol_buildup = {s['spin_t_pol_buildup_s']:.4e} s  "
                f"({s['spin_t_pol_buildup_turns']:.2f} turns)"
            )
        print(f"  num_particles = {s['num_particles']}")
        print(
            f"  |S| along track: mean = {s['spin_norm_mean']:.8f}, "
            f"std = {s['spin_norm_std']:.2e}"
        )
        print(f"  bunch polarization (final turn) = {s['pol_mean_final']:.6f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-turns",
        type=int,
        default=1000,
        help="Number of turns to track (default: 1000)",
    )
    parser.add_argument(
        "--radiation-model",
        choices=("mean", "quantum"),
        default="mean",
        help="Radiation model for twiss and tracking (default: mean)",
    )
    parser.add_argument(
        "--with-progress",
        type=int,
        default=0,
        metavar="N",
        help="Print tracking progress every N turns (0 = off)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the spin-vs-turn figure",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show()",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=300,
        help="Number of particles in the matched bunch (default: 300)",
    )
    parser.add_argument(
        "--bunch-seed",
        type=int,
        default=0,
        help="Random seed for initial position draws (default: 0)",
    )
    for coord in ("x", "px", "y", "py", "zeta", "delta"):
        parser.add_argument(
            f"--sigma-{coord.replace('_', '-')}",
            type=float,
            default=1e-4,
            dest=f"sigma_{coord}",
            metavar="SIGMA",
            help=f"Gaussian sigma for {coord} offset around CO [m or rad] (default: 1e-4)",
        )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("bare", "splineboris", "multipole", "all"),
        default=["all"],
        help="Which cases to run (default: all)",
    )
    args = parser.parse_args()

    if "all" in args.cases:
        models = [None, "splineboris", "multipole"]
    else:
        model_map = {"bare": None, "splineboris": "splineboris", "multipole": "multipole"}
        models = [model_map[c] for c in args.cases]

    field_fitter = load_field_fit() if any(m is not None for m in models) else None
    progress = args.with_progress if args.with_progress > 0 else False
    position_sigmas = {
        "sigma_x": args.sigma_x,
        "sigma_px": args.sigma_px,
        "sigma_y": args.sigma_y,
        "sigma_py": args.sigma_py,
        "sigma_zeta": args.sigma_zeta,
        "sigma_delta": args.sigma_delta,
    }

    results = []
    for model in models:
        label = CASE_LABELS[model]
        print(f"\nRunning case: {label} ...")
        results.append(
            run_case(
                name=label,
                model=model,
                field_fitter=field_fitter,
                num_turns=args.num_turns,
                radiation_model=args.radiation_model,
                with_progress=progress,
                num_particles=args.num_particles,
                bunch_seed=args.bunch_seed,
                position_sigmas=position_sigmas,
            )
        )

    print_summaries(results)
    plot_spin_vs_turn(results, save_path=args.save)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
