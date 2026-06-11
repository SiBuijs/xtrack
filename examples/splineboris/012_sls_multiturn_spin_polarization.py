"""
SLS multi-turn spin polarization: bare ring vs SplineBoris vs MultipoleKick undulator.

Loads sls.madx with RF cavities (005a), optionally inserts a fitted undulator at
ars11_uind_0610_1 (centered or horizontally offset by 1 mm), initializes around
the 6D closed orbit, then tracks particles with Gaussian offsets and configurable
initial spin polarization along y to study turn-by-turn polarization.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _sls_ring_common import (
    DEFAULT_UNDULATOR_SHIFT_X,
    WIGGLER_PLACES_SPIN,
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

# case_key -> (model, default undulator_shift_x [m])
CASE_SPECS = {
    "bare": (None, 0.0),
    "splineboris": ("splineboris", 0.0),
    "multipole": ("multipole", 0.0),
    "splineboris-offset": ("splineboris", -DEFAULT_UNDULATOR_SHIFT_X),
    "multipole-offset": ("multipole", -DEFAULT_UNDULATOR_SHIFT_X),
}
CASE_BASE_LABELS = {
    "bare": "bare",
    "splineboris": "SplineBoris",
    "multipole": "MultipoleKick",
}
CACHE_DIR = Path(__file__).resolve().parent / "_spin_cache"


def _cache_key(
    case_key,
    label,
    num_turns,
    radiation_model,
    num_particles,
    bunch_seed,
    position_sigmas,
    undulator_shift_x,
    num_wiggler_sites,
    spin_pol_y,
    spin_seed,
):
    sigma_part = "_".join(
        f"{k}-{position_sigmas[k]:.3e}"
        for k in ("sigma_x", "sigma_px", "sigma_y", "sigma_py", "sigma_zeta", "sigma_delta")
    )
    safe_label = (
        label.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "p")
        .replace("-", "m")
    )
    return (
        f"{case_key}_{safe_label}"
        f"_rad-{radiation_model}"
        f"_dx-{undulator_shift_x:.3e}"
        f"_nwig-{num_wiggler_sites}"
        f"_turns-{num_turns}"
        f"_np-{num_particles}"
        f"_seed-{bunch_seed}"
        f"_spy-{spin_pol_y:.4f}"
        f"_sspin-{spin_seed}"
        f"_{sigma_part}"
    )


def _legacy_cache_key(
    case_key,
    label,
    num_turns,
    radiation_model,
    num_particles,
    bunch_seed,
    position_sigmas,
    undulator_shift_x,
):
    sigma_part = "_".join(
        f"{k}-{position_sigmas[k]:.3e}"
        for k in ("sigma_x", "sigma_px", "sigma_y", "sigma_py", "sigma_zeta", "sigma_delta")
    )
    safe_label = (
        label.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "p")
        .replace("-", "m")
    )
    return (
        f"{case_key}_{safe_label}"
        f"_rad-{radiation_model}"
        f"_dx-{undulator_shift_x:.3e}"
        f"_turns-{num_turns}"
        f"_np-{num_particles}"
        f"_seed-{bunch_seed}"
        f"_{sigma_part}"
    )


def _save_case_cache(cache_path, monitor, summary):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        spin_x=np.asarray(monitor.spin_x),
        spin_y=np.asarray(monitor.spin_y),
        spin_z=np.asarray(monitor.spin_z),
        state=np.asarray(monitor.state),
        summary=np.array([summary], dtype=object),
    )


def _load_case_cache(cache_path):
    data = np.load(cache_path, allow_pickle=True)
    monitor = {
        "spin_x": data["spin_x"],
        "spin_y": data["spin_y"],
        "spin_z": data["spin_z"],
        "state": data["state"],
    }
    summary = data["summary"][0].item() if hasattr(data["summary"][0], "item") else data["summary"][0]
    return {"monitor": monitor, "summary": summary}


def _spin_arrays(monitor):
    if isinstance(monitor, dict):
        return monitor["spin_x"], monitor["spin_y"], monitor["spin_z"], monitor["state"]
    return monitor.spin_x, monitor.spin_y, monitor.spin_z, monitor.state


def bunch_spin_means(mon):
    """Bunch-averaged spin components per turn (alive particles only)."""
    spin_x, spin_y, spin_z, state = _spin_arrays(mon)
    mask_alive = state > 0
    n_alive = mask_alive.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sx = spin_x.sum(axis=0) / n_alive
        sy = spin_y.sum(axis=0) / n_alive
        sz = spin_z.sum(axis=0) / n_alive
    pol = np.sqrt(sx**2 + sy**2 + sz**2)
    return sx, sy, sz, pol


def closed_orbit_for_bunch_init(ring):
    """Get closed-orbit particle for bunch initialization only."""
    ring.build_tracker()
    return ring.twiss(method="6d")


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
    undulator_shift_x=0.0,
    spin_pol_y=1.0,
    spin_seed=None,
):
    p0 = default_particle_ref()
    env, ring = new_ring(p0)

    if model == "splineboris":
        und, _ = build_splineboris_undulator(
            env, p0, field_fitter, shift_x=undulator_shift_x
        )
        for place in WIGGLER_PLACES_SPIN:
            insert_undulator(ring, und, place=place)
    elif model == "multipole":
        for place in WIGGLER_PLACES_SPIN:
            und, _ = build_multipole_undulator(
                env,
                p0,
                field_fitter,
                shift_x=undulator_shift_x,
                name_prefix=f"und_kick_{place}",
            )
            insert_undulator(ring, und, place=place)
    elif model is not None:
        raise ValueError(f"Unknown model: {model!r}")

    configure_tracking_spin(ring)
    # Twiss/CO is not available with quantum radiation, so initialize on CO from mean model.
    ring.configure_radiation(model="mean")
    tw = closed_orbit_for_bunch_init(ring)
    mean_twiss_summary = None
    if radiation_model == "mean":
        tw_spin = twiss_spin(ring, radiation_model="mean")
        mean_twiss_summary = twiss_summary_dict(tw_spin)
        for key in ("spin_tune_fractional", "spin_tune", "q_spin", "nu_spin"):
            if hasattr(tw_spin, key):
                mean_twiss_summary[key] = float(getattr(tw_spin, key))
        mean_twiss_summary["spin_tune_agamma"] = float(
            ring.particle_ref.anomalous_magnetic_moment[0] * ring.particle_ref.gamma0[0]
        )
    ring.configure_radiation(model=radiation_model)

    particles = make_offset_bunch(
        ring,
        tw,
        num_particles=num_particles,
        seed=bunch_seed,
        spin_pol_y=spin_pol_y,
        spin_seed=spin_seed,
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

    summary = {"name": name}
    summary["wiggler_place"] = WIGGLER_PLACES_SPIN if model else None
    summary["undulator_shift_x"] = undulator_shift_x
    summary["co_radiation_model"] = "mean"
    summary["radiation_model"] = radiation_model
    summary["spin_pol_y"] = spin_pol_y
    summary["mean_twiss_spin"] = mean_twiss_summary

    spin_x, spin_y, spin_z, state = _spin_arrays(mon)
    spin_norm = np.sqrt(spin_x**2 + spin_y**2 + spin_z**2)
    summary["num_particles"] = num_particles
    summary["spin_norm_mean"] = float(np.mean(spin_norm[state > 0]))
    summary["spin_norm_std"] = float(np.std(spin_norm[state > 0]))
    _, _, _, pol = bunch_spin_means(mon)
    summary["pol_mean_final"] = float(pol[-1]) if len(pol) else np.nan

    return {"monitor": mon, "summary": summary}


def plot_spin_vs_turn(cases, save_path=None, spin_pol_y=1.0):
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    ylabels = (r"$\langle s_x \rangle$", r"$\langle s_y \rangle$", r"$\langle s_z \rangle$")

    for ax, ylab, idx in zip(axes[:3], ylabels, range(3)):
        for case in cases:
            mon = case["monitor"]
            sx, sy, sz, _ = bunch_spin_means(mon)
            vals = (sx, sy, sz)[idx]
            n_turns = len(vals)
            turns = np.arange(n_turns)
            ax.plot(turns, vals, label=case["summary"]["name"], linewidth=1.2)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)

    ax_pol = axes[3]
    for case in cases:
        mon = case["monitor"]
        _, _, _, pol = bunch_spin_means(mon)
        turns = np.arange(len(pol))
        ax_pol.plot(turns, pol, label=case["summary"]["name"], linewidth=1.4)
    ax_pol.set_ylabel(r"$|\langle \vec{S} \rangle|$")
    ax_pol.set_xlabel("Turn")
    ax_pol.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Turn")
    n_part = cases[0]["summary"]["num_particles"] if cases else 0
    axes[0].set_title(
        f"Bunch-averaged spin vs turn ({n_part} particles, "
        rf"Gaussian $\Delta$pos, initial $s_y={spin_pol_y:.3f}$)"
    )
    # Single legend on top-right of the <s_x> panel
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        leg = axes[0].legend(
            handles,
            labels,
            loc="upper right",
            fontsize=9,
            frameon=True,
        )
        leg.get_frame().set_alpha(0.8)
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
        if s.get("undulator_shift_x"):
            print(f"  undulator_shift_x = {s['undulator_shift_x']:.4e} m")
        print(f"  closed_orbit_radiation_model = {s['co_radiation_model']}")
        print(f"  radiation_model = {s['radiation_model']}")
        print(f"  num_particles = {s['num_particles']}")
        if s.get("spin_pol_y") is not None:
            print(f"  initial spin_pol_y = {s['spin_pol_y']:.4f}")
        print(
            f"  |S| along track: mean = {s['spin_norm_mean']:.8f}, "
            f"std = {s['spin_norm_std']:.2e}"
        )
        print(f"  bunch polarization (final turn) = {s['pol_mean_final']:.6f}")
    print("=" * 80)


def print_mean_spin_table(cases):
    mean_cases = [cc for cc in cases if cc["summary"].get("radiation_model") == "mean"]
    if not mean_cases:
        return

    headers = (
        "case",
        "spin_tune_frac",
        "spin_tune_a*gamma",
        "spin_pol_eq",
        "t_build_s",
        "t_build_turns",
    )
    rows = []
    for case in mean_cases:
        s = case["summary"]
        tws = s.get("mean_twiss_spin") or {}
        spin_tune_frac = tws.get(
            "spin_tune_fractional",
            tws.get("spin_tune", tws.get("q_spin", tws.get("nu_spin"))),
        )
        spin_tune_agamma = tws.get("spin_tune_agamma")
        rows.append(
            (
                s["name"],
                "n/a" if spin_tune_frac is None else f"{spin_tune_frac:.8f}",
                "n/a" if spin_tune_agamma is None else f"{spin_tune_agamma:.8f}",
                "n/a"
                if tws.get("spin_polarization_eq") is None
                else f"{tws['spin_polarization_eq']:.6e}",
                "n/a"
                if tws.get("spin_t_pol_buildup_s") is None
                else f"{tws['spin_t_pol_buildup_s']:.6e}",
                "n/a"
                if tws.get("spin_t_pol_buildup_turns") is None
                else f"{tws['spin_t_pol_buildup_turns']:.3f}",
            )
        )

    widths = [len(hh) for hh in headers]
    for row in rows:
        for ii, val in enumerate(row):
            widths[ii] = max(widths[ii], len(val))

    print("\nMEAN-RADIATION SPIN TWISS TABLE")
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    print(" | ".join(hh.ljust(widths[ii]) for ii, hh in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(val.ljust(widths[ii]) for ii, val in enumerate(row)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))


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
        default="quantum",
        help="Radiation model for tracking (default: quantum)",
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
    parser.add_argument(
        "--spin-pol-y",
        type=float,
        default=0.924,
        metavar="POL",
        help="Initial spin polarization along y, |S|=1 (default: 0.924)",
    )
    parser.add_argument(
        "--spin-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for s_x/s_z draws (default: bunch-seed + 1)",
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
        choices=tuple(CASE_SPECS) + ("all",),
        default=["all"],
        help="Which cases to run (default: all)",
    )
    parser.add_argument(
        "--undulator-shift-x",
        type=float,
        default=-DEFAULT_UNDULATOR_SHIFT_X,
        metavar="DX",
        help=(
            "Horizontal undulator field offset for *-offset cases [m] "
            f"(default: {-DEFAULT_UNDULATOR_SHIFT_X} = -1 mm)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute tracking even when cached data exists",
    )
    args = parser.parse_args()

    if "all" in args.cases:
        case_keys = list(CASE_SPECS)
    else:
        case_keys = args.cases

    field_fitter = load_field_fit() if any(CASE_SPECS[c][0] is not None for c in case_keys) else None
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
    for case_key in case_keys:
        model, default_shift_x = CASE_SPECS[case_key]
        shift_x = args.undulator_shift_x if case_key.endswith("-offset") else default_shift_x
        base_key = case_key.removesuffix("-offset")
        label = CASE_BASE_LABELS[base_key]
        if case_key.endswith("-offset"):
            label = f"{label} ({shift_x * 1e3:+.1f} mm x)"
        cache_key = _cache_key(
            case_key=case_key,
            label=label,
            num_turns=args.num_turns,
            radiation_model=args.radiation_model,
            num_particles=args.num_particles,
            bunch_seed=args.bunch_seed,
            position_sigmas=position_sigmas,
            undulator_shift_x=shift_x,
            num_wiggler_sites=len(WIGGLER_PLACES_SPIN) if model is not None else 0,
            spin_pol_y=args.spin_pol_y,
            spin_seed=args.spin_seed if args.spin_seed is not None else args.bunch_seed + 1,
        )
        cache_path = CACHE_DIR / f"{cache_key}.npz"
        legacy_cache_key = _legacy_cache_key(
            case_key=case_key,
            label=label,
            num_turns=args.num_turns,
            radiation_model=args.radiation_model,
            num_particles=args.num_particles,
            bunch_seed=args.bunch_seed,
            position_sigmas=position_sigmas,
            undulator_shift_x=shift_x,
        )
        legacy_cache_path = CACHE_DIR / f"{legacy_cache_key}.npz"

        if cache_path.exists() and not args.force:
            print(f"\nLoading cached case: {label}")
            results.append(_load_case_cache(cache_path))
        elif (
            args.spin_pol_y == 1.0
            and legacy_cache_path.exists()
            and not args.force
        ):
            print(f"\nLoading cached case (legacy key): {label}")
            results.append(_load_case_cache(legacy_cache_path))
        else:
            print(f"\nRunning case: {label} ...")
            case_result = run_case(
                name=label,
                model=model,
                field_fitter=field_fitter,
                num_turns=args.num_turns,
                radiation_model=args.radiation_model,
                with_progress=progress,
                num_particles=args.num_particles,
                bunch_seed=args.bunch_seed,
                position_sigmas=position_sigmas,
                undulator_shift_x=shift_x,
                spin_pol_y=args.spin_pol_y,
                spin_seed=args.spin_seed,
            )
            _save_case_cache(cache_path, case_result["monitor"], case_result["summary"])
            results.append(_load_case_cache(cache_path))

    print_summaries(results)
    print_mean_spin_table(results)
    plot_spin_vs_turn(results, save_path=args.save, spin_pol_y=args.spin_pol_y)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
