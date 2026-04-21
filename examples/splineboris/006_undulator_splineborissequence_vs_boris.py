import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import c as clight
from scipy.constants import e as qe

import xtrack as xt
from xtrack._temp.splineboris_sequence import SplineBorisSequence

TICK_FONTSIZE = 16
LABEL_FONTSIZE = 17
TITLE_FONTSIZE = 18

FIT_PARS_INDEX_COLS = [
    "field_component",
    "derivative_x",
    "region_name",
    "s_start",
    "s_end",
    "idx_start",
    "idx_end",
    "param_index",
]


def make_test_particle(p_ref, n_particles=1):
    p = p_ref.copy()
    p.x = np.full(n_particles, 1e-3)
    p.px = np.full(n_particles, 1e-4)
    p.y = np.full(n_particles, 0.5e-3)
    p.py = np.full(n_particles, -0.5e-4)
    return p


def benchmark_track(track_callable, n_warmup=1, n_repeats=20):
    t0 = time.perf_counter()
    track_callable()
    first_call_s = time.perf_counter() - t0

    for _ in range(n_warmup):
        track_callable()

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        track_callable()
        times.append(time.perf_counter() - t0)

    return {
        "first_call_s": float(first_call_s),
        "median_s": float(np.median(times)),
    }


def extract_multipole_strengths(seq, s_positions, multipole_order, brho,
                                dx=1e-6):
    """
    At each slice midpoint, evaluate the fitted field at several x-offsets
    (y=0), fit a polynomial of degree (multipole_order - 1), and extract
    derivatives d^n B / dx^n for n = 0 .. multipole_order - 1.

    Returns knl[n_slices, n_orders], ksl[n_slices, n_orders], ds[n_slices].
    """
    n_pts = multipole_order + 2
    x_eval = np.linspace(-n_pts // 2 * dx, n_pts // 2 * dx, n_pts)
    n_orders = multipole_order

    ds = np.diff(s_positions)
    s_mid = 0.5 * (s_positions[:-1] + s_positions[1:])
    n_slices = len(s_mid)

    knl = np.zeros((n_slices, n_orders))
    ksl = np.zeros((n_slices, n_orders))

    for i, s in enumerate(s_mid):
        By_vals = np.array(
            [seq.evaluate_field(x_pt, 0.0, s)[1] for x_pt in x_eval])
        Bx_vals = np.array(
            [seq.evaluate_field(x_pt, 0.0, s)[0] for x_pt in x_eval])

        coeff_By = np.polyfit(x_eval, By_vals, n_orders - 1)
        coeff_Bx = np.polyfit(x_eval, Bx_vals, n_orders - 1)

        for n in range(n_orders):
            dBy_n = np.polyval(np.polyder(coeff_By, m=n), 0.0)
            dBx_n = np.polyval(np.polyder(coeff_Bx, m=n), 0.0)
            knl[i, n] = dBy_n * ds[i] / brho
            ksl[i, n] = dBx_n * ds[i] / brho

    return knl, ksl, ds


def build_multipole_line(knl, ksl, ds, p_ref):
    """Build a Drift + thin Multipole line with drift-kick-drift splitting."""
    elements = []
    names = []

    for i in range(len(ds)):
        if i == 0:
            elements.append(xt.Drift(length=ds[i] / 2))
            names.append(f"drift_entry_{i}")
        else:
            elements.append(xt.Drift(length=(ds[i - 1] + ds[i]) / 2))
            names.append(f"drift_{i}")

        elements.append(xt.Multipole(
            knl=knl[i, :].tolist(),
            ksl=ksl[i, :].tolist(),
        ))
        names.append(f"kick_{i}")

    elements.append(xt.Drift(length=ds[-1] / 2))
    names.append("drift_exit")

    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = p_ref.copy()
    line.build_tracker()
    return line


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent

    df_fit_pars = pd.read_csv(
        base_dir / "test_data" / "sls" / "undulator_fit_pars.csv",
        index_col=FIT_PARS_INDEX_COLS,
    )
    multipole_order = 3

    p_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)
    P0_J = p_ref.p0c[0] * qe / clight
    brho = P0_J / qe

    df_reset = df_fit_pars.reset_index()
    s_start = df_reset["s_start"].min()
    s_end = df_reset["s_end"].max()
    print(f"Undulator s-range: [{s_start}, {s_end}] m")

    seq_ref = SplineBorisSequence(
        df_fit_pars=df_fit_pars,
        multipole_order=multipole_order,
        steps_per_point=1,
    )
    n_intervals = sum(int(e.n_steps) for e in seq_ref.elements)
    print(f"SplineBoris: {len(seq_ref.elements)} elements, "
          f"{n_intervals} intervals (spp=1)")

    # ── Reference solution ───────────────────────────────────────────────────
    # Pre-computed with SplineBorisSequence (steps_per_point=64).
    # Uncomment the block below to recompute:

    # recompute_reference = True
    # if recompute_reference:
    #     spp_ref = 64
    #     print(f"\nComputing reference (SplineBorisSequence, spp={spp_ref})…")
    #     seq_hi = xt.SplineBorisSequence(
    #         df_fit_pars=df_fit_pars,
    #         multipole_order=multipole_order,
    #         steps_per_point=spp_ref,
    #     )
    #     line_hi = seq_hi.to_line()
    #     line_hi.particle_ref = p_ref.copy()
    #     line_hi.build_tracker()
    #     p_hi = make_test_particle(p_ref)
    #     line_hi.track(p_hi)
    #     x_ref = p_hi.x.copy()
    #     px_ref = p_hi.px.copy()
    #     y_ref = p_hi.y.copy()
    #     py_ref = p_hi.py.copy()
    #     print(f"  x_ref  = np.array([{x_ref[0]:.17e}])")
    #     print(f"  px_ref = np.array([{px_ref[0]:.17e}])")
    #     print(f"  y_ref  = np.array([{y_ref[0]:.17e}])")
    #     print(f"  py_ref = np.array([{py_ref[0]:.17e}])")
    #
    #     seq_lo = xt.SplineBorisSequence(
    #         df_fit_pars=df_fit_pars,
    #         multipole_order=multipole_order,
    #         steps_per_point=spp_ref // 2,
    #     )
    #     line_lo = seq_lo.to_line()
    #     line_lo.particle_ref = p_ref.copy()
    #     line_lo.build_tracker()
    #     p_lo = make_test_particle(p_ref)
    #     line_lo.track(p_lo)
    #     print(f"  spp={spp_ref // 2} vs spp={spp_ref} agreement:")
    #     print(f"    |dx|  = {abs(p_lo.x[0] - x_ref[0]):.2e}")
    #     print(f"    |dpx| = {abs(p_lo.px[0] - px_ref[0]):.2e}")
    #     print(f"    |dy|  = {abs(p_lo.y[0] - y_ref[0]):.2e}")
    #     print(f"    |dpy| = {abs(p_lo.py[0] - py_ref[0]):.2e}")

    x_ref = np.array([1.78760557712828402e-03])
    px_ref = np.array([9.32072832640937443e-05])
    y_ref = np.array([5.91476566940394669e-05])
    py_ref = np.array([-5.40107590347957501e-05])

    def compute_errors(p):
        ex = np.max(np.abs((p.x - x_ref) / x_ref))
        epx = np.max(np.abs((p.px - px_ref) / px_ref))
        ey = np.max(np.abs((p.y - y_ref) / y_ref))
        epy = np.max(np.abs((p.py - py_ref) / py_ref))
        return ex, epx, ey, epy

    coord_keys = ["err_x", "err_px", "err_y", "err_py"]
    coord_labels = [
        r"$|\Delta x / x_\mathrm{ref}|$",
        r"$|\Delta p_x / p_{x,\mathrm{ref}}|$",
        r"$|\Delta y / y_\mathrm{ref}|$",
        r"$|\Delta p_y / p_{y,\mathrm{ref}}|$",
    ]
    n_warmup = 1
    n_repeats = 10

    # ══════════════════════════════════════════════════════════════════════════
    # Study 1: SplineBoris self-convergence (vary steps_per_point)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("Study 1: SplineBoris self-convergence")
    print(f"{'=' * 70}")

    steps_per_point_list = [1, 2, 4, 8, 16, 32]
    results_sb = {
        "steps_per_point": [], "n_steps": [], "median_s": [],
        "err_x": [], "err_px": [], "err_y": [], "err_py": [],
    }

    for spp in steps_per_point_list:
        n_steps_total = n_intervals * spp
        print(f"  spp={spp:>2d}  (total steps = {n_steps_total})", end="")

        seq_i = SplineBorisSequence(
            df_fit_pars=df_fit_pars,
            multipole_order=multipole_order,
            steps_per_point=spp,
        )
        line_i = seq_i.to_line()
        line_i.particle_ref = p_ref.copy()
        line_i.build_tracker()

        p_test = make_test_particle(p_ref)
        line_i.track(p_test)
        ex, epx, ey, epy = compute_errors(p_test)

        t = benchmark_track(
            lambda: line_i.track(make_test_particle(p_ref)),
            n_warmup=n_warmup, n_repeats=n_repeats,
        )

        results_sb["steps_per_point"].append(spp)
        results_sb["n_steps"].append(n_steps_total)
        results_sb["median_s"].append(t["median_s"])
        results_sb["err_x"].append(ex)
        results_sb["err_px"].append(epx)
        results_sb["err_y"].append(ey)
        results_sb["err_py"].append(epy)

        print(f"  err_x={ex:.4e}  t={t['median_s']:.4f}s")

    header = (f"{'spp':>4s} {'n_steps':>8s}  "
              f"{'err_x':>10s} {'err_px':>10s} "
              f"{'err_y':>10s} {'err_py':>10s} {'time':>8s}")
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")
    for i in range(len(steps_per_point_list)):
        print(f"{results_sb['steps_per_point'][i]:>4d} "
              f"{results_sb['n_steps'][i]:>8d}  "
              f"{results_sb['err_x'][i]:>10.4e} "
              f"{results_sb['err_px'][i]:>10.4e} "
              f"{results_sb['err_y'][i]:>10.4e} "
              f"{results_sb['err_py'][i]:>10.4e} "
              f"{results_sb['median_s'][i]:>8.4f}")
    print(f"{'=' * len(header)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Study 2: SplineBoris vs Naive Multipole Kicks
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("Study 2: SplineBoris vs Multipole kicks")
    print(f"{'=' * 70}")

    n_slices_list = [50, 100, 200, 500, 1000, 2000]
    results_mp = {
        "n_slices": [], "median_s": [],
        "err_x": [], "err_px": [], "err_y": [], "err_py": [],
    }

    for n_slices in n_slices_list:
        print(f"  n_slices={n_slices:>5d}  ", end="", flush=True)

        s_positions = np.linspace(s_start, s_end, n_slices + 1)
        knl, ksl, ds = extract_multipole_strengths(
            seq_ref, s_positions, multipole_order, brho)

        line_mp = build_multipole_line(knl, ksl, ds, p_ref)

        p_test = make_test_particle(p_ref)
        line_mp.track(p_test)
        ex, epx, ey, epy = compute_errors(p_test)

        t = benchmark_track(
            lambda: line_mp.track(make_test_particle(p_ref)),
            n_warmup=n_warmup, n_repeats=n_repeats,
        )

        results_mp["n_slices"].append(n_slices)
        results_mp["median_s"].append(t["median_s"])
        results_mp["err_x"].append(ex)
        results_mp["err_px"].append(epx)
        results_mp["err_y"].append(ey)
        results_mp["err_py"].append(epy)

        print(f"err_x={ex:.4e}  t={t['median_s']:.4f}s")

    header = (f"{'n_slices':>8s}  "
              f"{'MP err_x':>10s} {'MP err_px':>10s} "
              f"{'MP err_y':>10s} {'MP err_py':>10s} {'MP time':>8s}")
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")
    for i in range(len(n_slices_list)):
        print(f"{results_mp['n_slices'][i]:>8d}  "
              f"{results_mp['err_x'][i]:>10.4e} "
              f"{results_mp['err_px'][i]:>10.4e} "
              f"{results_mp['err_y'][i]:>10.4e} "
              f"{results_mp['err_py'][i]:>10.4e} "
              f"{results_mp['median_s'][i]:>8.4f}")
    print(f"{'=' * len(header)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════════════

    # Plot 1: Per-coordinate error vs total steps/slices
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, ck, cl in zip(axes.ravel(), coord_keys, coord_labels):
        ax.loglog(results_sb["n_steps"], results_sb[ck],
                  "o-", label="SplineBoris")
        ax.loglog(results_mp["n_slices"], results_mp[ck],
                  "s-", label="Multipole kicks")
        ax.set_xlabel(r"Total integration steps / slices", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(cl, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Per-coordinate error vs total steps", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()

    # Plot 2: Timing vs total steps/slices
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(results_sb["n_steps"], results_sb["median_s"],
              "o-", label="SplineBoris")
    ax.loglog(results_mp["n_slices"], results_mp["median_s"],
              "s-", label="Multipole kicks")
    ax.set_xlabel(r"Total integration steps / slices", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Median computing time [s]", fontsize=LABEL_FONTSIZE)
    ax.set_title("Computing time vs steps", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    # Plot 3: Per-coordinate error vs time (accuracy-vs-cost)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, ck, cl in zip(axes.ravel(), coord_keys, coord_labels):
        ax.loglog(results_sb["median_s"], results_sb[ck],
                  "o-", label="SplineBoris")
        ax.loglog(results_mp["median_s"], results_mp[ck],
                  "s-", label="Multipole kicks")
        ax.set_xlabel("Median computing time [s]", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(cl, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Per-coordinate error vs computation time", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()

    # Plot 4: SplineBoris self-convergence
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, ck, cl in zip(axes.ravel(), coord_keys, coord_labels):
        ax.loglog(results_sb["steps_per_point"], results_sb[ck], "o-")
        ax.set_xlabel("steps_per_point", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(cl, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("SplineBoris self-convergence", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()

    plt.show()
