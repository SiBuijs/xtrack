import json
import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import pandas as pd
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack._temp.field_fitter import FieldFitter
from xtrack._temp.splineboris_sequence import SplineBorisSequence
import matplotlib.pyplot as plt
import time

TICK_FONTSIZE = 16
LABEL_FONTSIZE = 17
TITLE_FONTSIZE = 18

# ── Setup (reused from 005_solenoid.py) ─────────────────────────────────────
interval = 8
dx = 0.001
dy = 0.001
multipole_order = 4

delta = np.array([0])
p0 = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, q0=1,
    energy0=45.6e6,
    x=1e-3,
    px=-1e-3 * (1 + delta),
    y=1e-3,
    delta=delta,
)

sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=4)


def get_field(x, y, z):
    return sf.get_field(x, y, z)


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


# ── Derived quantities needed for VariableSolenoid ──────────────────────────
P0_J = p0.p0c[0] * qe / clight
brho = P0_J / qe / p0.q0

# ── Reference solution ──────────────────────────────────────────────────────
# Computed with BorisSpatialIntegrator (n_steps=1_000_000, delta=0).
# n_steps_ref = 1_000_000
# print(f"Computing reference solution with BorisSpatialIntegrator "
#       f"(n_steps={n_steps_ref}) ...")
# boris_ref = xt.BorisSpatialIntegrator(
#     fieldmap_callable=get_field,
#     s_start=0,
#     s_end=interval,
#     n_steps=n_steps_ref,
# )
# boris_ref.log_trajectories = False
# p_ref = p0.copy()
# boris_ref.track(p_ref)
# x_ref = p_ref.x.copy()
# px_ref = p_ref.px.copy()
# y_ref = p_ref.y.copy()
# py_ref = p_ref.py.copy()
# print(f"  x_ref = {x_ref[0]:.17e}")
# print(f"  px_ref = {px_ref[0]:.17e}")
# print(f"  y_ref = {y_ref[0]:.17e}")
# print(f"  py_ref = {py_ref[0]:.17e}")

x_ref = -1.27416576780741407e-03
px_ref = -5.30211431567380841e-04
y_ref = 5.46316372713433196e-03
py_ref = 2.04827892292546408e-03

def compute_errors(p):
    """Return per-coordinate max relative errors vs reference."""
    ex = np.max(np.abs((p.x - x_ref) / x_ref))
    epx = np.max(np.abs((p.px - px_ref) / px_ref))
    ey = np.max(np.abs((p.y - y_ref) / y_ref))
    epy = np.max(np.abs((p.py - py_ref) / py_ref))
    return ex, epx, ey, epy


coord_keys = ["err_x", "err_px", "err_y", "err_py"]
coord_labels = [r"$|\Delta x / x_\mathrm{ref}|$",
                r"$|\Delta p_x / p_{x,\mathrm{ref}}|$",
                r"$|\Delta y / y_\mathrm{ref}|$",
                r"$|\Delta p_y / p_{y,\mathrm{ref}}|$"]
n_warmup = 1
n_repeats = 10

# ══════════════════════════════════════════════════════════════════════════════
# Study 1: Fix spline resolution, vary steps_per_point
# ══════════════════════════════════════════════════════════════════════════════
# Set to a file path to save/load results; only missing spp values will be run
study1_results_file = None  # e.g. "study1_results.json"

run_study_1 = True  # set to True to rerun study 1
if run_study_1:
    n_spline_points = 10001
    x_axis = np.linspace(
        -multipole_order * dx / 2, multipole_order * dx / 2,
        multipole_order + 1)
    y_axis = np.linspace(
        -multipole_order * dy / 2, multipole_order * dy / 2,
        multipole_order + 1)
    z_axis = np.linspace(0, interval, n_spline_points)
    x_grid, y_grid, z_grid = np.meshgrid(
        x_axis, y_axis, z_axis, indexing="ij")
    bx, by, bz = sf.get_field(
        x_grid.ravel(), y_grid.ravel(), z_grid.ravel())
    df_raw = pd.DataFrame(
        np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel(),
                         bx, by, bz]),
        columns=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
    ).set_index(["X", "Y", "Z"])
    fitter = FieldFitter(
        raw_data=df_raw, xy_point=(0, 0), distance_unit=1,
        min_region_size=10, deg=multipole_order - 1, field_tol=1e-8,
    )
    fitter.fit()
    df_fit_pars = fitter.df_fit_pars
    n_spline_intervals = n_spline_points - 1
    print(f"Spline fit built: {n_spline_intervals} intervals over "
          f"{interval} m "
          f"({interval / n_spline_intervals * 1e3:.1f} mm per interval)")

    steps_per_point_list = [1, 2, 4, 8, 16, 32]
    # methods = ["SplineBoris", "BorisSpatial", "VariableSolenoid"]
    methods = ["SplineBoris", "BorisSpatial"]
    results = {
        m: {"n_steps": [], "steps_per_point": [], "median_s": [],
             "err_x": [], "err_px": [], "err_y": [], "err_py": []}
        for m in methods
    }

    # Load existing results if available
    existing_spp = []
    if study1_results_file:
        try:
            with open(study1_results_file) as f:
                loaded = json.load(f)
            existing_spp = loaded["SplineBoris"]["steps_per_point"]
            for m in methods:
                for k in results[m]:
                    results[m][k] = loaded[m][k]
            print(f"Loaded Study 1 results for spp={existing_spp}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Could not load {study1_results_file}: {e}")

    spp_to_run = [s for s in steps_per_point_list if s not in existing_spp]
    if not spp_to_run:
        print("All spp already in results, skipping computation.")

    def store_result(method, spp, n_steps_total, t, ex, epx, ey, epy):
        r = results[method]
        r["steps_per_point"].append(spp)
        r["n_steps"].append(n_steps_total)
        r["median_s"].append(t["median_s"])
        r["err_x"].append(ex)
        r["err_px"].append(epx)
        r["err_y"].append(ey)
        r["err_py"].append(epy)

    for spp in spp_to_run:
        n_steps_total = n_spline_intervals * spp
        print(f"steps_per_point = {spp}  "
              f"(n_steps_total = {n_steps_total})")

        seq = SplineBorisSequence(
            df_fit_pars=df_fit_pars,
            multipole_order=multipole_order,
            steps_per_point=spp,
        )
        line_spline = seq.to_line()
        line_spline.build_tracker()
        p_test = p0.copy()
        line_spline.track(p_test)
        ex, epx, ey, epy = compute_errors(p_test)
        t = benchmark_track(lambda: line_spline.track(p0.copy()),
                            n_warmup=n_warmup, n_repeats=n_repeats)
        store_result("SplineBoris", spp, n_steps_total,
                     t, ex, epx, ey, epy)
        print(f"  SplineBoris       err_x={ex:.4e}  "
              f"t={t['median_s']:.4f}s")

        boris = xt.BorisSpatialIntegrator(
            fieldmap_callable=get_field, s_start=0, s_end=interval,
            n_steps=n_steps_total)
        boris.log_trajectories = False
        p_test = p0.copy()
        boris.track(p_test)
        ex, epx, ey, epy = compute_errors(p_test)
        t = benchmark_track(lambda: boris.track(p0.copy()),
                            n_warmup=n_warmup, n_repeats=n_repeats)
        store_result("BorisSpatial", spp, n_steps_total,
                     t, ex, epx, ey, epy)
        print(f"  BorisSpatial      err_x={ex:.4e}  "
              f"t={t['median_s']:.4f}s")

        # z_axis_vs = np.linspace(0, interval, n_steps_total + 1)
        # Bz_axis = sf.get_field(
        #     0 * z_axis_vs, 0 * z_axis_vs, z_axis_vs)[2]
        # ks = Bz_axis / brho
        # dz = z_axis_vs[1] - z_axis_vs[0]
        # line_varsol = xt.Line(elements=[
        #     xt.VariableSolenoid(
        #         length=dz, ks_profile=[ks[ii], ks[ii + 1]])
        #     for ii in range(len(z_axis_vs) - 1)
        # ])
        # line_varsol.build_tracker()
        # p_test = p0.copy()
        # line_varsol.track(p_test)
        # ex, epx, ey, epy = compute_errors(p_test)
        # t = benchmark_track(lambda: line_varsol.track(p0.copy()),
        #                     n_warmup=n_warmup, n_repeats=n_repeats)
        # store_result("VariableSolenoid", spp, n_steps_total,
        #              t, ex, epx, ey, epy)
        # print(f"  VariableSolenoid  err_x={ex:.4e}  "
        #       f"t={t['median_s']:.4f}s")
        print()

    if study1_results_file and spp_to_run:
        with open(study1_results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved Study 1 results to {study1_results_file}")

    # Sort by spp for consistent ordering (in case of load-then-append)
    order = np.argsort(results["SplineBoris"]["steps_per_point"])
    for m in methods:
        for k in results[m]:
            results[m][k] = [results[m][k][i] for i in order]

    header = (f"{'spp':>4s} {'n_steps':>8s}  "
              f"{'SB err_x':>10s} {'SB time':>8s}  "
              f"{'Boris err_x':>12s} {'Boris time':>10s}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i in range(len(results["SplineBoris"]["steps_per_point"])):
        spp = results["SplineBoris"]["steps_per_point"][i]
        ns = results["SplineBoris"]["n_steps"][i]
        row = f"{spp:>4d} {ns:>8d}  "
        for m in methods:
            row += (f"{results[m]['err_x'][i]:>10.4e} "
                    f"{results[m]['median_s'][i]:>8.4f}  ")
        print(row)
    print("=" * len(header))

    # markers = {"SplineBoris": "o", "BorisSpatial": "s",
    #            "VariableSolenoid": "^"}
    markers = {"SplineBoris": "o", "BorisSpatial": "s"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, ck, cl in zip(axes.ravel(), coord_keys, coord_labels):
        for m in methods:
            ax.loglog(results[m]["n_steps"], results[m][ck],
                      marker=markers[m], label=m)
        ax.set_xlabel(r"$n_{\mathrm{steps}}$", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(cl, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="x", labelbottom=True, labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="y", labelleft=True, labelsize=TICK_FONTSIZE)
        ax.set_title(f"{cl} vs number of integration steps", fontsize=TITLE_FONTSIZE)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Study 1: Per-coordinate error vs n_steps", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        ax.loglog(results[m]["n_steps"], results[m]["median_s"],
                  marker=markers[m], label=m)
    ax.set_xlabel(r"$n_{\mathrm{steps}}$")
    ax.set_ylabel(r"Median computing time [s]")
    ax.set_title("Study 1: Computing time vs n_steps")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, ck, cl in zip(axes.ravel(), coord_keys, coord_labels):
        for m in methods:
            ax.loglog(results[m]["median_s"], results[m][ck],
                      marker=markers[m], label=m)
        ax.set_xlabel(r"Median computing time [s]")
        ax.set_ylabel(cl)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_title(f"{cl} vs computation time")
        ax.legend(fontsize="small")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Study 1: Per-coordinate error vs time", fontsize=13)
    fig.tight_layout()

    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# Study 2: Isolate field approximation effect
#
# Fix data resolution and total integration steps, vary only the number of
# polynomial pieces (n_pieces) to isolate the field approximation error.
# ══════════════════════════════════════════════════════════════════════════════
run_study_2 = False  # set to True to rerun study 2
if run_study_2:
    n_data_points_s2 = 10001  # generous data resolution (10000 intervals)
    spp_s2 = 1                # steps_per_point fixed at 1
    total_steps_s2 = n_data_points_s2 - 1  # = 10000

    n_pieces_list = [5, 10, 20, 50, 100, 200, 500, 1000]#, 2000, 3000]

    results_field = {
        "n_pieces": [], "median_s": [],
        "err_x": [], "err_px": [], "err_y": [], "err_py": [],
        "field_err_B": [],
    }

    # Build field data grid once (same for all n_pieces)
    x_ax = np.linspace(
        -multipole_order * dx / 2, multipole_order * dx / 2,
        multipole_order + 1)
    y_ax = np.linspace(
        -multipole_order * dy / 2, multipole_order * dy / 2,
        multipole_order + 1)
    z_ax = np.linspace(0, interval, n_data_points_s2)
    xg, yg, zg = np.meshgrid(x_ax, y_ax, z_ax, indexing="ij")
    bx_f, by_f, bz_f = sf.get_field(xg.ravel(), yg.ravel(), zg.ravel())
    df_s2 = pd.DataFrame(
        np.column_stack([xg.ravel(), yg.ravel(), zg.ravel(),
                         bx_f, by_f, bz_f]),
        columns=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
    ).set_index(["X", "Y", "Z"])

    # Use BorisSpatial at the same total step count as reference,
    # with trajectory logging for field evaluation along the particle path
    boris_s2 = xt.BorisSpatialIntegrator(
        fieldmap_callable=get_field, s_start=0, s_end=interval,
        n_steps=total_steps_s2)
    boris_s2.log_trajectories = True
    p_boris_s2 = p0.copy()
    boris_s2.track(p_boris_s2)
    x_ref_s2 = p_boris_s2.x.copy()
    px_ref_s2 = p_boris_s2.px.copy()
    y_ref_s2 = p_boris_s2.y.copy()
    py_ref_s2 = p_boris_s2.py.copy()

    # Reference trajectory positions for field evaluation
    traj_x = np.array(boris_s2.x_log)[:, 0]
    traj_y = np.array(boris_s2.y_log)[:, 0]
    traj_z = np.clip(np.array(boris_s2.z_log)[:, 0], 0, interval)
    Bx_true_traj, By_true_traj, Bs_true_traj = sf.get_field(
        traj_x, traj_y, traj_z)
    B_mag_peak_traj = np.max(np.sqrt(
        Bx_true_traj**2 + By_true_traj**2 + Bs_true_traj**2))

    def compute_errors_s2(p):
        """Relative errors vs BorisSpatial at same total step count."""
        ex = np.max(np.abs((p.x - x_ref_s2) / x_ref_s2))
        epx = np.max(np.abs((p.px - px_ref_s2) / px_ref_s2))
        ey = np.max(np.abs((p.y - y_ref_s2) / y_ref_s2))
        epy = np.max(np.abs((p.py - py_ref_s2) / py_ref_s2))
        return ex, epx, ey, epy

    print(f"\n{'='*70}")
    print(f"Study 2: Field approximation (n_data={n_data_points_s2}, "
          f"spp={spp_s2}, total_steps={total_steps_s2})")
    print(f"  Reference: BorisSpatial at {total_steps_s2} steps")
    print(f"{'='*70}")

    for n_pieces in n_pieces_list:
        piece_size_mm = interval / n_pieces * 1e3
        print(f"  n_pieces={n_pieces:>5d}  "
              f"({piece_size_mm:.1f} mm/piece)", end="")

        fit = FieldFitter(
            raw_data=df_s2, xy_point=(0, 0), distance_unit=1,
            deg=multipole_order - 1, field_tol=1e-8,
            n_pieces=n_pieces,
        )
        fit.fit()
        seq_f = SplineBorisSequence(
            df_fit_pars=fit.df_fit_pars,
            multipole_order=multipole_order,
            steps_per_point=spp_s2,
        )

        # Field error along the reference particle trajectory
        B_spline_traj = np.array([
            seq_f.evaluate_field(tx, ty, tz)
            for tx, ty, tz in zip(traj_x, traj_y, traj_z)])
        Bx_sp = B_spline_traj[:, 0]
        By_sp = B_spline_traj[:, 1]
        Bs_sp = B_spline_traj[:, 2]
        field_err = np.max(np.sqrt(
            (Bx_sp - Bx_true_traj)**2 +
            (By_sp - By_true_traj)**2 +
            (Bs_sp - Bs_true_traj)**2)) / B_mag_peak_traj

        line_f = seq_f.to_line()
        line_f.build_tracker()

        p_test = p0.copy()
        line_f.track(p_test)
        ex, epx, ey, epy = compute_errors_s2(p_test)
        t = benchmark_track(lambda: line_f.track(p0.copy()),
                            n_warmup=n_warmup, n_repeats=n_repeats)

        results_field["n_pieces"].append(n_pieces)
        results_field["median_s"].append(t["median_s"])
        results_field["err_x"].append(ex)
        results_field["err_px"].append(epx)
        results_field["err_y"].append(ey)
        results_field["err_py"].append(epy)
        results_field["field_err_B"].append(field_err)
        print(f"  field_err={field_err:.4e}  err_x={ex:.4e}  "
              f"t={t['median_s']:.4f}s")

    # ── Plot 4: Per-coordinate error vs n_pieces (fixed total steps) ─────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, ck, cl in zip(axes.ravel(), coord_keys, coord_labels):
        ax.loglog(results_field["n_pieces"], results_field[ck], "o-",
                  label="SplineBoris vs BorisSpatial")
        ax.loglog(results_field["n_pieces"], results_field["field_err_B"],
                  "d--", color="C2", alpha=0.6,
                  label=r"$|\Delta \mathbf{B}| / |\mathbf{B}|_\mathrm{peak}$"
                        " (on trajectory)")
        ax.set_xlabel("Number of polynomial pieces")
        ax.set_ylabel(cl + r" (rel. to BorisSpatial)")
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_title(f"{cl} vs polynomial pieces")
        ax.legend(fontsize="small")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"SplineBoris error relative to BorisSpatial "
                 f"(total steps = {total_steps_s2})", fontsize=13)
    fig.tight_layout()

    # ── Plot 5: Computation time vs n_pieces ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(results_field["n_pieces"], results_field["median_s"],
              "o-", label="SplineBoris")
    ax.set_xlabel("Number of polynomial pieces")
    ax.set_ylabel("Median computing time [s]")
    ax.set_title(f"SplineBoris time vs polynomial pieces "
                 f"(total steps = {total_steps_s2})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    plt.show()
