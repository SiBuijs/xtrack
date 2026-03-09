import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import pandas as pd
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack._temp.field_fitter import FieldFitter
import matplotlib.pyplot as plt
import time

# ── Setup (reused from 005_solenoid.py) ─────────────────────────────────────
interval = 30
dx = 0.001
dy = 0.001
multipole_order = 2

delta = np.array([0])
p0 = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, q0=1,
    energy0=45.6e6,
    x=1e-3,
    px=-1e-3 * (1 + delta),
    y=1e-3,
    delta=delta,
)

sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)


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
# Pre-computed with BorisSpatialIntegrator (n_steps=1_000_000, delta=0).
# Uncomment the block below to recompute:

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

x_ref = np.array([-7.56888172431699896e-02])
px_ref = np.array([-8.14758308547530730e-03])
y_ref = np.array([1.07304136022629976e-01])
py_ref = np.array([1.17566359622342190e-02])

# ── Fixed spline fit (built once) ───────────────────────────────────────────
n_spline_points = 10001  # 10000 intervals ≈ 3 mm per spline over 30 m
x_axis = np.linspace(
    -multipole_order * dx / 2, multipole_order * dx / 2, multipole_order + 1)
y_axis = np.linspace(
    -multipole_order * dy / 2, multipole_order * dy / 2, multipole_order + 1)
z_axis = np.linspace(0, interval, n_spline_points)
x_grid, y_grid, z_grid = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
bx, by, bz = sf.get_field(x_grid.ravel(), y_grid.ravel(), z_grid.ravel())
df_raw = pd.DataFrame(
    np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel(),
                     bx, by, bz]),
    columns=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])
fitter = FieldFitter(
    raw_data=df_raw, xy_point=(0, 0), distance_unit=1,
    min_region_size=10, deg=multipole_order - 1, field_tol=1e-4,
)
fitter.fit()
df_fit_pars = fitter.df_fit_pars
n_spline_intervals = n_spline_points - 1
print(f"Spline fit built: {n_spline_intervals} intervals over {interval} m "
      f"({interval / n_spline_intervals * 1e3:.1f} mm per interval)")

# ── Sweep configuration ─────────────────────────────────────────────────────
steps_per_point_list = [1, 2, 4, 8, 16]
n_warmup = 1
n_repeats = 10

methods = ["SplineBoris", "BorisSpatial", "VariableSolenoid"]
results = {
    m: {"n_steps": [], "steps_per_point": [], "median_s": [],
         "err_x": [], "err_px": [], "err_y": [], "err_py": []}
    for m in methods
}

_COORD_LABELS = [r"$|\Delta x|$", r"$|\Delta p_x|$",
                 r"$|\Delta y|$", r"$|\Delta p_y|$"]


def compute_errors(p):
    """Return per-coordinate max-abs errors vs reference."""
    ex = np.max(np.abs(p.x - x_ref))
    epx = np.max(np.abs(p.px - px_ref))
    ey = np.max(np.abs(p.y - y_ref))
    epy = np.max(np.abs(p.py - py_ref))
    return ex, epx, ey, epy


def store_result(method, spp, n_steps_total, t, ex, epx, ey, epy):
    r = results[method]
    r["steps_per_point"].append(spp)
    r["n_steps"].append(n_steps_total)
    r["median_s"].append(t["median_s"])
    r["err_x"].append(ex)
    r["err_px"].append(epx)
    r["err_y"].append(ey)
    r["err_py"].append(epy)


# ── Main sweep ──────────────────────────────────────────────────────────────
for spp in steps_per_point_list:
    n_steps_total = n_spline_intervals * spp
    print(f"steps_per_point = {spp}  (n_steps_total = {n_steps_total})")

    # --- SplineBorisSequence (fixed fit, varying steps_per_point) ---
    seq = xt.SplineBorisSequence(
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
    store_result("SplineBoris", spp, n_steps_total, t, ex, epx, ey, epy)
    print(f"  SplineBoris       err_x={ex:.4e}  t={t['median_s']:.4f}s")

    # --- BorisSpatialIntegrator (matched total step count) ---
    boris = xt.BorisSpatialIntegrator(
        fieldmap_callable=get_field, s_start=0, s_end=interval,
        n_steps=n_steps_total)
    boris.log_trajectories = False

    p_test = p0.copy()
    boris.track(p_test)
    ex, epx, ey, epy = compute_errors(p_test)
    t = benchmark_track(lambda: boris.track(p0.copy()),
                        n_warmup=n_warmup, n_repeats=n_repeats)
    store_result("BorisSpatial", spp, n_steps_total, t, ex, epx, ey, epy)
    print(f"  BorisSpatial      err_x={ex:.4e}  t={t['median_s']:.4f}s")

    # --- VariableSolenoid (matched total step count) ---
    z_axis_vs = np.linspace(0, interval, n_steps_total + 1)
    Bz_axis = sf.get_field(
        0 * z_axis_vs, 0 * z_axis_vs, z_axis_vs)[2]
    ks = Bz_axis / brho
    dz = z_axis_vs[1] - z_axis_vs[0]
    line_varsol = xt.Line(elements=[
        xt.VariableSolenoid(length=dz, ks_profile=[ks[ii], ks[ii + 1]])
        for ii in range(len(z_axis_vs) - 1)
    ])
    line_varsol.build_tracker()

    p_test = p0.copy()
    line_varsol.track(p_test)
    ex, epx, ey, epy = compute_errors(p_test)
    t = benchmark_track(lambda: line_varsol.track(p0.copy()),
                        n_warmup=n_warmup, n_repeats=n_repeats)
    store_result("VariableSolenoid", spp, n_steps_total, t, ex, epx, ey, epy)
    print(f"  VariableSolenoid  err_x={ex:.4e}  t={t['median_s']:.4f}s")

    print()

# ── Summary table ────────────────────────────────────────────────────────────
header = (f"{'spp':>4s} {'n_steps':>8s}  "
          f"{'SB err_x':>10s} {'SB time':>8s}  "
          f"{'Boris err_x':>12s} {'Boris time':>10s}  "
          f"{'VS err_x':>10s} {'VS time':>8s}")
print("=" * len(header))
print(header)
print("-" * len(header))
for i, spp in enumerate(steps_per_point_list):
    ns = results["SplineBoris"]["n_steps"][i]
    row = f"{spp:>4d} {ns:>8d}  "
    for m in methods:
        row += (f"{results[m]['err_x'][i]:>10.4e} "
                f"{results[m]['median_s'][i]:>8.4f}  ")
    print(row)
print("=" * len(header))

# ── Plotting ─────────────────────────────────────────────────────────────────
markers = {"SplineBoris": "o", "BorisSpatial": "s", "VariableSolenoid": "^"}
coord_keys = ["err_x", "err_px", "err_y", "err_py"]
coord_labels = [r"$|\Delta x|$", r"$|\Delta p_x|$",
                r"$|\Delta y|$", r"$|\Delta p_y|$"]

# ── Plot 1: Per-coordinate error vs n_steps (2x2) ───────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for ax, ck, cl in zip(axes.ravel(), coord_keys, coord_labels):
    for m in methods:
        ax.loglog(results[m]["n_steps"], results[m][ck],
                  marker=markers[m], label=m)
    ax.set_xlabel(r"$n_{\mathrm{steps}}$")
    ax.set_ylabel(f"Max {cl}")
    ax.set_title(f"{cl} vs number of integration steps")
    ax.legend(fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
fig.suptitle("Per-coordinate error vs number of integration steps", fontsize=13)
fig.tight_layout()

# ── Plot 2: Computation time vs n_steps ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for m in methods:
    ax.loglog(results[m]["n_steps"], results[m]["median_s"],
              marker=markers[m], label=m)
ax.set_xlabel(r"$n_{\mathrm{steps}}$")
ax.set_ylabel(r"Median computing time [s]")
ax.set_title("Median computing time vs number of integration steps")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()

# ── Plot 3: Per-coordinate error vs computation time (2x2) ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for ax, ck, cl in zip(axes.ravel(), coord_keys, coord_labels):
    for m in methods:
        ax.loglog(results[m]["median_s"], results[m][ck],
                  marker=markers[m], label=m)
    ax.set_xlabel(r"Median computing time [s]")
    ax.set_ylabel(f"Max {cl}")
    ax.set_title(f"{cl} vs computation time")
    ax.legend(fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
fig.suptitle("Per-coordinate error vs computation time", fontsize=13)
fig.tight_layout()

plt.show()
