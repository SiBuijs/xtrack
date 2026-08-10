import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xtrack._temp.tube_fitter import KERNEL_TO_DEGREE, TubeFitter


'''
Does the tube fit's residual converge with n_frames at the rate B-spline
approximation theory predicts, and does that rate actually differ by
kernel? A degree-p B-spline approximates a smooth function with local error
~ h^(p+1) (h ~ 1/n_frames), so on a log-log plot of relative tube fit
residual vs n_frames, each kernel's slope should land near -(degree+1):
tent (degree 1) ~ -2, quadratic (degree 2) ~ -3, cubic (degree 3) ~ -4.

This fits the same SLS undulator field map as 003b/003d at a range of
n_frames for each kernel, and compares the measured log-log slope to that
prediction. Since total fit degrees of freedom is n_frames regardless of
kernel degree (see tube_fitter.py:_setup_frames -- the B-spline always has
exactly n_frames control points), any crossover between kernels here is a
direct look at what "more expressive per control point" (a higher-degree,
wider-support basis function) buys you in practice, versus its cost
(stronger smoothness coupling across neighbouring frames).

First run surfaced something the naive prediction doesn't account for: at
low n_frames, *every* kernel sits at ~100% residual regardless of degree,
then falls off a cliff before settling into the predicted power law. That
cliff lines up with the undulator's own period -- below ~2 frames per
period, no kernel (of any degree) can resolve the oscillation at all, so
the residual is dominated by that aliasing floor, not by the kernel's
approximation order. Fitting the log-log slope over the whole grid (i.e.
including the aliasing regime) badly understates the true convergence
order, so the period is estimated here (from on-axis By zero-crossings)
purely to know which points are past that floor and safe to use for the
slope fit.
'''

dz = 0.001
deg = 2
field_tol = 1e-3
ASYMPTOTIC_FRAMES_PER_PERIOD = 4.0  # only fit the convergence slope past this

file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
df_raw_data = pd.read_csv(
    file_path, sep=r"\s+", header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])

N_FRAMES_GRID = np.unique(np.geomspace(20, 1200, 9).astype(int))


def estimate_period(df_raw_data: pd.DataFrame) -> tuple[float, float]:
    """(period, n_periods) of the on-axis By oscillation, in raw Z units."""
    on_axis = df_raw_data.xs((0.0, 0.0), level=["X", "Y"]).sort_index()
    z = on_axis.index.get_level_values("Z").to_numpy(dtype=float)
    by = on_axis["By"].to_numpy(dtype=float)
    crossings = np.where(np.diff(np.sign(by)) != 0)[0]
    period = 2.0 * float(np.median(np.diff(z[crossings])))
    n_periods = (z.max() - z.min()) / period
    return period, n_periods


PERIOD, N_PERIODS = estimate_period(df_raw_data)
print(f"Estimated undulator period: {PERIOD:.3g} (raw Z units), "
      f"{N_PERIODS:.1f} periods over the full map\n")


def relative_residuals(fitter: TubeFitter) -> dict[str, float]:
    """Same metric TubeFitter itself uses for residual_tol (rms / field rms)."""
    b = fitter._b_vec
    n_bxby = fitter._n_bx_by_rows
    rel = {}
    for field, rows in (("Bskew", slice(0, n_bxby, 2)), ("Bnorm", slice(1, n_bxby, 2))):
        rms = fitter.tube_fit_residual_rms[field]
        field_rms = float(np.sqrt(np.mean(b[rows] ** 2)))
        rel[field] = rms / field_rms if field_rms > 0 else 0.0
    return rel


def fit_slope(n_frames: list[int], residual: list[float]) -> float:
    slope, _ = np.polyfit(np.log(n_frames), np.log(residual), 1)
    return float(slope)


results: dict[str, dict[str, list[float]]] = {}

for kernel in ("tent", "quadratic", "cubic"):
    n_min = KERNEL_TO_DEGREE[kernel] + 1
    n_values = [int(n) for n in N_FRAMES_GRID if n > n_min]
    bskew_rel, bnorm_rel = [], []
    print(f"=== kernel={kernel} ===")
    for n_frames in n_values:
        with contextlib.redirect_stdout(io.StringIO()):
            fitter = TubeFitter(
                raw_data=df_raw_data,
                n_frames=n_frames,
                distance_unit=dz,
                deg=deg,
                kernel=kernel,
                field_tol=field_tol,
                y_symmetry=False,
            )
            fitter.fit()
        rel = relative_residuals(fitter)
        bskew_rel.append(rel["Bskew"])
        bnorm_rel.append(rel["Bnorm"])
        frames_per_period = n_frames / N_PERIODS
        print(f"  n_frames={n_frames:5d}  ({frames_per_period:5.1f} frames/period)  "
              f"Bskew={rel['Bskew'] * 100:7.3f}%  Bnorm={rel['Bnorm'] * 100:7.3f}%")
    results[kernel] = {"n_frames": n_values, "Bskew": bskew_rel, "Bnorm": bnorm_rel}

asymptotic_n_frames = int(np.ceil(ASYMPTOTIC_FRAMES_PER_PERIOD * N_PERIODS))
print(f"\nAsymptotic regime: n_frames >= {asymptotic_n_frames} "
      f"(>= {ASYMPTOTIC_FRAMES_PER_PERIOD:.0f} frames/period)")

print("\n=== Measured vs predicted convergence order (log-log slope) ===")
for kernel, data in results.items():
    predicted = -(KERNEL_TO_DEGREE[kernel] + 1)
    mask = [n >= asymptotic_n_frames for n in data["n_frames"]]
    n_fit = [n for n, m in zip(data["n_frames"], mask) if m]
    if len(n_fit) < 2:
        print(f"{kernel:>10s}: not enough points past the aliasing regime to fit a slope")
        continue
    slope_skew = fit_slope(n_fit, [r for r, m in zip(data["Bskew"], mask) if m])
    slope_norm = fit_slope(n_fit, [r for r, m in zip(data["Bnorm"], mask) if m])
    slope_skew_naive = fit_slope(data["n_frames"], data["Bskew"])
    slope_norm_naive = fit_slope(data["n_frames"], data["Bnorm"])
    print(f"{kernel:>10s}: predicted {predicted:+.0f}   "
          f"asymptotic-only Bskew {slope_skew:+.2f} Bnorm {slope_norm:+.2f}   "
          f"(naive full-grid Bskew {slope_skew_naive:+.2f} Bnorm {slope_norm_naive:+.2f})")

fig, (ax_skew, ax_norm) = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
colors = {"tent": "tab:red", "quadratic": "tab:orange", "cubic": "tab:blue"}

for kernel, data in results.items():
    predicted = -(KERNEL_TO_DEGREE[kernel] + 1)
    mask = [n >= asymptotic_n_frames for n in data["n_frames"]]
    n_fit = [n for n, m in zip(data["n_frames"], mask) if m]
    for ax, field in ((ax_skew, "Bskew"), (ax_norm, "Bnorm")):
        n = data["n_frames"]
        r = data[field]
        r_fit = [v for v, m in zip(r, mask) if m]
        slope = fit_slope(n_fit, r_fit) if len(n_fit) >= 2 else float("nan")
        ax.plot(n, r, "o-", color=colors[kernel],
                label=f"{kernel} (measured {slope:.2f}, predicted {predicted:+.0f})")

for ax, field in ((ax_skew, "Bskew"), (ax_norm, "Bnorm")):
    ax.axvline(2.0 * N_PERIODS, color="k", linestyle=":", linewidth=1, alpha=0.6,
               label="~2 frames/period (Nyquist floor)")
    ax.axvline(asymptotic_n_frames, color="k", linestyle="--", linewidth=1, alpha=0.6,
               label=f"{ASYMPTOTIC_FRAMES_PER_PERIOD:.0f} frames/period (slope fit starts)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_frames")
    ax.set_ylabel(f"{field} relative tube fit residual")
    ax.set_title(f"{field} convergence vs n_frames")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

fig.suptitle("TubeFitter residual convergence vs n_frames, by kernel (SLS undulator map)")
plt.show()
