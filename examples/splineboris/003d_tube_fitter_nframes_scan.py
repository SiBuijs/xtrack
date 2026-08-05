import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xtrack._temp.tube_fitter import TubeFitter, _generate_pq_pairs


'''
How few frames can TubeFitter get away with on this field map, for a given
fit-quality target?

TubeFitter's own residual print (see tube_fit_residual_rms, added in
_report_tube_fit_residual) already tells us the tube fit is genuinely
overdetermined -- but how much of that comes from n_frames being generous
(2200 in 003c, ~1 per z grid point) versus actually needed? This script
answers that by searching for the smallest n_frames whose relative residual
(RMS residual / RMS field, same metric as the fit() printout) drops below a
target tolerance.

A dense linear/log scan over n_frames would work but wastes most of its
fits far from the interesting region. Instead: a short geometric probe
brackets the transition (doubling n_frames until the tolerance is met),
then integer bisection within that bracket converges to the minimal
n_frames in ~log2(n_max) fits. All evaluated points are kept and plotted so
the shape of the residual-vs-n_frames curve outside the bracket is visible
too.
'''

TARGET_REL_RESIDUAL = 0.001  # fraction of field RMS -- the convergence target

dz = 0.001
deg = 2
kernel = "cubic"

file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
df_raw_data = pd.read_csv(
    file_path, sep=r"\s+", header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])

n_x = df_raw_data.index.get_level_values("X").nunique()
n_y = df_raw_data.index.get_level_values("Y").nunique()
n_z = df_raw_data.index.get_level_values("Z").nunique()
basis_order = {"tent": 1, "quadratic": 2, "cubic": 3}[kernel]
n_min = basis_order + 1

# DOF ceiling: the tube system has 2*N_x*N_y*N_z equations (Bx, By at every
# grid point) and n_frames*n_pq unknowns (n_pq = size of the (p, q)
# multipole basis, same one TubeFitter builds via _generate_pq_pairs with
# its default y_symmetry=True, fit_skew=True). n_frames beyond this makes
# the system underdetermined regardless of how much z data exists -- unlike
# a plain "1 frame per z point" cap, this accounts for the 2*N_x*N_y
# transverse equations feeding each z sample.
n_pq = len(_generate_pq_pairs(deg + 1, y_symmetry=True, fit_skew=True))
n_max = (2 * n_x * n_y * n_z) // n_pq

results: dict[int, tuple[float, float]] = {}  # n_frames -> (rel_bskew, rel_bnorm)


def relative_residuals(n_frames: int) -> tuple[float, float]:
    if n_frames in results:
        return results[n_frames]

    with contextlib.redirect_stdout(io.StringIO()):
        fitter = TubeFitter(
            raw_data=df_raw_data,
            n_frames=n_frames,
            distance_unit=dz,
            deg=deg,
            kernel=kernel,
            field_tol=1e-3,
            y_symmetry=False,
        )
        fitter.fit()

    # field_rms is independent of n_frames (same raw data every time), so
    # recompute it from the stored residual/relative print logic directly:
    # tube_fit_residual_rms holds absolute RMS; rebuild the relative version
    # the same way _report_tube_fit_residual does.
    b = fitter._b_vec
    rels = {}
    for field, rows in (("Bskew", slice(0, None, 2)), ("Bnorm", slice(1, None, 2))):
        rms = fitter.tube_fit_residual_rms[field]
        field_rms = float(np.sqrt(np.mean(b[rows] ** 2)))
        rels[field] = rms / field_rms if field_rms > 0 else 0.0

    out = (rels["Bskew"], rels["Bnorm"])
    results[n_frames] = out
    print(f"  n_frames={n_frames:5d}  Bskew={out[0] * 100:6.2f}%  Bnorm={out[1] * 100:6.2f}%")
    return out


def meets_target(n_frames: int) -> bool:
    rel_bskew, rel_bnorm = relative_residuals(n_frames)
    return max(rel_bskew, rel_bnorm) <= TARGET_REL_RESIDUAL


print(f"Target: worst-case relative residual <= {TARGET_REL_RESIDUAL * 100:.1f}%")
print(f"Searching n_frames in [{n_min}, {n_max}]\n")

print("Coarse probe (doubling n_frames until target is met):")
lo = n_min
hi = n_max
if not meets_target(hi):
    print(f"\nTarget not reached even at n_frames={n_max} (residual floor of the fit).")
    n_star = None
else:
    n_probe = n_min
    while n_probe < hi and not meets_target(n_probe):
        lo = n_probe
        n_probe = min(n_probe * 2, hi)
    hi = n_probe

    print(f"\nBracketed between n_frames={lo} (fails) and n_frames={hi} (meets target).")
    print("Bisecting:")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if meets_target(mid):
            hi = mid
        else:
            lo = mid
    n_star = hi
    rel_bskew, rel_bnorm = results[n_star]
    print(f"\nSmallest n_frames meeting target: {n_star} "
          f"(Bskew={rel_bskew * 100:.2f}%, Bnorm={rel_bnorm * 100:.2f}%)")

# --- plot everything that was evaluated along the way ---
ns = sorted(results)
bskew = [results[n][0] * 100 for n in ns]
bnorm = [results[n][1] * 100 for n in ns]

fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
ax.plot(ns, bskew, "o-", color="tab:blue", label="Bskew")
ax.plot(ns, bnorm, "o-", color="tab:orange", label="Bnorm")
ax.axhline(TARGET_REL_RESIDUAL * 100, color="k", linestyle="--", linewidth=1,
           label=f"target ({TARGET_REL_RESIDUAL * 100:.1f}%)")
if n_star is not None:
    ax.axvline(n_star, color="tab:green", linestyle=":", linewidth=1.5,
               label=f"n_frames = {n_star}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("n_frames")
ax.set_ylabel("Tube fit residual (% of field RMS)")
ax.set_title("TubeFitter residual vs n_frames (undulator_field_map.txt)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()
