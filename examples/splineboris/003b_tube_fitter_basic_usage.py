from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from xtrack._temp.splineboris.tube_fitter import KERNEL_TO_DEGREE, TubeFitter


'''
Basic usage of TubeFitter.

This script fits the same field map as 003a_fieldfitter_basic_usage.py, but
using the tube approach (global sparse fit over a longitudinal B-spline
basis, see Riemann & Aiba, IPAC2021, TUPAB238) instead of FieldFitter's
sequential Hermite-piecewise regions.

It plots the fit results for each derivative order and the integrated field
along the longitudinal direction.

The raw data only has three transverse x positions, which means the highest
order polynomial that we can fit is 2. This also means that we can only
incorporate up to the second derivative of the field into the fit (sextupole
components).
'''

dz = 0.001  # Step size in the z (longitudinal) direction for numerical differentiation

# Convert the field map to a DataFrame. TubeFitter expects columns Bx, By, Bs
# (FieldFitter uses the equivalent Bskew, Bnorm, Bs naming for the same data).
file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
df_raw_data = pd.read_csv(
    file_path, sep=r"\s+", header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])

deg = 2

fitter = TubeFitter(
    raw_data=df_raw_data,
    n_frames=880,
    distance_unit=dz,
    kernel="quadratic",
    deg=deg,
    field_tol=1e-3,
)

fitter.fit()

for der in range(0, deg + 1):
    fitter.plot_fields(der=der)

fitter.plot_integrated_fields()

# --- Illustrate the three longitudinal B-spline kernels -------------------
# Show the individual basis functions beta_j(z) for each kernel choice
# (see TubeFitter's "Kernel choice" docstring), over a small number of
# frames so the individual basis functions stay visible. Dashed vertical
# lines mark the frame positions: for "tent" the interior knots coincide
# with the frames, while "quadratic"/"cubic" use fewer, wider-spaced knots
# since their basis functions span multiple frame-widths.
n_frames_demo = 10
z_min = df_raw_data.index.get_level_values("Z").min()
z_max = df_raw_data.index.get_level_values("Z").max()
s_dense = np.linspace(z_min, z_max, 2000)

fig, axes = plt.subplots(len(KERNEL_TO_DEGREE), 1, figsize=(10, 7), sharex=True, constrained_layout=True)
for ax, (kernel, k) in zip(axes, KERNEL_TO_DEGREE.items()):
    frames = np.linspace(z_min, z_max, n_frames_demo)
    if n_frames_demo > k + 1:
        interior = np.linspace(z_min, z_max, n_frames_demo - k + 1)[1:-1]
    else:
        interior = np.array([], dtype=float)
    knots = np.r_[np.full(k + 1, z_min), interior, np.full(k + 1, z_max)]

    total = np.zeros_like(s_dense)
    for j in range(n_frames_demo):
        coeffs = np.zeros(n_frames_demo)
        coeffs[j] = 1.0
        basis = BSpline(knots, coeffs, k)
        values = basis(s_dense)
        ax.plot(s_dense, values)
        total += values
    ax.plot(s_dense, total, color="k", linewidth=2, label="sum")

    for f in frames:
        ax.axvline(f, color="k", linestyle="--", linewidth=0.7, alpha=0.5)

    ax.set_ylabel(r"$\beta_j(z)$")
    ax.set_title(f"{kernel} (order {k})")
    ax.legend(loc="lower right")

axes[-1].set_xlabel("z")
fig.suptitle(f"Longitudinal B-spline basis functions ({n_frames_demo} frames)")
plt.show()
