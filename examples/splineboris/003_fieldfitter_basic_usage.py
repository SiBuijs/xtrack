from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xtrack._temp.field_fitter import FieldFitter


'''
Basic usage of FieldFitter.

This script fits a field map and saves the fit parameters to a file.

It plots the fit results for each derivative order.
It also plots the integrated field along the longitudinal direction.

The raw data only has three transverse x positions, which means the highest order polynomial that we can fit is 2.
This also means that we can only incorporate up to the second derivative of the field into the fit (sextupole components).
'''

dz = 0.001  # Step size in the z (longitudinal) direction for numerical differentiation

# Convert the field map to a DataFrame
file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
df_raw_data = pd.read_csv(
    file_path, sep=r"\s+", header=None,
    names=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
).set_index(["X", "Y", "Z"])

deg = 2

fitter = FieldFitter(
    raw_data=df_raw_data,
    xy_point=(0.0, 0.0),
    distance_unit=dz,
    min_region_size=10,
    deg=deg,
    field_tol=1e-3,
)

fitter.fit()

# Plot raw data with region boundaries (no fitted curves)
s = fitter.s_full
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
by_raw = fitter.df_on_axis_raw[("Bnorm", 0)].to_numpy()
by_abs_max = np.max(np.abs(by_raw))
y_pad = 0.05 * by_abs_max if by_abs_max > 0 else 1e-12
y_limits = (-by_abs_max - y_pad, by_abs_max + y_pad)
for ax, field, label in zip(
    axes,
    ["Bskew", "Bnorm", "Bs"],
    [r"$B_x$", r"$B_y$", r"$B_s$"],
):
    ax.plot(s, fitter.df_on_axis_raw[(field, 0)].to_numpy(), label="Raw data")
    sub_df = fitter.df_fit_pars.loc[(field, 0)].reset_index()
    s_edges = np.unique(np.concatenate((sub_df["s_start"].to_numpy(), sub_df["s_end"].to_numpy())))
    for s_edge in s_edges:
        ax.axvline(float(s_edge), color="k", linestyle="--", linewidth=1, alpha=0.25)
    ax.set_ylabel(f"{label} [T]")
    ax.set_ylim(*y_limits)
    ax.grid(True)
    ax.legend(loc="best")
axes[0].set_title("Raw on-axis field with fitted segment boundaries")
axes[-1].set_xlabel(r"Longitudinal position, $s$ [m]")

print("Fit parameters:")
print(fitter.df_fit_pars)

for der in range(0, deg + 1):
    fitter.plot_fields(der=der)

fitter.plot_integrated_fields()