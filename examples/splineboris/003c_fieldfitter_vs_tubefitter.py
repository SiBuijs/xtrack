from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xtrack._temp.field_fitter import FieldFitter
from xtrack._temp.tube_fitter import TubeFitter


'''
Compare FieldFitter (sequential Hermite-piecewise regions, boundaries chosen
adaptively via peak/valley detection) against TubeFitter (global sparse tube
fit, uniformly spaced frames) on the same field map, for each derivative
order.

For each derivative order this plots, per field component:
    - the raw data straight from the field-map file (der=0 only -- higher
      derivatives are not directly measured, only derived)
    - FieldFitter's raw on-axis series (der=0: identical to the file data;
      der>0: derived from a per-z transverse polynomial fit across the
      available X points)
    - TubeFitter's raw on-axis series (der=0: identical to the file data;
      der>0: the fitted multipole B-spline evaluated on-axis)
    - FieldFitter's Hermite fit (piecewise quartic Hermite regions)
    - TubeFitter's Hermite fit (same quartic Hermite export format, derived
      from the tube's global B-spline fit)

Note: for TubeFitter, the der>0 "raw" and "Hermite fit" curves come from
evaluating the same fitted B-spline, so they overlap exactly -- only der=0
distinguishes measured data from the fit for that fitter.
'''

dz = 0.001  # Step size in the z (longitudinal) direction for numerical differentiation
deg = 2

file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"

df_raw_fieldfitter = pd.read_csv(
    file_path, sep=r"\s+", header=None,
    names=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
).set_index(["X", "Y", "Z"])

field_fitter = FieldFitter(
    raw_data=df_raw_fieldfitter,
    xy_point=(0.0, 0.0),
    distance_unit=dz,
    min_region_size=10,
    deg=deg,
    field_tol=1e-3,
)
field_fitter.fit()

df_raw_tubefitter = pd.read_csv(
    file_path, sep=r"\s+", header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])

tube_fitter = TubeFitter(
    raw_data=df_raw_tubefitter,
    n_frames=2200,
    distance_unit=dz,
    deg=deg,
    field_tol=1e-3,
)
tube_fitter.fit()

s = field_fitter.s_full

# (FieldFitter field name, TubeFitter field name, plot label)
FIELDS = [
    ("Bskew", "Bx", "Horizontal"),
    ("Bnorm", "By", "Vertical"),
    ("Bs", "Bs", "Longitudinal"),
]


def get_series(df, field, der):
    try:
        return df[(field, der)].to_numpy()
    except KeyError:
        ref = df.iloc[:, 0].to_numpy()
        return np.zeros_like(ref)


def plot_comparison(der):
    fig, axes = plt.subplots(3, figsize=(11, 7), sharex=True, constrained_layout=True)

    for ax, (ff_field, tf_field, label) in zip(axes, FIELDS):
        if der == 0:
            ax.plot(s, get_series(field_fitter.df_on_axis_raw, ff_field, 0),
                     color="k", linewidth=1.5, label="Raw data (file)")

        ax.plot(s, get_series(field_fitter.df_on_axis_raw, ff_field, der),
                 color="tab:blue", linestyle=":", label="FieldFitter raw (transverse poly)")
        ax.plot(s, get_series(tube_fitter.df_on_axis_raw, tf_field, der),
                 color="tab:orange", linestyle=":", label="TubeFitter raw (tube fit)")

        ax.plot(s, get_series(field_fitter.df_on_axis_fit, ff_field, der),
                 color="tab:green", linestyle="--", label="FieldFitter Hermite fit")
        ax.plot(s, get_series(tube_fitter.df_on_axis_fit, tf_field, der),
                 color="tab:red", linestyle="--", label="TubeFitter Hermite fit")

        ax.set_ylabel(f"{label} field [T]")
        ax.grid()

    axes[0].set_title(f"FieldFitter vs TubeFitter, derivative order {der}")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel(r"Longitudinal Position, $s$ [m]")
    plt.show()


for der in range(0, deg + 1):
    plot_comparison(der)


