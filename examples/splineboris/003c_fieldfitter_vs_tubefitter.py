from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xtrack._temp.splineboris.field_fitter import FieldFitter
from xtrack._temp.splineboris.tube_fitter import TubeFitter
from xtrack._temp.splineboris.splineboris_sequence import SplineBorisSequence


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

TUBE_N_FRAMES = 2200

tube_fitter = TubeFitter(
    raw_data=df_raw_tubefitter,
    n_frames=TUBE_N_FRAMES,
    distance_unit=dz,
    deg=deg,
    kernel="tent",
    field_tol=1e-3,
    y_symmetry=True,
)
tube_fitter.fit()

# Same n_frames/deg/field_tol, only the longitudinal B-spline kernel differs
# -- isolates the effect of kernel order (tent=O(h^2), quadratic=O(h^3),
# cubic=O(h^4)) from everything else. See
# examples/splineboris/claude_notes/tubefitter_nframes_nonmonotonic.md for
# why this comparison matters.
tube_fitter_quadratic = TubeFitter(
    raw_data=df_raw_tubefitter,
    n_frames=TUBE_N_FRAMES,
    distance_unit=dz,
    deg=deg,
    kernel="quadratic",
    field_tol=1e-3,
    y_symmetry=True,
)
tube_fitter_quadratic.fit()

tube_fitter_cubic = TubeFitter(
    raw_data=df_raw_tubefitter,
    n_frames=TUBE_N_FRAMES,
    distance_unit=dz,
    deg=deg,
    kernel="cubic",
    field_tol=1e-3,
    y_symmetry=True,
)
tube_fitter_cubic.fit()

# --- SplineBoris field-prediction residual: FieldFitter vs TubeFitter -----
# Both fitters only ever export q=0 (skew) and q=1 (norm) content; the
# downstream SplineBoris Table-1/Schueren evaluator regenerates everything
# else (including off-axis structure) from that alone. Comparing predicted
# vs raw field here -- rather than each fitter's own internal regression
# residual -- is the fairer, apples-to-apples number: both go through the
# exact same field-evaluation code path from here on.
multipole_order = deg + 1
seq_field_fitter = SplineBorisSequence(
    df_fit_pars=field_fitter.df_fit_pars, multipole_order=multipole_order, steps_per_point=1,
)
seq_tube_fitter = SplineBorisSequence(
    df_fit_pars=tube_fitter.df_fit_pars, multipole_order=multipole_order, steps_per_point=1,
)
seq_tube_fitter_quadratic = SplineBorisSequence(
    df_fit_pars=tube_fitter_quadratic.df_fit_pars, multipole_order=multipole_order, steps_per_point=1,
)
seq_tube_fitter_cubic = SplineBorisSequence(
    df_fit_pars=tube_fitter_cubic.df_fit_pars, multipole_order=multipole_order, steps_per_point=1,
)


def spline_boris_residual(seq, df_raw, on_axis_only=False):
    """RMS / relative RMS of the SplineBoris field prediction vs raw (X,Y,Z)
    data, grouped by Z so ``get_field`` is called once per longitudinal slice
    (it only accepts a scalar ``s``, but array ``x``/``y``)."""
    preds = {"Bskew": [], "Bnorm": [], "Bs": []}
    meas = {"Bskew": [], "Bnorm": [], "Bs": []}
    for z0, grp in df_raw.groupby("Z"):
        if on_axis_only:
            grp = grp[(grp["X"] == 0) & (grp["Y"] == 0)]
            if grp.empty:
                continue
        x = grp["X"].to_numpy() * dz
        y = grp["Y"].to_numpy() * dz
        bx, by, bs = seq.get_field(x, y, float(z0 * dz))
        preds["Bskew"].append(np.atleast_1d(bx))
        preds["Bnorm"].append(np.atleast_1d(by))
        preds["Bs"].append(np.atleast_1d(bs))
        meas["Bskew"].append(grp["Bskew"].to_numpy())
        meas["Bnorm"].append(grp["Bnorm"].to_numpy())
        meas["Bs"].append(grp["Bs"].to_numpy())

    result = {}
    for field in ("Bskew", "Bnorm", "Bs"):
        p = np.concatenate(preds[field])
        m = np.concatenate(meas[field])
        rms = float(np.sqrt(np.mean((p - m) ** 2)))
        field_rms = float(np.sqrt(np.mean(m ** 2)))
        result[field] = (rms, rms / field_rms if field_rms > 0 else 0.0)
    return result


df_raw_for_residual = df_raw_fieldfitter.reset_index()

for title, on_axis_only in (("full grid", False), ("on-axis only (X=0, Y=0)", True)):
    print(f"\n=== SplineBoris field-prediction residual vs raw data ({title}) ===")
    print(f"{'field':8s} {'fitter':20s} {'RMS [T]':>12s} {'rel RMS':>10s}")
    for tag, seq in (
        ("FieldFitter", seq_field_fitter),
        ("TubeFitter (tent)", seq_tube_fitter),
        ("TubeFitter (quadratic)", seq_tube_fitter_quadratic),
        ("TubeFitter (cubic)", seq_tube_fitter_cubic),
    ):
        for field, (rms, rel) in spline_boris_residual(seq, df_raw_for_residual, on_axis_only).items():
            print(f"{field:8s} {tag:20s} {rms:12.4e} {rel * 100:9.3f}%")

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
