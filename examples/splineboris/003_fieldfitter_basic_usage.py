from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xtrack._temp.splineboris.field_fitter import FieldFitter
from xtrack._temp.splineboris.splineboris_sequence import SplineBorisSequence


'''
Basic usage of FieldFitter.

The constructor runs the fit.

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

# for der in range(0, deg + 1):
#     fitter.plot_fields(der=der)

# Poster-style panel: square figure + zoom on s so B_y oscillations stay readable.
# Comment out the ``plot_fields`` loop above if you only want this figure.
_s = np.asarray(fitter.s_full, dtype=float)
_by_raw = fitter.df_on_axis_raw[("Bnorm", 0)].to_numpy()
_by_fit = fitter.df_on_axis_fit[("Bnorm", 0)].to_numpy()
_s_zoom_m = 1.1  # half-width [m]; widen/narrow to taste for your A0 crop
_m = (_s >= -_s_zoom_m) & (_s <= _s_zoom_m)
_border_s = np.array([], dtype=float)
_dfp = fitter.df_fit_pars
if _dfp is not None and not _dfp.empty:
    _fc = np.asarray(_dfp.index.get_level_values("field_component"))
    _dx = np.asarray(_dfp.index.get_level_values("derivative_x")).astype(int)
    _sel = (_fc == "Bnorm") & (_dx == 0)
    if np.any(_sel):
        _ss = np.asarray(_dfp.index.get_level_values("s_start"), dtype=float)[_sel]
        _se = np.asarray(_dfp.index.get_level_values("s_end"), dtype=float)[_sel]
        _border_s = np.unique(np.concatenate((_ss, _se)))

_poster_rc = {
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
}
with plt.rc_context(_poster_rc):
    fig_poster, ax_poster = plt.subplots(figsize=(10, 4), layout="constrained")
    ax_poster.plot(_s[_m], _by_raw[_m], color="#1f77b4", label=r"$B_y$ data")
    ax_poster.plot(_s[_m], _by_fit[_m], "--", color="#ff7f0e", label=r"$B_y$ fit")
    for _sb in _border_s:
        if -_s_zoom_m <= float(_sb) <= _s_zoom_m:
            ax_poster.axvline(float(_sb), color="0.45", ls="--", lw=0.9, alpha=0.55)
    ax_poster.set_xlabel(r"Longitudinal position $s$ [m]")
    ax_poster.set_ylabel(r"Vertical field $B_y$ [T]")
    ax_poster.set_title(
        rf"Magnetic field at $(X,Y)=({fitter.xy_point[0]},\,{fitter.xy_point[1]})$ m"
    )
    ax_poster.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="0.75",
        framealpha=1.0,
    )
    ax_poster.grid(True, alpha=0.25)
    ax_poster.margins(x=0.02, y=0.06)
    # fig_poster.savefig("by_poster_panel.pdf", dpi=300, bbox_inches="tight")
    plt.show()
# fitter.plot_integrated_fields()

exit()

# Build a piecewise sequence of SplineBoris elements from fitted parameters.
seq = SplineBorisSequence(
    df_fit_pars=fitter.df_fit_pars,
    multipole_order=deg + 1,
    steps_per_point=1,
)

# Compare field continuity at boundaries:
# right edge of element i (s=length_i) vs left edge of element i+1 (s=0).
x0 = 0.001
y0 = 0.001

edge_diffs = []
edge_phi_diffs = []
for i in range(seq.n_pieces - 1):
    left_elem = seq.elements[i]
    right_elem = seq.elements[i + 1]
    left_field = np.array(left_elem.get_field(x0, y0, left_elem.length), dtype=float)
    right_field = np.array(right_elem.get_field(x0, y0, 0.0), dtype=float)
    left_phi = float(left_elem.get_phi(x0, y0, left_elem.length))
    right_phi = float(right_elem.get_phi(x0, y0, 0.0))
    edge_diffs.append(left_field - right_field)
    edge_phi_diffs.append(left_phi - right_phi)

edge_diffs = np.array(edge_diffs)
edge_phi_diffs = np.array(edge_phi_diffs, dtype=float)
edge_diffs_sum = np.sum(edge_diffs, axis=0)
edge_phi_diffs_sum = float(np.sum(edge_phi_diffs))

energy = 2.7e9
momentum = np.sqrt(energy**2 - 511e3**2) / 299792458.0

theta = edge_phi_diffs / momentum

# print("edge_diffs [dBx, dBy, dBs] per boundary:")
# print(edge_diffs)
# print("sum(edge_diffs):")
# print(edge_diffs_sum)
print("edge_phi_diffs [dphi] per boundary:")
print(edge_phi_diffs)
print("sum(edge_phi_diffs):")
print(edge_phi_diffs_sum)

print("theta [rad] per boundary:")
print(theta)
print("sum(theta):")
print(np.sum(theta))