from pathlib import Path

import numpy as np
import pandas as pd

from xtrack._temp.field_fitter import FieldFitter
from xtrack._temp.splineboris_sequence import SplineBorisSequence


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

# for der in range(0, deg + 1):
#     fitter.plot_fields(der=der)

# fitter.plot_integrated_fields()

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