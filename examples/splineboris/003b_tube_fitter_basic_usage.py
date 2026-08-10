from pathlib import Path

import pandas as pd

from xtrack._temp.splineboris.tube_fitter import TubeFitter


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
    residual_tol=1e-3,
    distance_unit=dz,
    deg=deg,
    field_tol=1e-3,
)

fitter.fit()

for der in range(0, deg + 1):
    fitter.plot_fields(der=der)

fitter.plot_integrated_fields()
