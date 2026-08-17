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
file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "simona_field_map.txt"
df_raw_data = pd.read_csv(
    file_path, sep="\t", header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
    dtype=float,
).set_index(["X", "Y", "Z"])

deg = 4

n_frames = 4441

# Below, you can either choose n_frames or residual_tol
# If you choose n_frames, the fitter will simply use that number of frames.
# If you choose residual_tol, the fitter will search for the smallest number of frames
# that meets the specified residual tolerance.
# Some reference times (depdendent on the machine and field map (simona_field_map.txt, deg=4 here)):
# n_frames=1000     : takes ~30 seconds     lands at ~5.7e-3 residual
# residual_tol=1e-3 : takes ~10 minutes     lands at n_frames=2324

import time

# Temporarily silence the automatic n_frames-search convergence plot so it
# doesn't pop up a blocking window mid-timing -- remove this line to see it
# again. (Only affects plot_n_frames_search; plot_fields/plot_integrated_fields
# below are untouched.)
TubeFitter.plot_n_frames_search = lambda self, *a, **kw: None

start_time = time.time()

fitter = TubeFitter(
    raw_data=df_raw_data,
    n_frames=n_frames,
    #residual_tol=1e-3,
    distance_unit=dz,
    deg=deg,
    field_tol=1e-3,
    #tube_radius=0.0005,
)
fitter.fit()

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds for {fitter.n_frames} frames")

for der in range(0, deg + 1):
    fitter.plot_fields(der=der)

fitter.plot_integrated_fields()
