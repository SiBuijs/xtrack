from pathlib import Path

import pandas as pd

from xtrack._temp.tube_fitter import TubeFitter


'''
How few frames can TubeFitter get away with on this field map, for a given
fit-quality target?

TubeFitter's own residual print (see tube_fit_residual_rms, added in
_report_tube_fit_residual) already tells us the tube fit is genuinely
overdetermined -- but how much of that comes from n_frames being generous
(2200 in 003c, ~1 per z grid point) versus actually needed? This script
answers that by passing residual_tol to TubeFitter, which internally
searches for the smallest n_frames whose relative residual (RMS residual /
RMS field, same metric as the fit() printout) drops below a target
tolerance (geometric-doubling bracket + integer bisection -- see
TubeFitter._search_n_frames), plots the search trace itself
(plot_n_frames_search()) once done, and falls back to the DOF ceiling with
a warning if the target turns out to be unreachable -- so there's nothing
left to handle here.
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

print(f"Target: worst-case relative residual <= {TARGET_REL_RESIDUAL * 100:.1f}%\n")

fitter = TubeFitter(
    raw_data=df_raw_data,
    distance_unit=dz,
    deg=deg,
    kernel=kernel,
    field_tol=1e-3,
    y_symmetry=False,
    residual_tol=TARGET_REL_RESIDUAL,
)
rel_bskew, rel_bnorm = fitter.n_frames_search_trace[fitter.n_frames]
print(f"\nn_frames selected: {fitter.n_frames} "
      f"(Bskew={rel_bskew * 100:.2f}%, Bnorm={rel_bnorm * 100:.2f}%)")
