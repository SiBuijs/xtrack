from pathlib import Path

import pandas as pd

from xtrack._temp.splineboris import TubeFitter, LongitudinalFitter


"""
Basic usage of the two-stage field-map fit.

Stage 1 (TubeFitter) finds the on-axis components at its frame positions:
    ("By", n) = d^n B_y / dx^n (0, 0, s),  ("Bx", n) = d^n B_x / dx^n (0, 0, s)
Stage 2 (LongitudinalFitter) fits each component, and the map's own on-axis
B_s, with a C3 quartic B-spline on uniform nodes, which is what the
SplineBoris elements store.

Things to play with below:
    - n_frames (stage 1): one frame per map plane avoids tent smoothing.
    - n_elements / points_per_element (stage 2): more elements follow the
      field better (error ~ Delta^5) but amplify noise in the s-derivatives.
    - end_condition: "zero" forces f, f', f'', f''' to zero at both ends,
      "free" does not. This map is not field-free at its ends, so "free".

The raw data only has three transverse x positions, so the highest
transverse order we can fit is 2 (deg=2, sextupole components).
"""

file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
df_raw_data = pd.read_csv(
    file_path, sep=r"\s+", header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])

deg = 2

# Stage 1
tube_fitter = TubeFitter(raw_data=df_raw_data, n_frames=2200, distance_unit=1e-3, deg=deg)
tube_fitter.fit()
z, F, names = tube_fitter.on_axis_multipoles()
z_bs, bs = tube_fitter.on_axis_bs()

# Stage 2
lf = LongitudinalFitter(
    s_start=z[0],
    s_end=z[-1],
    points_per_element=2,       # or n_elements=...
    end_condition="free",
    preserve_integral=False,
    period=0.036,               # undulator period, to warn if elements are too long
)
lf.fit(z, F, names)             # all multipoles share one factorisation
lf.fit(z_bs, bs, [("Bs", 0)])   # B_s has its own positions
print(f"{lf.n_elements} elements of {lf.delta * 1e3:.2f} mm")

for der in range(deg + 1):
    lf.plot_fields(der=der)
lf.plot_fields(der=0, integrated=True)

# The SplineBoris line: element k stores (f(s_k), f'(s_k), f(s_k+1),
# f'(s_k+1), mean_k) for every component.
line = lf.to_line(multipole_order=deg + 1)
print(line.get_table())
