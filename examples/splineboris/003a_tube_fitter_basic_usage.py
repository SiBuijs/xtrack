from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xtrack._temp.splineboris import TubeFitter, LongitudinalFitter


'''
Basic usage of TubeFitter (stage 1) on the large SLS field map, followed by
the longitudinal fit (stage 2, LongitudinalFitter).

The tube fit (Riemann & Aiba, IPAC2021, TUPAB238) fits a scalar potential
over all map points at once and passes on the on-axis components at its
frame positions. The longitudinal fit turns those into the C3 quartic
B-spline stored by the SplineBoris elements.

Plots: data vs fit of every on-axis component, their integrals, and the
field of the exported elements vs the raw map off axis.
'''

dz = 0.001  # the dataset uses mm

file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "simona_field_map.txt"
df_raw_data = pd.read_csv(
    file_path, sep="\t", header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
    dtype=float,
).set_index(["X", "Y", "Z"])

deg = 4

# Below, you can either choose n_frames or residual_tol
# If you choose n_frames, the fitter will simply use that number of frames.
# If you choose residual_tol, the fitter will search for the smallest number of frames
# that meets the specified residual tolerance.
# Some reference times (dependent on the machine and field map (simona_field_map.txt, deg=4 here)):
# n_frames=1000     : takes ~30 seconds     lands at ~5.7e-3 residual
# residual_tol=1e-3 : takes ~10 minutes     lands at n_frames=2324
# If we simply choose n_frames=4441 (one per plane), it takes ~35 seconds and lands at ~6e-6 residual
n_frames = 4441

# Temporarily silence the automatic n_frames-search convergence plot so it
# doesn't pop up a blocking window mid-timing -- remove this line to see it
# again.
TubeFitter.plot_n_frames_search = lambda self, *a, **kw: None

start_time = time.time()
fitter = TubeFitter(
    raw_data=df_raw_data,
    n_frames=n_frames,
    #residual_tol=1e-3,
    distance_unit=dz,
    deg=deg,
    #tube_radius=0.0005,
)
fitter.fit()
print(f"Stage 1: {time.time() - start_time:.1f} s for {fitter.n_frames} frames")

# Stage 2. ~5 frames per element by default (~14 elements per 36 mm period
# here); "free" ends since the map is not field-free at its ends.
start_time = time.time()
z, F, names = fitter.on_axis_multipoles()
lf = LongitudinalFitter(z[0], z[-1], end_condition="free", period=0.036)
lf.fit(z, F, names)
lf.fit(*fitter.on_axis_bs(), [("Bs", 0)])
print(f"Stage 2: {time.time() - start_time:.3f} s for {lf.n_elements} elements")

for der in range(deg + 1):
    lf.plot_fields(der=der)
lf.plot_fields(der=0, integrated=True)

# Field of the exported SplineBoris elements vs the raw map, along a line
# off axis. The tube fit is global, so this checks the whole chain: on-axis
# components, their longitudinal fit, and the off-axis field the elements
# reconstruct from them.
line = lf.to_line(multipole_order=deg + 1)


def field_from_line(x, y, s):
    k = np.clip(np.searchsorted(lf.nodes, s, side="right") - 1, 0, lf.n_elements - 1)
    b = np.zeros((len(s), 3))
    for i in np.unique(k):
        m = k == i
        s_local = np.clip(s[m] - lf.nodes[i], 0, line.elements[i].length)
        b[m] = np.column_stack(line.elements[i].get_field(x, y, s_local))
    return b


x_grid = np.unique(df_raw_data.index.get_level_values("X"))
y_grid = np.unique(df_raw_data.index.get_level_values("Y"))
for x_target, y_target in [(0.0, 0.0), (0.5, 0.0), (0.5, 0.1)]:  # [mm]
    x_mm = x_grid[np.argmin(np.abs(x_grid - x_target))]  # nearest map point
    y_mm = y_grid[np.argmin(np.abs(y_grid - y_target))]
    df_line = df_raw_data.xs((x_mm, y_mm), level=["X", "Y"]).sort_index()
    s = df_line.index.to_numpy() * dz
    b_raw = df_line[["Bx", "By", "Bs"]].to_numpy()
    b_fit = field_from_line(x_mm * dz, y_mm * dz, s)

    fig, axes = plt.subplots(3, 2, figsize=(14, 7), sharex=True, constrained_layout=True)
    for i, comp in enumerate(("Bx", "By", "Bs")):
        axes[i, 0].plot(s, b_raw[:, i], label="Raw map")
        axes[i, 0].plot(s, b_fit[:, i], "--", label="SplineBoris")
        axes[i, 1].plot(s, b_fit[:, i] - b_raw[:, i])
        axes[i, 0].set_ylabel(f"{comp} [T]")
        axes[i, 1].set_ylabel(f"{comp} fit - raw [T]")
        for ax in axes[i]:
            ax.grid()
    axes[0, 0].legend()
    axes[0, 0].set_title(f"Field at (x, y) = ({x_mm:g}, {y_mm:g}) mm")
    axes[0, 1].set_title("Difference")
    axes[-1, 0].set_xlabel("s [m]")
    axes[-1, 1].set_xlabel("s [m]")
    plt.show()
