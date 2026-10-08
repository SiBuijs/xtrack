from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xtrack._temp.splineboris import TubeFitter, LongitudinalFitter


'''
How accurately do the SplineBoris elements reproduce the raw field map, as a
function of the number of tube frames (stage 1) and of the element length
(stage 2)?

For every (n_frames, n_elements) the field of the exported elements is
evaluated at every raw-map point (on and off axis) and compared with the
map. Two effects show up:
    - stage 1: tent frames coarser than the map planes slightly smooth the
      on-axis components (relative bias ~ (k * Delta_frame)^2 / 12), which
      sets a floor no matter how short the elements are;
    - stage 2: longer elements follow the field less well (error ~ Delta^5)
      until the frame data themselves become the limit.
'''

PERIOD = 0.036  # [m]
DEG = 2
ELEMENTS_PER_PERIOD = [6, 9, 12, 18, 24, 36]
N_FRAMES_LIST = [550, 1100, 2201]  # 2201: one frame per map plane
END_CONDITION = "free"  # this map is not field-free at its ends

file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
df_raw_data = pd.read_csv(
    file_path, sep=r"\s+", header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])

idx = df_raw_data.index
x, y, s = (idx.get_level_values(lvl).to_numpy() * 1e-3 for lvl in "XYZ")
b_raw = df_raw_data[["Bx", "By", "Bs"]].to_numpy()
b_ref = np.max(np.abs(b_raw[:, 1]))


def field_from_line(line, nodes, x, y, s):
    '''Field of the SplineBoris elements at the points (x, y, s).'''
    k = np.clip(np.searchsorted(nodes, s, side="right") - 1, 0, len(nodes) - 2)
    b = np.zeros((len(s), 3))
    for i in np.unique(k):
        m = k == i
        s_local = np.clip(s[m] - nodes[i], 0, line.elements[i].length)
        b[m] = np.column_stack(line.elements[i].get_field(x[m], y[m], s_local))
    return b


results = {}
for n_frames in N_FRAMES_LIST:
    tube_fitter = TubeFitter(raw_data=df_raw_data, n_frames=n_frames,
                             distance_unit=1e-3, deg=DEG)
    tube_fitter.fit()
    z, F, names = tube_fitter.on_axis_multipoles()
    z_bs, bs = tube_fitter.on_axis_bs()

    for per_period in ELEMENTS_PER_PERIOD:
        n_elements = round((z[-1] - z[0]) / PERIOD * per_period)
        if n_elements + 4 > n_frames:
            continue  # "free" ends: E + 4 unknowns, need at least as many data points
        lf = LongitudinalFitter(z[0], z[-1], n_elements=n_elements,
                                end_condition=END_CONDITION)
        lf.fit(z, F, names)
        lf.fit(z_bs, bs, [("Bs", 0)])
        b_fit = field_from_line(lf.to_line(multipole_order=DEG + 1), lf.nodes, x, y, s)
        rms = np.sqrt(np.mean((b_fit - b_raw) ** 2, axis=0)) / b_ref
        results[n_frames, per_period] = rms
        print(f"n_frames={n_frames:5d}  elements/period={per_period:3d}  "
              f"(E={n_elements:5d})  RMS error / max|By|: "
              f"Bx {rms[0]:.2e}  By {rms[1]:.2e}  Bs {rms[2]:.2e}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, constrained_layout=True)
for i_comp, (ax, comp) in enumerate(zip(axes, ("Bx", "By", "Bs"))):
    for n_frames in N_FRAMES_LIST:
        pp = [p for p in ELEMENTS_PER_PERIOD if (n_frames, p) in results]
        ax.loglog(pp, [results[n_frames, p][i_comp] for p in pp], "o-",
                  label=f"n_frames={n_frames} ({n_frames * PERIOD / 2.2:.0f}/period)")
    ax.set_title(comp)
    ax.set_xlabel("elements per period")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("RMS field error / max|By| (all map points)")
axes[0].legend()
fig.suptitle(f"SplineBoris field vs raw undulator map, end_condition={END_CONDITION!r}")
plt.show()
