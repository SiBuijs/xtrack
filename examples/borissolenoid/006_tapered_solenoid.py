"""
Demonstrate BorisSolenoid edge tapering with SplineBoris.

This example builds a tapered representation with:
- left and right SplineBoris edge tapers,
- one central BorisSolenoid core.

The taper lengths are selected automatically from |B / (dB/ds)| and
the edge profile uses endpoint value/derivative constraints with
mean = B_edge / 2 for each field component.
"""

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt


L_COIL = 4.0
A_COIL = 0.3
B0 = 1.5
Z0 = 8.0      # Not centered in [0, LENGTH] to show asymmetric taper lengths.
LENGTH = 30.0
N_STEPS = 20000


def sample_line_on_axis(line, n_points=1201):
    s_edges = np.r_[0.0, np.cumsum([ee.length for ee in line.elements])]
    s_eval = np.linspace(0.0, s_edges[-1], n_points)
    bz = np.empty_like(s_eval)

    for ii, ee in enumerate(line.elements):
        if ii == len(line.elements) - 1:
            mask = (s_eval >= s_edges[ii]) & (s_eval <= s_edges[ii + 1])
        else:
            mask = (s_eval >= s_edges[ii]) & (s_eval < s_edges[ii + 1])
        s_local = s_eval[mask] - s_edges[ii]
        if isinstance(ee, xt.BorisSolenoid):
            bz[mask] = ee.get_field(
                0.0, 0.0, s_local, s_at_element=s_edges[ii])[2]
        else:
            bz[mask] = ee.get_field(0.0, 0.0, s_local)[2]
    return s_eval, bz


base = xt.BorisSolenoid(
    L_coil=L_COIL,
    a=A_COIL,
    B0=B0,
    z0=Z0,
    length=LENGTH,
    n_steps=N_STEPS,
    taper=True,
)

line_tapered = base.to_tapered_line()
left, core, right = line_tapered.elements
print(f"Auto taper lengths [m]: left={left.length:.6f}, right={right.length:.6f}")
print(f"Core length [m]: {core.length:.6f}")
print(f"Total length [m]: {line_tapered.get_length():.6f}")

s_tapered, bz_tapered = sample_line_on_axis(line_tapered)
s_raw = np.linspace(0.0, LENGTH, 1201)
bz_raw = base.get_field(0.0, 0.0, s_raw)[2]

fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
ax.plot(s_raw, bz_raw, label="Raw BorisSolenoid on-axis Bz", alpha=0.7)
ax.plot(s_tapered, bz_tapered, "--", label="Tapered composite on-axis Bz", linewidth=2)
ax.axvline(left.length, color="k", ls=":", alpha=0.4, label="taper-core interfaces")
ax.axvline(LENGTH - right.length, color="k", ls=":", alpha=0.4)
ax.set_xlabel("s [m]")
ax.set_ylabel("Bz [T]")
ax.set_title("BorisSolenoid edge tapering")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.show()


for fac in [0.5, 1.0, 1.5]:
    line_scan = base.to_tapered_line(taper_factor_lambda=fac)
    ll = line_scan.elements[0].length
    rr = line_scan.elements[2].length
    print(f"taper_factor_lambda={fac:.2f} -> left={ll:.6f} m, right={rr:.6f} m")
