"""
HelicalBoris vs SplineBoris (Strang-split Boris) on the same field map.

Both kernels use the same ``SplineBoris`` element geometry, field splines,
``evaluate_B`` path, and radiation/spin hooks; only ``use_helical_boris``
selects the per-step integrator. With ``use_helical_boris=True`` the C kernel
uses field-aligned exact constant-``B`` substeps (Newton in arc length plus
Rodrigues); with ``False`` it uses the Strang-split Boris integrator. This
script compares end-of-track coordinates and reports simple wall-clock timings.

Run from the xtrack repo root (or any environment where ``xtrack`` is importable):

    python examples/splineboris/008_helicalboris_vs_splineboris.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

import xtrack as xt
from xtrack.beam_elements.splineboris import Spline4

plt.rcParams.update({"font.size": 12})

# ---------------------------------------------------------------------------
# Homogeneous field in a short drift-like interval (same idea as 002)
# ---------------------------------------------------------------------------
length = 1.0  # [m]
n_steps = 500

B_x = 0.02  # [T]
B_y = 0.10  # [T]
B_s = 0.05  # [T]  longitudinal component exercises the helical rotation block

bs = Spline4(
    val_start=B_s,
    der_start=0.0,
    val_end=B_s,
    der_end=0.0,
    integral=B_s * length,
)
bx_spline = Spline4(
    val_start=B_x,
    der_start=0.0,
    val_end=B_x,
    der_end=0.0,
    integral=B_x * length,
)
by_spline = Spline4(
    val_start=B_y,
    der_start=0.0,
    val_end=B_y,
    der_end=0.0,
    integral=B_y * length,
)

common = dict(
    bs=bs,
    bx=(bx_spline,),
    by=(by_spline,),
    length=length,
    n_steps=n_steps,
)

elem_boris = xt.SplineBoris(**common, use_helical_boris=False)
elem_helic = xt.SplineBoris(**common, use_helical_boris=True)

line_b = xt.Line(elements=[elem_boris])
line_h = xt.Line(elements=[elem_helic])
line_b.build_tracker()
line_h.build_tracker()

p0 = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV,
    q0=1.0,
    energy0=1e9,
    x=1e-3,
    y=-0.5e-3,
    px=2e-4,
    py=-1e-4,
)

n_warm = 3
n_bench = 20

for _ in range(n_warm):
    line_b.track(p0.copy())

t0 = time.perf_counter()
for _ in range(n_bench):
    line_b.track(p0.copy())
t_boris = (time.perf_counter() - t0) / n_bench

t0 = time.perf_counter()
for _ in range(n_bench):
    line_h.track(p0.copy())
t_helic = (time.perf_counter() - t0) / n_bench

p_b = p0.copy()
p_h = p0.copy()
line_b.track(p_b)
line_h.track(p_h)

print("HelicalBoris vs SplineBoris (same field, same n_steps)")
print(f"  n_steps = {n_steps}, length = {length} m")
print(f"  mean wall time ({n_bench} runs): Boris {t_boris*1e3:.3f} ms, "
      f"Helical {t_helic*1e3:.3f} ms  (ratio helical/boris = {t_helic/t_boris:.3f})")
print(f"  end x:  boris {p_b.x}, helical {p_h.x},  dx = {p_h.x - p_b.x}")
print(f"  end y:  boris {p_b.y}, helical {p_h.y},  dy = {p_h.y - p_b.y}")
print(f"  end px: boris {p_b.px}, helical {p_h.px}, dpx = {p_h.px - p_b.px}")
print(f"  end py: boris {p_b.py}, helical {p_h.py}, dpy = {p_h.py - p_b.py}")

# ---------------------------------------------------------------------------
# Optional: trajectories with element monitor (single particle)
# ---------------------------------------------------------------------------
line_b.track(p0.copy(), turn_by_turn_monitor="ONE_TURN_EBE")
mon_b = line_b.record_last_track
line_h.track(p0.copy(), turn_by_turn_monitor="ONE_TURN_EBE")
mon_h = line_h.record_last_track

s_b = mon_b.s[0, :]
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
axes[0, 0].plot(s_b, mon_b.x[0, :] * 1e3, "-", label="SplineBoris", lw=2)
axes[0, 0].plot(s_b, mon_h.x[0, :] * 1e3, "--", label="HelicalBoris", lw=2)
axes[0, 0].set_ylabel("x [mm]")
axes[0, 0].set_title("x vs s")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(s_b, mon_b.y[0, :] * 1e3, "-", label="SplineBoris", lw=2)
axes[0, 1].plot(s_b, mon_h.y[0, :] * 1e3, "--", label="HelicalBoris", lw=2)
axes[0, 1].set_ylabel("y [mm]")
axes[0, 1].set_title("y vs s")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(s_b, mon_b.px[0, :], "-", label="SplineBoris", lw=2)
axes[1, 0].plot(s_b, mon_h.px[0, :], "--", label="HelicalBoris", lw=2)
axes[1, 0].set_xlabel("s [m]")
axes[1, 0].set_ylabel(r"$p_x$ / $p_0$")
axes[1, 0].set_title(r"$p_x$ vs s")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(s_b, mon_b.py[0, :], "-", label="SplineBoris", lw=2)
axes[1, 1].plot(s_b, mon_h.py[0, :], "--", label="HelicalBoris", lw=2)
axes[1, 1].set_xlabel("s [m]")
axes[1, 1].set_ylabel(r"$p_y$ / $p_0$")
axes[1, 1].set_title(r"$p_y$ vs s")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(
    "008: same SplineBoris element, "
    r"$B_x,B_y,B_s$ constant; "
    "SplineBoris vs field-aligned HelicalBoris"
)
plt.show()
