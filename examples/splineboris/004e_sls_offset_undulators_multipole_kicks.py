"""
SLS simulation with offset undulators represented by Multipole kicks.

This script mirrors 004c_sls_offset_undulators_closed_spin.py, but builds the
undulator as drift + thin-Multipole kicks sampled from the fitted field map.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xtrack as xt
from xtrack._temp.field_fitter import FieldFitter

from _undulator_multipole_builder import build_multipole_kick_undulator

multipole_order = 3
enable_plots = True  # Set True for interactive plots
non_blocking_plots = False  # If True, keep process alive until Enter is pressed
plot_stride = 5  # Use >1 to downsample and keep plotting responsive

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)

# Load SLS MADX file
madx_file = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "sls.madx"
env = xt.load(str(madx_file))
line_offset = env.ring

# Configure bend model
line_offset.configure_bend_model(core="mat-kick-mat")

# Set particle reference
line_offset.particle_ref = p0.copy()

base_dir = Path(__file__).resolve().parent
field_map_path = base_dir.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
df_raw_data = pd.read_csv(
    field_map_path,
    sep="\t",
    header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])

field_fitter = FieldFitter(
    raw_data=df_raw_data,
    xy_point=(0, 0),
    distance_unit=0.001,
    min_region_size=5,
    deg=multipole_order - 1,
)
field_fitter.fit()

x_off = 5e-4
y_off = 0.0

piecewise_undulator, l_wig = build_multipole_kick_undulator(
    env=env,
    p_ref=p0,
    df_fit_pars=field_fitter.df_fit_pars,
    multipole_order=multipole_order,
    shift_x=x_off,
    shift_y=y_off,
    name_prefix="und_o",
)

# Create env variables for corrector strengths (needed for matching)
env["k0l_corr1"] = 0.0
env["k0l_corr2"] = 0.0
env["k0l_corr3"] = 0.0
env["k0l_corr4"] = 0.0
env["k0sl_corr1"] = 0.0
env["k0sl_corr2"] = 0.0
env["k0sl_corr3"] = 0.0
env["k0sl_corr4"] = 0.0

# Create corrector elements with expressions referencing env variables
env.new("corr1", xt.Multipole, knl=["k0l_corr1"], ksl=["k0sl_corr1"])
env.new("corr2", xt.Multipole, knl=["k0l_corr2"], ksl=["k0sl_corr2"])
env.new("corr3", xt.Multipole, knl=["k0l_corr3"], ksl=["k0sl_corr3"])
env.new("corr4", xt.Multipole, knl=["k0l_corr4"], ksl=["k0sl_corr4"])

# Insert correctors at nearest existing boundary (no slicing, robust for reuse)
target_positions = {
    "corr1": 0.02,
    "corr2": 0.1,
    "corr3": l_wig - 0.1,
    "corr4": l_wig - 0.02,
}
base_names = list(piecewise_undulator.element_names)
tt_und = piecewise_undulator.get_table()
boundary_positions = [float(tt_und["s", nn]) for nn in base_names]
boundary_positions.append(l_wig)

corrector_insertions = {}
for corr_name, s_target in target_positions.items():
    idx = min(range(len(boundary_positions)), key=lambda ii: abs(boundary_positions[ii] - s_target))
    corrector_insertions.setdefault(idx, []).append(corr_name)
    print(f"{corr_name}: requested s={s_target:.4f}, inserting at boundary s={boundary_positions[idx]:.4f}")

names_with_correctors = []
for ii, ee_name in enumerate(base_names):
    if ii in corrector_insertions:
        names_with_correctors.extend(corrector_insertions[ii])
    names_with_correctors.append(ee_name)
if len(base_names) in corrector_insertions:
    names_with_correctors.extend(corrector_insertions[len(base_names)])

piecewise_undulator = xt.Line(env=env, element_names=names_with_correctors)
piecewise_undulator.particle_ref = p0.copy()

opt = piecewise_undulator.match(
    solve=False,
    betx=0,
    bety=0,
    only_orbit=True,
    include_collective=True,
    vary=xt.VaryList(
        [
            "k0l_corr1",
            "k0sl_corr1",
            "k0l_corr2",
            "k0sl_corr2",
            "k0l_corr3",
            "k0sl_corr3",
            "k0l_corr4",
            "k0sl_corr4",
        ],
        step=1e-6,
    ),
    targets=[
        xt.TargetSet(x=0, px=0, y=0, py=0.0, at=xt.END),
        xt.TargetSet(x=0.0, y=0, at="corr2"),
        xt.TargetSet(x=0.0, y=0, at="corr3"),
    ],
)
opt.step(2)

piecewise_undulator.discard_tracker()

wiggler_places = [
    "ars02_uind_0500_1",
    "ars03_uind_0380_1",
    "ars04_uind_0500_1",
    "ars05_uind_0650_1",
    "ars06_uind_0500_1",
    "ars07_uind_0200_1",
    "ars08_uind_0500_1",
    "ars09_uind_0790_1",
    "ars11_uind_0210_1",
    "ars11_uind_0610_1",
    "ars12_uind_0500_1",
]

tt = line_offset.get_table()
for ii, wig_place in enumerate(wiggler_places):
    und_this = piecewise_undulator.replicate(suffix=f"ins{ii}")
    print(f"Inserting piecewise_undulator {wig_place} at {tt['s', wig_place]}")
    line_offset.insert(und_this, anchor="start", at=tt["s", wig_place])

line_offset.build_tracker()
print("Tracker built, running twiss4d...")
t0 = time.perf_counter()
tw_offset = line_offset.twiss4d(radiation_integrals=True)
print(f"twiss4d completed in {time.perf_counter() - t0:.2f} s")

if enable_plots:
    print("Generating plots...")
    plt.close("all")
    s = np.asarray(tw_offset.s)[::plot_stride]
    x = np.asarray(tw_offset.x)[::plot_stride]
    y = np.asarray(tw_offset.y)[::plot_stride]
    betx = np.asarray(tw_offset.betx)[::plot_stride]
    bety = np.asarray(tw_offset.bety)[::plot_stride]
    dx = np.asarray(tw_offset.dx)[::plot_stride]
    dy = np.asarray(tw_offset.dy)[::plot_stride]
    betx2 = np.asarray(tw_offset.betx2)[::plot_stride]
    bety2 = np.asarray(tw_offset.bety2)[::plot_stride]

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(s, x, label="x")
    ax1.plot(s, y, label="y")
    ax1.set_xlabel("s [m]")
    ax1.set_ylabel("Closed orbit [m]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax2a.plot(s, betx, label="betx")
    ax2a.plot(s, bety, label="bety")
    ax2a.set_ylabel("Beta functions [m]")
    ax2a.grid(True, alpha=0.3)
    ax2a.legend()
    ax2b.plot(s, dx, label="dx")
    ax2b.plot(s, dy, label="dy")
    ax2b.set_xlabel("s [m]")
    ax2b.set_ylabel("Dispersion [m]")
    ax2b.grid(True, alpha=0.3)
    ax2b.legend()
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(s, betx2, label="betx2")
    ax3.plot(s, bety2, label="bety2")
    ax3.set_xlabel("s [m]")
    ax3.set_ylabel("Mode-2 beta [m]")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()

    if non_blocking_plots:
        plt.show(block=False)
        input("Plots are open. Press Enter to close and exit...")
    else:
        # Blocking show keeps GUI event loop alive reliably.
        plt.show()

print("=" * 80)
print("SLS WITH OFFSET MULTIPOLE-KICK UNDULATORS")
print("=" * 80)
print("Tunes:")
print(f"  qx = {tw_offset.qx:.4e}")
print(f"  qy = {tw_offset.qy:.4e}")
print(f"  qs = {tw_offset.qs:.4e}")
print()
print("Chromaticity:")
print(f"  dqx = {tw_offset.dqx:.4e}")
print(f"  dqy = {tw_offset.dqy:.4e}")
print()
print("Partition numbers:")
print(f"  J_x = {tw_offset.rad_int_partition_number_x:.4e}")
print(f"  J_y = {tw_offset.rad_int_partition_number_y:.4e}")
print(f"  J_zeta = {tw_offset.rad_int_partition_number_zeta:.4e}")
print()
print("Damping constants per second:")
print(f"  alpha_x = {tw_offset.rad_int_damping_constant_x_s:.4e}")
print(f"  alpha_y = {tw_offset.rad_int_damping_constant_y_s:.4e}")
print(f"  alpha_zeta = {tw_offset.rad_int_damping_constant_zeta_s:.4e}")
print()
print("Equilibrium emittances:")
print(f"  eq_gemitt_x = {tw_offset.rad_int_eq_gemitt_x:.4e}")
print(f"  eq_gemitt_y = {tw_offset.rad_int_eq_gemitt_y:.4e}")
print()
print(f"C^-: {tw_offset.c_minus:.4e}")
print()
print("=" * 80)
