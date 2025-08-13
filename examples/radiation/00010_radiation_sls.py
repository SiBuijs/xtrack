# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import time
import numpy as np

from cpymad.madx import Madx

import xtrack as xt

# Import a thick sequence
env = xt.load_madx_lattice('../../test_data/sls_2.0/b075_2024.09.25.madx')
line = env.ring
line.particle_ref = xt.Particles(energy0=2.7e9, mass0=xt.ELECTRON_MASS_EV)
line.configure_bend_model(num_multipole_kicks=20)

line['vrf'] = 1.8e6
line['frf'] = 499.6e6
line['lagrf'] = 180.

line.insert(
    env.new('cav', 'Cavity', voltage='vrf', frequency='frf', lag='lagrf', at=0))

tt = line.get_table()
tw4d_thick = line.twiss4d()
tw6d_thick = line.twiss()

env['ring_thick'] = env.ring.copy(shallow=True)

line.discard_tracker()
slicing_strategies = [
    xt.Strategy(slicing=None),  # Default
    xt.Strategy(slicing=xt.Teapot(20), element_type=xt.Bend),
    xt.Strategy(slicing=xt.Teapot(8), element_type=xt.Quadrupole),
]
line.slice_thick_elements(slicing_strategies)

print('Build tracker ...')
line.build_tracker()

################################
# Enable synchrotron radiation #
################################

# we choose the `mean` mode in which the mean power loss is applied without
# stochastic fluctuations (quantum excitation).
line.configure_radiation(model='mean')

#########
# Twiss #
#########

tw = line.twiss(eneloss_and_damping=True)

# By setting `eneloss_and_damping=True` we can get additional information
# from the twiss for example:
#  - tw['eneloss_turn'] provides the energy loss per turn (in eV).
#  - tw['damping_constants_s'] provides the damping constants in x, y and zeta.
#  - tw['partition_numbers'] provides the corresponding damping partion numbers.
#  - tw['eq_nemitt_x'] provides the equilibrium horizontal emittance.
#  - tw['eq_nemitt_y'] provides the equilibrium vertical emittance.
#  - tw['eq_nemitt_zeta'] provides the equilibrium longitudinal emittance.

############################################
# Generate particles and track (mean mode) #
############################################

# Build three particles (with action in x,y and zeta respectively)
part_co = tw['particle_on_co']
particles = line.build_particles(
    x_norm=[500., 0, 0], y_norm=[0, 500, 0], zeta=part_co.zeta[0],
    delta=np.array([0,0,1e-2]) + part_co.delta[0],
    nemitt_x=1e-9, nemitt_y=1e-9)

# Save initial state
particles_0 = particles.copy()

# Track
num_turns = 20000
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)

# Save monitor
mon_mean_mode = line.record_last_track

############################
# Switch to `quantum` mode #
############################

# We switch to the `quantum` mode in which the power loss from radiation is
# applied including stochastic fluctuations (quantum excitation).
# IMPORTANT: Note that this mode should not be used to compute twiss parameters
#            nor to match particle distributions. For this reason we switch
#            to quantum mode only after having generated the particles.


line.configure_radiation(model='quantum')

# We reuse the initial state saved before
particles = particles_0.copy()

line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
mon_quantum_mode = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
figs = []
for ii, mon in enumerate([mon_mean_mode, mon_quantum_mode]):
    fig = plt.figure(ii + 1)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)

    ax1.plot(1e3*mon.x[0, :].T)
    ax2.plot(1e3*mon.y[1, :].T)
    ax3.plot(1e3*mon.delta[2, :].T)
    i_turn = np.arange(num_turns)
    ax1.plot(1e3*(part_co.x[0]
        +(mon.x[0,0]-part_co.x[0])*np.exp(-i_turn*tw['damping_constants_turns'][0])))
    ax2.plot(1e3*(part_co.y[0]
        +(mon.y[1,0]-part_co.y[0])*np.exp(-i_turn*tw['damping_constants_turns'][1])))
    ax3.plot(1e3*(part_co.delta[0]
        +(mon.delta[2,0]-part_co.delta[0])*np.exp(-i_turn*tw['damping_constants_turns'][2])))

    ax1.set_ylabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    ax3.set_ylabel('delta [-]')
    ax3.set_xlabel('Turn')

plt.show()
# Twiss at the reference location (element 0)
alfx = float(tw.alfx[0]); betx = float(tw.betx[0]); gamx = (1+alfx**2)/betx
alfy = float(tw.alfy[0]); bety = float(tw.bety[0]); gamy = (1+alfy**2)/bety

# Pull turn-by-turn data from the quantum-mode run
nt = mon_quantum_mode.x.shape[1]
ix, iy = 0, 1  # indices of the particles with x- and y-action
x_t  = mon_quantum_mode.x[ix, :].astype(float)
px_t = mon_quantum_mode.px[ix, :].astype(float)
y_t  = mon_quantum_mode.y[iy, :].astype(float)
py_t = mon_quantum_mode.py[iy, :].astype(float)
# Energy readout (use the “longitudinal” particle or average)
E0 = float(line.particle_ref.energy0)
delta_t = mon_quantum_mode.delta[2, :].astype(float)
E_t = E0 * (1.0 + delta_t)

# Courant–Snyder invariants (single particle each plane)
emx_t = gamx*x_t**2 + 2*alfx*x_t*px_t + betx*px_t**2
emy_t = gamy*y_t**2 + 2*alfy*y_t*py_t + bety*py_t**2
emx_t = np.clip(emx_t, 0, None)
emy_t = np.clip(emy_t, 0, None)

# Precompute a normalized ellipse we will scale each frame
phi = np.linspace(0, 2*np.pi, 400)
x_norm  = np.cos(phi)
xp_norm = -(alfx*np.cos(phi) + np.sin(phi))    # this is p_x normalized by sqrt(eps/betx)

fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.35, wspace=0.35)
axx = fig.add_subplot(gs[0, 0]); axy = fig.add_subplot(gs[0, 1]); axe = fig.add_subplot(gs[1, :])

for ax in (axx, axy):
    ax.axhline(0, lw=0.5, alpha=0.3); ax.axvline(0, lw=0.5, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

axx.set_xlabel("x [m]");    axx.set_ylabel(r"$p_x$ [rad]")
axy.set_xlabel("y [m]");    axy.set_ylabel(r"$p_y$ [rad]")
axe.set_xlabel("turn");     axe.set_ylabel(r"emittance $\varepsilon$ [m·rad]")
fig.suptitle("Twiss Ellipses and Emittance vs Turn (Xtrack quantum mode)")

# Initial limits from first-turn invariants
pad = 1.4
def plane_limits(eps, alpha, beta):
    xr = np.sqrt(eps*beta)
    pr = np.sqrt(eps*(1+alpha**2)/beta)
    return (-pad*xr, pad*xr, -pad*pr, pad*pr)

xlo,xhi,pxlo,pxhi = plane_limits(max(emx_t[0], 1e-16), alfx, betx)
ylo,yhi,pylo,pyhi = plane_limits(max(emy_t[0], 1e-16), alfy, bety)
axx.set_xlim(xlo,xhi); axx.set_ylim(pxlo,pxhi)
axy.set_xlim(ylo,yhi); axy.set_ylim(pylo,pyhi)

# Artists
line_x, = axx.plot([], [], lw=2)
line_y, = axy.plot([], [], lw=2)
txt = axx.text(0.02, 0.02, "", transform=axx.transAxes,
               ha="left", va="bottom", fontsize=10,
               bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.8", alpha=0.8))

turns = np.arange(nt)
emx_hist = np.full(nt, np.nan); emy_hist = np.full(nt, np.nan)
line_emx, = axe.plot([], [], lw=1.8, label=r'$\varepsilon_x$')
line_emy, = axe.plot([], [], lw=1.8, label=r'$\varepsilon_y$')
pt_emx,   = axe.plot([], [], 'o', ms=4)
pt_emy,   = axe.plot([], [], 'o', ms=4)
axe.legend(loc="upper right")
axe.set_xlim(0, nt)
y0 = max(emx_t[0], emy_t[0], 1e-16)
axe.set_ylim(0, 1.2*y0)

def init():
    line_x.set_data([], []); line_y.set_data([], [])
    line_emx.set_data([], []); line_emy.set_data([], [])
    pt_emx.set_data([], []); pt_emy.set_data([], [])
    txt.set_text("")
    return line_x, line_y, line_emx, line_emy, pt_emx, pt_emy, txt

def update(i):
    # Ellipses for current invariants
    ex = max(emx_t[i], 0.0); ey = max(emy_t[i], 0.0)
    x_  = np.sqrt(ex*betx) * x_norm
    px_ = np.sqrt(ex/betx) * xp_norm
    y_  = np.sqrt(ey*bety) * x_norm
    py_ = np.sqrt(ey/bety) * (-(alfy*np.cos(phi) + np.sin(phi)))

    line_x.set_data(x_, px_)
    line_y.set_data(y_, py_)

    # History curves
    emx_hist[i] = ex; emy_hist[i] = ey
    t = turns[:i+1]
    line_emx.set_data(t, emx_hist[:i+1])
    line_emy.set_data(t, emy_hist[:i+1])
    pt_emx.set_data([i], [ex]); pt_emy.set_data([i], [ey])

    # Autoscale if needed
    y_now = np.nanmax([emx_hist[:i+1].max(), emy_hist[:i+1].max()])
    ylo, yhi = axe.get_ylim()
    if y_now > 0.9*yhi:
        axe.set_ylim(0, 1.2*y_now)

    txt.set_text(
        f"turn: {i:5d}\n"
        f"E: {E_t[i]: .3e} eV\n"
        f"εx: {ex:.3e}   εy: {ey:.3e}"
    )
    return line_x, line_y, line_emx, line_emy, pt_emx, pt_emy, txt

anim = FuncAnimation(fig, update, frames=nt, init_func=init, blit=True, interval=30)
plt.show()

# Animations
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.35, wspace=0.35)
axx = fig.add_subplot(gs[0, 0])        # x-x'
axy = fig.add_subplot(gs[0, 1])        # y-y'
axe = fig.add_subplot(gs[1, :])        # emittance time-series (spans bottom row)

for ax in (axx, axy):
    ax.axhline(0, lw=0.5, alpha=0.3)
    ax.axvline(0, lw=0.5, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

axx.set_xlabel("x");    axx.set_ylabel("x'")
axy.set_xlabel("y");    axy.set_ylabel("y'")
axe.set_xlabel("turn"); axe.set_ylabel("emittance [mrad]")

fig.suptitle("Twiss Ellipses and Emittance vs Turns")

# Axis limits for phase space from initial sizes
emx0 = gamx*state[0]**2 + 2*alfx*state[0]*state[1] + betx*state[1]**2 + 1e-16
emy0 = gamy*state[2]**2 + 2*alfy*state[2]*state[3] + bety*state[3]**2 + 1e-16
pad = 1.4
axx.set_xlim(-pad*np.sqrt(emx0*betx), pad*np.sqrt(emx0*betx))
axx.set_ylim(-pad*np.sqrt(emx0*betx), pad*np.sqrt(emx0*betx))
axy.set_xlim(-pad*np.sqrt(emy0*bety), pad*np.sqrt(emy0*bety))
axy.set_ylim(-pad*np.sqrt(emy0*(1+alfy**2)/bety), pad*np.sqrt(emy0*(1+alfy**2)/bety))

# Artists for phase-space ellipses
line_x, = axx.plot([], [], lw=2)
line_y, = axy.plot([], [], lw=2)

# Status text (attach to an Axes so blitting is happy)
txt = axx.text(0.02, 0.02, "", transform=axx.transAxes,
               ha="left", va="bottom", fontsize=10,
               bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.8", alpha=0.8))

# --- Emittance history + artists ---
turns = np.arange(N_FODO)
emx_hist = np.full(N_FODO, np.nan)
emy_hist = np.full(N_FODO, np.nan)

line_emx, = axe.plot([], [], lw=1.8, label=r'$\varepsilon_x$')
line_emy, = axe.plot([], [], lw=1.8, label=r'$\varepsilon_y$')
pt_emx,   = axe.plot([], [], 'o', ms=4)
pt_emy,   = axe.plot([], [], 'o', ms=4)
axe.legend(loc="upper right")
axe.set_xlim(0, N_FODO)
y0 = max(emx0, emy0)
axe.set_ylim(0, 1.2*y0 if y0 > 0 else 1.0)

def init():
    line_x.set_data([], []); line_y.set_data([], [])
    line_emx.set_data([], []); line_emy.set_data([], [])
    pt_emx.set_data([], []); pt_emy.set_data([], [])
    txt.set_text("")
    return line_x, line_y, line_emx, line_emy, pt_emx, pt_emy, txt

def update(frame):
    global state, Energy, T_FODO, E_avg, n_dot

    # --- one simulation step (same as before) ---
    state, Energy, T_FODO, E_avg, n_dot, emx, emy, nphot = step(
        frame, state, Energy, T_FODO, E_avg, n_dot
    )

    # phase-space ellipses
    x, xp = twiss_ellipse(alfx, betx, emx)
    y, yp = twiss_ellipse(alfy, bety, emy)
    line_x.set_data(x, xp)
    line_y.set_data(y, yp)

    # emittance history
    emx_hist[frame] = emx
    emy_hist[frame] = emy
    t = turns[:frame+1]
    line_emx.set_data(t, emx_hist[:frame+1])
    line_emy.set_data(t, emy_hist[:frame+1])
    pt_emx.set_data([frame], [emx])
    pt_emy.set_data([frame], [emy])

    # autoscale y if we approach the top
    y_now = np.nanmax([emx_hist[:frame+1].max(), emy_hist[:frame+1].max()])
    ylo, yhi = axe.get_ylim()
    if y_now > 0.9*yhi:
        axe.set_ylim(0, 1.2*y_now)

    # status text
    txt.set_text(
        f"turn: {frame:5d}\n"
        f"E: {Energy: .3e} eV\n"
        f"εx: {emx:.3e}   εy: {emy:.3e}\n"
        f"photons/turn: {nphot}"
    )

    return line_x, line_y, line_emx, line_emy, pt_emx, pt_emy, txt

anim = FuncAnimation(fig, update, frames=N_FODO, init_func=init,
                     blit=True, interval=30)
plt.show()