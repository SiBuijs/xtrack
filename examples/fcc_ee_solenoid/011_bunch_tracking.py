from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt

plt.close("all")

HERE = Path(__file__).resolve().parent
# Same SplineBoris lattice as 010a_dynamic_aperture_splineboris.py
# (from 004b_install_solenoids_in_fcc_ring.py and
# 004c_correct_solenoids_in_fcc_ring.py).
INPUT_LATTICE_JSON = (
    HERE / "fccee_z_lcc_splineboris_solenoids_coupling_corrected.json"
)

IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]
N_TURNS = 100
# Use the xtrack default (1 m), not the 5 cm limit from 009_momentum_acceptance.py:
# large beta functions amplify amplitudes well beyond 5 cm for sigma-levels ~ O(30).
GLOBAL_XY_LIMIT = 1.0


def _set_solenoid_knobs(line, *, with_solenoids, with_correctors):
    for ip_name in IP_NAMES:
        if f"on_sol_{ip_name}" in line.vars:
            line[f"on_sol_{ip_name}"] = float(with_solenoids)

        for corr_knob in (
            f"on_sol_corr_{ip_name}",
            f"on_comp_sol_{ip_name}",
            f"on_rot_doublet_left_{ip_name}",
            f"on_rot_doublet_right_{ip_name}",
            f"on_sol_orbit_corr_{ip_name}",
            f"on_sol_optics_corr_{ip_name}",
            f"on_sol_coupling_corr_{ip_name}",
        ):
            if corr_knob in line.vars:
                line[corr_knob] = float(with_correctors)


def _plot_bunch_phase_space(particles, title):
    fig = plt.figure(figsize=(6.4, 7))
    ax_x = fig.add_subplot(3, 1, 1)
    ax_y = fig.add_subplot(3, 1, 2)
    ax_z = fig.add_subplot(3, 1, 3)
    ax_x.plot(particles.x * 1000, particles.px, ".", markersize=1)
    ax_x.set_xlabel(r"x [mm]")
    ax_x.set_ylabel(r"px [-]")
    ax_y.plot(particles.y * 1000, particles.py, ".", markersize=1)
    ax_y.set_xlabel(r"y [mm]")
    ax_y.set_ylabel(r"py [-]")
    ax_z.plot(particles.zeta, particles.delta * 1000, ".", markersize=1)
    ax_z.set_xlabel(r"z [-]")
    ax_z.set_ylabel(r"$\delta$ [1e-3]")
    fig.suptitle(title)
    fig.subplots_adjust(bottom=0.08, top=0.93, hspace=0.33, left=0.18, right=0.96)
    return fig


print(f"\n=== SplineBoris: solenoids powered + correction scheme ===")
print(f"Loading lattice: {INPUT_LATTICE_JSON.name}")
env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring
line.cycle("ipa")
_set_solenoid_knobs(
    line,
    with_solenoids=True,
    with_correctors=True,
)

line.particle_ref.anomalous_magnetic_moment = 0.00115965218128
line.configure_radiation(model="mean")
line.compensate_radiation_energy_loss()

tw = line.twiss6d(radiation_analysis=True, strengths=True)
particle_on_co = tw.particle_on_co

delta0 = tw.delta[0]    # 9.401588655166826e-05
zeta0 = tw.zeta[0]      # -1.442824080284778e-06

print(
    f"Tune: Qx={tw['qx']:.6f}, Qy={tw['qy']:.6f}, "
    f"Qs={tw['qs']:.6f}"
)
print(
    f"Chromaticity: dQx/ddelta={tw['dqx']:.4f}, "
    f"dQy/ddelta={tw['dqy']:.4f}"
)
print(
    f"Beta at start: betx={tw['betx'][0]:.3f} m, "
    f"bety={tw['bety'][0]:.3f} m"
)
print(
    f"Dispersion at start: dx={tw['dx'][0]*1e3:.3f} mm, "
    f"dy={tw['dy'][0]*1e3:.3f} mm"
)

bunch_intensity = 1e11
n_part = 100

nemitt_x = 6.33e-5
nemitt_y = 1.69e-7

# beta_x = tw["betx"][0]
# beta_y = tw["bety"][0]
# beta0, gamma0 = float(tw["beta0"]), float(tw["gamma0"])
# geom_emitt_x = nemitt_x / (beta0 * gamma0)
# geom_emitt_y = nemitt_y / (beta0 * gamma0)
# sigma_x =  np.sqrt(beta_x * geom_emitt_x)
# sigma_y =  np.sqrt(beta_y * geom_emitt_y)

# Values from the Feasibility Study Vol 2.
sigma_x = 9e-6 
sigma_y = 40e-9 
sigma_z = 15.2e-3   # Including beamstrahlung. Should be 5.15 without.

particles = xp.generate_matched_gaussian_bunch(
         num_particles=n_part,
         total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x,
         nemitt_y=nemitt_y,
         sigma_z=sigma_z,
         line=line,
         particle_on_co=particle_on_co,
         engine='single-rf-harmonic')

# print(f"particles.x.mean():     {particles.x.mean()}")
# print(f"tw.x[0]:                {tw.x[0]}")
# print(f"particles.y.mean():     {particles.y.mean()}")
# print(f"tw.y[0]:                {tw.y[0]}")
# print(f"particles.px.mean():    {particles.px.mean()}")
# print(f"tw.px[0]:               {tw.px[0]}")
# print(f"particles.py.mean():    {particles.py.mean()}")
# print(f"tw.py[0]:               {tw.py[0]}")
# print(f"particles.delta.mean(): {particles.delta.mean()}")
# print(f"tw.delta[0]:            {tw.delta[0]}")
# print(f"particles.zeta.mean():  {particles.zeta.mean()}")
# print(f"tw.zeta[0]:             {tw.zeta[0]}")

# print("Shift zeta and delta to the closed orbit:")
particles.zeta  += zeta0
particles.delta += delta0
# print(f"particles.zeta.mean():  {particles.zeta.mean()}")
# print(f"particles.delta.mean(): {particles.delta.mean()}")

x_off = 0
px_off = 0
y_off = 0
py_off = 0
delta_off = 0
zeta_off = 0

particles.x += x_off
particles.px += px_off
particles.y += y_off
particles.py += py_off
particles.delta += delta_off
particles.zeta += zeta_off

particles_init = particles.copy()
_plot_bunch_phase_space(particles_init, "Initial bunch")

line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads="auto"))

print(f"Tracking {n_part} particles for {N_TURNS} turns")
line.config.XTRACK_GLOBAL_XY_LIMIT = GLOBAL_XY_LIMIT
line.track(particles, num_turns=N_TURNS, with_progress=1)
particles.sort(interleave_lost_particles=True)

lost = particles.state <= 0
frac_lost = lost.sum() / len(lost)
at_turn_mean = particles.at_turn.mean()
print(f"frac_lost={frac_lost:.6g}, at_turn_mean={at_turn_mean:.6g}")

_plot_bunch_phase_space(particles, f"After {N_TURNS} turns")
plt.show()

