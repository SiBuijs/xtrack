from pathlib import Path

import matplotlib.pyplot as plt
import xtrack as xt
import xpart as xp
import numpy as np

plt.close("all")

HERE = Path(__file__).resolve().parent
# Same SplineBoris lattice as 010a_dynamic_aperture_splineboris.py
# (from 004b_install_solenoids_in_fcc_ring.py and
# 004c_correct_solenoids_in_fcc_ring.py).
INPUT_LATTICE_JSON = (
    HERE / "fccee_z_lcc_splineboris_solenoids_coupling_corrected.json"
)

IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]


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

line.configure_radiation(model='mean')
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
n_part = 1000000

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

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1, figsize=(6.4, 7))
ax21 = fig1.add_subplot(3,1,1)
ax22 = fig1.add_subplot(3,1,2)
ax23 = fig1.add_subplot(3,1,3)
ax21.plot(particles.x*1000, particles.px, '.', markersize=1)
ax21.set_xlabel(r'x [mm]')
ax21.set_ylabel(r'px [-]')
ax22.plot(particles.y*1000, particles.py, '.', markersize=1)
ax22.set_xlabel(r'y [mm]')
ax22.set_ylabel(r'py [-]')
ax23.plot(particles.zeta, particles.delta*1000, '.', markersize=1)
ax23.set_xlabel(r'z [-]')
ax23.set_ylabel(r'$\delta$ [1e-3]')
fig1.subplots_adjust(bottom=.08, top=.93, hspace=.33, left=.18,
                     right=.96, wspace=.33)
plt.show()