import pathlib

import numpy as np
from cpymad.madx import Madx
import xtrack as xt

HERE = pathlib.Path(__file__).resolve().parent

# Import a line
env = xt.load(str(HERE / '../lattices/z/fccee_z.json'))
line = env.fccee_p_ring

tw0 = line.twiss4d()

# User-defined energy ramp
t_s = np.array([0., 0.0006, 0.0008, 0.001 , 0.0012, 0.0014, 0.0016, 0.0018,
                0.002 , 0.0022, 0.0024, 0.0026, 0.0028, 0.003, 0.01, 0.1])

E_kin_GeV = np.linspace(4.560, 4.473, 16) * 1e10

# Attach energy program to the line
line.energy_program = xt.EnergyProgram(t_s=t_s, kinetic_energy0=E_kin_GeV)

# t_s and kinetic_energy0 must have the same length; tracking must not exceed
# the last turn implied by the program (here ~330 turns for FCC-ee, not 9000).
n_turn_max = int(np.floor(line.energy_program.t_at_turn_interpolator.x[-1]))

# Plot energy and revolution frequency vs time
t_plot = np.linspace(0, 10e-3, 20)
E_kin_plot = line.energy_program.get_kinetic_energy0_at_t_s(t_plot)
f_rev_plot = line.energy_program.get_frev_at_t_s(t_plot)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(6.4 * 1.5, 4.8))
ax1 = plt.subplot(2,2,1)
plt.plot(t_plot * 1e3, E_kin_plot * 1e-6)
plt.ylabel(r'$E_{kin}$ [MeV]')
ax2 = plt.subplot(2,2,3, sharex=ax1)
plt.plot(t_plot * 1e3, f_rev_plot * 1e-3)
plt.ylabel(r'$f_{rev}$ [kHz]')
plt.xlabel('t [ms]')

# Keep rf400 on the harmonic of the revolution frequency during acceleration.
# Voltage and lag are already set in the lattice (rf_v_400, rf_lag_400).
h_rf = env["rf_harmon_400"]
line['rf400'].frequency = 0
line['rf400'].harmonic = h_rf

# When setting line['t_turn_s'] the reference energy is updated automatically;
# at tracking time the RF frequency is h_rf / T_rev.
line['t_turn_s'] = 0
line.particle_ref.kinetic_energy0[0]  # eV at start of ramp
h_rf * line.energy_program.get_frev_at_t_s(0)  # Hz

line['t_turn_s'] = 3e-3
line.particle_ref.kinetic_energy0[0]
h_rf * line.energy_program.get_frev_at_t_s(3e-3)

# Back to zero for tracking!
line['t_turn_s'] = 0

# Track a few particles to visualize the longitudinal phase space
p_test = line.build_particles(x_norm=0, zeta=np.linspace(0, line.get_length(), 101))

# Enable time-dependent variables (t_turn_s and all the variables that depend on
# it are automatically updated at each turn)
line.enable_time_dependent_vars = True

# Track
line.track(p_test, num_turns=n_turn_max, turn_by_turn_monitor=True, with_progress=True)
mon = line.record_last_track

# Plot
n_plot = min(200, n_turn_max)
plt.subplot2grid((2,2), (0,1), rowspan=2)
plt.plot(mon.zeta[:, -n_plot:].T, mon.delta[:, -n_plot:].T, color='C0')
plt.xlabel(r'$\zeta$ [m]')
plt.ylabel('$\delta$')
plt.xlim(-40, 30)
plt.ylim(-0.0025, 0.0025)
plt.title(f'Last {n_plot} turns')
plt.subplots_adjust(left=0.08, right=0.95, wspace=0.26)


# Check transverse beam size reduction
line['t_turn_s'] = 0
line.enable_time_dependent_vars = False

n_part_test = 500
# Generate Gaussian distribution with fixed rng seed
rng = np.random.default_rng(seed=123)
x_norm = rng.normal(loc=0, scale=1, size=n_part_test)
px_norm = rng.normal(loc=0, scale=1, size=n_part_test)
y_norm = rng.normal(loc=0, scale=1, size=n_part_test)
py_norm = rng.normal(loc=0, scale=1, size=n_part_test)

# rescale to have exact std dev.
x_norm = x_norm / np.std(x_norm)
px_norm = px_norm / np.std(px_norm)
y_norm = y_norm / np.std(y_norm)
py_norm = py_norm / np.std(py_norm)

p_test2 = line.build_particles(x_norm=x_norm, px_norm=px_norm,
                               y_norm=x_norm, py_norm=px_norm,
                               nemitt_x=3e-6, nemitt_y=3e-6,
                               delta=0)

line.enable_time_dependent_vars = True
line.track(p_test2, num_turns=n_turn_max, turn_by_turn_monitor=True, with_progress=True)
mon2 = line.record_last_track

std_y = np.std(mon2.y, axis=0)
std_x = np.std(mon2.x, axis=0)

# Apply moving average filter
from scipy.signal import savgol_filter
savgol_window = min(n_turn_max - 1, 101)
if savgol_window % 2 == 0:
    savgol_window -= 1
std_y_smooth = savgol_filter(std_y, savgol_window, 2)
std_x_smooth = savgol_filter(std_x, savgol_window, 2)

i_turn_match = n_turn_max // 5
std_y_expected = std_y_smooth[i_turn_match] * np.sqrt(
    mon2.gamma0[0, i_turn_match]* mon2.beta0[0, i_turn_match]
    / mon2.gamma0[0, :] / mon2.beta0[0, :])
std_x_expected = std_x_smooth[i_turn_match] * np.sqrt(
    mon2.gamma0[0, i_turn_match]* mon2.beta0[0, i_turn_match]
    / mon2.gamma0[0, :] / mon2.beta0[0, :])

d_sigma_x = std_x_expected[0] - std_x_expected[-1]
d_sigma_y = std_y_expected[0] - std_y_expected[-1]

import xobjects as xo
i_check_lo = max(i_turn_match, int(0.8 * n_turn_max))
i_check_hi = n_turn_max
xo.assert_allclose(std_y_expected[i_check_lo:i_check_hi].mean(),
                   std_y_smooth[i_check_lo:i_check_hi].mean(),
                   rtol=0, atol=0.07 * d_sigma_y)
xo.assert_allclose(std_x_expected[i_check_lo:i_check_hi].mean(),
                   std_x_smooth[i_check_lo:i_check_hi].mean(),
                   rtol=0, atol=0.07 * d_sigma_x)

plt.figure(2)
ax1 = plt.subplot(2,1,1)
plt.plot(std_x, label='raw')
plt.plot(std_x_smooth, label='smooth')
plt.plot(std_x_expected, label='expected')
plt.legend()

ax2 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(std_y, label='raw')
plt.plot(std_y_smooth, label='smooth')
plt.plot(std_y_expected, label='expected')


plt.show()
