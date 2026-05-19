import pathlib

import numpy as np
import xpart as xp
import xtrack as xt
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
test_data_folder = (HERE / '../../../test_data').resolve()

# fccee_z.json is an Environment; trackers and twiss are built on a Line.
env = xt.load(str(HERE / '../lattices/z/fccee_z.json'))
line = env.fccee_p_ring
line.build_tracker()

E0 = line.particle_ref.energy0[0]  # eV
print(f'E0: {E0:.3e} eV')

# Base frequency: 400848492.6737981 Hz
# Highest frequency that gives a closed orbit: 218 Hz
# Corresponding delta: -1.908e-02
# Corresponding energy: 4.473e+10 eV
# Formidable ~1 cm offset

# Lowest frequency that gives a closed orbit: -148 Hz
# Corresponding delta: 1.286e-02
# Corresponding energy: 4.619e+10 eV
df_hz = -148  # Frequency trim

tw = line.twiss()

h_rf = env["rf_harmon_400"]
f_rf = h_rf / tw.t_rev0

beta0 = line.particle_ref.beta0[0]

# T_ref = h_rf/f_rf
# dt = h_rf/(f_rf + df_hz) - h_rf/f_rf = h_rf/f_rf (1/(1+df_hz/f_rf) - 1)
#                                       ~= h_rf/f_rf * (1 - df_hz/f_rf -1)
#                                       = -h_rf/(f_rf^2) * df_hz
#                                       = -t_rev / f_rf * df_hz
# dzeta = -beta0 * clight * dt = circumference * df_hz / f_rf

env['circumference'] = tw.line_length
env['f_rf'] = f_rf
env['df_hz'] = float(df_hz)
env['dzeta'] = 'circumference * df_hz / f_rf'

env.new('zeta_shift', xt.ZetaShift, dzeta='dzeta')
line.discard_tracker()
line.append('zeta_shift')
line.build_tracker()

tw_6d_offmom = line.twiss()

delta_closed_orbit = tw_6d_offmom.delta[0]
energy_closed_orbit = tw_6d_offmom.particle_on_co.energy[0]

print(f'delta closed orbit: {delta_closed_orbit:.3e}')
print(f'energy closed orbit: {energy_closed_orbit:.3e} eV')

# Checks
eta = tw.slip_factor
f0 = 1 / tw.t_rev0
delta_trim = -1 / h_rf / eta / f0 * df_hz

# Use 4d twiss on machine without zeta shift
env['df_hz'] = 0.
tw_on_mom = line.twiss(delta0=0, method='4d')
tw_off_mom = line.twiss(delta0=delta_trim, method='4d')
dzeta_from_twiss = (tw_off_mom['zeta'][-1] - tw_off_mom['zeta'][0])

env['df_hz'] = float(df_hz)

print(f'delta_trim: {delta_trim}')
print(f'tw_6d_offmom.delta[0]: {tw_6d_offmom.delta[0]}')
print(f'tw_6d_offmom.ptau[0]: {tw_6d_offmom.ptau[0]}')
print(f'tw_6d_offmom.q_s: {tw_6d_offmom.qs}')
print(f'tw_6d_offmom.f_s: {f_rf * tw_6d_offmom.qs:.3e} Hz')

plt.close('all')
plt.figure(1, figsize=(12, 5))
plt.subplot(1, 2, 1)
tw_on_mom.plot('x', 'dx')
plt.subplot(1, 2, 2)
tw_off_mom.plot('x', 'dx')

# --- Physical ramp of df_hz from -148 to +218 Hz over many turns ---

n_turns_ramp = 20_000  # reduce for a quick smoke test; t_end scales with this
df_hz_lo, df_hz_hi = -148.0, 218.0
t_rev = tw.t_rev0
t_end = n_turns_ramp * t_rev

delta_expected_lo = -1 / h_rf / eta / f0 * df_hz_lo
delta_expected_hi = -1 / h_rf / eta / f0 * df_hz_hi

# build_particles calls twiss internally; keep df_hz scalar until after
env['df_hz'] = df_hz_lo
p_test = line.build_particles(
    x_norm=0,
    delta=np.linspace(delta_expected_lo - 8e-4, delta_expected_lo + 8e-4, 21),
)

line.functions['df_ramp'] = xt.FunctionPieceWiseLinear(
    x=np.array([0.0, t_end]),
    y=np.array([df_hz_lo, df_hz_hi]),
)
env['df_hz'] = line.functions['df_ramp'](line.ref['t_turn_s'])

line['t_turn_s'] = 0
line.enable_time_dependent_vars = True

line.track(p_test, num_turns=n_turns_ramp,
           turn_by_turn_monitor=True, with_progress=True)
mon = line.record_last_track

turns = np.arange(n_turns_ramp)
t_turns = turns * t_rev
df_hz_turns = df_hz_lo + (df_hz_hi - df_hz_lo) * t_turns / t_end
delta_expected_turns = -df_hz_turns / (h_rf * eta * f0)

data_dir = HERE.parent / 'data'
data_dir.mkdir(exist_ok=True)
ctx2np = line._context.nparray_from_context_array
out_file = data_dir / '007_radial_steering_ramp.npz'
np.savez(
    out_file,
    delta=ctx2np(mon.delta),
    zeta=ctx2np(mon.zeta),
    turns=turns,
    delta_expected_turns=delta_expected_turns,
    df_hz_turns=df_hz_turns,
)
print(f'Saved tracking data to {out_file}')

plt.figure(2, figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(turns, mon.delta.T * 1e4, color='C0', alpha=0.3)
plt.plot(turns, delta_expected_turns * 1e4, 'r--', lw=2,
         label=r'$\delta$ expected (slip factor)')
plt.xlabel('Turn')
plt.ylabel(r'$\delta$ [$10^{-4}$]')
plt.legend()
plt.title('Momentum vs turn during df_hz ramp')

n_plot = min(500, n_turns_ramp)
plt.subplot(1, 2, 2)
plt.plot(mon.zeta[:, :n_plot].T, mon.delta[:, :n_plot].T * 1e4,
         color='C0', alpha=0.3, label=f'first {n_plot} turns')
plt.plot(mon.zeta[:, -n_plot:].T, mon.delta[:, -n_plot:].T * 1e4,
         color='C1', alpha=0.3, label=f'last {n_plot} turns')
plt.xlabel(r'$\zeta$ [m]')
plt.ylabel(r'$\delta$ [$10^{-4}$]')
plt.legend()
plt.title('Longitudinal phase space')

# Endpoint check at df_hz = +218 Hz
line.enable_time_dependent_vars = False
env['df_hz'] = df_hz_hi
line['t_turn_s'] = t_end
tw_hi = line.twiss()
print(f'delta at +218 Hz (twiss): {tw_hi.delta[0]:.3e}')
print(f'delta at +218 Hz (track, mean last turn): {mon.delta[:, -1].mean():.3e}')
print(f'delta_expected_hi: {delta_expected_hi:.3e}')

plt.show()
