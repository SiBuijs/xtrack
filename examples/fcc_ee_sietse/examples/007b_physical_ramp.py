"""Time-dependent df_hz ramp (-148 to +218 Hz). Static CO / slip-factor checks: 007a_radial_steering_static.py."""

import pathlib
import sys

import numpy as np
import xtrack as xt
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
SIETSE = HERE.parent
sys.path.insert(0, str(SIETSE))
from dx_manual import (  # noqa: E402
    IPS,
    measure_dx_manual,
    plot_dx_scan,
    print_dx_comparison,
)

# fccee_z.json is an Environment; trackers and twiss are built on a Line.
env = xt.load(str(HERE / '../lattices/z/fccee_z.json'))
line = env.fccee_p_ring
line.build_tracker()

# --- ZetaShift (dzeta from df_hz); required for the ramp ---
tw = line.twiss()

h_rf = env["rf_harmon_400"]
f_rf = h_rf / tw.t_rev0
eta = tw.slip_factor
f0 = 1 / tw.t_rev0

df_hz_lo, df_hz_hi = -148.0, 218.0

env['circumference'] = tw.line_length
env['f_rf'] = f_rf
env['df_hz'] = df_hz_lo  # required before symbolic dzeta is evaluated
env['dzeta'] = 'circumference * df_hz / f_rf'

env.new('zeta_shift', xt.ZetaShift, dzeta='dzeta')
line.discard_tracker()
line.append('zeta_shift')
line.build_tracker()

# --- Physical ramp of df_hz from -148 to +218 Hz over many turns ---

n_turns_ramp = 1000  # reduce for a quick smoke test; t_end scales with this
t_rev = tw.t_rev0
t_end = n_turns_ramp * t_rev

delta_expected_lo = -1 / h_rf / eta / f0 * df_hz_lo
# delta_expected_hi = -1 / h_rf / eta / f0 * df_hz_hi

# build_particles calls twiss internally; keep df_hz at df_hz_lo until ramp is set
p_test = line.build_particles(
    x_norm=0,
    delta=np.linspace(delta_expected_lo - 4e-3, delta_expected_lo + 4e-3, 21),
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

delta_mon = ctx2np(mon.delta)
delta_init = delta_mon[:, 0]
cmap = plt.cm.viridis
norm = plt.Normalize(delta_init.min(), delta_init.max())
colors = cmap(norm(delta_init))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for i in range(delta_mon.shape[0]):
    ax1.plot(turns, delta_mon[i] * 1e4, color=colors[i], alpha=0.8)
    ax2.plot(df_hz_turns, delta_mon[i] * 1e4, color=colors[i], alpha=0.8)
ax1.plot(turns, delta_expected_turns * 1e4, 'r--', lw=2,
         label=r'$\delta$ expected (slip factor)')
ax1.set_xlabel('Turn')
ax1.set_ylabel(r'$\delta$ [$10^{-4}$]')
ax1.legend()
ax1.set_title('Momentum vs turn during df_hz ramp')
ax2.set_xlabel(r'$\Delta f$ [Hz]')
ax2.set_ylabel(r'$\delta$ [$10^{-4}$]')
ax2.set_title(r'Momentum vs RF frequency trim $\Delta f$')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.tight_layout()
fig.subplots_adjust(right=0.88)
fig.colorbar(
    sm, ax=ax2, pad=0.02, shrink=0.95,
    label=r'$\delta_0$ [$10^{-4}$]',
    format=lambda x, _: f'{x * 1e4:.2f}',
)

# Longitudinal phase space (HL-LHC-style; fixed trim, not ramp diagnostic)
plt.figure()
for i in range(delta_mon.shape[0]):
    plt.plot(ctx2np(mon.zeta)[i], delta_mon[i] * 1e4, alpha=0.5)
plt.xlabel(r'$\zeta$ [m]')
plt.ylabel(r'$\delta$ [$10^{-4}$]')

# --- Ramp endpoints: manual D_x at IPs (frozen trim, no ramp) ---
# Full static workflow: 007a_radial_steering_static.py
line.enable_time_dependent_vars = False
DELTA_SPAN = 4e-3
N_DELTA = 21

for df_hz_end in (df_hz_lo, df_hz_hi):
    env['df_hz'] = df_hz_end
    delta_expected_end = -df_hz_end / (h_rf * eta * f0)
    tw_end = line.twiss()
    co_guess0 = tw_end.particle_on_co.copy()
    delta_vals = np.linspace(
        delta_expected_end - DELTA_SPAN,
        delta_expected_end + DELTA_SPAN,
        N_DELTA,
    )
    res = measure_dx_manual(
        line,
        delta_vals,
        co_guess0=co_guess0,
        ips=IPS,
        delta0_twiss=delta_expected_end,
    )
    print_dx_comparison(
        res['dx_manual'],
        res['dx_twiss'],
        ips=IPS,
        df_hz=df_hz_end,
        dx_prime_manual=res['dx_prime_manual'],
    )
    plot_dx_scan(
        res['delta_vals'],
        res['x_at_ip'],
        res['dx_manual'],
        df_hz=df_hz_end,
        ips=IPS,
        dx_prime_manual=res['dx_prime_manual'],
        fit_coeff=res['fit_coeff'],
    )

plt.show()