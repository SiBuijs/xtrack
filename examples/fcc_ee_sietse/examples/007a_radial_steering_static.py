"""Static radial steering: slip-factor check, CO continuation, longitudinal tracking.

See 007b_physical_ramp.py for the time-dependent df_hz ramp.
"""

import pathlib
import sys

import numpy as np
import xtrack as xt
from xtrack.twiss import ClosedOrbitSearchError
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

# Endpoints where twiss() finds a CO without co_guess (found manually)
DF_HZ_CO_ANCHOR_LOW = -148.0
DF_HZ_CO_ANCHOR_HIGH = 218.0

df_hz_start = DF_HZ_CO_ANCHOR_HIGH# or DF_HZ_CO_ANCHOR_HIGH
step_hz_chain = None              # None: -1 from low anchor, +1 from high anchor
df_hz_chain_end = None            # optional cap in scan direction; None = until fail
max_chain_steps = 500

if step_hz_chain is None:
    # high anchor (+218): scan upward; low anchor (-148): scan downward
    step_hz_chain = 1.0 if df_hz_start >= 0 else -1.0
df_hz_next = df_hz_start + step_hz_chain  # single-step CO demo (1 Hz jump)

# --- Setup (no ramp, no time dependence) ---
env = xt.load(str(HERE / '../lattices/z/fccee_z.json'))
line = env.fccee_p_ring
line.build_tracker()
assert not line.enable_time_dependent_vars

tw0 = line.twiss()

h_rf = env['rf_harmon_400']
f_rf = h_rf / tw0.t_rev0
eta = tw0.slip_factor
f0 = 1 / tw0.t_rev0

env['circumference'] = tw0.line_length
env['f_rf'] = f_rf
env['df_hz'] = df_hz_start
env['dzeta'] = 'circumference * df_hz / f_rf'

env.new('zeta_shift', xt.ZetaShift, dzeta='dzeta')
line.discard_tracker()
line.append('zeta_shift')
line.build_tracker()

# --- Slip-factor check at anchor (twiss without co_guess) ---
env['df_hz'] = df_hz_start
delta_expected = -df_hz_start / (h_rf * eta * f0)

tw1 = line.twiss()
print(f'df_hz = {df_hz_start} Hz (anchor, twiss from zero guess)')
print(f'  delta_expected = {delta_expected:.6e}')
print(f'  delta CO (twiss) = {tw1.delta[0]:.6e}')
print(f'  |delta - expected| / |expected| = '
      f'{abs(tw1.delta[0] - delta_expected) / abs(delta_expected):.3e}')
# Linear slip-factor vs CO: ~3e-3 at -148 Hz, ~5e-3 at +218 Hz on FCC-ee
RTOL_DELTA = 5e-3 if df_hz_start >= 0 else 3e-3
assert np.isclose(tw1.delta[0], delta_expected, rtol=RTOL_DELTA, atol=0)

# --- CO continuation df_hz_start -> df_hz_next ---
p_co_prev = tw1.particle_on_co.copy()

env['df_hz'] = df_hz_next
delta_expected_next = -df_hz_next / (h_rf * eta * f0)

p_guess_raw = p_co_prev.copy()
p_co_raw = line.find_closed_orbit(co_guess=p_guess_raw)

p_guess = p_co_prev.copy()
p_guess.delta = delta_expected_next
p_co_next = line.find_closed_orbit(co_guess=p_guess)

print(f'\nCO continuation to df_hz = {df_hz_next} Hz')
print(f'  delta_expected = {delta_expected_next:.6e}')
print(f'  delta CO (raw co_guess)   = {p_co_raw.delta[0]:.6e}')
print(f'  delta CO (slip-factor nudge) = {p_co_next.delta[0]:.6e}')
assert np.isclose(p_co_raw.delta[0], delta_expected_next, rtol=RTOL_DELTA, atol=0)
assert np.isclose(p_co_next.delta[0], delta_expected_next, rtol=RTOL_DELTA, atol=0)

# Continuation chain: 1 Hz steps until CO search fails (optional df_hz_chain_end cap)
cap_msg = (f', cap at {df_hz_chain_end:.0f} Hz' if df_hz_chain_end is not None
           else ', until CO search fails')
print(f'\nContinuation chain ({step_hz_chain:+.0f} Hz steps from {df_hz_start:.0f}{cap_msg}):')
df_hz = df_hz_start
p_co = tw1.particle_on_co.copy()
df_hz_last_ok = df_hz_start
for _ in range(max_chain_steps):
    df_hz_try = df_hz + step_hz_chain
    if df_hz_chain_end is not None:
        if step_hz_chain > 0 and df_hz_try > df_hz_chain_end:
            print(f'  stopped: reached cap df_hz = {df_hz_chain_end:.0f} Hz')
            break
        if step_hz_chain < 0 and df_hz_try < df_hz_chain_end:
            print(f'  stopped: reached cap df_hz = {df_hz_chain_end:.0f} Hz')
            break
    df_hz = df_hz_try
    env['df_hz'] = float(df_hz)
    delta_exp = -df_hz / (h_rf * eta * f0)
    p_guess = p_co.copy()
    p_guess.delta = delta_exp
    try:
        p_co = line.find_closed_orbit(co_guess=p_guess)
    except ClosedOrbitSearchError:
        print(f'  stopped: no CO at df_hz = {df_hz:.0f} Hz '
              f'(last OK: {df_hz_last_ok:.0f} Hz)')
        break
    df_hz_last_ok = df_hz
    rel_err = abs(p_co.delta[0] - delta_exp) / abs(delta_exp)
    print(f'  df_hz={df_hz:4.0f}  rel_err={rel_err:.3e}')
else:
    print(f'  stopped: max_chain_steps = {max_chain_steps} reached')

# --- Multi-turn tracking validation ---
env['df_hz'] = df_hz_start
p_test = line.build_particles(
    x_norm=0,
    delta=np.linspace(delta_expected - 4e-3, delta_expected + 4e-3, 21),
)
line.track(p_test, num_turns=1000, turn_by_turn_monitor=True)
mon = line.record_last_track
ctx2np = line._context.nparray_from_context_array
delta_mon = ctx2np(mon.delta)
zeta_mon = ctx2np(mon.zeta)

plt.figure(figsize=(6, 5))
for i in range(delta_mon.shape[0]):
    plt.plot(zeta_mon[i], delta_mon[i] * 1e4, alpha=0.5)
plt.axhline(delta_expected * 1e4, color='r', linestyle='--', lw=2,
            label=r'$\delta$ expected (slip factor)')
plt.xlabel(r'$\zeta$ [m]')
plt.ylabel(r'$\delta$ [$10^{-4}$]')
plt.title(f'Static trim: $\\Delta f$ = {df_hz_start:.0f} Hz')
plt.legend()
plt.tight_layout()

# --- Manual horizontal dispersion at IPs (static trim, both CO anchors) ---
DELTA_SPAN = 4e-3
N_DELTA = 21

for df_hz_anchor in (DF_HZ_CO_ANCHOR_LOW, DF_HZ_CO_ANCHOR_HIGH):
    env['df_hz'] = df_hz_anchor
    delta_expected_anchor = -df_hz_anchor / (h_rf * eta * f0)
    tw_anchor = line.twiss()
    co_guess0 = tw_anchor.particle_on_co.copy()
    delta_vals = np.linspace(
        delta_expected_anchor - DELTA_SPAN,
        delta_expected_anchor + DELTA_SPAN,
        N_DELTA,
    )
    res = measure_dx_manual(
        line,
        delta_vals,
        co_guess0=co_guess0,
        ips=IPS,
        delta0_twiss=delta_expected_anchor,
    )
    print_dx_comparison(
        res['dx_manual'],
        res['dx_twiss'],
        ips=IPS,
        df_hz=df_hz_anchor,
        dx_prime_manual=res['dx_prime_manual'],
    )
    plot_dx_scan(
        res['delta_vals'],
        res['x_at_ip'],
        res['dx_manual'],
        df_hz=df_hz_anchor,
        ips=IPS,
        dx_prime_manual=res['dx_prime_manual'],
        fit_coeff=res['fit_coeff'],
    )

plt.show()