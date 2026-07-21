import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from solenoid_params import FIELD_TAG

parser = argparse.ArgumentParser()
parser.add_argument('--no-show', action='store_true',
                     help='Save the plot but do not open an interactive window.')
args = parser.parse_args()

HERE = Path(__file__).parent
IP = 'ipg'
RAMP_AMPS = [0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03,
             0.05, 0.07, 0.1]

# Window (in s, relative to the IP) over which the phase-advance change
# across the solenoid region is evaluated. The solenoid+compensation
# hardware itself only extends to about +-14 m, so +-20 m sits safely
# outside it in the following drift on both sides.
PHASE_WINDOW_DS = 20.0


def raw_coupling(path):
    env = xt.load(path)
    line = env.fccee_p_ring
    line.cycle(f'end_ds_start_straight_{IP}')

    s_ip = line.get_table()['s', IP]
    line.cut_at_s([s_ip - PHASE_WINDOW_DS, s_ip + PHASE_WINDOW_DS])

    for ip in ['ipa', 'ipd', 'ipg', 'ipj']:
        line[f'on_sol_{ip}'] = 0
        line[f'on_comp_sol_{ip}'] = 0

    tw0 = line.twiss4d(strengths=True)

    line[f'on_sol_{IP}'] = 1
    line[f'on_comp_sol_{IP}'] = 1

    tw_on = line.twiss(
        strengths=True,
        init_at=IP,
        betx=tw0['betx', IP],
        bety=tw0['bety', IP],
        start=f'end_ds_start_straight_{IP}',
        end=f'end_straight_start_ds_{IP}',
    )

    i_lo = np.argmin(np.abs(tw_on.s - (s_ip - PHASE_WINDOW_DS)))
    i_hi = np.argmin(np.abs(tw_on.s - (s_ip + PHASE_WINDOW_DS)))

    return {
        'betx2_max': np.max(np.abs(tw_on['betx2'])),
        'bety1_max': np.max(np.abs(tw_on['bety1'])),
        'alfx2_max': np.max(np.abs(tw_on['alfx2'])),
        'alfy1_max': np.max(np.abs(tw_on['alfy1'])),
        'dy_max': np.max(np.abs(tw_on['dy'])),
        'dmux': tw_on.mux[i_hi] - tw_on.mux[i_lo],
        'dmuy': tw_on.muy[i_hi] - tw_on.muy[i_lo],
    }


results = {}
for ramp_amp in RAMP_AMPS:
    if ramp_amp == 0.0:
        path = HERE / f'temp_fcc_ee_lcc_splineboris_solenoids_{FIELD_TAG}.json'
    else:
        subprocess.run(
            [sys.executable, '004a2_build_solenoid_bz_ramp.py',
             '--ramp-amp', str(ramp_amp)],
            cwd=HERE, check=True, capture_output=True)
        subprocess.run(
            [sys.executable, '004b2_install_solenoid_bz_ramp_in_fcc_ring.py',
             '--ramp-amp', str(ramp_amp), '--no-plot'],
            cwd=HERE, check=True, capture_output=True)
        path = (
            HERE
            / f'temp_fcc_ee_lcc_splineboris_solenoids_{FIELD_TAG}_bz_ramp_{ramp_amp:g}.json'
        )

    metrics = raw_coupling(path)
    results[ramp_amp] = metrics
    print(f'ramp_amp={ramp_amp:5.2f}  '
          + '  '.join(f'{k}={v:.4e}' for k, v in metrics.items()))

print()
print(f"{'ramp_amp':>10s} {'betx2_max':>12s} {'bety1_max':>12s} "
      f"{'alfx2_max':>12s} {'alfy1_max':>12s} {'dmux':>12s} {'dmuy':>12s}")
for ramp_amp, metrics in results.items():
    print(f"{ramp_amp:10.3f} {metrics['betx2_max']:12.4e} "
          f"{metrics['bety1_max']:12.4e} {metrics['alfx2_max']:12.4e} "
          f"{metrics['alfy1_max']:12.4e} {metrics['dmux']:12.4e} "
          f"{metrics['dmuy']:12.4e}")


########
# Plot #
########

ramp_amps = np.array(sorted(results))
log_metric_names = ['betx2_max', 'bety1_max', 'alfx2_max', 'alfy1_max']
lin_metric_names = ['dmux', 'dmuy']
baseline = results[0.0]

fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
for ax, name in zip(axes.flat, log_metric_names):
    values = np.array([results[ra][name] for ra in ramp_amps])
    nonzero = ramp_amps > 0
    ax.plot(ramp_amps[nonzero], values[nonzero], 'o-', color='C0',
            label=name)
    ax.axhline(baseline[name], color='0.4', linestyle='--', linewidth=1,
                label='baseline (ramp_amp=0)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ramp_amp (fraction of B0 swing per end)')
    ax.set_ylabel(name)
    ax.set_title(name)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=8)

# dmux/dmuy only move by ~1e-9-1e-8 (2pi rad) on top of an O(0.5) baseline,
# so plotting the raw values buries the trend in matplotlib's offset
# notation. Plot the deviation from the ramp_amp=0 baseline instead, so the
# baseline is exactly y=0 and the (tiny but real) variation is visible on
# its own scale.
mu_labels = {'dmux': r'$\Delta\mu_x - \Delta\mu_x(0)$ [$2\pi$ rad]',
             'dmuy': r'$\Delta\mu_y - \Delta\mu_y(0)$ [$2\pi$ rad]'}
for ax, name in zip(axes.flat[4:], lin_metric_names):
    values = np.array([results[ra][name] for ra in ramp_amps])
    nonzero = ramp_amps > 0
    ax.plot(ramp_amps[nonzero], values[nonzero] - baseline[name], 'o-',
            color='C0', label=f'{name} - baseline')
    ax.axhline(0.0, color='0.4', linestyle='--', linewidth=1,
               label='baseline (ramp_amp=0)')
    ax.set_xscale('log')
    ax.set_xlabel('ramp_amp (fraction of B0 swing per end)')
    ax.set_ylabel(mu_labels[name])
    ax.set_title(name)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=8)

fig.suptitle(
    f'Raw (uncorrected) coupling at {IP} vs Bz-ramp amplitude\n'
    f'($\\Delta\\mu$ over s = IP $\\pm$ {PHASE_WINDOW_DS:g} m)',
    fontsize=13)
fig.tight_layout()

out_png = HERE / 'temp_bz_ramp_amp_scan.png'
fig.savefig(out_png, dpi=130)
print(f'Saved plot to {out_png}')

if not args.no_show:
    plt.show()
