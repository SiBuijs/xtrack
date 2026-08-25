"""Bs knockout test: is the SB-vs-MK tune-shift gap explained by Bs?

Compares three field-model variants for the SAME undulator, using 007b's
local-optics approach (standalone undulator Twissed with the bare ring's
local optics as initial conditions, tune shift read off the phase advance
-- see claude_notes/sls_undulator_tune_displacement_local_optics.md):

  - SB          : TubeFitter.to_line(), full spatial field including Bs.
  - SB_zero_bs  : the SAME SplineBoris elements, but with the new
                  ``zero_bs`` flag set -- forces Bs to 0 inside the Boris
                  stepper's rotation step every substep, leaving Bx/By (and
                  hence the two half-kicks) completely untouched. Off-axis
                  Bs is not a free/stored parameter (it's generated from the
                  same bx/by Hermite data that also produces Bx/By, via
                  Maxwell consistency), so there is no data-level way to
                  suppress it -- this has to happen inside the stepper.
  - MK          : TubeFitter.to_multipole_line(), no Bs channel at all.

If SB_zero_bs's tune shift collapses onto MK's, the SB-vs-MK gap is
explained by Bs. If it stays close to SB instead, Bs is not the (whole)
story.

NOTE: line.match()/.twiss()/.track() auto-build a tracker with
use_prebuilt_kernels=True by default. zero_bs only works with a prebuilt
SplineBoris kernel that actually includes it -- confirmed by first testing
with a forced from-source build (build_tracker(use_prebuilt_kernels=False)),
then regenerating the prebuilt kernels so that build is no longer needed
here. If zero_bs ever silently stops doing anything again (e.g. after
pulling someone else's prebuilt-kernel cache), that forced rebuild is the
first thing to try before suspecting the physics.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xtrack as xt
from pathlib import Path
from xtrack._temp.splineboris.tube_fitter import TubeFitter


multipole_order = 3
E0 = 2.7e9
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)

madx_file = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'sls.madx'
file_path = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'simona_field_map.txt'
distance_unit = 0.001
n_frames = 4441

_tube_fitter = None


def get_tube_fitter():
    global _tube_fitter
    if _tube_fitter is None:
        print("[TubeFitter] loading raw field map and fitting (slow, once)...")
        df_raw_data = pd.read_csv(
            file_path, sep="\t", header=None,
            names=["X", "Y", "Z", "Bx", "By", "Bs"],
            dtype=float,
        ).set_index(["X", "Y", "Z"])
        tf = TubeFitter(
            raw_data=df_raw_data,
            n_frames=n_frames,
            distance_unit=distance_unit,
            deg=multipole_order - 1,
            field_tol=1e-4,
        )
        tf.fit()
        _tube_fitter = tf
    return _tube_fitter


PLACE = sys.argv[1] if len(sys.argv) > 1 else 'both'
WIGGLER_PLACES = {
    'ars11_uind_0210_1': ['ars11_uind_0210_1'],
    'ars11_uind_0610_1': ['ars11_uind_0610_1'],
    'both': ['ars11_uind_0210_1', 'ars11_uind_0610_1'],
}[PLACE]

CORR_VAR_NAMES = ['k0l_corr1', 'k0sl_corr1', 'k0l_corr2', 'k0sl_corr2',
                  'k0l_corr3', 'k0sl_corr3', 'k0l_corr4', 'k0sl_corr4']

N_TUNES = 30
HOR_OFF_LIST = np.linspace(-0.5e-3, 0.5e-3, N_TUNES)

MODEL_LABELS = ('SB', 'SB_zero_bs', 'MK')


def _match_orbit(undulator):
    opt = undulator.match(
        solve=False,
        betx=0, bety=0,
        only_orbit=True,
        include_collective=True,
        vary=xt.VaryList(CORR_VAR_NAMES, step=1e-6),
        targets=[
            xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.START),
            xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
        ],
    )
    opt.step(2)


def build_undulator(model_label):
    tube_fitter = get_tube_fitter()
    und_env = xt.Environment()
    und_env.particle_ref = p0.copy()

    if model_label.startswith('SB'):
        undulator_line = tube_fitter.to_line(multipole_order=multipole_order)
    else:
        undulator_line = tube_fitter.to_multipole_line(
            multipole_order=multipole_order, p0c=E0, field_at='mean')
    undulator = und_env.import_line(undulator_line, line_name='undulator')
    l_wig = undulator.get_length()

    for kk in CORR_VAR_NAMES:
        und_env[kk] = 0.
    und_env.new('corr1', xt.Multipole, knl=['k0l_corr1'], ksl=['k0sl_corr1'])
    und_env.new('corr2', xt.Multipole, knl=['k0l_corr2'], ksl=['k0sl_corr2'])
    und_env.new('corr3', xt.Multipole, knl=['k0l_corr3'], ksl=['k0sl_corr3'])
    und_env.new('corr4', xt.Multipole, knl=['k0l_corr4'], ksl=['k0sl_corr4'])
    undulator.insert([
        und_env.place('corr1', at=0.02),
        und_env.place('corr2', at=0.1),
        und_env.place('corr3', at=l_wig - 0.1),
        und_env.place('corr4', at=l_wig - 0.02),
    ], s_tol=5e-3)

    field_names = [nn for nn in undulator.element_names
                   if nn.startswith('tubefitter')]
    if model_label == 'SB_zero_bs':
        for nn in field_names:
            undulator[nn].zero_bs = 1

    # Zero-offset match happens AFTER zero_bs is set, so the correctors are
    # matched self-consistently against whichever physics is actually active
    # for this variant.
    _match_orbit(undulator)

    return undulator, l_wig, field_names


def scan_model(model_label, init_conditions, tw_no_undulator):
    undulator, l_wig, field_names = build_undulator(model_label)

    deltaqx_list = []
    deltaqy_list = []
    for dx in HOR_OFF_LIST:
        for nn in field_names:
            undulator[nn].shift_x = dx
        _match_orbit(undulator)

        dqx_total = 0.0
        dqy_total = 0.0
        for ic in init_conditions:
            tw_p = undulator.twiss4d(
                betx=ic['betx'], bety=ic['bety'],
                alfx=ic['alfx'], alfy=ic['alfy'],
                include_collective=True)

            mux_bare_start = np.interp(ic['s0'], tw_no_undulator.s, tw_no_undulator.mux)
            mux_bare_end = np.interp(ic['s0'] + l_wig, tw_no_undulator.s, tw_no_undulator.mux)
            muy_bare_start = np.interp(ic['s0'], tw_no_undulator.s, tw_no_undulator.muy)
            muy_bare_end = np.interp(ic['s0'] + l_wig, tw_no_undulator.s, tw_no_undulator.muy)

            dqx_total += tw_p.mux[-1] - (mux_bare_end - mux_bare_start)
            dqy_total += tw_p.muy[-1] - (muy_bare_end - muy_bare_start)

        deltaqx_list.append(dqx_total)
        deltaqy_list.append(dqy_total)

    return np.array(deltaqx_list), np.array(deltaqy_list)


def main():
    env = xt.load(str(madx_file))
    line_sls = env.ring
    line_sls.configure_bend_model(core='mat-kick-mat')
    line_sls.particle_ref = p0.copy()
    tw_no_undulator = line_sls.twiss4d(include_collective=True)

    init_conditions = []
    for wig_place in WIGGLER_PLACES:
        s0 = float(tw_no_undulator['s', wig_place])
        init_conditions.append(dict(
            betx=float(tw_no_undulator['betx', wig_place]),
            bety=float(tw_no_undulator['bety', wig_place]),
            alfx=float(tw_no_undulator['alfx', wig_place]),
            alfy=float(tw_no_undulator['alfy', wig_place]),
            s0=s0,
        ))

    results = {}
    for model_label in MODEL_LABELS:
        print("=" * 80)
        print(f"Scanning model={model_label}  place={PLACE}")
        print("=" * 80)
        dqx, dqy = scan_model(model_label, init_conditions, tw_no_undulator)
        results[model_label] = (dqx, dqy)

        coef_x = np.polyfit(HOR_OFF_LIST, dqx, 2)
        coef_y = np.polyfit(HOR_OFF_LIST, dqy, 2)
        print(f"  d(DQx)/dx = {coef_x[1]:.6e}   (1/2)d2(DQx)/dx2 = {coef_x[0]:.6e}")
        print(f"  d(DQy)/dx = {coef_y[1]:.6e}   (1/2)d2(DQy)/dx2 = {coef_y[0]:.6e}")

    # Fraction of the SB-vs-MK slope gap that closes when Bs is knocked out.
    # ~1 => Bs explains the gap; ~0 => Bs is not responsible.
    slope = {ml: np.polyfit(HOR_OFF_LIST, results[ml][0], 2)[1] for ml in MODEL_LABELS}
    slope_y = {ml: np.polyfit(HOR_OFF_LIST, results[ml][1], 2)[1] for ml in MODEL_LABELS}
    denom_x = slope['SB'] - slope['MK']
    denom_y = slope_y['SB'] - slope_y['MK']
    frac_x = (slope['SB'] - slope['SB_zero_bs']) / denom_x if denom_x else float('nan')
    frac_y = (slope_y['SB'] - slope_y['SB_zero_bs']) / denom_y if denom_y else float('nan')
    print("=" * 80)
    print(f"Fraction of (SB - MK) d(DQx)/dx gap closed by zeroing Bs: {frac_x:.3f}")
    print(f"Fraction of (SB - MK) d(DQy)/dx gap closed by zeroing Bs: {frac_y:.3f}")
    print("=" * 80)

    colors = {'SB': 'tab:blue', 'SB_zero_bs': 'tab:red', 'MK': 'tab:green'}
    labels = {'SB': 'SB (real Bs)', 'SB_zero_bs': 'SB (Bs forced to 0)',
              'MK': 'MK (no Bs channel)'}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for model_label in MODEL_LABELS:
        dqx, dqy = results[model_label]
        ax1.plot(HOR_OFF_LIST, dqx, marker='o', color=colors[model_label],
                  label=labels[model_label])
        ax2.plot(HOR_OFF_LIST, dqy, marker='o', color=colors[model_label],
                  label=labels[model_label])
    ax1.set_ylabel('Delta Qx')
    ax1.set_title(f'Bs knockout test (local optics, phase advance) -- place={PLACE}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.set_xlabel('Horizontal offset [m]')
    ax2.set_ylabel('Delta Qy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
