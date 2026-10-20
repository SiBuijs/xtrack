"""Q''y after 004c comes from the QD0 -> sdy1 phase error, not from the beta beat
at the sdm1 sextupole that 004c targets.

Two candidate causes of the Q''y error of a corrected lattice (claude_notes/08,
12): (A) the vertical phase advance from QD0 to the first local-chromatic-
correction sextupole sdy1, which must stay at 0.5 for the -I scheme to cancel
the final-focus chromatic beta beat, and (B) the beta beat at sdm1, the
sextupole on each side that the 004c optics match targets (sext_left/right).

The test uses the SB lattices with and without mid-bend trims (2T and 3T, the
004c output of RESULTS L19-L26). Per lattice, on the closed ring:
- Q''y (true d2Q/ddelta2, 21-point fit as 004n) of the bare ring and of the
  corrected ring;
- per side (4 IPs x left/right): the corrected - bare QD0 -> sdy1 phase error
  (004c's convention: QD0R entrance -> first sdy1r slice, first sdy1l slice ->
  QD0L exit), and the corrected - bare beta beat at sdm1, sdy1 and sdy2;
- the chromatic weight sum(beta_y k2l Dx) of sdm1 and sdy1 per side (bare).

Three tests, with a pure vertical phase trombone (LineSegmentMap with qy only,
matched to the local twiss, claude_notes/08) 20 m from each IP, between the
compensation solenoid and sdm1. It shifts the QD0 -> sdy1 phase and leaves beta
and dispersion unchanged everywhere, so it separates (A) from (B):
1. Calibration: dQ''y/dmu per side in the bare ring (ipa left and right,
   +-1e-4), and the ring Delta Q''y predicted from the phase errors alone.
2. Cancel: each side's trombone set to minus its phase error. (A) predicts Q''y
   back at the bare value; the beta beat at sdm1 is untouched.
3. Transplant: the trims lattice with the no-trims phase errors put in by the
   trombones. (A) predicts the no-trims Q''y, with the trims lattice's (small)
   beta beat at sdm1.

The clue (second figure, `_ir_profile`), around ipa:
- the vertical phase error accumulated outward from QD0, corrected - bare, along
  s. It builds up between QD0 and sdy1, where the trims act, and none of the
  optics targets (straight edge, sdm1) sees it;
- the chromatic function W_y along s, out into the arcs. The -I pair confines
  the final-focus W_y (~4e3) to the IR in the bare ring; a QD0 -> sdy1 phase
  error lets it leak, and the arc W_y grows with the error.

Writes data/q2y_phase_test/004o_q2y_phase_vs_beta_beat.json and
004o_q2y_phase_vs_beta_beat{,_ir_profile}.pdf in the studies plot folder.
Runtime ~15 min (18 Q'' fits of 21 twisses each).

Usage: python 004o_q2y_phase_vs_beta_beat.py [--b0 2.0 3.0] [--replot] [--no-show]
(--replot opens the saved results in seconds, for zooming and panning.)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from aperture_study_io import PLOT_DIR, RUN_PROVENANCE, pdf_metadata
from lattice_knobs import IP_NAMES, set_lattice_knobs
from solenoid_params import field_tag

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / 'nonlinear_tunes'))
from detuning import get_nonlinear_chromaticity  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--b0', type=float, nargs='+', default=[2.0, 3.0],
                    help='Main-solenoid fields to test (default: 2.0 3.0).')
parser.add_argument('--replot', action='store_true',
                    help='Redraw the figures from the saved JSON (seconds).')
parser.add_argument('--no-show', action='store_true',
                    help='Save the figures without opening them.')
args = parser.parse_args()

OUT_DIR = HERE / 'data' / 'q2y_phase_test'
OUT_STEM = '004o_q2y_phase_vs_beta_beat'
OUT_JSON = OUT_DIR / f'{OUT_STEM}.json'
OUT_PDF = PLOT_DIR / f'{OUT_STEM}.pdf'
OUT_PDF_PROFILE = PLOT_DIR / f'{OUT_STEM}_ir_profile.pdf'

PROFILE_IP = 'ipa'
PROFILE_HALF_WIDTH = 3000.0   # m around PROFILE_IP stored for the profile plot
ARC_MIN_DISTANCE = 3000.0     # m from every IP: "arc" for the W_y rms

CHROM_POINTS = 21          # as 004n, so Q''y matches RESULTS L20/L24
CALIBRATION_DMU = 1e-4     # trombone step for dQ''y/dmu (linear up to >3e-4)
CALIBRATION_IP = 'ipa'
TROMBONE_DISTANCE = 20.0   # m from the IP: past the compensation solenoid
SIDES = ('left', 'right')

# From 004c's config (004c runs on import, so it is copied, not imported): the
# QD0 at the IP end of each doublet, and the sextupole the optics match targets.
QD0_LEFT = {'ipa': 'qd0al.3', 'ipd': 'qd0al.0', 'ipg': 'qd0al.1',
            'ipj': 'qd0al.2'}
QD0_RIGHT = {'ipa': 'qd0ar.0', 'ipd': 'qd0ar.1', 'ipg': 'qd0ar.2',
             'ipj': 'qd0ar.3'}
SDM1 = {'ipa': ('sdm1l.7', 'sdm1r.0'), 'ipd': ('sdm1l.1', 'sdm1r.2'),
        'ipg': ('sdm1l.3', 'sdm1r.4'), 'ipj': ('sdm1l.5', 'sdm1r.6')}


def lattice_json(b0, trims):
    suffix = '' if trims else '_notrims'
    return (HERE / 'fccee_z_lcc_splineboris_solenoids_coupling_corrected_'
            f'{field_tag(b0)}{suffix}.json')


def trombone_name(ip_name, side):
    return f'trombone_qd0_sdy1_{side}_{ip_name}'


def load_line(path):
    """The ring from `path`, started mid-arc between ipa and ipd (as 004n, so
    no straight straddles s = 0), with an identity trombone on each side of
    every IP."""
    env = xt.load(path)
    line = env.fccee_p_ring
    tt = line.get_table()
    s_a, s_d = tt['s', 'ipa'], tt['s', 'ipd']
    s_mid = (s_a + s_d) / 2 if s_d > s_a else (s_a + s_d + tt.s[-1]) / 2
    ii = int(np.argmin(np.abs(tt.s[:-1] - s_mid)))
    while 'drift' not in tt.name[ii]:
        ii += 1
    line.cycle(tt.name[ii])

    tt = line.get_table()
    places = []
    for ip_name in IP_NAMES:
        for side, sign in (('left', -1), ('right', +1)):
            name = trombone_name(ip_name, side)
            env.elements[name] = xt.LineSegmentMap(
                qx=0.0, qy=0.0, betx=1.0, bety=1.0)
            places.append(env.place(
                name, at=tt['s', ip_name] + sign * TROMBONE_DISTANCE))
    line.insert(places)
    return line


def side_elements(line):
    """Per (ip, side): the QD0 -> sdy1 phase range (start, end) in beam order,
    the targeted sdm1 and the slices of sdy1 and sdy2."""
    tt = line.get_table()
    names = np.asarray(tt.name)
    is_sext = np.asarray(tt.element_type) == 'Sextupole'
    out = {}
    for ip_name in IP_NAMES:
        s_ip = tt['s', ip_name]
        for side in SIDES:
            window = ((tt.s > s_ip) & (tt.s < s_ip + 1000) if side == 'right'
                      else (tt.s < s_ip) & (tt.s > s_ip - 1000))
            tag = side[0]
            fam = {
                kk: [str(nn) for nn in names[window & is_sext]
                     if nn.startswith(f'{kk}{tag}')]
                for kk in ('sdy1', 'sdy2')
            }
            if not fam['sdy1'] or not fam['sdy2']:
                raise SystemExit(f'sdy1/sdy2 not found at {ip_name} {side}.')
            if side == 'right':
                rng = (QD0_RIGHT[ip_name], fam['sdy1'][0])
            else:
                ii_qd0 = np.flatnonzero(names == QD0_LEFT[ip_name])[-1]
                rng = (fam['sdy1'][0], str(names[ii_qd0 + 1]))
            out[ip_name, side] = dict(
                phase_range=rng, sdm1=SDM1[ip_name][side == 'right'],
                sdy1=fam['sdy1'], sdy2=fam['sdy2'])
    return out


def phase(tw, rng):
    return float(np.mod(tw['muy', rng[1]] - tw['muy', rng[0]], tw.qy))


def set_trombones(line, tw, dmu):
    """Trombone on each side matched to `tw` (the state being tested), adding
    dmu[(ip, side)] to that side's QD0 -> sdy1 phase. Missing sides: 0."""
    for ip_name in IP_NAMES:
        for side in SIDES:
            name = trombone_name(ip_name, side)
            el = line[name]
            for aa in ('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx', 'dy', 'dpy'):
                getattr(el, aa)[0] = getattr(el, aa)[1] = tw[aa, name]
            for aa, cc in (('x_ref', 'x'), ('px_ref', 'px'),
                           ('y_ref', 'y'), ('py_ref', 'py')):
                getattr(el, aa)[0] = getattr(el, aa)[1] = tw[cc, name]
            el.qx = 0.0
            el.qy = dmu.get((ip_name, side), 0.0)


def q2y(line):
    """True Q''y and Q'y from the 21-point fit (detuning.py stores Q''/4)."""
    chrom = get_nonlinear_chromaticity(line, npoints=CHROM_POINTS, order=2)
    return 4 * float(chrom.qy_derivatives[2]), float(chrom.qy_derivatives[1])


def beta_beat(tw, tw0, names):
    """Largest |corrected/bare - 1| over `names`, for beta_x and beta_y."""
    return {pl: float(max(abs(tw[f'bet{pl}', nn] / tw0[f'bet{pl}', nn] - 1)
                          for nn in names))
            for pl in ('x', 'y')}


def chromatic_weight(line, tw0, names):
    """sum(beta_y k2l Dx) over the slices `names`, bare ring."""
    return float(sum(tw0['bety', nn] * line[nn].k2 * line[nn].length
                     * tw0['dx', nn] for nn in names))


def ir_profile(line, elems):
    """Phase error from QD0 and W_y along s around PROFILE_IP, and the arc W_y
    rms, for the bare (knobs off) and corrected (on) ring. Leaves the knobs on."""
    set_lattice_knobs(line, with_solenoids=False, with_correctors=False)
    tw0 = line.twiss4d(chrom=True)
    set_lattice_knobs(line, with_solenoids=True, with_correctors=True)
    tw1 = line.twiss4d(chrom=True)
    s0 = np.asarray(tw0.s)
    s = s0 - tw0['s', PROFILE_IP]
    dmu = np.asarray(tw1.muy) - np.asarray(tw0.muy)
    # Referenced to the QD0 end facing the IP, not to the IP marker (whose
    # muy jumps with the waist position, claude_notes/08), and signed so that
    # + means more phase from QD0 to s, as the per-side phase errors.
    ref_r = elems[PROFILE_IP, 'right']['phase_range'][0]
    ref_l = elems[PROFILE_IP, 'left']['phase_range'][1]
    from_qd0 = np.where(
        s > 0, dmu - (tw1['muy', ref_r] - tw0['muy', ref_r]),
        -(dmu - (tw1['muy', ref_l] - tw0['muy', ref_l])))
    window = np.abs(s) <= PROFILE_HALF_WIDTH
    circ = s0[-1]
    dist = np.min([np.abs((s0 - tw0['s', ip] + circ / 2) % circ - circ / 2)
                   for ip in IP_NAMES], axis=0)
    arc = dist > ARC_MIN_DISTANCE

    def rms(vv):
        return float(np.sqrt(np.mean(np.asarray(vv)[arc] ** 2)))
    markers = {}
    for side in SIDES:
        ee = elems[PROFILE_IP, side]
        for kk, nn in (('sdy1', ee['sdy1'][0]), ('sdy2', ee['sdy2'][0]),
                       ('sdm1', ee['sdm1'])):
            markers[f'{kk}{side[0]}'] = float(tw0['s', nn] - tw0['s', PROFILE_IP])
    return dict(
        s=s[window].tolist(),
        dmu_from_qd0=from_qd0[window].tolist(),
        wy_bare=np.asarray(tw0.wy_chrom)[window].tolist(),
        wy_on=np.asarray(tw1.wy_chrom)[window].tolist(),
        wy_arc_rms_bare=rms(tw0.wy_chrom), wy_arc_rms_on=rms(tw1.wy_chrom),
        markers=markers)


def check_trombones(line, tw_ref, elems, dmu, label):
    """The trombones must shift each phase by exactly dmu and leave beta
    unchanged; otherwise the tests below would not separate (A) from (B)."""
    tw = line.twiss4d()
    worst_phase, worst_beta = 0.0, 0.0
    for key, ee in elems.items():
        want = phase(tw_ref, ee['phase_range']) + dmu.get(key, 0.0)
        worst_phase = max(worst_phase, abs(phase(tw, ee['phase_range']) - want))
        bb = beta_beat(tw, tw_ref, [ee['sdm1'], *ee['sdy1'], *ee['sdy2']])
        worst_beta = max(worst_beta, bb['x'], bb['y'])
    print(f'  trombone check ({label}): phase off by {worst_phase:.1e}, '
          f'beta at sdm1/sdy by {worst_beta:.1e}')
    if worst_phase > 1e-6 or worst_beta > 1e-5:
        raise SystemExit('Trombones are not a pure phase shift; stopping.')
    return tw


def run():
    results = dict(provenance=RUN_PROVENANCE, chrom_points=CHROM_POINTS,
                   trombone_distance_m=TROMBONE_DISTANCE, lattices={},
                   transplant={})
    for b0 in args.b0:
        ftag = field_tag(b0)
        dmu_by_trims = {}
        lines = {}
        for trims in (True, False):
            label = f'SB-{ftag} {"trims" if trims else "no trims"}'
            path = lattice_json(b0, trims)
            print(f'\n=== {label}: {path.name}')
            line = load_line(path)
            elems = side_elements(line)
            lines[trims] = (line, elems)

            set_lattice_knobs(line, with_solenoids=False, with_correctors=False)
            tw0 = line.twiss4d()
            q2y_bare, q1y_bare = q2y(line)

            calibration = None
            if trims and 'calibration' not in results:
                # dQ''y/dmu per side, in the bare ring, by central difference.
                calibration = {}
                for side in SIDES:
                    key = (CALIBRATION_IP, side)
                    vals = []
                    for sign in (+1, -1):
                        dmu = {key: sign * CALIBRATION_DMU}
                        set_trombones(line, tw0, dmu)
                        check_trombones(line, tw0, elems, dmu,
                                        f'calibration {side} {sign:+d}')
                        vals.append(q2y(line)[0])
                    calibration[side] = (vals[0] - vals[1]) / (2 * CALIBRATION_DMU)
                    print(f'  dQ\'\'y/dmu {side}: {calibration[side]:.4g}')
                set_trombones(line, tw0, {})
                results['calibration'] = calibration
            calibration = results['calibration']

            set_lattice_knobs(line, with_solenoids=True, with_correctors=True)
            tw1 = line.twiss4d()
            q2y_on, q1y_on = q2y(line)

            sides = {}
            dmu_err = {}
            for (ip_name, side), ee in elems.items():
                err = phase(tw1, ee['phase_range']) - phase(tw0, ee['phase_range'])
                dmu_err[ip_name, side] = err
                sides[f'{ip_name}_{side}'] = dict(
                    phase_range=list(ee['phase_range']),
                    dmu=err,
                    predicted_dq2y=calibration[side] * err,
                    sdm1=ee['sdm1'],
                    beat_sdm1=beta_beat(tw1, tw0, [ee['sdm1']]),
                    beat_sdy1=beta_beat(tw1, tw0, ee['sdy1']),
                    beat_sdy2=beta_beat(tw1, tw0, ee['sdy2']),
                    weight_sdm1=chromatic_weight(line, tw0, [ee['sdm1']]),
                    weight_sdy1=chromatic_weight(line, tw0, ee['sdy1']),
                )
            dmu_by_trims[trims] = dmu_err

            # Test 2: cancel every side's phase error with its trombone.
            cancel = {kk: -vv for kk, vv in dmu_err.items()}
            set_trombones(line, tw1, cancel)
            check_trombones(line, tw1, elems, cancel, 'cancel')
            q2y_cancel, q1y_cancel = q2y(line)
            set_trombones(line, tw1, {})

            profile = ir_profile(line, elems)
            predicted = sum(ss['predicted_dq2y'] for ss in sides.values())
            results['lattices'][f'{ftag}_{"trims" if trims else "notrims"}'] = dict(
                label=label, lattice=path.name, b0=b0, trims=trims,
                q2y_bare=q2y_bare, q1y_bare=q1y_bare,
                q2y_on=q2y_on, q1y_on=q1y_on,
                dq2y_measured=q2y_on - q2y_bare,
                dq2y_predicted_from_phase=predicted,
                q2y_phase_cancelled=q2y_cancel, q1y_phase_cancelled=q1y_cancel,
                sides=sides, profile=profile)
            print(f'  Q\'\'y bare {q2y_bare:.1f}, corrected {q2y_on:.1f} '
                  f'(Delta {q2y_on - q2y_bare:+.1f}; from the phase errors '
                  f'{predicted:+.1f}); phase errors cancelled {q2y_cancel:.1f}')

        # Test 3: the trims lattice with the no-trims phase errors.
        line, elems = lines[True]
        tw1 = line.twiss4d()
        transplant = {kk: dmu_by_trims[False][kk] - dmu_by_trims[True][kk]
                      for kk in dmu_by_trims[True]}
        set_trombones(line, tw1, transplant)
        check_trombones(line, tw1, elems, transplant, 'transplant')
        q2y_tr, _ = q2y(line)
        set_trombones(line, tw1, {})
        results['transplant'][ftag] = dict(
            q2y_trims_with_notrims_phase=q2y_tr,
            q2y_notrims=results['lattices'][f'{ftag}_notrims']['q2y_on'])
        print(f'  SB-{ftag} trims + no-trims phase errors: Q\'\'y {q2y_tr:.1f} '
              f'(no-trims lattice: '
              f'{results["transplant"][ftag]["q2y_notrims"]:.1f})')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=1))
    print(f'\nSaved {OUT_JSON}')
    return results


def report(results):
    print('\n--- Q\'\'y (true d2Q/ddelta2) ---')
    print(f'{"lattice":20s} {"bare":>8s} {"corrected":>10s} {"Delta":>9s} '
          f'{"pred.(phase)":>12s} {"phase cancelled":>16s}')
    for rr in results['lattices'].values():
        print(f'{rr["label"]:20s} {rr["q2y_bare"]:8.1f} {rr["q2y_on"]:10.1f} '
              f'{rr["dq2y_measured"]:+9.1f} {rr["dq2y_predicted_from_phase"]:+12.1f} '
              f'{rr["q2y_phase_cancelled"]:16.1f}')
    for ftag, tr in results['transplant'].items():
        print(f'SB-{ftag} trims + no-trims phase errors: '
              f'{tr["q2y_trims_with_notrims_phase"]:.1f} '
              f'(no-trims lattice {tr["q2y_notrims"]:.1f})')
    cal = results['calibration']
    print(f'dQ\'\'y/dmu ({CALIBRATION_IP}, bare): left {cal["left"]:.4g}, '
          f'right {cal["right"]:.4g}')

    print('\n--- per side: phase error vs beta beat (corrected - bare) ---')
    print(f'{"lattice":20s} {"side":10s} {"dmu_y":>9s} {"sdm1 dby/by":>12s} '
          f'{"sdm1 dbx/bx":>12s} {"sdy1 dby/by":>12s} {"sdy2 dby/by":>12s}')
    for rr in results['lattices'].values():
        for side, ss in rr['sides'].items():
            print(f'{rr["label"]:20s} {side:10s} {ss["dmu"]:+9.1e} '
                  f'{ss["beat_sdm1"]["y"]:12.1e} {ss["beat_sdm1"]["x"]:12.1e} '
                  f'{ss["beat_sdy1"]["y"]:12.1e} {ss["beat_sdy2"]["y"]:12.1e}')
    print(f'\n--- W_y rms in the arcs (> {ARC_MIN_DISTANCE:.0f} m from every IP) ---')
    for rr in results['lattices'].values():
        if 'profile' in rr:
            print(f'{rr["label"]:20s} bare {rr["profile"]["wy_arc_rms_bare"]:6.2f}   '
                  f'corrected {rr["profile"]["wy_arc_rms_on"]:6.2f}')
    first = next(iter(results['lattices'].values()))['sides']
    print('\nChromatic weight sum(beta_y k2l Dx), bare: '
          + ', '.join(f'{side} sdm1 {ss["weight_sdm1"]:.3g} / sdy1 '
                      f'{ss["weight_sdy1"]:.4g}' for side, ss in first.items()))


# Lattice colours (panels b, c): fixed order, one per lattice.
LATTICE_COLORS = {'2T_trims': 'C0', '2T_notrims': 'C1',
                  '3T_trims': 'C2', '3T_notrims': 'C3'}


def plot(results):
    lat = results['lattices']
    keys = [kk for kk in LATTICE_COLORS if kk in lat]
    side_keys = list(lat[keys[0]]['sides'])

    fig, axs = plt.subplots(3, 1, figsize=(7.0, 9.0),
                            gridspec_kw=dict(height_ratios=[1.2, 1, 1]))

    # (a) Delta Q''y per lattice: measured, predicted from the phase errors,
    # left after cancelling them, and (no-trims columns) reproduced by putting
    # the no-trims phase errors into the trims lattice. Grey levels + hatching
    # for the four kinds, so the lattice colours stay reserved for (b) and (c).
    ax = axs[0]
    kinds = [
        ('measured (corrected - bare)', '0.15', None,
         lambda kk: lat[kk]['dq2y_measured']),
        ('predicted from the phase errors alone', '0.6', None,
         lambda kk: lat[kk]['dq2y_predicted_from_phase']),
        ('phase errors cancelled by trombones', 'white', None,
         lambda kk: lat[kk]['q2y_phase_cancelled'] - lat[kk]['q2y_bare']),
        ('trims lattice + no-trims phase errors', 'white', '////',
         lambda kk: (results['transplant'][kk.split('_')[0]]
                     ['q2y_trims_with_notrims_phase'] - lat[kk]['q2y_bare'])
         if kk.endswith('notrims') else np.nan),
    ]
    width = 0.2
    xx = np.arange(len(keys))
    for jj, (label, color, hatch, fn) in enumerate(kinds):
        vals = [fn(kk) for kk in keys]
        ax.bar(xx + (jj - 1.5) * width, vals, width * 0.9, color=color,
               edgecolor='0.15', linewidth=0.6, hatch=hatch, label=label)
    ax.axhline(0, color='0.5', linewidth=0.6)
    ax.set_xticks(xx, [lat[kk]['label'] for kk in keys])
    ax.set_ylabel(r"$\Delta Q''_y$ (vs bare ring)")
    ax.set_title(r"(a) $Q''_y$ error follows the QD0$\to$sdy1 phase", loc='left')
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.grid(axis='y', color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)

    # (b), (c): per side, the phase error and the beta beat at sdm1.
    width = 0.8 / len(keys)
    xs = np.arange(len(side_keys))
    for ax, title, ylabel, fn in (
            (axs[1], r'(b) QD0$\to$sdy1 vertical phase error',
             r'$\Delta\mu_y$ [$10^{-4}\times 2\pi$]', lambda ss: 1e4 * ss['dmu']),
            (axs[2], r'(c) $\beta_y$ beat at the targeted sdm1 sextupole',
             r'$|\Delta\beta_y/\beta_y|$ at sdm1', lambda ss: ss['beat_sdm1']['y'])):
        for jj, kk in enumerate(keys):
            ax.bar(xs + (jj - (len(keys) - 1) / 2) * width,
                   [fn(lat[kk]['sides'][sk]) for sk in side_keys],
                   width * 0.9, color=LATTICE_COLORS[kk], label=lat[kk]['label'])
        ax.axhline(0, color='0.5', linewidth=0.6)
        ax.set_xticks(xs, [sk.replace('_', ' ') for sk in side_keys])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left')
        ax.grid(axis='y', color='0.9', linewidth=0.6)
        ax.set_axisbelow(True)
    axs[1].legend(fontsize=7, frameon=False, ncol=2)
    # A fixed, wide log range: the beat is ~1e-6 everywhere, and a tight
    # autoscale would draw that as tall bars.
    axs[2].set_yscale('log')
    axs[2].set_ylim(1e-8, 1e-2)
    axs[2].yaxis.set_major_locator(plt.LogLocator(numticks=7))
    first = lat[keys[0]]['sides']
    w_sdm1 = max(abs(ss['weight_sdm1']) for ss in first.values())
    w_sdy1 = min(abs(ss['weight_sdy1']) for ss in first.values())
    axs[2].text(
        0.01, 0.95,
        r'chromatic weight $|\Sigma\,\beta_y k_2 l\, D_x|$ (bare): '
        f'sdm1 $\\leq$ {w_sdm1:.2g}, sdy1 $\\geq$ {w_sdy1:.0f}',
        transform=axs[2].transAxes, fontsize=7, va='top')

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, metadata=pdf_metadata(results['provenance']))
    print(f'Saved {OUT_PDF}')


def plot_profile(results):
    """The clue: where the phase error builds up, and what it does to W_y."""
    lat = results['lattices']
    keys = [kk for kk in LATTICE_COLORS if kk in lat]
    prof = {kk: lat[kk]['profile'] for kk in keys}
    markers = prof[keys[0]]['markers']

    fig, axs = plt.subplots(2, 1, figsize=(7.0, 7.0))
    ax = axs[0]
    xlim = (-800, 500)
    plateaus = [1e4 * lat[kk]['sides'][f'{PROFILE_IP}_{side}']['dmu']
                for kk in keys for side in SIDES]
    ylim = (min(min(plateaus), 0) - 0.6, max(max(plateaus), 0) + 0.6)
    lowest, off_scale = 0.0, []
    for kk in keys:
        ss = np.asarray(prof[kk]['s'])
        dd = 1e4 * np.asarray(prof[kk]['dmu_from_qd0'])
        ax.plot(ss, dd, color=LATTICE_COLORS[kk], linewidth=1.2,
                label=lat[kk]['label'])
        inside = (ss > xlim[0]) & (ss < xlim[1])
        lowest = min(lowest, float(np.nanmin(dd[inside])))
        off_scale.extend(ss[inside & (dd < ylim[0])])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if off_scale:
        # The low-beta stretch around sdm1l (beta_y of a few m, so mu moves
        # fast and a small beta change is a large local Delta mu). Every
        # curve is back on its plateau before sdy1l.
        s_lo, s_hi = min(off_scale), max(off_scale)
        ax.annotate(f'off scale (down to {lowest:.0f}) between\n'
                    f's = {s_lo:.0f} and {s_hi:.0f} m: low-$\\beta_y$ stretch',
                    xy=((s_lo + s_hi) / 2, ylim[0]), xytext=(30, ylim[0] + 0.12),
                    fontsize=7, color='0.3',
                    arrowprops=dict(arrowstyle='->', color='0.5', linewidth=0.6))
    ax.axhline(0, color='0.5', linewidth=0.6)
    ax.set_ylabel(r'$\Delta\mu_y$ from QD0 [$10^{-4}\times 2\pi$]')
    ax.set_xlabel(f's from {PROFILE_IP} [m]')
    ax.set_title(r'(a) Vertical phase error from QD0 outward (corrected $-$ bare)',
                 loc='left')
    ax.legend(fontsize=7, frameon=False, loc='upper right')

    ax = axs[1]
    ax.plot(prof[keys[0]]['s'], prof[keys[0]]['wy_bare'], color='0.15',
            linewidth=1.0, linestyle='--',
            label=f'bare ring (arc rms {prof[keys[0]]["wy_arc_rms_bare"]:.1f})')
    for kk in keys:
        ax.plot(prof[kk]['s'], prof[kk]['wy_on'], color=LATTICE_COLORS[kk],
                linewidth=1.0,
                label=f'{lat[kk]["label"]} (arc rms {prof[kk]["wy_arc_rms_on"]:.1f})')
    ax.set_yscale('log')
    # Room below the curves for the legend.
    ax.set_ylim(1e-4, 3e4)
    ax.set_xlim(-PROFILE_HALF_WIDTH, PROFILE_HALF_WIDTH)
    ax.set_ylabel(r'$W_y$')
    ax.set_xlabel(f's from {PROFILE_IP} [m]')
    ax.set_title(r'(b) $W_y$ leaks past the sdy pair into the arcs', loc='left')
    ax.legend(fontsize=7, frameon=False, loc='lower center', ncol=2)

    # All markers in (a); only the ends of the -I pairs in (b), where the
    # others would sit on top of each other.
    for ax, names in ((axs[0], list(markers)), (axs[1], ['sdy2l', 'sdy2r'])):
        lo, hi = ax.get_xlim()
        for name in names:
            ss = markers[name]
            if lo < ss < hi:
                ax.axvline(ss, color='0.75', linewidth=0.6, linestyle=':')
                ax.text(ss, 1.0, name, transform=ax.get_xaxis_transform(),
                        fontsize=6.5, color='0.4', ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(OUT_PDF_PROFILE, metadata=pdf_metadata(results['provenance']))
    print(f'Saved {OUT_PDF_PROFILE}')


if args.replot:
    results = json.loads(OUT_JSON.read_text())
else:
    results = run()
report(results)
plot(results)
plot_profile(results)
if not args.no_show:
    plt.show()
