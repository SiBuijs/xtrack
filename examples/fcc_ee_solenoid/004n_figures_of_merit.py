"""Figures of merit of a corrected 004c lattice, for paper/RESULTS.md.

Reads `fccee_z_lcc_splineboris_solenoids_coupling_corrected_{tag}.json` and
writes nothing but a JSON summary to `data/figures_of_merit/`. Three knob
states of the same lattice (`lattice_knobs.set_lattice_knobs`): `on`
(solenoids + corrections), `nocorr` (solenoids on, every correction knob off,
compensation solenoids and doublet rotation included) and `off` (the bare
ring).

Per state:
- tunes, |C-| and Q'x/Q'y (twiss `dqx/dqy`) from a 4D twiss, and twiss
  `ddqx/ddqy` from the same twiss with `chrom=True`;
- Q', Q'' from the 21-point delta sweep of 004c (`detuning.py`). That helper
  returns the Taylor coefficients divided by n! once more, so its index 2 is
  Q''/4; the true Q'' = d2Q/ddelta2 is stored as `q2x/q2y`, the raw value as
  `d2qx_printed/d2qy_printed` (claude_notes/10);
- equilibrium emittances from the 6D radiative twiss (Chao, normal modes),
  with the recipe of 004j: `configure_radiation('mean')` +
  `compensate_radiation_energy_loss`, then restored.

For `on`, against `off` (corrected - bare, closed ring):
- worst residual over the 4 IPs and over the 8 straight edges of dy, dpy,
  beta beat, alfa, the coupled beta ratios and the x/y orbit;
- the closed-ring Edwards-Teng coupling matrix at the straight edges,
  normalised G_y R G_x^-1 with the bare Floquet matrices. The edges are
  field-free, so ET is valid there (inside the solenoids SplineBoris uses
  kinetic momenta and it is not, claude_notes/01 and 11);
- the largest coupled beta ratio in the arcs (more than 1.5 km from an IP);
- the x-y leak of each straight: off-diagonal norm of its transfer map
  between the two edges, normalised with the bare W matrices.

Usage: python 004n_figures_of_merit.py --b0 3.0 [--max-transverse-order N]
       [--output-tag TAG]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import xtrack as xt

from lattice_knobs import IP_NAMES, set_lattice_knobs
from solenoid_params import (
    MAIN_SOLENOID_B0,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    order_tag,
)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / 'nonlinear_tunes'))
from detuning import get_nonlinear_chromaticity  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
add_b0_argument(parser, default=MAIN_SOLENOID_B0)
add_max_order_argument(parser)
parser.add_argument('--output-tag', default='',
                    help='Same as 004c --output-tag: selects the lattice.')
args = parser.parse_args()

TAG = (f'{field_tag(args.b0)}{order_tag(args.max_transverse_order)}'
       + (f'_{args.output_tag}' if args.output_tag else ''))
LATTICE_JSON = (
    HERE / f'fccee_z_lcc_splineboris_solenoids_coupling_corrected_{TAG}.json')
OUT_DIR = HERE / 'data' / 'figures_of_merit'
OUT_JSON = OUT_DIR / f'figures_of_merit_{TAG}.json'

CHROM_POINTS = 21
ARC_MIN_DISTANCE = 1500.0  # m from the nearest IP
STATES = {'on': (True, True), 'nocorr': (True, False), 'off': (False, False)}
EDGES = {ip: (f'end_ds_start_straight_{ip}', f'end_straight_start_ds_{ip}')
         for ip in IP_NAMES}

env = xt.load(LATTICE_JSON)
line = env.fccee_p_ring

# Start the line mid-arc between ipa and ipd so that no straight straddles
# s = 0: get_R_matrix only works forward in the table.
_tt = line.get_table()
_s_a, _s_d = _tt['s', 'ipa'], _tt['s', 'ipd']
_s_mid = ((_s_a + _s_d) / 2 if _s_d > _s_a
          else (_s_a + _s_d + _tt.s[-1]) / 2)
_ii = int(np.argmin(np.abs(_tt.s[:-1] - _s_mid)))
while 'drift' not in _tt.name[_ii]:
    _ii += 1
line.cycle(_tt.name[_ii])


def chao_emittances(line):
    """Equilibrium emittances from the 6D radiative twiss (004j's recipe)."""
    cavities = [nn for nn in line.element_names
                if isinstance(line[nn], xt.Cavity)]
    snapshot = {nn: {ff: getattr(line[nn], ff)
                     for ff in ('voltage', 'frequency', 'lag')}
                for nn in cavities}
    try:
        line.configure_radiation(model='mean')
        line.compensate_radiation_energy_loss(verbose=False)
        tw6d = line.twiss(method='6d', radiation_analysis=True)
        return dict(eq_gemitt_x=float(tw6d.eq_gemitt_x),
                    eq_gemitt_y=float(tw6d.eq_gemitt_y),
                    eq_gemitt_zeta=float(tw6d.eq_gemitt_zeta),
                    energy_loss=float(tw6d.energy_loss))
    except Exception as exc:  # noqa: BLE001 -- e.g. no closed orbit
        return dict(emit_error=repr(exc))
    finally:
        line.configure_radiation(model=None)
        for nn, fields in snapshot.items():
            for ff, value in fields.items():
                setattr(line[nn], ff, value)


def state_figures(line, with_chromaticity):
    out = {}
    try:
        tw = line.twiss4d(chrom=True)
    except Exception as exc:  # noqa: BLE001 -- e.g. no closed orbit
        out['twiss_error'] = repr(exc)
        out.update(chao_emittances(line))
        return out, None
    out.update(qx=float(tw.qx), qy=float(tw.qy), c_minus=float(tw.c_minus),
               dqx=float(tw.dqx), dqy=float(tw.dqy),
               ddqx=float(tw.ddqx), ddqy=float(tw.ddqy))
    if with_chromaticity:
        chrom = get_nonlinear_chromaticity(line, npoints=CHROM_POINTS, order=2)
        d2qx = float(chrom.qx_derivatives[2])
        d2qy = float(chrom.qy_derivatives[2])
        out.update(fit_dqx=float(chrom.qx_derivatives[1]),
                   fit_dqy=float(chrom.qy_derivatives[1]),
                   d2qx_printed=d2qx, d2qy_printed=d2qy,
                   q2x=4 * d2qx, q2y=4 * d2qy)
    out.update(chao_emittances(line))
    return out, tw


results = {'lattice': LATTICE_JSON.name}
twiss = {}
for state, (with_solenoids, with_correctors) in STATES.items():
    set_lattice_knobs(line, with_solenoids=with_solenoids,
                      with_correctors=with_correctors)
    results[state], twiss[state] = state_figures(
        line, with_chromaticity=(state != 'nocorr'))

# Corrected - bare, at the IPs and straight edges.
set_lattice_knobs(line, with_solenoids=True, with_correctors=True)
t0, t1 = twiss['off'], line.twiss4d(coupling_edw_teng=True)
assert np.all(t0.name == t1.name)


def worst_residuals(points):
    def worst(fn):
        return float(max(fn(nn) for nn in points))
    return dict(
        dy=worst(lambda nn: abs(t1['dy', nn] - t0['dy', nn])),
        dpy=worst(lambda nn: abs(t1['dpy', nn] - t0['dpy', nn])),
        beta_beat=worst(lambda nn: max(
            abs(t1['betx', nn] / t0['betx', nn] - 1),
            abs(t1['bety', nn] / t0['bety', nn] - 1))),
        alfa=worst(lambda nn: max(abs(t1['alfx', nn] - t0['alfx', nn]),
                                  abs(t1['alfy', nn] - t0['alfy', nn]))),
        coupled_beta_ratio=worst(lambda nn: max(
            t1['betx2', nn] / t1['betx1', nn],
            t1['bety1', nn] / t1['bety2', nn])),
        orbit=worst(lambda nn: max(abs(t1['x', nn] - t0['x', nn]),
                                   abs(t1['y', nn] - t0['y', nn]))),
    )


def floquet(bet, alf):
    return np.array([[1 / math.sqrt(bet), 0.0],
                     [alf / math.sqrt(bet), math.sqrt(bet)]])


def edwards_teng_normalised(nn):
    r_et = np.array([[t1['r11_edw_teng', nn], t1['r12_edw_teng', nn]],
                     [t1['r21_edw_teng', nn], t1['r22_edw_teng', nn]]])
    g_y = floquet(t0['bety', nn], t0['alfy', nn])
    g_x = floquet(t0['betx', nn], t0['alfx', nn])
    return g_y @ r_et @ np.linalg.inv(g_x)


def straight_leak(start, end):
    def w0(nn):
        return t0.W_matrix[np.where(t0.name == nn)[0][0]][:4, :4]
    r_map = t1.get_R_matrix(start=start, end=end)[:4, :4]
    r_norm = np.linalg.inv(w0(end)) @ r_map @ w0(start)
    return float(np.sqrt(np.sum(r_norm[:2, 2:]**2)
                         + np.sum(r_norm[2:, :2]**2)))


edge_names = [nn for pair in EDGES.values() for nn in pair]
s_ips = np.array([t1['s', ip] for ip in IP_NAMES])
circumference = t1.s[-1]
dist_to_ip = np.min(np.abs(
    ((t1.s[:, None] - s_ips[None, :]) + circumference / 2) % circumference
    - circumference / 2), axis=1)
in_arcs = dist_to_ip > ARC_MIN_DISTANCE
results['match'] = dict(
    ip=worst_residuals(IP_NAMES),
    edge=worst_residuals(edge_names),
    edwards_teng_edge_max=float(max(
        np.abs(edwards_teng_normalised(nn)).max() for nn in edge_names)),
    arc_coupled_beta_ratio_max=float(max(
        (t1.betx2 / t1.betx1)[in_arcs].max(),
        (t1.bety1 / t1.bety2)[in_arcs].max())),
    straight_leak={ip: straight_leak(*EDGES[ip]) for ip in IP_NAMES},
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
print(f'Wrote {OUT_JSON}')
