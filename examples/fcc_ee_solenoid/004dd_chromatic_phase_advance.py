"""Chromatic phase advance along the ring: mu(s), mu'(s), mu''(s).

get_nonlinear_chromaticity (examples/nonlinear_tunes/detuning.py) fits a
quadratic to the *total* tunes of npoints off-momentum 4D twisses. Here the
same off-momentum twisses are run, but mux(s) and muy(s) are kept at every
element and the quadratic in delta is fitted at each s:

    mu(s, delta) ~ mu(s) + mu'(s) delta + mu''(s)/2 delta^2

with the primes true derivatives with respect to delta. At the end of the
ring these reduce to Q, Q' and Q''; along s they show where the chromatic
phase error builds up (IR straights vs arcs).

NOTE on conventions: detuning.py's Chromaticity stores coef / n!, so its
q{x,y}_derivatives[2] (printed as d2qx/d2qy by 004c/004j) is Q''/4 in the
convention above. The end-of-ring report prints that value as well so the
numbers can be compared directly.

Two cases are scanned on the 004c output lattice: bare (solenoids and
corrections off) and corrected (both on). The fit result is cached in
DATA_DIR; --replot redraws from the cache without twissing.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from aperture_study_io import DATA_DIR, PLOT_DIR
from lattice_knobs import IP_NAMES, set_lattice_knobs
from solenoid_params import (
    MAIN_SOLENOID_B0,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    order_tag,
)


parser = argparse.ArgumentParser(
    description='Fit mux, muy vs delta at every s of the corrected FCC ring.')
add_b0_argument(parser, default=MAIN_SOLENOID_B0)
add_max_order_argument(parser)
parser.add_argument(
    '--npoints', type=int, default=21,
    help='Number of off-momentum twisses per case (default: 21, as '
         'get_nonlinear_chromaticity).')
parser.add_argument(
    '--delta-max', type=float, default=1e-3,
    help='Deltas span [-delta_max, +delta_max] (default: 1e-3, as '
         'get_nonlinear_chromaticity).')
parser.add_argument(
    '--start', default='ipa',
    help='Element the ring is cycled to; mu(s) accumulates from here '
         '(default: ipa, as the 004c chromaticity report).')
parser.add_argument(
    '--replot', action='store_true',
    help='Load the cached fit instead of running the twisses.')
parser.add_argument(
    '--zoom-ip', nargs='*', default=['ipd'], metavar='IP',
    help='IPs for the tw.plot zoom with the magnet structure (default: ipd; '
         'the --start IP sits at s=0, so its window would be cut in half). '
         'Pass no value to skip the zoom plots.')
parser.add_argument(
    '--zoom-half-width', type=float, default=1400.0, metavar='M',
    help='Half-width of the zoom window around the IP in metres '
         '(default: 1400, just beyond the half-straight).')
args = parser.parse_args()

FIELD_TAG = field_tag(args.b0)
ORDER_TAG = order_tag(args.max_transverse_order)

HERE = Path(__file__).parent
INPUT_LATTICE_JSON = (
    HERE / ('fccee_z_lcc_splineboris_solenoids_coupling_corrected_'
            f'{FIELD_TAG}{ORDER_TAG}.json'))
CACHE_FILE = (
    DATA_DIR / f'004dd_chromatic_phase_{FIELD_TAG}{ORDER_TAG}_{args.start}.npz')

CASES = {
    'bare': dict(with_solenoids=False, with_correctors=False),
    'corrected': dict(with_solenoids=True, with_correctors=True),
}
CASE_STYLE = {
    'bare': dict(color='tab:gray', lw=1.0),
    'corrected': dict(color='tab:red', lw=1.0),
}
PLANES = ('x', 'y')


def load_line():
    env = xt.load(INPUT_LATTICE_JSON)
    line = env.fccee_p_ring
    line.cycle(args.start)
    return line


def scan_phase_advance(line, deltas):
    """Return s, names and mu_{x,y} arrays of shape (len(deltas), n_rows).

    Closed-orbit guess as in get_nonlinear_chromaticity: the on-momentum
    dispersion at the start of the line, scaled by delta.
    """
    tw_ref = line.twiss4d()
    mu = {plane: np.zeros((len(deltas), len(tw_ref))) for plane in PLANES}

    for ii, delta in enumerate(deltas):
        co_guess = line.particle_ref.copy()
        co_guess.x = delta * tw_ref.dx[0]
        co_guess.px = delta * tw_ref.dpx[0]
        co_guess.y = delta * tw_ref.dy[0]
        co_guess.py = delta * tw_ref.dpy[0]
        tw = line.twiss4d(delta0=delta, co_guess=co_guess)
        assert np.array_equal(tw.name, tw_ref.name)
        mu['x'][ii] = tw.mux
        mu['y'][ii] = tw.muy
        print(f'    delta={delta:+.2e}  qx={tw.qx:.6f}  qy={tw.qy:.6f}')

    return tw_ref.s.copy(), tw_ref.name.copy(), mu


def fit_quadratic(deltas, mu):
    """Fit mu = c0 + c1 delta + c2 delta^2 at every s, vectorised over s.

    Fitted in u = delta / delta_max for conditioning, then rescaled. Returns
    (mu, mu', mu'', rms residual), each of shape (n_rows,).
    """
    delta_max = np.max(np.abs(deltas))
    u = deltas / delta_max
    coef = np.polynomial.polynomial.polyfit(u, mu, 2)
    residual = mu - np.polynomial.polynomial.polyval(u, coef).T
    c0 = coef[0]
    c1 = coef[1] / delta_max
    c2 = coef[2] / delta_max**2
    return c0, c1, 2 * c2, np.sqrt(np.mean(residual**2, axis=0))


###############################
# Scan or load from the cache #
###############################

def straight_edges(names, s_col):
    """Map each IP to (s at start of its straight, s at end of its straight)."""
    def s_of(name):
        return s_col[np.flatnonzero(names == name)[0]]
    return {ip: (s_of(f'end_ds_start_straight_{ip}'),
                 s_of(f'end_straight_start_ds_{ip}'))
            for ip in IP_NAMES}


if args.replot:
    cache = np.load(CACHE_FILE, allow_pickle=False)
    deltas = cache['deltas']
    s = cache['s']
    s_ip = dict(zip(IP_NAMES, cache['s_ip']))
    fits = {
        case: {plane: tuple(cache[f'{case}_{plane}_{kk}'] for kk in range(4))
               for plane in PLANES}
        for case in CASES}
    print(f'Loaded {CACHE_FILE}')
    line = load_line() if args.zoom_ip or 's_straight' not in cache else None
    if 's_straight' in cache:
        s_straight = {ip: tuple(edges)
                      for ip, edges in zip(IP_NAMES, cache['s_straight'])}
    else:  # cache written before the straight edges were stored
        table = line.get_table()
        s_straight = straight_edges(table.name, table.s)
else:
    line = load_line()

    deltas = np.linspace(-args.delta_max, args.delta_max, args.npoints)
    fits = {}
    for case, knobs in CASES.items():
        print(f'Scanning case "{case}" ({args.npoints} deltas)')
        set_lattice_knobs(line, **knobs)
        s, names, mu = scan_phase_advance(line, deltas)
        fits[case] = {plane: fit_quadratic(deltas, mu[plane])
                      for plane in PLANES}

    s_ip = {ip: s[np.flatnonzero(names == ip)[0]] for ip in IP_NAMES}
    s_straight = straight_edges(names, s)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_FILE, deltas=deltas, s=s,
        s_ip=np.array([s_ip[ip] for ip in IP_NAMES]),
        s_straight=np.array([s_straight[ip] for ip in IP_NAMES]),
        **{f'{case}_{plane}_{kk}': fits[case][plane][kk]
           for case in CASES for plane in PLANES for kk in range(4)})
    print(f'Saved {CACHE_FILE}')


##########################
# End-of-ring comparison #
##########################

print()
print('End of ring (primes are derivatives w.r.t. delta; '
      'Q\'\'/4 is detuning.py\'s qx_derivatives[2] = 004c\'s d2q):')
for case in CASES:
    for plane in PLANES:
        q, dq, d2q, res = fits[case][plane]
        print(f"  {case:10s} {plane}:  Q={q[-1]:12.6f}  Q'={dq[-1]:11.4f}  "
              f"Q''={d2q[-1]:13.4f}  Q''/4={d2q[-1] / 4:12.4f}  "
              f"max rms fit residual={np.max(res):.2e}")


#########
# Plots #
#########

STRAIGHT_EDGE_STYLE = dict(color='tab:green', ls='--', lw=0.8, alpha=0.8)


def mark_ips(ax, set_xlim=True):
    """Dashed lines at the IPs (black) and at the straight edges (green)."""
    for ip, s_val in s_ip.items():
        ax.axvline(s_val, color='k', ls='--', lw=0.6, alpha=0.5)
        for s_edge in s_straight[ip]:
            ax.axvline(s_edge, **STRAIGHT_EDGE_STYLE)
    if set_xlim:
        ax.set_xlim(s[0], s[-1])


def label_ips(ax):
    for ip, s_val in s_ip.items():
        ax.text(s_val, 1.02, ip, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8)


ROW_LABELS = (r'$\mu_{p}$', r"$\mu_{p}'$", r"$\mu_{p}''$")
PLOT_TAG = f'{FIELD_TAG}{ORDER_TAG}_{args.start}'

fig1, axs1 = plt.subplots(3, 2, sharex=True, figsize=(12, 9))
for jj, plane in enumerate(PLANES):
    for case in CASES:
        for kk in range(3):
            axs1[kk, jj].plot(s, fits[case][plane][kk], label=case,
                              **CASE_STYLE[case])
    for kk in range(3):
        axs1[kk, jj].set_ylabel(ROW_LABELS[kk].replace('p', plane))
        mark_ips(axs1[kk, jj])
    label_ips(axs1[0, jj])
    axs1[-1, jj].set_xlabel('s [m]')
axs1[0, 0].legend(loc='upper left')
fig1.suptitle(f'Phase advance vs delta, {FIELD_TAG}{ORDER_TAG} '
              f'(from {args.start}, |delta| <= {np.max(np.abs(deltas)):.0e}; '
              f'black: IPs, green: straight edges)')
fig1.tight_layout()
fig1.savefig(PLOT_DIR / f'004dd_chromatic_phase_{PLOT_TAG}.png', dpi=200)

fig2, axs2 = plt.subplots(3, 2, sharex=True, figsize=(12, 9))
for jj, plane in enumerate(PLANES):
    for kk in (1, 2):
        axs2[kk - 1, jj].plot(
            s, fits['corrected'][plane][kk] - fits['bare'][plane][kk],
            color='tab:blue', lw=1.0)
        axs2[kk - 1, jj].set_ylabel(
            r'$\Delta$' + ROW_LABELS[kk].replace('p', plane)
            + ' (corrected - bare)')
    for case in CASES:
        axs2[2, jj].semilogy(s, fits[case][plane][3], label=case,
                             **CASE_STYLE[case])
    axs2[2, jj].set_ylabel(f'rms fit residual $\\mu_{plane}$')
    for kk in range(3):
        mark_ips(axs2[kk, jj])
    label_ips(axs2[0, jj])
    axs2[-1, jj].set_xlabel('s [m]')
axs2[2, 0].legend(loc='upper left')
fig2.suptitle(f'Corrected - bare chromatic phase advance, {FIELD_TAG}{ORDER_TAG} '
              f'(black: IPs, green: straight edges)')
fig2.tight_layout()
fig2.savefig(PLOT_DIR / f'004dd_chromatic_phase_diff_{PLOT_TAG}.png', dpi=200)


#####################################################
# Zoom on an IP against the magnet structure (tw.plot) #
#####################################################

# The fitted functions are attached as extra columns to an on-momentum twiss
# of the corrected ring, so tw.plot draws them over its lattice strip. The
# rows line up because the scan twisses used the same line and start point.
# Column names avoid 'dmux'/'dmuy', which twiss already uses for its own
# chromatic functions. mu{p}_d1 / mu{p}_d2 are mu' / mu'' of the corrected
# ring, the *_bare columns those of the bare ring, and *_corr_minus_bare their
# difference (the top rows of figure 2).
if args.zoom_ip:
    set_lattice_knobs(line, **CASES['corrected'])
    tw = line.twiss4d(strengths=True)
    assert len(tw) == len(s) and np.allclose(tw.s, s)

    for plane in PLANES:
        for kk, order in ((1, 'd1'), (2, 'd2')):
            col = f'mu{plane}_{order}'
            tw[col] = fits['corrected'][plane][kk]
            tw[f'{col}_bare'] = fits['bare'][plane][kk]
            tw[f'{col}_corr_minus_bare'] = (
                fits['corrected'][plane][kk] - fits['bare'][plane][kk])
            prime = "'" * kk
            TwissPlot = xt.twissplot.TwissPlot
            TwissPlot.lglabel[f'{col}_corr_minus_bare'] = (
                rf"$\Delta\mu_{plane}{prime}$ (corr - bare)")
            TwissPlot.axlabel[f'{col}_corr_minus_bare'] = (
                rf"$\Delta\mu_{plane}{prime}$")

    for ip in args.zoom_ip:
        s_lo = s_ip[ip] - args.zoom_half_width
        s_hi = s_ip[ip] + args.zoom_half_width
        if s_lo < s[0] or s_hi > s[-1]:
            print(f'NOTE: zoom window around {ip} is cut at the ring start/end '
                  f'(start is {args.start}).')
        # Slice rows rather than pass mask=: TwissPlot masks the strengths
        # but not s when drawing the lattice bars, so with a mask the magnets
        # land outside the window.
        tw_window = tw.rows[s_lo:s_hi:'s']
        for plane in PLANES:
            pl = tw_window.plot(
                yl=f'mu{plane}_d2_corr_minus_bare',
                yr=f'mu{plane}_d1_corr_minus_bare',
                figlabel=f'004dd {ip} {plane}', figsize=(12, 5))
            xlim = pl.ax.get_xlim()
            mark_ips(pl.ax, set_xlim=False)
            pl.ax.set_xlim(xlim)
            pl.ax.set_title(f'{ip}, {plane}: corrected - bare, '
                            f'{FIELD_TAG}{ORDER_TAG} (black: IP, '
                            f'green: straight edges)')
            pl.figure.savefig(
                PLOT_DIR / f'004dd_chromatic_phase_zoom_{ip}_{plane}_{PLOT_TAG}.png',
                dpi=200, bbox_inches='tight')

plt.show()
