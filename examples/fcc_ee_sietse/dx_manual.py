"""Manual horizontal dispersion D_x = dx/dδ at IP markers via CO + one-turn track."""

from __future__ import annotations

import warnings

import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
from xtrack.twiss import ClosedOrbitSearchError

IPS = ['ip.0', 'ip.2', 'ip.4', 'ip.6']


def measure_dx_manual(
    line,
    delta_vals,
    co_guess0,
    ips=None,
    delta0_twiss=None,
    twiss_method='4d',
):
    """D_x, D_x' at each IP from quadratic fit x(δ) after CO + one-turn track.

    Fit: x(δ) = D_x' δ² + D_x δ + c₀; linear coeff is D_x, quadratic is D_x'.

    Parameters
    ----------
    line : Line
        Steered ring with tracker built; time-dependent vars must be off.
    delta_vals : array-like
        Momentum offsets for the scan.
    co_guess0 : Particles
        Warm-start CO guess (e.g. twiss particle_on_co); ``delta`` is updated per point.
    ips : list of str, optional
        Marker names for ``multi_element_monitor_at`` (default :data:`IPS`).
    delta0_twiss : float, optional
        ``delta0`` passed to ``line.twiss`` for reference ``dx``; default is
        ``delta_vals[len(delta_vals) // 2]``.

    Returns
    -------
    dict with keys ``dx_manual``, ``dx_prime_manual``, ``dx_twiss``, ``x_at_ip``,
    ``delta_vals``, ``fit_coeff``, ``fit_r2``.
    """
    if ips is None:
        ips = IPS
    delta_vals = np.asarray(delta_vals, dtype=float)
    if delta0_twiss is None:
        delta0_twiss = float(delta_vals[len(delta_vals) // 2])

    def _co_at_delta(d, p_guess):
        """CO at fixed momentum offset δ (dispersion slope requires fixed δ)."""
        pg = p_guess.copy()
        pg.delta = float(d)
        try:
            return line.find_closed_orbit(co_guess=pg, delta0=d)
        except ClosedOrbitSearchError:
            try:
                return line.twiss(
                    method=twiss_method, delta0=d,
                ).particle_on_co.copy()
            except ClosedOrbitSearchError:
                return None

    # Scan from anchor δ outward so each CO guess is close to the previous one.
    order = np.argsort(np.abs(delta_vals - delta0_twiss))
    co_list = []
    delta_ok = []
    p_guess = co_guess0.copy()
    for ii in order:
        d = float(delta_vals[ii])
        p_co = _co_at_delta(d, p_guess)
        if p_co is None:
            warnings.warn(f'No CO at delta={d:.6e}; skipping scan point')
            continue
        co_list.append(p_co)
        delta_ok.append(d)
        p_guess = p_co.copy()

    if len(co_list) < 3:
        raise ClosedOrbitSearchError(
            f'Fewer than 3 CO points in delta scan (got {len(co_list)})')
    delta_vals = np.asarray(delta_ok, dtype=float)

    p_batch = xt.Particles.merge(co_list)
    line.track(p_batch, num_turns=1, multi_element_monitor_at=list(ips))
    mon = line.record_multi_element_last_track
    ctx2np = line._context.nparray_from_context_array

    x_at_ip = {}
    dx_manual = {}
    dx_prime_manual = {}
    fit_coeff = {}
    fit_r2 = {}
    for ip in ips:
        x_ip = ctx2np(mon.get('x', obs_name=ip, turn=0))
        x_at_ip[ip] = x_ip
        coeff = np.polyfit(delta_vals, x_ip, 2)
        fit_coeff[ip] = coeff
        dx_prime_manual[ip] = coeff[0]
        dx_manual[ip] = coeff[1]
        y_fit = np.polyval(coeff, delta_vals)
        ss_res = np.sum((x_ip - y_fit) ** 2)
        ss_tot = np.sum((x_ip - np.mean(x_ip)) ** 2)
        fit_r2[ip] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    tw_ref = line.twiss(method=twiss_method, delta0=delta0_twiss)
    dx_twiss = {ip: float(tw_ref['dx', ip]) for ip in ips}

    return {
        'dx_manual': dx_manual,
        'dx_prime_manual': dx_prime_manual,
        'dx_twiss': dx_twiss,
        'x_at_ip': x_at_ip,
        'delta_vals': delta_vals,
        'fit_coeff': fit_coeff,
        'fit_r2': fit_r2,
    }


def print_dx_comparison(
    dx_manual,
    dx_twiss,
    ips=None,
    df_hz=None,
    dx_prime_manual=None,
):
    if ips is None:
        ips = list(dx_manual.keys())
    if dx_prime_manual is not None:
        hdr = ('ip          Dx_manual [m]   Dx_prime [m]   Dx_twiss [m]   '
               'rel_err')
    else:
        hdr = 'ip          Dx_manual [m]   Dx_twiss [m]   rel_err'
    if df_hz is not None:
        print(f'\nManual D_x vs twiss (df_hz = {df_hz:.0f} Hz)')
    else:
        print('\nManual D_x vs twiss')
    print(hdr)
    for ip in ips:
        dm = dx_manual[ip]
        dt = dx_twiss[ip]
        denom = max(abs(dt), 1e-12)
        if dx_prime_manual is not None:
            dp = dx_prime_manual[ip]
            print(
                f'{ip:12s}  {dm:14.6e}  {dp:14.6e}  {dt:14.6e}  '
                f'{abs(dm - dt) / denom:.3e}'
            )
        else:
            print(
                f'{ip:12s}  {dm:14.6e}  {dt:14.6e}  '
                f'{abs(dm - dt) / denom:.3e}'
            )


def plot_dx_scan(
    delta_vals,
    x_at_ip,
    dx_manual,
    df_hz,
    ips=None,
    dx_prime_manual=None,
    fit_coeff=None,
    figsize=(8, 7),
):
    if ips is None:
        ips = list(x_at_ip.keys())
    delta_vals = np.asarray(delta_vals, dtype=float)
    delta_fine = np.linspace(delta_vals.min(), delta_vals.max(), 100)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    for ax, ip in zip(axes.flat, ips):
        x_ip = x_at_ip[ip]
        ax.plot(delta_vals, x_ip, 'o', ms=4)
        if fit_coeff is not None and ip in fit_coeff:
            coeff = fit_coeff[ip]
        else:
            coeff = np.polyfit(delta_vals, x_ip, 2)
        ax.plot(delta_fine, np.polyval(coeff, delta_fine), '-', lw=1.5)
        dp = coeff[0]
        ax.set_title(
            f'{ip}  $D_x$ = {dx_manual[ip]:.3e} m, '
            f"$D_x'$ = {dp:.3e} m"
        )
        ax.set_xlabel(r'$\delta$')
        ax.set_ylabel(r'$x$ [m]')
    fig.suptitle(
        f'Manual $D_x$, $D_x\'$ scan: $\\Delta f$ = {df_hz:.0f} Hz'
    )
    fig.tight_layout()
    return fig
