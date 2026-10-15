"""Where does the undulator's linear focusing come from? Term-by-term
decomposition of K_x, K_y on the wiggling closed orbit, SB vs MK.

Exact split (differences, not Taylor terms), with the gradients evaluated at
the points named (lab coordinates; the magnet axis sits at x = dx, y = 0):

    term                    B rho K_x                          B rho K_y
    (a) axis gradient       dBy/dx(axis)                       -dBx/dy(axis)
    (b) feed-down from x_w  dBy/dx(x_w, 0) - dBy/dx(axis)      -[dBx/dy(x_w, 0) - dBx/dy(axis)]
    (c) feed-down from y_w  dBy/dx(x_w, y_w) - dBy/dx(x_w, 0)  -[dBx/dy(x_w, y_w) - dBx/dy(x_w, 0)]
    (d) B_s on the wiggle   -y_w' dBs/dx(orbit)                +x_w' dBs/dy(orbit)

(a)+(b)+(c)+(d) is the full K of 007a's field_focusing(). To first order (b)
and (c) are x_w d2By/dx2 and y_w d2Bx/dy2, which focus one plane and defocus
the other equally. Maxwell fixes the sum,
    int (K_x + K_y) ds = int (B_x^2 + B_y^2) / (B rho)^2 ds,
and (d) alone integrates to int B_x^2/(B rho)^2 (x) and int B_y^2/(B rho)^2 (y)
-- both checked below. Derivation and discussion:
claude_notes/undulator_focusing_writeup.md.

MK (to_multipole_line(), 2D thick multipoles) has (a) = K1, (b) = K2 x,
(c) = -K2_skew y and no (d); its thin MultipoleEdge kicks are not included
in the decomposition (they are in its one-turn-matrix tune shift).

Each term is reported as int K ds and as its tune-shift contribution
(1/4 pi) int K beta_0 ds, next to the exact tune shift from the one-turn
matrix (bare ring + undulator transfer matrix, as in 007b). A summary, a bar
chart of the terms and a plot of the on-axis field curvature (their origin)
follow.
"""
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xtrack as xt
from scipy.integrate import cumulative_trapezoid
from xtrack import linear_normal_form as lnf
from xtrack._temp.splineboris import TubeFitter, LongitudinalFitter

PLACE = 'ars11_uind_0610_1'
OFFSETS = [-0.4e-3, 0., 0.4e-3]   # undulator shift_x values [m]

multipole_order = 3
E0 = 2.7e9
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)
FIELD_TOL = dict(field_tol=1e-4, r_ref=2e-3)

ROOT = Path(__file__).resolve().parent
madx_file = ROOT.parent.parent / 'test_data' / 'sls' / 'sls.madx'
field_map_file = ROOT.parent.parent / 'test_data' / 'sls' / 'simona_field_map.txt'
# The fitted LongitudinalFitter is small (~0.4 MB) -- cached so that only the
# first run has to load and fit the ~1.8 GB field map. Same fit settings as
# 007a; delete the file after changing them.
FIT_CACHE = ROOT / 'data' / 'undulator_longitudinal_fit.pkl'
OUT_DIR = Path(os.environ.get(
    'SLS_UNDULATOR_PLOT_DIR',
    Path.home() / 'cernbox' / 'Pictures' / 'SLS_Undulator_Studies'))
OUT_DIR.mkdir(parents=True, exist_ok=True)
K_U = 2 * np.pi / 0.036   # undulator wavenumber [1/m] (period 36 mm)


def get_undulator_fit():
    if FIT_CACHE.exists():
        return pickle.loads(FIT_CACHE.read_bytes())
    print("[TubeFitter] loading raw field map and fitting...")
    df_raw_data = pd.read_csv(
        field_map_file, sep="\t", header=None,
        names=["X", "Y", "Z", "Bx", "By", "Bs"], dtype=float,
    ).set_index(["X", "Y", "Z"])
    tf = TubeFitter(raw_data=df_raw_data, n_frames=4441, distance_unit=0.001,
                    deg=multipole_order - 1)
    tf.fit()
    z, F, names = tf.on_axis_multipoles()
    lf = LongitudinalFitter(z[0], z[-1], end_condition="free", period=0.036)
    lf.fit(z, F, names)
    lf.fit(*tf.on_axis_bs(), [("Bs", 0)])
    FIT_CACHE.parent.mkdir(exist_ok=True)
    FIT_CACHE.write_bytes(pickle.dumps(lf))
    return lf


def build_undulator(lf, model_label):
    """Undulator line with four orbit correctors, as in 007a."""
    und_env = xt.Environment()
    und_env.particle_ref = p0.copy()
    if model_label == 'SB':
        line = lf.to_line(multipole_order=multipole_order, **FIELD_TOL)
    else:
        line = lf.to_multipole_line(multipole_order=multipole_order, p0c=E0,
                                    field_at='mean', **FIELD_TOL)
    undulator = und_env.import_line(line, line_name='undulator')
    l_wig = undulator.get_length()
    corrector_vars = []
    for i in range(1, 5):
        for kind in ('k0l', 'k0sl'):
            und_env[f'{kind}_corr{i}'] = 0.
            corrector_vars.append(f'{kind}_corr{i}')
        und_env.new(f'corr{i}', xt.Multipole, knl=[f'k0l_corr{i}'],
                    ksl=[f'k0sl_corr{i}'])
    undulator.insert([
        und_env.place('corr1', at=0.02),
        und_env.place('corr2', at=0.1),
        und_env.place('corr3', at=l_wig - 0.1),
        und_env.place('corr4', at=l_wig - 0.02),
    ], s_tol=5e-3)
    field_names = [nn for nn in undulator.element_names
                   if nn.startswith(('splineboris_', 'multipole_'))]

    def set_offset(dx):
        for nn in field_names:
            undulator[nn].shift_x = dx
        undulator.match(
            solve=False, betx=0, bety=0, only_orbit=True,
            include_collective=True,
            vary=xt.VaryList(corrector_vars, step=1e-6),
            targets=[xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.START),
                     xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END)],
        ).step(2)

    return undulator, field_names, set_offset


TERMS = ('a', 'b', 'c', 'd')
TERM_NAMES = {'a': '(a) axis gradient',
              'b': '(b) feed-down x_w',
              'c': '(c) feed-down y_w',
              'd': '(d) B_s on wiggle'}


def decompose_sb(undulator, field_names, tw, dx, brho0, n_sub=20):
    """Per term and plane: K(s) [1/m^2] on n_sub + 1 points per element,
    plus the field on the orbit (for the B^2 identities) and the field and
    its transverse curvature on the magnet axis. Orbit through each element
    integrated from its Twiss entrance values, as in 007a's
    field_focusing()."""
    d = 1e-4
    tab = undulator.get_table()
    u = np.linspace(0, 1, n_sub + 1)
    out = {(t, p): [] for t in TERMS for p in 'xy'}
    s_all, Bx_all, By_all = [], [], []
    axis = {kk: [] for kk in ('Bx', 'By', 'dxx_By', 'dyy_By', 'dss_By',
                              'dxx_Bx', 'dyy_Bx', 'dss_Bx')}
    for nn in field_names:
        el = undulator[nn]
        s_loc = el.length * u
        s_glob = tab['s', nn] + s_loc
        Bx, By, _ = el.get_field(np.interp(s_glob, tw.s, tw.x),
                                 np.interp(s_glob, tw.s, tw.y), s_loc)
        xp = np.interp(s_glob[0], tw.s, tw.px) - cumulative_trapezoid(By, s_loc, initial=0) / brho0
        yp = np.interp(s_glob[0], tw.s, tw.py) + cumulative_trapezoid(Bx, s_loc, initial=0) / brho0
        x = np.interp(s_glob[0], tw.s, tw.x) + cumulative_trapezoid(xp, s_loc, initial=0)
        y = np.interp(s_glob[0], tw.s, tw.y) + cumulative_trapezoid(yp, s_loc, initial=0)
        Bx, By, _ = el.get_field(x, y, s_loc)   # on the integrated orbit

        def grad(xx, yy):
            """(dBy/dx, dBx/dy, dBs/dx, dBs/dy) at (xx, yy)."""
            fxp = np.array(el.get_field(xx + d, yy, s_loc))
            fxm = np.array(el.get_field(xx - d, yy, s_loc))
            fyp = np.array(el.get_field(xx, yy + d, s_loc))
            fym = np.array(el.get_field(xx, yy - d, s_loc))
            gx = (fxp - fxm) / (2 * d)
            gy = (fyp - fym) / (2 * d)
            return gx[1], gy[0], gx[2], gy[2]

        zero = np.zeros_like(s_loc)
        # field and curvature on the magnet axis (second differences; the
        # field is polynomial in x, y, so these are exact up to round-off)
        f0 = np.array(el.get_field(dx + zero, zero, s_loc))
        fxx = (np.array(el.get_field(dx + d + zero, zero, s_loc)) - 2 * f0
               + np.array(el.get_field(dx - d + zero, zero, s_loc))) / d**2
        fyy = (np.array(el.get_field(dx + zero, zero + d, s_loc)) - 2 * f0
               + np.array(el.get_field(dx + zero, zero - d, s_loc))) / d**2
        # d2/ds2 inside the element (points within ds of an edge use the
        # nearest interior stencil)
        ds = 1e-4
        s_c = np.clip(s_loc, ds, el.length - ds)
        fss = (np.array(el.get_field(dx + zero, zero, s_c + ds))
               - 2 * np.array(el.get_field(dx + zero, zero, s_c))
               + np.array(el.get_field(dx + zero, zero, s_c - ds))) / ds**2
        axis['dss_By'].append(fss[1])
        axis['dss_Bx'].append(fss[0])
        axis['Bx'].append(f0[0])
        axis['By'].append(f0[1])
        axis['dxx_By'].append(fxx[1])
        axis['dyy_By'].append(fyy[1])
        axis['dxx_Bx'].append(fxx[0])
        axis['dyy_Bx'].append(fyy[0])
        g_axis = grad(dx + zero, zero)
        g_xw = grad(x, zero)
        g_orb = grad(x, y)
        out[('a', 'x')].append(g_axis[0] / brho0)
        out[('b', 'x')].append((g_xw[0] - g_axis[0]) / brho0)
        out[('c', 'x')].append((g_orb[0] - g_xw[0]) / brho0)
        out[('d', 'x')].append(-yp * g_orb[2] / brho0)
        out[('a', 'y')].append(-g_axis[1] / brho0)
        out[('b', 'y')].append(-(g_xw[1] - g_axis[1]) / brho0)
        out[('c', 'y')].append(-(g_orb[1] - g_xw[1]) / brho0)
        out[('d', 'y')].append(xp * g_orb[3] / brho0)
        s_all.append(s_glob)
        Bx_all.append(Bx)
        By_all.append(By)
    return (np.concatenate(s_all),
            {k: np.concatenate(v) for k, v in out.items()},
            np.concatenate(Bx_all), np.concatenate(By_all),
            {k: np.concatenate(v) for k, v in axis.items()})


def decompose_mk(undulator, field_names, tw, dx):
    """Same terms for the thick multipoles: K constant per region, orbit
    position = mean of the region's entry/exit Twiss orbit."""
    tab = undulator.get_table()
    names = [nn for nn in field_names
             if not isinstance(undulator[nn], xt.MultipoleEdge)]
    s, length, out = [], [], {(t, p): [] for t in TERMS for p in 'xy'}
    for nn in names:
        el = undulator[nn]
        s0 = tab['s', nn]
        x = np.interp([s0, s0 + el.length], tw.s, tw.x).mean() - dx
        y = np.interp([s0, s0 + el.length], tw.s, tw.y).mean()
        K1 = el.knl[1] / el.length
        K2 = el.knl[2] / el.length
        K2s = el.ksl[2] / el.length
        # normal: dBy/dx = brho (K1 + K2 x - K2s y); skew parts only couple
        for p, sgn in (('x', 1), ('y', -1)):
            out[('a', p)].append(sgn * K1)
            out[('b', p)].append(sgn * K2 * x)
            out[('c', p)].append(-sgn * K2s * y)
            out[('d', p)].append(0.)
        s.append(s0 + el.length / 2)
        length.append(el.length)
    return np.array(s), np.array(length), {k: np.array(v) for k, v in out.items()}


def main():
    lf = get_undulator_fit()
    brho0 = p0.rigidity0[0]

    line_sls = xt.load(str(madx_file)).ring
    line_sls.configure_bend_model(core='mat-kick-mat')
    line_sls.particle_ref = p0.copy()
    tw0 = line_sls.twiss4d()
    ic = {kk: float(tw0[kk, PLACE]) for kk in ('betx', 'alfx', 'bety', 'alfy')}

    # Bare optics inside the undulator span (a drift: zero-strength
    # placeholders only), and the bare one-turn matrix at its entrance.
    def beta0(plane, s_loc):
        b, a = ic[f'bet{plane}'], ic[f'alf{plane}']
        return b - 2 * a * s_loc + (1 + a**2) / b * s_loc**2

    row = int(np.where(tw0.name == PLACE)[0][0])
    W = tw0.W_matrix[row][:4, :4]
    rot = np.zeros((4, 4))
    rot[:2, :2] = lnf.Rot2D(2 * np.pi * tw0.qx)
    rot[2:, 2:] = lnf.Rot2D(2 * np.pi * tw0.qy)
    M_bare = W @ rot @ np.linalg.inv(W)
    qx_f, qy_f = tw0.qx % 1, tw0.qy % 1

    def exact_tune_shift(tw, l_wig):
        D_inv = np.eye(4)
        D_inv[0, 1] = D_inv[2, 3] = -l_wig
        M = M_bare @ D_inv @ tw.get_R_matrix(0, len(tw.name) - 1)[:4, :4]
        q = np.abs(np.angle(np.linalg.eigvals(M))) / (2 * np.pi)
        return (q[np.argmin(np.abs(q - qx_f))] - qx_f,
                q[np.argmin(np.abs(q - qy_f))] - qy_f)

    results = {}
    for model_label in ('SB', 'MK'):
        undulator, field_names, set_offset = build_undulator(lf, model_label)
        l_wig = undulator.get_length()
        for dx in OFFSETS:
            set_offset(dx)
            tw = undulator.twiss4d(include_collective=True, **ic)
            res = {'exact': exact_tune_shift(tw, l_wig)}
            if model_label == 'SB':
                s, K, Bx, By, axis = decompose_sb(undulator, field_names, tw, dx, brho0)
                res['s'], res['axis'] = s, axis
                # elements are sampled separately -> integrate per element
                # (the shared end points have zero-width gaps between them)
                n = len(s) // len(field_names)
                integ = lambda f: sum(np.trapezoid(f[i:i + n], s[i:i + n])
                                      for i in range(0, len(s), n))
                res['Bx2'] = integ(Bx**2) / brho0**2
                res['By2'] = integ(By**2) / brho0**2
            else:
                s, length, K = decompose_mk(undulator, field_names, tw, dx)
                integ = lambda f: np.sum(f * length)
            for (t, p), k in K.items():
                res[('intK', t, p)] = integ(k)
                res[('dQ', t, p)] = integ(k * beta0(p, s)) / (4 * np.pi)
            results[(model_label, dx)] = res

    # ---- report ---------------------------------------------------------
    def fmt(v):
        return f'{v:+.4e}'

    for dx in OFFSETS:
        print()
        print('=' * 96)
        print(f'{PLACE}, dx = {dx * 1e3:+.2f} mm')
        print('=' * 96)
        for quantity, title in (('intK', 'int K ds [1/m]'),
                                ('dQ', 'tune-shift contribution (1/4pi) int K beta_0 ds')):
            print(f'\n{title}')
            print(f'{"term":24s}' + ''.join(f'{h:>14s}' for h in (
                'SB x', 'MK x', 'SB-MK x', 'SB y', 'MK y', 'SB-MK y')))
            sb, mk = results[('SB', dx)], results[('MK', dx)]
            for t in TERMS + ('sum',):
                vals = []
                for p in 'xy':
                    if t == 'sum':
                        a = sum(sb[(quantity, tt, p)] for tt in TERMS)
                        b = sum(mk[(quantity, tt, p)] for tt in TERMS)
                    else:
                        a, b = sb[(quantity, t, p)], mk[(quantity, t, p)]
                    vals += [a, b, a - b]
                name = TERM_NAMES.get(t, 'sum (a)+(b)+(c)+(d)')
                print(f'{name:24s}' + ''.join(f'{fmt(v):>14s}' for v in vals))
            if quantity == 'dQ':
                vals = []
                for i in range(2):
                    a, b = sb['exact'][i], mk['exact'][i]
                    vals += [a, b, a - b]
                print(f'{"exact (one-turn matrix)":24s}'
                      + ''.join(f'{fmt(v):>14s}' for v in vals))
        sb = results[('SB', dx)]
        dx_int = sb[('intK', 'd', 'x')]
        dy_int = sb[('intK', 'd', 'y')]
        sum_xy = sum(sb[('intK', t, p)] for t in TERMS for p in 'xy')
        print('\nSB identities (int K ds):')
        print(f'  (d)_x = {fmt(dx_int)}   vs  int Bx^2/Brho^2 = {fmt(sb["Bx2"])}')
        print(f'  (d)_y = {fmt(dy_int)}   vs  int By^2/Brho^2 = {fmt(sb["By2"])}')
        print(f'  int (K_x + K_y) = {fmt(sum_xy)}   vs  '
              f'int (Bx^2 + By^2)/Brho^2 = {fmt(sb["Bx2"] + sb["By2"])}')

    summarize(results)
    plot_decomposition(results)
    plot_curvature(results)
    plt.show()


def summarize(results):
    """The finding in words: each field component's natural focusing lands
    in the plane of the wiggle it drives (B_y -> x, B_x -> y), because of
    the map's transverse curvature (terms b, c), not in the perpendicular
    plane as for a flat-pole planar undulator."""
    sb, mk = results[('SB', 0.)], results[('MK', 0.)]
    By2, Bx2 = sb['By2'], sb['Bx2']
    I = lambda r, t, p: r[('intK', t, p)]
    print()
    print('=' * 96)
    print('Summary (SB, dx = 0): where does the natural focusing land?')
    print('=' * 96)
    print(f'  int By^2/Brho^2 = {By2:.3e} 1/m,  int Bx^2/Brho^2 = {Bx2:.3e} 1/m')
    print(f'  B_y part:  (d)_y  = {I(sb, "d", "y") / By2:+.2f} x int By^2   '
          f'(B_s on the horizontal wiggle: focuses y, as in a planar undulator)')
    print(f'             (b)_x  = {I(sb, "b", "x") / By2:+.2f} x int By^2   '
          f'(x_w d2By/dx2: focuses x and defocuses y by as much)')
    print(f'             -> net B_y focusing: x {I(sb, "b", "x") / By2:+.2f}, '
          f'y {(I(sb, "d", "y") + I(sb, "b", "y")) / By2:+.2f}  (x int By^2)')
    print(f'  B_x part:  (d)_x  = {I(sb, "d", "x") / Bx2:+.2f} x int Bx^2   '
          f'(B_s on the vertical wiggle: focuses x)')
    print(f'             (c)_y  = {I(sb, "c", "y") / Bx2:+.2f} x int Bx^2   '
          f'(y_w d2Bx/dy2: focuses y and defocuses x by as much)')
    print(f'             -> net B_x focusing: x {(I(sb, "d", "x") + I(sb, "c", "x")) / Bx2:+.2f}, '
          f'y {I(sb, "c", "y") / Bx2:+.2f}  (x int Bx^2)')
    ax = sb['axis']
    print('  Origin -- transverse curvature on axis, int B d2B / (k_u^2 int B^2):')
    for comp, own, other, flat in (('B_y', 'dxx', 'dyy', '0 / +1'),
                                   ('B_x', 'dyy', 'dxx', '0 / +1 (rotated)')):
        c = comp.replace('_', '')
        B = ax[c]
        r = {kk: np.sum(B * ax[f'{kk}_{c}']) / (K_U**2 * np.sum(B**2))
             for kk in ('dxx', 'dyy', 'dss')}
        print(f'    {comp}: {own} {r[own]:+.2f},  {other} {r[other]:+.2f}   '
              f'(flat-pole planar: {flat});  dss {r["dss"]:+.2f};  '
              f'Laplace dxx+dyy+dss = {r["dxx"] + r["dyy"] + r["dss"]:+.3f}')
    print(f'  MK has (b) and (c) from its 2D sextupole but no (d): net y '
          f'{sum(I(mk, t, "y") for t in TERMS):+.3e} vs SB '
          f'{sum(I(sb, t, "y") for t in TERMS):+.3e} 1/m '
          f'-> its large vertical tune shift / beta beat.')


def plot_decomposition(results):
    """Bar chart of int K ds per term, SB vs MK, per plane, at dx = 0, with
    the B^2 integrals as reference lines."""
    sb, mk = results[('SB', 0.)], results[('MK', 0.)]
    By2, Bx2 = sb['By2'], sb['Bx2']
    labels = ['(b) $x_w\\,\\partial_x^2 B_y$', '(c) $y_w\\,\\partial_y^2 B_x$',
              '(d) $B_s$ on wiggle', 'sum']
    keys = ['b', 'c', 'd', 'sum']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    w = 0.38
    for ax, p in zip(axes, 'xy'):
        for i, (r, name, color) in enumerate(((sb, 'SplineBoris', 'tab:blue'),
                                              (mk, 'Multipole kick', 'tab:green'))):
            vals = [sum(r[('intK', t, p)] for t in TERMS) if k == 'sum'
                    else r[('intK', k, p)] for k in keys]
            xs = np.arange(len(keys)) + (i - 0.5) * w
            ax.bar(xs, np.array(vals) * 1e3, w, color=color, label=name)
            print(f'[decomposition plot] {name} {p}: ' + ', '.join(
                f'{k} {v:+.3e}' for k, v in zip(keys, vals)))
        for val, ls, lab in ((By2, '--', '$\\pm\\int B_y^2/(B\\rho)^2$'),
                             (Bx2, ':', '$\\pm\\int B_x^2/(B\\rho)^2$')):
            for sgn in (1, -1):
                ax.axhline(sgn * val * 1e3, color='0.4', linestyle=ls,
                           linewidth=1, label=lab if sgn == 1 else None)
        ax.axhline(0, color='k', linewidth=0.8)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f'$\\int K_{p}\\,ds$')
        ax.grid(True, axis='y', alpha=0.3)
    axes[0].set_ylabel('$\\int K\\,ds$ [$10^{-3}$ m$^{-1}$]')
    axes[1].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'{PLACE}: linear focusing on the wiggling orbit by term '
                 '($\\delta x = 0$; (a) axis gradient is zero)')
    fig.tight_layout()
    out_path = OUT_DIR / f'{PLACE}_focusing_decomposition.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')
    return fig


def plot_curvature(results):
    """The origin: on-axis transverse curvature of each field component next
    to k_u^2 B, over a few periods in the middle of the undulator. A
    flat-pole planar undulator has d2By/dx2 = 0, d2By/dy2 = k_u^2 By."""
    sb = results[('SB', 0.)]
    s, ax_ = sb['s'], sb['axis']
    mid = 0.5 * (s.min() + s.max())
    m = np.abs(s - mid) < 2 * 0.036
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, comp, own, other in (
            (axes[0], 'By', 'dxx_By', 'dyy_By'),
            (axes[1], 'Bx', 'dyy_Bx', 'dxx_Bx')):
        c = comp[1]
        ax.plot(s[m], K_U**2 * ax_[comp][m], color='k', label=f'$k_u^2 B_{c}$')
        ax.plot(s[m], ax_[own][m], color='tab:red',
                label=f'$\\partial_{own[1]}^2 B_{c}$ (plane of its own wiggle)')
        ax.plot(s[m], ax_[other][m], color='tab:blue',
                label=f'$\\partial_{other[1]}^2 B_{c}$')
        ax.set_ylabel('[T/m$^2$]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    axes[0].set_title('On-axis transverse curvature of the field (SplineBoris, middle of the undulator)')
    axes[1].set_xlabel('s [m]')
    fig.tight_layout()
    out_path = OUT_DIR / f'{PLACE}_field_curvature.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved {out_path}')
    return fig


if __name__ == '__main__':
    main()
