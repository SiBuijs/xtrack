"""
Move the start (inner edge) of the IDEA screening solenoid and rebuild the
screening and total on-axis fields on the 0.1 m spline grid.

Same data, ripple removal and main-solenoid fit as 001d (copied here, 001e is
standalone). Notation as in 001d: G = main solenoid, S = screening *as
plotted* (positive; its field is -S), total = G - S; on the solenoid axis,
mirror-symmetric about the IP.

Edge: the point where S/G = 0.5, i.e. where the beam sees half of the main
field. In the data it is at z_e0 = 1.404 m (half of max S: 1.397 m).

Field strength: MAIN_FIELD sets the main solenoid, and the screening
follows it (S = G * fraction). Main, screening and total all scale by
MAIN_FIELD / OPERA_FIELD, and the edges stay where they are. The scaling is
applied to the digitized curves when they are loaded. Every step after that
is linear in the field (the exponential tail of B keeps its decay length),
so MAIN_FIELD = OPERA_FIELD reproduces the data. At 2 T the fitted G(0) is
2.0014 T, within the digitization (1 px = 2.2 mT).
The scaling assumes the iron does not saturate. The Opera model has iron,
so at a different field the real shape (e.g. G's fast fall-off at 2.5-4 m)
may change.

For each edge z_e in SCREENING_EDGES:
1. Ripple removal as 001d step 1 (B_clean), G fitted as 001d step 2
   (0.2 m elements) and projected onto the 0.1 m grid of the edge (see
   Choices). On the grid with nodes at multiples of 0.1 m the projection is
   exact (the 0.2 m nodes are a subset). On the grid offset by 0.05 m the
   error is <= 16 uT, with G = 0 beyond +-S_END.
2. Screening fraction r(z) = S/G = 1 - B_clean/G, split into the two edges
   of the symmetric pair, r(z) = rho(z - z_e0) + rho(-z - z_e0). rho is the
   data for u >= -z_e0. Beyond that (one edge's inner tail seen past the IP)
   it decays exponentially, with an e-folding length fitted
   self-consistently on INNER_TAIL_WINDOW.
3. Target S_target(z) = G(z) [rho(z - z_e) + rho(-z - z_e)].
   What moves is the screening *fraction*, not S: past its edge S follows
   G, which falls by up to ~1 T/m at 3-4 m, so shifting S itself by 0.2 m
   would leave up to 0.28 T uncancelled. With the fraction, the height of
   the edge follows the local G and S never exceeds G.
4. Match point z_m = z_e + MATCH_DISTANCE, rounded up to a node; it moves
   with the edge. For |z| >= z_m, S = G exactly: the B-spline coefficients
   whose support reaches past z_m are G's. The total is identically zero
   there, so its elements can stop at +-z_m.
5. The other coefficients: least squares to S_target on |z| <= z_m (1 mm
   grid, mirrored), integral preserved (as LongitudinalFitter's
   preserve_integral). A quartic B-spline on simple knots is C3 for any
   coefficients, so the connection between the moved edge and the match
   point needs no extra conditions.
6. Beam-frame A_0..A_4 of the total, as 001d step 3 (same TILT_ANGLE).

Choices (2026-11-15):
- The edge always sits on a node. With the edge between two nodes the fit
  is about twice as bad at the edge (prototype, edge 1.23 m: 13/35 mT
  rms/max against 5.5/18 mT). Edges therefore come in 0.05 m steps:
  - multiples of 0.1 m use the grid with nodes at k * 0.1 m;
  - odd multiples of 0.05 m use the grid offset by 0.05 m (range
    +-(S_END + 0.05), nodes at +-0.05, +-0.15, ...).
  On the offset grid one element straddles the IP. 004b places the IP
  marker at the centre of the line, so for elements it is split at s = 0
  into two 0.05 m elements. The split is exact: a quartic on half the
  interval is a quartic. `element_boundaries` per case holds the result,
  |z| <= z_m.
- The match point moves with the edge.
- Symmetric pair of edges, so S(0) (the sum of both inner tails) changes
  with the edge: 11.5 mT for 1.20 m, 6.8 mT for 1.40 m, 4.2 mT for 1.60 m.
  The inner tail of one edge is extrapolated with e-folding length 42.9 cm.

Results (2026-11-15), 2 T, edges 1.20-1.60 m in 0.05 m steps:
- S/G = 0.5 lands on the chosen node to 0.2 mm. The total is exactly zero
  beyond z_m, and never negative between the edge and z_m.
- Fit - target: 5.4-5.7 mT rms (18 max) at the edge, < 1 mT rms elsewhere,
  the same on both grids.
- Total elements on |z| <= z_m: 44 (1.20 m) to 52 (1.60 m). The 0.05 m
  offset edges include two 0.05 m elements at the IP.
- Integral of the total over z = 0-6 m: 1.88 T m per m of edge shift
  (2.729 T m for 1.40 m).
- The 1.40 m case is 001d moved by 3.7 mm (data edge 1.404 m -> node). With
  that move applied to 001d too, they agree to 1.4 mT rms (2.5 max) at the
  edge. Beyond z_m, 001d's total is 1.2 mT rms (6.4 max), because G and S
  are fitted independently there; here it is 0.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

from aperture_study_io import PLOT_DIR, pdf_metadata
from solenoid_params import field_tag
from xtrack._temp.splineboris.longitudinal_fitter import LongitudinalFitter

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / 'IDEA_solenoid_beam_axis_raw.json'

MAIN_FIELD = 2.5           # T, main-solenoid field (nominal, at the IP)
OPERA_FIELD = 2.0          # T, nominal main field of the digitized Opera curves

# Ripple removal and G fit: as in 001d.
DZ = 1e-3                  # m, common grid step (raw resolution ~3.4 mm)
Z_CUT = 2.05               # m, ripples in B start here
FIT_WINDOW = (1.70, Z_CUT) # m, exponential fit of the clean edge of B
BLEND = 0.05               # m, cosine blend raw -> fit, ending at Z_CUT
S_END = 6.0                # m, spline range +-S_END
MAIN_ELEMENT_LENGTH = 0.2  # m, G fit

ELEMENT_LENGTH = 0.1       # m, grid of the screening and total splines
SCREENING_EDGES = [1.20, 1.25, 1.30, 1.35, 1.40,  # m, S/G = 0.5;
                   1.45, 1.50, 1.55, 1.60]        # multiples of 0.05 m
MATCH_DISTANCE = 1.0       # m, S = G beyond edge + this (rounded up to a node)
INNER_TAIL_WINDOW = (0.5, 1.0)  # m, e-folding fit of one edge's inner tail

TILT_ANGLE = 0.015         # rad, theta_b, as 001d (0 -> A_m = a_m)

DEG = 4                    # quartic B-splines, as LongitudinalFitter


def load_raw_curve(curves, key):
    """Raw points with repeated z (row-scanned steep parts) averaged."""
    z = np.array(curves[key]['z'])
    bz = np.array(curves[key]['Bz'])
    z_unique, inverse = np.unique(z, return_inverse=True)
    bz_mean = np.bincount(inverse, bz) / np.bincount(inverse)
    return z_unique, bz_mean


def fit_on_axis_bs(z_data, bs_data, element_length):
    """001d's fit: C3 quartic B-spline of ("Bs", 0) on +-S_END, data mirrored
    about the IP, field zero at both ends, integral preserved."""
    z_mirror = np.r_[-z_data[::-1], z_data]
    bs_mirror = np.r_[bs_data[::-1], bs_data]
    unique = np.r_[True, np.diff(z_mirror) > 0]      # z = 0 only once
    fitter = LongitudinalFitter(
        s_start=-S_END, s_end=S_END,
        n_elements=int(round(2 * S_END / element_length)),
        end_condition='zero', preserve_integral=True)
    fitter.fit(z_mirror[unique], bs_mirror[unique], ('Bs', 0))
    return fitter, fitter.splines[('Bs', 0)]


def beam_frame_coefficients(spline, s, theta):
    """A_m at x = y = 0 in the beam frame, to O(theta^2), as 001d step 3
    (b_s^(5) term of A_4 dropped)."""
    b0, b1, b2, b3, b4 = (spline.derivative(k)(s) if k else spline(s)
                          for k in range(5))
    th2 = theta ** 2
    return {
        'A0': b0 - th2 * (s / 2 * b0 + s ** 2 / 4 * b1),
        'A1': theta * (b0 + s / 2 * b1),
        'A2': -0.5 * b1 + th2 * (1.5 * b1 + 1.25 * s * b2 + 3 / 16 * s ** 2 * b3),
        'A3': -theta * (1.5 * b2 + 0.375 * s * b3),
        'A4': 0.375 * b3 - th2 * (3.75 * b3 + 27 / 16 * s * b4),
    }


# --------------------------------------- data, ripple removal (001d step 1)
with open(DATA_FILE) as fid:
    curves = json.load(fid)['curves']

z_g, bz_g = load_raw_curve(curves, 'main_only')
z_b, bz_b = load_raw_curve(curves, 'both')

field_scale = MAIN_FIELD / OPERA_FIELD          # everything below is linear
bz_g = field_scale * bz_g
bz_b = field_scale * bz_b
print(f'Main field {MAIN_FIELD:g} T: Opera curves ({OPERA_FIELD:g} T) '
      f'scaled by {field_scale:.4f}')

z = np.round(np.arange(0.0, z_g.max() + 0.5 * DZ, DZ), 9)
main = np.interp(z, z_g, bz_g)
both_raw = np.interp(z, z_b, bz_b, right=np.nan)

mask_fit = (z >= FIT_WINDOW[0]) & (z <= FIT_WINDOW[1])
slope, intercept = np.polyfit(z[mask_fit], np.log(both_raw[mask_fit]), 1)
both_tail = np.exp(intercept + slope * z)
weight = np.clip((z - (Z_CUT - BLEND)) / BLEND, 0.0, 1.0)
weight = 0.5 * (1.0 - np.cos(np.pi * weight))
both_raw_or_tail = np.where(np.isnan(both_raw), both_tail, both_raw)
both_clean = (1.0 - weight) * both_raw_or_tail + weight * both_tail
screening_clean = main - both_clean

# ------------------------- G fit (001d step 2), on the two 0.1 m grids
main_fitter, main_spline = fit_on_axis_bs(z_g, bz_g, MAIN_ELEMENT_LENGTH)
# 001d's screening fit, the reference for the 1.40 m edge
_, screening_001d = fit_on_axis_bs(z, screening_clean, ELEMENT_LENGTH)
main_zero_beyond = BSpline(main_spline.t, main_spline.c, DEG,
                           extrapolate=False)    # NaN beyond +-S_END -> 0


def make_grid(offset):
    """ELEMENT_LENGTH grid on +-(S_END + offset), nodes at offset + k *
    ELEMENT_LENGTH, field zero at the ends; G projected onto it."""
    s_end = S_END + offset
    grid = LongitudinalFitter(s_start=-s_end, s_end=s_end,
                              n_elements=int(round(2 * s_end / ELEMENT_LENGTH)),
                              end_condition='zero')
    z_dense = np.linspace(-s_end, s_end, int(round(2 * s_end / 5e-4)) + 1)
    g_dense = np.nan_to_num(main_zero_beyond(z_dense))
    grid.fit(z_dense, g_dense, ('Bs', 0))
    main = grid.splines[('Bs', 0)]
    knots = grid.knots
    support_lo, support_hi = knots[:-DEG - 1], knots[DEG + 1:]
    return dict(offset=offset, s_end=s_end, knots=knots, nodes=grid.nodes,
                main=main, main_coeffs=main.c.copy(),
                support_lo=support_lo, support_hi=support_hi,
                basis_integral=(support_hi - support_lo) / (DEG + 1),
                projection_error=np.abs(main(z_dense) - g_dense).max())


grids = {offset: make_grid(offset)
         for offset in (0.0, 0.5 * ELEMENT_LENGTH)}
for grid in grids.values():
    print(f'G ({main_fitter.n_elements} elements of {main_fitter.delta:.2f} m) '
          f'on the {ELEMENT_LENGTH:.2f} m grid with nodes at '
          f'{grid["offset"]:.2f} m + k {ELEMENT_LENGTH:.2f} m: max error '
          f'{grid["projection_error"]:.1e} T')

# --------------------------------- screening fraction, split into two edges
z_profile = z[z <= 4.0]
fraction = 1.0 - both_clean[z <= 4.0] / main_spline(z_profile)   # S/G
i_half = np.argmax(fraction >= 0.5)
edge_data = np.interp(0.5, fraction[i_half - 1:i_half + 1],
                      z_profile[i_half - 1:i_half + 1])
half_at_ip = 0.5 * fraction[0]                 # each edge's share at z = 0

mask_tail = ((z_profile >= INNER_TAIL_WINDOW[0])
             & (z_profile <= INNER_TAIL_WINDOW[1]))
decay = 0.3
for _ in range(200):
    one_edge = fraction - half_at_ip * np.exp(-z_profile / decay)
    decay_new = 1.0 / np.polyfit(z_profile[mask_tail],
                                 np.log(one_edge[mask_tail]), 1)[0]
    if abs(decay_new - decay) < 1e-9:
        break
    decay = decay_new
else:
    raise RuntimeError('inner-tail e-folding length did not converge')
u_data = z_profile - edge_data


def one_edge_fraction(u):
    """rho(u), u = distance from the edge (negative towards the IP)."""
    u = np.asarray(u, dtype=float)
    beyond_ip = half_at_ip * np.exp(np.minimum(u + edge_data, 0.0) / decay)
    return np.where(u >= -edge_data, np.interp(u, u_data, one_edge),
                    beyond_ip)


print(f'Data edge (S/G = 0.5): {edge_data:.4f} m; S/G at the IP '
      f'{fraction[0] * 1e3:.2f}e-3 = 2 x {half_at_ip * 1e3:.2f}e-3; inner tail '
      f'of one edge decays with e-folding length {decay * 100:.1f} cm '
      f'(self-consistent, z = {INNER_TAIL_WINDOW[0]}-{INNER_TAIL_WINDOW[1]} m)')

# ------------------------------------------ constrained fit for each edge
def grid_for_edge(edge):
    """The grid with a node on `edge`: nodes at k * ELEMENT_LENGTH, or offset
    by ELEMENT_LENGTH / 2."""
    half_steps = edge / (0.5 * ELEMENT_LENGTH)
    if abs(half_steps - round(half_steps)) > 1e-9:
        raise ValueError(f'edge {edge} m is not a multiple of '
                         f'{0.5 * ELEMENT_LENGTH} m')
    return grids[0.0 if round(half_steps) % 2 == 0 else 0.5 * ELEMENT_LENGTH]


def build_screening(edge):
    """S for an edge at `edge` (S/G = 0.5 there): S = G for |z| >= z_match,
    least squares to the moved target inside, integral preserved."""
    grid = grid_for_edge(edge)
    knots, nodes = grid['knots'], grid['nodes']
    support_lo, support_hi = grid['support_lo'], grid['support_hi']
    basis_integral, main_coeffs = grid['basis_integral'], grid['main_coeffs']
    z_match = nodes[np.searchsorted(nodes, edge + MATCH_DISTANCE - 1e-9)]
    if z_match > z_profile.max() - 0.5:
        raise ValueError(f'match point {z_match} m too far out')

    z_half = z[z <= z_match + 1e-9]
    target_half = main_spline(z_half) * (one_edge_fraction(z_half - edge)
                                         + one_edge_fraction(-z_half - edge))
    z_t = np.r_[-z_half[:0:-1], z_half]
    target = np.r_[target_half[:0:-1], target_half]

    free = (support_lo >= -z_match - 1e-9) & (support_hi <= z_match + 1e-9)
    coeffs = main_coeffs.copy()                  # fixed ones: G's
    K = BSpline.design_matrix(z_t, knots, DEG).tocsc()
    K_free = K[:, np.flatnonzero(free)]
    rhs = target - K[:, np.flatnonzero(~free)] @ coeffs[~free]
    normal = (K_free.T @ K_free).toarray()
    c_lsq = np.linalg.solve(normal, K_free.T @ rhs)

    # integral over |z| <= z_m equal to the target's (S = G beyond)
    required = (np.trapezoid(target, z_t)
                + 2 * grid['main'].integrate(z_match, grid['s_end'])
                - basis_integral[~free] @ coeffs[~free])
    w = basis_integral[free]
    normal_inv_w = np.linalg.solve(normal, w)
    coeffs[free] = (c_lsq - normal_inv_w * (w @ c_lsq - required)
                    / (w @ normal_inv_w))

    screening = BSpline(knots, coeffs, DEG)
    total = BSpline(knots, main_coeffs - coeffs, DEG)
    # elements of the total on |z| <= z_m, with a boundary at the IP
    inside = nodes[np.abs(nodes) <= z_match + 1e-9]
    boundaries = np.unique(np.round(np.r_[inside, 0.0], 12))
    return dict(edge=edge, z_match=z_match, offset=grid['offset'],
                main=grid['main'], screening=screening, total=total,
                z_target=z_half, target=target_half,
                element_boundaries=boundaries)


cases = [build_screening(edge) for edge in SCREENING_EDGES]

z_fine = np.round(np.arange(0.0, S_END + 0.5 * DZ, DZ), 9)
s_coef = np.round(np.arange(-S_END, S_END + 0.5 * DZ, DZ), 9)
for case in cases:
    edge, z_match = case['edge'], case['z_match']
    s_fit, t_fit = case['screening'](z_fine), case['total'](z_fine)
    z_near = z_fine[z_fine < 4.0]
    ratio = case['screening'](z_near) / case['main'](z_near)
    i = np.argmax(ratio >= 0.5)
    case['edge_fit'] = np.interp(0.5, ratio[i - 1:i + 1], z_near[i - 1:i + 1])
    lengths = np.diff(case['element_boundaries'])
    case['n_elements'] = len(lengths)
    case['n_short'] = int(np.sum(lengths < ELEMENT_LENGTH - 1e-9))
    case['residual'] = case['screening'](case['z_target']) - case['target']
    case['coefficients'] = beam_frame_coefficients(case['total'], s_coef,
                                                   TILT_ANGLE)
    beyond = z_fine >= z_match
    between = (z_fine >= edge) & (z_fine <= z_match)
    case['total_beyond_max'] = np.abs(t_fit[beyond]).max()
    case['total_between_min'] = t_fit[between].min()

# ----------------------------------------------------------------- report
print(f'\nScreening edge scan: {ELEMENT_LENGTH} m elements, S = G beyond '
      f'z_m = edge + {MATCH_DISTANCE} m (rounded up to a node)')
print('  edge   z_m  node    S/G=0.5 at   S(0)  total(0)    int total'
      '   total elements       min total  |total| beyond z_m')
print('   [m]   [m]  offset         [m]   [mT]       [T]  0-6 m [T m]'
      '   on |z| <= z_m   edge..z_m [mT]                 [T]')
for case in cases:
    elements = f'{case["n_elements"]}'
    if case['n_short']:
        elements += f' ({case["n_short"]} x {0.5 * ELEMENT_LENGTH:.2f} m)'
    print(f'  {case["edge"]:.2f}  {case["z_match"]:.2f}  {case["offset"]:+.2f}'
          f'  {case["edge_fit"]:12.4f}  {case["screening"](0.0) * 1e3:5.1f}'
          f'  {case["total"](0.0):8.4f}  {case["total"].integrate(0.0, S_END):11.4f}'
          f'   {elements:>14s}  {case["total_between_min"] * 1e3:14.2f}'
          f'  {case["total_beyond_max"]:18.1e}')

sum_check = max(np.abs(case['main'](z_fine) - case['screening'](z_fine)
                       - case['total'](z_fine)).max() for case in cases)
print(f'  G - S - total, all edges: max {sum_check:.1e} T')

print('\n  fit - target [mT], rms / max:')
for case in cases:
    edge, zt, res = case['edge'], case['z_target'], case['residual']
    parts = []
    for name, mask in [
            ('inner', zt < edge - 0.25),
            ('edge', (zt >= edge - 0.25) & (zt <= edge + 0.35)),
            ('tail', zt > edge + 0.35)]:
        parts.append(f'{name} {np.sqrt(np.mean(res[mask] ** 2)) * 1e3:5.2f} / '
                     f'{np.abs(res[mask]).max() * 1e3:5.1f}')
    print(f'  {edge:.2f} m:  ' + ',   '.join(parts))

print(f'\n  max |A_m| of the total, theta_b = {TILT_ANGLE} rad:')
print('  edge      A0 [T]    A1 [T]  A2 [T/m]  A3 [T/m2]  A4 [T/m3]')
for case in cases:
    c = case['coefficients']
    print(f'  {case["edge"]:.2f}  ' + '  '.join(
        f'{np.abs(c[key]).max():8.4g}' for key in ('A0', 'A1', 'A2', 'A3', 'A4')))

reference = next((case for case in cases if abs(case['edge'] - 1.40) < 1e-9),
                 None)
if reference is not None:
    # 001d's total with its edge moved the same way (data edge -> node)
    shift = edge_data - reference['edge']
    z_cmp = z_fine[z_fine <= reference['z_match']]
    fraction_001d = (screening_001d(z_cmp + shift)
                     / main_spline(z_cmp + shift))
    diff = (reference['total'](z_cmp)
            - main_spline(z_cmp) * (1.0 - fraction_001d))
    print(f'\n  edge 1.40 m against 001d (unconstrained S fit), its edge moved '
          f'by the same {-shift * 1e3:.1f} mm, difference of the totals:')
    for z_min, z_max in [(0.0, 1.2), (1.2, 1.7), (1.7, reference['z_match'])]:
        mask = (z_cmp >= z_min) & (z_cmp <= z_max)
        print(f'    z = {z_min:.2f}-{z_max:.2f} m: rms '
              f'{np.sqrt(np.mean(diff[mask] ** 2)) * 1e3:.2f} mT, max '
              f'{np.abs(diff[mask]).max() * 1e3:.1f} mT')
    beyond = z_fine >= reference['z_match']
    total_001d = main_spline(z_fine[beyond]) - screening_001d(z_fine[beyond])
    print(f'    beyond z_m = {reference["z_match"]:.2f} m, 001d total: rms '
          f'{np.sqrt(np.mean(total_001d ** 2)) * 1e3:.2f} mT, max '
          f'{np.abs(total_001d).max() * 1e3:.1f} mT (here: 0)')

# -------------------------------------------------------------------- plot
# main, screening and total, one panel per edge
n_cols = 1 if len(cases) <= 5 else 2
n_rows = -(-len(cases) // n_cols)
fig_parts, axes_parts = plt.subplots(n_rows, n_cols,
                                     figsize=(8 * n_cols, 1 + 2.2 * n_rows),
                                     sharex=True, sharey=True, squeeze=False)
axes_flat = axes_parts.T.ravel()                 # fill column by column
for ax, case in zip(axes_flat, cases):
    ax.axhline(0, color='gray', lw=0.8)
    ax.axvline(case['edge'], color='gray', lw=0.8, ls=':',
               label='edge (S/G = 0.5)')
    ax.axvline(case['z_match'], color='gray', lw=0.8, ls='--',
               label='match point $z_m$ (S = G beyond)')
    ax.plot(z_fine, case['main'](z_fine), color='green', lw=1.5,
            label='main (G)')
    ax.plot(z_fine, case['screening'](z_fine), color='darkred', lw=1.2,
            ls='--', label='screening (S, as plotted; its field is $-$S)')
    ax.plot(z_fine, case['total'](z_fine), color='black', lw=1.5,
            label='total = G $-$ S')
    ax.text(0.98, 0.94, f'edge {case["edge"]:.2f} m, '
            f'$z_m$ = {case["z_match"]:.2f} m',
            transform=ax.transAxes, ha='right', va='top', fontsize=9)
    ax.grid(True, alpha=0.4)
for ax in axes_parts[:, 0]:
    ax.set_ylabel(r'$B_s$ [T]')
axes_parts[0, 0].set_ylim(-0.05 * MAIN_FIELD, 1.075 * MAIN_FIELD)
axes_parts[0, 0].set_xlim(0, S_END)
handles, labels = axes_parts[0, 0].get_legend_handles_labels()
spare = axes_flat[len(cases):]
if len(spare):                                   # legend in an empty panel
    for ax in spare:
        ax.axis('off')
    spare[0].legend(handles, labels, fontsize=9, loc='center')
    above = axes_flat[len(cases) - 1]            # its column's last panel
    above.xaxis.set_tick_params(labelbottom=True)
    above.set_xlabel('z [m]')
else:
    axes_parts[0, 0].legend(fontsize=8, loc='center right')
for ax in axes_parts[-1, :]:
    if ax.axison:
        ax.set_xlabel('z [m]')
fig_parts.suptitle(f'IDEA solenoids, main field {MAIN_FIELD:g} T: main, '
                   'screening and total per screening edge')
fig_parts.tight_layout()

PLOT_DIR.mkdir(parents=True, exist_ok=True)
out_path = (PLOT_DIR / 'IDEA_solenoid_screening_edge_scan_components_'
            f'{field_tag(MAIN_FIELD)}.pdf')
fig_parts.savefig(out_path, bbox_inches='tight', metadata=pdf_metadata())
print(f'\nSaved {out_path}')

# real field on the beam axis, beam frame: B_s = A_0, B_x = A_1; the
# screening solenoid's field is -S
fig_real, axes_real = plt.subplots(len(cases), 2,
                                   figsize=(14, 1.5 + 2.0 * len(cases)),
                                   sharex=True, sharey='col', squeeze=False)
for row, case in zip(axes_real, cases):
    main_field = beam_frame_coefficients(case['main'], s_coef, TILT_ANGLE)
    screening_field = beam_frame_coefficients(case['screening'], s_coef,
                                              TILT_ANGLE)
    for ax, key in zip(row, ('A0', 'A1')):
        ax.axhline(0, color='gray', lw=0.8)
        ax.plot(s_coef, main_field[key], color='green', lw=1.5,
                label='main')
        ax.plot(s_coef, -screening_field[key], color='darkred', lw=1.2,
                ls='--', label='screening')
        ax.plot(s_coef, case['coefficients'][key], color='black', lw=1.5,
                label='total = main + screening')
        ax.grid(True, alpha=0.4)
    row[0].set_ylabel(r'$B_s$ [T]')
    row[1].set_ylabel(r'$B_x$ [T]')
    row[0].text(0.98, 0.94, f'edge {case["edge"]:.2f} m',
                transform=row[0].transAxes, ha='right', va='top', fontsize=9)
for ax in axes_real[-1, :]:
    ax.set_xlabel('S [m]')
axes_real[0, 0].set_xlim(-S_END, S_END)
handles, labels = axes_real[0, 0].get_legend_handles_labels()
fig_real.legend(handles, labels, loc='upper center', ncol=3, fontsize=9,
                bbox_to_anchor=(0.5, 0.975))
fig_real.suptitle(f'IDEA solenoids, main field {MAIN_FIELD:g} T: field on '
                  r'the beam axis, $\theta_b$ = ' f'{TILT_ANGLE} rad')
fig_real.tight_layout(rect=(0, 0, 1, 0.965))

out_path = (PLOT_DIR / 'IDEA_solenoid_screening_edge_scan_real_field_'
            f'{field_tag(MAIN_FIELD)}.pdf')
fig_real.savefig(out_path, bbox_inches='tight', metadata=pdf_metadata())
print(f'Saved {out_path}')

plt.show()
