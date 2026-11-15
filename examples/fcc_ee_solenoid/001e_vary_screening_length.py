"""
Move the start (inner edge) of the IDEA screening solenoid and rebuild the
screening and total on-axis fields on the 0.1 m spline grid.

Same data, ripple removal and main-solenoid fit as 001d (copied here, 001e is
standalone). Notation as in 001d: G = main solenoid, S = screening *as
plotted* (positive; its field is -S), total = G - S; on the solenoid axis,
mirror-symmetric about the IP.

The construction (steps 1-5 below) lives in idea_solenoid_field.py since
2026-11-15, shared with 004a_build_idea_solenoid.py (the BFieldExpansion line
for the ring). This script runs the edge scan, the report and the figures;
the move left its output bit-identical.

Edge: where S reaches half of its own plateau, S/G = (1 + eps)/2 with the
over-screening eps below (0.5 without: the beam sees half of the main
field). In the data (eps = 0) it is at z_e0 = 1.404 m (half of max S:
1.397 m).

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
3. Target S_target(z) = G(z) [rho(z - z_e) + rho(-z - z_e)] k(z), with
   k = 1 + eps w the over-screening factor (below; k = 1 for eps = 0).
   What moves is the screening *fraction*, not S: past its edge S follows
   G, which falls by up to ~1 T/m at 3-4 m, so shifting S itself by 0.2 m
   would leave up to 0.28 T uncancelled. With the fraction, the height of
   the edge follows the local G, and for eps = 0 S never exceeds G.
4. Peak z_p of the eps = 0 target (where S stops rising and starts to
   follow G down): the vertex of a parabola fitted over +-PEAK_WINDOW
   around its maximum. The peak is flat (+-5 cm costs 2-5 mT), so the raw
   argmax follows the digitization (steps of up to 3 cm between edges);
   the parabola gives z_p = z_e + 0.411 m (1.20 m) ... z_e + 0.338 m
   (1.60 m), within 1.3 cm of the fitted spline's peak.
   Match point z_m = z_p + TAPER_LENGTH, rounded up to a node: z_e + 1.0 m
   for edges 1.20 and 1.25 m, z_e + 0.9 m for the others. Both move with
   the edge. For |z| >= z_m, S = G exactly: the B-spline coefficients whose
   support reaches past z_m are G's. The total is identically zero there,
   so its elements can stop at +-z_m.
5. The other coefficients: least squares to S_target on |z| <= z_m (1 mm
   grid, mirrored), integral preserved (as LongitudinalFitter's
   preserve_integral). A quartic B-spline on simple knots is C3 for any
   coefficients, so the connection between the moved edge and the match
   point needs no extra conditions.
6. Beam-frame A_0..A_4 of the total, as 001d step 3. TILT_ANGLE = theta_b
   = -solenoid_params.THETA = +0.015: theta_b is the solenoid axis' angle
   from the beam, THETA (TiltedSolenoid, 004a, 010a) the beam's angle from
   the solenoid axis. Checked against the 004a 2 T template: B_x(IP) =
   +0.030 T there, +theta_b B_s here, and both have a negative edge peak.

Over-screening (SCREENING_OVERSHOOT = eps, 2026-11-15), to mimic 010a, whose
local solenoid next to the main coil pulls the total down to -0.67 B0, with
a negative region of -0.358 m x B0:
- k(z) = 1 + eps w(z), with w = 1 for |z| <= z_p and a septic smoothstep
  (C3) to 0 at z_p + TAPER_LENGTH. So S is (1 + eps) x the eps = 0 target
  from the IP (inner tail included) to z_p, and the total has a single
  negative lobe of about -eps G, deepest just past z_p.
- The taper starts at the peak (user, 2026-11-15). Both S_eps=0 and w have
  zero slope there, so the target's peak stays at z_p for every eps (fitted
  peaks within 1.4 cm). Before, the taper ended at z_m = edge + 1.2 m. The
  total then sat flat at about -1.05 T (eps = 0.5, edge 1.40 m) from the S
  peak (1.76 m) to the taper start (2.10 m), and rose only in the taper: a
  shoulder in B_s, and a B_x tail with its opposite-sign peak at 2.37 m.
- Anchored at the peak the lobe is shorter, so the same integral needs a
  larger eps. eps = 0.75 reproduces 010a's lobe at edge 1.40 m (-0.650 B0,
  -0.348 m x B0; at edge 1.30 m, 010a's own local-solenoid start: -0.661 B0,
  -0.364 m). The total is linear in eps, if a target integral is preferred.
- z_m <= 2.5 m, before the main solenoid's own edge (G = G(0)/2 at
  ~2.9 m), so the screening matches the main field there.
- The edge keeps its measured shape, scaled by 1 + eps; its half-plateau
  point stays on the node.
- TAPER_LENGTH >= 0.4 m: the 0.1 m spline cannot follow anything shorter.
  Prototype at edge 1.40 m: a hard switch at z_m leaves 0.8 T fit error, and
  the integral constraint pushes it into S(0) (+54 mT); a 0.3 m taper gives
  0.17 T. Peak-anchored, at the same integral (edge 1.40 m): 0.6 m instead
  of 0.5 m needs eps 0.68 instead of 0.73, and lowers the opposite-sign B_x
  peak from 0.096 to 0.075 T, with z_m 0.1 m further out.
- Alternatives prototyped (2026-11-15, edge 1.40 m, 2.5 T, all lowering the
  integral of the total by 0.90 T m):
  - an extra anti-solenoid section 1.40-2.00 m with the measured edge at
    both ends (dB = 1.49 T): its outer fringe needs S = G only from 2.6 m,
    and it gives the largest B_x (0.18 T);
  - a prescribed negative lobe of the total: its shape is not tied to a
    coil, and a >= 0.4 m ramp moves the zero crossing out to 1.53 m.
  The user chose the scaling.

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
- The peak z_p and the match point move with the edge.
- Symmetric pair of edges, so S(0) (the sum of both inner tails) changes
  with the edge: 11.5 mT for 1.20 m, 6.8 mT for 1.40 m, 4.2 mT for 1.60 m.
  The inner tail of one edge is extrapolated with e-folding length 42.9 cm.

Results (2026-11-15), 2 T, eps = 0, edges 1.20-1.60 m in 0.05 m steps:
- S/G = 0.5 lands on the chosen node to 0.2 mm. The total is exactly zero
  beyond z_m, and never negative between the edge and z_m.
- Fit - target: 5.4-5.7 mT rms (18 max) at the edge, < 1 mT rms elsewhere,
  the same on both grids.
- Total elements on |z| <= z_m: 44 (1.20 m) to 50 (1.60 m). The 0.05 m
  offset edges include two 0.05 m elements at the IP.
- Integral of the total over z = 0-6 m: 1.88 T m per m of edge shift
  (2.729 T m for 1.40 m).
- The 1.40 m case is 001d moved by 3.7 mm (data edge 1.404 m -> node). With
  that move applied to 001d too, they agree to 1.4 mT rms (2.5 max) at the
  edge. Beyond z_m, 001d's total is 1.2 mT rms (6.4 max), because G and S
  are fitted independently there; here it is 0.
- Moving z_m from edge + 1.0 m (MATCH_DISTANCE, until 2026-11-15) to the
  node past z_p + TAPER_LENGTH changes the total by <= 1.6 mT.

Results (2026-11-15), 2.5 T, eps = 0.75, TAPER_LENGTH = 0.5 m:
- The edge stays on its node to 0.2 mm, and the total is exactly zero
  beyond z_m. Same number of elements as eps = 0.
- The total crosses zero 1.6 cm past the edge. Minimum -1.67 T (-0.669 B0,
  edge 1.20 m) to -1.56 T (-0.625 B0, 1.60 m), 3-4 cm past z_p. Negative
  part -0.95 to -0.78 T m (-0.379 to -0.313 m x B0).
- Integral of the total over 0-6 m: 1.77 (1.20 m) to 2.89 T m (1.60 m),
  0.99-1.17 T m below eps = 0 (2.328 against 3.411 T m for 1.40 m).
- S(0) x 1.67-1.82 (24.9 mT at 1.20 m, 8.7 mT at 1.60 m); total(0)
  3.5-10.5 mT lower.
- Fit - target: 38-40 mT max at the edge (it scales with the edge height,
  x 1.75), 5-19 mT max in the taper.
- max |A_1| (B_x) 0.17-0.22 T, against 0.08-0.11 T for eps = 0: the total's
  edge is steeper. A second B_x peak, 0.094-0.108 T of the opposite sign,
  sits about 0.26 m past z_p (mid-taper). A_2-A_4 grow by x 1.75.
"""

import matplotlib.pyplot as plt
import numpy as np

from aperture_study_io import PLOT_DIR, pdf_metadata
from idea_solenoid_field import (
    DZ,
    ELEMENT_LENGTH,
    INNER_TAIL_WINDOW,
    OPERA_FIELD,
    S_END,
    TILT_ANGLE,
    IdeaSolenoidData,
    beam_frame_coefficients,
    overshoot_factor,
)
from solenoid_params import field_tag

MAIN_FIELD = 2.5           # T, main-solenoid field (nominal, at the IP)

SCREENING_EDGES = [1.20, 1.25, 1.30, 1.35, 1.40,  # m, S/G = 0.5;
                   1.45, 1.50, 1.55, 1.60]        # multiples of 0.05 m

# Over-screening (as 010a's local solenoid): S x (1 + eps) from the IP to
# z_p, the peak of S for eps = 0, then a C3 taper of TAPER_LENGTH back to x 1.
# S = G from the first node past the taper (z_m). eps = 0 -> no overshoot.
SCREENING_OVERSHOOT = 0.75  # eps (0.75: 010a's negative lobe at edge 1.40 m)
TAPER_LENGTH = 0.5         # m, >= 0.4 m (the 0.1 m spline cannot follow less)

# --------------------- data, ripple removal, G fit, rho (idea_solenoid_field)
data = IdeaSolenoidData(MAIN_FIELD)
print(f'Main field {MAIN_FIELD:g} T: Opera curves ({OPERA_FIELD:g} T) '
      f'scaled by {data.field_scale:.4f}')
for grid in data.grids.values():
    print(f'G ({data.main_fitter.n_elements} elements of '
          f'{data.main_fitter.delta:.2f} m) '
          f'on the {ELEMENT_LENGTH:.2f} m grid with nodes at '
          f'{grid["offset"]:.2f} m + k {ELEMENT_LENGTH:.2f} m: max error '
          f'{grid["projection_error"]:.1e} T')
print(f'Data edge (S/G = 0.5): {data.edge_data:.4f} m; S/G at the IP '
      f'{data.fraction[0] * 1e3:.2f}e-3 = 2 x {data.half_at_ip * 1e3:.2f}e-3; '
      f'inner tail of one edge decays with e-folding length '
      f'{data.decay * 100:.1f} cm (self-consistent, z = '
      f'{INNER_TAIL_WINDOW[0]}-{INNER_TAIL_WINDOW[1]} m)')

# ------------------------------------------ constrained fit for each edge
cases = [data.build_case(edge, SCREENING_OVERSHOOT, TAPER_LENGTH)
         for edge in SCREENING_EDGES]

z_fine = np.round(np.arange(0.0, S_END + 0.5 * DZ, DZ), 9)
s_coef = np.round(np.arange(-S_END, S_END + 0.5 * DZ, DZ), 9)
for case in cases:
    edge, z_match = case['edge'], case['z_match']
    s_fit, t_fit = case['screening'](z_fine), case['total'](z_fine)
    z_near = z_fine[z_fine < 4.0]
    ratio = (case['screening'](z_near) / case['main'](z_near)
             / overshoot_factor(z_near, case['z_peak'], SCREENING_OVERSHOOT,
                                TAPER_LENGTH))   # S/(k G), 0.5 at edge
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
print(f'\nScreening edge scan: {ELEMENT_LENGTH} m elements; z_p = peak of S '
      f'for eps = 0; S = G beyond z_m = z_p + {TAPER_LENGTH:g} m (rounded up '
      'to a node)')
print(f'Over-screening eps = {SCREENING_OVERSHOOT:g}: S x {1 + SCREENING_OVERSHOOT:g}'
      f' up to z_p, C3 taper of {TAPER_LENGTH:g} m to x 1; '
      f'edge where S/G = {0.5 * (1 + SCREENING_OVERSHOOT):g}')
print('  edge    z_p   z_m  node   edge of fit   S(0)  total(0)    int total'
      '   total elements    min total  |total| beyond z_m')
print('   [m]    [m]   [m]  offset         [m]   [mT]       [T]  0-6 m [T m]'
      '   on |z| <= z_m   edge..z_m [T]                 [T]')
for case in cases:
    elements = f'{case["n_elements"]}'
    if case['n_short']:
        elements += f' ({case["n_short"]} x {0.5 * ELEMENT_LENGTH:.2f} m)'
    print(f'  {case["edge"]:.2f}  {case["z_peak"]:.3f}  {case["z_match"]:.2f}'
          f'  {case["offset"]:+.2f}'
          f'  {case["edge_fit"]:12.4f}  {case["screening"](0.0) * 1e3:5.1f}'
          f'  {case["total"](0.0):8.4f}  {case["total"].integrate(0.0, S_END):11.4f}'
          f'   {elements:>14s}  {case["total_between_min"]:13.4f}'
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
if reference is not None and SCREENING_OVERSHOOT != 0:
    print('\n  (comparison with 001d only for SCREENING_OVERSHOOT = 0)')
elif reference is not None:
    # 001d's total with its edge moved the same way (data edge -> node)
    shift = data.edge_data - reference['edge']
    z_cmp = z_fine[z_fine <= reference['z_match']]
    fraction_001d = (data.screening_001d(z_cmp + shift)
                     / data.main_spline(z_cmp + shift))
    diff = (reference['total'](z_cmp)
            - data.main_spline(z_cmp) * (1.0 - fraction_001d))
    print(f'\n  edge 1.40 m against 001d (unconstrained S fit), its edge moved '
          f'by the same {-shift * 1e3:.1f} mm, difference of the totals:')
    for z_min, z_max in [(0.0, 1.2), (1.2, 1.7), (1.7, reference['z_match'])]:
        mask = (z_cmp >= z_min) & (z_cmp <= z_max)
        print(f'    z = {z_min:.2f}-{z_max:.2f} m: rms '
              f'{np.sqrt(np.mean(diff[mask] ** 2)) * 1e3:.2f} mT, max '
              f'{np.abs(diff[mask]).max() * 1e3:.1f} mT')
    beyond = z_fine >= reference['z_match']
    total_001d = (data.main_spline(z_fine[beyond])
                  - data.screening_001d(z_fine[beyond]))
    print(f'    beyond z_m = {reference["z_match"]:.2f} m, 001d total: rms '
          f'{np.sqrt(np.mean(total_001d ** 2)) * 1e3:.2f} mT, max '
          f'{np.abs(total_001d).max() * 1e3:.1f} mT (here: 0)')

# -------------------------------------------------------------------- plot
overshoot_note = (f', screening overshoot $\\varepsilon$ = {SCREENING_OVERSHOOT:g}'
                  if SCREENING_OVERSHOOT else '')
overshoot_tag = (f'_eps{SCREENING_OVERSHOOT:.2f}'.replace('.', 'p')
                 if SCREENING_OVERSHOOT else '')

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
               label=f'edge (S/G = {0.5 * (1 + SCREENING_OVERSHOOT):g})')
    ax.axvline(case['z_peak'], color='gray', lw=0.8, ls='-.',
               label=r'$z_p$: peak of S for $\varepsilon$ = 0'
               + (r', taper start (S $\times$ '
                  f'{1 + SCREENING_OVERSHOOT:g} before)'
                  if SCREENING_OVERSHOOT else ''))
    ax.axvline(case['z_match'], color='gray', lw=0.8, ls='--',
               label='match point $z_m$ (S = G beyond)')
    ax.plot(z_fine, case['main'](z_fine), color='green', lw=1.5,
            label='main (G)')
    ax.plot(z_fine, case['screening'](z_fine), color='darkred', lw=1.2,
            ls='--', label='screening (S, as plotted; its field is $-$S)')
    ax.plot(z_fine, case['total'](z_fine), color='black', lw=1.5,
            label='total = G $-$ S')
    ax.text(0.98, 0.94, f'edge {case["edge"]:.2f} m, '
            f'$z_p$ = {case["z_peak"]:.2f} m, $z_m$ = {case["z_match"]:.2f} m',
            transform=ax.transAxes, ha='right', va='top', fontsize=9)
    ax.grid(True, alpha=0.4)
for ax in axes_parts[:, 0]:
    ax.set_ylabel(r'$B_s$ [T]')
s_max = max(case['screening'](z_fine).max() for case in cases)
t_min = min(case['total'](z_fine).min() for case in cases)
axes_parts[0, 0].set_ylim(min(-0.05 * MAIN_FIELD, t_min - 0.05 * MAIN_FIELD),
                          max(1.075 * MAIN_FIELD, s_max + 0.075 * MAIN_FIELD))
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
fig_parts.suptitle(f'IDEA solenoids, main field {MAIN_FIELD:g} T'
                   f'{overshoot_note}: main, screening and total per '
                   'screening edge')
fig_parts.tight_layout()

PLOT_DIR.mkdir(parents=True, exist_ok=True)
out_path = (PLOT_DIR / 'IDEA_solenoid_screening_edge_scan_components_'
            f'{field_tag(MAIN_FIELD)}{overshoot_tag}.pdf')
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
fig_real.suptitle(f'IDEA solenoids, main field {MAIN_FIELD:g} T'
                  f'{overshoot_note}: field on the beam axis, '
                  r'$\theta_b$ = ' f'{TILT_ANGLE} rad')
fig_real.tight_layout(rect=(0, 0, 1, 0.965))

out_path = (PLOT_DIR / 'IDEA_solenoid_screening_edge_scan_real_field_'
            f'{field_tag(MAIN_FIELD)}{overshoot_tag}.pdf')
fig_real.savefig(out_path, bbox_inches='tight', metadata=pdf_metadata())
print(f'Saved {out_path}')

plt.show()
