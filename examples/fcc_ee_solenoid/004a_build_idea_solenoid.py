"""
Build the IDEA detector solenoid (main + screening, total = G - S, from
idea_solenoid_field.py / 001e) as a line of xt.BFieldExpansion elements, with
the compensation solenoids, for 004b_install_idea_solenoids_in_fcc_ring.py.
Exploratory, not in the paper's pipeline (claude_notes/18).

Steps:
1. The total on-axis profile b(s) for one case (--b0, --edge, --eps,
   --taper-length): a C3 quartic B-spline on 0.1 m elements, |s| <= z_m,
   identically zero beyond.
2. First-corrector segments as in 010a: two segments with 010a's weights,
   placed on the opposite-sign B_x lobe (where the over-screened total comes
   back to zero). 010a's segments are mapped affinely from 010a's own lobe
   (1.758-2.283 m) onto this one. Element boundaries are added at the segment
   ends, so the installer can overlay the corrector on whole elements.
3. One BFieldExpansion per element, exact to all orders in the tilt:
   inside an element b is a quartic, so the tilted field is a polynomial in
   (x, y, s) and its beam-frame coefficients are polynomials in s
   (idea_solenoid_field.exact_beam_frame_polynomials, theta_b = -THETA).
   ksolc = A_0 / B rho = B_s on axis, ksc[i] = A_(i+1) / B rho = d^i B_x /
   dx^i for i = 0..4 (as the 004a templates), knc = 0 (B_y = 0 on y = 0).
   No refit, no SplineBoris step. (The O(theta^2) A_m of 001d/001e are off
   by up to ~2.5 mT in B_x at this sharp edge, where s theta ~ 2 cm is not
   small against the edge width, and 001d's A_0 had a wrong theta^2 term.)
4. Compensation solenoids: the 004a 2 T template (its shape does not depend
   on B0), rescaled so that main + 2 x compensation int(Bs) = 0, converted
   to BFieldExpansion (bfe_setup).
5. Checks: (a) the same A_m -> BFE code on the analytic 2 T main solenoid
   against the established 004a 2 T template (tilt sign and convention);
   (b) the off-axis field against the exact field of the tilted axisymmetric
   total (paraxial series, exact for quartic pieces); (c) tracking against
   BorisSpatialIntegrator through that exact field, RK4 step convergence,
   symplecticity; (d) int(Bs).

Writes 004_idea_solenoid_lines_{field_tag}_{idea_case_tag}.json and a figure.
"""
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack._temp.splineboris.longitudinal_fitter import LongitudinalFitter

from aperture_study_io import PLOT_DIR, pdf_metadata
from bfe_setup import field_tesla, sb_line_to_bfe_line
from idea_solenoid_field import (
    TILT_ANGLE,
    IdeaSolenoidData,
    exact_beam_frame_polynomials,
)
from solenoid_params import (
    IDEA_DEFAULT_TAPER_LENGTH,
    MAIN_SOLENOID_A,
    THETA,
    field_tag,
    half_length_for_b0,
    idea_case_tag,
)
from spline_boris_setup import symplectic_error
from tilted_solenoid import TiltedSolenoid

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument('--b0', type=float, default=2.5,
                    help='Main-solenoid field [T] (default 2.5, as 001e).')
parser.add_argument('--edge', type=float, default=1.40,
                    help='Screening edge [m], a multiple of 0.05 m.')
parser.add_argument('--eps', type=float, default=0.75,
                    help='Over-screening eps (default 0.75, as 001e).')
parser.add_argument('--taper-length', type=float,
                    default=IDEA_DEFAULT_TAPER_LENGTH,
                    help='Over-screening taper length [m], >= 0.4.')
parser.add_argument('--steps', type=int, default=None,
                    help='RK4 steps per 0.1 m element (default: '
                         'DEFAULT_STEPS).')
parser.add_argument('--no-show', action='store_true')
args = parser.parse_args()

HERE = Path(__file__).parent
TAG = f'{field_tag(args.b0)}_{idea_case_tag(args.edge, args.eps, args.taper_length)}'
OUTPUT_JSON = HERE / f'004_idea_solenoid_lines_{TAG}.json'
COMP_TEMPLATE_JSON = HERE / '004_solenoid_lines_2T.json'

PARTICLE_REF = xt.Particles('positron', energy0=45.6e9)
RIGIDITY0 = PARTICLE_REF.rigidity0[0]

# Free drift between each IP and the first QD0 element (hcor/vcor_qd0*),
# measured in fccee_z_lcc.json on 2026-11-15: 2.400 m at all four IPs.
SLOT_HALF_LENGTH = 2.400

# RK4 steps per 0.1 m element; from the step-convergence check below (the
# analytic templates use 2 steps per 2.4 cm slice, bfe_setup).
DEFAULT_STEPS = 8
STEPS = args.steps or DEFAULT_STEPS

# 010a's first corrector (010a_parametric_version.py, first_corr_segments)
# and the opposite-sign B_x lobe it covers: B_x > 5 % of the lobe's peak, from
# 010a's analytic fields (theta = -0.015; the shape does not depend on B0),
# computed 2026-11-15. Only the weight ratio matters: one knob per side and
# plane multiplies all segments.
CORR_SEGMENTS_010A = (((1.78, 1.965), 1.7799320875667033),
                      ((2.015, 2.2), 1.0))
LOBE_010A = (1.758, 2.283)
LOBE_THRESHOLD = 0.05


# --------------------------------------------------------------- helpers
def bfe_line_from_profile(profile, boundaries, theta_b, steps, prefix):
    """One BFieldExpansion per [boundaries[i], boundaries[i+1]] carrying the
    beam-frame field of the on-axis profile, s = 0 at the IP."""
    elements, names, polys = [], [], []
    width = len(str(len(boundaries) - 2))
    for ii, (s_lo, s_hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        coeffs = exact_beam_frame_polynomials(profile, s_lo, s_hi, theta_b)
        polys.append(coeffs)
        elements.append(xt.BFieldExpansion(
            length=s_hi - s_lo,
            ksolc=coeffs['A0'].coef / RIGIDITY0,
            ksc=[coeffs[f'A{m}'].coef / RIGIDITY0 for m in range(1, 6)],
            knl=[0.0], ksl=[0.0],       # in-solenoid corrector (004b)
            s_start=0.0,
            num_integration_steps=steps,
            pkin_const=True))
        names.append(f'{prefix}_{ii:0{width}d}')
    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = PARTICLE_REF.copy()
    return line, polys


def line_field(line, s_start, x, y, s):
    """(Bx, By, Bs) [T] of a solenoid line placed from s_start, at x, y and
    the (sorted) positions s."""
    out = np.zeros((3, len(s)))
    edges = s_start + np.r_[0.0, np.cumsum([el.length for el in line.elements])]
    idx = np.clip(np.searchsorted(edges, s, side='right') - 1, 0,
                  len(line.elements) - 1)
    for ii in np.unique(idx):
        mask = idx == ii
        element = line.elements[ii]
        if not hasattr(element, 'get_field'):
            continue
        s_local = s[mask] - edges[ii]
        fields = field_tesla(element, np.full(mask.sum(), x),
                             np.full(mask.sum(), y), s_local,
                             rigidity0=RIGIDITY0)
        out[:, mask] = np.array([np.asarray(f).ravel() for f in fields])
    return out


def exact_tilted_field(profile, theta):
    """Field of the axisymmetric on-axis profile, tilted as TiltedSolenoid
    (theta = beam angle from the solenoid axis), beam frame. The paraxial
    series is exact for quartic pieces: Bz = b - r^2 b''/4 + r^4 b''''/64,
    Br = -r b'/2 + r^3 b'''/16."""
    derivs = [profile] + [profile.derivative(k) for k in range(1, 5)]
    ct, st = np.cos(theta), np.sin(theta)

    def field(x, y, s):
        x, y, s = np.broadcast_arrays(np.asarray(x, float),
                                      np.asarray(y, float),
                                      np.asarray(s, float))
        x_sol = x * ct + s * st
        z_sol = -x * st + s * ct
        b0, b1, b2, b3, b4 = (np.nan_to_num(d(z_sol)) for d in derivs)
        r2 = x_sol ** 2 + y ** 2
        bz_sol = b0 - r2 / 4 * b2 + r2 ** 2 / 64 * b4
        radial = -b1 / 2 + r2 * b3 / 16
        bx_sol, by_sol = x_sol * radial, y * radial
        return (bx_sol * ct - bz_sol * st, by_sol,
                bx_sol * st + bz_sol * ct)

    return field


def opposite_lobe(s, a1, edge):
    """[start, end] of the B_x lobe past the edge with the sign opposite to
    the edge peak, where |B_x| > LOBE_THRESHOLD x its peak."""
    past = s > edge
    edge_sign = np.sign(a1[np.argmax(np.abs(a1) * (np.abs(s - edge) < 0.1))])
    lobe_values = np.where(past, -edge_sign * a1, 0.0)
    i_peak = int(np.argmax(lobe_values))
    above = lobe_values > LOBE_THRESHOLD * lobe_values[i_peak]
    lo = hi = i_peak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < len(s) - 1 and above[hi + 1]:
        hi += 1
    return s[lo], s[hi], s[i_peak], -edge_sign * lobe_values[i_peak]


# ---------------------------------------------------------- 1. the profile
data = IdeaSolenoidData(args.b0)
case = data.build_case(args.edge, args.eps, args.taper_length)
z_m = case['z_match']
profile = case['total']
print(f'IDEA {args.b0:g} T, edge {args.edge:.2f} m, eps {args.eps:g}, taper '
      f'{args.taper_length:g} m: z_p = {case["z_peak"]:.3f} m, z_m = '
      f'{z_m:.2f} m, {len(case["element_boundaries"]) - 1} elements')
if z_m > SLOT_HALF_LENGTH + 1e-9:
    raise SystemExit(f'z_m = {z_m:.2f} m does not fit in the {SLOT_HALF_LENGTH} '
                     'm between the IP and QD0: choose a smaller edge or taper.')

# --------------------------------------------- 2. first-corrector segments
def polys_on_grid(polys, boundaries, key, s):
    """Evaluate one coefficient of per-element polynomials at s."""
    idx = np.clip(np.searchsorted(boundaries, s, side='right') - 1, 0,
                  len(polys) - 1)
    return np.array([polys[i][key](si - boundaries[i])
                     for i, si in zip(idx, s)])


_, base_polys = bfe_line_from_profile(profile, case['element_boundaries'],
                                      TILT_ANGLE, STEPS, 'base')
s_fine = np.round(np.arange(0.0, z_m + 5e-4, 1e-3), 9)
a1_fine = polys_on_grid(base_polys, case['element_boundaries'], 'A1', s_fine)
lobe_lo, lobe_hi, lobe_peak_s, lobe_peak = opposite_lobe(s_fine, a1_fine,
                                                         args.edge)


def map_from_010a(s010a):
    return lobe_lo + ((s010a - LOBE_010A[0]) * (lobe_hi - lobe_lo)
                      / (LOBE_010A[1] - LOBE_010A[0]))


corr_segments = [
    {'s_range': [round(float(map_from_010a(lo)), 3),
                 round(float(map_from_010a(hi)), 3)],
     'weight': weight}
    for (lo, hi), weight in CORR_SEGMENTS_010A]
print(f'Opposite-sign B_x lobe: {lobe_lo:.3f}-{lobe_hi:.3f} m, peak '
      f'{lobe_peak * 1e3:+.1f} mT at {lobe_peak_s:.3f} m (010a: '
      f'{LOBE_010A[0]}-{LOBE_010A[1]} m)')
for seg in corr_segments:
    print(f'  first-corrector segment {seg["s_range"][0]:.3f}-'
          f'{seg["s_range"][1]:.3f} m, weight {seg["weight"]:.4f}')

seg_edges = np.array([e for seg in corr_segments for e in seg['s_range']])
boundaries = case['element_boundaries']
# no near-zero slivers if a segment end is (almost) on a node
seg_edges = seg_edges[~np.any(np.isclose(seg_edges[:, None], boundaries,
                                         rtol=0, atol=1e-6), axis=1)]
boundaries = np.unique(np.round(np.r_[boundaries, seg_edges, -seg_edges], 12))

# ------------------------------------------------- 3. BFieldExpansion line
main_line, polys = bfe_line_from_profile(profile, boundaries, TILT_ANGLE,
                                         STEPS, 'idea_sol')
lengths = np.diff(boundaries)
print(f'Main line: {len(main_line.elements)} BFieldExpansion elements '
      f'({lengths.min() * 100:.1f}-{lengths.max() * 100:.1f} cm), '
      f'{-boundaries[0]:.2f} m either side of the IP, {STEPS} RK4 steps each')

main_bs_integral = sum(float(el.ksoll[0]) for el in main_line.elements) * RIGIDITY0
print(f'int Bs: BFE line {main_bs_integral:.6f} T m, spline '
      f'{profile.integrate(boundaries[0], boundaries[-1]):.6f} T m')

# --------------------------------------------- 4. compensation solenoids
comp_template = xt.Line.from_dict(
    xt.json.load(COMP_TEMPLATE_JSON)['lines']['compensation_solenoid'])
comp_bs_integral = sum(el.bs[4] * el.length * el.scale_b
                       for el in comp_template.elements
                       if isinstance(el, xt.SplineBoris))
comp_rescale = main_bs_integral / (-2 * comp_bs_integral)
for el in comp_template.elements:
    if isinstance(el, xt.SplineBoris):
        el.scale_b *= comp_rescale
comp_line = sb_line_to_bfe_line(comp_template, rigidity0=RIGIDITY0)
comp_line.particle_ref = PARTICLE_REF.copy()
comp_bfe_integral = sum(float(el.ksoll[0]) for el in comp_line.elements
                        if isinstance(el, xt.BFieldExpansion)) * RIGIDITY0
print(f'Compensation solenoid: int Bs {comp_bs_integral:.6f} -> '
      f'{comp_bfe_integral:.6f} T m (x {comp_rescale:.6f}); net with the '
      f'main: {main_bs_integral + 2 * comp_bfe_integral:.2e} T m')

# ------------------------------------- 5a. convention check against 004a 2 T
half_2t = half_length_for_b0(2.0)
s_ax = np.linspace(-2.399, 2.399, 4799)
analytic = SolenoidField(L=2 * half_2t, a=MAIN_SOLENOID_A, B0=2.0, z0=0)
bs_axis = np.asarray(analytic.get_field(0 * s_ax, 0 * s_ax, s_ax)[2])
# 6 mm elements: with the 2.4 cm of 004a the profile fit itself (zero end
# condition at +-2.399 m, where the field is still 6 mT) leaves 0.8 mT here.
fit_2t = LongitudinalFitter(s_start=-2.399, s_end=2.399, n_elements=800)
fit_2t.fit(s_ax, bs_axis, ('Bs', 0))
line_2t, _ = bfe_line_from_profile(fit_2t.splines[('Bs', 0)], fit_2t.nodes,
                                   TILT_ANGLE, STEPS, 'check_2t')
template_2t = xt.Line.from_dict(
    xt.json.load(COMP_TEMPLATE_JSON)['lines']['main_solenoid'])
s_chk = np.linspace(-2.2, 2.2, 2201)        # inside 004a's 0.15 m end tapers
tilted_2t = TiltedSolenoid(L=2 * half_2t, a=MAIN_SOLENOID_A, B0=2.0,
                           theta=THETA)
print('Convention check, analytic 2 T main solenoid (|s| <= 2.2 m): this '
      'code vs TiltedSolenoid (exact) and vs the 004a template:')
for x0, y0 in ((0.0, 0.0), (0.005, 0.0), (0.0, 0.005)):
    mine = line_field(line_2t, -2.399, x0, y0, s_chk)
    for label, ref in (
            ('TiltedSolenoid', np.array([np.asarray(f) for f in
                                         tilted_2t.get_field(
                                             np.full_like(s_chk, x0),
                                             np.full_like(s_chk, y0),
                                             s_chk)])),
            ('004a template', line_field(template_2t, -2.399, x0, y0,
                                         s_chk))):
        print(f'  x = {x0 * 1e3:.0f} mm, y = {y0 * 1e3:.0f} mm, vs {label:14s}'
              ': max |dB| '
              + ', '.join(f'{c} {np.abs(mine[k] - ref[k]).max() * 1e3:.3f} mT'
                          for k, c in enumerate(('Bx', 'By', 'Bs')))
              + f'; Bx(IP) {mine[0][1100]:+.5f} vs {ref[0][1100]:+.5f} T')

# ---------------------------------- 5b. off-axis field against the exact one
exact = exact_tilted_field(profile, THETA)
s_main = np.linspace(boundaries[0], boundaries[-1], 9201)
offsets = ((0.0, 0.0), (0.005, 0.0), (0.0, 0.005), (0.01, 0.0), (0.0, 0.01),
           (0.007, 0.007))
residuals = {}
print('Field of the BFE line minus the exact tilted field:')
for x0, y0 in offsets:
    mine = line_field(main_line, boundaries[0], x0, y0, s_main)
    ref = np.array(exact(x0, y0, s_main))
    residuals[x0, y0] = mine - ref
    # within 0.3 mm past a knot the two switch spline pieces on differently
    # tilted planes (z' = t_k here, s = t_k in the elements)
    away = np.min(np.abs(np.abs(s_main)[:, None] - boundaries[None, :]),
                  axis=1) > 3e-4
    print(f'  x = {x0 * 1e3:4.1f} mm, y = {y0 * 1e3:4.1f} mm: max |dB| '
          + ', '.join(f'{c} {np.abs(mine[k] - ref[k]).max() * 1e6:7.2f}'
                      f' ({np.abs(mine[k] - ref[k])[away].max() * 1e6:.2f})'
                      for k, c in enumerate(('Bx', 'By', 'Bs')))
          + ' uT (more than 0.3 mm from a knot)')

# ------------------------------------------- 5c. tracking, steps, symplecticity
boris = xt.BorisSpatialIntegrator(
    fieldmap_callable=exact, s_start=boundaries[0], s_end=boundaries[-1],
    n_steps=int(round((boundaries[-1] - boundaries[0]) / 2.5e-4)))
delta = np.array([0.0, 1e-2])
p0 = xt.Particles(mass0=PARTICLE_REF.mass0, q0=PARTICLE_REF.q0,
                  energy0=PARTICLE_REF.energy0[0], x=1e-3,
                  px=-1e-4 * (1 + delta), y=1e-3, py=2e-4, delta=delta)
p_boris = p0.copy()
boris.track(p_boris)
coords = ('x', 'px', 'y', 'py')


def track_with_steps(n_steps):
    line, _ = bfe_line_from_profile(profile, boundaries, TILT_ANGLE, n_steps,
                                    'trk')
    line.build_tracker()
    p = p0.copy()
    line.track(p)
    return np.array([getattr(p, c) for c in coords]), line


reference_steps = 64
ref_exit, _ = track_with_steps(reference_steps)
print(f'Tracking from s = {boundaries[0]:.2f} m, delta = {delta}; max exit '
      f'difference over x, px, y, py:')
print(f'  Boris through the exact field (0.25 mm steps) vs BFE '
      f'{reference_steps} steps: '
      f'{np.abs(np.array([getattr(p_boris, c) for c in coords]) - ref_exit).max():.2e}')
for n_steps in (1, 2, 4, 8, 16, 32):
    exit_n, _ = track_with_steps(n_steps)
    flag = '  <- default' if n_steps == STEPS else ''
    print(f'  BFE {n_steps:2d} steps vs {reference_steps}: '
          f'{np.abs(exit_n - ref_exit).max():.2e}{flag}')
main_line.build_tracker()
for n_steps in sorted({4, STEPS, 16}):
    _, line_n = track_with_steps(n_steps)
    print(f'Symplecticity error |R^T S R - S|_2, BFE line, {n_steps:2d} steps: '
          f'{symplectic_error(line=line_n, particle_ref=PARTICLE_REF):.2e}')

# ------------------------------------------------------------------- save
metadata = {
    'description': 'IDEA main + screening total field (idea_solenoid_field, '
                   '001e) as BFieldExpansion elements; compensation '
                   'solenoids from the 004a 2 T template, rescaled',
    'b0': args.b0, 'edge': args.edge, 'eps': args.eps,
    'taper_length': args.taper_length,
    'z_peak': float(case['z_peak']), 'z_match': float(z_m),
    'theta_b': TILT_ANGLE, 'theta': THETA,
    'element_boundaries': [float(b) for b in boundaries],
    'num_integration_steps': STEPS,
    'first_corr_segments': corr_segments,
    'opposite_bx_lobe': [float(lobe_lo), float(lobe_hi)],
    'main_bs_integral': main_bs_integral,
    'comp_bs_integral': comp_bfe_integral,
    'comp_rescale_from_004a_2T': comp_rescale,
}
xt.json.dump({'metadata': metadata,
              'lines': {'main_solenoid': main_line.to_dict(),
                        'compensation_solenoid': comp_line.to_dict()}},
             OUTPUT_JSON, indent=1)
print(f'Wrote {OUTPUT_JSON} ({OUTPUT_JSON.stat().st_size / 1e3:.0f} kB)')

# ------------------------------------------------------------------- plot
s_plot = np.linspace(boundaries[0], boundaries[-1], 9201)
on_axis = line_field(main_line, boundaries[0], 0.0, 0.0, s_plot)
exact_axis = exact(0.0, 0.0, s_plot)
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axes[0].plot(s_plot, on_axis[2], color='black', lw=1.5, label='BFE line')
axes[0].plot(s_plot, exact_axis[2], color='C1', lw=1, ls='--',
             label='exact tilted field')
axes[0].set_ylabel('$B_s$ on axis [T]')
axes[1].plot(s_plot, on_axis[0], color='black', lw=1.5, label='BFE line')
axes[1].plot(s_plot, exact_axis[0], color='C1', lw=1, ls='--',
             label='exact tilted field')
axes[1].set_ylabel('$B_x$ on axis [T]')
for (x0, y0), color in zip(((0.01, 0.0), (0.0, 0.01), (0.007, 0.007)),
                           ('C0', 'C2', 'C3')):
    res = residuals[x0, y0]
    axes[2].plot(s_main, np.max(np.abs(res), axis=0) * 1e6, color=color,
                 lw=1, label=f'x = {x0 * 1e3:g} mm, y = {y0 * 1e3:g} mm')
axes[2].set_yscale('log')
axes[2].set_ylabel(r'max$_c$ |BFE $-$ exact| [$\mu$T]')
axes[2].set_xlabel('s from the IP [m]')
for ax in axes:
    for seg in corr_segments:
        for sign in (1, -1):
            lo, hi = sorted(sign * np.array(seg['s_range']))
            ax.axvspan(lo, hi, color='C4', alpha=0.15, lw=0)
    for sign in (1, -1):
        ax.axvline(sign * args.edge, color='gray', lw=0.8, ls=':')
        ax.axvline(sign * case['z_peak'], color='gray', lw=0.8, ls='-.')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
axes[1].plot([], [], color='C4', alpha=0.3, lw=6,
             label='first-corrector segments')
axes[1].legend(fontsize=8, loc='upper right')
fig.suptitle(f'IDEA solenoid as BFieldExpansion, {args.b0:g} T, edge '
             f'{args.edge:.2f} m, $\\varepsilon$ = {args.eps:g}: '
             f'{len(main_line.elements)} elements, $\\theta_b$ = '
             f'{TILT_ANGLE:+g} rad')
fig.tight_layout()
PLOT_DIR.mkdir(parents=True, exist_ok=True)
plot_path = PLOT_DIR / f'004a_idea_bfe_line_{TAG}.pdf'
fig.savefig(plot_path, metadata=pdf_metadata())
print(f'Saved {plot_path}')

if not args.no_show:
    plt.show()
