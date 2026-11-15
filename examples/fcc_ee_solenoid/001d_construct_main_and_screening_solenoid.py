"""
Construct the on-axis Bz(z) of the IDEA main and screening solenoids from the
digitized Opera curves (IDEA_solenoid_beam_axis_raw.json, plotted by 001c).

Curves (one side of the IP, z = 0..~6 m):
    G = Main Solenoid Only
    S = Screening Only, *as plotted* (positive); the screening solenoid's
        own field is -S
    B = Both Energized = G - S

Step 1 -- remove the ripples from B beyond Z_CUT.
    The ripples in B come from the screening solenoid: where all three
    curves are visible (z = 4.1-5.0 m) G is smooth and S + B reproduces it to
    ~1 mT, so the dip at 4.45 m and the bump at 4.75 m are in S (screening
    field exceeds the main field, then ends at ~4.6 m).
    Method: clean B, then S = G - B_clean, so that G = S + B holds exactly.
    B is the right curve to clean: there the ripple sits on a ~0 baseline,
    whereas in S it is a few mT on top of a curve falling from 1.7 T.
    B_clean = B_raw for z < Z_CUT - BLEND, an exponential tail fitted to B on
    FIT_WINDOW beyond Z_CUT, and a cosine blend in between.

Note: the cleaned S also absorbs the places where the raw curves do not add
up (G - S_raw - B_raw is 5-9 mT at z ~ 1.85-2.6 m; the Opera model has
iron), and the 1-2 px digitization jitter of G where G is hidden under S
(z ~ 2-4.1 m). The bottom panel separates the two.

    Choice (2026-11-15): idealized screening. Cleaning all of B beyond
    Z_CUT also removes the end of the screening solenoid (B dips to -25 mT
    at 4.45 m and rises to +35 mT at 4.75 m in the raw data), so the
    cleaned S cancels G up to the end of the data instead of stopping at
    ~4.6 m.

Step 2 -- fit G and S with LongitudinalFitter; total = G_fit - S_fit.
    The stage-2 fitter of the SplineBoris pipeline
    (xtrack/_temp/splineboris/longitudinal_fitter.py): least-squares C3
    quartic B-spline on uniform nodes, component ("Bs", 0), on the
    solenoid's own axis, data mirrored about the IP (Bs(-z) = Bs(z)),
    field forced to zero at +-S_END, integral preserved. Its elements map
    one-to-one onto SplineBoris Spline4 / BFieldExpansion.
    G: raw digitized points. MAIN_ELEMENT_LENGTH = 0.2 m: 0.1 m fits the
    digitization noise (Bs'''' jumps by +-600 T/m^4 between elements),
    0.3 m starts to bias the steep part near z = 3 m.
    S: the cleaned (idealized) S of step 1, on the 1 mm grid. Its turn-on
    edge (z ~ 1.25-1.55 m, up to ~9 T/m) is much sharper than G:
    with 0.2 m elements the spline misses it by up to ~70 mT and rings back
    to the IP (~40 mT at z < 1.2 m); 0.1 m gives ~20 mT at the edge and
    ~4 mT near the IP (SCREENING_ELEMENT_LENGTH).
    (One analytic SolenoidField fits G only to +-35 mT: G droops near the IP
    like a large coil but falls off at z ~ 2.5-4 m faster than an air coil
    can -- the Opera model has iron.)
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from aperture_study_io import PLOT_DIR, pdf_metadata
from xtrack._temp.splineboris.longitudinal_fitter import LongitudinalFitter

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / 'IDEA_solenoid_beam_axis_raw.json'

DZ = 1e-3                  # m, common grid step (raw resolution ~3.4 mm)
Z_CUT = 2.05               # m, ripples in B start here
FIT_WINDOW = (1.70, Z_CUT) # m, exponential fit of the clean edge of B
BLEND = 0.05               # m, cosine blend raw -> fit, ending at Z_CUT

S_END = 6.0                # m, spline range +-S_END (data end at 5.95 m)
MAIN_ELEMENT_LENGTH = 0.2  # m, spline element (knot interval) length for G
SCREENING_ELEMENT_LENGTH = MAIN_ELEMENT_LENGTH  # m, for S (0.1 m resolves its edge)


def load_raw_curve(curves, key):
    """Raw points with repeated z (row-scanned steep parts) averaged."""
    z = np.array(curves[key]['z'])
    bz = np.array(curves[key]['Bz'])
    z_unique, inverse = np.unique(z, return_inverse=True)
    bz_mean = np.bincount(inverse, bz) / np.bincount(inverse)
    return z_unique, bz_mean


with open(DATA_FILE) as fid:
    curves = json.load(fid)['curves']

z_g, bz_g = load_raw_curve(curves, 'main_only')
z_s, bz_s = load_raw_curve(curves, 'screening_only')
z_b, bz_b = load_raw_curve(curves, 'both')

# Common grid over the main-solenoid data. S and B end at 5.73 m (Opera logo);
# beyond that the raw S, B are only used for comparison and are left as NaN.
z = np.round(np.arange(0.0, z_g.max() + 0.5 * DZ, DZ), 9)
main = np.interp(z, z_g, bz_g)
screening_raw = np.interp(z, z_s, bz_s, right=np.nan)
both_raw = np.interp(z, z_b, bz_b, right=np.nan)

# ------------------------------------------------- step 1: clean B, derive S
mask_fit = (z >= FIT_WINDOW[0]) & (z <= FIT_WINDOW[1])
slope, intercept = np.polyfit(z[mask_fit], np.log(both_raw[mask_fit]), 1)
decay_length = -1.0 / slope
both_tail = np.exp(intercept + slope * z)

weight = np.clip((z - (Z_CUT - BLEND)) / BLEND, 0.0, 1.0)
weight = 0.5 * (1.0 - np.cos(np.pi * weight))
both_raw_or_tail = np.where(np.isnan(both_raw), both_tail, both_raw)
both_clean = (1.0 - weight) * both_raw_or_tail + weight * both_tail
screening_clean = main - both_clean

# ------------------------------------------------------------------ report
def integral(bz, z_min=0.0, z_max=np.inf):
    mask = (z >= z_min) & (z <= z_max) & np.isfinite(bz)
    return np.trapezoid(bz[mask], z[mask])

z_common = np.nanmax(np.where(np.isfinite(both_raw), z, np.nan))
mismatch_raw = main - screening_raw - both_raw

print(f'Exponential tail of B fitted on z = {FIT_WINDOW[0]:.2f}-{FIT_WINDOW[1]:.2f} m: '
      f'decay length {decay_length * 100:.1f} cm, '
      f'B({Z_CUT:.2f} m) = {both_tail[np.argmin(abs(z - Z_CUT))] * 1e3:.2f} mT '
      f'(raw {both_raw[np.argmin(abs(z - Z_CUT))] * 1e3:.2f} mT)')
print(f'Integrals over z = 0-{z_common:.2f} m [T m]:')
for name, raw, clean in [('main', main, main),
                         ('screening (plotted)', screening_raw, screening_clean),
                         ('both', both_raw, both_clean)]:
    print(f'  {name:20s} raw {integral(raw, 0, z_common):.4f}   '
          f'clean {integral(clean, 0, z_common):.4f}')
print(f'  int B over z = {Z_CUT:.2f}-{z_common:.2f} m: raw '
      f'{integral(both_raw, Z_CUT, z_common) * 1e3:+.2f} mT m, clean '
      f'{integral(both_clean, Z_CUT, z_common) * 1e3:+.2f} mT m')
for z_min, z_max in [(0.0, 1.2), (1.85, 2.6), (2.6, 4.1), (4.1, z_common)]:
    mask = (z >= z_min) & (z <= z_max)
    print(f'  raw G - S - B on z = {z_min:.2f}-{z_max:.2f} m: '
          f'mean {np.nanmean(mismatch_raw[mask]) * 1e3:+.1f} mT, '
          f'rms {np.sqrt(np.nanmean(mismatch_raw[mask] ** 2)) * 1e3:.1f} mT')

# -------------------------------------------------------------------- plot
fig, (ax_full, ax_both, ax_diff) = plt.subplots(
    3, 1, figsize=(8, 9), sharex=True,
    gridspec_kw={'height_ratios': [2, 1.2, 1.2]})

ax_full.plot(z, main, color='green', lw=1.5, label='Main Solenoid Only (G)')
ax_full.plot(z, screening_raw, color='red', lw=1, alpha=0.5,
             label='Screening Only, raw (S, as plotted)')
ax_full.plot(z, screening_clean, color='darkred', lw=1.5, ls='--',
             label='Screening Only, clean = G - B clean')
ax_full.plot(z, both_raw, color='gray', lw=1, alpha=0.7,
             label='Both Energized, raw (B)')
ax_full.plot(z, both_clean, color='black', lw=1.5, ls='--',
             label='Both Energized, clean')
ax_full.set_ylabel(r'$B_z$ [T]')
ax_full.set_ylim(-0.05, 2.1)
ax_full.legend(fontsize=8, loc='upper right')
ax_full.set_title('IDEA solenoids: ripple removal (step 1)')

ax_both.axhline(0, color='gray', lw=0.8)
ax_both.axvspan(*FIT_WINDOW, color='tab:blue', alpha=0.12,
                label='exponential fit window')
ax_both.axvline(Z_CUT, color='tab:blue', lw=1, ls=':',
                label=f'Z_CUT = {Z_CUT} m')
ax_both.plot(z, both_raw * 1e3, color='gray', lw=1, label='B raw')
ax_both.plot(z, both_clean * 1e3, color='black', lw=1.5, ls='--',
             label='B clean')
ax_both.set_ylim(-40, 60)
ax_both.set_ylabel(r'$B_z$ [mT]')
ax_both.legend(fontsize=8, loc='upper right')

ax_diff.axhline(0, color='gray', lw=0.8)
ax_diff.plot(z, (screening_clean - screening_raw) * 1e3, color='darkred',
             lw=1, label='S clean - S raw')
ax_diff.plot(z, (both_clean - both_raw) * 1e3, color='black', lw=1,
             label='B clean - B raw')
ax_diff.plot(z, mismatch_raw * 1e3, color='tab:orange', lw=1, alpha=0.8,
             label='G - S raw - B raw (raw curves do not add up)')
ax_diff.set_ylim(-40, 40)
ax_diff.set_ylabel(r'$\Delta B_z$ [mT]')
ax_diff.set_xlabel('z [m]')
ax_diff.legend(fontsize=8, loc='upper left')
ax_diff.set_xlim(0, z.max())

for ax in (ax_full, ax_both, ax_diff):
    ax.grid(True, alpha=0.4)

fig.tight_layout()

PLOT_DIR.mkdir(parents=True, exist_ok=True)
out_path = PLOT_DIR / 'IDEA_solenoid_beam_axis_ripple_removal.pdf'
fig.savefig(out_path, bbox_inches='tight', metadata=pdf_metadata())
print(f'Saved {out_path}')

# ------------------------- step 2: fit G and S with LongitudinalFitter
def fit_on_axis_bs(z_data, bs_data, element_length):
    """C3 quartic B-spline fit of ("Bs", 0) on +-S_END, data mirrored about
    the IP, field zero at both ends, integral preserved."""
    z_mirror = np.r_[-z_data[::-1], z_data]
    bs_mirror = np.r_[bs_data[::-1], bs_data]
    unique = np.r_[True, np.diff(z_mirror) > 0]      # z = 0 only once
    fitter = LongitudinalFitter(
        s_start=-S_END, s_end=S_END,
        n_elements=int(round(2 * S_END / element_length)),
        end_condition='zero', preserve_integral=True)
    fitter.fit(z_mirror[unique], bs_mirror[unique], ('Bs', 0))
    return fitter, fitter.splines[('Bs', 0)]


# G from the raw digitized points; S is derived (step 1) and lives on the grid.
main_fitter, main_spline = fit_on_axis_bs(z_g, bz_g, MAIN_ELEMENT_LENGTH)
screening_fitter, screening_spline = fit_on_axis_bs(
    z, screening_clean, SCREENING_ELEMENT_LENGTH)

z_fit = np.round(np.arange(0.0, S_END + 0.5 * DZ, DZ), 9)
main_fit = main_spline(z_fit)
screening_fit = screening_spline(z_fit)
total_fit = main_fit - screening_fit             # = Both Energized

main_residual = main_spline(z_g) - bz_g          # at the digitized points
screening_residual = screening_spline(z) - screening_clean
total_residual = (main_spline(z) - screening_spline(z)) - both_clean

print(f'\nStep 2: LongitudinalFitter fits on z = {-S_END:.2f}..{S_END:.2f} m '
      f'(mirrored data, end_condition="zero", integral preserved)')
for label, fitter, spline, z_data, residual in [
        ('G', main_fitter, main_spline, z_g, main_residual),
        ('S', screening_fitter, screening_spline, z, screening_residual),
        ('total - B clean', None, None, z, total_residual)]:
    if fitter is not None:
        d2_max = np.abs(spline.derivative(2)(z_fit)).max()
        d4_max = np.abs(spline.derivative(4)(z_fit)).max()
        print(f'  {label}: {fitter.n_elements} elements of {fitter.delta:.3f} m, '
              f'max |d2 Bs/dz2| = {d2_max:.2f} T/m^2, '
              f'max |d4 Bs/dz4| = {d4_max:.0f} T/m^4')
    for z_min, z_max in [(0.0, 1.2), (1.2, 1.7), (1.7, 5.73), (5.73, z_data.max())]:
        mask = (z_data >= z_min) & (z_data <= z_max)
        if mask.any():
            print(f'    fit - data, {label} on z = {z_min:.2f}-{z_max:.2f} m: '
                  f'rms {np.sqrt(np.mean(residual[mask] ** 2)) * 1e3:.2f} mT, '
                  f'max {np.abs(residual[mask]).max() * 1e3:.1f} mT')
print(f'  int Bs over z = 0-{S_END:.2f} m [T m]: '
      f'G fit {np.trapezoid(main_fit, z_fit):.4f} (data {np.trapezoid(bz_g, z_g):.4f}), '
      f'S fit {np.trapezoid(screening_fit, z_fit):.4f} '
      f'(data {np.trapezoid(screening_clean, z):.4f}), '
      f'total {np.trapezoid(total_fit, z_fit):.4f} '
      f'(B clean {np.trapezoid(both_clean, z):.4f})')

fig_fit, (ax_fit, ax_res, ax_d2, ax_d4) = plt.subplots(
    4, 1, figsize=(8, 10), sharex=True,
    gridspec_kw={'height_ratios': [2, 1.2, 1, 1]})
ax_fit.plot(z_g, bz_g, color='green', lw=1.5, label='Main Solenoid Only (G)')
ax_fit.plot(z_fit, main_fit, color='tab:blue', lw=1.2, ls='--',
            label=f'G fit ({main_fitter.delta:.2f} m elements)')
ax_fit.plot(z, screening_clean, color='red', lw=1.5,
            label='Screening, clean (S, as plotted; field is $-B_s$)')
ax_fit.plot(z_fit, screening_fit, color='darkred', lw=1.2, ls='--',
            label=f'S fit ({screening_fitter.delta:.2f} m elements)')
ax_fit.plot(z, both_clean, color='gray', lw=2.5, alpha=0.5,
            label='Both Energized, clean (B)')
ax_fit.plot(z_fit, total_fit, color='black', lw=1.2, ls='--',
            label='total = G fit $-$ S fit')
ax_fit.set_ylabel(r'$B_s$ [T]')
ax_fit.set_ylim(-0.1, 2.2)
ax_fit.legend(fontsize=8, loc='upper right')
ax_fit.set_title('IDEA solenoids: spline fits (step 2)')

ax_res.axhline(0, color='gray', lw=0.8)
ax_res.plot(z_g, main_residual * 1e3, '.', color='tab:blue', ms=1.5,
            label='G fit $-$ G')
ax_res.plot(z, screening_residual * 1e3, color='darkred', lw=1,
            label='S fit $-$ S')
ax_res.plot(z, total_residual * 1e3, color='black', lw=1, ls=':',
            label='total $-$ B clean')
ax_res.set_ylabel(r'fit $-$ data [mT]')
ax_res.legend(fontsize=8, loc='upper right')

for spline, color, label in [(main_spline, 'tab:blue', 'G'),
                             (screening_spline, 'darkred', 'S')]:
    ax_d2.plot(z_fit, spline.derivative(2)(z_fit), color=color, lw=1, label=label)
    ax_d4.plot(z_fit, spline.derivative(4)(z_fit), color=color, lw=1, label=label)
ax_d2.set_ylabel(r"$B_s''$ [T/m$^2$]")
ax_d2.legend(fontsize=8, loc='upper right')
ax_d4.set_ylabel(r"$B_s''''$ [T/m$^4$]")
ax_d4.set_xlabel('z [m]')
ax_d4.set_xlim(0, S_END)

for ax in (ax_fit, ax_res, ax_d2, ax_d4):
    ax.grid(True, alpha=0.4)
fig_fit.tight_layout()

out_path = PLOT_DIR / 'IDEA_solenoid_beam_axis_spline_fits.pdf'
fig_fit.savefig(out_path, bbox_inches='tight', metadata=pdf_metadata())
print(f'Saved {out_path}')

plt.show()
