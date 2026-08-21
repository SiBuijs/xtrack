import numpy as np
import xtrack as xt
import pandas as pd

#################################################
# Polynomial fit on the data from the field map #
#################################################

# Load the raw field map data from shared test_data
field_map_path = "../../test_data/sls/simona_field_map.txt"
df_raw_data = pd.read_csv(
    field_map_path,
    sep=r"\s+",
    header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],  # TubeFitter expects Bx/By, not Bskew/Bnorm
).set_index(["X", "Y", "Z"])

# Use fitting procedure to extract field and derivatives on the reference
# trajectory ("tube approach", B. Riemann & M. Aiba, IPAC2021 TUPAB238). This
# class is taylored for this example data, use your own fitting procedure for
# other datasets.
from xtrack._temp.splineboris.tube_fitter import TubeFitter
from xtrack._temp.splineboris.splineboris_sequence import SplineBorisSequence
MULTIPOLE_ORDER = 4  # deg=2 -> Bx/By fit up to and including order 2

# Rigid transverse misalignment applied to the undulator field elements
# (both models) -- change these to scan different misalignments.
SHIFT_X = 0.000#5  # [m]
SHIFT_Y = 0.000  # [m]

# e+/e- anomalous magnetic moment (g-2)/2, needed for spin (Thomas-BMT)
# tracking -- same value as 012_sls_undulator_spin_emittance_evolution.py.
ANOMALOUS_MAGNETIC_MOMENT = 0.00115965218128
# Initial spin, fully polarized vertical (matches 012's tracked bunch).
SPIN_X0 = 0.0
SPIN_Y0 = 1.0
SPIN_Z0 = 0.0

fitter = TubeFitter(
    raw_data=df_raw_data,
    distance_unit=0.001, # dataset uses mm
    n_frames=4401,#1701,
    deg=MULTIPOLE_ORDER - 1,
)
fitter.fit()

#######################################
# Build Xsuite model of the undulator #
#######################################

env = xt.Environment()
env.set_particle_ref('positron', p0c=2.7e9)
env.particle_ref.anomalous_magnetic_moment = ANOMALOUS_MAGNETIC_MOMENT

# TubeFitter.to_line() builds a Line of SplineBoris elements directly from
# the fit -- one element per longitudinal region (region grid shared across
# all field components, so no boundary reconciliation is needed), one Boris
# step per interval between adjacent field-map points. It lives in its own
# implicit Environment, so import it into `env` to use it alongside the
# correctors built below.
undulator_line = fitter.to_line(multipole_order=MULTIPOLE_ORDER, steps_per_point=1,
                                 shift_x=SHIFT_X, shift_y=SHIFT_Y)
undulator = env.import_line(undulator_line, line_name='undulator')

###########################################################################
# Install thin dipole correctors at the edges of the undulator to control #
# trajectory along the undulator.                                         #
###########################################################################

# Knobs controlling the correctors
env['k0l_corr1'] = 0.
env['k0l_corr2'] = 0.
env['k0l_corr3'] = 0.
env['k0l_corr4'] = 0.
env['k0sl_corr1'] = 0.
env['k0sl_corr2'] = 0.
env['k0sl_corr3'] = 0.
env['k0sl_corr4'] = 0.

# Create correcto elements
env.new('corr1', xt.Multipole, knl=['k0l_corr1'], ksl=['k0sl_corr1'])
env.new('corr2', xt.Multipole, knl=['k0l_corr2'], ksl=['k0sl_corr2'])
env.new('corr3', xt.Multipole, knl=['k0l_corr3'], ksl=['k0sl_corr3'])
env.new('corr4', xt.Multipole, knl=['k0l_corr4'], ksl=['k0sl_corr4'])

# Insert correctors
l_undulator = undulator.get_length()
undulator.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_undulator - 0.1),
    env.place('corr4', at=l_undulator - 0.02),
], s_tol=5e-3) # large s_tol avoids slicing the SplineBoris elements

# Use optimizer to control the orbit
opt = undulator.match(
    solve=False,
    betx=1, bety=1,
    include_collective=True,
    vary=xt.VaryList(['k0l_corr1', 'k0sl_corr1',
                      'k0l_corr2', 'k0sl_corr2',
                      'k0l_corr3', 'k0sl_corr3',
                      'k0l_corr4', 'k0sl_corr4',
                      ], step=1e-6),
    targets=[
        xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
        xt.TargetSet(x=0., y=0, at='corr2'),
        xt.TargetSet(x=0., y=0, at='corr3')
        ],
)
opt.solve()

###############################
# Save undulator to json file #
###############################

undulator.to_json('sls_undulator.json')

#############################################################################
# Build a Multipole-based undulator (coarse thick-Multipole approximation, #
# TubeFitter.to_multipole_line()) from the same fit, reusing the SAME      #
# (already matched) correctors rather than rematching independently -- so #
# the comparison below isolates the field-model difference between        #
# SplineBoris and Multipole instead of letting each model's own           #
# correction reabsorb it (same rationale as the "shared" corrector        #
# strategy in 012_sls_undulator_spin_emittance_evolution.py).             #
#############################################################################

multipole_undulator_line = fitter.to_multipole_line(
    multipole_order=MULTIPOLE_ORDER, p0c=2.7e9, field_at='midpoint',
    shift_x=SHIFT_X, shift_y=SHIFT_Y)
undulator_mult = env.import_line(multipole_undulator_line, line_name='undulator_mult')

# Same region grid as to_line() (shared across every field component), so
# the two undulators have identical length -- the same corrector insertion
# offsets apply unchanged.
undulator_mult.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_undulator - 0.1),
    env.place('corr4', at=l_undulator - 0.02),
], s_tol=5e-3)

##################################
# Plot orbit along the undulator #
##################################

import matplotlib.pyplot as plt

tw_undulator = undulator.twiss4d(
    betx=1, bety=1, spin=True, spin_x=SPIN_X0, spin_y=SPIN_Y0, spin_z=SPIN_Z0)
tw_undulator_mult = undulator_mult.twiss4d(
    betx=1, bety=1, spin=True, spin_x=SPIN_X0, spin_y=SPIN_Y0, spin_z=SPIN_Z0)

# The correctors are matched to x=y=0 at the end of the SplineBoris undulator
# only -- the Multipole undulator reuses those same corrector strengths
# without its own matching, so any nonzero end-orbit here is a direct,
# model-independent measure of the SplineBoris-vs-Multipole field
# discrepancy (rather than each model's own correction reabsorbing it).
dx_end = tw_undulator_mult.x[-1] - tw_undulator.x[-1]
dy_end = tw_undulator_mult.y[-1] - tw_undulator.y[-1]
print(f"Final orbit difference (Multipole - SplineBoris): "
      f"dx = {dx_end:.6e} m, dy = {dy_end:.6e} m")

fig_orbit = plt.figure(1, figsize=(10, 6))
ax_orbit = fig_orbit.add_subplot(111, projection='3d')
ax_orbit.plot(tw_undulator.s, tw_undulator.x, tw_undulator.y, label='SplineBoris')
ax_orbit.plot(tw_undulator_mult.s, tw_undulator_mult.x, tw_undulator_mult.y, '-.', label='Multipole')
ax_orbit.set_xlabel('s [m]')
ax_orbit.set_ylabel('x [m]')
ax_orbit.set_zlabel('y [m]')
ax_orbit.set_title('Undulator trajectory: SplineBoris vs Multipole')
ax_orbit.legend()
fig_orbit.savefig('splineboris_undulator_trajectory.png', dpi=200,
                  bbox_inches='tight')

#################################################################
# Plot the field the particle actually feels along its orbit,   #
# using the (x, y, s) trajectory from the SplineBoris Twiss.    #
#################################################################

# There's no ready-made "plot field along an arbitrary trajectory" method,
# but SplineBorisSequence.get_field(x, y, s) (built directly from the same
# fitter.df_fit_pars used by to_line()) is the closest built-in evaluator --
# confirmed in examples/splineboris/claude_notes/tube_fitter_to_line.md to
# reproduce to_line()'s field to ~1e-14 T. Field-only (no correctors), which
# is what we want here: the correctors are discrete kicks, not part of the
# smoothly-varying undulator field.
field_seq = SplineBorisSequence(
    fitter.df_fit_pars, multipole_order=MULTIPOLE_ORDER, steps_per_point=1,
    shift_x=SHIFT_X, shift_y=SHIFT_Y)

# field_seq.s_starts/s_ends keep the raw field-map's s convention (centered
# on 0, e.g. -1.1..+1.1 m) -- df_fit_pars is built directly from the fitted
# frames, which never get reset to a line-local origin. Twiss's s, in
# contrast, always starts at 0 at the line's entrance. to_line() sidesteps
# this (it only ever uses s_end - s_start, i.e. lengths, when chaining
# elements), but looking a Twiss s up against field_seq.s_starts directly
# needs this offset correction first.
s_starts = np.asarray(field_seq.s_starts)
s_offset = s_starts[0]
s_raw = tw_undulator.s + s_offset

# get_field() itself does a linear scan over all regions per call: look up
# the region index for every trajectory point up front (searchsorted over
# the sorted region boundaries) so each get_field() call below is O(1).
region_idx = np.clip(
    np.searchsorted(s_starts, s_raw, side='right') - 1,
    0, len(field_seq.elements) - 1)

bx_on_orbit = np.empty(len(tw_undulator.s))
by_on_orbit = np.empty(len(tw_undulator.s))
bs_on_orbit = np.empty(len(tw_undulator.s))
for i, i_reg in enumerate(region_idx):
    elem = field_seq.elements[i_reg]
    s_local = np.clip(s_raw[i] - s_starts[i_reg], 0.0, elem.length)
    bx_on_orbit[i], by_on_orbit[i], bs_on_orbit[i] = elem.get_field(
        tw_undulator.x[i], tw_undulator.y[i], s_local)

fig_field, (ax_bx, ax_by, ax_bs) = plt.subplots(
    3, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
ax_bx.plot(tw_undulator.s, bx_on_orbit)
ax_by.plot(tw_undulator.s, by_on_orbit)
ax_bs.plot(tw_undulator.s, bs_on_orbit)
ax_bx.set_ylabel(r'$B_x$ [T]')
ax_by.set_ylabel(r'$B_y$ [T]')
ax_bs.set_ylabel(r'$B_s$ [T]')
ax_bs.set_xlabel('s [m]')
ax_bx.set_title('Field on the SplineBoris undulator orbit (from Twiss x, y, s)')
for ax in (ax_bx, ax_by, ax_bs):
    ax.grid()
fig_field.savefig('splineboris_undulator_field_on_orbit.png', dpi=200,
                   bbox_inches='tight')

##################################################################
# Plot the integrated field components along s (Figure 3)        #
##################################################################

from scipy.integrate import cumulative_trapezoid

s_orbit = tw_undulator.s

int_bx = cumulative_trapezoid(bx_on_orbit, s_orbit, initial=0.0)
int_by = cumulative_trapezoid(by_on_orbit, s_orbit, initial=0.0)
int_bs = cumulative_trapezoid(bs_on_orbit, s_orbit, initial=0.0)

# "Average" transverse field strength, integrated along the orbit rather than
# divided by length -- B_perp = integral of sqrt(Bx^2 + By^2) ds.
b_perp_on_orbit = np.sqrt(bx_on_orbit**2 + by_on_orbit**2)
int_b_perp = cumulative_trapezoid(b_perp_on_orbit, s_orbit, initial=0.0)

int_bx_total = int_bx[-1]
int_by_total = int_by[-1]
int_bs_total = int_bs[-1]
int_b_perp_total = int_b_perp[-1]
bs_over_bperp = int_bs_total / int_b_perp_total

print(f"Integrated fields: int(Bx)ds = {int_bx_total:.6e} T*m, "
      f"int(By)ds = {int_by_total:.6e} T*m, int(Bs)ds = {int_bs_total:.6e} T*m, "
      f"B_perp = int(sqrt(Bx^2+By^2))ds = {int_b_perp_total:.6e} T*m, "
      f"int(Bs)ds / B_perp = {bs_over_bperp:.6e}")

fig_int, (ax_ibx, ax_iby, ax_ibs) = plt.subplots(
    3, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
ax_ibx.plot(s_orbit, int_bx)
ax_iby.plot(s_orbit, int_by)
ax_ibs.plot(s_orbit, int_bs)
ax_ibx.set_ylabel(r'$\int B_x\,ds$ [T$\cdot$m]')
ax_iby.set_ylabel(r'$\int B_y\,ds$ [T$\cdot$m]')
ax_ibs.set_ylabel(r'$\int B_s\,ds$ [T$\cdot$m]')
ax_ibs.set_xlabel('s [m]')
ax_ibx.set_title('Integrated field along the SplineBoris undulator orbit')
for ax in (ax_ibx, ax_iby, ax_ibs):
    ax.grid()

text_box_kwargs = dict(va='top', ha='left',
                        bbox=dict(boxstyle='round', fc='white', alpha=0.8))
ax_ibx.text(0.02, 0.95,
            f'$\\int B_x\\,ds$ = {int_bx_total:.4e} T$\\cdot$m',
            transform=ax_ibx.transAxes, **text_box_kwargs)
ax_iby.text(0.02, 0.95,
            f'$\\int B_y\\,ds$ = {int_by_total:.4e} T$\\cdot$m',
            transform=ax_iby.transAxes, **text_box_kwargs)
ax_ibs.text(0.02, 0.95,
            f'$\\int B_s\\,ds$ = {int_bs_total:.4e} T$\\cdot$m\n'
            f'$B_\\perp = \\int\\sqrt{{B_x^2+B_y^2}}\\,ds$ = {int_b_perp_total:.4e} T$\\cdot$m\n'
            f'$\\int B_s\\,ds\\,/\\,B_\\perp$ = {bs_over_bperp:.4e}',
            transform=ax_ibs.transAxes, **text_box_kwargs)

fig_int.savefig('splineboris_undulator_integrated_field.png', dpi=200,
                 bbox_inches='tight')

##################################################################
# Compare spin components along the undulator: SplineBoris vs    #
# Multipole (Figure 4, one panel per component, starting with    #
# vertical spin).                                                #
##################################################################

fig_spin, axes_spin = plt.subplots(
    3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)

for ax_spin, (spin_key, plane_label) in zip(axes_spin, [
    ('spin_y', 'vertical'),
    ('spin_x', 'horizontal'),
    ('spin_z', 'longitudinal'),
]):
    # spin_y sits near 1 (fully polarized vertical at entrance), so plot
    # 1 - spin_y instead -- avoids matplotlib's confusing "1e-6+9.9999e-1"
    # offset notation on an axis whose values barely move off 1.
    plot_1_minus = (spin_key == 'spin_y')

    # Diff computed on the raw (untransformed) spin component -- physically
    # meaningful regardless of the 1-minus display transform below.
    sb_raw = getattr(tw_undulator, spin_key)
    mult_raw = getattr(tw_undulator_mult, spin_key)
    spin_diff = mult_raw - sb_raw
    diff_final = spin_diff[-1]
    diff_max_abs = np.max(np.abs(spin_diff))

    sb_vals, mult_vals = sb_raw, mult_raw
    if plot_1_minus:
        sb_vals = 1 - sb_vals
        mult_vals = 1 - mult_vals

    ax_spin.plot(tw_undulator.s, sb_vals, label='SplineBoris')
    ax_spin.plot(tw_undulator_mult.s, mult_vals, '-.', label='Multipole')
    if plot_1_minus:
        ax_spin.set_ylabel(f'$1 - S_{{{spin_key[-1]}}}$')
    else:
        ax_spin.set_ylabel(f'${spin_key[0].upper()}_{{{spin_key[-1]}}}$')
    ax_spin.set_title(f'{plane_label.capitalize()} spin component')
    ax_spin.grid()
    ax_spin.legend(loc='upper left')

    diff_text = (
        f'$\\Delta S_{{{spin_key[-1]}}}$ = Multipole $-$ SplineBoris\n'
        f'  final = {diff_final:.3e}\n'
        f'  max $|\\Delta|$ = {diff_max_abs:.3e}'
    )
    ax_spin.text(0.98, 0.03, diff_text, transform=ax_spin.transAxes,
                 ha='right', va='bottom', fontsize=8, linespacing=1.4,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.85, edgecolor='0.7'))

axes_spin[-1].set_xlabel('s [m]')
fig_spin.suptitle('Spin components along the undulator: SplineBoris vs Multipole')
fig_spin.savefig('splineboris_undulator_spin.png', dpi=200, bbox_inches='tight')

plt.show()
