import sys
import xtrack as xt
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as sc_const
from xtrack._temp.splineboris.tube_fitter import TubeFitter


multipole_order = 3

E0 = 2.7e9

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)

# Load SLS MADX file
madx_file = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'sls.madx'

BASE_DIR = Path(__file__).resolve().parent

# Raw field map data (shared test_data) -- ~1.8 GB, so loading and fitting it
# is by far the slowest part of this script. Deferred to get_tube_fitter(),
# called lazily from compute_case() only for cases that are actually being
# (re)computed -- a `--replot` run where every case already has cached data
# in DATA_DIR never touches this file at all.
file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "simona_field_map.txt"

# Distance unit in meters (the dataset uses mm, so 1 mm = 0.001 m)
distance_unit = 0.001

n_frames = 4441

_tube_fitter = None


def get_tube_fitter():
    global _tube_fitter
    if _tube_fitter is None:
        print("[TubeFitter] loading raw field map and fitting "
              "(this is the slow part, and independent of undulator "
              "placement/model, so it only happens once per run)...")
        df_raw_data = pd.read_csv(
            file_path, sep="\t", header=None,
            names=["X", "Y", "Z", "Bx", "By", "Bs"],
            dtype=float,
        ).set_index(["X", "Y", "Z"])
        tf = TubeFitter(
            raw_data=df_raw_data,
            n_frames=n_frames,
            #residual_tol=1e-3,
            distance_unit=distance_unit,
            deg=multipole_order - 1,
            field_tol=1e-4,
            #tube_radius=0.0005,
        )
        tf.fit()
        _tube_fitter = tf
    return _tube_fitter

# for der in range(0, multipole_order):
#     get_tube_fitter().plot_fields(der=der)

# plt.show()

OUT_DIR = Path('/home/simonfan/cernbox/Pictures/SLS_Undulator_Studies')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cached per-case scan/formula results, so that re-running just to tweak a
# plot doesn't require redoing the fit/match/scan pipeline (which needs the
# ~1.8 GB field map above). Pass --replot on the command line to load from
# here instead of recomputing, for any case whose data is already cached.
DATA_DIR = Path(__file__).resolve().parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPLOT = '--replot' in sys.argv

# Three placement cases (only ars11_uind_0210_1, only ars11_uind_0610_1,
# both) x two undulator models (SB = SplineBoris, from to_line(); MK =
# Multipole Kick, from to_multipole_line()) -- compute_case()/plot_case()
# below build and analyse one such case end to end and save its 3 figures.
WIGGLER_CASES = [
    ('ars11_uind_0210_1', ['ars11_uind_0210_1']),
    ('ars11_uind_0610_1', ['ars11_uind_0610_1']),
    ('both', ['ars11_uind_0210_1', 'ars11_uind_0610_1']),
]
MODEL_LABELS = ('SB', 'MK')

# Digitized reference measurement (blue "Measurements" diamonds from a plot
# titled "X11MA, gap = 11.5 mm", pixel-extracted from test_data/sls/image004.png
# -- see examples/splineboris/claude_notes/ for the digitization method).
# NOTE: the correspondence between this "X11MA" beamline measurement and the
# ars11_uind_0210_1 / ars11_uind_0610_1 elements simulated below has NOT been
# confirmed -- overlaid here purely as a shape/order-of-magnitude reference,
# plotted directly on the same Delta Qx/Qy scale as the simulated curves.
MEASURED_TUNE_SHIFT_CSV = (
    Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls'
    / 'x11ma_gap11p5mm_tune_shift_digitized.csv'
)


def load_measured_tune_shift():
    if not MEASURED_TUNE_SHIFT_CSV.exists():
        return None
    return pd.read_csv(MEASURED_TUNE_SHIFT_CSV)


def compute_case(place_label, wiggler_places, model_label):
    """Build, match and scan one (placement, model) case, returning a dict
    of everything plot_case() needs -- nothing here touches matplotlib.
    """
    print("=" * 80)
    print(f"Case: place={place_label}  model={model_label}")
    case_label = f'{place_label} ({model_label})'
    print("=" * 80)

    tube_fitter = get_tube_fitter()

    # Load SLS MADX file
    env = xt.load(str(madx_file))
    line_sls = env.ring

    # Configure bend model
    line_sls.configure_bend_model(core='mat-kick-mat')

    # Set particle reference
    line_sls.particle_ref = p0.copy()

    # Build undulator -- SB uses TubeFitter.to_line() (one SplineBoris
    # element per polynomial piece, full spatial field integration); MK uses
    # TubeFitter.to_multipole_line() (one thick Multipole per region, a
    # coarser rigidity-normalized approximation). Either way it lives in its
    # own implicit Environment, so import it into its own `und_env` to add
    # correctors and match it (same build path as
    # 004b_undulators_in_sls_ring.py / 004a_build_undulator.py) before
    # placing it into the SLS ring.
    und_env = xt.Environment()
    und_env.particle_ref = p0.copy()

    if model_label == 'SB':
        undulator_line = tube_fitter.to_line(multipole_order=multipole_order)
    else:
        undulator_line = tube_fitter.to_multipole_line(
            multipole_order=multipole_order, p0c=E0, field_at='mean')
    undulator = und_env.import_line(undulator_line, line_name='undulator')

    l_wig = undulator.get_length()

    # Create env variables for corrector strengths (needed for matching)
    und_env['k0l_corr1'] = 0.
    und_env['k0l_corr2'] = 0.
    und_env['k0l_corr3'] = 0.
    und_env['k0l_corr4'] = 0.
    und_env['k0sl_corr1'] = 0.
    und_env['k0sl_corr2'] = 0.
    und_env['k0sl_corr3'] = 0.
    und_env['k0sl_corr4'] = 0.

    # Create corrector elements with expressions referencing env variables
    und_env.new('corr1', xt.Multipole, knl=['k0l_corr1'], ksl=['k0sl_corr1'])
    und_env.new('corr2', xt.Multipole, knl=['k0l_corr2'], ksl=['k0sl_corr2'])
    und_env.new('corr3', xt.Multipole, knl=['k0l_corr3'], ksl=['k0sl_corr3'])
    und_env.new('corr4', xt.Multipole, knl=['k0l_corr4'], ksl=['k0sl_corr4'])

    # Insert correctors at nearest element boundary (s_tol avoids slicing)
    undulator.insert([
        und_env.place('corr1', at=0.02),
        und_env.place('corr2', at=0.1),
        und_env.place('corr3', at=l_wig - 0.1),
        und_env.place('corr4', at=l_wig - 0.02),
    ], s_tol=5e-3)

    opt = undulator.match(
        solve=False,
        betx=0, bety=0,
        only_orbit=True,
        include_collective=True,
        vary=xt.VaryList(['k0l_corr1', 'k0sl_corr1',
                          'k0l_corr2', 'k0sl_corr2',
                          'k0l_corr3', 'k0sl_corr3',
                          'k0l_corr4', 'k0sl_corr4',
                          ], step=1e-6),
        targets=[
            xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.START),
            xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
            ],
    )
    opt.step(2)

    # Twiss with no undulator inserted -- reused both as the tune baseline
    # below and as the "no undulator" closed-orbit reference in the
    # comparison plot further down.
    tw_no_undulator = line_sls.twiss4d(include_collective=True)
    qx_0 = tw_no_undulator.qx
    qy_0 = tw_no_undulator.qy

    tt = line_sls.get_table()
    insertions = []
    for i, wig_place in enumerate(wiggler_places):
        # Fresh import per placement -- an element repeated at multiple
        # positions in the same line confuses downstream analyses (same
        # rationale as 004b_undulators_in_sls_ring.py's _build_and_twiss).
        print(f"Inserting undulator at {wig_place} ({tt['s', wig_place]})")
        import_name = f'undulator_{i}'
        env.import_line(undulator, line_name=import_name)
        insertions.append(
            env.place(env[import_name], anchor='start', at=tt['s', wig_place]))
    line_sls.insert(insertions)

    # Field-model element names, in both the ring copy and the standalone
    # `undulator` line -- name-based (not isinstance-based) because for the
    # MK model both the field elements and the correctors are xt.Multipole,
    # so isinstance alone can't tell them apart. to_line()/to_multipole_line()
    # name every region "tubefitter..." (import renaming only ever appends a
    # suffix), so this is robust to both models and to multiple placements.
    field_element_names = [
        nn for nn in line_sls.element_names if nn.startswith('tubefitter')
        ]
    undulator_field_element_names = [
        nn for nn in undulator.element_names if nn.startswith('tubefitter')
        ]

    # s ranges (start, end) of the active undulators, for marking their
    # location on the plots below. `tt` still reflects the pre-insertion
    # element table, so its s values are the undulator start positions.
    undulator_s_ranges = [
        (tt['s', wig_place], tt['s', wig_place] + l_wig)
        for wig_place in wiggler_places
        ]

    # On-axis Twiss (shift_x=0) -- needed below for the x-deflection-baseline
    # Twiss-interpolation check.
    for nn in field_element_names:
        line_sls[nn].shift_x = 0.
    tw_onaxis = line_sls.twiss4d(include_collective=True)

    deltaqx_list = []
    deltaqy_list = []
    orbit_scan_x = []
    orbit_scan_y = []
    orbit_scan_s = None
    betx_scan = []
    bety_scan = []
    betx2_scan = []
    bety1_scan = []

    n_tunes = 30

    hor_off_list = np.linspace(-0.5e-3, 0.5e-3, n_tunes)

    # Correctors were only matched once, at shift_x=0 -- as the offset is
    # scanned, the field (and hence the kick) seen by the field elements
    # changes, so those fixed corrector strengths stop restoring
    # x=px=y=py=0 at the undulator exit. Left uncorrected, that residual
    # orbit distortion leaks into the rest of the ring and contaminates the
    # tune shift with an orbit-distortion effect on top of the field
    # nonlinearity we're trying to isolate. Re-match the correctors (on the
    # standalone `undulator` line, same targets as the initial match above)
    # for every offset instead.
    corrector_vars = ['k0l_corr1', 'k0sl_corr1', 'k0l_corr2', 'k0sl_corr2',
                       'k0l_corr3', 'k0sl_corr3', 'k0l_corr4', 'k0sl_corr4']

    # Local x-deflection baseline: even at shift_x=0, the beam deflects
    # horizontally *inside* the undulator (correctors only force x=px=0 at
    # the undulator's own start/end, not throughout its length). Read via a
    # literal single-particle track through the (still unshifted) standalone
    # `undulator` line at element-by-element resolution, since the coarser
    # per-region Twiss orbit (tw_onaxis) may not resolve the same
    # intra-region curvature -- print both so the gap motivating this
    # addition is visible, not just assumed.
    # Folded into x_local in the deltaK1 loop below, as a first attempt at
    # explaining part of the theory-vs-simulation gap; may need revisiting.
    p_deflection = p0.copy()
    undulator.track(p_deflection, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_deflection = undulator.record_last_track
    x_deflection_baseline = float(np.mean(mon_deflection.x[0, :]))
    x_deflection_twiss_check = float(np.mean(np.interp(
        mon_deflection.s[0, :] + undulator_s_ranges[0][0],
        tw_onaxis.s, tw_onaxis.x)))
    print(f"[x-deflection baseline] literal single-particle track: "
          f"mean x = {x_deflection_baseline * 1e6:.3f} um "
          f"(Twiss-orbit-interpolated equivalent: "
          f"{x_deflection_twiss_check * 1e6:.3f} um)")

    for dx in hor_off_list:   # dx in meters
        # apply horizontal offset to all undulator slices (standalone line
        # and the copy inserted in the ring)
        for nn in undulator_field_element_names:
            undulator[nn].shift_x = dx
        for nn in field_element_names:
            line_sls[nn].shift_x = dx

        # re-correct the orbit for this offset
        opt_dx = undulator.match(
            solve=False,
            betx=0, bety=0,
            only_orbit=True,
            include_collective=True,
            vary=xt.VaryList(corrector_vars, step=1e-6),
            targets=[
                xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.START),
                xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
                ],
        )
        opt_dx.step(2)

        # push the re-matched corrector strengths to the copy inserted in
        # the ring
        for kk in corrector_vars:
            env[kk] = und_env[kk]

        # then compute tune for this offset
        tw = line_sls.twiss4d(include_collective=True)
        deltaqx_list.append(tw.qx - qx_0)
        deltaqy_list.append(tw.qy - qy_0)
        orbit_scan_x.append(tw.x)
        orbit_scan_y.append(tw.y)
        betx_scan.append(tw.betx)
        bety_scan.append(tw.bety)
        betx2_scan.append(tw.betx2)
        bety1_scan.append(tw.bety1)
        if orbit_scan_s is None:
            orbit_scan_s = tw.s

    # Independent analytic cross-check of the tune shift, via
    #   deltaQ = 1/(4 pi) * oint deltaK1(s) * beta(s) ds
    # (standard first-order perturbation theory), instead of reading it off
    # the full nonlinear Twiss above. deltaK1(s) has three sources within the
    # undulator: the fitted on-axis quadrupole itself (K1(s), ~0 here per
    # the fit -- see "Bnorm der=1" in the fit report above); normal-sextupole
    # feed-down, K2(s) * x(s), from evaluating the fitted sextupole (Bnorm
    # der=2) away from the magnet's own physical axis; and skew-sextupole
    # feed-down, -K2_skew(s) * y(s) (Bskew der=2 -- a vertical shift of a
    # skew sextupole also produces a normal-quadrupole component). The sign
    # on the skew term follows from the same complex-field convention xtrack's
    # own multipole kick uses internally
    # (track_magnet_kick.h::evaluate_field_from_strengths: By + iBx =
    # brho0/length * sum_n (knl[n]+i*ksl[n])/n! * (x+iy)^n): writing z=x+iy for the n=2 term,
    # Re(z^2)=x^2-y^2 and Im(z^2)=2xy, so d(By)/dx at fixed y works out to
    # K2*x - K2_skew*y. In this scan y is always ~0 (only shift_x is varied),
    # so this term is expected to evaluate to ~0 -- included for
    # completeness/correctness rather than because it moves the curve here.
    # to_multipole_line() gives K1(s), K2(s), K2_skew(s) directly per region
    # (thick Multipole knl[1]/knl[2]/ksl[2], built unshifted -- the canonical
    # field expansion about the magnet's own physical center); x(s)/y(s) are
    # read off the already-corrected orbit from the scan above, minus the
    # offset dx itself (shift_x moves the MAGNET, not the beam, so the field
    # is evaluated at the particle's position relative to the shifted
    # magnet), plus x_deflection_baseline (see above -- the local x-wiggle
    # present even at shift_x=0). This cross-check is built the same way
    # regardless of which model (SB/MK) is the primary undulator here -- for
    # the MK case it's effectively the same field model as the Twiss itself,
    # so the two are expected to agree closely.
    multipole_line = tube_fitter.to_multipole_line(
        multipole_order=multipole_order, p0c=E0, field_at='mean')
    mult_table = multipole_line.get_table()
    # to_multipole_line() also brackets each region with thin xt.MultipoleEdge
    # kicks (see tube_fitter.py) -- excluded here since K1(s)/K2(s)/K2_skew(s)
    # is a per-region (thick-body) quantity; the edge kicks are picked up by
    # the Twiss itself (via tw.qx/tw.qy or, in 007b, the phase advance) but
    # aren't part of this analytic deltaK1(s)*beta(s) integral.
    mult_names = [nn for nn in multipole_line.element_names
                  if not isinstance(multipole_line[nn], xt.MultipoleEdge)]
    mult_length = np.array([multipole_line[nn].length for nn in mult_names])
    mult_K1 = np.array([multipole_line[nn].knl[1] / multipole_line[nn].length
                         for nn in mult_names])
    mult_K2 = np.array([multipole_line[nn].knl[2] / multipole_line[nn].length
                         for nn in mult_names])
    mult_K2_skew = np.array([multipole_line[nn].ksl[2] / multipole_line[nn].length
                              for nn in mult_names])
    mult_s_mid = np.array([mult_table['s', nn] for nn in mult_names]) + mult_length / 2

    deltaqx_formula_list = []
    deltaqy_formula_list = []
    # Same integral, but using the actual perturbed beta (betx_scan/
    # bety_scan -- the Twiss beta at this same dx, with the undulator
    # already in the ring) instead of the unperturbed tw_no_undulator beta
    # -- isolates how much of the formula/Twiss gap comes from beta-beat
    # feedback (formula is 1st-order perturbation theory and normally uses
    # the unperturbed beta) rather than from delta K(s) itself.
    deltaqx_formula_pert_list = []
    deltaqy_formula_pert_list = []
    for dx, x_orbit, y_orbit, betx_pert, bety_pert in zip(
            hor_off_list, orbit_scan_x, orbit_scan_y, betx_scan, bety_scan):
        integral_x = 0.0
        integral_y = 0.0
        integral_x_pert = 0.0
        integral_y_pert = 0.0
        for s_start_und, _ in undulator_s_ranges:
            s_global = s_start_und + mult_s_mid
            x_local = (np.interp(s_global, orbit_scan_s, x_orbit) - dx
                       + x_deflection_baseline)
            # No "- dx"-like offset for y: shift_y is never varied in the
            # scan (only shift_x is), so the magnet's y-position stays at 0
            # and y_local is just the actual orbit position.
            y_local = np.interp(s_global, orbit_scan_s, y_orbit)
            deltaK1 = mult_K1 + mult_K2 * x_local - mult_K2_skew * y_local
            betx_at_s = np.interp(s_global, tw_no_undulator.s, tw_no_undulator.betx)
            bety_at_s = np.interp(s_global, tw_no_undulator.s, tw_no_undulator.bety)
            integral_x += np.sum(deltaK1 * betx_at_s * mult_length)
            integral_y += np.sum(deltaK1 * bety_at_s * mult_length)

            betx_pert_at_s = np.interp(s_global, orbit_scan_s, betx_pert)
            bety_pert_at_s = np.interp(s_global, orbit_scan_s, bety_pert)
            integral_x_pert += np.sum(deltaK1 * betx_pert_at_s * mult_length)
            integral_y_pert += np.sum(deltaK1 * bety_pert_at_s * mult_length)
        # Sign convention (deltaQx = +1/4pi * oint K1 betax ds, deltaQy =
        # -1/4pi * oint K1 betay ds) empirically verified: inserted a small
        # known-K1L thin Multipole into line_sls and compared the resulting
        # Twiss qx/qy shift against both sign choices -- this is the one
        # that matched.
        deltaqx_formula_list.append(integral_x / (4 * np.pi))
        deltaqy_formula_list.append(-integral_y / (4 * np.pi))
        deltaqx_formula_pert_list.append(integral_x_pert / (4 * np.pi))
        deltaqy_formula_pert_list.append(-integral_y_pert / (4 * np.pi))

    # tw_no_undulator carries a different element grid (no undulator
    # slices), so interpolate it onto the scan's (shared, undulator-
    # including) s grid to overlay it as a reference in the beta plots.
    betx_no_und_i = np.interp(orbit_scan_s, tw_no_undulator.s, tw_no_undulator.betx)
    bety_no_und_i = np.interp(orbit_scan_s, tw_no_undulator.s, tw_no_undulator.bety)
    betx2_no_und_i = np.interp(orbit_scan_s, tw_no_undulator.s, tw_no_undulator.betx2)
    bety1_no_und_i = np.interp(orbit_scan_s, tw_no_undulator.s, tw_no_undulator.bety1)

    return dict(
        case_label=case_label,
        hor_off_list=hor_off_list,
        deltaqx_list=np.array(deltaqx_list),
        deltaqy_list=np.array(deltaqy_list),
        orbit_scan_s=orbit_scan_s,
        betx_scan=np.array(betx_scan),
        bety_scan=np.array(bety_scan),
        betx2_scan=np.array(betx2_scan),
        bety1_scan=np.array(bety1_scan),
        betx_no_und_i=betx_no_und_i,
        bety_no_und_i=bety_no_und_i,
        betx2_no_und_i=betx2_no_und_i,
        bety1_no_und_i=bety1_no_und_i,
        deltaqx_formula_list=np.array(deltaqx_formula_list),
        deltaqy_formula_list=np.array(deltaqy_formula_list),
        deltaqx_formula_pert_list=np.array(deltaqx_formula_pert_list),
        deltaqy_formula_pert_list=np.array(deltaqy_formula_pert_list),
        undulator_s_ranges=np.array(undulator_s_ranges),
    )


def get_case_data(place_label, wiggler_places, model_label):
    data_path = DATA_DIR / f'{place_label}_{model_label}.npz'
    if REPLOT and data_path.exists():
        print(f"[replot] loading cached data from {data_path}")
        with np.load(data_path, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}
        data['case_label'] = str(data['case_label'])
        return data
    data = compute_case(place_label, wiggler_places, model_label)
    np.savez(data_path, **data)
    print(f"Saved case data to {data_path}")
    return data


def plot_case(data, place_label, model_label):
    """Build and save the 3 figures for one case, purely from the dict
    returned by compute_case()/get_case_data() -- no line building, matching
    or twissing happens here, so this is cheap and safe to re-run on its own
    (e.g. via --replot) to iterate on plot styling.
    """
    case_label = data['case_label']

    def mark_undulator_bounds(ax):
        for s_start, s_end in data['undulator_s_ranges']:
            ax.axvline(s_start, color='0.4', linestyle='--', linewidth=1)
            ax.axvline(s_end, color='0.4', linestyle='--', linewidth=1)

    hor_off_list = data['hor_off_list']
    deltaqx_list = data['deltaqx_list']
    deltaqy_list = data['deltaqy_list']
    orbit_scan_s = data['orbit_scan_s']
    betx_scan = data['betx_scan']
    bety_scan = data['bety_scan']
    betx2_scan = data['betx2_scan']
    bety1_scan = data['bety1_scan']
    betx_no_und_i = data['betx_no_und_i']
    bety_no_und_i = data['bety_no_und_i']
    betx2_no_und_i = data['betx2_no_und_i']
    bety1_no_und_i = data['bety1_no_und_i']
    deltaqx_formula_list = data['deltaqx_formula_list']
    deltaqy_formula_list = data['deltaqy_formula_list']
    deltaqx_formula_pert_list = data['deltaqx_formula_pert_list']
    deltaqy_formula_pert_list = data['deltaqy_formula_pert_list']

    # Colour scale for the offset-coloured beta-beat scan plots below (a
    # per-curve legend would be unreadable with n_tunes=30 curves).
    norm = plt.Normalize(vmin=hor_off_list.min(), vmax=hor_off_list.max())
    cmap = plt.cm.viridis

    # Beta beat at each offset of the tune scan, relative to the
    # no-undulator baseline and normalized by that baseline to show the
    # relative scale -- same colour-coded-by-offset layout as the orbit
    # scan above.
    fig_beta_diff, (ax_dbetx, ax_dbety) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for dx, betx, bety in zip(hor_off_list, betx_scan, bety_scan):
        color = cmap(norm(dx))
        ax_dbetx.plot(orbit_scan_s, (betx - betx_no_und_i) / betx_no_und_i, color=color)
        ax_dbety.plot(orbit_scan_s, (bety - bety_no_und_i) / bety_no_und_i, color=color)
    mark_undulator_bounds(ax_dbetx)
    mark_undulator_bounds(ax_dbety)
    ax_dbetx.set_ylabel(r'$\Delta\beta_x/\beta_{x,0}$')
    ax_dbetx.set_title('Relative beta beat across the tune scan (relative to no undulator)')
    ax_dbetx.grid(True, alpha=0.3)
    ax_dbety.set_xlabel('s [m]')
    ax_dbety.set_ylabel(r'$\Delta\beta_y/\beta_{y,0}$')
    ax_dbety.grid(True, alpha=0.3)

    sm_beta = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm_beta.set_array([])
    fig_beta_diff.colorbar(sm_beta, ax=[ax_dbetx, ax_dbety], label='Horizontal offset [m]')
    fig_beta_diff.suptitle(case_label)

    # Same colour-coded-by-offset layout, but for the coupled beta functions
    # betx2/bety1 (Edwards-Teng), which are zero without coupling and hence
    # a direct probe of the coupling introduced by the undulator. Normalized
    # by the *primary* beta (betx/bety, no undulator) rather than by
    # betx2/bety1 itself, since the latter is near zero along most of the
    # ring (only the ring's residual imperfection coupling) and would blow
    # up the ratio wherever it happens to dip towards zero.
    fig_beta_diff_coupled, (ax_dbetx2, ax_dbety1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for dx, betx2, bety1 in zip(hor_off_list, betx2_scan, bety1_scan):
        color = cmap(norm(dx))
        ax_dbetx2.plot(orbit_scan_s, (betx2 - betx2_no_und_i) / betx_no_und_i, color=color)
        ax_dbety1.plot(orbit_scan_s, (bety1 - bety1_no_und_i) / bety_no_und_i, color=color)
    mark_undulator_bounds(ax_dbetx2)
    mark_undulator_bounds(ax_dbety1)
    ax_dbetx2.set_ylabel(r'$\Delta\beta_{x2}/\beta_{x,0}$')
    ax_dbetx2.set_title('Coupled beta beat across the tune scan (relative to no undulator)')
    ax_dbetx2.grid(True, alpha=0.3)
    ax_dbety1.set_xlabel('s [m]')
    ax_dbety1.set_ylabel(r'$\Delta\beta_{y1}/\beta_{y,0}$')
    ax_dbety1.grid(True, alpha=0.3)

    sm_beta_coupled = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm_beta_coupled.set_array([])
    fig_beta_diff_coupled.colorbar(sm_beta_coupled, ax=[ax_dbetx2, ax_dbety1], label='Horizontal offset [m]')
    fig_beta_diff_coupled.suptitle(case_label)

    coef_qx = np.polyfit(hor_off_list, deltaqx_list, 2)
    coef_qy = np.polyfit(hor_off_list, deltaqy_list, 2)

    poly_qx = np.poly1d(coef_qx)
    poly_qy = np.poly1d(coef_qy)

    # coef_q*[0]/[1]/[2] are just the quadratic-fit Taylor coefficients of
    # DeltaQ(dx) itself -- (1/2)*d^2(DeltaQ)/dx^2, d(DeltaQ)/dx, DeltaQ(0) --
    # not yet deconvolved into magnetic field gradients.
    print(f"(1/2) d²(ΔQx)/dx² = {coef_qx[0]}")
    print(f"d(ΔQx)/dx         = {coef_qx[1]}")
    print(f"ΔQx(0)            = {coef_qx[2]}")
    print(f"(1/2) d²(ΔQy)/dx² = {coef_qy[0]}")
    print(f"d(ΔQy)/dx         = {coef_qy[1]}")
    print(f"ΔQy(0)            = {coef_qy[2]}")

    coef_qx_formula = np.polyfit(hor_off_list, deltaqx_formula_list, 2)
    coef_qy_formula = np.polyfit(hor_off_list, deltaqy_formula_list, 2)
    print(f"[formula] (1/2) d²(ΔQx)/dx² = {coef_qx_formula[0]}")
    print(f"[formula] d(ΔQx)/dx         = {coef_qx_formula[1]}")
    print(f"[formula] ΔQx(0)            = {coef_qx_formula[2]}")
    print(f"[formula] (1/2) d²(ΔQy)/dx² = {coef_qy_formula[0]}")
    print(f"[formula] d(ΔQy)/dx         = {coef_qy_formula[1]}")
    print(f"[formula] ΔQy(0)            = {coef_qy_formula[2]}")

    coef_qx_formula_pert = np.polyfit(hor_off_list, deltaqx_formula_pert_list, 2)
    coef_qy_formula_pert = np.polyfit(hor_off_list, deltaqy_formula_pert_list, 2)
    print(f"[formula, perturbed beta] (1/2) d²(ΔQx)/dx² = {coef_qx_formula_pert[0]}")
    print(f"[formula, perturbed beta] d(ΔQx)/dx         = {coef_qx_formula_pert[1]}")
    print(f"[formula, perturbed beta] ΔQx(0)            = {coef_qx_formula_pert[2]}")
    print(f"[formula, perturbed beta] (1/2) d²(ΔQy)/dx² = {coef_qy_formula_pert[0]}")
    print(f"[formula, perturbed beta] d(ΔQy)/dx         = {coef_qy_formula_pert[1]}")
    print(f"[formula, perturbed beta] ΔQy(0)            = {coef_qy_formula_pert[2]}")

    text_box_kwargs = dict(va='top', ha='left', fontsize=8, linespacing=1.4,
                            bbox=dict(boxstyle='round', fc='white', alpha=0.85, edgecolor='0.7'))

    fig_tune_shift, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(hor_off_list, deltaqx_list, marker='o', color='tab:blue', label='Twiss')
    ax1.plot(hor_off_list, poly_qx(hor_off_list), linestyle='--', color='k', label='Quadratic fit')
    ax1.plot(hor_off_list, deltaqx_formula_list, marker='^', linestyle='none',
              color='tab:green', label=r'$\frac{1}{4\pi}\oint\delta K_1\,\beta_{x,0}\,ds$')
    ax1.plot(hor_off_list, deltaqx_formula_pert_list, marker='v', linestyle='none',
              color='tab:red', label=r'$\frac{1}{4\pi}\oint\delta K_1\,\beta_x\,ds$')
    ax1.set_ylabel('Delta Qx')
    ax1.set_title('Tune shift vs undulator horizontal offset')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.text(0.02, 0.95,
              f'$\\frac{{1}}{{2}}\\frac{{d^2\\Delta Q_x}}{{dx^2}}$ = {coef_qx[0]:.4e}\n'
              f'$\\frac{{d\\Delta Q_x}}{{dx}}$ = {coef_qx[1]:.4e}\n'
              f'$\\Delta Q_x(0)$ = {coef_qx[2]:.4e}',
              transform=ax1.transAxes, **text_box_kwargs)

    ax2.plot(hor_off_list, deltaqy_list, marker='s', color='tab:orange', label='Twiss')
    ax2.plot(hor_off_list, poly_qy(hor_off_list), linestyle='--', color='k', label='Quadratic fit')
    ax2.plot(hor_off_list, deltaqy_formula_list, marker='^', linestyle='none',
              color='tab:green', label=r'$\frac{1}{4\pi}\oint(-\delta K_1)\beta_{y,0}\,ds$')
    ax2.plot(hor_off_list, deltaqy_formula_pert_list, marker='v', linestyle='none',
              color='tab:red', label=r'$\frac{1}{4\pi}\oint(-\delta K_1)\beta_y\,ds$')
    ax2.set_xlabel('Horizontal offset [m]')
    ax2.set_ylabel('Delta Qy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.text(0.02, 0.95,
              f'$\\frac{{1}}{{2}}\\frac{{d^2\\Delta Q_y}}{{dx^2}}$ = {coef_qy[0]:.4e}\n'
              f'$\\frac{{d\\Delta Q_y}}{{dx}}$ = {coef_qy[1]:.4e}\n'
              f'$\\Delta Q_y(0)$ = {coef_qy[2]:.4e}',
              transform=ax2.transAxes, **text_box_kwargs)

    measured = load_measured_tune_shift()
    if measured is not None:
        ax1.plot(measured['bump_amplitude_m'], measured['dtune_x_measured'],
                  marker='d', linestyle='none', markersize=4,
                  color='tab:purple', label='X11MA, gap=11.5mm (measured)')
        ax1.legend()

        ax2.plot(measured['bump_amplitude_m'], measured['dtune_y_measured'],
                  marker='d', linestyle='none', markersize=4,
                  color='tab:purple', label='X11MA, gap=11.5mm (measured)')
        ax2.legend()

    fig_tune_shift.suptitle(case_label)
    fig_tune_shift.tight_layout()

    # Save the 3 figures for this case, named
    # "<place_label>_<model_label>_<what the figure shows>.pdf".
    figures = [
        (fig_beta_diff, 'beta_beat'),
        (fig_beta_diff_coupled, 'beta_beat_coupled'),
        (fig_tune_shift, 'tune_shift'),
        ]
    for fig, suffix in figures:
        out_path = OUT_DIR / f'{place_label}_{model_label}_{suffix}.pdf'
        fig.savefig(out_path)
        print(f"Saved {out_path}")


for place_label, wiggler_places in WIGGLER_CASES:
    for model_label in MODEL_LABELS:
        case_data = get_case_data(place_label, wiggler_places, model_label)
        plot_case(case_data, place_label, model_label)

# All 3*2*3 = 18 figures across every case (3 figures x 2 models x 3
# placements) are kept open (not closed inside plot_case()) so they can all
# be reviewed interactively here, in addition to having been saved as PDFs
# above.
plt.show()
