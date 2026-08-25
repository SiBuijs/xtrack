import sys
import xtrack as xt
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as sc_const
from xtrack._temp.splineboris.tube_fitter import TubeFitter
from xtrack._temp.splineboris.splineboris_sequence import SplineBorisSequence


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
# below build and analyse one such case end to end and save its 8 figures.
WIGGLER_CASES = [
    ('ars11_uind_0210_1', ['ars11_uind_0210_1']),
    ('ars11_uind_0610_1', ['ars11_uind_0610_1']),
    ('both', ['ars11_uind_0210_1', 'ars11_uind_0610_1']),
]
MODEL_LABELS = ('SB', 'MK')


def _multipole_field_bx_by(knl, ksl, length, brho0, x, y):
    """(Bx, By) [T] at (x, y) from a thick Multipole's knl/ksl, matching the
    field convention xtrack's own kick uses internally
    (track_magnet_kick.h::evaluate_field_from_strengths /
    kick_simple_single_coordinates): knl[n]/ksl[n] are length-integrated
    (i.e. K_n*length, J_n*length), so
    By + i*Bx = brho0/length * sum_n (knl[n]+i*ksl[n])/n! * (x+iy)**n.
    """
    z = complex(x, y)
    s = 0j
    zpow = 1 + 0j
    fact = 1.0
    for n in range(len(knl)):
        if n > 0:
            fact *= n
            zpow *= z
        s += (knl[n] + 1j * ksl[n]) / fact * zpow
    field = s * brho0 / length
    return field.imag, field.real


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

    brho0 = E0 / (sc_const.c * 1.0)

    # Closed orbit comparison: no undulator vs. on-axis (shift_x=0) vs.
    # off-axis (shift_x=0.5 mm) undulator.
    for nn in field_element_names:
        line_sls[nn].shift_x = 0.
    tw_onaxis = line_sls.twiss4d(include_collective=True)

    for nn in field_element_names:
        line_sls[nn].shift_x = 0.5e-3
    tw_offaxis = line_sls.twiss4d(include_collective=True)

    # Field actually seen along the closed-orbit trajectory (tw_onaxis.x/y/s,
    # tw_offaxis.x/y/s -- the Twiss closed orbit is sufficient here, no need
    # to track a probe particle separately), evaluated with whichever model
    # is primary for this case: SB reads the real fitted field via
    # SplineBoris.get_field() (Bx, By, Bs all included); MK has no field
    # evaluator (it's a thick Multipole, no spatial field structure within
    # the element), so its field is reconstructed from its own knl/ksl and
    # the reference rigidity via _multipole_field_bx_by() -- exactly the
    # field xtrack's own multipole kick is implicitly using (Bs is not
    # defined for MK, no solenoid equivalent -- see
    # to_multipole_line()).
    field_tt = line_sls.get_table()
    n_sub = 6  # field-sample points per field element (for smoother curves)

    def _sample_field(traj_s, traj_x, traj_y, dx):
        s_out, bx_out, by_out, bs_out = [], [], [], []
        for nn in field_element_names:
            s0 = field_tt['s', nn]
            length = line_sls[nn].length
            for s_local in np.linspace(0.0, length, n_sub, endpoint=False):
                s_glob = s0 + s_local
                x_local = np.interp(s_glob, traj_s, traj_x) - dx
                y_local = np.interp(s_glob, traj_s, traj_y)
                if model_label == 'SB':
                    bx, by, bs = line_sls[nn].get_field(x_local, y_local, s_local)
                else:
                    bx, by = _multipole_field_bx_by(
                        line_sls[nn].knl, line_sls[nn].ksl, length, brho0,
                        x_local, y_local)
                    bs = 0.0
                s_out.append(s_glob)
                bx_out.append(bx)
                by_out.append(by)
                bs_out.append(bs)
        return (np.array(s_out), np.array(bx_out), np.array(by_out),
                np.array(bs_out))

    field_s_on, field_bx_on, field_by_on, field_bs_on = _sample_field(
        tw_onaxis.s, tw_onaxis.x, tw_onaxis.y, 0.0)
    field_s_off, field_bx_off, field_by_off, field_bs_off = _sample_field(
        tw_offaxis.s, tw_offaxis.x, tw_offaxis.y, 0.5e-3)

    deltaqx_list = []
    deltaqy_list = []
    orbit_scan_x = []
    orbit_scan_y = []
    orbit_scan_s = None
    betx_scan = []
    bety_scan = []
    c_minus_scan = []

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
    # per-region Twiss orbit (tw_onaxis, used above for the field-sampling
    # comparison) may not resolve the same intra-region curvature -- print
    # both so the gap motivating this addition is visible, not just assumed.
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
        c_minus_scan.append(tw.c_minus)
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
    # on the skew term follows from the same complex-field convention used
    # in _multipole_field_bx_by() above (By + iBx = brho0/length * sum_n
    # (knl[n]+i*ksl[n])/n! * (x+iy)^n): writing z=x+iy for the n=2 term,
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
    mult_length = np.array([multipole_line[nn].length
                             for nn in multipole_line.element_names])
    mult_K1 = np.array([multipole_line[nn].knl[1] / multipole_line[nn].length
                         for nn in multipole_line.element_names])
    mult_K2 = np.array([multipole_line[nn].knl[2] / multipole_line[nn].length
                         for nn in multipole_line.element_names])
    mult_K2_skew = np.array([multipole_line[nn].ksl[2] / multipole_line[nn].length
                              for nn in multipole_line.element_names])
    mult_s_mid = mult_table['s'][:-1] + mult_length / 2

    # Longitudinal-field (solenoid) focusing cross-check: unlike a normal
    # quadrupole, a solenoid focuses BOTH planes, with an effective
    # K_sol(s) = (Bs(s) / (2*Brho0))^2. On-axis Bs is ~0 here (fit flagged
    # it "to_fit=False"), but Bs is NOT zero off-axis: Maxwell's equations
    # tie the z-varying transverse (quadrupole/sextupole) field to a
    # longitudinal component that grows away from the physical axis even
    # when the on-axis Bs is exactly zero (confirmed empirically: this
    # fit's Bz(x=0.5mm) ~ 0.012 T vs Bz(x=0) = 0 exactly). So Bs must be
    # read at the actual local orbit position x_local(s) -- same x_local as
    # the K1/K2 feed-down term below -- not on-axis, and hence recomputed
    # per dx rather than once up front. Read via the SplineBoris field
    # evaluator, get_field(x, y, s_local); a dedicated, always-SplineBoris-
    # based `bs_line` is built fresh for this regardless of which model
    # (SB/MK) is primary here, since to_multipole_line() drops Bs entirely
    # (no Multipole equivalent) and the MK `undulator` therefore has no Bs
    # to read. Same region grid as multipole_line (both come from the same
    # df_fit_pars/frames), so element i in each lines up with region i in
    # the other.
    bs_line = tube_fitter.to_line(multipole_order=multipole_order)
    assert len(bs_line.element_names) == len(multipole_line.element_names)
    bs_line_names = bs_line.element_names
    bs_s_local = mult_length / 2

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
    # Quadrupole-effect-only variant (deltaK1 term -- intrinsic K1 plus
    # sextupole feed-down -- without the solenoid/Bs focusing term), at the
    # unperturbed beta0, to isolate how much of the beta0 formula curve above
    # comes from the longitudinal field component.
    deltaqx_formula_quadonly_list = []
    deltaqy_formula_quadonly_list = []
    for dx, x_orbit, y_orbit, betx_pert, bety_pert in zip(
            hor_off_list, orbit_scan_x, orbit_scan_y, betx_scan, bety_scan):
        integral_x = 0.0
        integral_y = 0.0
        integral_x_pert = 0.0
        integral_y_pert = 0.0
        integral_x_sol = 0.0
        integral_y_sol = 0.0
        integral_x_sol_pert = 0.0
        integral_y_sol_pert = 0.0
        for s_start_und, _ in undulator_s_ranges:
            s_global = s_start_und + mult_s_mid
            x_local = (np.interp(s_global, orbit_scan_s, x_orbit) - dx
                       + x_deflection_baseline)
            # No "- dx"-like offset for y: shift_y is never varied in the
            # scan (only shift_x is), so the magnet's y-position stays at 0
            # and y_local is just the actual orbit position.
            y_local = np.interp(s_global, orbit_scan_s, y_orbit)
            deltaK1 = mult_K1 + mult_K2 * x_local - mult_K2_skew * y_local
            mult_Bs_local = np.array([
                bs_line[nn].get_field(x_val, y_val, s_local)[2]
                for nn, s_local, x_val, y_val in zip(
                    bs_line_names, bs_s_local, x_local, y_local)
                ])
            mult_K_sol = (mult_Bs_local / (2 * brho0)) ** 2
            betx_at_s = np.interp(s_global, tw_no_undulator.s, tw_no_undulator.betx)
            bety_at_s = np.interp(s_global, tw_no_undulator.s, tw_no_undulator.bety)
            integral_x += np.sum(deltaK1 * betx_at_s * mult_length)
            integral_y += np.sum(deltaK1 * bety_at_s * mult_length)
            integral_x_sol += np.sum(mult_K_sol * betx_at_s * mult_length)
            integral_y_sol += np.sum(mult_K_sol * bety_at_s * mult_length)

            betx_pert_at_s = np.interp(s_global, orbit_scan_s, betx_pert)
            bety_pert_at_s = np.interp(s_global, orbit_scan_s, bety_pert)
            integral_x_pert += np.sum(deltaK1 * betx_pert_at_s * mult_length)
            integral_y_pert += np.sum(deltaK1 * bety_pert_at_s * mult_length)
            integral_x_sol_pert += np.sum(mult_K_sol * betx_pert_at_s * mult_length)
            integral_y_sol_pert += np.sum(mult_K_sol * bety_pert_at_s * mult_length)
        # Sign convention (deltaQx = +1/4pi * oint K1 betax ds, deltaQy =
        # -1/4pi * oint K1 betay ds) empirically verified: inserted a small
        # known-K1L thin Multipole into line_sls and compared the resulting
        # Twiss qx/qy shift against both sign choices -- this is the one
        # that matched. The solenoid term is added with the SAME sign in
        # both planes (it focuses x AND y), unlike the quadrupole-feed-down
        # term above -- hence "+integral_y_sol" despite the "-integral_y".
        deltaqx_formula_list.append((integral_x + integral_x_sol) / (4 * np.pi))
        deltaqy_formula_list.append((-integral_y + integral_y_sol) / (4 * np.pi))
        deltaqx_formula_pert_list.append((integral_x_pert + integral_x_sol_pert) / (4 * np.pi))
        deltaqy_formula_pert_list.append((-integral_y_pert + integral_y_sol_pert) / (4 * np.pi))
        deltaqx_formula_quadonly_list.append(integral_x / (4 * np.pi))
        deltaqy_formula_quadonly_list.append(-integral_y / (4 * np.pi))

    # Closest-tune-approach (betatron coupling) correction of the analytic
    # formula, using |C-| read directly off the coupled Twiss (tw.c_minus,
    # computed automatically by twiss4d whenever mux/muy are available -- no
    # extra tracking needed). The formula above gives *uncoupled* tunes; the
    # actual observable eigentunes are pulled apart from their uncoupled
    # values by the coupling resonance,
    #   Qx_obs = Qbar + sign(Delta)/2 * sqrt(Delta^2 + |C-|^2)
    #   Qy_obs = Qbar - sign(Delta)/2 * sqrt(Delta^2 + |C-|^2)
    # with Qbar=(Qx+Qy)/2, Delta=Qx-Qy. This has to be applied to the
    # *absolute* tunes and then differenced -- not added directly to
    # deltaQ -- and the reference (tw_no_undulator) needs the same
    # treatment with its own c_minus before subtracting, since it is not in
    # general perfectly decoupled either.
    def _coupling_corrected_tunes(qx, qy, c_minus):
        qbar = 0.5 * (qx + qy)
        delta = qx - qy
        sign = np.where(delta == 0, 1.0, np.sign(delta))
        sqrt_term = np.sqrt(delta ** 2 + c_minus ** 2)
        return qbar + 0.5 * sign * sqrt_term, qbar - 0.5 * sign * sqrt_term

    c_minus_scan_arr = np.array(c_minus_scan)
    c_minus_0 = tw_no_undulator.c_minus

    qx_obs_ref, qy_obs_ref = _coupling_corrected_tunes(qx_0, qy_0, c_minus_0)

    qx_obs_beta0, qy_obs_beta0 = _coupling_corrected_tunes(
        qx_0 + np.array(deltaqx_formula_list),
        qy_0 + np.array(deltaqy_formula_list),
        c_minus_scan_arr)
    qx_obs_pert, qy_obs_pert = _coupling_corrected_tunes(
        qx_0 + np.array(deltaqx_formula_pert_list),
        qy_0 + np.array(deltaqy_formula_pert_list),
        c_minus_scan_arr)
    qx_obs_quadonly, qy_obs_quadonly = _coupling_corrected_tunes(
        qx_0 + np.array(deltaqx_formula_quadonly_list),
        qy_0 + np.array(deltaqy_formula_quadonly_list),
        c_minus_scan_arr)

    deltaqx_formula_corr_list = qx_obs_beta0 - qx_obs_ref
    deltaqy_formula_corr_list = qy_obs_beta0 - qy_obs_ref
    deltaqx_formula_pert_corr_list = qx_obs_pert - qx_obs_ref
    deltaqy_formula_pert_corr_list = qy_obs_pert - qy_obs_ref
    deltaqx_formula_quadonly_corr_list = qx_obs_quadonly - qx_obs_ref
    deltaqy_formula_quadonly_corr_list = qy_obs_quadonly - qy_obs_ref

    deltaqx_resid_corr_beta0 = np.array(deltaqx_list) - deltaqx_formula_corr_list
    deltaqy_resid_corr_beta0 = np.array(deltaqy_list) - deltaqy_formula_corr_list
    deltaqx_resid_corr_pert = np.array(deltaqx_list) - deltaqx_formula_pert_corr_list
    deltaqy_resid_corr_pert = np.array(deltaqy_list) - deltaqy_formula_pert_corr_list
    deltaqx_resid_corr_quadonly = np.array(deltaqx_list) - deltaqx_formula_quadonly_corr_list
    deltaqy_resid_corr_quadonly = np.array(deltaqy_list) - deltaqy_formula_quadonly_corr_list

    # Report the arithmetic at the largest-offset scan point, as requested:
    # c_minus, Delta (uncoupled formula tunes), and the residual before/after
    # the coupling correction.
    i_report = -1
    delta_report = ((qx_0 + deltaqx_formula_list[i_report])
                     - (qy_0 + deltaqy_formula_list[i_report]))
    print(f"[coupling check] dx = {hor_off_list[i_report] * 1e3:.4f} mm")
    print(f"[coupling check] c_minus (Twiss)            = {c_minus_scan_arr[i_report]:.6e}")
    print(f"[coupling check] Delta = Qx-Qy (uncoupled)  = {delta_report:.6e}")
    print(f"[coupling check] deltaQx residual, uncorrected (beta0) = "
          f"{deltaqx_list[i_report] - deltaqx_formula_list[i_report]:.6e}")
    print(f"[coupling check] deltaQx residual, corrected   (beta0) = "
          f"{deltaqx_resid_corr_beta0[i_report]:.6e}")
    print(f"[coupling check] deltaQy residual, uncorrected (beta0) = "
          f"{deltaqy_list[i_report] - deltaqy_formula_list[i_report]:.6e}")
    print(f"[coupling check] deltaQy residual, corrected   (beta0) = "
          f"{deltaqy_resid_corr_beta0[i_report]:.6e}")

    # tw_no_undulator carries a different element grid (no undulator
    # slices), so interpolate it onto the scan's (shared, undulator-
    # including) s grid to overlay it as a reference in the beta plots.
    betx_no_und_i = np.interp(orbit_scan_s, tw_no_undulator.s, tw_no_undulator.betx)
    bety_no_und_i = np.interp(orbit_scan_s, tw_no_undulator.s, tw_no_undulator.bety)

    return dict(
        case_label=case_label,
        hor_off_list=hor_off_list,
        deltaqx_list=np.array(deltaqx_list),
        deltaqy_list=np.array(deltaqy_list),
        orbit_scan_s=orbit_scan_s,
        orbit_scan_x=np.array(orbit_scan_x),
        orbit_scan_y=np.array(orbit_scan_y),
        betx_scan=np.array(betx_scan),
        bety_scan=np.array(bety_scan),
        betx_no_und_i=betx_no_und_i,
        bety_no_und_i=bety_no_und_i,
        deltaqx_formula_list=np.array(deltaqx_formula_list),
        deltaqy_formula_list=np.array(deltaqy_formula_list),
        deltaqx_formula_pert_list=np.array(deltaqx_formula_pert_list),
        deltaqy_formula_pert_list=np.array(deltaqy_formula_pert_list),
        deltaqx_formula_quadonly_list=np.array(deltaqx_formula_quadonly_list),
        deltaqy_formula_quadonly_list=np.array(deltaqy_formula_quadonly_list),
        deltaqx_resid_corr_beta0=deltaqx_resid_corr_beta0,
        deltaqy_resid_corr_beta0=deltaqy_resid_corr_beta0,
        deltaqx_resid_corr_pert=deltaqx_resid_corr_pert,
        deltaqy_resid_corr_pert=deltaqy_resid_corr_pert,
        deltaqx_resid_corr_quadonly=deltaqx_resid_corr_quadonly,
        deltaqy_resid_corr_quadonly=deltaqy_resid_corr_quadonly,
        undulator_s_ranges=np.array(undulator_s_ranges),
        tw_no_undulator_s=tw_no_undulator.s,
        tw_no_undulator_x=tw_no_undulator.x,
        tw_no_undulator_y=tw_no_undulator.y,
        tw_onaxis_s=tw_onaxis.s,
        tw_onaxis_x=tw_onaxis.x,
        tw_onaxis_y=tw_onaxis.y,
        tw_offaxis_s=tw_offaxis.s,
        tw_offaxis_x=tw_offaxis.x,
        tw_offaxis_y=tw_offaxis.y,
        field_s_on=field_s_on, field_bx_on=field_bx_on,
        field_by_on=field_by_on, field_bs_on=field_bs_on,
        field_s_off=field_s_off, field_bx_off=field_bx_off,
        field_by_off=field_by_off, field_bs_off=field_bs_off,
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
    """Build and save the 8 figures for one case, purely from the dict
    returned by compute_case()/get_case_data() -- no line building, matching
    or twissing happens here, so this is cheap and safe to re-run on its own
    (e.g. via --replot) to iterate on plot styling.
    """
    case_label = data['case_label']

    def mark_undulator_bounds(ax):
        for s_start, s_end in data['undulator_s_ranges']:
            ax.axvline(s_start, color='0.4', linestyle='--', linewidth=1)
            ax.axvline(s_end, color='0.4', linestyle='--', linewidth=1)

    n_field_rows = 3 if model_label == 'SB' else 2
    fig_field_traj, field_axes = plt.subplots(
        n_field_rows, 1, figsize=(10, 3.2 * n_field_rows), sharex=True)
    ax_bx, ax_by = field_axes[0], field_axes[1]
    ax_bx.plot(data['field_s_on'], data['field_bx_on'],
               color='tab:blue', label='On-axis')
    ax_bx.plot(data['field_s_off'], data['field_bx_off'],
               color='tab:orange', label='Off-axis (shift_x=0.5 mm)')
    mark_undulator_bounds(ax_bx)
    ax_bx.set_ylabel(r'$B_x$ [T]')
    ax_bx.set_title(f'Field along the tracked trajectory ({model_label} model)')
    ax_bx.grid(True, alpha=0.3)
    ax_bx.legend()

    ax_by.plot(data['field_s_on'], data['field_by_on'],
               color='tab:blue', label='On-axis')
    ax_by.plot(data['field_s_off'], data['field_by_off'],
               color='tab:orange', label='Off-axis (shift_x=0.5 mm)')
    mark_undulator_bounds(ax_by)
    ax_by.set_ylabel(r'$B_y$ [T]')
    ax_by.grid(True, alpha=0.3)
    ax_by.legend()

    if model_label == 'SB':
        ax_bs = field_axes[2]
        ax_bs.plot(data['field_s_on'], data['field_bs_on'],
                   color='tab:blue', label='On-axis')
        ax_bs.plot(data['field_s_off'], data['field_bs_off'],
                   color='tab:orange', label='Off-axis (shift_x=0.5 mm)')
        mark_undulator_bounds(ax_bs)
        ax_bs.set_ylabel(r'$B_s$ [T]')
        ax_bs.grid(True, alpha=0.3)
        ax_bs.legend()

    field_axes[-1].set_xlabel('s [m]')
    fig_field_traj.suptitle(case_label)
    fig_field_traj.tight_layout()

    fig_orbit, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for s_arr, x_arr, y_arr, label in [
            (data['tw_no_undulator_s'], data['tw_no_undulator_x'], data['tw_no_undulator_y'], 'No undulator'),
            (data['tw_onaxis_s'], data['tw_onaxis_x'], data['tw_onaxis_y'], 'On-axis undulator'),
            (data['tw_offaxis_s'], data['tw_offaxis_x'], data['tw_offaxis_y'], 'Off-axis undulator (shift_x=0.5 mm)')]:
        ax_x.plot(s_arr, x_arr, label=label)
        ax_y.plot(s_arr, y_arr, label=label)
    mark_undulator_bounds(ax_x)
    mark_undulator_bounds(ax_y)
    ax_x.set_ylabel('x [m]')
    ax_x.set_title('Horizontal closed orbit around the SLS ring')
    ax_x.grid(True, alpha=0.3)
    ax_x.legend()
    ax_y.set_xlabel('s [m]')
    ax_y.set_ylabel('y [m]')
    ax_y.set_title('Vertical closed orbit around the SLS ring')
    ax_y.grid(True, alpha=0.3)
    ax_y.legend()
    fig_orbit.suptitle(case_label)
    fig_orbit.tight_layout()

    hor_off_list = data['hor_off_list']
    deltaqx_list = data['deltaqx_list']
    deltaqy_list = data['deltaqy_list']
    orbit_scan_s = data['orbit_scan_s']
    orbit_scan_x = data['orbit_scan_x']
    orbit_scan_y = data['orbit_scan_y']
    betx_scan = data['betx_scan']
    bety_scan = data['bety_scan']
    betx_no_und_i = data['betx_no_und_i']
    bety_no_und_i = data['bety_no_und_i']
    deltaqx_formula_list = data['deltaqx_formula_list']
    deltaqy_formula_list = data['deltaqy_formula_list']
    deltaqx_formula_pert_list = data['deltaqx_formula_pert_list']
    deltaqy_formula_pert_list = data['deltaqy_formula_pert_list']
    deltaqx_formula_quadonly_list = data['deltaqx_formula_quadonly_list']
    deltaqy_formula_quadonly_list = data['deltaqy_formula_quadonly_list']

    # Closed orbit at each offset of the tune scan above -- same panel
    # layout as the no-undulator/on-axis/off-axis comparison plot, but
    # colored by offset (a per-curve legend would be unreadable with
    # n_tunes=30 curves).
    norm = plt.Normalize(vmin=hor_off_list.min(), vmax=hor_off_list.max())
    cmap = plt.cm.viridis

    fig_orbit_scan, (ax_x_scan, ax_y_scan) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for dx, x_orbit, y_orbit in zip(hor_off_list, orbit_scan_x, orbit_scan_y):
        color = cmap(norm(dx))
        ax_x_scan.plot(orbit_scan_s, x_orbit, color=color)
        ax_y_scan.plot(orbit_scan_s, y_orbit, color=color)
    mark_undulator_bounds(ax_x_scan)
    mark_undulator_bounds(ax_y_scan)
    ax_x_scan.set_ylabel('x [m]')
    ax_x_scan.set_title('Horizontal closed orbit across the tune scan')
    ax_x_scan.grid(True, alpha=0.3)
    ax_y_scan.set_xlabel('s [m]')
    ax_y_scan.set_ylabel('y [m]')
    ax_y_scan.set_title('Vertical closed orbit across the tune scan')
    ax_y_scan.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig_orbit_scan.colorbar(sm, ax=[ax_x_scan, ax_y_scan], label='Horizontal offset [m]')
    fig_orbit_scan.suptitle(case_label)

    # Beta functions at each offset of the tune scan -- same colour-coded-
    # by-offset layout as the orbit scan above.
    fig_beta_scan, (ax_betx_scan, ax_bety_scan) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for dx, betx, bety in zip(hor_off_list, betx_scan, bety_scan):
        color = cmap(norm(dx))
        ax_betx_scan.plot(orbit_scan_s, betx, color=color)
        ax_bety_scan.plot(orbit_scan_s, bety, color=color)
    ax_betx_scan.plot(orbit_scan_s, betx_no_und_i, color='k', linestyle='--',
                       linewidth=1, label='No undulator')
    ax_bety_scan.plot(orbit_scan_s, bety_no_und_i, color='k', linestyle='--',
                       linewidth=1, label='No undulator')
    mark_undulator_bounds(ax_betx_scan)
    mark_undulator_bounds(ax_bety_scan)
    ax_betx_scan.set_ylabel(r'$\beta_x$ [m]')
    ax_betx_scan.set_title('Beta functions across the tune scan')
    ax_betx_scan.grid(True, alpha=0.3)
    ax_betx_scan.legend()
    ax_bety_scan.set_xlabel('s [m]')
    ax_bety_scan.set_ylabel(r'$\beta_y$ [m]')
    ax_bety_scan.grid(True, alpha=0.3)
    ax_bety_scan.legend()

    sm_beta_abs = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm_beta_abs.set_array([])
    fig_beta_scan.colorbar(sm_beta_abs, ax=[ax_betx_scan, ax_bety_scan], label='Horizontal offset [m]')
    fig_beta_scan.suptitle(case_label)

    # Beta beat at each offset of the tune scan, relative to the
    # no-undulator baseline -- same colour-coded-by-offset layout as the
    # orbit scan above.
    fig_beta_diff, (ax_dbetx, ax_dbety) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for dx, betx, bety in zip(hor_off_list, betx_scan, bety_scan):
        color = cmap(norm(dx))
        ax_dbetx.plot(orbit_scan_s, betx - betx_no_und_i, color=color)
        ax_dbety.plot(orbit_scan_s, bety - bety_no_und_i, color=color)
    mark_undulator_bounds(ax_dbetx)
    mark_undulator_bounds(ax_dbety)
    ax_dbetx.set_ylabel(r'$\Delta\beta_x$ [m]')
    ax_dbetx.set_title('Beta beat across the tune scan (relative to no undulator)')
    ax_dbetx.grid(True, alpha=0.3)
    ax_dbety.set_xlabel('s [m]')
    ax_dbety.set_ylabel(r'$\Delta\beta_y$ [m]')
    ax_dbety.grid(True, alpha=0.3)

    sm_beta = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm_beta.set_array([])
    fig_beta_diff.colorbar(sm_beta, ax=[ax_dbetx, ax_dbety], label='Horizontal offset [m]')
    fig_beta_diff.suptitle(case_label)

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

    coef_qx_formula_quadonly = np.polyfit(hor_off_list, deltaqx_formula_quadonly_list, 2)
    coef_qy_formula_quadonly = np.polyfit(hor_off_list, deltaqy_formula_quadonly_list, 2)
    print(f"[formula, quadrupole only, no Bs] (1/2) d²(ΔQx)/dx² = {coef_qx_formula_quadonly[0]}")
    print(f"[formula, quadrupole only, no Bs] d(ΔQx)/dx         = {coef_qx_formula_quadonly[1]}")
    print(f"[formula, quadrupole only, no Bs] ΔQx(0)            = {coef_qx_formula_quadonly[2]}")
    print(f"[formula, quadrupole only, no Bs] (1/2) d²(ΔQy)/dx² = {coef_qy_formula_quadonly[0]}")
    print(f"[formula, quadrupole only, no Bs] d(ΔQy)/dx         = {coef_qy_formula_quadonly[1]}")
    print(f"[formula, quadrupole only, no Bs] ΔQy(0)            = {coef_qy_formula_quadonly[2]}")

    text_box_kwargs = dict(va='top', ha='left', fontsize=8, linespacing=1.4,
                            bbox=dict(boxstyle='round', fc='white', alpha=0.85, edgecolor='0.7'))

    fig_tune_shift, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(hor_off_list, deltaqx_list, marker='o', color='tab:blue', label='Twiss')
    ax1.plot(hor_off_list, poly_qx(hor_off_list), linestyle='--', color='k', label='Quadratic fit')
    ax1.plot(hor_off_list, deltaqx_formula_list, marker='^', linestyle='none',
              color='tab:green', label=r'$\frac{1}{4\pi}\oint(\delta K_1+K_{sol})\beta_{x,0}\,ds$')
    ax1.plot(hor_off_list, deltaqx_formula_pert_list, marker='v', linestyle='none',
              color='tab:red', label=r'$\frac{1}{4\pi}\oint(\delta K_1+K_{sol})\beta_x\,ds$')
    ax1.plot(hor_off_list, deltaqx_formula_quadonly_list, marker='D', linestyle='none',
              color='tab:purple', label=r'$\frac{1}{4\pi}\oint\delta K_1\,\beta_{x,0}\,ds$ (no $B_s$)')
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
              color='tab:green', label=r'$\frac{1}{4\pi}\oint(K_{sol}-\delta K_1)\beta_{y,0}\,ds$')
    ax2.plot(hor_off_list, deltaqy_formula_pert_list, marker='v', linestyle='none',
              color='tab:red', label=r'$\frac{1}{4\pi}\oint(K_{sol}-\delta K_1)\beta_y\,ds$')
    ax2.plot(hor_off_list, deltaqy_formula_quadonly_list, marker='D', linestyle='none',
              color='tab:purple', label=r'$\frac{1}{4\pi}\oint(-\delta K_1)\beta_{y,0}\,ds$ (no $B_s$)')
    ax2.set_xlabel('Horizontal offset [m]')
    ax2.set_ylabel('Delta Qy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.text(0.02, 0.95,
              f'$\\frac{{1}}{{2}}\\frac{{d^2\\Delta Q_y}}{{dx^2}}$ = {coef_qy[0]:.4e}\n'
              f'$\\frac{{d\\Delta Q_y}}{{dx}}$ = {coef_qy[1]:.4e}\n'
              f'$\\Delta Q_y(0)$ = {coef_qy[2]:.4e}',
              transform=ax2.transAxes, **text_box_kwargs)

    fig_tune_shift.suptitle(case_label)
    fig_tune_shift.tight_layout()

    # Tracked-vs-calculated tune shift residual: how far the analytic
    # perturbation-theory formula (both beta variants) is from the actual
    # Twiss/tracked result plotted above, as a function of offset -- makes
    # the (generally much smaller) formula/Twiss gap visible on its own
    # scale rather than as an overlap of near-identical curves.
    deltaqx_diff_beta0 = deltaqx_list - deltaqx_formula_list
    deltaqx_diff_beta = deltaqx_list - deltaqx_formula_pert_list
    deltaqy_diff_beta0 = deltaqy_list - deltaqy_formula_list
    deltaqy_diff_beta = deltaqy_list - deltaqy_formula_pert_list
    deltaqx_diff_quadonly = deltaqx_list - deltaqx_formula_quadonly_list
    deltaqy_diff_quadonly = deltaqy_list - deltaqy_formula_quadonly_list

    fig_tune_shift_diff, (ax1_diff, ax2_diff) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1_diff.plot(hor_off_list, deltaqx_diff_beta0, marker='^', color='tab:green',
                  label=r'Twiss $-\ \frac{1}{4\pi}\oint(\delta K_1+K_{sol})\beta_{x,0}\,ds$')
    ax1_diff.plot(hor_off_list, deltaqx_diff_beta, marker='v', color='tab:red',
                  label=r'Twiss $-\ \frac{1}{4\pi}\oint(\delta K_1+K_{sol})\beta_x\,ds$')
    ax1_diff.plot(hor_off_list, deltaqx_diff_quadonly, marker='D', color='tab:purple',
                  label=r'Twiss $-\ \frac{1}{4\pi}\oint\delta K_1\,\beta_{x,0}\,ds$ (no $B_s$)')
    ax1_diff.axhline(0, color='0.4', linewidth=1)
    ax1_diff.set_ylabel(r'$\Delta Q_x$ residual')
    ax1_diff.set_title('Tracked minus calculated tune shift vs undulator horizontal offset')
    ax1_diff.grid(True, alpha=0.3)
    ax1_diff.legend()

    ax2_diff.plot(hor_off_list, deltaqy_diff_beta0, marker='^', color='tab:green',
                  label=r'Twiss $-\ \frac{1}{4\pi}\oint(K_{sol}-\delta K_1)\beta_{y,0}\,ds$')
    ax2_diff.plot(hor_off_list, deltaqy_diff_beta, marker='v', color='tab:red',
                  label=r'Twiss $-\ \frac{1}{4\pi}\oint(K_{sol}-\delta K_1)\beta_y\,ds$')
    ax2_diff.plot(hor_off_list, deltaqy_diff_quadonly, marker='D', color='tab:purple',
                  label=r'Twiss $-\ \frac{1}{4\pi}\oint(-\delta K_1)\beta_{y,0}\,ds$ (no $B_s$)')
    ax2_diff.axhline(0, color='0.4', linewidth=1)
    ax2_diff.set_xlabel('Horizontal offset [m]')
    ax2_diff.set_ylabel(r'$\Delta Q_y$ residual')
    ax2_diff.grid(True, alpha=0.3)
    ax2_diff.legend()

    fig_tune_shift_diff.suptitle(case_label)
    fig_tune_shift_diff.tight_layout()

    # Same residual, before vs. after applying the closest-tune-approach
    # (betatron coupling) correction computed above -- if the coupling
    # picture accounts for the residual seen in fig_tune_shift_diff, the
    # solid (corrected) curves should sit much closer to zero than the
    # dashed (uncorrected) ones.
    fig_tune_shift_corr, (ax1_corr, ax2_corr) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1_corr.plot(hor_off_list, deltaqx_diff_beta0, marker='^', linestyle='--',
                  color='tab:green', alpha=0.4, label=r'Uncorrected ($\beta_{x,0}$)')
    ax1_corr.plot(hor_off_list, data['deltaqx_resid_corr_beta0'], marker='^',
                  color='tab:green', label=r'$C^-$-corrected ($\beta_{x,0}$)')
    ax1_corr.plot(hor_off_list, deltaqx_diff_beta, marker='v', linestyle='--',
                  color='tab:red', alpha=0.4, label=r'Uncorrected ($\beta_x$)')
    ax1_corr.plot(hor_off_list, data['deltaqx_resid_corr_pert'], marker='v',
                  color='tab:red', label=r'$C^-$-corrected ($\beta_x$)')
    ax1_corr.plot(hor_off_list, deltaqx_diff_quadonly, marker='D', linestyle='--',
                  color='tab:purple', alpha=0.4, label=r'Uncorrected (no $B_s$)')
    ax1_corr.plot(hor_off_list, data['deltaqx_resid_corr_quadonly'], marker='D',
                  color='tab:purple', label=r'$C^-$-corrected (no $B_s$)')
    ax1_corr.axhline(0, color='0.4', linewidth=1)
    ax1_corr.set_ylabel(r'$\Delta Q_x$ residual')
    ax1_corr.set_title('Tune-shift residual before/after closest-tune-approach correction')
    ax1_corr.grid(True, alpha=0.3)
    ax1_corr.legend(fontsize=7)

    ax2_corr.plot(hor_off_list, deltaqy_diff_beta0, marker='^', linestyle='--',
                  color='tab:green', alpha=0.4, label=r'Uncorrected ($\beta_{y,0}$)')
    ax2_corr.plot(hor_off_list, data['deltaqy_resid_corr_beta0'], marker='^',
                  color='tab:green', label=r'$C^-$-corrected ($\beta_{y,0}$)')
    ax2_corr.plot(hor_off_list, deltaqy_diff_beta, marker='v', linestyle='--',
                  color='tab:red', alpha=0.4, label=r'Uncorrected ($\beta_y$)')
    ax2_corr.plot(hor_off_list, data['deltaqy_resid_corr_pert'], marker='v',
                  color='tab:red', label=r'$C^-$-corrected ($\beta_y$)')
    ax2_corr.plot(hor_off_list, deltaqy_diff_quadonly, marker='D', linestyle='--',
                  color='tab:purple', alpha=0.4, label=r'Uncorrected (no $B_s$)')
    ax2_corr.plot(hor_off_list, data['deltaqy_resid_corr_quadonly'], marker='D',
                  color='tab:purple', label=r'$C^-$-corrected (no $B_s$)')
    ax2_corr.axhline(0, color='0.4', linewidth=1)
    ax2_corr.set_xlabel('Horizontal offset [m]')
    ax2_corr.set_ylabel(r'$\Delta Q_y$ residual')
    ax2_corr.grid(True, alpha=0.3)
    ax2_corr.legend(fontsize=7)

    fig_tune_shift_corr.suptitle(case_label)
    fig_tune_shift_corr.tight_layout()

    # Save the 8 figures for this case, named
    # "<place_label>_<model_label>_<what the figure shows>.pdf".
    figures = [
        (fig_field_traj, 'field_along_trajectory'),
        (fig_orbit, 'orbit_comparison'),
        (fig_orbit_scan, 'orbit_scan'),
        (fig_beta_scan, 'beta_functions'),
        (fig_beta_diff, 'beta_beat'),
        (fig_tune_shift, 'tune_shift'),
        (fig_tune_shift_diff, 'tune_shift_difference'),
        (fig_tune_shift_corr, 'tune_shift_coupling_corrected'),
        ]
    for fig, suffix in figures:
        out_path = OUT_DIR / f'{place_label}_{model_label}_{suffix}.pdf'
        fig.savefig(out_path)
        print(f"Saved {out_path}")


for place_label, wiggler_places in WIGGLER_CASES:
    for model_label in MODEL_LABELS:
        case_data = get_case_data(place_label, wiggler_places, model_label)
        plot_case(case_data, place_label, model_label)

# All 8*3*2 = 48 figures across every case are kept open (not closed inside
# plot_case()) so they can all be reviewed interactively here, in addition
# to having been saved as PDFs above.
plt.show()
