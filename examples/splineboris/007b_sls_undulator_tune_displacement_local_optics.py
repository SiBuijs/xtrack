import os
import sys
import xtrack as xt
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as sc_const
from xtrack._temp.splineboris import TubeFitter, LongitudinalFitter


multipole_order = 3

E0 = 2.7e9

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)

# Load SLS MADX file
madx_file = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'sls.madx'

BASE_DIR = Path(__file__).resolve().parent

# Raw field map data (shared test_data) -- ~1.8 GB, so loading and fitting it
# is by far the slowest part of this script. Deferred to get_undulator_fit(),
# called lazily from compute_case() only for cases that are actually being
# (re)computed -- a `--replot` run where every case already has cached data
# in DATA_DIR never touches this file at all.
file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "simona_field_map.txt"

# Distance unit in meters (the dataset uses mm, so 1 mm = 0.001 m)
distance_unit = 0.001

n_frames = 4441

_undulator_fit = None

# Drop components whose scale max|f| * r_ref^n / n! is below field_tol times
# the largest field (r_ref: the map's transverse half-width).
FIELD_TOL = dict(field_tol=1e-4, r_ref=2e-3)


def get_undulator_fit():
    global _undulator_fit
    if _undulator_fit is None:
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
            distance_unit=distance_unit,
            deg=multipole_order - 1,
        )
        tf.fit()
        # Longitudinal fit of the on-axis components ("free" ends: the map
        # ends are not field-free; ~5 frames per element by default).
        z, F, names = tf.on_axis_multipoles()
        lf = LongitudinalFitter(z[0], z[-1], end_condition="free", period=0.036)
        lf.fit(z, F, names)
        lf.fit(*tf.on_axis_bs(), [("Bs", 0)])
        _undulator_fit = lf
    return _undulator_fit

# Same base dir as 007a (default ~/cernbox/Pictures/SLS_Undulator_Studies,
# overridable via the SLS_UNDULATOR_PLOT_DIR env var), in a local_optics
# subdirectory.
OUT_DIR = Path(os.environ.get(
    'SLS_UNDULATOR_PLOT_DIR',
    Path.home() / 'cernbox' / 'Pictures' / 'SLS_Undulator_Studies')) / 'local_optics'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Separate cache dir from 007a's -- compute_case()'s return schema here is
# entirely different (per-placement arrays, no ring-wide/c_minus data), so
# sharing 007a's DATA_DIR would either collide on filenames or leave stale/
# incompatible keys behind.
DATA_DIR = Path(__file__).resolve().parent / 'data_007b'
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPLOT = '--replot' in sys.argv

# Three placement cases (only ars11_uind_0210_1, only ars11_uind_0610_1,
# both) x two undulator models (SB = SplineBoris, from to_line(); MK =
# Multipole Kick, from to_multipole_line()) -- compute_case()/plot_case()
# below build and analyse one such case end to end and save its figures.
WIGGLER_CASES = [
    ('ars11_uind_0210_1', ['ars11_uind_0210_1']),
    ('ars11_uind_0610_1', ['ars11_uind_0610_1']),
    ('both', ['ars11_uind_0210_1', 'ars11_uind_0610_1']),
]
MODEL_LABELS = ('SB', 'MK')

# Digitized reference measurement (X11MA, gap = 11.5 mm) -- see 007a for
# its provenance and caveats (correspondence with the simulated placements
# NOT confirmed; dtune_*_measured = closed-gap minus open-gap tunes,
# subtracted point by point; tune_{x,y}_{open,closed}_gap are absolute).
MEASURED_TUNE_SHIFT_CSV = (
    Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls'
    / 'x11ma_gap11p5mm_tune_shift_digitized.csv'
)


def load_measured_tune_shift():
    if not MEASURED_TUNE_SHIFT_CSV.exists():
        return None
    return pd.read_csv(MEASURED_TUNE_SHIFT_CSV)


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

    Unlike 007a, the undulator is never inserted into the SLS ring. Instead:
    the bare ring (no undulator at all) is Twissed once to get its tunes and
    its optics (betx, bety, alfx, alfy) at the start of each undulator
    placement; those four numbers seed a Twiss of the *standalone* undulator
    line only (line.twiss4d(betx=..., bety=..., alfx=..., alfy=...)) for
    every offset in the scan. The tune shift is then read directly off the
    phase advance (mux/muy, already in units of 2 pi) the undulator adds,
    relative to what that same physical span of bare ring would have
    contributed -- a decoupled, first-order-in-isolation picture that never
    needs to know about the rest of the ring. For the 'both' placement case
    this also means the two undulators are two independent Twiss problems
    (same design, different initial optics) whose phase-advance
    contributions are simply summed; there is no cross-talk between them and
    no periodic/closed-ring quantity (like c_minus) available here, so
    unlike 007a there is no closest-tune-approach coupling correction.
    """
    print("=" * 80)
    print(f"Case: place={place_label}  model={model_label}")
    case_label = f'{place_label} ({model_label}, local optics)'
    print("=" * 80)

    undulator_fit = get_undulator_fit()

    # Load SLS MADX file (only needed for the bare-ring Twiss below -- the
    # undulator itself is never inserted into it in this version)
    env = xt.load(str(madx_file))
    line_sls = env.ring
    line_sls.configure_bend_model(core='mat-kick-mat')
    line_sls.particle_ref = p0.copy()

    # Build undulator -- SB uses LongitudinalFitter.to_line() (one SplineBoris
    # element per polynomial piece, full spatial field integration); MK uses
    # LongitudinalFitter.to_multipole_line() (one thick Multipole per region, a
    # coarser rigidity-normalized approximation).
    und_env = xt.Environment()
    und_env.particle_ref = p0.copy()

    if model_label == 'SB':
        undulator_line = undulator_fit.to_line(multipole_order=multipole_order, **FIELD_TOL)
    else:
        undulator_line = undulator_fit.to_multipole_line(
            multipole_order=multipole_order, p0c=E0, field_at='mean', **FIELD_TOL)
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

    und_env.new('corr1', xt.Multipole, knl=['k0l_corr1'], ksl=['k0sl_corr1'])
    und_env.new('corr2', xt.Multipole, knl=['k0l_corr2'], ksl=['k0sl_corr2'])
    und_env.new('corr3', xt.Multipole, knl=['k0l_corr3'], ksl=['k0sl_corr3'])
    und_env.new('corr4', xt.Multipole, knl=['k0l_corr4'], ksl=['k0sl_corr4'])

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

    # Bare-ring Twiss -- tune baseline (qx_0/qy_0) and, per placement, the
    # source of both the local initial conditions (betx, bety, alfx, alfy)
    # and the bare-ring phase advance across that same physical span.
    tw_no_undulator = line_sls.twiss4d(include_collective=True)
    qx_0 = tw_no_undulator.qx
    qy_0 = tw_no_undulator.qy

    n_placements = len(wiggler_places)
    init_conditions = []
    for wig_place in wiggler_places:
        s0 = float(tw_no_undulator['s', wig_place])
        ic = dict(
            betx=float(tw_no_undulator['betx', wig_place]),
            bety=float(tw_no_undulator['bety', wig_place]),
            alfx=float(tw_no_undulator['alfx', wig_place]),
            alfy=float(tw_no_undulator['alfy', wig_place]),
            s0=s0,
        )
        init_conditions.append(ic)
        print(f"Undulator at {wig_place} (s={s0:.4f} m): "
              f"betx={ic['betx']:.4f} bety={ic['bety']:.4f} "
              f"alfx={ic['alfx']:.4f} alfy={ic['alfy']:.4f}")

    undulator_field_element_names = [
        nn for nn in undulator.element_names if nn.startswith(('splineboris_', 'multipole_'))
        ]
    # For the MK model, to_multipole_line() also inserts a pair of thin
    # xt.MultipoleEdge kicks around each region (see longitudinal_fitter.py) -- they
    # match the field-element name filter above (so they still get shifted
    # together with their parent region, which is physically correct) but
    # have no .length/.knl/.ksl/.get_field(), so per-element field sampling
    # below needs the thick-body-only subset instead.
    undulator_field_element_names_thick = [
        nn for nn in undulator_field_element_names
        if not isinstance(undulator[nn], xt.MultipoleEdge)
        ]
    undulator_s_ranges = [(ic['s0'], ic['s0'] + l_wig) for ic in init_conditions]

    brho0 = E0 / (sc_const.c * 1.0)

    def _twiss_undulator(ic):
        return undulator.twiss4d(
            betx=ic['betx'], bety=ic['bety'],
            alfx=ic['alfx'], alfy=ic['alfy'],
            include_collective=True)

    # Field along the closed-orbit trajectory, on-axis vs. off-axis, per
    # placement -- same field-evaluation logic as 007a, but the trajectory
    # now comes from Twissing only the standalone undulator, seeded with
    # that placement's own initial conditions from the bare ring.
    field_tt = undulator.get_table()
    n_sub = 6  # field-sample points per field element (for smoother curves)

    def _sample_field(traj_s, traj_x, traj_y, dx):
        s_out, bx_out, by_out, bs_out = [], [], [], []
        for nn in undulator_field_element_names_thick:
            s0 = field_tt['s', nn]
            length = undulator[nn].length
            for s_local in np.linspace(0.0, length, n_sub, endpoint=False):
                s_glob = s0 + s_local
                x_local = np.interp(s_glob, traj_s, traj_x) - dx
                y_local = np.interp(s_glob, traj_s, traj_y)
                if model_label == 'SB':
                    bx, by, bs = undulator[nn].get_field(x_local, y_local, s_local)
                else:
                    bx, by = _multipole_field_bx_by(
                        undulator[nn].knl, undulator[nn].ksl, length, brho0,
                        x_local, y_local)
                    bs = 0.0
                s_out.append(s_glob)
                bx_out.append(bx)
                by_out.append(by)
                bs_out.append(bs)
        return (np.array(s_out), np.array(bx_out), np.array(by_out),
                np.array(bs_out))

    field_s_on_pp, field_bx_on_pp, field_by_on_pp, field_bs_on_pp = [], [], [], []
    field_s_off_pp, field_bx_off_pp, field_by_off_pp, field_bs_off_pp = [], [], [], []
    tw_no_undulator_local_x_pp = []
    tw_no_undulator_local_y_pp = []
    tw_onaxis_s_pp, tw_onaxis_x_pp, tw_onaxis_y_pp = [], [], []
    tw_offaxis_s_pp, tw_offaxis_x_pp, tw_offaxis_y_pp = [], [], []

    for ic in init_conditions:
        for nn in undulator_field_element_names:
            undulator[nn].shift_x = 0.
        tw_onaxis = _twiss_undulator(ic)

        for nn in undulator_field_element_names:
            undulator[nn].shift_x = 0.5e-3
        tw_offaxis = _twiss_undulator(ic)

        s_on, bx_on, by_on, bs_on = _sample_field(
            tw_onaxis.s, tw_onaxis.x, tw_onaxis.y, 0.0)
        s_off, bx_off, by_off, bs_off = _sample_field(
            tw_offaxis.s, tw_offaxis.x, tw_offaxis.y, 0.5e-3)

        field_s_on_pp.append(ic['s0'] + s_on)
        field_bx_on_pp.append(bx_on)
        field_by_on_pp.append(by_on)
        field_bs_on_pp.append(bs_on)
        field_s_off_pp.append(ic['s0'] + s_off)
        field_bx_off_pp.append(bx_off)
        field_by_off_pp.append(by_off)
        field_bs_off_pp.append(bs_off)

        tw_onaxis_s_pp.append(ic['s0'] + tw_onaxis.s)
        tw_onaxis_x_pp.append(tw_onaxis.x)
        tw_onaxis_y_pp.append(tw_onaxis.y)
        tw_offaxis_s_pp.append(ic['s0'] + tw_offaxis.s)
        tw_offaxis_x_pp.append(tw_offaxis.x)
        tw_offaxis_y_pp.append(tw_offaxis.y)
        tw_no_undulator_local_x_pp.append(
            np.interp(ic['s0'] + tw_onaxis.s, tw_no_undulator.s, tw_no_undulator.x))
        tw_no_undulator_local_y_pp.append(
            np.interp(ic['s0'] + tw_onaxis.s, tw_no_undulator.s, tw_no_undulator.y))

    # local twiss s-grid per placement -- identical across dx (element
    # boundaries don't move with shift_x), so the shift_x=0 field-trajectory
    # pass above already gives the canonical grid.
    orbit_scan_s_pp = np.array(tw_onaxis_s_pp)

    deltaqx_list = []
    deltaqy_list = []
    orbit_scan_x_pp = [[] for _ in range(n_placements)]
    orbit_scan_y_pp = [[] for _ in range(n_placements)]
    betx_scan_pp = [[] for _ in range(n_placements)]
    bety_scan_pp = [[] for _ in range(n_placements)]
    betx2_scan_pp = [[] for _ in range(n_placements)]
    bety1_scan_pp = [[] for _ in range(n_placements)]

    # Scan the offsets at the measured bump amplitudes when the measurement
    # is available, so the simulated tunes can be compared point by point
    # with the measured open-gap tunes (see plot_models_combined()).
    measured = load_measured_tune_shift()
    if measured is not None:
        hor_off_list = np.asarray(measured['bump_amplitude_m'], dtype=float)
    else:
        n_tunes = 30
        hor_off_list = np.linspace(-0.5e-3, 0.5e-3, n_tunes)

    corrector_vars = ['k0l_corr1', 'k0sl_corr1', 'k0l_corr2', 'k0sl_corr2',
                       'k0l_corr3', 'k0sl_corr3', 'k0l_corr4', 'k0sl_corr4']

    # Local x-deflection baseline (same rationale as 007a): even at
    # shift_x=0, the beam deflects horizontally inside the undulator.
    # Reset shift_x first -- the off-axis field-trajectory pass above
    # leaves the undulator at shift_x=0.5 mm.
    for nn in undulator_field_element_names:
        undulator[nn].shift_x = 0.
    p_deflection = p0.copy()
    undulator.track(p_deflection, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_deflection = undulator.record_last_track
    x_deflection_baseline = float(np.mean(mon_deflection.x[0, :]))
    print(f"[x-deflection baseline] literal single-particle track: "
          f"mean x = {x_deflection_baseline * 1e6:.3f} um")

    for dx in hor_off_list:   # dx in meters
        for nn in undulator_field_element_names:
            undulator[nn].shift_x = dx

        # re-correct the orbit for this offset (same rationale as 007a --
        # fixed correctors matched only at shift_x=0 would otherwise leak a
        # residual orbit distortion into the phase-advance calculation)
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

        # Twiss the standalone undulator, once per placement, seeded with
        # that placement's own bare-ring initial conditions; the tune shift
        # contribution of each placement is the phase advance it adds
        # (mux/muy, already in units of 2 pi) minus the phase advance the
        # same physical span of bare ring would have contributed -- summed
        # over placements for the 'both' case.
        dqx_total = 0.0
        dqy_total = 0.0
        for i, ic in enumerate(init_conditions):
            tw_p = _twiss_undulator(ic)

            mux_bare_start = np.interp(ic['s0'], tw_no_undulator.s, tw_no_undulator.mux)
            mux_bare_end = np.interp(ic['s0'] + l_wig, tw_no_undulator.s, tw_no_undulator.mux)
            muy_bare_start = np.interp(ic['s0'], tw_no_undulator.s, tw_no_undulator.muy)
            muy_bare_end = np.interp(ic['s0'] + l_wig, tw_no_undulator.s, tw_no_undulator.muy)

            dqx_total += tw_p.mux[-1] - (mux_bare_end - mux_bare_start)
            dqy_total += tw_p.muy[-1] - (muy_bare_end - muy_bare_start)

            orbit_scan_x_pp[i].append(tw_p.x)
            orbit_scan_y_pp[i].append(tw_p.y)
            betx_scan_pp[i].append(tw_p.betx)
            bety_scan_pp[i].append(tw_p.bety)
            betx2_scan_pp[i].append(tw_p.betx2)
            bety1_scan_pp[i].append(tw_p.bety1)

        deltaqx_list.append(dqx_total)
        deltaqy_list.append(dqy_total)

    orbit_scan_x_pp = np.array(orbit_scan_x_pp)  # (n_placements, n_offsets, n_pts)
    orbit_scan_y_pp = np.array(orbit_scan_y_pp)
    betx_scan_pp = np.array(betx_scan_pp)
    bety_scan_pp = np.array(bety_scan_pp)
    betx2_scan_pp = np.array(betx2_scan_pp)
    bety1_scan_pp = np.array(bety1_scan_pp)

    # Independent analytic cross-check of the tune shift (same formula and
    # rationale as 007a), summed over placements the same way the
    # phase-advance calculation above is.
    multipole_line = undulator_fit.to_multipole_line(
        multipole_order=multipole_order, p0c=E0, field_at='mean', **FIELD_TOL)
    mult_table = multipole_line.get_table()
    # to_multipole_line() also brackets each region with thin xt.MultipoleEdge
    # kicks (see longitudinal_fitter.py) -- excluded here since K1(s)/K2(s)/K2_skew(s)
    # is a per-region (thick-body) quantity; the edge kicks are picked up by
    # the Twiss itself (via mux/muy above) but aren't part of this analytic
    # deltaK1(s)*beta(s) integral.
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
    deltaqx_formula_pert_list = []
    deltaqy_formula_pert_list = []
    for i_dx in range(len(hor_off_list)):
        dx = hor_off_list[i_dx]
        integral_x = 0.0
        integral_y = 0.0
        integral_x_pert = 0.0
        integral_y_pert = 0.0
        for i, ic in enumerate(init_conditions):
            s_global = ic['s0'] + mult_s_mid
            x_local = (np.interp(s_global, orbit_scan_s_pp[i], orbit_scan_x_pp[i, i_dx]) - dx
                       + x_deflection_baseline)
            y_local = np.interp(s_global, orbit_scan_s_pp[i], orbit_scan_y_pp[i, i_dx])
            deltaK1 = mult_K1 + mult_K2 * x_local - mult_K2_skew * y_local
            betx_at_s = np.interp(s_global, tw_no_undulator.s, tw_no_undulator.betx)
            bety_at_s = np.interp(s_global, tw_no_undulator.s, tw_no_undulator.bety)
            integral_x += np.sum(deltaK1 * betx_at_s * mult_length)
            integral_y += np.sum(deltaK1 * bety_at_s * mult_length)

            betx_pert_at_s = np.interp(s_global, orbit_scan_s_pp[i], betx_scan_pp[i, i_dx])
            bety_pert_at_s = np.interp(s_global, orbit_scan_s_pp[i], bety_scan_pp[i, i_dx])
            integral_x_pert += np.sum(deltaK1 * betx_pert_at_s * mult_length)
            integral_y_pert += np.sum(deltaK1 * bety_pert_at_s * mult_length)
        # Sign convention as established/verified in 007a.
        deltaqx_formula_list.append(integral_x / (4 * np.pi))
        deltaqy_formula_list.append(-integral_y / (4 * np.pi))
        deltaqx_formula_pert_list.append(integral_x_pert / (4 * np.pi))
        deltaqy_formula_pert_list.append(-integral_y_pert / (4 * np.pi))

    # tw_no_undulator carries a different element grid than the standalone
    # undulator's, so interpolate it onto each placement's local twiss grid.
    betx_no_und_i_pp = np.array([
        np.interp(orbit_scan_s_pp[i], tw_no_undulator.s, tw_no_undulator.betx)
        for i in range(n_placements)])
    bety_no_und_i_pp = np.array([
        np.interp(orbit_scan_s_pp[i], tw_no_undulator.s, tw_no_undulator.bety)
        for i in range(n_placements)])
    betx2_no_und_i_pp = np.array([
        np.interp(orbit_scan_s_pp[i], tw_no_undulator.s, tw_no_undulator.betx2)
        for i in range(n_placements)])
    bety1_no_und_i_pp = np.array([
        np.interp(orbit_scan_s_pp[i], tw_no_undulator.s, tw_no_undulator.bety1)
        for i in range(n_placements)])

    return dict(
        case_label=case_label,
        n_placements=n_placements,
        hor_off_list=hor_off_list,
        qx_0=qx_0,
        qy_0=qy_0,
        deltaqx_list=np.array(deltaqx_list),
        deltaqy_list=np.array(deltaqy_list),
        undulator_s_ranges=np.array(undulator_s_ranges),
        orbit_scan_s_pp=orbit_scan_s_pp,
        orbit_scan_x_pp=orbit_scan_x_pp,
        orbit_scan_y_pp=orbit_scan_y_pp,
        betx_scan_pp=betx_scan_pp,
        bety_scan_pp=bety_scan_pp,
        betx2_scan_pp=betx2_scan_pp,
        bety1_scan_pp=bety1_scan_pp,
        betx_no_und_i_pp=betx_no_und_i_pp,
        bety_no_und_i_pp=bety_no_und_i_pp,
        betx2_no_und_i_pp=betx2_no_und_i_pp,
        bety1_no_und_i_pp=bety1_no_und_i_pp,
        deltaqx_formula_list=np.array(deltaqx_formula_list),
        deltaqy_formula_list=np.array(deltaqy_formula_list),
        deltaqx_formula_pert_list=np.array(deltaqx_formula_pert_list),
        deltaqy_formula_pert_list=np.array(deltaqy_formula_pert_list),
        tw_onaxis_s_pp=np.array(tw_onaxis_s_pp),
        tw_onaxis_x_pp=np.array(tw_onaxis_x_pp),
        tw_onaxis_y_pp=np.array(tw_onaxis_y_pp),
        tw_offaxis_s_pp=np.array(tw_offaxis_s_pp),
        tw_offaxis_x_pp=np.array(tw_offaxis_x_pp),
        tw_offaxis_y_pp=np.array(tw_offaxis_y_pp),
        tw_no_undulator_local_x_pp=np.array(tw_no_undulator_local_x_pp),
        tw_no_undulator_local_y_pp=np.array(tw_no_undulator_local_y_pp),
        field_s_on_pp=np.array(field_s_on_pp),
        field_bx_on_pp=np.array(field_bx_on_pp),
        field_by_on_pp=np.array(field_by_on_pp),
        field_bs_on_pp=np.array(field_bs_on_pp),
        field_s_off_pp=np.array(field_s_off_pp),
        field_bx_off_pp=np.array(field_bx_off_pp),
        field_by_off_pp=np.array(field_by_off_pp),
        field_bs_off_pp=np.array(field_bs_off_pp),
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


# Style of the fit-coefficient text boxes stacked under the legends.
text_box_kwargs = dict(va='top', ha='left', fontsize=8, linespacing=1.4,
                       bbox=dict(boxstyle='round', fc='white', alpha=0.85, edgecolor='0.7'))


def legend_outside(ax):
    # Anchor the legend just outside the right edge of the axes (same as
    # 007a) -- figures using this are saved with bbox_inches='tight' and a
    # subplots_adjust(right=...) so the legend stays on-canvas.
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8)


def stack_boxes_under_legend(ax, boxes):
    # Stack coefficient text boxes down the right margin, starting just
    # below the legend and left-aligned with it (same as 007a). Must run
    # *after* tight_layout()/subplots_adjust(), since it measures the
    # rendered legend/box geometry.
    fig = ax.get_figure()
    fig.canvas.draw()
    to_axes = ax.transAxes.inverted()

    def frame_bbox(artist):
        patch = artist.get_bbox_patch()
        return (patch or artist).get_window_extent().transformed(to_axes)

    legend_bbox = ax.get_legend().get_window_extent().transformed(to_axes)
    y = legend_bbox.y0
    for text, kwargs in boxes:
        y -= 0.04
        artist = ax.text(1.01, y, text, transform=ax.transAxes, **kwargs)
        fig.canvas.draw()
        # line the box's frame up with the legend's frame
        artist.set_x(artist.get_position()[0]
                     + (legend_bbox.x0 - frame_bbox(artist).x0))
        fig.canvas.draw()
        y = frame_bbox(artist).y0


def fit_box_text(plane, coef, prefix=''):
    return (f'{prefix}'
            f'$\\frac{{1}}{{2}}\\frac{{d^2\\Delta Q_{plane}}}{{dx^2}}$ = {coef[0]:.4e}\n'
            f'$\\frac{{d\\Delta Q_{plane}}}{{dx}}$ = {coef[1]:.4e}\n'
            f'$\\Delta Q_{plane}(0)$ = {coef[2]:.4e}')


def tune_shift_comparison_figure(sim_series, x_meas, dq_meas, meas_label,
                                 title, suptitle):
    """Simulated vs measured Delta Q_x,y with a quadratic fit through each
    and coefficient boxes in the right margin (same as 007a). sim_series is
    a list of (x, dq, label, color, marker, fit_box_prefix), one per
    simulated curve; dq and dq_meas are dicts keyed 'x'/'y', and labels may
    contain {p} for the plane.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes_boxes = []
    for ax, plane in zip(axes, ('x', 'y')):
        boxes = []
        for x_sim, dq_sim, label, color, marker, box_prefix in sim_series:
            coef_sim = np.polyfit(x_sim, dq_sim[plane], 2)
            print(f"[{suptitle}: {title}] {label.format(p=plane)}: "
                  f"ΔQ{plane} quadratic fit coefficients {coef_sim}")
            ax.plot(x_sim, dq_sim[plane], marker=marker, markersize=4,
                    color=color, label=label.format(p=plane))
            ax.plot(x_sim, np.poly1d(coef_sim)(x_sim), linestyle='--',
                    color='k', linewidth=1)
            boxes.append((fit_box_text(plane, coef_sim, box_prefix),
                          text_box_kwargs))
        coef_meas = np.polyfit(x_meas, dq_meas[plane], 2)
        ax.plot(x_meas, dq_meas[plane], marker='d', linestyle='none',
                markersize=4, color='tab:purple',
                label=meas_label.format(p=plane))
        ax.plot(x_meas, np.poly1d(coef_meas)(x_meas), linestyle='--',
                color='tab:red', label='Quadratic fit (measured)')
        # one legend entry for all the simulated curves' fits
        ax.plot([], [], linestyle='--', color='k', linewidth=1,
                label='Quadratic fit')
        boxes.append((fit_box_text(plane, coef_meas, 'measured fit:\n'),
                      text_box_kwargs))
        ax.set_ylabel(f'$\\Delta Q_{plane}$')
        ax.grid(True, alpha=0.3)
        legend_outside(ax)
        axes_boxes.append(boxes)
    axes[0].set_title(title)
    axes[1].set_xlabel('Horizontal offset [m]')

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.subplots_adjust(right=0.62)
    for ax, boxes in zip(axes, axes_boxes):
        stack_boxes_under_legend(ax, boxes)
    return fig


def plot_case(data, place_label, model_label):
    """Build and save the figures for one case, purely from the dict
    returned by compute_case()/get_case_data() -- no line building, matching
    or twissing happens here, so this is cheap and safe to re-run on its own
    (e.g. via --replot) to iterate on plot styling.
    """
    case_label = data['case_label']
    n_placements = int(data['n_placements'])

    def mark_undulator_bounds(ax):
        for s_start, s_end in data['undulator_s_ranges']:
            ax.axvline(s_start, color='0.4', linestyle='--', linewidth=1)
            ax.axvline(s_end, color='0.4', linestyle='--', linewidth=1)

    n_field_rows = 3 if model_label == 'SB' else 2
    fig_field_traj, field_axes = plt.subplots(
        n_field_rows, 1, figsize=(10, 3.2 * n_field_rows), sharex=True)
    ax_bx, ax_by = field_axes[0], field_axes[1]
    for i in range(n_placements):
        label_on = 'On-axis' if i == 0 else None
        label_off = 'Off-axis (shift_x=0.5 mm)' if i == 0 else None
        ax_bx.plot(data['field_s_on_pp'][i], data['field_bx_on_pp'][i],
                   color='tab:blue', label=label_on)
        ax_bx.plot(data['field_s_off_pp'][i], data['field_bx_off_pp'][i],
                   color='tab:orange', label=label_off)
        ax_by.plot(data['field_s_on_pp'][i], data['field_by_on_pp'][i],
                   color='tab:blue', label=label_on)
        ax_by.plot(data['field_s_off_pp'][i], data['field_by_off_pp'][i],
                   color='tab:orange', label=label_off)
    mark_undulator_bounds(ax_bx)
    mark_undulator_bounds(ax_by)
    ax_bx.set_ylabel(r'$B_x$ [T]')
    ax_bx.set_title(f'Field along the tracked trajectory ({model_label} model, local optics)')
    ax_bx.grid(True, alpha=0.3)
    ax_bx.legend()
    ax_by.grid(True, alpha=0.3)
    ax_by.legend()

    if model_label == 'SB':
        ax_bs = field_axes[2]
        for i in range(n_placements):
            label_on = 'On-axis' if i == 0 else None
            label_off = 'Off-axis (shift_x=0.5 mm)' if i == 0 else None
            ax_bs.plot(data['field_s_on_pp'][i], data['field_bs_on_pp'][i],
                       color='tab:blue', label=label_on)
            ax_bs.plot(data['field_s_off_pp'][i], data['field_bs_off_pp'][i],
                       color='tab:orange', label=label_off)
        mark_undulator_bounds(ax_bs)
        ax_bs.set_ylabel(r'$B_s$ [T]')
        ax_bs.grid(True, alpha=0.3)
        ax_bs.legend()

    field_axes[-1].set_xlabel('s [m]')
    fig_field_traj.suptitle(case_label)
    fig_field_traj.tight_layout()

    # orbit_comparison -- MK-model version dropped (redundant with the SB
    # one for reviewing this case).
    fig_orbit = None
    if model_label != 'MK':
        fig_orbit, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for i in range(n_placements):
            lbl = (i == 0)
            ax_x.plot(data['tw_onaxis_s_pp'][i], data['tw_no_undulator_local_x_pp'][i],
                      color='tab:blue', label='No undulator' if lbl else None)
            ax_x.plot(data['tw_onaxis_s_pp'][i], data['tw_onaxis_x_pp'][i],
                      color='tab:orange', label='On-axis undulator' if lbl else None)
            ax_x.plot(data['tw_offaxis_s_pp'][i], data['tw_offaxis_x_pp'][i],
                      color='tab:green', label='Off-axis undulator (shift_x=0.5 mm)' if lbl else None)
            ax_y.plot(data['tw_onaxis_s_pp'][i], data['tw_no_undulator_local_y_pp'][i],
                      color='tab:blue', label='No undulator' if lbl else None)
            ax_y.plot(data['tw_onaxis_s_pp'][i], data['tw_onaxis_y_pp'][i],
                      color='tab:orange', label='On-axis undulator' if lbl else None)
            ax_y.plot(data['tw_offaxis_s_pp'][i], data['tw_offaxis_y_pp'][i],
                      color='tab:green', label='Off-axis undulator (shift_x=0.5 mm)' if lbl else None)
        mark_undulator_bounds(ax_x)
        mark_undulator_bounds(ax_y)
        ax_x.set_ylabel('x [m]')
        ax_x.set_title('Horizontal orbit through the undulator(s) (local optics)')
        ax_x.grid(True, alpha=0.3)
        ax_x.legend()
        ax_y.set_xlabel('s [m]')
        ax_y.set_ylabel('y [m]')
        ax_y.set_title('Vertical orbit through the undulator(s) (local optics)')
        ax_y.grid(True, alpha=0.3)
        ax_y.legend()
        fig_orbit.suptitle(case_label)
        fig_orbit.tight_layout()

    hor_off_list = data['hor_off_list']
    deltaqx_list = data['deltaqx_list']
    deltaqy_list = data['deltaqy_list']
    orbit_scan_s_pp = data['orbit_scan_s_pp']
    orbit_scan_x_pp = data['orbit_scan_x_pp']
    orbit_scan_y_pp = data['orbit_scan_y_pp']
    betx_scan_pp = data['betx_scan_pp']
    bety_scan_pp = data['bety_scan_pp']
    betx2_scan_pp = data['betx2_scan_pp']
    bety1_scan_pp = data['bety1_scan_pp']
    betx_no_und_i_pp = data['betx_no_und_i_pp']
    bety_no_und_i_pp = data['bety_no_und_i_pp']
    betx2_no_und_i_pp = data['betx2_no_und_i_pp']
    bety1_no_und_i_pp = data['bety1_no_und_i_pp']
    deltaqx_formula_list = data['deltaqx_formula_list']
    deltaqy_formula_list = data['deltaqy_formula_list']
    deltaqx_formula_pert_list = data['deltaqx_formula_pert_list']
    deltaqy_formula_pert_list = data['deltaqy_formula_pert_list']

    norm = plt.Normalize(vmin=hor_off_list.min(), vmax=hor_off_list.max())
    cmap = plt.cm.viridis

    # orbit_scan -- MK-model version dropped (redundant with the SB one).
    fig_orbit_scan = None
    if model_label != 'MK':
        fig_orbit_scan, (ax_x_scan, ax_y_scan) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for i in range(n_placements):
            for i_dx, dx in enumerate(hor_off_list):
                color = cmap(norm(dx))
                ax_x_scan.plot(orbit_scan_s_pp[i], orbit_scan_x_pp[i, i_dx], color=color)
                ax_y_scan.plot(orbit_scan_s_pp[i], orbit_scan_y_pp[i, i_dx], color=color)
        mark_undulator_bounds(ax_x_scan)
        mark_undulator_bounds(ax_y_scan)
        ax_x_scan.set_ylabel('x [m]')
        ax_x_scan.set_title('Horizontal orbit across the tune scan (local optics)')
        ax_x_scan.grid(True, alpha=0.3)
        ax_y_scan.set_xlabel('s [m]')
        ax_y_scan.set_ylabel('y [m]')
        ax_y_scan.set_title('Vertical orbit across the tune scan (local optics)')
        ax_y_scan.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig_orbit_scan.colorbar(sm, ax=[ax_x_scan, ax_y_scan], label='Horizontal offset [m]')
        fig_orbit_scan.suptitle(case_label)

    # Beta functions -- SB-model version dropped (redundant with the
    # relative beta_beat plot kept below).
    fig_beta_scan = None
    if model_label != 'SB':
        fig_beta_scan, (ax_betx_scan, ax_bety_scan) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for i in range(n_placements):
            for i_dx, dx in enumerate(hor_off_list):
                color = cmap(norm(dx))
                ax_betx_scan.plot(orbit_scan_s_pp[i], betx_scan_pp[i, i_dx], color=color)
                ax_bety_scan.plot(orbit_scan_s_pp[i], bety_scan_pp[i, i_dx], color=color)
            ax_betx_scan.plot(orbit_scan_s_pp[i], betx_no_und_i_pp[i], color='k', linestyle='--',
                               linewidth=1, label='No undulator' if i == 0 else None)
            ax_bety_scan.plot(orbit_scan_s_pp[i], bety_no_und_i_pp[i], color='k', linestyle='--',
                               linewidth=1, label='No undulator' if i == 0 else None)
        mark_undulator_bounds(ax_betx_scan)
        mark_undulator_bounds(ax_bety_scan)
        ax_betx_scan.set_ylabel(r'$\beta_x$ [m]')
        ax_betx_scan.set_title('Beta functions across the tune scan (local optics)')
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

    # Beta beat, relative to the bare-ring optics fed in as the initial
    # condition -- same colour-coded-by-offset layout as the orbit scan.
    fig_beta_diff, (ax_dbetx, ax_dbety) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i in range(n_placements):
        for i_dx, dx in enumerate(hor_off_list):
            color = cmap(norm(dx))
            ax_dbetx.plot(orbit_scan_s_pp[i],
                          (betx_scan_pp[i, i_dx] - betx_no_und_i_pp[i]) / betx_no_und_i_pp[i],
                          color=color)
            ax_dbety.plot(orbit_scan_s_pp[i],
                          (bety_scan_pp[i, i_dx] - bety_no_und_i_pp[i]) / bety_no_und_i_pp[i],
                          color=color)
    mark_undulator_bounds(ax_dbetx)
    mark_undulator_bounds(ax_dbety)
    ax_dbetx.set_ylabel(r'$\Delta\beta_x/\beta_{x,0}$')
    ax_dbetx.set_title('Relative beta beat across the tune scan (local optics, vs. bare ring)')
    ax_dbetx.grid(True, alpha=0.3)
    ax_dbety.set_xlabel('s [m]')
    ax_dbety.set_ylabel(r'$\Delta\beta_y/\beta_{y,0}$')
    ax_dbety.grid(True, alpha=0.3)

    sm_beta = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm_beta.set_array([])
    fig_beta_diff.colorbar(sm_beta, ax=[ax_dbetx, ax_dbety], label='Horizontal offset [m]')
    fig_beta_diff.suptitle(case_label)

    # Same colour-coded-by-offset layout, but for the coupled beta functions
    # betx2/bety1 (Edwards-Teng). Normalized by the *primary* beta (betx/
    # bety, no undulator) rather than by betx2/bety1 itself -- see 007a for
    # the rationale (betx2/bety1 is near zero along most of the span and
    # would blow up the ratio wherever it dips towards zero).
    fig_beta_diff_coupled, (ax_dbetx2, ax_dbety1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i in range(n_placements):
        for i_dx, dx in enumerate(hor_off_list):
            color = cmap(norm(dx))
            ax_dbetx2.plot(orbit_scan_s_pp[i],
                           (betx2_scan_pp[i, i_dx] - betx2_no_und_i_pp[i]) / betx_no_und_i_pp[i],
                           color=color)
            ax_dbety1.plot(orbit_scan_s_pp[i],
                           (bety1_scan_pp[i, i_dx] - bety1_no_und_i_pp[i]) / bety_no_und_i_pp[i],
                           color=color)
    mark_undulator_bounds(ax_dbetx2)
    mark_undulator_bounds(ax_dbety1)
    ax_dbetx2.set_ylabel(r'$\Delta\beta_{x2}/\beta_{x,0}$')
    ax_dbetx2.set_title('Coupled beta beat across the tune scan (local optics, vs. bare ring)')
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

    # Coefficient boxes live outside the axes, stacked down the right margin
    # underneath the legend (see stack_boxes_under_legend) -- collected
    # per-axes here and only placed once the figure has been laid out.
    ax1_boxes = []
    ax2_boxes = []

    fig_tune_shift, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    ax1.plot(hor_off_list, deltaqx_list, marker='o', color='tab:blue', label='Twiss (phase advance)')
    ax1.plot(hor_off_list, poly_qx(hor_off_list), linestyle='--', color='k', label='Quadratic fit')
    ax1.plot(hor_off_list, deltaqx_formula_list, marker='^', linestyle='none',
              color='tab:green', label=r'$\frac{1}{4\pi}\oint\delta K_1\,\beta_{x,0}\,ds$')
    ax1.plot(hor_off_list, deltaqx_formula_pert_list, marker='v', linestyle='none',
              color='tab:red', label=r'$\frac{1}{4\pi}\oint\delta K_1\,\beta_x\,ds$')
    ax1.set_ylabel(r'$\Delta Q_x$')
    ax1.set_title('Tune shift vs undulator horizontal offset (local optics)')
    ax1.grid(True, alpha=0.3)
    legend_outside(ax1)
    ax1_boxes.append((fit_box_text('x', coef_qx), text_box_kwargs))

    ax2.plot(hor_off_list, deltaqy_list, marker='s', color='tab:orange', label='Twiss (phase advance)')
    ax2.plot(hor_off_list, poly_qy(hor_off_list), linestyle='--', color='k', label='Quadratic fit')
    ax2.plot(hor_off_list, deltaqy_formula_list, marker='^', linestyle='none',
              color='tab:green', label=r'$\frac{1}{4\pi}\oint(-\delta K_1)\beta_{y,0}\,ds$')
    ax2.plot(hor_off_list, deltaqy_formula_pert_list, marker='v', linestyle='none',
              color='tab:red', label=r'$\frac{1}{4\pi}\oint(-\delta K_1)\beta_y\,ds$')
    ax2.set_xlabel('Horizontal offset [m]')
    ax2.set_ylabel(r'$\Delta Q_y$')
    ax2.grid(True, alpha=0.3)
    legend_outside(ax2)
    ax2_boxes.append((fit_box_text('y', coef_qy), text_box_kwargs))

    measured = load_measured_tune_shift()
    fig_tune_shift_measured = None
    if measured is not None:
        meas_x = np.asarray(measured['bump_amplitude_m'], dtype=float)
        meas_dqx = np.asarray(measured['dtune_x_measured'], dtype=float)
        meas_dqy = np.asarray(measured['dtune_y_measured'], dtype=float)

        # Quadratic fit through the digitized measurement, same Taylor-
        # coefficient convention as coef_qx/coef_qy above.
        coef_qx_measured = np.polyfit(meas_x, meas_dqx, 2)
        coef_qy_measured = np.polyfit(meas_x, meas_dqy, 2)
        poly_qx_measured = np.poly1d(coef_qx_measured)
        poly_qy_measured = np.poly1d(coef_qy_measured)

        print(f"[measured] (1/2) d²(ΔQx)/dx² = {coef_qx_measured[0]}")
        print(f"[measured] d(ΔQx)/dx         = {coef_qx_measured[1]}")
        print(f"[measured] ΔQx(0)            = {coef_qx_measured[2]}")
        print(f"[measured] (1/2) d²(ΔQy)/dx² = {coef_qy_measured[0]}")
        print(f"[measured] d(ΔQy)/dx         = {coef_qy_measured[1]}")
        print(f"[measured] ΔQy(0)            = {coef_qy_measured[2]}")

        # --- overlay measured points + their quadratic fit on the main plot ---
        ax1.plot(meas_x, meas_dqx,
                  marker='d', linestyle='none', markersize=4,
                  color='tab:purple', label='Measurements')
        ax1.plot(hor_off_list, poly_qx_measured(hor_off_list),
                  linestyle='--', color='tab:red', label='Quadratic fit (measured)')
        ax1_boxes.append((fit_box_text('x', coef_qx_measured, 'measured fit:\n'),
                          text_box_kwargs))
        legend_outside(ax1)

        ax2.plot(meas_x, meas_dqy,
                  marker='d', linestyle='none', markersize=4,
                  color='tab:purple', label='Measurements')
        ax2.plot(hor_off_list, poly_qy_measured(hor_off_list),
                  linestyle='--', color='tab:red', label='Quadratic fit (measured)')
        ax2_boxes.append((fit_box_text('y', coef_qy_measured, 'measured fit:\n'),
                          text_box_kwargs))
        legend_outside(ax2)

    fig_tune_shift.suptitle(case_label)
    fig_tune_shift.tight_layout()
    fig_tune_shift.subplots_adjust(right=0.62)
    stack_boxes_under_legend(ax1, ax1_boxes)
    stack_boxes_under_legend(ax2, ax2_boxes)

    # --- standalone figure: measured tune data only + quadratic fit, on the
    #     same vertical axis as the main tune-shift plot above ---
    if measured is not None:
        ylim_qx = ax1.get_ylim()
        ylim_qy = ax2.get_ylim()

        meas_x_dense = np.linspace(meas_x.min(), meas_x.max(), 200)

        fig_tune_shift_measured, (ax1_m, ax2_m) = plt.subplots(
            2, 1, figsize=(9, 9), sharex=True)

        ax1_m.plot(meas_x, meas_dqx, marker='d', linestyle='none', markersize=5,
                    color='tab:purple', label='Measurements')
        ax1_m.plot(meas_x_dense, poly_qx_measured(meas_x_dense),
                    linestyle='--', color='tab:red', label='Quadratic fit')
        ax1_m.set_ylabel(r'$\Delta Q_x$')
        ax1_m.set_ylim(ylim_qx)
        ax1_m.set_title('Measured tune shift vs undulator horizontal offset')
        ax1_m.grid(True, alpha=0.3)
        legend_outside(ax1_m)

        ax2_m.plot(meas_x, meas_dqy, marker='d', linestyle='none', markersize=5,
                    color='tab:purple', label='Measurements')
        ax2_m.plot(meas_x_dense, poly_qy_measured(meas_x_dense),
                    linestyle='--', color='tab:red', label='Quadratic fit')
        ax2_m.set_xlabel('Horizontal offset [m]')
        ax2_m.set_ylabel(r'$\Delta Q_y$')
        ax2_m.set_ylim(ylim_qy)
        ax2_m.grid(True, alpha=0.3)
        legend_outside(ax2_m)

        fig_tune_shift_measured.suptitle(case_label)
        fig_tune_shift_measured.tight_layout()
        fig_tune_shift_measured.subplots_adjust(right=0.62)
        stack_boxes_under_legend(
            ax1_m, [(fit_box_text('x', coef_qx_measured), text_box_kwargs)])
        stack_boxes_under_legend(
            ax2_m, [(fit_box_text('y', coef_qy_measured), text_box_kwargs)])

    # --- measured closed-gap Q_x,y (closed-minus-open shift with the open
    #     gap added back in) minus the bare-ring Twiss tune, next to the
    #     usual simulated (phase-advance) shift -- same as 007a. ---
    fig_tune_shift_closed_gap = None
    if measured is not None and 'qx_0' not in data:
        print("[tune_shift_closed_gap_vs_model] cached data has no "
              "qx_0/qy_0 (computed before this plot existed) -- rerun "
              "without --replot to produce it")
    elif measured is not None:
        q_0 = {'x': float(data['qx_0']), 'y': float(data['qy_0'])}
        dq_meas_closed_vs_model = {
            'x': meas_dqx + np.asarray(measured['tune_x_open_gap'], dtype=float) - q_0['x'],
            'y': meas_dqy + np.asarray(measured['tune_y_open_gap'], dtype=float) - q_0['y'],
        }
        fig_tune_shift_closed_gap = tune_shift_comparison_figure(
            [(hor_off_list, {'x': deltaqx_list, 'y': deltaqy_list},
              'Twiss (phase advance, with $-$ without undulator)',
              'tab:blue', 'o', '')],
            meas_x, dq_meas_closed_vs_model,
            'Measured closed-gap $Q_{p}$ $-$ Twiss $Q_{p}$ without undulator',
            'Tune shift vs undulator horizontal offset '
            '(relative to Twiss tune without undulator, local optics)',
            case_label)

    # Tracked (phase-advance)-vs-calculated tune shift residual -- MK-model
    # version dropped (the formula is built from MK's own field model, so
    # this residual is near-trivially small there and not informative).
    fig_tune_shift_diff = None
    if model_label != 'MK':
        deltaqx_diff_beta0 = deltaqx_list - deltaqx_formula_list
        deltaqx_diff_beta = deltaqx_list - deltaqx_formula_pert_list
        deltaqy_diff_beta0 = deltaqy_list - deltaqy_formula_list
        deltaqy_diff_beta = deltaqy_list - deltaqy_formula_pert_list

        fig_tune_shift_diff, (ax1_diff, ax2_diff) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax1_diff.plot(hor_off_list, deltaqx_diff_beta0, marker='^', color='tab:green',
                      label=r'Twiss $-\ \frac{1}{4\pi}\oint\delta K_1\,\beta_{x,0}\,ds$')
        ax1_diff.plot(hor_off_list, deltaqx_diff_beta, marker='v', color='tab:red',
                      label=r'Twiss $-\ \frac{1}{4\pi}\oint\delta K_1\,\beta_x\,ds$')
        ax1_diff.axhline(0, color='0.4', linewidth=1)
        ax1_diff.set_ylabel(r'$\Delta Q_x$ residual')
        ax1_diff.set_title('Tracked (phase-advance) minus calculated tune shift (local optics)')
        ax1_diff.grid(True, alpha=0.3)
        ax1_diff.legend()

        ax2_diff.plot(hor_off_list, deltaqy_diff_beta0, marker='^', color='tab:green',
                      label=r'Twiss $-\ \frac{1}{4\pi}\oint(-\delta K_1)\beta_{y,0}\,ds$')
        ax2_diff.plot(hor_off_list, deltaqy_diff_beta, marker='v', color='tab:red',
                      label=r'Twiss $-\ \frac{1}{4\pi}\oint(-\delta K_1)\beta_y\,ds$')
        ax2_diff.axhline(0, color='0.4', linewidth=1)
        ax2_diff.set_xlabel('Horizontal offset [m]')
        ax2_diff.set_ylabel(r'$\Delta Q_y$ residual')
        ax2_diff.grid(True, alpha=0.3)
        ax2_diff.legend()

        fig_tune_shift_diff.suptitle(case_label)
        fig_tune_shift_diff.tight_layout()

    # Save whichever figures were built for this case. Some entries are None
    # (skipped for this model_label -- see above) and are filtered out here
    # rather than saved.
    figures = [
        (fig_field_traj, 'field_along_trajectory'),
        (fig_orbit, 'orbit_comparison'),
        (fig_orbit_scan, 'orbit_scan'),
        (fig_beta_scan, 'beta_functions'),
        (fig_beta_diff, 'beta_beat'),
        (fig_beta_diff_coupled, 'beta_beat_coupled'),
        (fig_tune_shift, 'tune_shift'),
        (fig_tune_shift_measured, 'tune_shift_measured'),
        (fig_tune_shift_closed_gap, 'tune_shift_closed_gap_vs_model'),
        (fig_tune_shift_diff, 'tune_shift_difference'),
        ]
    for fig, suffix in figures:
        if fig is None:
            continue
        out_path = OUT_DIR / f'{place_label}_{model_label}_{suffix}_local_optics.pdf'
        # bbox_inches='tight' so the out-of-axes legends aren't clipped.
        fig.savefig(out_path, bbox_inches='tight')
        print(f"Saved {out_path}")


def plot_measured_open_gap_drift():
    """Measurement-only figure: the open-gap tune drift with bump amplitude
    next to the undulator signal (closed minus open gap), both relative to
    their own mean -- identical to 007a's, see there for the rationale.
    """
    measured = load_measured_tune_shift()
    if measured is None or 'tune_x_open_gap' not in measured:
        return None
    meas_x_um = np.asarray(measured['bump_amplitude_um'], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, plane in zip(axes, ('x', 'y')):
        q_open = np.asarray(measured[f'tune_{plane}_open_gap'], dtype=float)
        dq = np.asarray(measured[f'dtune_{plane}_measured'], dtype=float)
        ax.plot(meas_x_um, q_open - q_open.mean(), marker='o', markersize=4,
                color='tab:blue',
                label=f'Open gap: $Q_{plane} - \\langle Q_{plane}\\rangle$')
        ax.plot(meas_x_um, dq - dq.mean(), marker='d', markersize=4,
                color='tab:purple',
                label=f'Closed $-$ open: $\\Delta Q_{plane} - '
                      f'\\langle\\Delta Q_{plane}\\rangle$')
        ax.set_ylabel(f'Tune {plane} variation')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title('X11MA, gap = 11.5 mm: open-gap drift vs undulator signal')
    axes[1].set_xlabel(r'Bump amplitude x [$\mu$m]')
    fig.tight_layout()

    out_path = OUT_DIR / 'x11ma_open_gap_drift.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved {out_path}")
    return fig


MODEL_STYLES = {'SB': ('SplineBoris', 'tab:blue', 'o'),
                'MK': ('Multipole kick', 'tab:green', 's')}


def plot_models_combined(place_label, case_datas, reference,
                         zero_at_origin=False):
    """Simulated tune shift for both undulator models (case_datas:
    {model_label: data}) in one figure, next to the measured
    closed-minus-open-gap shift -- same as 007a, with the simulated shift
    taken from the local-optics phase advance. reference sets what the
    simulated Q_x,y with undulator (bare-ring qx_0/qy_0 + phase-advance
    shift) is taken relative to:
      'open_gap': the *measured* open-gap Q_x,y at the same bump amplitude;
      'model':    the bare-ring Twiss Q_x,y (as in the main tune_shift
                  figure).
    With zero_at_origin, every curve is shifted by its own (linearly
    interpolated) value at x = 0, so only the progression with offset is
    compared, not the absolute level.
    """
    measured = load_measured_tune_shift()
    if measured is None:
        return None
    if (reference == 'open_gap'
            and any('qx_0' not in data for data in case_datas.values())):
        print(f"[{place_label}_tune_shift_vs_open_gap] cached data has no "
              "qx_0/qy_0 (computed before this plot existed) -- rerun "
              "without --replot to produce it")
        return None
    meas_x = np.asarray(measured['bump_amplitude_m'], dtype=float)
    meas_q_open = {plane: np.asarray(measured[f'tune_{plane}_open_gap'], dtype=float)
                   for plane in ('x', 'y')}
    meas_dq = {plane: np.asarray(measured[f'dtune_{plane}_measured'], dtype=float)
               for plane in ('x', 'y')}

    def shift_to_origin(x, dq):
        if not zero_at_origin:
            return dq
        return {plane: dq[plane] - np.interp(0., x, dq[plane])
                for plane in ('x', 'y')}

    sim_series = []
    for model_label, data in case_datas.items():
        hor_off_list = data['hor_off_list']
        # Simulated tunes interpolated onto the measured bump amplitudes
        # (exact when the scan was run at them, see hor_off_list in
        # compute_case()); points outside the simulated range are dropped.
        in_range = ((meas_x >= hor_off_list.min())
                    & (meas_x <= hor_off_list.max()))
        x_sim = meas_x[in_range]
        dq_twiss = {'x': np.interp(x_sim, hor_off_list, data['deltaqx_list']),
                    'y': np.interp(x_sim, hor_off_list, data['deltaqy_list'])}
        model_name, color, marker = MODEL_STYLES[model_label]
        if reference == 'open_gap':
            q_0 = {'x': float(data['qx_0']), 'y': float(data['qy_0'])}
            dq_sim = {plane: (dq_twiss[plane] + q_0[plane]
                              - meas_q_open[plane][in_range])
                      for plane in ('x', 'y')}
            label = (f'{model_name}: Twiss $Q_{{p}}$ $-$ '
                     f'measured open-gap $Q_{{p}}$')
        else:
            dq_sim = dq_twiss
            label = f'{model_name}: Twiss (with $-$ without undulator)'
        sim_series.append((
            x_sim, shift_to_origin(x_sim, dq_sim), label,
            color, marker, f'{model_name} fit:\n'))

    if reference == 'open_gap':
        title = ('Tune shift vs undulator horizontal offset '
                 '(relative to measured open-gap tune, local optics)')
        suffix = 'tune_shift_vs_open_gap'
    else:
        title = ('Tune shift vs undulator horizontal offset '
                 '(relative to Twiss tune without undulator, local optics)')
        suffix = 'tune_shift_models'
    if zero_at_origin:
        title += ',\nshifted to $\\Delta Q(0) = 0$'
        suffix += '_zeroed'
    fig = tune_shift_comparison_figure(
        sim_series, meas_x, shift_to_origin(meas_x, meas_dq),
        'Measurements (closed $-$ open gap)', title, place_label)
    out_path = OUT_DIR / f'{place_label}_{suffix}_local_optics.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved {out_path}")
    return fig


plot_measured_open_gap_drift()

for place_label, wiggler_places in WIGGLER_CASES:
    case_datas = {}
    for model_label in MODEL_LABELS:
        case_datas[model_label] = get_case_data(
            place_label, wiggler_places, model_label)
        plot_case(case_datas[model_label], place_label, model_label)
    plot_models_combined(place_label, case_datas, 'open_gap')
    plot_models_combined(place_label, case_datas, 'open_gap',
                         zero_at_origin=True)
    plot_models_combined(place_label, case_datas, 'model',
                         zero_at_origin=True)

# All figures (per placement: the per-model ones from plot_case() plus the
# SB/MK-combined ones from plot_models_combined()) are kept open so they can
# all be reviewed interactively here, in addition to having been saved as
# PDFs above.
plt.show()
