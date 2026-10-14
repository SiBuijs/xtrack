import os
import sys
import xtrack as xt
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as sc_const
from scipy.integrate import cumulative_trapezoid
from xtrack._temp.splineboris import TubeFitter, LongitudinalFitter
from xtrack import linear_normal_form as lnf

# Placement cases to simulate/plot (keys of WIGGLER_CASES below).
CASES = ['ars11_uind_0610_1']#, 'ars11_uind_0210_1']#, 'both']


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
            #residual_tol=1e-3,
            distance_unit=distance_unit,
            deg=multipole_order - 1,
            #tube_radius=0.0005,
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

# for der in range(0, multipole_order):
#     get_undulator_fit().plot_fields(der=der)

# plt.show()

# Default ~/cernbox/Pictures/SLS_Undulator_Studies (works on any machine/user),
# overridable via the SLS_UNDULATOR_PLOT_DIR env var -- same as 012.
OUT_DIR = Path(os.environ.get(
    'SLS_UNDULATOR_PLOT_DIR',
    Path.home() / 'cernbox' / 'Pictures' / 'SLS_Undulator_Studies'))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cached per-case scan/formula results, so that re-running just to tweak a
# plot doesn't require redoing the fit/match/scan pipeline (which needs the
# ~1.8 GB field map above). Pass --replot on the command line to load from
# here instead of recomputing, for any case whose data is already cached.
# Separate from 007a's DATA_DIR (different schema); 007a's cache is only
# read, to overlay its full-ring Twiss tune shift for comparison.
DATA_DIR = Path(__file__).resolve().parent / 'data_007b'
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR_007A = Path(__file__).resolve().parent / 'data'
REPLOT = '--replot' in sys.argv

# Same as 007a, except that the undulator is never inserted into the ring:
# the standalone undulator line is Twissed from the bare-ring optics at its
# start, and the tune shift is read off the phase advance it adds (see
# compute_case()).
#
# Three possible placement cases (only ars11_uind_0210_1, only
# ars11_uind_0610_1, both -- the ones run are selected by CASES at the top)
# x two undulator models (SB = SplineBoris, from to_line(); MK = Multipole
# Kick, from to_multipole_line()) -- compute_case() below builds and
# analyses one such (placement, model) case end to end.
WIGGLER_CASES = {
    'ars11_uind_0210_1': ['ars11_uind_0210_1'],
    'ars11_uind_0610_1': ['ars11_uind_0610_1'],
    'both': ['ars11_uind_0210_1', 'ars11_uind_0610_1'],
}
MODEL_LABELS = ('SB', 'MK')

# Digitized reference measurement (blue "Measurements" diamonds from a plot
# titled "X11MA, gap = 11.5 mm", pixel-extracted from test_data/sls/image004.png
# -- see examples/splineboris/claude_notes/ for the digitization method).
# NOTE: the correspondence between this "X11MA" beamline measurement and the
# ars11_uind_0210_1 / ars11_uind_0610_1 elements simulated below has NOT been
# confirmed -- overlaid here purely as a shape/order-of-magnitude reference,
# plotted directly on the same Delta Qx/Qy scale as the simulated curves.
# The dtune_*_measured columns are closed-gap minus open-gap tunes, subtracted
# point by point (verified against the raw open/closed-gap tune scans, also
# digitized into this CSV as tune_{x,y}_{open,closed}_gap, absolute tunes
# with the plot's offsets Qx = 39.37, Qy = 15.22 added back).
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
    case_label = f'{place_label} ({model_label}, local optics)'
    print("=" * 80)

    undulator_fit = get_undulator_fit()

    # Load SLS MADX file
    env = xt.load(str(madx_file))
    line_sls = env.ring

    # Configure bend model
    line_sls.configure_bend_model(core='mat-kick-mat')

    # Set particle reference
    line_sls.particle_ref = p0.copy()

    # Build undulator -- SB uses LongitudinalFitter.to_line() (one SplineBoris
    # element per polynomial piece, full spatial field integration); MK uses
    # LongitudinalFitter.to_multipole_line() (one thick Multipole per region, a
    # coarser rigidity-normalized approximation). Either way it lives in its
    # own implicit Environment, so import it into its own `und_env` to add
    # correctors and match it (same build path as
    # 004b_undulators_in_sls_ring.py / 004a_build_undulator.py) before
    # placing it into the SLS ring.
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

    # Bare-ring Twiss (the undulator is never inserted, unlike 007a) -- the
    # tune baseline, and the source of the initial optics at each placement.
    tw_no_undulator = line_sls.twiss4d(include_collective=True)
    qx_0 = tw_no_undulator.qx
    qy_0 = tw_no_undulator.qy

    # The standalone `undulator` line is Twissed from the bare-ring optics
    # (betx, alfx, bety, alfy) at the start of each placement. The span it
    # would replace in the ring (ars11_uind_*_1 .. _2 and the start of the
    # next drift) only holds zero-strength placeholders, i.e. it is a drift
    # of length L = l_wig.
    #
    # Tune shift (main result): the one-turn matrix at the first undulator
    # entrance, built without inserting anything into the ring,
    #   M_turn = ... (T_next D^-1) M_und ...,
    # with M_und the undulator's transfer matrix (from its Twiss), D the
    # drift it replaces and T_next the bare-ring transfer to the next
    # placement (for the last one, round the ring back to the first:
    # M_bare,one-turn T_first->last^-1). Its eigen-tunes are exact, i.e.
    # identical to 007a's closed-ring Twiss.
    #
    # Diagnostic: the phase advance across the undulator itself,
    #   Delta mu_local = mu_und - mu_drift,
    #   2 pi mu_drift = arctan2(L, beta0 - alpha0 L)   (mux/muy in turns),
    # is NOT the tune shift: with the entrance optics frozen it only counts
    # (1/4 pi) int K beta (1 - cos 2 psi) ds, psi the phase advance to the
    # undulator exit, about 1/8 of the shift here. The rest is accumulated in
    # the rest of the ring. See claude_notes/local_phase_advance_vs_tune_shift.md.
    init_conditions = []
    undulator_s_ranges = []
    for wig_place in wiggler_places:
        s0 = float(tw_no_undulator['s', wig_place])
        ic = {kk: float(tw_no_undulator[kk, wig_place])
              for kk in ('betx', 'alfx', 'bety', 'alfy')}
        init_conditions.append(ic)
        undulator_s_ranges.append((s0, s0 + l_wig))
        print(f"Undulator at {wig_place} (s = {s0:.4f} m): "
              + ", ".join(f"{kk} = {vv:.4f}" for kk, vv in ic.items()))

    def drift_phase_advance(bet, alf):
        return np.arctan2(l_wig, bet - alf * l_wig) / (2 * np.pi)

    mux_drift = [drift_phase_advance(ic['betx'], ic['alfx']) for ic in init_conditions]
    muy_drift = [drift_phase_advance(ic['bety'], ic['alfy']) for ic in init_conditions]

    def twiss_undulator(ic):
        return undulator.twiss4d(include_collective=True, **ic)

    # Bare-ring linear maps (4D: x, px, y, py) for the one-turn matrix.
    # Placements are processed in order of s (they don't overlap).
    order = np.argsort([s0 for s0, _ in undulator_s_ranges])
    place_rows = [int(np.where(tw_no_undulator.name == wiggler_places[i])[0][0])
                  for i in order]
    W_first = tw_no_undulator.W_matrix[place_rows[0]][:4, :4]
    rot = np.zeros((4, 4))
    rot[:2, :2] = lnf.Rot2D(2 * np.pi * qx_0)
    rot[2:, 2:] = lnf.Rot2D(2 * np.pi * qy_0)
    M_bare_turn = W_first @ rot @ np.linalg.inv(W_first)
    D_inv = np.eye(4)
    D_inv[0, 1] = D_inv[2, 3] = -l_wig
    # bare transfer from each placement's start to the next one's start
    # (round the ring back to the first for the last placement)
    T_next = []
    for k, row in enumerate(place_rows):
        if k + 1 < len(place_rows):
            T_next.append(tw_no_undulator.get_R_matrix(row, place_rows[k + 1])[:4, :4])
        else:
            T_first_to_last = tw_no_undulator.get_R_matrix(place_rows[0], row)[:4, :4]
            T_next.append(M_bare_turn @ np.linalg.inv(T_first_to_last))
    qx_frac0, qy_frac0 = qx_0 % 1, qy_0 % 1

    def one_turn_tune_shift(tws_sorted):
        M = np.eye(4)
        for tw_p, T in zip(tws_sorted, T_next):
            M_und = tw_p.get_R_matrix(0, len(tw_p.name) - 1)[:4, :4]
            M = T @ D_inv @ M_und @ M
        # eigen-tunes, assigned to the plane whose bare tune they're closest
        # to (fine for shifts << |Qx - Qy| mod 1)
        q = np.abs(np.angle(np.linalg.eigvals(M))) / (2 * np.pi)
        return (q[np.argmin(np.abs(q - qx_frac0))] - qx_frac0,
                q[np.argmin(np.abs(q - qy_frac0))] - qy_frac0)

    # Field-model element names in the standalone `undulator` line -- name-
    # based (not isinstance-based) because for the MK model both the field
    # elements and the correctors are xt.Multipole. to_line()/
    # to_multipole_line() name every element "splineboris_..."/"multipole_...".
    undulator_field_element_names = [
        nn for nn in undulator.element_names if nn.startswith(('splineboris_', 'multipole_'))
        ]

    deltaqx_list = []         # one-turn matrix (exact)
    deltaqy_list = []
    deltaqx_local_list = []   # phase advance across the undulator (diagnostic)
    deltaqy_local_list = []
    orbit_scan_x = []    # per offset: (n_placements, n_pts), local s grid
    orbit_scan_y = []
    orbit_scan_s = None  # local s grid of the undulator Twiss (same for all)
    betx_scan = []
    bety_scan = []

    # Scan the offsets at the measured bump amplitudes when the measurement
    # is available, so the simulated tunes can be compared point by point
    # with the measured tunes.
    measured = load_measured_tune_shift()
    if measured is not None:
        hor_off_list = np.asarray(measured['bump_amplitude_m'], dtype=float)
    else:
        n_tunes = 30
        hor_off_list = np.linspace(-0.5e-3, 0.5e-3, n_tunes)

    # Correctors are re-matched for every offset, same as 007a: the fixed
    # strengths matched at shift_x=0 would otherwise leave a residual orbit
    # at the undulator exit.
    corrector_vars = ['k0l_corr1', 'k0sl_corr1', 'k0l_corr2', 'k0sl_corr2',
                       'k0l_corr3', 'k0sl_corr3', 'k0l_corr4', 'k0sl_corr4']

    # Local x-deflection baseline (same as 007a): even at shift_x=0 the beam
    # deflects horizontally inside the undulator. Literal single-particle
    # track at element-by-element resolution, with the (coarser) Twiss-orbit
    # equivalent printed alongside.
    p_deflection = p0.copy()
    undulator.track(p_deflection, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_deflection = undulator.record_last_track
    x_deflection_baseline = float(np.mean(mon_deflection.x[0, :]))
    tw_onaxis = twiss_undulator(init_conditions[0])
    x_deflection_twiss_check = float(np.mean(np.interp(
        mon_deflection.s[0, :], tw_onaxis.s, tw_onaxis.x)))
    print(f"[x-deflection baseline] literal single-particle track: "
          f"mean x = {x_deflection_baseline * 1e6:.3f} um "
          f"(Twiss-orbit-interpolated equivalent: "
          f"{x_deflection_twiss_check * 1e6:.3f} um)")

    # Linear focusing on the wiggling orbit, straight from the 3D field --
    # SB only (the MK model is 2D thick multipoles, for which the multipole
    # deltaK1 formula further below is exact). Derivation in
    # claude_notes/undulator_linear_focusing_derivation.md: from the
    # paraxial Lorentz force
    #   x'' = (y' B_s - B_y) / B rho,   y'' = (B_x - x' B_s) / B rho,
    # linearized about the closed orbit (x_w, y_w), keeping only the
    # diagonal terms (the rest only couple, shifting the tunes at 2nd order):
    #   K_x = (dB_y/dx - y_w' dB_s/dx) / B rho
    #   K_y = (x_w' dB_s/dy - dB_x/dy) / B rho      (y'' + K_y y = 0 etc.)
    # with all gradients on the orbit. Besides K1 and the feed-downs this
    # includes the undulator's natural focusing: the B_s terms (Guignard's
    # M x' term on a wiggling orbit) and the 3D d^2B_x/ds^2 part of dB_y/dx
    # at y_w != 0, which a 2D multipole expansion misses (for this
    # elliptical map it moves the <B_x^2> focusing from x to y).
    # The field is SplineBoris.get_field() -- the same Maxwell-consistent
    # expansion the tracking uses, shift_x included -- on n_sub + 1 points
    # per element. The orbit oscillates on the undulator period, which the
    # Twiss rows don't resolve, so x_w', y_w' (and from them x_w, y_w) are
    # integrated through each element from its Twiss entrance values
    # (x_w' = x_k' - int B_y ds / B rho). Gradients by central differences
    # (exact: the field is polynomial in x, y). Computed per offset inside
    # the scan below, as element-mean strengths (the SplineBoris elements
    # share their nodes with the to_multipole_line() regions).
    field_K_included = model_label == 'SB'
    brho0 = p0.rigidity0[0]
    n_sub = 20
    u_sub = np.linspace(0, 1, n_sub + 1)
    und_table = undulator.get_table()
    und_s_entry = np.array([und_table['s', nn]
                            for nn in undulator_field_element_names])
    d_fd = 1e-4   # finite-difference step for the field gradients [m]
    K_x_scan = []    # per offset: (n_placements, n_elements), full K_x
    K_y_scan = []
    Ks_x_scan = []   # same, B_s terms only (for reference)
    Ks_y_scan = []

    def field_focusing(tw, s_start_und):
        n_el = len(undulator_field_element_names)
        K_x, K_y, Ks_x, Ks_y = (np.zeros(n_el) for _ in range(4))
        for k, nn in enumerate(undulator_field_element_names):
            el = undulator[nn]
            s_loc = el.length * u_sub
            s_glob = s_start_und + und_s_entry[k] + s_loc
            # orbit slope through the element, from the field along the
            # (coarse) Twiss orbit, then the orbit itself from the slope
            Bx, By, _ = el.get_field(np.interp(s_glob, tw.s, tw.x),
                                     np.interp(s_glob, tw.s, tw.y), s_loc)
            xp = (np.interp(s_glob[0], tw.s, tw.px)
                  - cumulative_trapezoid(By, s_loc, initial=0) / brho0)
            yp = (np.interp(s_glob[0], tw.s, tw.py)
                  + cumulative_trapezoid(Bx, s_loc, initial=0) / brho0)
            x = np.interp(s_glob[0], tw.s, tw.x) + cumulative_trapezoid(xp, s_loc, initial=0)
            y = np.interp(s_glob[0], tw.s, tw.y) + cumulative_trapezoid(yp, s_loc, initial=0)

            dB_dx = (np.array(el.get_field(x + d_fd, y, s_loc))
                     - np.array(el.get_field(x - d_fd, y, s_loc))) / (2 * d_fd)
            dB_dy = (np.array(el.get_field(x, y + d_fd, s_loc))
                     - np.array(el.get_field(x, y - d_fd, s_loc))) / (2 * d_fd)
            dBy_dx, dBs_dx = dB_dx[1], dB_dx[2]
            dBx_dy, dBs_dy = dB_dy[0], dB_dy[2]

            def mean(f):
                return np.trapezoid(f, s_loc) / el.length / brho0
            Ks_x[k] = mean(-yp * dBs_dx)
            Ks_y[k] = mean(xp * dBs_dy)
            K_x[k] = mean(dBy_dx) + Ks_x[k]
            K_y[k] = mean(-dBx_dy) + Ks_y[k]
        return K_x, K_y, Ks_x, Ks_y

    for dx in hor_off_list:   # dx in meters
        # apply horizontal offset to all undulator slices
        for nn in undulator_field_element_names:
            undulator[nn].shift_x = dx

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

        # Twiss each placement from its own bare-ring initial optics; tune
        # shift from the one-turn matrix, plus the local phase advance
        tws = [twiss_undulator(ic) for ic in init_conditions]
        dqx, dqy = one_turn_tune_shift([tws[i] for i in order])
        deltaqx_list.append(dqx)
        deltaqy_list.append(dqy)
        dqx = dqy = 0.
        for tw_p, mx_d, my_d in zip(tws, mux_drift, muy_drift):
            dqx += tw_p.mux[-1] - mx_d
            dqy += tw_p.muy[-1] - my_d
        deltaqx_local_list.append(dqx)
        deltaqy_local_list.append(dqy)
        orbit_scan_x.append([tw_p.x for tw_p in tws])
        orbit_scan_y.append([tw_p.y for tw_p in tws])
        betx_scan.append([tw_p.betx for tw_p in tws])
        bety_scan.append([tw_p.bety for tw_p in tws])
        if field_K_included:
            # local Twiss: s starts at 0 at the undulator entrance
            kk = [field_focusing(tw_p, 0.) for tw_p in tws]
            K_x_scan.append([k[0] for k in kk])
            K_y_scan.append([k[1] for k in kk])
            Ks_x_scan.append([k[2] for k in kk])
            Ks_y_scan.append([k[3] for k in kk])
        if orbit_scan_s is None:
            orbit_scan_s = tws[0].s

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
    multipole_line = undulator_fit.to_multipole_line(
        multipole_order=multipole_order, p0c=E0, field_at='mean', **FIELD_TOL)
    mult_table = multipole_line.get_table()
    # to_multipole_line() also brackets each region with thin xt.MultipoleEdge
    # kicks (see longitudinal_fitter.py) -- excluded here since K1(s)/K2(s)/K2_skew(s)
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
    # bety_scan -- the local undulator Twiss beta at this same dx, from the
    # bare-ring initial optics) instead of the unperturbed tw_no_undulator beta
    # -- isolates how much of the formula/Twiss gap comes from beta-beat
    # feedback (formula is 1st-order perturbation theory and normally uses
    # the unperturbed beta) rather than from delta K(s) itself.
    deltaqx_formula_pert_list = []
    deltaqy_formula_pert_list = []
    # The B_s terms alone (SB, unperturbed beta), for reference.
    deltaqx_bs_list = []
    deltaqy_bs_list = []
    for i_dx, (dx, x_orbit, y_orbit, betx_pert, bety_pert) in enumerate(zip(
            hor_off_list, orbit_scan_x, orbit_scan_y, betx_scan, bety_scan)):
        integral_x = 0.0
        integral_y = 0.0
        integral_x_pert = 0.0
        integral_y_pert = 0.0
        integral_x_bs = 0.0
        integral_y_bs = 0.0
        for i_und, (s_start_und, _) in enumerate(undulator_s_ranges):
            # orbit_scan_s is the local (undulator-entrance = 0) s grid
            s_global = s_start_und + mult_s_mid
            x_local = (np.interp(mult_s_mid, orbit_scan_s, x_orbit[i_und]) - dx
                       + x_deflection_baseline)
            # No "- dx"-like offset for y: shift_y is never varied in the
            # scan (only shift_x is), so the magnet's y-position stays at 0
            # and y_local is just the actual orbit position.
            y_local = np.interp(mult_s_mid, orbit_scan_s, y_orbit[i_und])
            deltaK1 = mult_K1 + mult_K2 * x_local - mult_K2_skew * y_local
            # Focusing strengths in the deltaK1 convention (x: +K_x, y:
            # -K_y, the y integral gets an overall minus sign below). For SB
            # the full 3D-field K_x, K_y replace deltaK1 entirely.
            if field_K_included:
                Kx_eff = K_x_scan[i_dx][i_und]
                Ky_eff = -K_y_scan[i_dx][i_und]
                Ks_x = Ks_x_scan[i_dx][i_und]
                Ks_y = Ks_y_scan[i_dx][i_und]
            else:
                Kx_eff = Ky_eff = deltaK1
                Ks_x = Ks_y = np.zeros_like(mult_length)
            betx_at_s = np.interp(s_global, tw_no_undulator.s, tw_no_undulator.betx)
            bety_at_s = np.interp(s_global, tw_no_undulator.s, tw_no_undulator.bety)
            integral_x += np.sum(Kx_eff * betx_at_s * mult_length)
            integral_y += np.sum(Ky_eff * bety_at_s * mult_length)
            integral_x_bs += np.sum(Ks_x * betx_at_s * mult_length)
            integral_y_bs += np.sum(Ks_y * bety_at_s * mult_length)

            betx_pert_at_s = np.interp(mult_s_mid, orbit_scan_s, betx_pert[i_und])
            bety_pert_at_s = np.interp(mult_s_mid, orbit_scan_s, bety_pert[i_und])
            integral_x_pert += np.sum(Kx_eff * betx_pert_at_s * mult_length)
            integral_y_pert += np.sum(Ky_eff * bety_pert_at_s * mult_length)
        # Sign convention (deltaQx = +1/4pi * oint K1 betax ds, deltaQy =
        # -1/4pi * oint K1 betay ds) as verified in 007a.
        deltaqx_formula_list.append(integral_x / (4 * np.pi))
        deltaqy_formula_list.append(-integral_y / (4 * np.pi))
        deltaqx_formula_pert_list.append(integral_x_pert / (4 * np.pi))
        deltaqy_formula_pert_list.append(-integral_y_pert / (4 * np.pi))
        deltaqx_bs_list.append(integral_x_bs / (4 * np.pi))
        deltaqy_bs_list.append(integral_y_bs / (4 * np.pi))
    if field_K_included:
        i0 = np.argmin(np.abs(hor_off_list))
        print(f"[B_s focusing] at dx = {hor_off_list[i0] * 1e6:.1f} um: "
              f"ΔQx = {deltaqx_bs_list[i0]:.3e}, ΔQy = {deltaqy_bs_list[i0]:.3e}; "
              f"ΔQy quadratic fit coefficients "
              f"{np.polyfit(hor_off_list, deltaqy_bs_list, 2)}")

    return dict(
        case_label=case_label,
        hor_off_list=hor_off_list,
        qx_0=qx_0,
        qy_0=qy_0,
        deltaqx_list=np.array(deltaqx_list),
        deltaqy_list=np.array(deltaqy_list),
        deltaqx_local_list=np.array(deltaqx_local_list),
        deltaqy_local_list=np.array(deltaqy_local_list),
        orbit_scan_s=orbit_scan_s,          # local s, per placement below
        orbit_scan_x=np.array(orbit_scan_x),  # (n_offsets, n_placements, n_pts)
        orbit_scan_y=np.array(orbit_scan_y),
        betx_scan=np.array(betx_scan),
        bety_scan=np.array(bety_scan),
        init_conditions=np.array([[ic[kk] for kk in ('betx', 'alfx', 'bety', 'alfy')]
                                  for ic in init_conditions]),
        deltaqx_formula_list=np.array(deltaqx_formula_list),
        deltaqy_formula_list=np.array(deltaqy_formula_list),
        deltaqx_formula_pert_list=np.array(deltaqx_formula_pert_list),
        deltaqy_formula_pert_list=np.array(deltaqy_formula_pert_list),
        undulator_s_ranges=np.array(undulator_s_ranges),
        field_K_included=field_K_included,
        deltaqx_bs_list=np.array(deltaqx_bs_list),
        deltaqy_bs_list=np.array(deltaqy_bs_list),
    )


def get_case_data(place_label, wiggler_places, model_label):
    # (suffix: data_007b/ also holds the old, incompatible 007b cache)
    data_path = DATA_DIR / f'{place_label}_{model_label}_one_turn.npz'
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


def load_ring_tune_shift_007a(place_label, model_label, hor_off_list):
    """007a's closed-ring Twiss tune shift (undulator inserted in the ring)
    for the same case, from its cache -- None if not cached or scanned at
    different offsets."""
    data_path = DATA_DIR_007A / f'{place_label}_{model_label}.npz'
    if not data_path.exists():
        return None
    with np.load(data_path, allow_pickle=True) as npz:
        if not np.allclose(npz['hor_off_list'], hor_off_list):
            return None
        return {'x': npz['deltaqx_list'], 'y': npz['deltaqy_list']}


# Style of the fit-coefficient text boxes stacked under the legends.
text_box_kwargs = dict(va='top', ha='left', fontsize=8, linespacing=1.4,
                       bbox=dict(boxstyle='round', fc='white', alpha=0.85, edgecolor='0.7'))


def legend_outside(ax):
    # Anchor the legend to the top of the margin just outside the right
    # edge of the axes, so it doesn't sit on top of the data. Figures
    # using this get saved with bbox_inches='tight' and a
    # subplots_adjust(right=...) so the legend stays on-canvas both on
    # screen and in the PDF.
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8)

def stack_boxes_under_legend(ax, boxes):
    # Stack coefficient text boxes down the right margin, starting just
    # below the legend and left-aligned with it. The legend's (and each
    # box's) rendered size is only known once the figure has been laid
    # out and drawn, hence the canvas.draw() calls -- so this must run
    # *after* tight_layout()/subplots_adjust(), otherwise the measured
    # geometry refers to an axes about to change size.
    fig = ax.get_figure()
    fig.canvas.draw()
    to_axes = ax.transAxes.inverted()

    def frame_bbox(artist):
        # The drawn rounded frame, not the text inside it -- so boxes are
        # spaced and aligned frame-to-frame, like the legend.
        patch = artist.get_bbox_patch()
        return (patch or artist).get_window_extent().transformed(to_axes)

    legend_bbox = ax.get_legend().get_window_extent().transformed(to_axes)
    y = legend_bbox.y0
    for text, kwargs in boxes:
        y -= 0.04
        artist = ax.text(1.01, y, text, transform=ax.transAxes, **kwargs)
        fig.canvas.draw()
        # ax.text() anchors the *text*, while the legend's bbox_to_anchor
        # anchors its *frame*, so at the same x the rounded box overhangs
        # to the left of the legend by its padding. Measure that and
        # shift x by the difference to line the two frames up.
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
                                 title, suptitle, extra_meas=()):
    """Simulated vs measured Delta Q_x,y with a quadratic fit through each
    and coefficient boxes in the right margin -- same layout as the main
    tune-shift figure. sim_series is a list of (x, dq, label, color,
    marker, fit_box_prefix), one per simulated curve; dq and dq_meas are
    dicts keyed 'x'/'y', and labels may contain {p} for the plane.
    extra_meas is a list of (x, dq, label, color, marker) for further
    measured series, shown as points only (no fit / coefficient box).
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 11), sharex=True)
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
        for x_extra, dq_extra, label, color, marker in extra_meas:
            ax.plot(x_extra, dq_extra[plane], marker=marker, linestyle='none',
                    markersize=4, markerfacecolor='none', color=color,
                    label=label.format(p=plane))
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


MODEL_STYLES = {'SB': ('SplineBoris', 'tab:blue', 'o'),
                'MK': ('Multipole kick', 'tab:green', 's')}


def plot_tune_shift_panel(data, ax_x, ax_y, measured, ring=None):
    """Phase-advance tune shift vs offset for one (placement, model) case into
    (ax_x, ax_y), with its quadratic fit, the first-order-perturbation
    cross-checks and, if available, the measured closed-minus-open-gap
    shift. ring: optional {'x': dq, 'y': dq}, 007a's closed-ring Twiss tune
    shift, overlaid for reference. Returns the coefficient boxes for each
    axes, to be stacked under the legends once the figure layout is final
    (stack_boxes_under_legend).
    """
    hor_off_list = data['hor_off_list']
    dq_twiss = {'x': data['deltaqx_list'], 'y': data['deltaqy_list']}
    dq_formula = {'x': data['deltaqx_formula_list'],
                  'y': data['deltaqy_formula_list']}
    dq_formula_pert = {'x': data['deltaqx_formula_pert_list'],
                       'y': data['deltaqy_formula_pert_list']}
    styles = {'x': ('o', 'tab:blue', r'\delta K_1', r'\,'),
              'y': ('s', 'tab:orange', r'(-\delta K_1)', '')}
    # SB: full linear focusing from the 3D field on the wiggling orbit
    # (cached data from before this was added has no flag).
    if 'field_K_included' in data and bool(data['field_K_included']):
        styles = {'x': ('o', 'tab:blue', r'\,K_x', r'\,'),
                  'y': ('s', 'tab:orange', r'\,K_y', r'\,')}

    # coef[0]/[1]/[2] are just the quadratic-fit Taylor coefficients of
    # DeltaQ(dx) itself -- (1/2)*d^2(DeltaQ)/dx^2, d(DeltaQ)/dx, DeltaQ(0) --
    # not yet deconvolved into magnetic field gradients.
    all_boxes = []
    for ax, plane in zip((ax_x, ax_y), ('x', 'y')):
        marker, color, dk1, sep = styles[plane]
        coef = np.polyfit(hor_off_list, dq_twiss[plane], 2)
        for tag, dq in (('formula', dq_formula),
                        ('formula, perturbed beta', dq_formula_pert)):
            print(f"[{data['case_label']}, {tag}] ΔQ{plane} quadratic fit "
                  f"coefficients {np.polyfit(hor_off_list, dq[plane], 2)}")
        print(f"[{data['case_label']}] ΔQ{plane} quadratic fit "
              f"coefficients {coef}")

        ax.plot(hor_off_list, dq_twiss[plane], marker=marker, color=color,
                label='One-turn matrix')
        ax.plot(hor_off_list, data[f'deltaq{plane}_local_list'], marker=marker,
                linestyle=':', markerfacecolor='none', color=color,
                label=f'Local $\\Delta\\mu_{plane}$ (not $\\Delta Q$)')
        if ring is not None:
            ax.plot(hor_off_list, ring[plane], marker=marker, linestyle='none',
                    markerfacecolor='none', color='0.5',
                    label='Closed-ring Twiss (007a)')
        ax.plot(hor_off_list, np.poly1d(coef)(hor_off_list), linestyle='--',
                color='k', label='Quadratic fit')
        ax.plot(hor_off_list, dq_formula[plane], marker='^', linestyle='none',
                color='tab:green',
                label=f'$\\frac{{1}}{{4\\pi}}\\oint{dk1}{sep}\\beta_{{{plane},0}}\\,ds$')
        ax.plot(hor_off_list, dq_formula_pert[plane], marker='v',
                linestyle='none', color='tab:red',
                label=f'$\\frac{{1}}{{4\\pi}}\\oint{dk1}{sep}\\beta_{plane}\\,ds$')
        boxes = [(fit_box_text(plane, coef), text_box_kwargs)]

        if measured is not None:
            meas_x = np.asarray(measured['bump_amplitude_m'], dtype=float)
            meas_dq = np.asarray(measured[f'dtune_{plane}_measured'], dtype=float)
            coef_meas = np.polyfit(meas_x, meas_dq, 2)
            print(f"[measured] ΔQ{plane} quadratic fit coefficients {coef_meas}")
            ax.plot(meas_x, meas_dq, marker='d', linestyle='none',
                    markersize=4, color='tab:purple', label='Measurements')
            ax.plot(hor_off_list, np.poly1d(coef_meas)(hor_off_list),
                    linestyle='--', color='tab:red',
                    label='Quadratic fit (measured)')
            boxes.append((fit_box_text(plane, coef_meas, 'measured fit:\n'),
                          text_box_kwargs))

        ax.set_ylabel(f'$\\Delta Q_{plane}$')
        ax.grid(True, alpha=0.3)
        legend_outside(ax)
        all_boxes.append(boxes)
    return all_boxes


def plot_tune_shift_models_side_by_side(place_label, case_datas):
    """The per-model tune-shift figure, MK (left) and SB (right) next to
    each other on shared vertical axes for direct comparison."""
    measured = load_measured_tune_shift()
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), sharex=True, sharey='row')
    axes_boxes = []
    for col, model_label in enumerate(('MK', 'SB')):
        ax_x, ax_y = axes[:, col]
        data = case_datas[model_label]
        ring = load_ring_tune_shift_007a(place_label, model_label,
                                         data['hor_off_list'])
        if ring is not None:
            for plane in ('x', 'y'):
                print(f"[{place_label} {model_label}] one-turn matrix vs 007a "
                      f"closed ring: max |difference| in ΔQ{plane} = "
                      f"{np.max(np.abs(data[f'deltaq{plane}_list'] - ring[plane])):.2e}")
        boxes = plot_tune_shift_panel(data, ax_x, ax_y, measured, ring)
        axes_boxes.extend(zip((ax_x, ax_y), boxes))
        ax_x.set_title(MODEL_STYLES[model_label][0])
        ax_y.set_xlabel('Horizontal offset [m]')
        # sharey hides the right column's tick labels, show them anyway
        ax_x.tick_params(labelleft=True)
        ax_y.tick_params(labelleft=True)

    fig.suptitle(f'{place_label}: tune shift vs undulator horizontal offset\n'
                 '(open undulator from bare-ring initial optics, one-turn matrix)')
    fig.tight_layout()
    # Room right of each column for its legend + coefficient boxes.
    fig.subplots_adjust(right=0.81, wspace=0.85)
    # Only now that the axes geometry is final can the boxes be stacked
    # under the legend (their placement depends on its rendered height).
    for ax, boxes in axes_boxes:
        stack_boxes_under_legend(ax, boxes)

    out_path = OUT_DIR / f'{place_label}_tune_shift_MK_vs_SB_local_optics.pdf'
    # bbox_inches='tight' so the out-of-axes legends aren't clipped.
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved {out_path}")
    return fig


def plot_closed_gap_tune_shift(place_label, case_datas):
    """Simulated (one-turn-matrix) tune shift for both undulator models
    (case_datas: {model_label: data}) next to the measured closed-gap tune
    (the closed-minus-open-gap shift with the measured open-gap tune added
    back in). Every curve is shifted by its own (linearly interpolated)
    value at x = 0, so only the progression with offset is compared, not
    the absolute level.
    """
    measured = load_measured_tune_shift()
    if measured is None:
        return None
    meas_x = np.asarray(measured['bump_amplitude_m'], dtype=float)
    meas_q_open = {
        plane: np.asarray(measured[f'tune_{plane}_open_gap'], dtype=float)
        for plane in ('x', 'y')}
    meas_q_closed = {
        plane: (np.asarray(measured[f'dtune_{plane}_measured'], dtype=float)
                + meas_q_open[plane])
        for plane in ('x', 'y')}

    def shift_to_origin(x, dq):
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
        sim_series.append((
            x_sim, shift_to_origin(x_sim, dq_twiss),
            f'{model_name}: one-turn matrix $Q_{{p}}$',
            color, marker, f'{model_name} fit:\n'))

    fig = tune_shift_comparison_figure(
        sim_series, meas_x, shift_to_origin(meas_x, meas_q_closed),
        'Measured closed-gap $Q_{p}$',
        'Tune shift vs undulator horizontal offset (local optics),\n'
        'shifted to $\\Delta Q(0) = 0$',
        place_label,
        # Open-gap tune drift with bump amplitude (bump orbit through the
        # ring's own sextupoles, no undulator), for reference.
        extra_meas=[(meas_x, shift_to_origin(meas_x, meas_q_open),
                     'Measured open-gap $Q_{p}$', 'tab:gray', 'o')])
    out_path = OUT_DIR / f'{place_label}_tune_shift_vs_closed_gap_zeroed_local_optics.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved {out_path}")
    return fig


for place_label in CASES:
    wiggler_places = WIGGLER_CASES[place_label]
    case_datas = {model_label: get_case_data(place_label, wiggler_places,
                                             model_label)
                  for model_label in MODEL_LABELS}
    plot_tune_shift_models_side_by_side(place_label, case_datas)
    plot_closed_gap_tune_shift(place_label, case_datas)

# All figures are kept open so they can be reviewed interactively here, in
# addition to having been saved as PDFs above.
plt.show()
