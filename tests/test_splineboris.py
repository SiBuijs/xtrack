import matplotlib
matplotlib.use("Agg")  # headless -- TubeFitter's residual_tol search auto-plots via plt.show()

import numpy as np
from scipy import interpolate as sc_interpolate
from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import epsilon_0, hbar
import pytest
import xobjects as xo
from xobjects.test_helpers import fix_random_seed, for_all_test_contexts
import pandas as pd
from pathlib import Path

import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack._temp.splineboris.tube_fitter import TubeFitter, DEFAULT_N_FRAMES
from xtrack._temp.splineboris.longitudinal_fitter import LongitudinalFitter
from xtrack.beam_elements.splineboris_src.spline_B_field_eval_python import (
    evaluate_B, hermite_to_polynomial,
)

SOLENOID_MODEL_PARAMS = {
    "L": 4.0,
    "a": 0.3,
    "B0": 1.5,
    "z0": 20.0,
}
SOLENOID_INTERVAL = 30.0
SOLENOID_DX = 0.001
SOLENOID_DY = 0.001
SOLENOID_MULTIPOLE_ORDER = 2
SOLENOID_N_STEPS = 5000
SOLENOID_Z_POINT_COUNT = SOLENOID_N_STEPS + 1

@pytest.fixture(scope="module")
def test_data_dir():
    return Path(__file__).parent.parent / "test_data"

@pytest.fixture
def make_uniform_splineboris():
    def _make(Bx=0, By=0, Bs=0, s_start=0, s_end=1, n_steps=100,
                multipole_order=1, radiation_flag=0, scale_b=1.0, kn=None, ks=None):
        # Uniform field: Hermite params (val_start, der_start, val_end, der_end, mean)
        # For a constant field B, all boundary values = B, derivatives = 0, mean = B
        Bx_h = [Bx, 0, Bx, 0, Bx]
        By_h = [By, 0, By, 0, By]
        Bs_h = [Bs, 0, Bs, 0, Bs]

        # Verify the polynomials evaluate to constants (polynomial is in local s = s - s_start)
        s_test = np.linspace(s_start, s_end, 100)
        s_local = s_test - s_start
        from xtrack.beam_elements.splineboris_src.spline_B_field_eval_python import hermite_to_polynomial
        xo.assert_allclose(hermite_to_polynomial(s_start, s_end, Bx_h)(s_local), Bx, rtol=1e-12, atol=1e-12)
        xo.assert_allclose(hermite_to_polynomial(s_start, s_end, By_h)(s_local), By, rtol=1e-12, atol=1e-12)
        xo.assert_allclose(hermite_to_polynomial(s_start, s_end, Bs_h)(s_local), Bs, rtol=1e-12, atol=1e-12)

        splineboris = xt.SplineBoris(
            bs=xt.Spline4(*Bs_h),
            by=(xt.Spline4(*By_h),),
            bx=(xt.Spline4(*Bx_h),),
            length=s_end - s_start,
            n_steps=n_steps,
            scale_b=scale_b,
            radiation_flag=radiation_flag,
        )
        return splineboris
    return _make

@pytest.fixture(scope="module")
def make_segment_field():
    def _make(bs, by, bx, L, multipole_order_local, s_start=0.0):
        # ensure we have simple lists/arrays
        bs_arr = np.asarray(bs, dtype=float).tolist()
        B_norm_list = [np.asarray(b, dtype=float).tolist() for b in by]
        B_skew_list = [np.asarray(b, dtype=float).tolist() for b in bx]
        def field(x, y, z):
            s_loc = z - s_start
            Bx, By, Bs = evaluate_B(x, y, s_loc,
                                    bs_arr,
                                    B_norm_list,
                                    B_skew_list,
                                    L,
                                    multipole_order_local)
            return Bx, By, Bs
        return field
    return _make

@pytest.fixture(scope="module")
def solenoid_field():
    return SolenoidField(**SOLENOID_MODEL_PARAMS)

@pytest.fixture(scope="module")
def solenoid_fit(solenoid_field):
    """
    Fit a solenoid field map with TubeFitter (stage 1) and LongitudinalFitter
    (stage 2). TubeFitter only ever passes on the q=0 and q=1 rows of its
    fitted potential, and Bs is taken from the map's on-axis data -- q>=2
    content (needed to represent this field's dBy/dy) is never exported,
    and is expected to be regenerated downstream by the Schueren/Table-1
    field evaluator from the on-axis components alone. See
    examples/splineboris/claude_notes/tube_schueren_integration.md.

    This combination used to diverge by an order of magnitude from
    VariableSolenoid, back when TubeFitter tried to fit Bs *jointly* with
    Bx/By via a div(B)=0 elimination (removed -- see
    examples/splineboris/claude_notes/transverse_bs_coupling_gap.md); the
    test below checks that fitting Bs independently resolves it.
    """
    sf = solenoid_field

    x_axis = np.linspace(
        -SOLENOID_MULTIPOLE_ORDER * SOLENOID_DX / 2,
        SOLENOID_MULTIPOLE_ORDER * SOLENOID_DX / 2,
        SOLENOID_MULTIPOLE_ORDER + 1,
    )
    y_axis = np.linspace(
        -SOLENOID_MULTIPOLE_ORDER * SOLENOID_DY / 2,
        SOLENOID_MULTIPOLE_ORDER * SOLENOID_DY / 2,
        SOLENOID_MULTIPOLE_ORDER + 1,
    )
    z_axis = np.linspace(0, SOLENOID_INTERVAL, SOLENOID_Z_POINT_COUNT)
    x_grid, y_grid, z_grid = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    bx, by, bz = sf.get_field(x_grid.ravel(), y_grid.ravel(), z_grid.ravel())

    df_raw_data = pd.DataFrame(
        np.column_stack(
            [x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), bx.ravel(), by.ravel(), bz.ravel()]
        ),
        columns=["X", "Y", "Z", "Bx", "By", "Bs"],
    ).set_index(["X", "Y", "Z"])

    fitter = TubeFitter(
        raw_data=df_raw_data,
        n_frames=2000,
        distance_unit=1,
        deg=SOLENOID_MULTIPOLE_ORDER - 1,
    )
    fitter.fit()

    diag = fitter.check_trace_consistency()
    assert diag["relative_rms"] < 1e-2, (
        f"TubeFitter's own (unused) q=2 fit is inconsistent with the "
        f"div(B)=0/Bs' prediction at the {diag['relative_rms']:.3%} level "
        f"-- the underlying tube fit may be under-resolved."
    )

    z, F, names = fitter.on_axis_multipoles()
    lf = LongitudinalFitter(0.0, SOLENOID_INTERVAL)
    lf.fit(z, F, names)
    lf.fit(*fitter.on_axis_bs(), [("Bs", 0)])
    return lf

UNDULATOR_PERIOD = 0.036
UNDULATOR_MULTIPOLE_ORDER = 3


def _read_undulator_map(test_data_dir):
    return pd.read_csv(
        test_data_dir / "sls" / "undulator_field_map.txt", sep=r"\s+", header=None,
        names=["X", "Y", "Z", "Bx", "By", "Bs"],
    ).set_index(["X", "Y", "Z"])


def _fit_undulator(df_raw_data):
    """Stage 1 + stage 2 on the SLS undulator map (2201 planes, 1 mm apart),
    one tube frame per plane."""
    tf = TubeFitter(raw_data=df_raw_data, n_frames=2201, distance_unit=1e-3, deg=2)
    tf.fit()
    z, F, names = tf.on_axis_multipoles()
    lf = LongitudinalFitter(z[0], z[-1], n_elements=750, end_condition="free",
                            period=UNDULATOR_PERIOD)
    lf.fit(z, F, names)
    lf.fit(*tf.on_axis_bs(), [("Bs", 0)])
    return lf


@pytest.fixture(scope="module")
def undulator_raw_data(test_data_dir):
    return _read_undulator_map(test_data_dir)


@pytest.fixture(scope="module")
def undulator_fit(undulator_raw_data):
    return _fit_undulator(undulator_raw_data)


# Small, fast synthetic dataset for exercising TubeFitter's n_frames/residual_tol
# selection logic in isolation (i.e. not the full solenoid-vs-VariableSolenoid
# physics comparison above). The field is a low-order sinusoid in z plus fixed
# per-point noise, so:
#   - a loose residual_tol is reachable once enough frames resolve the sinusoid
#   - a residual_tol far below the noise floor is never reachable, regardless
#     of n_frames, exercising the warn-and-fall-back-to-DOF-ceiling path
TUBEFITTER_NFRAMES_TEST_B0 = 0.05
TUBEFITTER_NFRAMES_TEST_NOISE_STD = 6e-4
TUBEFITTER_NFRAMES_TEST_N_CYCLES = 3
TUBEFITTER_NFRAMES_TEST_N_X = 3
TUBEFITTER_NFRAMES_TEST_N_Y = 3
TUBEFITTER_NFRAMES_TEST_N_Z = 201

@pytest.fixture(scope="module")
def tubefitter_noisy_sine_raw_data():
    rng = np.random.default_rng(12345)

    x_axis = np.linspace(-1e-3, 1e-3, TUBEFITTER_NFRAMES_TEST_N_X)
    y_axis = np.linspace(-1e-3, 1e-3, TUBEFITTER_NFRAMES_TEST_N_Y)
    z_axis = np.linspace(0.0, 1.0, TUBEFITTER_NFRAMES_TEST_N_Z)
    x_grid, y_grid, z_grid = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")

    phase = 2 * np.pi * TUBEFITTER_NFRAMES_TEST_N_CYCLES * z_grid
    by = (TUBEFITTER_NFRAMES_TEST_B0 * np.sin(phase)
          + rng.normal(scale=TUBEFITTER_NFRAMES_TEST_NOISE_STD, size=z_grid.shape))
    bx = (0.3 * TUBEFITTER_NFRAMES_TEST_B0 * np.cos(phase)
          + rng.normal(scale=TUBEFITTER_NFRAMES_TEST_NOISE_STD, size=z_grid.shape))
    bs = np.zeros_like(z_grid)

    return pd.DataFrame(
        {
            "X": x_grid.ravel(),
            "Y": y_grid.ravel(),
            "Z": z_grid.ravel(),
            "Bx": bx.ravel(),
            "By": by.ravel(),
            "Bs": bs.ravel(),
        }
    ).set_index(["X", "Y", "Z"])

def test_splineboris_tubefitter_solenoid_vs_variable_solenoid(
        solenoid_field, solenoid_fit):
    """
    Track through the fitted solenoid (see ``solenoid_fit``) and compare
    with a VariableSolenoid built from the on-axis field.

    A paraxial solenoid's transverse field (Bx=-x*Bz'/2, By=-y*Bz'/2) needs
    a q=2 term (Psi[0,2]) to represent dBy/dy, which TubeFitter never
    exports. This test checks that tracking still agrees with
    VariableSolenoid regardless, because the Schueren/Table-1 field
    evaluator used by SplineBoris regenerates that q=2 content from the
    exported on-axis components via div(B)=0.
    """
    interval = SOLENOID_INTERVAL

    delta = np.array([0, 4])
    p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                    energy0=45.6e6,
                    x=1e-3,
                    px=-1e-3*(1+delta),
                    y=1e-3,
                    delta=delta)

    sf = solenoid_field

    line_splineboris = solenoid_fit.to_line(multipole_order=SOLENOID_MULTIPOLE_ORDER)
    line_splineboris.build_tracker()
    p_splineboris = p0.copy()
    line_splineboris.track(p_splineboris, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_splineboris = line_splineboris.record_last_track

    n_steps = SOLENOID_N_STEPS
    z_axis_ref = np.linspace(0, interval, n_steps)
    Bz_axis = sf.get_field(0 * z_axis_ref, 0 * z_axis_ref, z_axis_ref)[2]
    P0_J = p0.p0c[0] * qe / clight
    brho = P0_J / qe / p0.q0
    ks = Bz_axis / brho
    ks_entry = ks[:-1]
    ks_exit = ks[1:]
    dz = z_axis_ref[1] - z_axis_ref[0]
    line_varsol = xt.Line(elements=[
        xt.VariableSolenoid(length=dz, ks_profile=[ks_entry[ii], ks_exit[ii]])
        for ii in range(len(z_axis_ref) - 1)
    ])
    line_varsol.build_tracker()
    p_varsol = p0.copy()
    line_varsol.track(p_varsol, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_varsol = line_varsol.record_last_track

    # Compare positions at the SplineBoris element boundaries (where the
    # monitor records), with the fine VariableSolenoid track interpolated
    # there. Slopes from finite differences would only resolve the element
    # length.
    n_part = mon_splineboris.x.shape[0]
    for i_part in range(n_part):
        s_sb = mon_splineboris.s[i_part, :]
        for coord in ("x", "y"):
            sb = getattr(mon_splineboris, coord)[i_part, :]
            vs = getattr(mon_varsol, coord)[i_part, :]
            vs_check = np.interp(s_sb, mon_varsol.s[i_part, :], vs)
            xo.assert_allclose(sb, vs_check, rtol=0, atol=3e-2 * np.ptp(vs))

def _sine_bump(z, L=1.0):
    return np.sin(6 * np.pi * z / L) * np.sin(np.pi * z / L) ** 4


def _fit_sine_bump(n_elements, n_data, **kwargs):
    z = (np.arange(n_data) + 0.5) / n_data
    lf = LongitudinalFitter(0.0, 1.0, n_elements=n_elements, **kwargs)
    lf.fit(z, _sine_bump(z), [("By", 0)])
    return lf


def _piece_limits(spl, nodes, der):
    """Left and right limits of the der-th derivative at the interior nodes."""
    pp = sc_interpolate.PPoly.from_spline(spl)
    if der:
        pp = pp.derivative(der)
    i = np.searchsorted(pp.x, nodes[1:-1], side="right") - 1
    left = np.array([np.polyval(pp.c[:, j - 1], pp.x[j] - pp.x[j - 1]) for j in i])
    right = pp.c[-1, i]
    return left, right


def test_longitudinal_fitter_convergence_and_c3():
    """Synthetic field at cell centres, "zero" ends: error ~ Delta^5, C3 at
    interior nodes, f..f''' zero at both ends."""
    zz = np.linspace(0, 1, 20001)
    errors = {}
    for n in (40, 100):
        lf = _fit_sine_bump(n, n)
        errors[n] = np.max(np.abs(lf.splines[("By", 0)](zz) - _sine_bump(zz)))
    assert errors[40] < 3e-5
    assert errors[40] / errors[100] > 80  # 2.5x shorter elements -> ~2.5^5 = 98x

    lf = _fit_sine_bump(40, 40)
    spl = lf.splines[("By", 0)]
    for der in range(4):
        d_spl = spl.derivative(der) if der else spl
        scale = np.max(np.abs(d_spl(zz)))
        left, right = _piece_limits(spl, lf.nodes, der)
        assert np.max(np.abs(left - right)) < 1e-12 * scale
        xo.assert_allclose(d_spl([0.0, 1.0]), 0.0, rtol=0, atol=1e-14 * scale)


def test_longitudinal_fitter_free_ends_reproduce_quartic():
    z = np.sort(np.random.default_rng(1).uniform(0.0, 2.0, 300))
    poly = np.polynomial.Polynomial([1.0, 2.0, -3.0, 0.5, -0.25])
    lf = LongitudinalFitter(0.0, 2.0, n_elements=30, end_condition="free")
    lf.fit(z, poly(z), [("Bs", 0)])
    zz = np.linspace(0, 2, 2001)
    xo.assert_allclose(lf.splines[("Bs", 0)](zz), poly(zz), rtol=0, atol=1e-12)


def test_longitudinal_fitter_element_export_reproduces_spline():
    """Rebuilding each element's quartic from its 5 numbers reproduces the
    B-spline inside the element."""
    lf = _fit_sine_bump(40, 200)
    spl = lf.splines[("By", 0)]
    params = lf.element_params(("By", 0))
    scale = np.max(np.abs(spl(np.linspace(0, 1, 2001))))
    for k in range(lf.n_elements):
        s = np.linspace(lf.nodes[k], lf.nodes[k + 1], 11)
        poly = hermite_to_polynomial(lf.nodes[k], lf.nodes[k + 1], params[k])
        xo.assert_allclose(poly(s - lf.nodes[k]), spl(s), rtol=0, atol=1e-14 * scale)


def test_longitudinal_fitter_preserve_integral():
    lf = _fit_sine_bump(40, 40, preserve_integral=True)
    z, f = lf.data[("By", 0)]
    xo.assert_allclose(lf.splines[("By", 0)].integrate(0, 1), np.trapezoid(f, z),
                       rtol=0, atol=1e-15)


def test_longitudinal_fitter_raises_on_empty_element():
    lf = LongitudinalFitter(0.0, 1.0, n_elements=50)
    with pytest.raises(ValueError, match="contains no data point"):
        lf.fit(np.linspace(0, 1, 40), np.zeros(40), [("By", 0)])
    with pytest.raises(ValueError, match="outside"):
        lf.fit(np.linspace(-0.1, 1, 400), np.zeros(400), [("By", 0)])


def test_longitudinal_fitter_warns_few_elements_per_period():
    with pytest.warns(UserWarning, match="elements per period"):
        LongitudinalFitter(0.0, 1.0, n_elements=100, period=0.1)


def test_undulator_fit_vs_raw_map(undulator_raw_data, undulator_fit):
    """Regression on the SLS undulator map: the exported elements reproduce
    the raw field (on and off axis) and stage 2 reproduces its own data.
    One frame per plane avoids TubeFitter's tent smoothing; the old tent
    export gave ~2.5e-3 / 8.8e-3 / 8.1e-3 RMS (Bx/By/Bs) on the same frames."""
    lf = undulator_fit
    idx = undulator_raw_data.index
    x, y, z = (idx.get_level_values(lvl).to_numpy() * 1e-3 for lvl in "XYZ")
    b_raw = undulator_raw_data[["Bx", "By", "Bs"]].to_numpy()
    ref = np.max(np.abs(b_raw[:, 1]))

    line = lf.to_line(multipole_order=UNDULATOR_MULTIPOLE_ORDER)
    k = np.clip(np.searchsorted(lf.nodes, z, side="right") - 1, 0, lf.n_elements - 1)
    b_fit = np.zeros_like(b_raw)
    for i in np.unique(k):
        m = k == i
        s_local = np.clip(z[m] - lf.nodes[i], 0, lf.delta)
        b_fit[m] = np.column_stack(line.elements[i].get_field(x[m], y[m], s_local))

    rms = np.sqrt(np.mean((b_fit - b_raw) ** 2, axis=0)) / ref
    assert np.all(rms < [3e-4, 3e-4, 2.5e-4]), rms

    on_axis = (x == 0) & (y == 0)
    int_raw = np.trapezoid(b_raw[on_axis, 1], z[on_axis])
    int_fit = np.trapezoid(b_fit[on_axis, 1], z[on_axis])
    assert abs(int_fit - int_raw) < 5e-6

    # Stage 2 alone: the splines reproduce their own data points (der 0
    # only -- the higher multipoles from this 3x3 grid are noise-dominated).
    for name, (zd, fd) in lf.data.items():
        if name[1] > 0:
            continue
        err = np.max(np.abs(lf.splines[name](zd) - fd)) / ref
        assert err < 5e-4, (name, err)


def test_longitudinal_fitter_to_multipole_line_matches_mean_field():
    """``to_multipole_line()`` converts each element's mean field via
    ``knl[n] = length / brho0 * (By, n)``, ``ksl[n] = length / brho0 * (Bx, n)``
    (see ``track_magnet_kick.h::evaluate_field_from_strengths``). A constant
    field is reproduced exactly with "free" ends."""
    B0, G, A0, Gs = 0.5, 20.0, 0.1, -8.0
    z = np.linspace(0.0, 1.0, 21)
    names = [("By", 0), ("By", 1), ("Bx", 0), ("Bx", 1)]
    F = np.tile([B0, G, A0, Gs], (len(z), 1))
    lf = LongitudinalFitter(0.0, 1.0, n_elements=4, end_condition="free")
    lf.fit(z, F, names)

    p0c = 2.7e9
    q0 = 1.0
    brho0 = p0c / (clight * q0)
    line = lf.to_multipole_line(p0c=p0c, multipole_order=3, q0=q0)

    assert len(line.elements) == lf.n_elements
    assert all(isinstance(el, xt.Multipole) and el.isthick for el in line.elements)
    for el in line.elements:
        xo.assert_allclose(np.array(el.knl) / el.length * brho0, [B0, G, 0.0], rtol=1e-12, atol=1e-12)
        xo.assert_allclose(np.array(el.ksl) / el.length * brho0, [A0, Gs, 0.0], rtol=1e-12, atol=1e-12)
    xo.assert_allclose(sum(el.length for el in line.elements), 1.0, rtol=0, atol=1e-12)


def test_longitudinal_fitter_to_multipole_line_drops_bs_with_warning(capsys):
    z = np.linspace(0.0, 1.0, 21)
    lf = LongitudinalFitter(0.0, 1.0, n_elements=4, end_condition="free")
    lf.fit(z, np.full(len(z), 0.5), [("By", 0)])
    lf.fit(z, np.full(len(z), 0.3), [("Bs", 0)])

    capsys.readouterr()
    line = lf.to_multipole_line(p0c=2.7e9)
    assert "no Multipole equivalent" in capsys.readouterr().out
    for el in line.elements:
        assert el.knl[0] != 0.0


def test_longitudinal_fitter_field_tol_drops_small_components():
    z = np.linspace(0.0, 1.0, 21)
    lf = LongitudinalFitter(0.0, 1.0, n_elements=4, end_condition="free")
    lf.fit(z, np.column_stack([np.full(21, 1.0), np.full(21, 1e-5), np.full(21, 1.0)]),
           [("By", 0), ("Bx", 0), ("By", 1)])
    line = lf.to_line(field_tol=1e-3, r_ref=1e-2)
    el = line.elements[0]
    assert el.by[0, 4] == pytest.approx(1.0)
    assert el.bx[0, 4] == 0.0       # 1e-5 < 1e-3 * 1.0
    assert el.by[1, 4] == pytest.approx(1.0)  # 1.0 * r_ref = 1e-2 >= 1e-3

def test_tubefitter_n_frames_and_residual_tol_mutually_exclusive(tubefitter_noisy_sine_raw_data):
    """Passing both n_frames and residual_tol is an ambiguous request, and should
    raise rather than silently prioritizing one over the other."""
    with pytest.raises(ValueError):
        TubeFitter(
            raw_data=tubefitter_noisy_sine_raw_data,
            distance_unit=1.0,
            deg=1,
            n_frames=50,
            residual_tol=0.05,
        )

def test_tubefitter_n_frames_default(tubefitter_noisy_sine_raw_data):
    """With neither n_frames nor residual_tol given, TubeFitter should fall back
    to DEFAULT_N_FRAMES (clamped into the valid range) without running a search."""
    fitter = TubeFitter(raw_data=tubefitter_noisy_sine_raw_data, distance_unit=1.0, deg=1)

    assert fitter.n_frames == DEFAULT_N_FRAMES
    assert fitter.n_frames_search_trace is None

@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive.*")
def test_tubefitter_residual_tol_search_reaches_target(tubefitter_noisy_sine_raw_data):
    """A loose residual_tol should be reachable well below the DOF ceiling, and
    the selected n_frames should actually meet the target."""
    target = 0.05
    fitter = TubeFitter(
        raw_data=tubefitter_noisy_sine_raw_data,
        distance_unit=1.0,
        deg=1,
        residual_tol=target,
    )

    dof_ceiling = fitter._dof_ceiling()
    assert 2 <= fitter.n_frames <= dof_ceiling
    assert fitter.n_frames < dof_ceiling, (
        "search should find a frame count well below the DOF ceiling for such "
        "a loose target -- getting the ceiling suggests the search silently "
        "fell back instead of converging"
    )

    assert fitter.n_frames_search_trace
    assert fitter.n_frames in fitter.n_frames_search_trace
    rel_bskew, rel_bnorm = fitter.n_frames_search_trace[fitter.n_frames]
    assert max(rel_bskew, rel_bnorm) <= target

@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive.*")
def test_tubefitter_residual_tol_unreachable_falls_back(tubefitter_noisy_sine_raw_data, capsys):
    """A residual_tol far below the noise floor can never be met, no matter how
    many frames are used -- construction should still succeed, falling back to
    the DOF ceiling with a warning instead of raising."""
    target = 1e-5
    fitter = TubeFitter(
        raw_data=tubefitter_noisy_sine_raw_data,
        distance_unit=1.0,
        deg=1,
        residual_tol=target,
    )

    dof_ceiling = fitter._dof_ceiling()
    assert fitter.n_frames == dof_ceiling
    assert fitter.n_frames_search_trace

    captured = capsys.readouterr()
    assert "WARNING" in captured.out

@for_all_test_contexts
def test_splineboris_to_dict_from_dict_roundtrip(test_context):
    element = xt.SplineBoris(
        length=1.2,
        n_steps=4,
        shift_x=1e-3,
        shift_y=-2e-3,
        scale_b=1.7,
        radiation_flag=1,
        bs=xt.Spline4(0.1, 0.2, 0.3, 0.4, 0.5),
        by=(
            xt.Spline4(1.0, 1.1, 1.2, 1.3, 1.4),
            None,
            xt.Spline4(3.0, 3.1, 3.2, 3.3, 3.4),
        ),
        bx=(
            None,
            xt.Spline4(-2.0, -2.1, -2.2, -2.3, -2.4),
        ),
        knl=[0.01, -0.02, 0.03],
        ksl=[-0.04, 0.05],
        _context=test_context,
    )

    element_dict = element.to_dict()

    assert 'bs' in element_dict
    assert 'by' in element_dict
    assert 'bx' in element_dict
    assert isinstance(element_dict['bs'], dict)
    assert isinstance(element_dict['by'], list)
    assert isinstance(element_dict['bx'], list)
    assert element_dict['bs']['mean'] == 0.5
    assert 'integral' not in element_dict['bs']
    assert 'multipole_order' not in element_dict

    legacy_ctor_dict = element_dict.copy()
    legacy_ctor_dict.pop('__class__', None)
    legacy_ctor_dict['bs'] = legacy_ctor_dict['bs'].copy()
    legacy_ctor_dict['bs']['integral'] = legacy_ctor_dict['bs'].pop('mean')
    with pytest.raises(ValueError, match='mean'):
        xt.SplineBoris(_context=test_context, **legacy_ctor_dict)

    roundtrip = xt.SplineBoris.from_dict(element_dict, _context=test_context)

    ctor_dict = element_dict.copy()
    ctor_dict.pop('__class__', None)
    ctor_roundtrip = xt.SplineBoris(_context=test_context, **ctor_dict)

    element_cpu = element.copy(_context=xo.ContextCpu())
    roundtrip_cpu = roundtrip.copy(_context=xo.ContextCpu())
    ctor_roundtrip_cpu = ctor_roundtrip.copy(_context=xo.ContextCpu())

    for candidate in (roundtrip_cpu, ctor_roundtrip_cpu):
        xo.assert_allclose(candidate.length, element_cpu.length, atol=0, rtol=0)
        xo.assert_allclose(candidate.n_steps, element_cpu.n_steps, atol=0, rtol=0)
        xo.assert_allclose(candidate.shift_x, element_cpu.shift_x, atol=0, rtol=0)
        xo.assert_allclose(candidate.shift_y, element_cpu.shift_y, atol=0, rtol=0)
        xo.assert_allclose(candidate.scale_b, element_cpu.scale_b, atol=0, rtol=0)
        xo.assert_allclose(candidate.radiation_flag, element_cpu.radiation_flag, atol=0, rtol=0)
        xo.assert_allclose(candidate.bs, element_cpu.bs, atol=0, rtol=0)
        xo.assert_allclose(candidate.by, element_cpu.by, atol=0, rtol=0)
        xo.assert_allclose(candidate.bx, element_cpu.bx, atol=0, rtol=0)
        xo.assert_allclose(candidate.knl, element_cpu.knl, atol=0, rtol=0)
        xo.assert_allclose(candidate.ksl, element_cpu.ksl, atol=0, rtol=0)

def test_splineboris_multipole_kick_matches_stepwise_thin_multipoles():
    length = 1.7
    n_steps = 17
    knl = np.array([1.2e-5, -3.4e-4, 1.5e-3])
    ksl = np.array([-4e-6, 2.1e-5, -7e-4])

    splineboris = xt.SplineBoris(
        length=length,
        n_steps=n_steps,
        knl=knl,
        ksl=ksl,
    )

    reference_elements = []
    for _ in range(n_steps):
        reference_elements.append(xt.Drift(length=0.5 * length / n_steps))
        reference_elements.append(xt.Multipole(
            knl=knl / n_steps,
            ksl=ksl / n_steps,
        ))
        reference_elements.append(xt.Drift(length=0.5 * length / n_steps))
    reference = xt.Line(elements=reference_elements)

    p0 = xt.Particles(
        p0c=7e9,
        x=1.3e-3,
        px=2.1e-4,
        y=-0.7e-3,
        py=-1.2e-4,
        delta=3e-4,
    )

    p_splineboris = p0.copy()
    p_splineboris_line = p0.copy()
    p_ref = p0.copy()

    splineboris.track(p_splineboris)
    line_splineboris = xt.Line(elements=[splineboris.copy()])
    line_splineboris.build_tracker()
    line_splineboris.track(p_splineboris_line)
    reference.track(p_ref)

    xo.assert_allclose(p_splineboris_line.x, p_splineboris.x, rtol=0, atol=5e-14)
    xo.assert_allclose(p_splineboris_line.px, p_splineboris.px, rtol=0, atol=5e-14)
    xo.assert_allclose(p_splineboris_line.y, p_splineboris.y, rtol=0, atol=5e-14)
    xo.assert_allclose(p_splineboris_line.py, p_splineboris.py, rtol=0, atol=5e-14)

    xo.assert_allclose(p_splineboris.x, p_ref.x, rtol=0, atol=5e-11)
    xo.assert_allclose(p_splineboris.px, p_ref.px, rtol=0, atol=5e-11)
    xo.assert_allclose(p_splineboris.y, p_ref.y, rtol=0, atol=5e-11)
    xo.assert_allclose(p_splineboris.py, p_ref.py, rtol=0, atol=5e-11)
    xo.assert_allclose(p_splineboris.zeta, p_ref.zeta, rtol=0, atol=5e-11)

def test_splineboris_multipole_field_contributes_to_mean_radiation():
    length = 1.0
    n_steps = 100
    B_T = 2.0
    p0c = 5e9
    brho = p0c / clight
    knl0 = B_T * length / brho

    particles = xt.Particles(
        p0c=p0c,
        px=1e-4,
        py=-1e-4,
        delta=0,
        mass0=xt.ELECTRON_MASS_EV,
    )
    particles_before = particles.copy()

    splineboris = xt.SplineBoris(
        length=length,
        n_steps=n_steps,
        knl=[knl0],
        radiation_flag=1,
    )
    splineboris.track(particles)

    gamma = (particles_before.energy / particles_before.mass0)[0]
    gamma0 = particles_before.gamma0[0]
    rho_0 = brho / B_T
    mass0_kg = particles_before.mass0 * qe / clight**2
    r0 = qe**2 / (4 * np.pi * epsilon_0 * mass0_kg * clight**2)
    ps = (2 * r0 * clight * mass0_kg * clight**2 * gamma0**2 * gamma**2) / (3 * rho_0**2)
    expected_delta_e_ev = -ps * (length / clight) / qe
    tracked_delta_e_ev = ((particles.ptau - particles_before.ptau) * particles.p0c)[0]

    xo.assert_allclose(tracked_delta_e_ev, expected_delta_e_ev, rtol=5e-3, atol=0)

def test_splineboris_scale_b_does_not_scale_multipole_components():
    length = 1.2
    n_steps = 30
    knl = [2.0e-3, -1.0e-2]
    ksl = [-1.5e-3, 4.0e-3]

    el_scale_1 = xt.SplineBoris(
        length=length,
        n_steps=n_steps,
        knl=knl,
        ksl=ksl,
        scale_b=1.0,
    )
    el_scale_7 = xt.SplineBoris(
        length=length,
        n_steps=n_steps,
        knl=knl,
        ksl=ksl,
        scale_b=7.0,
    )

    p_scale_1 = xt.Particles(
        p0c=7e9,
        x=1.3e-3,
        px=2.1e-4,
        y=-0.7e-3,
        py=-1.2e-4,
        delta=3e-4,
    )
    p_scale_7 = p_scale_1.copy()

    xt.Line(elements=[el_scale_1]).track(p_scale_1)
    xt.Line(elements=[el_scale_7]).track(p_scale_7)

    xo.assert_allclose(p_scale_7.x, p_scale_1.x, rtol=0, atol=1e-15)
    xo.assert_allclose(p_scale_7.px, p_scale_1.px, rtol=0, atol=1e-15)
    xo.assert_allclose(p_scale_7.y, p_scale_1.y, rtol=0, atol=1e-15)
    xo.assert_allclose(p_scale_7.py, p_scale_1.py, rtol=0, atol=1e-15)
    xo.assert_allclose(p_scale_7.zeta, p_scale_1.zeta, rtol=0, atol=1e-15)

def test_splineboris_spline_and_multipole_fields_add_for_mean_radiation():
    length = 1.0
    n_steps = 100
    B_spline_T = 1.25
    B_multipole_T = 0.75
    B_total_T = B_spline_T + B_multipole_T
    p0c = 5e9
    brho = p0c / clight
    knl0 = B_multipole_T * length / brho

    by_h = [B_spline_T, 0, B_spline_T, 0, B_spline_T]

    particles = xt.Particles(
        p0c=p0c,
        px=1e-4,
        py=-1e-4,
        delta=0,
        mass0=xt.ELECTRON_MASS_EV,
    )
    particles_before = particles.copy()

    splineboris = xt.SplineBoris(
        length=length,
        n_steps=n_steps,
        by=(xt.Spline4(*by_h),),
        knl=[knl0],
        radiation_flag=1,
    )
    splineboris.track(particles)

    gamma = (particles_before.energy / particles_before.mass0)[0]
    gamma0 = particles_before.gamma0[0]
    rho_0 = brho / B_total_T
    mass0_kg = particles_before.mass0 * qe / clight**2
    r0 = qe**2 / (4 * np.pi * epsilon_0 * mass0_kg * clight**2)
    ps = (2 * r0 * clight * mass0_kg * clight**2 * gamma0**2 * gamma**2) / (3 * rho_0**2)
    expected_delta_e_ev = -ps * (length / clight) / qe
    tracked_delta_e_ev = ((particles.ptau - particles_before.ptau) * particles.p0c)[0]

    xo.assert_allclose(tracked_delta_e_ev, expected_delta_e_ev, rtol=5e-3, atol=0)

def test_splineboris_scale_b_scales_field_and_tracking(make_uniform_splineboris):
    scale_b = 2.5
    Bx = 0.03
    By = -0.07
    Bs = 0.01

    scaled = make_uniform_splineboris(
        Bx=Bx, By=By, Bs=Bs, n_steps=20, scale_b=scale_b)
    reference = make_uniform_splineboris(
        Bx=scale_b * Bx, By=scale_b * By, Bs=scale_b * Bs, n_steps=20)

    xo.assert_allclose(scaled.scale_b, scale_b, atol=0, rtol=0)
    xo.assert_allclose(
        scaled.get_field(1e-3, -2e-3, 0.4),
        reference.get_field(1e-3, -2e-3, 0.4),
        atol=1e-14,
        rtol=0,
    )

    particle_ref = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1.0,
        energy0=1e9,
    )

    p_scaled = particle_ref.copy()
    p_reference = particle_ref.copy()
    for pp in (p_scaled, p_reference):
        pp.x = 1e-3
        pp.y = -2e-3
        pp.px = 3e-4
        pp.py = -4e-4

    line_scaled = xt.Line(elements=[scaled])
    line_reference = xt.Line(elements=[reference])
    line_scaled.particle_ref = particle_ref.copy()
    line_reference.particle_ref = particle_ref.copy()

    line_scaled.track(p_scaled)
    line_reference.track(p_reference)

    xo.assert_allclose(p_scaled.x, p_reference.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_scaled.y, p_reference.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_scaled.px, p_reference.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_scaled.py, p_reference.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_scaled.zeta, p_reference.zeta, atol=1e-15, rtol=0)

def test_splineboris_backtrack_twiss_checks_s():
    def make_splineboris(length, scale=1.0):
        return xt.SplineBoris(
            bs=xt.Spline4(
                scale * 0.020, scale * 0.003, scale * 0.017,
                scale * -0.002, scale * 0.018),
            by=(xt.Spline4(
                scale * -0.010, scale * 0.004, scale * -0.012,
                scale * 0.001, scale * -0.011),),
            bx=(xt.Spline4(
                scale * 0.006, scale * -0.002, scale * 0.009,
                scale * 0.003, scale * 0.007),),
            length=length,
            n_steps=8,
        )

    splineboris_0 = make_splineboris(length=1.3, scale=1.0)
    splineboris_1 = make_splineboris(length=0.7, scale=-0.6)

    assert splineboris_0.has_backtrack
    assert splineboris_1.has_backtrack

    line = xt.Line(elements={
        'sb0': splineboris_0,
        'sb1': splineboris_1,
        'end': xt.Marker(),
    })
    line.particle_ref = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1.0,
        energy0=1e9,
    )
    line.build_tracker()

    tw_forward = line.twiss(
        method='4d',
        start='sb0',
        end='end',
        init_at='sb0',
        x=1.2e-3,
        px=2.0e-4,
        y=-0.8e-3,
        py=-3.0e-4,
        betx=1.0,
        bety=1.0,
    )
    tw_backtrack = line.twiss(
        method='4d',
        start='sb0',
        end='end',
        init=tw_forward,
        init_at='end',
    )

    xo.assert_allclose(
        tw_forward.s,
        [0.0, splineboris_0.length,
         splineboris_0.length + splineboris_1.length,
         splineboris_0.length + splineboris_1.length],
        atol=1e-14,
        rtol=0,
    )
    xo.assert_allclose(tw_backtrack.s, tw_forward.s, atol=1e-14, rtol=0)

    for name in ('sb0', 'sb1', 'end', '_end_point'):
        assert name in tw_forward.name
        assert name in tw_backtrack.name

    for column in ('x', 'px', 'y', 'py', 'delta'):
        xo.assert_allclose(
            tw_backtrack[column],
            tw_forward[column],
            atol=1e-12,
            rtol=0,
        )

    for column in ('betx', 'bety', 'alfx', 'alfy'):
        xo.assert_allclose(
            tw_backtrack[column],
            tw_forward[column],
            atol=1e-9,
            rtol=0,
        )

@pytest.mark.parametrize('field_angle', [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 4*np.pi/9, np.pi/7])
def test_splineboris_homogeneous_analytic(field_angle, make_uniform_splineboris):
    """
    Test the SplineBoris element with a homogeneous field, which has an analytic solution.
    Knowing the angle of the field, we can rotate our coordinates such that the field points in the y-direction.
    We then use helix geometry to calculate the end-point/angle of the particle.
    We then rotate back to the original coordinates and check this solution against the SplineBoris.
    """

    s_start = 0
    s_end = 1
    n_steps = 100

    # Field strength and orientation in the transverse plane
    B_0 = 0.1
    B_x = B_0 * np.cos(field_angle)
    B_y = B_0 * np.sin(field_angle)

    splineboris = make_uniform_splineboris(Bx=B_x, By=B_y, Bs=0, s_start=s_start, s_end=s_end, n_steps=n_steps)

    # Reference and test particle
    line = xt.Line(elements=[splineboris])
    line.particle_ref = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1.0,
        energy0=1e9,
    )

    p = line.particle_ref.copy()
    p.x = 1e-3  # 1 mm offset
    p.px = 1e-3  # small transverse momentum to create a visible helix

    # Analytic solution for the helix angle
    kin_xp = p.kin_xp[0]
    kin_yp = p.kin_yp[0]
    x = p.x[0]
    y = p.y[0]

    # Transform the coordinates to a frame where the field points in the y-direction:
    # In this frame, x' = dx/dz (where z is along the field direction)
    R = np.array([[np.sin(field_angle), -np.cos(field_angle)],
                  [np.cos(field_angle), np.sin(field_angle)]])
    x_rot, y_rot = R @ np.array([x, y])
    xp_rot, yp_rot = R @ np.array([kin_xp, kin_yp])
    
    q_C = abs(p.q0) * qe  # Coulomb
    p0_SI = p.p0c[0] * qe / clight  # (eV/c) -> kg m/s
    px_SI = p.kin_px[0] * p0_SI
    ps_SI = p.kin_ps[0] * p0_SI

    # In the rotated frame where B || y, p_perp is in (x,s)
    p_perp_SI = np.sqrt(px_SI**2 + ps_SI**2)
    rho = p_perp_SI / (q_C * B_0)

    assert rho > (s_end - s_start) * 2 / np.pi


    sqrt_term = np.sqrt(1.0 + xp_rot**2)

    # Two candidate centers (from helix geometry):
    # x_c = x0 ∓ rho/sqrt(1+xp0^2)
    # s_c = s0 ± rho*xp0/sqrt(1+xp0^2)
    x_c_plus  = x_rot - rho / sqrt_term
    s_c_plus  = s_start + rho * xp_rot / sqrt_term

    x_c_minus = x_rot + rho / sqrt_term
    s_c_minus = s_start - rho * xp_rot / sqrt_term

    def xp_from_center(xc, sc, x0, s0):
        # x' = dx/ds = -(s - s_c)/(x - x_c)
        return -(s0 - sc) / (x0 - xc)

    # Pick the center whose implied slope matches xp_rot best
    xp_pred_plus  = xp_from_center(x_c_plus,  s_c_plus,  x_rot, s_start)
    xp_pred_minus = xp_from_center(x_c_minus, s_c_minus, x_rot, s_start)

    if abs(xp_pred_plus - xp_rot) <= abs(xp_pred_minus - xp_rot):
        x_c, s_c = x_c_plus, s_c_plus
    else:
        x_c, s_c = x_c_minus, s_c_minus

    # Determine the correct branch for x(s) from the initial point (NOT from the center sign)
    x0_diff = x_rot - x_c
    s0_diff = s_start - s_c

    # Guard numerical noise
    rad0 = rho**2 - s0_diff**2
    if rad0 < -1e-15 * rho**2:
        raise ValueError(f"Initial point not on circle: rho^2-(s0-s_c)^2 = {rad0}")
    rad0 = max(0.0, rad0)
    sqrt0 = np.sqrt(rad0)

    sigma_branch = np.sign(x0_diff)
    if sigma_branch == 0:
        sigma_branch = 1.0

    # Sanity: reconstruct x0
    x0_recon = x_c + sigma_branch * sqrt0
    if abs(x0_recon - x_rot) > 1e-12:
        raise ValueError(
            f"Branch selection failed: x0_recon={x0_recon}, x0={x_rot}, "
            f"diff={x0_recon - x_rot}"
        )

    # Now compute end-point x(s_end), x'(s_end) consistently on the same branch
    s_diff = s_end - s_c
    rad_end = rho**2 - s_diff**2
    if rad_end < -1e-15 * rho**2:
        raise ValueError(f"s_end outside circle: rho^2-(s_end-s_c)^2 = {rad_end}")
    rad_end = max(0.0, rad_end)
    sqrt_end = np.sqrt(rad_end)

    x_end_rot = x_c + sigma_branch * sqrt_end
    x_diff = x_end_rot - x_c

    # x' = -(s-s_c)/(x-x_c)
    xp_end_rot = -s_diff / x_diff

    # --- Optional: robust phase change (wrapped to [-pi, pi]) ---
    initial_phase = np.arctan2(s0_diff, x0_diff)
    final_phase   = np.arctan2(s_diff,  x_diff)
    phase_change  = np.arctan2(np.sin(final_phase - initial_phase),
                            np.cos(final_phase - initial_phase))

    # y_end_rot and yp_end_rot (your existing formulas)
    y_end_rot  = y_rot + yp_rot * x0_diff * phase_change
    yp_end_rot = yp_rot * x0_diff / x_diff

    
    # Transform back to original (x, y) coordinates
    R_inv = np.linalg.inv(R)  # Inverse rotation (transpose of orthogonal matrix)
    x_final, y_final = R_inv @ np.array([x_end_rot, y_end_rot])
    xp_final, yp_final = R_inv @ np.array([xp_end_rot, yp_end_rot])
    
    # Track the particle with SplineBoris
    line.track(p)
    x_end_splineboris = p.x[0]
    y_end_splineboris = p.y[0]
    xp_final_splineboris = p.kin_xp[0]
    yp_final_splineboris = p.kin_yp[0]

    xo.assert_allclose(x_final, x_end_splineboris, atol=1e-12, rtol=1e-5)
    xo.assert_allclose(y_final, y_end_splineboris, atol=1e-12, rtol=1e-5)
    xo.assert_allclose(xp_final, xp_final_splineboris, atol=1e-12, rtol=1e-5)
    xo.assert_allclose(yp_final, yp_final_splineboris, atol=1e-12, rtol=1e-5)

@pytest.mark.parametrize('field_angle', [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 4*np.pi/9, np.pi/7])
def test_splineboris_homogeneous_rbend(field_angle, make_uniform_splineboris):
    """
    Test the SplineBoris element with a homogeneous field against the RBend.
    """

    s_start = 0
    s_end = 1
    length = s_end - s_start
    n_steps = 100

    # Field strength and orientation in the transverse plane
    B_0 = 0.1
    B_x = B_0 * np.cos(field_angle)
    B_y = B_0 * np.sin(field_angle)

    splineboris = make_uniform_splineboris(Bx=B_x, By=B_y, Bs=0, s_start=s_start, s_end=s_end, n_steps=n_steps)

    # Reference and test particle
    line_splineboris = xt.Line(elements=[splineboris])
    line_splineboris.particle_ref = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1.0,
        energy0=1e9,
    )

    p_splineboris = line_splineboris.particle_ref.copy()
    p_splineboris.x = 1e-3
    p_splineboris.px = 1e-3

    p_rbend = p_splineboris.copy()
    p_rbend.x = 1e-3
    p_rbend.px = 1e-3

    k0 =  B_0 * clight / p_rbend.p0c[0]

    edge_model = 'suppressed'       # Ignore the edge effects
    rot_s_rad = field_angle-np.pi/2 # For field_angle = 0, the field is in the x-direction. To have the bend reflect this, we need to rotate the coordinate system in the opposite direction.
    b_rbend = xt.RBend(k0=k0, k0_from_h=False, length_straight=length, angle=0, edge_entry_angle=0, edge_exit_angle=0, rot_s_rad=rot_s_rad)
    b_rbend.edge_entry_model = edge_model
    b_rbend.edge_exit_model = edge_model
    b_rbend.model = 'bend-kick-bend'

    line_rbend = xt.Line(elements=[b_rbend])
    line_rbend.particle_ref = p_rbend

    line_rbend.track(p_rbend)
    x_end_rbend = p_rbend.x[0]
    y_end_rbend = p_rbend.y[0]
    px_end_rbend = p_rbend.kin_px[0]
    py_end_rbend = p_rbend.kin_py[0]

    # Track the particle with SplineBoris
    line_splineboris.track(p_splineboris)
    x_end_splineboris = p_splineboris.x[0]
    y_end_splineboris = p_splineboris.y[0]
    px_final_splineboris = p_splineboris.kin_px[0]
    py_final_splineboris = p_splineboris.kin_py[0]

    xo.assert_allclose(x_end_rbend, x_end_splineboris, atol=1e-12, rtol=1e-5)
    xo.assert_allclose(y_end_rbend, y_end_splineboris, atol=1e-12, rtol=1e-5)
    xo.assert_allclose(px_end_rbend, px_final_splineboris, atol=1e-12, rtol=1e-5)
    xo.assert_allclose(py_end_rbend, py_final_splineboris, atol=1e-12, rtol=1e-5)

def test_splineboris_undulator_vs_boris_spatial(undulator_fit, make_segment_field):
    """
    Build the undulator from the fitted SLS map and check that tracking with
    SplineBoris and BorisSpatialIntegrator gives consistent end coordinates.
    """
    multipole_order = UNDULATOR_MULTIPOLE_ORDER
    line_spline = undulator_fit.to_line(multipole_order=multipole_order)

    # This undulator is part of the SLS, so we use the nominal energy of the SLS.
    p_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)
    line_spline.particle_ref = p_ref.copy()

    p_spline = line_spline.particle_ref.copy()
    p_spline.x = 1e-3
    p_spline.px = 1e-4
    p_spline.y = 0.5e-3
    p_spline.py = -0.5e-4

    line_spline.track(p_spline)

    # ------------------------------------------------------------------
    # Build a parallel undulator line using BorisSpatialIntegrator
    # from the same elements' Hermite parameters
    # ------------------------------------------------------------------
    boris_elems = []
    nodes = undulator_fit.nodes
    for elem, s_start, s_end in zip(line_spline.elements, nodes[:-1], nodes[1:]):
        bs = [elem.bs[i] for i in range(5)]
        by = [
            [elem.by[i, j] for j in range(5)]
            for i in range(multipole_order)
        ]
        bx = [
            [elem.bx[i, j] for j in range(5)]
            for i in range(multipole_order)
        ]
        L = float(elem.length)
        field_i = make_segment_field(
            bs,
            by,
            bx,
            L,
            multipole_order,
            s_start=float(s_start),
        )

        boris_elems.append(
            xt.BorisSpatialIntegrator(
                fieldmap_callable=field_i,
                s_start=float(s_start),
                s_end=float(s_end),
                n_steps=int(elem.n_steps),
            )
        )

    line_boris = xt.Line(elements=boris_elems)
    line_boris.particle_ref = p_ref.copy()

    p_boris = line_boris.particle_ref.copy()
    p_boris.x = 1e-3
    p_boris.px = 1e-4
    p_boris.y = 0.5e-3
    p_boris.py = -0.5e-4

    line_boris.track(p_boris)

    # ------------------------------------------------------------------
    # Compare end coordinates
    # ------------------------------------------------------------------
    xo.assert_allclose(p_spline.x, p_boris.x, rtol=1e-12, atol=5e-11)
    xo.assert_allclose(p_spline.px, p_boris.px, rtol=1e-12, atol=5e-11)
    xo.assert_allclose(p_spline.y, p_boris.y, rtol=1e-12, atol=5e-11)
    xo.assert_allclose(p_spline.py, p_boris.py, rtol=1e-12, atol=5e-11)
    xo.assert_allclose(p_spline.zeta, p_boris.zeta, rtol=1e-12, atol=5e-11)
    xo.assert_allclose(p_spline.delta, p_boris.delta, rtol=1e-12, atol=5e-11)

def test_undulator_rotated_fit_obeys_rotation_rule(undulator_raw_data, undulator_fit):
    """
    Rotate the field map by 90 degrees about s, (x, y) -> (y, -x), and check
    that the fitted on-axis components obey the rotation rule:
    Bx_rotated == By_original, By_rotated == -Bx_original,
    Bs_rotated == Bs_original. The tube basis (all p + q <= M) is closed
    under this rotation, so the rule holds to rounding.
    """
    df = undulator_raw_data.reset_index()
    df_rot = pd.DataFrame({
        "X": df["Y"], "Y": -df["X"], "Z": df["Z"],
        "Bx": df["By"], "By": -df["Bx"], "Bs": df["Bs"],
    }).set_index(["X", "Y", "Z"])
    lf_rot = _fit_undulator(df_rot)

    orig = undulator_fit
    scale = np.max(np.abs(orig.element_params(("By", 0))))
    for rot_name, orig_name, sign in ((("Bx", 0), ("By", 0), 1),
                                      (("By", 0), ("Bx", 0), -1),
                                      (("Bs", 0), ("Bs", 0), 1)):
        xo.assert_allclose(lf_rot.element_params(rot_name),
                           sign * orig.element_params(orig_name),
                           rtol=0, atol=1e-10 * scale)

@fix_random_seed(645284)
def test_splineboris_bend_radiation(make_uniform_splineboris):
    """
    Test synchrotron radiation in SplineBoris element.

    This test creates a SplineBoris element with a uniform dipole field (constant By),
    tracks particles with both average and quantum radiation models, and compares
    the energy loss against theoretical predictions from the Larmor formula.
    """
    # Dipole parameters
    L_bend = 1.0  # [m]
    B_T = 2.0     # [T] - dipole field strength

    # Create test particles (5 GeV electrons)
    n_particles = 100000
    particles_mean = xt.Particles(
        p0c=5e9,  # 5 GeV
        x=np.zeros(n_particles),
        px=1e-4,
        py=-1e-4,
        delta=0,
        mass0=xt.ELECTRON_MASS_EV,
    )

    particles_mean_0 = particles_mean.copy()
    gamma = (particles_mean.energy / particles_mean.mass0)[0]
    gamma0 = particles_mean.gamma0[0]
    particles_qntm_0 = particles_mean.copy()
    particles_qntm_kick_0 = particles_mean.copy()

    # Calculate bend angle from field
    P0_J = particles_mean.p0c[0] / clight * qe
    h_bend = B_T * qe / P0_J
    theta_bend = h_bend * L_bend
    rho_0 = L_bend / theta_bend  # bending radius

    # Create SplineBoris element with uniform By field (dipole)
    # For a dipole, we need constant By = B_T
    s_start = 0.0
    s_end = L_bend
    n_steps = 100

    # Create SplineBoris elements with radiation
    splineboris_mean = make_uniform_splineboris(Bx=0, By=B_T, Bs=0, s_start=s_start, s_end=s_end, n_steps=n_steps, radiation_flag=1)
    splineboris_qntm = make_uniform_splineboris(Bx=0, By=B_T, Bs=0, s_start=s_start, s_end=s_end, n_steps=n_steps, radiation_flag=2)
    splineboris_qntm_kick = make_uniform_splineboris(Bx=0, By=B_T, Bs=0, s_start=s_start, s_end=s_end, n_steps=n_steps, radiation_flag=3)

    # Initialize random number generators
    particles_mean_0._init_random_number_generator()
    particles_qntm_0._init_random_number_generator()
    particles_qntm_kick_0._init_random_number_generator()

    dct_mean_before = particles_mean_0.to_dict()

    # Track particles
    splineboris_mean.track(particles_mean_0)
    splineboris_qntm.track(particles_qntm_0)
    splineboris_qntm_kick.track(particles_qntm_kick_0)

    dct_mean = particles_mean_0.to_dict()
    dct_qntm = particles_qntm_0.to_dict()
    dct_qntm_kick = particles_qntm_kick_0.to_dict()

    # Test 1: Average and stochastic models should give same mean energy loss
    xo.assert_allclose(dct_mean['delta'], np.mean(dct_qntm['delta']),
                       atol=0, rtol=5e-3)
    xo.assert_allclose(dct_mean['delta'], np.mean(dct_qntm_kick['delta']),
                       atol=0, rtol=5e-3)

    # Test 2: Compare energy loss against Larmor formula
    mass0_kg = dct_mean['mass0'] * qe / clight**2
    r0 = qe**2 / (4 * np.pi * epsilon_0 * mass0_kg * clight**2)
    Ps = (2 * r0 * clight * mass0_kg * clight**2 * gamma0**2 * gamma**2) / (3 * rho_0**2)  # [W]

    Delta_E_eV = -Ps * (L_bend / clight) / qe  # Theoretical energy loss
    Delta_E_qntm = (dct_mean['ptau'] - dct_mean_before['ptau']) * dct_mean['p0c']  # Tracked energy loss

    # Allow ~0.5% tolerance due to integration steps
    xo.assert_allclose(Delta_E_eV, np.mean(Delta_E_qntm), atol=0, rtol=5e-3)

    # Test 3: Check photon statistics using internal logging
    line = xt.Line(elements=[
        xt.Drift(length=1.0),
        make_uniform_splineboris(Bx=0.0, By=B_T, Bs=0.0, s_start=s_start, s_end=s_end, n_steps=n_steps),
        xt.Drift(length=1.0),
        make_uniform_splineboris(Bx=0.0, By=B_T, Bs=0.0, s_start=s_start, s_end=s_end, n_steps=n_steps),
    ])
    
    line.build_tracker()
    line.configure_radiation(model='quantum')

    sum_photon_energy = 0
    sum_photon_energy_sq = 0
    tot_n_recorded = 0

    for _ in range(10):
        record_capacity = int(10e6)
        record = line.start_internal_logging_for_elements_of_type(
            xt.SplineBoris, capacity=record_capacity
        )
        particles_test = particles_mean_0.copy()
        particles_test_before = particles_test.copy()
        line.track(particles_test)

        Delta_E_test = (particles_test.ptau - particles_test_before.ptau) * particles_test.p0c
        n_recorded = record._index.num_recorded
        assert n_recorded < record_capacity

        # Verify energy conservation: particle energy loss = photon energy
        xo.assert_allclose(
            -np.sum(Delta_E_test),
            np.sum(record.photon_energy[:n_recorded]),
            atol=0, rtol=1e-6,
        )

        sum_photon_energy += np.sum(record.photon_energy[:n_recorded])
        sum_photon_energy_sq += np.sum(record.photon_energy[:n_recorded]**2)
        tot_n_recorded += n_recorded

    # Compute theoretical photon statistics
    p0_J = particles_mean_0.p0c[0] / clight * qe
    B_T_actual = p0_J / qe / rho_0
    mass_0_kg = particles_mean_0.mass0 * qe / clight**2
    E_crit_J = 3 * qe * hbar * gamma**2 * B_T_actual / (2 * mass_0_kg)

    E_ave_J = 8 * np.sqrt(3) / 45 * E_crit_J
    E_ave_eV = E_ave_J / qe

    E_sq_ave_J = 11 / 27 * E_crit_J**2
    E_sq_ave_eV = E_sq_ave_J / qe**2

    mean_photon_energy = sum_photon_energy / tot_n_recorded
    mean_photon_energy_sq = sum_photon_energy_sq / tot_n_recorded
    std_photon_energy = np.sqrt(mean_photon_energy_sq - mean_photon_energy**2)

    xo.assert_allclose(mean_photon_energy, E_ave_eV, rtol=1e-2, atol=0)
    xo.assert_allclose(std_photon_energy, np.sqrt(E_sq_ave_eV - E_ave_eV**2), rtol=2e-3, atol=0)

    line.configure_radiation(model='quantum-kick')
    record = line.start_internal_logging_for_elements_of_type(
        xt.SplineBoris, capacity=record_capacity
    )
    particles_test = particles_mean_0.copy()
    particles_test_before = particles_test.copy()
    line.track(particles_test)

    Delta_E_test = (particles_test.ptau - particles_test_before.ptau) * particles_test.p0c
    assert -np.sum(Delta_E_test) > 0
    assert record._index.num_recorded == 0

def test_splineboris_variable_solenoid_radiation(solenoid_field, solenoid_fit):

    delta=np.array([0, 4])
    p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                    energy0=45.6e9,
                    x=[-5e-3, -5e-3], px=-1e-3*(1+delta), y=5e-3,
                    delta=delta)

    sf = solenoid_field

    # --- SplineBoris tracking ---
    line_boris = solenoid_fit.to_line(multipole_order=SOLENOID_MULTIPOLE_ORDER)
    line_boris.build_tracker()
    line_boris.configure_radiation(model='mean')

    p_boris = p0.copy()
    line_boris.track(p_boris, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_boris = line_boris.record_last_track

    # --- VariableSolenoid reference ---
    z_axis = np.linspace(0, SOLENOID_INTERVAL, SOLENOID_Z_POINT_COUNT)
    Bz_axis = sf.get_field(0 * z_axis, 0 * z_axis, z_axis)[2]

    P0_J = p0.p0c[0] * qe / clight
    brho = P0_J / qe / p0.q0

    ks = Bz_axis / brho
    ks_entry = ks[:-1]
    ks_exit = ks[1:]

    dz = z_axis[1]-z_axis[0]

    line_varsol = xt.Line(elements=[xt.VariableSolenoid(length=dz,
                                        ks_profile=[ks_entry[ii], ks_exit[ii]])
                                for ii in range(len(z_axis)-1)])
    line_varsol.build_tracker()
    line_varsol.configure_radiation(model='mean')

    p_xt = p0.copy()
    line_varsol.track(p_xt, turn_by_turn_monitor='ONE_TURN_EBE')
    mon = line_varsol.record_last_track

    p_xt = p0.copy()
    line_varsol.configure_radiation(model=None)
    line_varsol.track(p_xt, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_no_rad = line_varsol.record_last_track

    Bz_mid = 0.5 * (Bz_axis[:-1] + Bz_axis[1:])
    Bz_mon = 0 * Bz_axis
    Bz_mon[1:] = Bz_mid

    # Wolsky Eq. 3.114
    Ax = -0.5 * Bz_mon * mon.y
    Ay =  0.5 * Bz_mon * mon.x

    # Wolsky Eq. 2.74
    ax_ref = Ax * p0.q0 * qe / P0_J
    ay_ref = Ay * p0.q0 * qe / P0_J

    dx_ds = np.diff(mon.x, axis=1) / np.diff(mon.s, axis=1)
    dy_ds = np.diff(mon.y, axis=1) / np.diff(mon.s, axis=1)

    dE_ds = 0*mon.ptau
    # Central differences
    dE_ds[:, 1:-1] = -((mon.ptau[:, 2:] - mon.ptau[:, :-2]) / (mon.s[:, 2:] - mon.s[:, :-2])
                            * p_xt.energy0[0])

    emitted_dpx = -(np.diff(mon.kin_px, axis=1) - np.diff(mon_no_rad.kin_px, axis=1))
    emitted_dpy = -(np.diff(mon.kin_py, axis=1) - np.diff(mon_no_rad.kin_py, axis=1))

    # --- Comparisons ---
    for i_part in range(len(delta)):

        # SplineBoris vs VariableSolenoid: positions and energy loss at
        # the SplineBoris element boundaries (where its monitor records),
        # with the fine VariableSolenoid track interpolated there.
        # Derivatives from finite differences would only resolve the
        # element length.
        s_boris = mon_boris.s[i_part, :]
        e_loss_boris = -(mon_boris.ptau[i_part, :] - mon_boris.ptau[i_part, 0]) * p_boris.energy0[0]
        e_loss_xsuite = -(mon.ptau[i_part, :] - mon.ptau[i_part, 0]) * p_xt.energy0[0]
        for boris, xsuite, tol in ((mon_boris.x[i_part, :], mon.x[i_part, :], 2.8e-2),
                                   (mon_boris.y[i_part, :], mon.y[i_part, :], 2.8e-2),
                                   (e_loss_boris, e_loss_xsuite, 2.5e-2)):
            xo.assert_allclose(np.interp(s_boris, mon.s[i_part, :], xsuite), boris,
                               rtol=0, atol=tol * np.ptp(xsuite))

        this_emitted_dpx = emitted_dpx[i_part, :]
        this_emitted_dpy = emitted_dpy[i_part, :]
        this_dE_ds = dE_ds[i_part, :]
        this_dx_ds = dx_ds[i_part, :]
        this_dy_ds = dy_ds[i_part, :]

        xo.assert_allclose(ax_ref[i_part, :], mon.ax[i_part, :],
                        rtol=0, atol=np.max(np.abs(ax_ref)*3e-2))
        xo.assert_allclose(ay_ref[i_part, :], mon.ay[i_part, :],
                        rtol=0, atol=np.max(np.abs(ay_ref)*3e-2))

        xo.assert_allclose(this_emitted_dpx,
                0.5 * (this_dE_ds[:-1] + this_dE_ds[1:]) * this_dx_ds * np.diff(mon.s[i_part, :])/p0.p0c[0],
                rtol=0, atol=2e-2 * (np.max(this_emitted_dpx) - np.min(this_emitted_dpx)))
        xo.assert_allclose(this_emitted_dpy,
                0.5 * (this_dE_ds[:-1] + this_dE_ds[1:]) * this_dy_ds * np.diff(mon.s[i_part, :])/p0.p0c[0],
                rtol=0, atol=5e-2 * (np.max(this_emitted_dpy) - np.min(this_emitted_dpy)))



# Use the same test cases as in test_spin.py
COMMON_TEST_CASES = [
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'base'
    },
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': -0.01,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'delta=-0.01'
    },
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': -0.005,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'delta=-0.005'
    },
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': 0,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'delta=0'
    },
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': 0.005,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'delta=0.005'
    },
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': 0.01,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'delta=0.01'
    },
    {
        'case': {
            'x': 0.001,
            'px': -0.03,
            'y': 0.002,
            'py': -0.02,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'px=-0.03, py=-0.02'
    },
    {
        'case': {
            'x': 0.001,
            'px': -0.015,
            'y': 0.002,
            'py': -0.01,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'px=-0.015, py=-0.01'
    },
    {
        'case': {
            'x': 0.001,
            'px': 0,
            'y': 0.002,
            'py': 0,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'px=0, py=0'
    },
    {
        'case': {
            'x': 0.001,
            'px': 0.015,
            'y': 0.002,
            'py': 0.01,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'px=0.015, py=0.01'
    },
    {
        'case': {
            'x': 0.001,
            'px': 0.03,
            'y': 0.002,
            'py': 0.02,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'px=0.03, py=0.02'
    }
]

@pytest.mark.parametrize(
    'case,atol',
    list(zip(
        [case['case'].copy() for case in COMMON_TEST_CASES],
        [3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 2e-5, 1e-5, 2e-8, 1e-5, 2e-5],
    )),
    ids=[case['id'] for case in COMMON_TEST_CASES],
)
def test_splineboris_spin_uniform_solenoid(case, atol, make_uniform_splineboris):
    case['spin_y'] = np.sqrt(1 - case['spin_x']**2 - case['spin_z']**2)

    p = xt.Particles(
        p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128,
        **case,
    )
    p_ref = p.copy()

    Bz_T = 0.05
    ks = Bz_T / (p.p0c[0] / clight / p.q0)

    length = 0.02
    s_start = 0
    s_end = length
    n_steps = 1000

    # --- xsuite UniformSolenoid reference ---
    env = xt.Environment()
    line_ref = env.new_line(
        components=[
            env.new('mysolenoid', xt.UniformSolenoid, length=length, ks=ks),
            env.new('mymarker', xt.Marker),
        ]
    )
    line_ref.configure_spin(spin_model='auto')
    line_ref.track(p_ref)

    splineboris = make_uniform_splineboris(Bx=0, By=0, Bs=Bz_T, s_start=s_start, s_end=s_end, n_steps=n_steps)

    line_splineboris = xt.Line(elements=[splineboris])
    line_splineboris.particle_ref = p.copy()

    line_splineboris.configure_spin(spin_model='auto')

    line_splineboris.track(p)

    xo.assert_allclose(p.spin_x[0], p_ref.spin_x[0], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_y[0], p_ref.spin_y[0], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_z[0], p_ref.spin_z[0], atol=atol, rtol=0)

def test_splineboris_spin_multipole_dipole_component():
    spin_x = 0.1
    spin_z = 0.2
    spin_y = np.sqrt(1 - spin_x**2 - spin_z**2)

    p = xt.Particles(
        p0c=700e9,
        mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128,
        x=1e-3,
        px=1e-5,
        y=2e-3,
        py=2e-5,
        delta=1e-3,
        spin_x=spin_x,
        spin_y=spin_y,
        spin_z=spin_z,
    )
    p_ref = p.copy()

    length = 0.02
    k0 = 0.01
    n_steps = 100

    line_ref = xt.Line(elements=[
        xt.Bend(length=length, angle=0.0, k0=k0),
        xt.Marker(),
    ])
    line_ref.configure_spin(spin_model='auto')
    line_ref.track(p_ref)

    line_splineboris = xt.Line(elements=[
        xt.SplineBoris(
            length=length,
            n_steps=n_steps,
            knl=[k0 * length],
        )
    ])
    line_splineboris.particle_ref = p.copy()
    line_splineboris.configure_spin(spin_model='auto')
    line_splineboris.track(p)

    xo.assert_allclose(p.spin_x[0], p_ref.spin_x[0], atol=3e-8, rtol=0)
    xo.assert_allclose(p.spin_y[0], p_ref.spin_y[0], atol=3e-8, rtol=0)
    xo.assert_allclose(p.spin_z[0], p_ref.spin_z[0], atol=3e-8, rtol=0)

@pytest.mark.parametrize(
    'case,atol',
    list(zip(
        [case['case'].copy() for case in COMMON_TEST_CASES],
        [6e-8, 6e-8, 6e-8, 6e-8, 6e-8, 6e-8, 6e-5, 3e-5, 2e-7, 3e-5, 6e-5],
    )),
    ids=[case['id'] for case in COMMON_TEST_CASES],
)
def test_splineboris_spin_quadrupole(case, atol):
    case['spin_y'] = np.sqrt(1 - case['spin_x']**2 - case['spin_z']**2)

    p = xt.Particles(
        p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128,
        **case,
    )
    p_ref = p.copy()

    k1 = 0.01
    quad_gradient = k1 * p.p0c[0] / clight / p.q0

    length = 0.02
    s_start = 0
    s_end = length
    n_steps = 100

    # --- xsuite Quadrupole reference ---
    env = xt.Environment()
    line_ref = env.new_line(
        components=[
            env.new('myquad', xt.Quadrupole, k1=k1, length=length),
            env.new('mymarker', xt.Marker),
        ]
    )
    line_ref.configure_spin(spin_model='auto')
    line_ref.track(p_ref)

    # --- SplineBoris ---
    # Uniform quadrupole: Hermite params (f_left, df_left, f_right, df_right, average)
    kn_1_hermite = [quad_gradient, 0, quad_gradient, 0, quad_gradient]
    bs = [0, 0, 0, 0, 0]

    # Verify the polynomial evaluates to a constant gradient.
    # hermite_to_polynomial returns a poly in local coordinate s_local = s - s_start.
    from xtrack.beam_elements.splineboris_src.spline_B_field_eval_python import hermite_to_polynomial
    kn_1_poly = hermite_to_polynomial(s_start, s_end, kn_1_hermite)
    s_test = np.linspace(s_start, s_end, 100)
    xo.assert_allclose(kn_1_poly(s_test - s_start), quad_gradient, rtol=1e-12, atol=1e-12)

    splineboris = xt.SplineBoris(
        bs=xt.Spline4(*bs),
        by=(None, xt.Spline4(*kn_1_hermite)),
        length=s_end - s_start,
        n_steps=n_steps,
    )

    # Reference and test particle
    line_splineboris = xt.Line(elements=[splineboris])
    line_splineboris.particle_ref = p.copy()

    line_splineboris.configure_spin(spin_model='auto')

    line_splineboris.track(p)

    xo.assert_allclose(p.spin_x[0], p_ref.spin_x[0], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_y[0], p_ref.spin_y[0], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_z[0], p_ref.spin_z[0], atol=atol, rtol=0)

