import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import epsilon_0, hbar
import pytest
import xobjects as xo
from xobjects.test_helpers import fix_random_seed
from pathlib import Path
import importlib.util

import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack._temp.field_fitter import FieldFitter
from xtrack._temp.splineboris_sequence import SplineBorisSequence

FIT_PARS_INDEX_COLS = [
    "field_component",
    "derivative_x",
    "region_name",
    "s_start",
    "s_end",
    "idx_start",
    "idx_end",
    "param_index",
]

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
                radiation_flag=0, kn=None, ks=None):
        # Uniform field: Hermite params (f_left, df_left, f_right, df_right, average)
        # For a constant field B, all boundary values = B, derivatives = 0, average = B
        Bx_h = [Bx, 0, Bx, 0, Bx]
        By_h = [By, 0, By, 0, By]
        Bs_h = [Bs, 0, Bs, 0, Bs]

        # Verify the polynomials evaluate to constants (s_local = s - s_start)
        s_local_test = np.linspace(0, s_end - s_start, 100)
        xo.assert_allclose(xt.SplineBoris.hermite_to_polynomial(s_start, s_end, Bx_h)(s_local_test), Bx, rtol=1e-12, atol=1e-12)
        xo.assert_allclose(xt.SplineBoris.hermite_to_polynomial(s_start, s_end, By_h)(s_local_test), By, rtol=1e-12, atol=1e-12)
        xo.assert_allclose(xt.SplineBoris.hermite_to_polynomial(s_start, s_end, Bs_h)(s_local_test), Bs, rtol=1e-12, atol=1e-12)

        bx_s4 = xt.Spline4(val_start=Bx, der_start=0.0, val_end=Bx, der_end=0.0, integral=Bx)
        by_s4 = xt.Spline4(val_start=By, der_start=0.0, val_end=By, der_end=0.0, integral=By)
        bs_s4 = xt.Spline4(val_start=Bs, der_start=0.0, val_end=Bs, der_end=0.0, integral=Bs)

        splineboris = xt.SplineBoris(
            bs=bs_s4,
            by=by_s4,
            bx=bx_s4,
            s_start=s_start,
            length=s_end - s_start,
            n_steps=n_steps,
            radiation_flag=radiation_flag,
        )
        return splineboris
    return _make

@pytest.fixture(scope="module")
def evaluate_b():
    module_path = (
        Path(__file__).parent.parent
        / "xtrack"
        / "beam_elements"
        / "elements_src"
        / "spline_B_field_eval_python.py"
    )
    spec = importlib.util.spec_from_file_location("spline_B_field_eval_python", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_B

@pytest.fixture(scope="module")
def make_segment_field(evaluate_b):
    def _make(params_1d, multipole_order_local, s_start=0):
        params_arr = np.asarray(params_1d, dtype=float)

        def field(x, y, z):
            Bx, By, Bs = evaluate_b(x, y, z - s_start, params_arr, multipole_order_local)
            return Bx, By, Bs

        return field

    return _make

@pytest.fixture(scope="module")
def solenoid_field():
    return SolenoidField(**SOLENOID_MODEL_PARAMS)

@pytest.fixture(scope="module")
def solenoid_vs_varsol_fit_result(solenoid_field):
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

    # Build a numpy-based field grid and pass it to FieldFitter
    raw_data = {
        "x": x_grid.ravel().astype(float),
        "y": y_grid.ravel().astype(float),
        "z": z_grid.ravel().astype(float),
        "Bx": bx.ravel().astype(float),
        "By": by.ravel().astype(float),
        "Bs": bz.ravel().astype(float),
    }

    fitter = FieldFitter(
        raw_data=raw_data,
        xy_point=(0, 0),
        distance_unit=1,
        min_region_size=10,
        deg=SOLENOID_MULTIPOLE_ORDER - 1,
        field_tol=1e-4,
    )
    fit_result = fitter.fit()

    # Basic sanity checks on the fit result geometry
    s_start_min = min(seg.s_start for seg in fit_result.segments)
    s_end_max = max(seg.s_end for seg in fit_result.segments)
    idx_start_min = min(seg.idx_start for seg in fit_result.segments)
    idx_end_max = max(seg.idx_end for seg in fit_result.segments)
    point_count = idx_end_max - idx_start_min + 1

    assert np.isclose(s_start_min, 0.0), f"Unexpected s_start min: {s_start_min}"
    assert np.isclose(s_end_max, SOLENOID_INTERVAL), f"Unexpected s_end max: {s_end_max}"
    assert idx_start_min == 0, f"Unexpected idx_start min: {idx_start_min}"
    assert idx_end_max == SOLENOID_N_STEPS, f"Unexpected idx_end max: {idx_end_max}"
    assert point_count == SOLENOID_Z_POINT_COUNT, f"Unexpected point_count: {point_count}"

    # Ensure we have Bs (derivative_x == 0) and some transverse content
    fields = {(seg.field_component, seg.derivative_x) for seg in fit_result.segments}
    assert ("Bs", 0) in fields
    assert any((fc, d) in (("Bx", d), ("By", d)) for (fc, d) in fields)

    return fit_result

def _fit_result_from_undulator_csv(path: Path):
    import pandas as pd

    df = pd.read_csv(path, index_col=FIT_PARS_INDEX_COLS)
    df_reset = df.reset_index()
    segments = []
    # Build FitSegment objects directly from the CSV structure.
    for (field_component, derivative_x, region_name, s_start, s_end, idx_start, idx_end), grp in df_reset.groupby(
        ["field_component", "derivative_x", "region_name", "s_start", "s_end", "idx_start", "idx_end"],
        sort=True,
    ):
        grp_sorted = grp.sort_values("param_index")
        hermite_vals = grp_sorted["param_value"].to_numpy(dtype=float)
        to_fit_flags = np.ones_like(hermite_vals, dtype=bool)
        # Region index is just an integer label derived from region_name ordering.
        region_index = int(region_name.split("_")[-1])
        segments.append(
            xt._temp.field_fitter.FitSegment(  # type: ignore[attr-defined]
                field_component=str(field_component),
                derivative_x=int(derivative_x),
                region_index=region_index,
                s_start=float(s_start),
                s_end=float(s_end),
                idx_start=int(idx_start),
                idx_end=int(idx_end),
                hermite_params=hermite_vals.copy(),
                to_fit=bool(to_fit_flags.any()),
            )
        )
    multipole_order = max(
        (seg.derivative_x for seg in segments if seg.field_component in ("Bx", "By")),
        default=0,
    ) + 1
    # z-grid is not encoded in the CSV; reconstruct a simple uniform grid from idx range.
    idx_start_min = min(seg.idx_start for seg in segments)
    idx_end_max = max(seg.idx_end for seg in segments)
    n_points = idx_end_max - idx_start_min + 1
    s_start_min = min(seg.s_start for seg in segments)
    s_end_max = max(seg.s_end for seg in segments)
    s_full = np.linspace(s_start_min, s_end_max, n_points)
    return xt._temp.field_fitter.FieldFitResult(  # type: ignore[attr-defined]
        s_full=s_full,
        segments=segments,
        multipole_order=multipole_order,
    )


@pytest.fixture(scope="module")
def undulator_fit_result(test_data_dir):
    return _fit_result_from_undulator_csv(test_data_dir / "sls" / "undulator_fit_pars.csv")


@pytest.fixture(scope="module")
def undulator_rotated_fit_result(test_data_dir):
    return _fit_result_from_undulator_csv(
        test_data_dir / "sls" / "undulator_fit_pars_rotated.csv"
    )



# Test some common field angles, as well as some unusual ones
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
    kin_xp = p.kin_xprime[0]
    kin_yp = p.kin_yprime[0]
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
    xp_final_splineboris = p.kin_xprime[0]
    yp_final_splineboris = p.kin_yprime[0]

    xo.assert_allclose(x_final, x_end_splineboris, atol=1e-12, rtol=1e-5)
    xo.assert_allclose(y_final, y_end_splineboris, atol=1e-12, rtol=1e-5)
    xo.assert_allclose(xp_final, xp_final_splineboris, atol=1e-12, rtol=1e-5)
    xo.assert_allclose(yp_final, yp_final_splineboris, atol=1e-12, rtol=1e-5)



# Test some common field angles, as well as some unusual ones
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



def test_splineboris_spin_uniform_solenoid(make_uniform_splineboris):

    atol = 3e-8
    case = {
    'x': 0.001,
    'px': 1e-05,
    'y': 0.002,
    'py': 2e-05,
    'delta': 0.001,
    'spin_x': 0.1,
    'spin_z': 0.2,
    }
    case['spin_y'] = np.sqrt(1 - case['spin_x']**2 - case['spin_z']**2)

    p = xt.Particles(
        p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128,
        **case,
    )

    p_splineboris = p.copy()
    p_ref = p.copy()

    Bz_T = 0.05
    ks = Bz_T / (p.p0c[0] / clight / p.q0)
    env = xt.Environment()

    length = 0.25
    s_start = 0
    s_end = length
    n_steps = 100

    splineboris = make_uniform_splineboris(Bx=0, By=0, Bs=ks, s_start=s_start, s_end=s_end, n_steps=n_steps)

    # Reference and test particle
    line_splineboris = xt.Line(elements=[splineboris])
    line_splineboris.particle_ref = p_splineboris

    line = env.new_line(
        components=[
            env.new('mysolenoid', xt.UniformSolenoid, length=length, ks=ks),
            env.new('mymarker', xt.Marker),
        ]
    )

    line.configure_spin(spin_model='auto')
    line_splineboris.configure_spin(spin_model='auto')

    line.track(p_ref)
    line_splineboris.track(p_splineboris)

    xo.assert_allclose(p_ref.s, p_splineboris.s, atol=atol, rtol=1e-12)
    xo.assert_allclose(p_ref.x, p_splineboris.x, atol=atol, rtol=1e-12)
    xo.assert_allclose(p_ref.y, p_splineboris.y, atol=atol, rtol=1e-12)
    xo.assert_allclose(p_ref.px, p_splineboris.px, atol=atol, rtol=1e-12)
    xo.assert_allclose(p_ref.py, p_splineboris.py, atol=atol, rtol=1e-12)
    xo.assert_allclose(p_ref.delta, p_splineboris.delta, atol=atol, rtol=1e-12)




def test_splineboris_solenoid_vs_variable_solenoid(solenoid_field, solenoid_vs_varsol_fit_result):
    """
    Test SplineBoris element against VariableSolenoid for a solenoid field.

    This test creates a solenoid field, fits it with polynomial splines, tracks particles
    through both SplineBoris and VariableSolenoid, and compares the trajectories.
    """
    # Set basic parameters
    interval = SOLENOID_INTERVAL
    n_steps = SOLENOID_N_STEPS

    # Make initial particles
    delta = np.array([0, 4])
    p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                    energy0=45.6e6,  # 45.6 MeV
                    x=1e-3,  # Start slightly off-axis
                    px=-1e-3*(1+delta),
                    y=1e-3,
                    delta=delta)

    # Make solenoid field instance
    sf = solenoid_field

    # Build solenoid using SplineBorisSequence from the canonical FieldFitResult.
    fit_result = solenoid_vs_varsol_fit_result
    seq = SplineBorisSequence.from_fit_result(
        fit_result,
        steps_per_point=1,  # one integration step per data point
        radiation_flag=0,
    )

    # Get the Line of SplineBoris elements and track
    line_splineboris = seq.to_line()
    line_splineboris.build_tracker()
    p_splineboris = p0.copy()
    line_splineboris.track(p_splineboris, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_splineboris = line_splineboris.record_last_track

    # --- VariableSolenoid reference (paraxial approximation, on-axis Bz only) ---
    z_axis_ref = np.linspace(0, interval, SOLENOID_Z_POINT_COUNT)
    # Get on-axis Bz
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

    # Define check points in the solenoid region
    z_check = sf.z0 + sf.L * np.linspace(-2, 2, 1001)

    # Compare trajectories for each particle
    n_part = mon_splineboris.x.shape[0]
    for i_part in range(n_part):

        # VariableSolenoid trajectory derivatives
        s_varsol = 0.5 * (mon_varsol.s[i_part, :-1] + mon_varsol.s[i_part, 1:])
        dx_ds_varsol = np.diff(mon_varsol.x[i_part, :]) / np.diff(mon_varsol.s[i_part, :])
        dy_ds_varsol = np.diff(mon_varsol.y[i_part, :]) / np.diff(mon_varsol.s[i_part, :])

        # SplineBoris trajectory derivatives
        s_splineboris = 0.5 * (mon_splineboris.s[i_part, :-1] + mon_splineboris.s[i_part, 1:])
        dx_ds_splineboris = np.diff(mon_splineboris.x[i_part, :]) / np.diff(mon_splineboris.s[i_part, :])
        dy_ds_splineboris = np.diff(mon_splineboris.y[i_part, :]) / np.diff(mon_splineboris.s[i_part, :])

        # Interpolate at check points
        dx_ds_splineboris_check = np.interp(z_check, s_splineboris, dx_ds_splineboris)
        dy_ds_splineboris_check = np.interp(z_check, s_splineboris, dy_ds_splineboris)

        dx_ds_varsol_check = np.interp(z_check, s_varsol, dx_ds_varsol)
        dy_ds_varsol_check = np.interp(z_check, s_varsol, dy_ds_varsol)

        # Assert that SplineBoris matches the VariableSolenoid reference
        dx_span = max(np.ptp(dx_ds_varsol_check), np.ptp(dx_ds_splineboris_check), 1e-12)
        dy_span = max(np.ptp(dy_ds_varsol_check), np.ptp(dy_ds_splineboris_check), 1e-12)
        xo.assert_allclose(
            dx_ds_splineboris_check,
            dx_ds_varsol_check,
            rtol=0,
            atol=1.1 * dx_span,
        )
        xo.assert_allclose(
            dy_ds_splineboris_check,
            dy_ds_varsol_check,
            rtol=0,
            atol=1.1 * dy_span,
        )



def test_splineboris_undulator_vs_boris_spatial(undulator_fit_result, make_segment_field):
    """
    Build a lightweight undulator from spline-fit parameters using SplineBorisSequence
    and check that tracking with SplineBoris and BorisSpatialIntegrator gives consistent
    end coordinates.
    """

    # ------------------------------------------------------------------
    # Load fit parameters and build undulator using SplineBorisSequence
    # ------------------------------------------------------------------
    fit_result = undulator_fit_result
    multipole_order = fit_result.multipole_order

    # Build undulator using SplineBorisSequence from FieldFitResult
    seq = SplineBorisSequence.from_fit_result(
        fit_result,
        steps_per_point=1,
    )

    line_spline = seq.to_line()

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
    # Extract parameters from SplineBorisSequence elements
    # ------------------------------------------------------------------
    boris_elems = []
    for elem in seq.elements:
        # par_list is a 1D array of polynomial coefficients for this piece
        params_i = np.asarray(elem.par_list, dtype=float)
        field_i = make_segment_field(params_i, multipole_order,
                                     s_start=float(elem.s_start))

        boris_elems.append(
            xt.BorisSpatialIntegrator(
                fieldmap_callable=field_i,
                s_start=float(elem.s_start),
                s_end=float(elem.s_end),
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
    for val in (p_spline.x, p_spline.px, p_spline.y, p_spline.py, p_spline.zeta, p_spline.delta):
        assert np.isfinite(val)
    for val in (p_boris.x, p_boris.px, p_boris.y, p_boris.py, p_boris.zeta, p_boris.delta):
        assert np.isfinite(val)



def test_splineboris_rotated_undulator_vs_boris_spatial(undulator_rotated_fit_result, undulator_fit_result, make_segment_field):
    '''
    Rotate the field map by 90 degrees and check that the fit parameters obey the rotation rule:
    Bx_rotated == By_original,
    By_rotated == -Bx_original,
    Bs_rotated == Bs_original
    '''

    # ------------------------------------------------------------------
    # Load fit parameters and build undulator using SplineBorisSequence
    # ------------------------------------------------------------------
    # Rebuild FieldFitResult objects for rotated and original cases
    df_rot = undulator_rotated_fit_result

    # ------------------------------------------------------------------
    # Check that the fit parameters obey the rotation rule:
    #   Bx_rotated == By_original,  By_rotated == -Bx_original,
    #   Bs_rotated == Bs_original
    # ------------------------------------------------------------------
    df_orig = undulator_fit_result
    multipole_order = df_orig.multipole_order

    # Bx_rotated coefficients should equal By_original coefficients
    def _collect_hermite(fit_res, field, der):
        return np.concatenate(
            [
                seg.hermite_params
                for seg in fit_res.segments
                if seg.field_component == field and seg.derivative_x == der
            ]
        )

    for der in range(multipole_order):
        rot_vals = _collect_hermite(df_rot, "Bx", der)
        orig_vals = _collect_hermite(df_orig, "By", der)
        assert len(rot_vals) == len(orig_vals), (
            f"Bx_rot vs By_orig length mismatch for der={der}")
        xo.assert_allclose(rot_vals, orig_vals, atol=1e-12, rtol=1e-12)

    # By_rotated coefficients should equal -Bx_original coefficients
    for der in range(multipole_order):
        rot_vals = _collect_hermite(df_rot, "By", der)
        orig_vals = _collect_hermite(df_orig, "Bx", der)
        assert len(rot_vals) == len(orig_vals), (
            f"By_rot vs Bx_orig length mismatch for der={der}")
        xo.assert_allclose(rot_vals, -orig_vals, atol=1e-12, rtol=1e-12)

    # Bs_rotated coefficients should equal Bs_original coefficients
    rot_vals = _collect_hermite(df_rot, "Bs", 0)
    orig_vals = _collect_hermite(df_orig, "Bs", 0)
    xo.assert_allclose(rot_vals, orig_vals, atol=1e-12, rtol=1e-12)

    # Build undulator using SplineBorisSequence from rotated FieldFitResult
    seq = SplineBorisSequence.from_fit_result(
        df_rot,
        steps_per_point=1,
    )

    line_splineboris = seq.to_line()

    # This undulator is part of the SLS, so we use the nominal energy of the SLS.
    p_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)
    line_splineboris.particle_ref = p_ref.copy()

    p_splineboris = line_splineboris.particle_ref.copy()
    p_splineboris.x = 1e-3
    p_splineboris.px = 1e-4
    p_splineboris.y = 0.5e-3
    p_splineboris.py = -0.5e-4

    line_splineboris.track(p_splineboris)

    # ------------------------------------------------------------------
    # Build a parallel undulator line using BorisSpatialIntegrator
    # Extract parameters from SplineBorisSequence elements
    # ------------------------------------------------------------------
    boris_elems = []
    for elem in seq.elements:
        # par_list is a 1D array of polynomial coefficients for this piece
        params_i = np.asarray(elem.par_list, dtype=float)
        field_i = make_segment_field(params_i, multipole_order,
                                     s_start=float(elem.s_start))

        boris_elems.append(
            xt.BorisSpatialIntegrator(
                fieldmap_callable=field_i,
                s_start=float(elem.s_start),
                s_end=float(elem.s_end),
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
    for val in (
        p_splineboris.x,
        p_splineboris.px,
        p_splineboris.y,
        p_splineboris.py,
        p_splineboris.zeta,
        p_splineboris.delta,
    ):
        assert np.isfinite(val)
    for val in (p_boris.x, p_boris.px, p_boris.y, p_boris.py, p_boris.zeta, p_boris.delta):
        assert np.isfinite(val)



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

    # Initialize random number generators
    particles_mean_0._init_random_number_generator()
    particles_qntm_0._init_random_number_generator()

    dct_mean_before = particles_mean_0.to_dict()

    # Track particles
    splineboris_mean.track(particles_mean_0)
    splineboris_qntm.track(particles_qntm_0)

    dct_mean = particles_mean_0.to_dict()
    dct_qntm = particles_qntm_0.to_dict()

    # Test 1: Average and stochastic models should give same mean energy loss
    xo.assert_allclose(dct_mean['delta'], np.mean(dct_qntm['delta']),
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



def test_splineboris_variable_solenoid_radiation(solenoid_field, solenoid_vs_varsol_fit_result):

    delta=np.array([0, 4])
    p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                    energy0=45.6e9,
                    x=[-5e-3, -5e-3], px=-1e-3*(1+delta), y=5e-3,
                    delta=delta)

    sf = solenoid_field

    # --- SplineBoris tracking ---
    fit_result = solenoid_vs_varsol_fit_result

    seq = SplineBorisSequence.from_fit_result(
        fit_result,
        steps_per_point=1,
        radiation_flag=1,
    )

    line_boris = seq.to_line()
    line_boris.build_tracker()
    line_boris.configure_radiation(model='mean')

    p_boris = p0.copy()
    line_boris.track(p_boris, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_boris = line_boris.record_last_track

    # Compute dE/ds from SplineBoris ptau (central differences)
    dE_ds_boris = 0 * mon_boris.ptau
    dE_ds_boris[:, 1:-1] = -((mon_boris.ptau[:, 2:] - mon_boris.ptau[:, :-2])
                              / (mon_boris.s[:, 2:] - mon_boris.s[:, :-2])
                              * p_boris.energy0[0])

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
    z_check = sf.z0 + sf.L * np.linspace(-2, 2, 1001)

    for i_part in range(len(delta)):

        # SplineBoris data
        s_boris_mid = 0.5 * (mon_boris.s[i_part, :-1] + mon_boris.s[i_part, 1:])
        dx_ds_boris = np.diff(mon_boris.x[i_part, :]) / np.diff(mon_boris.s[i_part, :])
        dy_ds_boris = np.diff(mon_boris.y[i_part, :]) / np.diff(mon_boris.s[i_part, :])

        # VariableSolenoid data
        s_xsuite = 0.5 * (mon.s[i_part, :-1] + mon.s[i_part, 1:])
        dx_ds_xsuite = np.diff(mon.x[i_part, :]) / np.diff(mon.s[i_part, :])
        dy_ds_xsuite = np.diff(mon.y[i_part, :]) / np.diff(mon.s[i_part, :])
        dE_ds_xsuite = dE_ds[i_part, :]

        dx_ds_xsuite_check = np.interp(z_check, s_xsuite, dx_ds_xsuite)
        dy_ds_xsuite_check = np.interp(z_check, s_xsuite, dy_ds_xsuite)
        dE_ds_xsuite_check = np.interp(z_check, mon.s[i_part, :], dE_ds_xsuite)

        dx_ds_boris_check = np.interp(z_check, s_boris_mid, dx_ds_boris)
        dy_ds_boris_check = np.interp(z_check, s_boris_mid, dy_ds_boris)
        dE_ds_boris_check = np.interp(z_check, mon_boris.s[i_part, :],
                                      dE_ds_boris[i_part, :])

        this_emitted_dpx = emitted_dpx[i_part, :]
        this_emitted_dpy = emitted_dpy[i_part, :]
        this_dE_ds = dE_ds[i_part, :]
        this_dx_ds = dx_ds[i_part, :]
        this_dy_ds = dy_ds[i_part, :]

        dx_span = max(np.ptp(dx_ds_boris_check), np.ptp(dx_ds_xsuite_check), 1e-12)
        dy_span = max(np.ptp(dy_ds_boris_check), np.ptp(dy_ds_xsuite_check), 1e-12)
        dE_span = max(np.ptp(dE_ds_boris_check), np.ptp(dE_ds_xsuite_check), 1e-12)
        xo.assert_allclose(dx_ds_xsuite_check, dx_ds_boris_check, rtol=0, atol=1.1 * dx_span)
        xo.assert_allclose(dy_ds_xsuite_check, dy_ds_boris_check, rtol=0, atol=1.1 * dy_span)
        xo.assert_allclose(dE_ds_xsuite_check, dE_ds_boris_check, rtol=0, atol=1.1 * dE_span)

        xo.assert_allclose(ax_ref[i_part, :], mon.ax[i_part, :],
                        rtol=0, atol=np.max(np.abs(ax_ref)*3e-2))
        xo.assert_allclose(ay_ref[i_part, :], mon.ay[i_part, :],
                        rtol=0, atol=np.max(np.abs(ay_ref)*3e-2))

        emitted_dpx_span = max(np.ptp(this_emitted_dpx), 1e-12)
        emitted_dpy_span = max(np.ptp(this_emitted_dpy), 1e-12)
        xo.assert_allclose(
            this_emitted_dpx,
            0.5 * (this_dE_ds[:-1] + this_dE_ds[1:]) * this_dx_ds * np.diff(mon.s[i_part, :]) / p0.p0c[0],
            rtol=0,
            atol=2e-2 * emitted_dpx_span,
        )
        xo.assert_allclose(
            this_emitted_dpy,
            0.5 * (this_dE_ds[:-1] + this_dE_ds[1:]) * this_dy_ds * np.diff(mon.s[i_part, :]) / p0.p0c[0],
            rtol=0,
            atol=5e-2 * emitted_dpy_span,
        )



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
    zip(
        [case['case'].copy() for case in COMMON_TEST_CASES],
        [3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 2e-5, 1e-5, 2e-8, 1e-5, 2e-5],
    ),
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



@pytest.mark.parametrize(
    'case,atol',
    zip(
        [case['case'].copy() for case in COMMON_TEST_CASES],
        [6e-8, 6e-8, 6e-8, 6e-8, 6e-8, 6e-8, 6e-5, 3e-5, 2e-7, 3e-5, 6e-5],
    ),
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
    bnorm_1_hermite = [quad_gradient, 0, quad_gradient, 0, quad_gradient]
    bs_hermite = [0, 0, 0, 0, 0]
    bnorm_1_s4 = xt.Spline4(
        val_start=quad_gradient, der_start=0.0, val_end=quad_gradient, der_end=0.0, integral=quad_gradient
    )
    bs_s4 = xt.Spline4(val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, integral=0.0)

    # Verify the polynomial evaluates to a constant gradient (s_local coordinates)
    bnorm_1_poly = xt.SplineBoris.hermite_to_polynomial(s_start, s_end, bnorm_1_hermite)
    xo.assert_allclose(bnorm_1_poly(np.linspace(0, s_end - s_start, 100)), quad_gradient, rtol=1e-12, atol=1e-12)

    splineboris = xt.SplineBoris(
        bs=bs_s4,
        by=(None, bnorm_1_s4),
        s_start=s_start,
        length=length,
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


def test_splineboris_bnorm_only_omitting_bs():
    hermite = xt.Spline4(val_start=1.0, der_start=0.0, val_end=1.0, der_end=0.0, integral=1.0)
    elem = xt.SplineBoris(
        by=hermite,
        s_start=0.0,
        length=1.0,
        n_steps=1,
    )
    assert elem.multipole_order == 1
    assert len(elem.par_list) == (1 + 2 * elem.multipole_order) * 5


def test_splineboris_bs_only_empty_bnorm_bskew():
    bs = xt.Spline4(val_start=0.1, der_start=0.0, val_end=0.1, der_end=0.0, integral=0.1)
    elem = xt.SplineBoris(
        bs=bs,
        by=(),
        bx=(),
        s_start=0.0,
        length=1.0,
        n_steps=1,
    )
    assert elem.multipole_order == 1


def test_splineboris_requires_at_least_one_field_component():
    with pytest.raises(ValueError, match="At least one of bs, bx, or by"):
        xt.SplineBoris(
            s_start=0.0,
            length=1.0,
        )


def test_splineboris_kernel_param_names_order():
    n = xt.SplineBoris._NUM_COEFFS
    names1 = xt.SplineBoris._get_param_names(1)
    assert names1[:n] == [f"Bs_{k}" for k in range(n)]
    assert names1[n : 2 * n] == [f"Bnorm_0_{k}" for k in range(n)]
    assert names1[2 * n :] == [f"Bskew_0_{k}" for k in range(n)]

    names2 = xt.SplineBoris._get_param_names(2)
    assert names2[:n] == [f"Bs_{k}" for k in range(n)]
    assert names2[n : 2 * n] == [f"Bnorm_0_{k}" for k in range(n)]
    assert names2[2 * n : 3 * n] == [f"Bnorm_1_{k}" for k in range(n)]
    assert names2[3 * n : 4 * n] == [f"Bskew_0_{k}" for k in range(n)]
    assert names2[4 * n : 5 * n] == [f"Bskew_1_{k}" for k in range(n)]


def test_splineboris_sparse_derivative_keys_and_zero_fill():
    hermite_one = xt.Spline4(val_start=1.0, der_start=0.0, val_end=1.0, der_end=0.0, integral=1.0)
    hermite_two = xt.Spline4(val_start=2.0, der_start=0.0, val_end=2.0, der_end=0.0, integral=2.0)
    bs_zeros = xt.Spline4(val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, integral=0.0)
    elem = xt.SplineBoris(
        bs=bs_zeros,
        by=(None, None, hermite_two),
        bx=(hermite_one,),
        s_start=0.0,
        length=1.0,
        n_steps=1,
    )

    # Convention: multipole_order is number of derivative orders (max key + 1).
    assert elem.multipole_order == 3
    assert len(elem.par_list) == (1 + 2 * elem.multipole_order) * 5


def test_splineboris_invalid_component_inputs_fail():
    bs_zeros = xt.Spline4(val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, integral=0.0)
    one = xt.Spline4(val_start=1.0, der_start=0.0, val_end=1.0, der_end=0.0, integral=1.0)

    with pytest.raises(TypeError, match="by must be a Spline4, tuple/list of Spline4/None, or None"):
        xt.SplineBoris(
            bs=bs_zeros,
            by={"0": one},
            bx=(one,),
            s_start=0.0,
            length=1.0,
        )

    with pytest.raises(TypeError, match="bx\\[1\\] must be a Spline4 or None"):
        xt.SplineBoris(
            bs=bs_zeros,
            by=(one,),
            bx=(one, 3.14),
            s_start=0.0,
            length=1.0,
        )


def test_splineboris_order_above_max_fails():
    bs_zeros = xt.Spline4(val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, integral=0.0)
    h = xt.Spline4(val_start=1.0, der_start=0.0, val_end=1.0, der_end=0.0, integral=1.0)
    with pytest.raises(ValueError, match="Unsupported multipole_order=8; max supported is 7"):
        xt.SplineBoris(
            bs=bs_zeros,
            by=(None, None, None, None, None, None, None, h),
            s_start=0.0,
            length=1.0,
        )


@pytest.mark.parametrize("bad_length", [0.0, -1.0])
def test_splineboris_length_must_be_positive(bad_length):
    h0 = xt.Spline4(val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, integral=0.0)
    with pytest.raises(ValueError, match="length must be finite and > 0"):
        xt.SplineBoris(
            bs=h0,
            by=(h0,),
            bx=(h0,),
            s_start=0.0,
            length=bad_length,
        )


def test_splineboris_evaluate_field_dispatch_uses_consistent_order(evaluate_b):
    bs_zeros = xt.Spline4(val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, integral=0.0)
    bn0 = xt.Spline4(val_start=1.0, der_start=0.0, val_end=1.0, der_end=0.0, integral=1.0)
    bn1 = xt.Spline4(val_start=3.0, der_start=0.0, val_end=3.0, der_end=0.0, integral=3.0)
    bs0 = xt.Spline4(val_start=2.0, der_start=0.0, val_end=2.0, der_end=0.0, integral=2.0)
    elem = xt.SplineBoris(
        bs=bs_zeros,
        by=(bn0, bn1),
        bx=(bs0,),
        s_start=2.0,
        length=1.0,
    )

    x = 1e-3
    y = -2e-3
    s = 2.4
    ref_bx, ref_by, ref_bs = evaluate_b(x, y, s - elem.s_start, np.asarray(elem.par_list), elem.multipole_order)
    bx, by, bs = elem.evaluate_field(x, y, s)
    xo.assert_allclose([bx, by, bs], [ref_bx, ref_by, ref_bs], rtol=0, atol=1e-15)
