import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.special import ellipk, ellipe, elliprf, elliprj
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts, skip_if_forbid_compile
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack.general import _pkg_root as pkg_root
from xtrack.beam_elements.borissolenoid_src.solenoid_B_field_eval_python import (
    evaluate_solenoid_B,
)


SOLENOID_MODEL_PARAMS = {
    "L_coil": 4.0,
    "a": 0.3,
    "B0": 1.5,
    "z0": 20.0,
}
INTERVAL = 30.0
N_STEPS = 5000


def _rotation_lab_to_zeta(Bx, By, Bz, vx, vy, vz):
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_perp = np.sqrt(Bx**2 + By**2)
    if B_mag < 1e-30 or B_perp < 1e-30:
        return vx, vy, (Bx * vx + By * vy + Bz * vz) / max(B_mag, 1e-30)
    inv_B_mag = 1.0 / B_mag
    inv_B_perp = 1.0 / B_perp
    Bz_over_B_perp = Bz * inv_B_mag * inv_B_perp
    ox = -By * inv_B_perp * vx + Bx * inv_B_perp * vy
    oy = (
        -Bx * Bz_over_B_perp * vx
        - By * Bz_over_B_perp * vy
        + B_perp * inv_B_mag * vz
    )
    oz = Bx * inv_B_mag * vx + By * inv_B_mag * vy + Bz * inv_B_mag * vz
    return ox, oy, oz


def _rotation_zeta_to_lab(Bx, By, Bz, vx, vy, vz):
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_perp = np.sqrt(Bx**2 + By**2)
    if B_mag < 1e-30 or B_perp < 1e-30:
        return vx, vy, Bz * vz / max(B_mag, 1e-30)
    inv_B_mag = 1.0 / B_mag
    inv_B_perp = 1.0 / B_perp
    Bz_over_B_perp = Bz * inv_B_mag * inv_B_perp
    ox = -By * inv_B_perp * vx - Bx * Bz_over_B_perp * vy + Bx * inv_B_mag * vz
    oy = Bx * inv_B_perp * vx - By * Bz_over_B_perp * vy + By * inv_B_mag * vz
    oz = B_perp * inv_B_mag * vy + Bz * inv_B_mag * vz
    return ox, oy, oz


def _sinc(theta):
    if abs(theta) < 1e-14:
        return 1.0
    return np.sin(theta) / theta


def _vers_over_theta(theta):
    if abs(theta) < 1e-14:
        return 0.0
    return (1.0 - np.cos(theta)) / theta


def _cos_minus_one_over_theta(theta):
    if abs(theta) < 1e-14:
        return 0.0
    return (np.cos(theta) - 1.0) / theta


def _helical_F_step_py(x, y, px, py, B_mag, P_z, q_coulomb, h):
    """Pure helical map in the B-aligned frame (B_x = B_y = 0)."""
    theta = q_coulomb * B_mag * h / P_z
    st = np.sin(theta)
    ct = np.cos(theta)
    h_over_Pz = h / P_z
    x_out = x + h_over_Pz * _sinc(theta) * px + h_over_Pz * _vers_over_theta(theta) * py
    y_out = y + h_over_Pz * _cos_minus_one_over_theta(theta) * px + h_over_Pz * _sinc(theta) * py
    px_out = ct * px + st * py
    py_out = -st * px + ct * py
    return x_out, y_out, px_out, py_out


def _helical_step_lab_py(x, y, px, py, ps, Bx, By, Bz, q_coulomb, h):
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    P_z = (Bx * px + By * py + Bz * ps) / B_mag
    x_z, y_z, _ = _rotation_lab_to_zeta(Bx, By, Bz, x, y, 0.0)
    px_z, py_z, _ = _rotation_lab_to_zeta(Bx, By, Bz, px, py, ps)
    x_z, y_z, px_z, py_z = _helical_F_step_py(
        x_z, y_z, px_z, py_z, B_mag, P_z, q_coulomb, h
    )
    _, _, z_z = _rotation_lab_to_zeta(Bx, By, Bz, x, y, 0.0)
    x_out, y_out, _ = _rotation_zeta_to_lab(Bx, By, Bz, x_z, y_z, z_z)
    px_out, py_out, _ = _rotation_zeta_to_lab(Bx, By, Bz, px_z, py_z, P_z)
    return x_out, y_out, px_out, py_out


def _add_borissolenoid_test_kernels(ctx):
    elliptic_knl = xo.Kernel(
        c_name='borissolenoid_test_elliptic',
        args=[
            xo.Arg(xo.Float64, name='m'),
            xo.Arg(xo.Float64, name='n'),
            xo.Arg(xo.Float64, name='k_out', pointer=True),
            xo.Arg(xo.Float64, name='e_out', pointer=True),
            xo.Arg(xo.Float64, name='pi_out', pointer=True),
        ],
    )
    field_knl = xo.Kernel(
        c_name='borissolenoid_test_field',
        args=[
            xo.Arg(xo.Float64, name='x'),
            xo.Arg(xo.Float64, name='y'),
            xo.Arg(xo.Float64, name='z'),
            xo.Arg(xo.Float64, name='L'),
            xo.Arg(xo.Float64, name='a'),
            xo.Arg(xo.Float64, name='B0'),
            xo.Arg(xo.Float64, name='z0'),
            xo.Arg(xo.Float64, name='bx_out', pointer=True),
            xo.Arg(xo.Float64, name='by_out', pointer=True),
            xo.Arg(xo.Float64, name='bz_out', pointer=True),
        ],
    )
    helical_knl = xo.Kernel(
        c_name='borissolenoid_test_helical_step',
        args=[
            xo.Arg(xo.Float64, name='x'),
            xo.Arg(xo.Float64, name='y'),
            xo.Arg(xo.Float64, name='px'),
            xo.Arg(xo.Float64, name='py'),
            xo.Arg(xo.Float64, name='ps'),
            xo.Arg(xo.Float64, name='Bx'),
            xo.Arg(xo.Float64, name='By'),
            xo.Arg(xo.Float64, name='Bz'),
            xo.Arg(xo.Float64, name='q_coulomb'),
            xo.Arg(xo.Float64, name='h'),
            xo.Arg(xo.Float64, name='x_out', pointer=True),
            xo.Arg(xo.Float64, name='y_out', pointer=True),
            xo.Arg(xo.Float64, name='px_out', pointer=True),
            xo.Arg(xo.Float64, name='py_out', pointer=True),
        ],
    )
    ctx.add_kernels(
        kernels={
            'borissolenoid_test_elliptic': elliptic_knl,
            'borissolenoid_test_field': field_knl,
            'borissolenoid_test_helical_step': helical_knl,
        },
        sources=[pkg_root / 'beam_elements/borissolenoid_src/test_kernels.h'],
    )


@for_all_test_contexts
def test_elliptic_integrals_vs_scipy(test_context):
    skip_if_forbid_compile()
    ctx = test_context
    _add_borissolenoid_test_kernels(ctx)

    m_vals = np.linspace(0.0, 0.98, 20)
    n_vals = np.linspace(-0.5, 0.5, 10)

    for m in m_vals:
        k_out = np.array([0.0])
        e_out = np.array([0.0])
        pi_out = np.array([0.0])
        ctx.kernels.borissolenoid_test_elliptic(
            m=float(m), n=0.0,
            k_out=k_out, e_out=e_out, pi_out=pi_out,
        )
        xo.assert_allclose(k_out[0], ellipk(m), rtol=1e-10, atol=1e-12)
        xo.assert_allclose(e_out[0], ellipe(m), rtol=1e-10, atol=1e-12)

    for m in m_vals[::5]:
        for n in n_vals:
            pi_out = np.array([0.0])
            k_out = np.array([0.0])
            e_out = np.array([0.0])
            ctx.kernels.borissolenoid_test_elliptic(
                m=float(m), n=float(n),
                k_out=k_out, e_out=e_out, pi_out=pi_out,
            )
            pi_ref = elliprf(0.0, 1.0 - m, 1.0) + elliprj(0.0, 1.0 - m, 1.0, 1.0 - n) * n / 3.0
            xo.assert_allclose(pi_out[0], pi_ref, rtol=1e-9, atol=1e-11)


@for_all_test_contexts
def test_solenoid_field_eval_c_vs_python(test_context):
    skip_if_forbid_compile()
    ctx = test_context
    _add_borissolenoid_test_kernels(ctx)

    L = SOLENOID_MODEL_PARAMS["L_coil"]
    a = SOLENOID_MODEL_PARAMS["a"]
    B0 = SOLENOID_MODEL_PARAMS["B0"]
    z0 = SOLENOID_MODEL_PARAMS["z0"]

    x_grid = np.linspace(-0.01, 0.01, 5)
    y_grid = np.linspace(-0.01, 0.01, 5)
    z_grid = np.linspace(18.0, 22.0, 9)
    xg, yg, zg = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

    bx_py, by_py, bz_py = evaluate_solenoid_B(xg, yg, zg, L, a, B0, z0)

    bx_c = np.zeros_like(xg)
    by_c = np.zeros_like(xg)
    bz_c = np.zeros_like(xg)

    flat_x = xg.ravel()
    flat_y = yg.ravel()
    flat_z = zg.ravel()
    for ii in range(flat_x.size):
        bx_out = np.array([0.0])
        by_out = np.array([0.0])
        bz_out = np.array([0.0])
        ctx.kernels.borissolenoid_test_field(
            x=float(flat_x[ii]), y=float(flat_y[ii]), z=float(flat_z[ii]),
            L=L, a=a, B0=B0, z0=z0,
            bx_out=bx_out, by_out=by_out, bz_out=bz_out,
        )
        bx_c.ravel()[ii] = bx_out[0]
        by_c.ravel()[ii] = by_out[0]
        bz_c.ravel()[ii] = bz_out[0]

    xo.assert_allclose(bx_c, bx_py, rtol=1e-10, atol=1e-12)
    xo.assert_allclose(by_c, by_py, rtol=1e-10, atol=1e-12)
    xo.assert_allclose(bz_c, bz_py, rtol=1e-10, atol=1e-12)


@for_all_test_contexts
def test_borissolenoid_get_field_vs_solenoid_field(test_context):
    skip_if_forbid_compile()
    el = xt.BorisSolenoid(
        _context=test_context,
        **SOLENOID_MODEL_PARAMS,
        length=INTERVAL,
        n_steps=10,
    )
    sf = SolenoidField(
        L=SOLENOID_MODEL_PARAMS["L_coil"],
        a=SOLENOID_MODEL_PARAMS["a"],
        B0=SOLENOID_MODEL_PARAMS["B0"],
        z0=SOLENOID_MODEL_PARAMS["z0"],
    )

    s_local = np.linspace(0, INTERVAL, 31)
    x = np.array([1e-3, 0.0, -5e-4])
    y = np.array([1e-3, 2e-3, 0.0])

    for xi, yi in zip(x, y):
        bx_el, by_el, bz_el = el.get_field(xi, yi, s_local, s_at_element=0.0)
        bx_sf, by_sf, bz_sf = sf.get_field(
            xi * np.ones_like(s_local),
            yi * np.ones_like(s_local),
            s_local,
        )
        xo.assert_allclose(bx_el, bx_sf, rtol=1e-10, atol=1e-12)
        xo.assert_allclose(by_el, by_sf, rtol=1e-10, atol=1e-12)
        xo.assert_allclose(bz_el, bz_sf, rtol=1e-10, atol=1e-12)


@for_all_test_contexts
def test_helical_map_uniform_B(test_context):
    skip_if_forbid_compile()
    ctx = test_context
    _add_borissolenoid_test_kernels(ctx)

    B_cases = [
        (0.0, 0.0, 1.5),
        (0.2, 0.1, 1.4),
    ]
    q_coulomb = qe
    h = 0.01
    P = 1.0e-19 * 1e4

    for Bx, By, Bz in B_cases:
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        B_perp = np.sqrt(Bx**2 + By**2)
        for px, py in [(P * 0.01, P * 0.005), (P * 0.02, -P * 0.01)]:
            ps = np.sqrt(max(P**2 - px**2 - py**2, 0.0))
            for x, y in [(1e-3, 2e-3), (-5e-4, 1e-3)]:
                x_ref, y_ref, px_ref, py_ref = _helical_step_lab_py(
                    x, y, px, py, ps, Bx, By, Bz, q_coulomb, h
                )
                x_out = np.array([0.0])
                y_out = np.array([0.0])
                px_out = np.array([0.0])
                py_out = np.array([0.0])
                ctx.kernels.borissolenoid_test_helical_step(
                    x=float(x), y=float(y),
                    px=float(px), py=float(py), ps=float(ps),
                    Bx=float(Bx), By=float(By), Bz=float(Bz),
                    q_coulomb=float(q_coulomb), h=float(h),
                    x_out=x_out, y_out=y_out, px_out=px_out, py_out=py_out,
                )
                xo.assert_allclose(x_out[0], x_ref, rtol=1e-12, atol=1e-15)
                xo.assert_allclose(y_out[0], y_ref, rtol=1e-12, atol=1e-15)
                xo.assert_allclose(px_out[0], px_ref, rtol=1e-12, atol=1e-15)
                xo.assert_allclose(py_out[0], py_ref, rtol=1e-12, atol=1e-15)

        if B_perp > 1e-30:
            v = np.array([0.3, -0.2, 0.5])
            R = np.array([
                [-By / B_perp, Bx / B_perp, 0.0],
                [-Bx * Bz / (B_mag * B_perp), -By * Bz / (B_mag * B_perp), B_perp / B_mag],
                [Bx / B_mag, By / B_mag, Bz / B_mag],
            ])
            xo.assert_allclose(R @ R.T, np.eye(3), rtol=1e-12, atol=1e-12)
            xo.assert_allclose(np.linalg.det(R), 1.0, rtol=1e-12, atol=1e-12)
            ox, oy, oz = _rotation_lab_to_zeta(Bx, By, Bz, v[0], v[1], v[2])
            vx, vy, vz = _rotation_zeta_to_lab(Bx, By, Bz, ox, oy, oz)
            xo.assert_allclose([vx, vy, vz], v, rtol=1e-12, atol=1e-12)


@for_all_test_contexts
def test_borissolenoid_tracking_vs_boris_spatial(test_context):
    skip_if_forbid_compile()

    delta = np.array([0, 4])
    p0 = xt.Particles(
        _context=test_context,
        mass0=xt.ELECTRON_MASS_EV,
        q0=1,
        energy0=45.6e9 / 1000,
        x=[-1e-3, -1e-3],
        px=-1e-3 * (1 + delta),
        y=1e-3,
        delta=delta,
    )

    sf = SolenoidField(
        L=SOLENOID_MODEL_PARAMS["L_coil"],
        a=SOLENOID_MODEL_PARAMS["a"],
        B0=SOLENOID_MODEL_PARAMS["B0"],
        z0=SOLENOID_MODEL_PARAMS["z0"],
    )

    line = xt.Line(elements=[
        xt.BorisSolenoid(
            _context=test_context,
            **SOLENOID_MODEL_PARAMS,
            length=INTERVAL,
            n_steps=N_STEPS,
        )
    ])
    line.build_tracker()

    p_elem = p0.copy()
    line.track(p_elem)

    integrator = xt.BorisSpatialIntegrator(
        fieldmap_callable=sf.get_field,
        s_start=0,
        s_end=INTERVAL,
        n_steps=N_STEPS,
    )
    p_boris = p0.copy()
    integrator.track(p_boris)

    # Helical exponential map vs split Boris — benchmark, not bitwise equality.
    xo.assert_allclose(p_elem.x, p_boris.x, rtol=5e-3, atol=1e-6)
    xo.assert_allclose(p_elem.y, p_boris.y, rtol=5e-3, atol=1e-6)
    xo.assert_allclose(p_elem.px, p_boris.px, rtol=5e-3, atol=1e-6)
    xo.assert_allclose(p_elem.py, p_boris.py, rtol=5e-3, atol=1e-6)
    xo.assert_allclose(p_elem.zeta, p_boris.zeta, rtol=5e-3, atol=1e-6)
