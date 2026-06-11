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
    ctx.add_kernels(
        kernels={
            'borissolenoid_test_elliptic': elliptic_knl,
            'borissolenoid_test_field': field_knl,
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

    xo.assert_allclose(p_elem.x, p_boris.x, rtol=1e-9, atol=1e-10)
    xo.assert_allclose(p_elem.y, p_boris.y, rtol=1e-9, atol=1e-10)
    xo.assert_allclose(p_elem.px, p_boris.px, rtol=1e-9, atol=1e-10)
    xo.assert_allclose(p_elem.py, p_boris.py, rtol=1e-9, atol=1e-10)
    xo.assert_allclose(p_elem.zeta, p_boris.zeta, rtol=1e-9, atol=1e-9)
