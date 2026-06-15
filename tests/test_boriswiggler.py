import numpy as np
import pytest
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt
from xtrack.beam_elements.boriswiggler_src.wiggler_B_field_eval_python import evaluate_wiggler_B
from xtrack.beam_elements.elements import _resolve_wiggler_period_params


def test_resolve_wiggler_period_params_from_n_periods():
    n_periods, lambda_u, k_u = _resolve_wiggler_period_params(
        length=2.0, n_periods=4,
    )
    assert n_periods == 4
    xo.assert_allclose(lambda_u, 0.5)
    xo.assert_allclose(k_u, 2.0 * np.pi / 0.5)


def test_resolve_wiggler_period_params_from_lambda_u():
    n_periods, lambda_u, k_u = _resolve_wiggler_period_params(
        length=2.0, lambda_u=0.5,
    )
    assert n_periods == 4
    xo.assert_allclose(lambda_u, 0.5)


def test_resolve_wiggler_period_params_from_k_u():
    n_periods, lambda_u, k_u = _resolve_wiggler_period_params(
        length=2.0, k_u=2.0 * np.pi / 0.5,
    )
    assert n_periods == 4
    xo.assert_allclose(lambda_u, 0.5)


def test_resolve_wiggler_period_params_non_integer_periods():
    with pytest.raises(ValueError, match="integer number of periods"):
        _resolve_wiggler_period_params(length=2.0, lambda_u=0.6)


def test_resolve_wiggler_period_params_requires_exactly_one():
    with pytest.raises(ValueError, match="Exactly one"):
        _resolve_wiggler_period_params(length=2.0)
    with pytest.raises(ValueError, match="Exactly one"):
        _resolve_wiggler_period_params(length=2.0, n_periods=4, lambda_u=0.5)


def test_boriswiggler_default_n_steps():
    el = xt.BorisWiggler(length=2.0, g=0.02, B_r=1.0, n_periods=4)
    assert el.n_steps == 40
    assert el.n_periods == 4
    xo.assert_allclose(el.k_u, 2.0 * np.pi / 0.5)
    xo.assert_allclose(el.b_tilde, 1.0 / np.cosh(np.pi * 0.02 / 0.5))


def test_boriswiggler_get_field_matches_analytic():
    length = 2.0
    g = 0.02
    B_r = 1.2
    n_periods = 4
    el = xt.BorisWiggler(length=length, g=g, B_r=B_r, n_periods=n_periods)

    x = np.array([0.0, 1e-3])
    y = np.array([0.0, 2e-3])
    s_local = np.array([0.25, 0.75])

    bx_el, by_el, bs_el = el.get_field(x, y, s_local)
    bx_ref, by_ref, bs_ref = evaluate_wiggler_B(x, y, s_local, el.k_u, el.b_tilde)

    xo.assert_allclose(bx_el, bx_ref)
    xo.assert_allclose(by_el, by_ref)
    xo.assert_allclose(bs_el, bs_ref)


def test_boriswiggler_to_dict_from_dict_roundtrip():
    el = xt.BorisWiggler(length=2.0, g=0.02, B_r=1.0, n_periods=4, n_steps=20)
    dct = el.to_dict()
    el2 = xt.BorisWiggler.from_dict(dct)
    assert el2.length == el.length
    assert el2.g == el.g
    assert el2.B_r == el.B_r
    assert el2.n_periods == el.n_periods
    assert el2.n_steps == el.n_steps
    xo.assert_allclose(el2.k_u, el.k_u)
    xo.assert_allclose(el2.b_tilde, el.b_tilde)


def test_boriswiggler_split_into_segments():
    wiggler = xt.BorisWiggler(
        length=2.0, g=0.02, B_r=1.0, n_periods=4, n_steps=40,
    )
    segments = wiggler.split_into_segments([0.25, 0.25, 0.25, 0.25])
    assert len(segments) == 4
    xo.assert_allclose(sum(seg.length for seg in segments), 2.0)
    assert sum(seg.n_steps for seg in segments) == 40
    xo.assert_allclose([seg.s_offset for seg in segments], [0.0, 0.5, 1.0, 1.5])
    for seg in segments:
        xo.assert_allclose(seg.k_u, wiggler.k_u)
        xo.assert_allclose(seg.b_tilde, wiggler.b_tilde)


def test_boriswiggler_sliced_tracking_matches_unsliced():
    wiggler = xt.BorisWiggler(
        length=2.0, g=0.02, B_r=1.0, n_periods=4, n_steps=200,
    )
    p0 = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1.0,
        energy0=1e9,
        x=1e-4,
        y=2e-4,
        px=1e-4,
        py=-5e-5,
    )

    line = xt.Line(elements=[wiggler])
    line.build_tracker(use_prebuilt_kernels=False)
    p_unsliced = p0.copy()
    line.track(p_unsliced)

    line_sliced = line.copy()
    line_sliced.slice_thick_elements([
        xt.Strategy(
            slicing=xt.Uniform(8, mode='thick'),
            element_type=xt.BorisWiggler,
        ),
    ])
    line_sliced.build_tracker(use_prebuilt_kernels=False)
    p_sliced = p0.copy()
    line_sliced.track(p_sliced)

    for name in ('x', 'y', 'px', 'py', 'zeta'):
        xo.assert_allclose(
            getattr(p_unsliced, name),
            getattr(p_sliced, name),
            rtol=0.0,
            atol=1e-5,
        )


@for_all_test_contexts
def test_boriswiggler_tracking_smoke(test_context):
    el = xt.BorisWiggler(
        length=1.0,
        g=0.02,
        B_r=0.5,
        n_periods=2,
        n_steps=20,
        _context=test_context,
    )
    line = xt.Line(elements=[el])
    line.particle_ref = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1.0,
        energy0=1e9,
        _context=test_context,
    )
    p = line.particle_ref.copy()
    p.x = 1e-4
    p.y = 2e-4
    p.px = 1e-4
    p.py = -5e-5

    line.build_tracker(use_prebuilt_kernels=False)
    line.track(p)
    assert np.all(np.isfinite(p.x))
    assert np.all(np.isfinite(p.y))
    assert np.all(np.isfinite(p.px))
    assert np.all(np.isfinite(p.py))
