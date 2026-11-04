import numpy as np
import pytest
import xobjects as xo
import xtrack as xt

COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')


def make_element(integrator='boris', num_integration_steps=40, pkin_const=False):
    # Round solenoid-like profile (transverse seed -1/2 d ksolc/ds) on top of
    # dipole, quadrupole, sextupole and skew content.
    ksolc = np.array([0.1, 1.2, -1.2])
    dks = np.polynomial.Polynomial(ksolc).deriv().coef
    return xt.BFieldExpansion(
        length=1.0, s_start=0.05, ksolc=ksolc,
        ksc=[[0.], -0.5 * dks], knc=[[0.002], [0.3, 0.1]],
        k2=0.5, k1s=0.05, ksl=[0., 0.01],
        integrator=integrator, num_integration_steps=num_integration_steps,
        pkin_const=pkin_const)


def make_particles():
    return xt.Particles(
        p0c=45.6e9, mass0=xt.ELECTRON_MASS_EV,
        x=[1e-3, -2e-3, 0.5e-3, 0.], px=[1e-4, 0., -2e-4, 3e-4],
        y=[0., 1.5e-3, -1e-3, 2e-3], py=[2e-4, -1e-4, 0., 0.],
        zeta=[0., 1e-3, -2e-3, 0.], delta=[0., 1e-3, -2e-3, 5e-3],
        chi=[1., 0.7, -1., 1.], mass_ratio=[1., 2., 1., 1.])


def track(element, particles=None):
    particles = make_particles() if particles is None else particles
    element.track(particles)
    return particles


@pytest.mark.parametrize('pkin_const', [False, True])
def test_boris_converges_to_rk4_at_second_order(pkin_const):
    reference = track(make_element('rk4', 2000, pkin_const))
    errors = []
    for nstep in (20, 40, 80, 160):
        particles = track(make_element('boris', nstep, pkin_const))
        assert np.all(particles.state == 1)
        for name in ('s', 'ay'):
            xo.assert_allclose(getattr(particles, name),
                               getattr(reference, name), rtol=0, atol=1e-15)
        # ax is the exit potential at the tracked position (0 if pkin_const).
        errors.append(max(np.max(np.abs(getattr(particles, name)
                                        - getattr(reference, name)))
                          for name in (*COORDS, 'ax')))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    xo.assert_allclose(orders, 2., rtol=0, atol=0.05)
    assert errors[-1] < 1e-7


@pytest.mark.parametrize('pkin_const', [False, True])
def test_boris_backtrack_retraces_forward_steps(pkin_const):
    line = xt.Line(elements={'e': make_element('boris', 37, pkin_const)})
    initial = make_particles()
    particles = initial.copy()
    line.track(particles)
    line.track(particles, backtrack=True)
    assert np.all(particles.state == 1)
    # Boris is time-reversible: the round trip is exact up to rounding.
    for name in (*COORDS, 's'):
        xo.assert_allclose(getattr(particles, name), getattr(initial, name),
                           rtol=0, atol=1e-17)
    if pkin_const:
        xo.assert_allclose(particles.ax, 0., rtol=0, atol=0)


@pytest.mark.parametrize('pkin_const', [False, True])
def test_boris_thick_slices_match_parent(pkin_const):
    parent = make_element('boris', 40, pkin_const)
    reference = track(parent.copy())
    line = xt.Line(elements={'e': parent})
    line.slice_thick_elements([xt.Strategy(xt.Uniform(4, mode='thick'))])
    slices = [name for name in line.element_names
              if isinstance(line[name], xt.ThickSliceBFieldExpansion)]
    assert len(slices) == 4
    assert all(line[name].integrator == 'boris' for name in slices)
    particles = make_particles()
    line.track(particles, _force_no_end_turn_actions=True)
    # Slices split the parent's 40 steps into 4 x 10 identical steps.
    for name in (*COORDS, 's', 'ax', 'ay'):
        xo.assert_allclose(getattr(particles, name), getattr(reference, name),
                           rtol=0, atol=1e-16)


def test_boris_api_and_serialization():
    assert xt.BFieldExpansion.get_available_integrators() == ['rk4', 'boris']
    element = make_element('boris', 12)
    assert element.integrator == 'boris'
    assert element.to_dict()['integrator'] == 'boris'
    restored = xt.BFieldExpansion.from_dict(element.to_dict())
    assert restored.integrator == 'boris'
    for name in COORDS:
        xo.assert_allclose(getattr(track(restored), name),
                           getattr(track(element), name), rtol=0, atol=0)

    element.integrator = 'rk4'
    xo.assert_allclose(track(element).x, track(make_element('rk4', 12)).x,
                       rtol=0, atol=0)
    element.integrator = 'boris'
    assert element.integrator == 'boris'

    with pytest.raises(ValueError, match='only supports'):
        element.integrator = 'yoshida4'
    with pytest.raises(ValueError, match='straight'):
        xt.BFieldExpansion(length=1., h=0.3, integrator='boris')
    bent = xt.BFieldExpansion(length=1., h=0.3)
    with pytest.raises(ValueError, match='straight'):
        bent.integrator = 'boris'
    assert bent.integrator == 'rk4'


def test_boris_uniform_field_matches_exact_helix():
    # In a uniform longitudinal field the kinetic transverse momentum rotates
    # at omega = chi*bs/kl per unit s, with kl the constant longitudinal
    # kinetic momentum; the orbit is a helix. Boris reproduces it at second
    # order (Cayley angle 2*atan(omega*ds/2) instead of omega*ds).
    bs, length = 0.4, 1.3
    initial = xt.Particles(p0c=1e9, x=[1e-3, -2e-3, 0.], px=[1e-4, -3e-4, 2e-3],
                           y=[2e-3, 0., 1e-3], py=[0., 2e-4, -1e-3],
                           zeta=[0., 1e-3, 0.], delta=[0., 1e-2, -5e-3],
                           chi=[1., 1., -0.5], mass_ratio=[1., 1., 2.])
    kx0, ky0 = initial.px, initial.py  # pkin_const: ax = ay = 0 at entry
    delta1 = 1 + initial.delta
    kl = np.sqrt(delta1**2 - kx0**2 - ky0**2)
    phase = initial.chi * bs / kl * length
    expected = {
        'x': initial.x + (kx0 * np.sin(phase) + ky0 * (1 - np.cos(phase)))
                          / (initial.chi * bs),
        'y': initial.y + (ky0 * np.sin(phase) - kx0 * (1 - np.cos(phase)))
                          / (initial.chi * bs),
        'px': kx0 * np.cos(phase) + ky0 * np.sin(phase),
        'py': ky0 * np.cos(phase) - kx0 * np.sin(phase),
        'zeta': initial.zeta + length * (1 - delta1 / (initial.rvv * kl)),
        'delta': initial.delta,
    }
    for nstep, atol in ((50, 1e-8), (500, 1e-10)):
        element = xt.BFieldExpansion(length=length, ksolc=[bs],
                                     integrator='boris', pkin_const=True,
                                     num_integration_steps=nstep)
        particles = initial.copy()
        element.track(particles)
        for name, value in expected.items():
            xo.assert_allclose(getattr(particles, name), value,
                               rtol=0, atol=atol)
