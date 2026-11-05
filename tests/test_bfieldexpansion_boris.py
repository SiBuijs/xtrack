import numpy as np
import pytest
from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import epsilon_0, hbar
import xobjects as xo
from xobjects.test_helpers import fix_random_seed
import xtrack as xt

COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')
# Radiation and spin are implemented for both integrators.
INTEGRATORS = ['boris', 'rk4']


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
    # Curved elements support Boris as well
    bent = xt.BFieldExpansion(length=1., h=0.3)
    bent.integrator = 'boris'
    assert bent.integrator == 'boris'
    assert xt.BFieldExpansion(length=1., h=0.3, integrator='boris').integrator == 'boris'


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


# ---------------------------------------------------------------------------
# Synchrotron radiation
# ---------------------------------------------------------------------------

def _straight_dipole_bfe(k0, length, nstep, radiation_flag=0, integrator='boris'):
    return xt.BFieldExpansion(length=length, k0=k0, integrator=integrator,
                              num_integration_steps=nstep, pkin_const=True,
                              radiation_flag=radiation_flag)


@pytest.mark.parametrize('integrator', INTEGRATORS)
@fix_random_seed(2389457)
def test_dipole_radiation(integrator):
    # Uniform 2 T dipole field in a straight frame, 5 GeV electrons
    # (as test_splineboris_bend_radiation).
    length, b_tesla, p0c = 1.0, 2.0, 5e9
    k0 = b_tesla / (p0c / clight)
    n_particles = 100000
    particles0 = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV,
                              x=np.zeros(n_particles), px=1e-4, py=-1e-4)
    particles0._init_random_number_generator()

    tracked = {}
    for flag in (0, 1, 2, 3):
        particles = particles0.copy()
        _straight_dipole_bfe(k0, length, 100, radiation_flag=flag,
                             integrator=integrator).track(particles)
        tracked[flag] = particles

    # Average and stochastic models give the same mean energy loss
    xo.assert_allclose(np.mean(tracked[2].delta), tracked[1].delta, atol=0, rtol=5e-3)
    xo.assert_allclose(np.mean(tracked[3].delta), tracked[1].delta, atol=0, rtol=5e-3)

    # Energy loss against the Larmor formula
    gamma = (particles0.energy / particles0.mass0)[0]
    gamma0 = particles0.gamma0[0]
    mass0_kg = particles0.mass0 * qe / clight**2
    r0 = qe**2 / (4 * np.pi * epsilon_0 * mass0_kg * clight**2)
    rho = 1 / k0
    power = 2 * r0 * clight * mass0_kg * clight**2 * gamma0**2 * gamma**2 / (3 * rho**2)
    expected_loss_ev = power * length / clight / qe
    loss_ev = -(tracked[1].ptau - particles0.ptau) * particles0.p0c
    xo.assert_allclose(loss_ev, expected_loss_ev, atol=0, rtol=5e-3)

    # The same field in a straight xt.Bend radiates the same energy and gives
    # the same radiation kick on the transverse momenta (both converge to the
    # same value with the number of steps).
    def radiation_effect(element):
        line = xt.Line(elements=[element])
        line.build_tracker()
        out = []
        for model in ('mean', None):
            line.configure_radiation(model=model)
            particles = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV,
                                     x=[0., 1e-3], px=[1e-4, -2e-4],
                                     py=[-1e-4, 3e-4], delta=[0., 1e-2])
            line.track(particles)
            out.append(particles)
        with_rad, without_rad = out
        return {name: getattr(with_rad, name) - getattr(without_rad, name)
                for name in ('ptau', 'px', 'py')}

    bfe = radiation_effect(_straight_dipole_bfe(k0, length, 1000,
                                               integrator=integrator))
    bend = radiation_effect(xt.Bend(length=length, angle=0, k0=k0,
                                    num_multipole_kicks=1000,
                                    edge_entry_active=False,
                                    edge_exit_active=False))
    xo.assert_allclose(bfe['ptau'], bend['ptau'], atol=0, rtol=1e-6)
    xo.assert_allclose(bfe['py'], bend['py'], atol=0, rtol=1e-5)
    # The transverse radiation kick must be there (it is lost if the
    # integrator does not hand its momenta to the radiation routine).
    assert np.all(np.abs(bfe['px']) > 1e-6)
    xo.assert_allclose(bfe['px'], bend['px'], atol=0, rtol=1e-2)


@pytest.mark.parametrize('integrator', INTEGRATORS)
@fix_random_seed(9834751)
def test_dipole_photon_record(integrator):
    length, b_tesla, p0c = 1.0, 2.0, 5e9
    k0 = b_tesla / (p0c / clight)
    line = xt.Line(elements=[
        xt.Drift(length=1.0), _straight_dipole_bfe(k0, length, 100, integrator=integrator),
        xt.Drift(length=1.0), _straight_dipole_bfe(k0, length, 100, integrator=integrator)])
    line.build_tracker()
    line.configure_radiation(model='quantum')

    particles0 = xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV,
                              x=np.zeros(20000), px=1e-4, py=-1e-4)
    particles0._init_random_number_generator()
    gamma = (particles0.energy / particles0.mass0)[0]

    photon_energy = []
    for _ in range(5):
        record = line.start_internal_logging_for_elements_of_type(
            xt.BFieldExpansion, capacity=int(5e6))
        particles = particles0.copy()
        line.track(particles)
        n_recorded = record._index.num_recorded
        assert 0 < n_recorded < int(5e6)
        energy = np.asarray(record.photon_energy[:n_recorded])
        # Energy lost by the particles is carried by the recorded photons
        xo.assert_allclose(np.sum(energy),
                           -np.sum((particles.ptau - particles0.ptau) * particles.p0c),
                           atol=0, rtol=1e-6)
        photon_energy.append(energy)
    photon_energy = np.concatenate(photon_energy)

    mass0_kg = particles0.mass0 * qe / clight**2
    e_crit = 3 * qe * hbar * gamma**2 * b_tesla / (2 * mass0_kg) / qe
    xo.assert_allclose(np.mean(photon_energy), 8 * np.sqrt(3) / 45 * e_crit,
                       atol=0, rtol=1e-2)
    xo.assert_allclose(np.mean(photon_energy**2), 11 / 27 * e_crit**2,
                       atol=0, rtol=2e-2)

    line.configure_radiation(model='quantum-kick')
    record = line.start_internal_logging_for_elements_of_type(
        xt.BFieldExpansion, capacity=int(5e6))
    particles = particles0.copy()
    line.track(particles)
    assert np.all(particles.ptau < particles0.ptau)
    assert record._index.num_recorded == 0


@pytest.mark.parametrize('integrator', INTEGRATORS)
@pytest.mark.parametrize('pkin_const', [False, True])
def test_radiation_slices_match_parent(integrator, pkin_const):
    parent = make_element(integrator, 40, pkin_const)
    reference_line = xt.Line(elements={'e': parent.copy()})
    line = xt.Line(elements={'e': parent})
    line.slice_thick_elements([xt.Strategy(xt.Uniform(4, mode='thick'))])
    for this_line in (line, reference_line):
        this_line.build_tracker()
        this_line.configure_radiation(model='mean')
    reference = make_particles()
    particles = make_particles()
    reference_line.track(reference, _force_no_end_turn_actions=True)
    line.track(particles, _force_no_end_turn_actions=True)
    assert np.all(reference.delta < make_particles().delta)
    for name in (*COORDS, 's', 'ax', 'ay'):
        xo.assert_allclose(getattr(particles, name), getattr(reference, name),
                           rtol=0, atol=1e-16)


@pytest.mark.parametrize('integrator', INTEGRATORS)
def test_radiation_and_spin_skipped_when_backtracking(integrator):
    element = make_element(integrator, 20)
    line = xt.Line(elements={'e': element})
    line.build_tracker()
    line.configure_radiation(model='mean')
    line.configure_spin(spin_model='auto')
    initial = make_particles()
    initial.spin_z = 1.
    particles = initial.copy()
    line.track(particles, backtrack=True)
    reference = initial.copy()
    line.configure_radiation(model=None)
    line.track(reference, backtrack=True)
    assert np.all(particles.state == 1)
    for name in (*COORDS, 'spin_x', 'spin_y', 'spin_z'):
        xo.assert_allclose(getattr(particles, name), getattr(reference, name),
                           rtol=0, atol=0)


def _robinson_ring(insertion):
    env = xt.Environment()
    env.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=3e9)
    ncells = 16
    env.new('mb', xt.Bend, length=2.0, angle=np.pi / ncells, k0_from_h=True)
    env.new('qf', xt.Quadrupole, length=0.4, k1=0.6)
    env.new('qd', xt.Quadrupole, length=0.4, k1=-0.6)
    cell = env.new_line(components=[
        env.place('qf', at=0.2), env.place('mb', at=2.4), env.place('qd', at=4.6),
        env.place('mb', at=6.8), env.new('cell_end', xt.Marker, at=9.2)])
    ring = env.new_line(components=[cell] * ncells)
    ring.insert([env.new('cav', xt.Cavity, voltage=3e6, frequency=500e6,
                         phase=np.pi, at=0.9)])
    env.elements['ins'] = insertion
    ring.insert([env.place('ins', at=8.5)])  # in the 1.4 m drift of cell 1
    return ring


@pytest.mark.parametrize('integrator', INTEGRATORS)
def test_robinson_theorem(integrator):
    # Robinson's theorem: for any static magnetic lattice the damping
    # partition numbers add up to Jx + Jy + Jz = 4. The BFE insertion has a
    # 3 T wiggler-like By profile (zero integral and first moment, so the
    # orbit closes outside it), a solenoid with its radial fringe field and a
    # skew quadrupole, so it contributes energy loss, damping and coupling.
    # The sum is only 4 if both the energy loss and the radiation kick on the
    # transverse momenta are applied correctly in every integration step.
    length = 1.2
    ksolc = np.array([0., 4 * 0.3 / length, -4 * 0.3 / length**2])
    dks = np.polynomial.Polynomial(ksolc).deriv().coef
    insertion = xt.BFieldExpansion(
        length=length, knc=[0.3 * np.array([1., -6 / length, 6 / length**2])],
        ksolc=ksolc, ksc=[[0.], -0.5 * dks], k1s=0.02,
        integrator=integrator, num_integration_steps=200)
    ring = _robinson_ring(insertion)
    ring.configure_radiation(model='mean')
    tw = ring.twiss(radiation_analysis=True)

    ring['ins'].radiation_flag = 0
    tw_no_ins = ring.twiss(radiation_analysis=True)

    # The insertion adds a few percent of energy loss and changes all modes
    assert tw.energy_loss > 1.03 * tw_no_ins.energy_loss
    assert np.all(np.abs(tw.partition_numbers - tw_no_ins.partition_numbers) > 4e-3)
    xo.assert_allclose(tw.partition_numbers.sum(), 4., atol=3e-4, rtol=0)


# ---------------------------------------------------------------------------
# Spin (cases and tolerances as in the SplineBoris spin tests)
# ---------------------------------------------------------------------------

SPIN_CASES = {
    'base': dict(x=0.001, px=1e-05, y=0.002, py=2e-05, delta=0.001),
    'delta=-0.01': dict(x=0.001, px=1e-05, y=0.002, py=2e-05, delta=-0.01),
    'delta=0.01': dict(x=0.001, px=1e-05, y=0.002, py=2e-05, delta=0.01),
    'px=-0.03, py=-0.02': dict(x=0.001, px=-0.03, y=0.002, py=-0.02, delta=0.001),
    'px=0, py=0': dict(x=0.001, px=0., y=0.002, py=0., delta=0.001),
    'px=0.015, py=0.01': dict(x=0.001, px=0.015, y=0.002, py=0.01, delta=0.001),
}


def _spin_particles(case):
    spin_x, spin_z = 0.1, 0.2
    return xt.Particles(p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
                        anomalous_magnetic_moment=0.00115965218128,
                        spin_x=spin_x, spin_z=spin_z,
                        spin_y=np.sqrt(1 - spin_x**2 - spin_z**2), **case)


def _compare_spin(reference_elements, element, case, atol,
                  reference_case=None, check_orbit=False):
    particles = _spin_particles(case)
    reference = _spin_particles(reference_case or case)
    for these_elements, this_particles in ((reference_elements, reference),
                                           ([element], particles)):
        line = xt.Line(elements=list(these_elements) + [xt.Marker()])
        line.particle_ref = _spin_particles(case)
        line.configure_spin(spin_model='auto')
        line.track(this_particles)
    # Quadrupole and Bend use an expanded drift, BFE the exact one: at large
    # angles their orbits differ, so the orbit is only compared on request.
    names = ('x', 'y') * check_orbit + ('spin_x', 'spin_y', 'spin_z')
    for name in names:
        xo.assert_allclose(getattr(particles, name), getattr(reference, name),
                           atol=atol, rtol=0)
    # The spin has actually moved, well beyond the tolerance
    assert np.abs(particles.spin_x[0] - 0.1) > 100 * atol


@pytest.mark.parametrize('integrator', INTEGRATORS)
@pytest.mark.parametrize('name', list(SPIN_CASES))
def test_spin_uniform_solenoid(integrator, name):
    # 2 T over 1 m: the spin turns by ~1e-3 rad at 700 GeV. Off axis the
    # transverse kinetic momentum matters (a*gamma ~ 1600 amplifies its
    # tilt with respect to Bs), so both elements must start with the same
    # kinetic momentum. BFE with pkin_const=True has no edge kick; the
    # UniformSolenoid uses k = p - a with the symmetric gauge, so its
    # canonical momenta are shifted by a. BFE's own gauge differs
    # (Ax = -ks*y, Ay = 0), so pkin_const=False would not be comparable.
    # UniformSolenoid evaluates spin once per element: split it to converge.
    ks, length, npieces = 2.0 / (700e9 / clight), 1.0, 100
    case = SPIN_CASES[name]
    reference_case = dict(case, px=case['px'] - 0.5 * ks * case['y'],
                          py=case['py'] + 0.5 * ks * case['x'])
    atol = 2e-7 if 'px=-0.03' in name or 'px=0.015' in name else 1e-9
    _compare_spin([xt.UniformSolenoid(length=length / npieces, ks=ks)
                   for _ in range(npieces)],
                  xt.BFieldExpansion(length=length, ksolc=[ks], integrator=integrator,
                                     pkin_const=True, num_integration_steps=200),
                  case, atol, reference_case=reference_case, check_orbit=True)


@pytest.mark.parametrize('integrator', INTEGRATORS)
@pytest.mark.parametrize('name', list(SPIN_CASES))
def test_spin_quadrupole(integrator, name):
    k1, length = 0.01, 0.02
    atol = 1e-7 if 'px=-0.03' in name or 'px=0.015' in name else 1e-9
    _compare_spin([xt.Quadrupole(k1=k1, length=length)],
                  xt.BFieldExpansion(length=length, k1=k1, integrator=integrator,
                                     num_integration_steps=100),
                  SPIN_CASES[name], atol)


@pytest.mark.parametrize('integrator', INTEGRATORS)
def test_spin_dipole(integrator):
    k0, length = 0.01, 0.02
    _compare_spin([xt.Bend(length=length, angle=0., k0=k0)],
                  xt.BFieldExpansion(length=length, k0=k0, integrator=integrator,
                                     num_integration_steps=100),
                  SPIN_CASES['base'], 1e-9)


# ---------------------------------------------------------------------------
# Curved frame: the half drifts are exact sector bends in the design field
# b0 = h, the kicks apply By - b0 (and Bx), the rotation Bs.
# ---------------------------------------------------------------------------

def make_bent_element(integrator='boris', num_integration_steps=40,
                      pkin_const=False, h=0.4):
    # Design dipole plus s-dependent dipole error, quadrupole, sextupole,
    # skew quadrupole and a solenoid ramp, all in curved-frame components.
    return xt.BFieldExpansion(
        length=1.0, h=h, k0=h, s_start=0.05,
        knc=[[0.0, 0.02, -0.02], [0.3, -0.2, 0.1], [2.0, 1.0]],
        ksc=[[0.], [0.05, 0.05]], ksolc=[0.1, 0.3, -0.3],
        integrator=integrator, num_integration_steps=num_integration_steps,
        pkin_const=pkin_const)


def make_bent_particles():
    return xt.Particles(
        p0c=2e9, x=[1e-3, -2e-3, 4e-3, 0.], px=[2e-4, -1e-3, 3e-3, 0.],
        y=[-1.5e-3, 2e-3, 3e-3, 0.], py=[1e-4, 5e-4, -2e-3, 0.],
        zeta=[0., 1e-3, -1e-3, 0.], delta=[1e-3, -2e-3, 5e-3, 0.],
        chi=[1., 1.01, 0.99, 1.], mass_ratio=[1., 1.01, 1., 1.])


@pytest.mark.parametrize('pkin_const', [False, True])
def test_boris_bent_converges_to_rk4_at_second_order(pkin_const):
    # The bent RK4 integrates the canonical Hamiltonian, independently of the
    # kinetic-momentum formulation of the Boris step.
    reference = track(make_bent_element('rk4', 4000, pkin_const),
                      make_bent_particles())
    errors = []
    for nstep in (20, 40, 80, 160):
        particles = track(make_bent_element('boris', nstep, pkin_const),
                          make_bent_particles())
        assert np.all(particles.state == 1)
        xo.assert_allclose(particles.s, reference.s, rtol=0, atol=1e-15)
        errors.append(max(np.max(np.abs(getattr(particles, name)
                                        - getattr(reference, name)))
                          for name in (*COORDS, 'ax', 'ay')))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    xo.assert_allclose(orders, 2., rtol=0, atol=0.05)
    assert errors[-1] < 3e-7


@pytest.mark.parametrize('pkin_const', [False, True])
def test_boris_bent_backtrack_retraces_forward_steps(pkin_const):
    line = xt.Line(elements={'e': make_bent_element('boris', 37, pkin_const)})
    initial = make_bent_particles()
    particles = initial.copy()
    line.track(particles)
    line.track(particles, backtrack=True)
    assert np.all(particles.state == 1)
    for name in (*COORDS, 's'):
        xo.assert_allclose(getattr(particles, name), getattr(initial, name),
                           rtol=0, atol=1e-15)


@pytest.mark.parametrize('pkin_const', [False, True])
def test_boris_bent_thick_slices_match_parent(pkin_const):
    parent = make_bent_element('boris', 40, pkin_const)
    reference = track(parent.copy(), make_bent_particles())
    line = xt.Line(elements={'e': parent})
    line.slice_thick_elements([xt.Strategy(xt.Uniform(4, mode='thick'))])
    particles = make_bent_particles()
    line.track(particles, _force_no_end_turn_actions=True)
    for name in (*COORDS, 's', 'ax', 'ay'):
        xo.assert_allclose(getattr(particles, name), getattr(reference, name),
                           rtol=0, atol=1e-15)


@pytest.mark.parametrize('nstep', [1, 7])
def test_boris_bent_design_dipole_is_exact(nstep):
    # In the pure design field the step is the exact sector-bend map: the
    # reference particle stays on the frame and off-momentum particles (and
    # other charge-to-mass ratios) follow the exact xt.Bend map, for any
    # number of steps.
    h = 0.4
    element = xt.BFieldExpansion(length=1.0, h=h, k0=h, integrator='boris',
                                 num_integration_steps=nstep)
    reference = xt.Particles(p0c=2e9, x=0., px=0., y=0., py=0.)
    element.track(reference)
    for name in ('x', 'px', 'y', 'py'):
        xo.assert_allclose(getattr(reference, name), 0., rtol=0, atol=1e-17)
    xo.assert_allclose(reference.zeta, 0., rtol=0, atol=1e-16)

    bend = xt.Bend(length=1.0, angle=h, k0=h, model='bend-kick-bend',
                   edge_entry_active=False, edge_exit_active=False)
    particles = track(element, make_bent_particles())
    expected = track(bend, make_bent_particles())
    for name in (*COORDS, 's'):
        xo.assert_allclose(getattr(particles, name), getattr(expected, name),
                           rtol=0, atol=2e-15)


@pytest.mark.parametrize('integrator', INTEGRATORS)
def test_bent_spin_matches_bend(integrator):
    # Uniform dipole with a quadrupole: the spin turns by ~1e-2 relative to the
    # frame. magnet_spin gets the frame rotation h*ds per step; without it the
    # spin would be off by ~h*length = 4e-2.
    p0c, length, h, k1 = 45.6e9, 2.0, 0.02, 0.05

    def spin_particles():
        return xt.Particles(
            p0c=p0c, mass0=xt.ELECTRON_MASS_EV,
            anomalous_magnetic_moment=0.00115965218128,
            x=[0., 1e-3, -2e-3], px=[0., 1e-4, -2e-4], y=[0., 5e-4, 1e-3],
            py=[0., -1e-4, 2e-4], delta=[0., 1e-3, -2e-3],
            spin_x=0.1, spin_y=np.sqrt(1 - 0.05), spin_z=0.2)

    out = []
    for element in (
            xt.BFieldExpansion(length=length, h=h, k0=h, k1=k1,
                               integrator=integrator, num_integration_steps=1000),
            xt.Bend(length=length, angle=h * length, k0=h, k1=k1,
                    model='bend-kick-bend', num_multipole_kicks=10000,
                    edge_entry_active=False, edge_exit_active=False)):
        line = xt.Line(elements=[element])
        line.particle_ref = spin_particles()
        line.configure_spin(spin_model='auto')
        particles = spin_particles()
        line.track(particles)
        out.append(particles)
    bfe, bend = out
    for name in ('spin_x', 'spin_y', 'spin_z'):
        xo.assert_allclose(getattr(bfe, name), getattr(bend, name),
                           rtol=0, atol=2e-7)
    assert np.all(np.abs(bfe.spin_x - 0.1) > 1e-2)


def _bent_ring(use_bfe, nstep=50, integrator='boris'):
    env = xt.Environment()
    env.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=3e9)
    ncells, length = 16, 2.0
    angle = np.pi / ncells
    if use_bfe:
        env.elements['mb'] = xt.BFieldExpansion(
            length=length, h=angle / length, k0=angle / length, k1=0.01,
            integrator=integrator, num_integration_steps=nstep)
    else:
        env.new('mb', xt.Bend, length=length, angle=angle, k0_from_h=True,
                k1=0.01, model='bend-kick-bend', num_multipole_kicks=nstep,
                edge_entry_active=False, edge_exit_active=False)
    env.new('qf', xt.Quadrupole, length=0.4, k1=0.6)
    env.new('qd', xt.Quadrupole, length=0.4, k1=-0.6)
    cell = env.new_line(components=[
        env.place('qf', at=0.2), env.place('mb', at=2.4), env.place('qd', at=4.6),
        env.place('mb', at=6.8), env.new('cell_end', xt.Marker, at=9.2)])
    ring = env.new_line(components=[cell] * ncells)
    ring.insert([env.new('cav', xt.Cavity, voltage=3e6, frequency=500e6,
                         phase=np.pi, at=0.9)])
    return ring


@pytest.mark.parametrize('integrator', INTEGRATORS)
def test_bent_ring_matches_bend_ring(integrator):
    # A ring whose bends are all curved Boris BFieldExpansions has the optics
    # and radiation properties of the same ring with exact xt.Bend elements,
    # and the damping partition numbers add up to 4 (Robinson's theorem).
    results = []
    for use_bfe in (True, False):
        ring = _bent_ring(use_bfe, integrator=integrator)
        tw4d = ring.twiss4d()
        ring.configure_radiation(model='mean')
        results.append((tw4d, ring.twiss(radiation_analysis=True)))
    (tw4d, tw), (tw4d_bend, tw_bend) = results
    xo.assert_allclose(tw4d.qx, tw4d_bend.qx, rtol=0, atol=1e-5)
    xo.assert_allclose(tw4d.qy, tw4d_bend.qy, rtol=0, atol=1e-5)
    xo.assert_allclose(tw4d.dqx, tw4d_bend.dqx, rtol=0, atol=1e-4)
    xo.assert_allclose(tw.energy_loss, tw_bend.energy_loss, rtol=1e-6, atol=0)
    xo.assert_allclose(tw.partition_numbers, tw_bend.partition_numbers,
                       rtol=0, atol=1e-3)
    xo.assert_allclose(tw.partition_numbers.sum(), 4., rtol=0, atol=2e-4)
