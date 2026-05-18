import xtrack as xt
import xobjects as xo
import pathlib
import numpy as np
import pytest

fcc_ee_lattice_folder = pathlib.Path(__file__).parent.parent

@pytest.mark.parametrize('ftype', ['py', 'json', 'cpymad'])
def test_twiss_z_lattice(ftype):
    if ftype == 'py':
        # Load lattice from python files
        env = xt.Environment()
        env.call(fcc_ee_lattice_folder / 'lattices/z/fccee_z_lattice.py')
        env.call(fcc_ee_lattice_folder / 'lattices/z/fccee_z_strengths.py')
        line = env['fccee_p_ring']
    elif ftype == 'json':
        env = xt.load(fcc_ee_lattice_folder / 'lattices/z/fccee_z.json')
        line = env.fccee_p_ring
    elif ftype == 'cpymad':
        from cpymad.madx import Madx
        mad = Madx(stdout=False)
        mad.call(str(fcc_ee_lattice_folder / 'lattices/z/fccee_z.madx'))
        mad.beam()
        mad.use('fccee_p_ring')
        line = xt.Line.from_madx_sequence(mad.sequence.fccee_p_ring,
                                        deferred_expressions=True)
        line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=45.6e9)

        tt = line.get_table()
        tt_bend = tt.rows[(tt.element_type=='Bend') | (tt.element_type=='RBend')]
        tt_quad = tt.rows[(tt.element_type=='Quadrupole')]
        tt_sext = tt.rows[(tt.element_type=='Sextupole')]

        line.set(tt_bend, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')
        line.set(tt_quad, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')
        line.set(tt_sext, integrator='yoshida4', num_multipole_kicks=1)

    tw4d = line.twiss4d()
    tw6d = line.twiss()

    tt = line.get_table(attr=True)
    tt_cav = tt.rows[tt.element_type=='Cavity']
    xo.assert_allclose(tt_cav.phase, np.pi, atol=1e-10, rtol=0)
    xo.assert_allclose(tt_cav.rows['.*400.*'].frequency, 400.0e6, rtol=0.01)
    xo.assert_allclose(tt_cav.rows['.*800.*'].frequency, 800.0e6, rtol=0.01)

    for tw in [tw4d, tw6d]:

        # Check that there is no orbit and no vertical dispersion in the arcs
        xo.assert_allclose(tw.x, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw.y, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw.rows['qf2a.*'].dy, 0, rtol=0, atol=2e-8) # checking one arc family

        xo.assert_allclose(tw.qx, 194.160148, atol=1e-5, rtol=0)
        xo.assert_allclose(tw.qy, 170.199897, atol=1e-5, rtol=0)
        xo.assert_allclose(tw.dqx, 12.5, atol=0.5, rtol=0)
        xo.assert_allclose(tw.dqy, 5.5, atol=0.5, rtol=0)
        xo.assert_allclose(tw.momentum_compaction_factor, 2.86e-5, atol=1e-7, rtol=0)
        xo.assert_allclose(tw.c_minus, 0.0, atol=2e-7, rtol=0)

        tw_ips = tw.rows['ip.*']
        xo.assert_allclose(tw_ips.betx, 0.09, atol=0, rtol=6.5e-5)
        xo.assert_allclose(tw_ips.bety, 0.0007, atol=0, rtol=5e-5)
        xo.assert_allclose(tw_ips.x, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw_ips.y, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw_ips.delta, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw_ips.zeta, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw_ips.dx, 0, rtol=0, atol=3e-6)
        xo.assert_allclose(tw_ips.dy, 0, rtol=0, atol=3e-6)
        xo.assert_allclose(tw_ips.dpx, 0, rtol=0, atol=1e-5)
        xo.assert_allclose(tw_ips.dpy, 0, rtol=0, atol=1e-5)
        assert np.all(tw_ips.wy_chrom < 10)

    xo.assert_allclose(tw6d.qs, 0.03307785923, atol=1e-4, rtol=0)

    if ftype == 'cpymad':

        # Check the madx results
        mad.elements['rf400'].volt = 0
        tw_mad = mad.twiss(chrom=True)
        xo.assert_allclose(tw_mad.summary.q1, 194.160148, atol=1e-5, rtol=0)
        xo.assert_allclose(tw_mad.summary.q2, 170.199897, atol=1e-5, rtol=0)

        # Madx chromaticities are quite off
        xo.assert_allclose(tw_mad.summary.dq1, 12.5, atol=0.5, rtol=0)
        xo.assert_allclose(tw_mad.summary.dq2, 5.5, atol=0.5, rtol=0)

        tw_ips_mad = xt.Table(tw_mad).rows['ip.*']
        xo.assert_allclose(tw_ips_mad.betx, 0.09, atol=0, rtol=6.5e-5)
        xo.assert_allclose(tw_ips_mad.bety, 0.0007, atol=0, rtol=5e-5)
        assert np.all(tw_ips_mad.wx < 10.)
        assert np.all(tw_ips_mad.wy < 10.)

@pytest.mark.parametrize('ftype', ['py', 'json', 'cpymad'])
def test_twiss_t_lattice(ftype):
    if ftype == 'py':
        # Load lattice from python files
        env = xt.Environment()
        env.call(fcc_ee_lattice_folder / 'lattices/t/fccee_t_lattice.py')
        env.call(fcc_ee_lattice_folder / 'lattices/t/fccee_t_strengths.py')
        line = env['fccee_p_ring']
    elif ftype == 'json':
        env = xt.load(fcc_ee_lattice_folder / 'lattices/t/fccee_t.json')
        line = env.fccee_p_ring
    elif ftype == 'cpymad':
        from cpymad.madx import Madx
        mad = Madx(stdout=False)
        mad.call(str(fcc_ee_lattice_folder / 'lattices/t/fccee_t.madx'))
        mad.beam()
        mad.use('fccee_p_ring')
        line = xt.Line.from_madx_sequence(mad.sequence.fccee_p_ring,
                                        deferred_expressions=True)
        line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=182.5e9)

        tt = line.get_table()
        tt_bend = tt.rows[(tt.element_type=='Bend') | (tt.element_type=='RBend')]
        tt_quad = tt.rows[(tt.element_type=='Quadrupole')]
        tt_sext = tt.rows[(tt.element_type=='Sextupole')]

        line.set(tt_bend, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')
        line.set(tt_quad, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')
        line.set(tt_sext, integrator='yoshida4', num_multipole_kicks=1)

    tw4d = line.twiss4d()
    tw6d = line.twiss()
    tt = line.get_table(attr=True)
    tt_cav = tt.rows[tt.element_type=='Cavity']
    xo.assert_allclose(tt_cav.phase, np.pi, atol=1e-10, rtol=0)
    xo.assert_allclose(tt_cav.rows['.*400.*'].frequency, 400.0e6, rtol=0.01)
    xo.assert_allclose(tt_cav.rows['.*800.*'].frequency, 800.0e6, rtol=0.01)

    for tw in [tw4d, tw6d]:

        # Check that there is no orbit and no vertical dispersion in the arcs
        xo.assert_allclose(tw.x, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw.y, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw.rows['qf2a.*'].dy, 0, rtol=0, atol=2e-8) # checking one arc quad family

        xo.assert_allclose(tw.qx, 346.130347, atol=3e-5, rtol=0)
        xo.assert_allclose(tw.qy, 262.269742, atol=3e-5, rtol=0)
        xo.assert_allclose(tw.dqx, 1.6, atol=0.3, rtol=0)
        xo.assert_allclose(tw.dqy, 1.4, atol=0.3, rtol=0)
        xo.assert_allclose(tw.momentum_compaction_factor, 9.411578e-6, atol=1e-8, rtol=0)
        xo.assert_allclose(tw.c_minus, 0.0, atol=2e-7, rtol=0)

        tw_ips = tw.rows['ip.*']
        xo.assert_allclose(tw_ips.betx, 0.9, atol=0, rtol=3e-4)
        xo.assert_allclose(tw_ips.bety, 0.0014, atol=0, rtol=3.5e-4)
        xo.assert_allclose(tw_ips.x, 0, rtol=0, atol=1e-12)
        xo.assert_allclose(tw_ips.y, 0, rtol=0, atol=1e-12)
        xo.assert_allclose(tw_ips.delta, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw_ips.zeta, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw_ips.dx, 0, rtol=0, atol=3e-5)
        xo.assert_allclose(tw_ips.dy, 0, rtol=0, atol=3e-5)
        xo.assert_allclose(tw_ips.dpx, 0, rtol=0, atol=5.4e-5)
        xo.assert_allclose(tw_ips.dpy, 0, rtol=0, atol=3e-5)
        assert np.all(tw_ips.wx_chrom < 8)
        assert np.all(tw_ips.wy_chrom < 5.6)

    xo.assert_allclose(tw6d.qs, 0.13854873549977959, atol=2e-4, rtol=0)

    if ftype == 'cpymad':

        # Check the madx results (we go to 4d, in 6d madx chromatic properties are qure off)
        mad.elements['rf400.0'].volt = 0
        mad.elements['rf800.0'].volt = 0
        mad.elements['rf800.1'].volt = 0
        mad.elements['rf400.1'].volt = 0
        tw_mad = mad.twiss()
        xo.assert_allclose(tw_mad.summary.q1, 346.130347, atol=5e-5, rtol=0)
        xo.assert_allclose(tw_mad.summary.q2, 262.269742, atol=5e-5, rtol=0)

        xo.assert_allclose(tw_mad.summary.dq1, 1.6, atol=0.3, rtol=0)
        xo.assert_allclose(tw_mad.summary.dq2, 1.4, atol=0.3, rtol=0)

        tw_ips_mad = xt.Table(tw_mad).rows['ip.*']
        xo.assert_allclose(tw_ips_mad.betx, 0.9, atol=0, rtol=3e-4)
        xo.assert_allclose(tw_ips_mad.bety, 0.0014, atol=0, rtol=3.5e-4)
        assert np.all(tw_ips_mad.wx < 8.)
        assert np.all(tw_ips_mad.wy < 5.6)

@pytest.mark.parametrize('ftype', ['py', 'json', 'cpymad'])
def test_twiss_w_lattice(ftype):
    if ftype == 'py':
        # Load lattice from python files
        env = xt.Environment()
        env.call(fcc_ee_lattice_folder / 'lattices/w/fccee_w_lattice.py')
        env.call(fcc_ee_lattice_folder / 'lattices/w/fccee_w_strengths.py')
        line = env['fccee_p_ring']
    elif ftype == 'json':
        env = xt.load(fcc_ee_lattice_folder / 'lattices/w/fccee_w.json')
        line = env.fccee_p_ring
    elif ftype == 'cpymad':
        from cpymad.madx import Madx
        mad = Madx(stdout=False)
        mad.call(str(fcc_ee_lattice_folder / 'lattices/w/fccee_w.madx'))
        mad.beam()
        mad.use('fccee_p_ring')
        line = xt.Line.from_madx_sequence(mad.sequence.fccee_p_ring,
                                        deferred_expressions=True)
        line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=80.0e9)

        tt = line.get_table()
        tt_bend = tt.rows[(tt.element_type=='Bend') | (tt.element_type=='RBend')]
        tt_quad = tt.rows[(tt.element_type=='Quadrupole')]
        tt_sext = tt.rows[(tt.element_type=='Sextupole')]

        line.set(tt_bend, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')
        line.set(tt_quad, integrator='uniform', num_multipole_kicks=3, model='mat-kick-mat')
        line.set(tt_sext, integrator='yoshida4', num_multipole_kicks=1)

    tw4d = line.twiss4d()
    tw6d = line.twiss()

    tt = line.get_table(attr=True)
    tt_cav = tt.rows[tt.element_type=='Cavity']
    xo.assert_allclose(tt_cav.phase, np.pi, atol=1e-10, rtol=0)
    xo.assert_allclose(tt_cav.rows['.*400.*'].frequency, 400.0e6, rtol=0.01)
    xo.assert_allclose(tt_cav.rows['.*800.*'].frequency, 800.0e6, rtol=0.01)

    for tw in [tw4d, tw6d]:

        # Check that there is no orbit and no vertical dispersion in the arcs
        xo.assert_allclose(tw.x, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw.y, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw.rows['qf2a.*'].dy, 0, rtol=0, atol=2e-8) # checking one arc family

        xo.assert_allclose(tw.qx,194.179148, atol=1e-5, rtol=0)
        xo.assert_allclose(tw.qy,170.240898, atol=1e-5, rtol=0)
        xo.assert_allclose(tw.dqx, 0.5, atol=0.5, rtol=0)
        xo.assert_allclose(tw.dqy, 3.0, atol=0.1, rtol=0)
        xo.assert_allclose(tw.momentum_compaction_factor, 2.86e-5, atol=1e-7, rtol=0)
        xo.assert_allclose(tw.c_minus, 0.0, atol=2e-7, rtol=0)

        tw_ips = tw.rows['ip.*']
        xo.assert_allclose(tw_ips.betx, 0.22, atol=0, rtol=2.0e-4)
        xo.assert_allclose(tw_ips.bety, 0.0010, atol=0, rtol=4.1e-4)
        xo.assert_allclose(tw_ips.x, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw_ips.y, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw_ips.delta, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw_ips.zeta, 0, rtol=0, atol=1e-10)
        xo.assert_allclose(tw_ips.dx, 0, rtol=0, atol=6.0e-6)
        xo.assert_allclose(tw_ips.dy, 0, rtol=0, atol=1.0e-6)
        xo.assert_allclose(tw_ips.dpx, 0, rtol=0, atol=1e-5)
        xo.assert_allclose(tw_ips.dpy, 0, rtol=0, atol=1e-5)
        assert np.all(tw_ips.wy_chrom < 10)

    xo.assert_allclose(tw6d.qs,0.084061, atol=1e-4, rtol=0)

    if ftype == 'cpymad':

        # Check the madx results
        mad.elements['rf400'].volt = 0
        tw_mad = mad.twiss(chrom=True)
        xo.assert_allclose(tw_mad.summary.q1, 194.179148, atol=1e-5, rtol=0)
        xo.assert_allclose(tw_mad.summary.q2, 170.240898, atol=1e-5, rtol=0)

        # Madx chromaticities are quite off
        xo.assert_allclose(tw_mad.summary.dq1, 0.5 , atol=0.5, rtol=0)
        xo.assert_allclose(tw_mad.summary.dq2, 3.0, atol=0.1, rtol=0)

        tw_ips_mad = xt.Table(tw_mad).rows['ip.*']
        xo.assert_allclose(tw_ips_mad.betx, 0.22, atol=0, rtol=2.0e-4)
        xo.assert_allclose(tw_ips_mad.bety, 0.0010, atol=0, rtol=4.1e-4)
        assert np.all(tw_ips_mad.wx < 10.)
        assert np.all(tw_ips_mad.wy < 10.)


