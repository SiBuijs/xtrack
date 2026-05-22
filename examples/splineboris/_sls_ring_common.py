"""
Shared SLS ring setup: RF cavities, field fit, undulator build/match, spin twiss.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xtrack as xt
from xtrack._temp.splineboris.field_fitter import FieldFitter
from xtrack._temp.splineboris.splineboris_sequence import SplineBorisSequence

from _undulator_multipole_builder import build_multipole_kick_undulator

E0 = 2.7e9  # eV
ELECTRON_AMM = 0.00115965218128

V_RF_MAIN_STRAIGHT = 890e3  # V per short straight (2 cavities × 445 kV)
F_RF_MAIN = 499.6537e6  # Hz
LAG_RF_MAIN = 237.0  # deg

V_RF_3HC = 540e3  # V
F_RF_3HC = 1498.95e6  # Hz
LAG_RF_3HC = 270.0  # deg

MAIN_RF_ANCHORS = ("ars02_gsrc_0500", "ars08_gsrc_0500")
HC3_ANCHOR = "ars07_gsrc_0390"

WIGGLER_PLACE = "ars11_uind_0210_1"
DEFAULT_MULTIPOLE_ORDER = 3

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MADX_FILE = _REPO_ROOT / "test_data" / "sls" / "sls.madx"
FIELD_MAP_FILE = _REPO_ROOT / "test_data" / "sls" / "simona_field_map.txt"


def default_particle_ref():
    return xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)


def build_sls_ring_with_rf(env, line):
    """Insert main RF and third-harmonic cavities (005a pattern)."""
    env["vrf_main_straight"] = V_RF_MAIN_STRAIGHT
    env["frf_main"] = F_RF_MAIN
    env["lag_main"] = LAG_RF_MAIN
    env["vrf_3hc"] = V_RF_3HC
    env["frf_3hc"] = F_RF_3HC
    env["lag_3hc"] = LAG_RF_3HC

    env.new(
        "cav_main_x02",
        xt.Cavity,
        voltage="vrf_main_straight",
        frequency="frf_main",
        lag="lag_main",
    )
    env.new(
        "cav_main_x08",
        xt.Cavity,
        voltage="vrf_main_straight",
        frequency="frf_main",
        lag="lag_main",
    )
    env.new(
        "cav_3hc_x07",
        xt.Cavity,
        voltage="vrf_3hc",
        frequency="frf_3hc",
        lag="lag_3hc",
    )

    line.insert(
        [
            env.place("cav_main_x02", at=f"{MAIN_RF_ANCHORS[0]}@start"),
            env.place("cav_main_x08", at=f"{MAIN_RF_ANCHORS[1]}@start"),
            env.place("cav_3hc_x07", at=f"{HC3_ANCHOR}@start"),
        ]
    )


def load_field_fit(multipole_order=DEFAULT_MULTIPOLE_ORDER):
    """FieldFitter on simona_field_map.txt (007 pattern)."""
    df_raw_data = pd.read_csv(
        FIELD_MAP_FILE,
        sep=r"\s+",
        header=None,
        names=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
    ).set_index(["X", "Y", "Z"])

    return FieldFitter(
        raw_data=df_raw_data,
        xy_point=(0, 0),
        distance_unit=0.001,
        min_region_size=10,
        deg=multipole_order - 1,
        field_tol=1e-4,
    )


def new_ring(p0=None):
    """Fresh SLS ring with bend model, particle ref, and RF cavities."""
    if p0 is None:
        p0 = default_particle_ref()
    env = xt.load(str(MADX_FILE))
    line = env.ring
    line.configure_bend_model(core="mat-kick-mat")
    line.particle_ref = p0.copy()
    build_sls_ring_with_rf(env, line)
    return env, line


def _setup_corrector_env(env):
    for key in (
        "k0l_corr1",
        "k0l_corr2",
        "k0l_corr3",
        "k0l_corr4",
        "k0sl_corr1",
        "k0sl_corr2",
        "k0sl_corr3",
        "k0sl_corr4",
    ):
        env[key] = 0.0

    env.new("corr1", xt.Multipole, knl=["k0l_corr1"], ksl=["k0sl_corr1"])
    env.new("corr2", xt.Multipole, knl=["k0l_corr2"], ksl=["k0sl_corr2"])
    env.new("corr3", xt.Multipole, knl=["k0l_corr3"], ksl=["k0sl_corr3"])
    env.new("corr4", xt.Multipole, knl=["k0l_corr4"], ksl=["k0sl_corr4"])


def _match_orbit_correctors(undulator_line, env, l_wig):
    _setup_corrector_env(env)
    undulator_line.insert(
        [
            env.place("corr1", at=0.02),
            env.place("corr2", at=0.1),
            env.place("corr3", at=l_wig - 0.1),
            env.place("corr4", at=l_wig - 0.02),
        ],
        s_tol=5e-3,
    )

    undulator_line.build_tracker()
    opt = undulator_line.match(
        solve=False,
        betx=0,
        bety=0,
        only_orbit=True,
        include_collective=True,
        vary=xt.VaryList(
            [
                "k0l_corr1",
                "k0sl_corr1",
                "k0l_corr2",
                "k0sl_corr2",
                "k0l_corr3",
                "k0sl_corr3",
                "k0l_corr4",
                "k0sl_corr4",
            ],
            step=1e-6,
        ),
        targets=[
            xt.Target(lambda tw: np.mean(tw.x), value=0.0, tol=1e-8, tag="avg_orbit"),
            xt.Target(lambda tw: np.mean(tw.y), value=0.0, tol=1e-8, tag="avg_orbit"),
            xt.TargetSet(x=0, px=0, y=0, py=0.0, at=xt.END),
        ],
    )
    opt.step(2)
    undulator_line.discard_tracker()


def build_splineboris_undulator(env, p0, field_fitter, multipole_order=DEFAULT_MULTIPOLE_ORDER):
    """SplineBorisSequence line with orbit correctors (007 pattern)."""
    seq = SplineBorisSequence(
        df_fit_pars=field_fitter.df_fit_pars,
        multipole_order=multipole_order,
        steps_per_point=1,
    )
    undulator_line = seq.to_line(env=env)
    undulator_line.particle_ref = p0.copy()
    _match_orbit_correctors(undulator_line, env, seq.length)
    return undulator_line, seq.length


def build_multipole_undulator(
    env,
    p0,
    field_fitter,
    multipole_order=DEFAULT_MULTIPOLE_ORDER,
    name_prefix="und_kick",
):
    """Thin MultipoleKick undulator with orbit correctors."""
    undulator_line, l_wig = build_multipole_kick_undulator(
        env=env,
        p_ref=p0,
        df_fit_pars=field_fitter.df_fit_pars,
        multipole_order=multipole_order,
        name_prefix=name_prefix,
    )
    _match_orbit_correctors(undulator_line, env, l_wig)
    return undulator_line, l_wig


def insert_undulator(ring, undulator_line, place=WIGGLER_PLACE):
    s_wig = float(ring.get_table()["s", place])
    ring.insert(undulator_line, anchor="start", at=s_wig)
    return s_wig


def configure_tracking_spin(ring):
    ring.configure_spin("auto")
    ring.particle_ref.anomalous_magnetic_moment = ELECTRON_AMM


def make_offset_bunch(
    ring,
    tw,
    num_particles=100,
    seed=0,
    sigma_x=1e-4,
    sigma_px=1e-4,
    sigma_y=1e-4,
    sigma_py=1e-4,
    sigma_zeta=1e-4,
    sigma_delta=1e-4,
):
    """Particles on CO + Gaussian offsets; common spin along y."""
    rng = np.random.default_rng(seed)
    n = num_particles

    def draws(sigma):
        if sigma == 0:
            return 0.0
        return rng.normal(0.0, sigma, n)

    particles = ring.build_particles(
        particle_on_co=tw.particle_on_co,
        mode="shift",
        x=draws(sigma_x),
        px=draws(sigma_px),
        y=draws(sigma_y),
        py=draws(sigma_py),
        zeta=draws(sigma_zeta),
        delta=draws(sigma_delta),
    )
    particles.spin_x = 0.0
    particles.spin_y = 1.0
    particles.spin_z = 0.0
    return particles


def twiss_spin(ring, radiation_model="mean"):
    ring.configure_radiation(model=radiation_model)
    ring.build_tracker()
    return ring.twiss(
        radiation_integrals=True,
        spin=True,
        polarization=True,
        radiation_method="full",
    )


def twiss_summary_dict(tw):
    out = {
        "qx": float(tw.qx),
        "qy": float(tw.qy),
        "qs": float(tw.qs),
        "spin_polarization_eq": float(tw.spin_polarization_eq),
    }
    if hasattr(tw, "spin_t_pol_buildup_s"):
        out["spin_t_pol_buildup_s"] = float(tw.spin_t_pol_buildup_s)
    if hasattr(tw, "T_rev0"):
        t_build = out.get("spin_t_pol_buildup_s")
        if t_build is not None and tw.T_rev0 > 0:
            out["spin_t_pol_buildup_turns"] = t_build / float(tw.T_rev0)
    return out
