"""
Benchmark minimal Twiss runtime for SLS with and without undulators.

This script reuses the undulator construction strategy from
`004a_sls_with_undulators.py`, but only times the Twiss call itself.
Line construction, fitting, insertion, and tracker build are done outside
the timing window.
"""

from pathlib import Path
import time
from statistics import median

import numpy as np
import pandas as pd
import xtrack as xt

from xtrack._temp.field_fitter import FieldFitter
from xtrack._temp.splineboris_sequence import SplineBorisSequence


BASE_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = BASE_DIR.parent.parent / "test_data" / "sls"
MADX_FILE = TEST_DATA_DIR / "sls.madx"
FIELD_MAP_FILE = TEST_DATA_DIR / "undulator_field_map.txt"

WIGGLER_PLACES = [
    "ars02_uind_0500_1",
    "ars03_uind_0380_1",
    "ars04_uind_0500_1",
    "ars05_uind_0650_1",
    "ars06_uind_0500_1",
    "ars07_uind_0200_1",
    "ars08_uind_0500_1",
    "ars09_uind_0790_1",
    "ars11_uind_0210_1",
    "ars11_uind_0610_1",
    "ars12_uind_0500_1",
]


def make_reference_particle():
    return xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)


def make_sls_line():
    env = xt.load(str(MADX_FILE))
    line = env.ring
    line.configure_bend_model(core="mat-kick-mat")
    line.particle_ref = make_reference_particle()
    return env, line


def _make_seq_field_evaluator(seq):
    """Vectorized field evaluator backed by SplineBorisSequence/SplineBoris."""
    n_pieces = len(seq.elements)

    def _eval_field(x, y, s):
        x_arr, y_arr, s_arr = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(s, dtype=float),
        )

        bx = np.zeros_like(x_arr, dtype=float)
        by = np.zeros_like(x_arr, dtype=float)
        bs = np.zeros_like(x_arr, dtype=float)
        assigned = np.zeros_like(x_arr, dtype=bool)

        for ii, (elem, s_start, s_end) in enumerate(
            zip(seq.elements, seq.s_starts, seq.s_ends)
        ):
            if ii < n_pieces - 1:
                mask = (s_arr >= s_start) & (s_arr < s_end)
            else:
                mask = (s_arr >= s_start) & (s_arr <= s_end)
            if not np.any(mask):
                continue
            bx_i, by_i, bs_i = elem.get_field(
                x_arr[mask],
                y_arr[mask],
                s_arr[mask] - s_start,
            )
            bx[mask] = bx_i
            by[mask] = by_i
            bs[mask] = bs_i
            assigned[mask] = True

        if not np.all(assigned):
            s_bad = s_arr[~assigned]
            raise ValueError(
                f"s contains values outside the sequence range "
                f"[{min(seq.s_starts)}, {max(seq.s_ends)}], "
                f"bad min/max: {float(np.min(s_bad))} / {float(np.max(s_bad))}"
            )

        if bx.shape == ():
            return float(bx), float(by), float(bs)
        return bx, by, bs

    return _eval_field


def build_piecewise_undulator(env, multipole_order=3, integrator="splineboris"):
    if integrator not in ("splineboris", "boris_spatial"):
        raise ValueError(f"Unknown integrator '{integrator}'")

    df_raw_data = pd.read_csv(
        FIELD_MAP_FILE,
        sep="\t",
        header=None,
        names=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
    ).set_index(["X", "Y", "Z"])

    field_fitter = FieldFitter(
        raw_data=df_raw_data,
        xy_point=(0, 0),
        distance_unit=0.001,
        min_region_size=5,
        deg=multipole_order - 1,
    )
    field_fitter.fit()

    seq = SplineBorisSequence(
        df_fit_pars=field_fitter.df_fit_pars,
        multipole_order=multipole_order,
        steps_per_point=1,
    )
    seq.to_line(env=env)  # registers sequence elements in env

    l_wig = seq.length
    env["k0l_corr1"] = 0.0
    env["k0l_corr2"] = 0.0
    env["k0l_corr3"] = 0.0
    env["k0l_corr4"] = 0.0
    env["k0sl_corr1"] = 0.0
    env["k0sl_corr2"] = 0.0
    env["k0sl_corr3"] = 0.0
    env["k0sl_corr4"] = 0.0

    env.new("corr1", xt.Multipole, knl=["k0l_corr1"], ksl=["k0sl_corr1"])
    env.new("corr2", xt.Multipole, knl=["k0l_corr2"], ksl=["k0sl_corr2"])
    env.new("corr3", xt.Multipole, knl=["k0l_corr3"], ksl=["k0sl_corr3"])
    env.new("corr4", xt.Multipole, knl=["k0l_corr4"], ksl=["k0sl_corr4"])

    s0 = float(seq.s_starts[0])
    boundaries = [0.0]
    for s_end in seq.s_ends:
        boundaries.append(float(s_end) - s0)

    desired_positions = {
        "corr1": 0.02,
        "corr2": 0.1,
        "corr3": l_wig - 0.1,
        "corr4": l_wig - 0.02,
    }

    def nearest_boundary_idx(s_target):
        return min(range(len(boundaries)), key=lambda i: abs(boundaries[i] - s_target))

    insertions = {}
    for corr_name, s_target in desired_positions.items():
        idx = nearest_boundary_idx(s_target)
        insertions.setdefault(idx, []).append(corr_name)

    element_names_with_correctors = []
    for i, sb_name in enumerate(seq.element_names):
        if i in insertions:
            element_names_with_correctors.extend(insertions[i])
        element_names_with_correctors.append(sb_name)
    if len(seq.element_names) in insertions:
        element_names_with_correctors.extend(insertions[len(seq.element_names)])

    piecewise_undulator = xt.Line(env=env, element_names=element_names_with_correctors)
    piecewise_undulator.particle_ref = make_reference_particle()
    piecewise_undulator.build_tracker()

    opt = piecewise_undulator.match(
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
            xt.TargetSet(x=0, px=0, y=0, py=0, at=xt.END),
            xt.TargetSet(x=0, y=0, at="corr2"),
            xt.TargetSet(x=0, y=0, at="corr3"),
        ],
    )
    opt.step(2)
    piecewise_undulator.discard_tracker()

    if integrator == "splineboris":
        return piecewise_undulator

    seq_field = _make_seq_field_evaluator(seq)
    boris_element_names = []
    for i, (elem, s_start, s_end) in enumerate(zip(seq.elements, seq.s_starts, seq.s_ends)):
        for corr_name in insertions.get(i, []):
            boris_element_names.append(corr_name)
        boris_name = f"boris_spatial_seg_{i:03d}"
        env.elements[boris_name] = xt.BorisSpatialIntegrator(
            fieldmap_callable=seq_field,
            s_start=float(s_start),
            s_end=float(s_end),
            n_steps=int(elem.n_steps),
        )
        boris_element_names.append(boris_name)
    for corr_name in insertions.get(len(seq.elements), []):
        boris_element_names.append(corr_name)

    boris_undulator = xt.Line(env=env, element_names=boris_element_names)
    boris_undulator.particle_ref = make_reference_particle()
    return boris_undulator


def build_sls_with_undulators(integrator="splineboris"):
    env, line = make_sls_line()
    undulator = build_piecewise_undulator(env=env, integrator=integrator)
    tt = line.get_table()
    for place in WIGGLER_PLACES:
        line.insert(undulator, anchor="start", at=tt["s", place])
    line.build_tracker()
    return line


def benchmark_twiss(line, n_warmup=1, n_repeats=5):
    t0 = time.perf_counter()
    line.twiss4d(include_collective=True)
    first_call_s = time.perf_counter() - t0

    for _ in range(n_warmup):
        line.twiss4d(include_collective=True)

    run_times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        line.twiss4d(radiation_integrals=False, include_collective=True)
        run_times.append(time.perf_counter() - t0)

    return {
        "first_call_s": first_call_s,
        "median_s": median(run_times),
        "min_s": min(run_times),
        "max_s": max(run_times),
    }


def main():
    _, line_bare = make_sls_line()
    line_bare.build_tracker()

    line_with_und = build_sls_with_undulators(integrator="splineboris")
    line_with_und_boris = build_sls_with_undulators(integrator="boris_spatial")

    n_warmup = 1
    n_repeats = 5
    print(f"Timing twiss4d only (warmup={n_warmup}, repeats={n_repeats})")
    print("No line construction/fitting/insertion time included.")
    print()

    bare_stats = benchmark_twiss(line_bare, n_warmup=n_warmup, n_repeats=n_repeats)
    und_stats = benchmark_twiss(line_with_und, n_warmup=n_warmup, n_repeats=n_repeats)
    und_boris_stats = benchmark_twiss(
        line_with_und_boris, n_warmup=n_warmup, n_repeats=n_repeats
    )

    ratio_spline = und_stats["median_s"] / bare_stats["median_s"]
    ratio_boris = und_boris_stats["median_s"] / bare_stats["median_s"]
    print("Twiss timing results")
    print("-" * 72)
    print(
        f"{'Case':36s} {'first [s]':>12s} {'median [s]':>12s} {'min/max [s]':>18s}"
    )
    print("-" * 72)
    print(
        f"{'SLS bare ring':36s} "
        f"{bare_stats['first_call_s']:12.6f} "
        f"{bare_stats['median_s']:12.6f} "
        f"{bare_stats['min_s']:8.6f}/{bare_stats['max_s']:.6f}"
    )
    print(
        f"{'SLS with 11 undulators':36s} "
        f"{und_stats['first_call_s']:12.6f} "
        f"{und_stats['median_s']:12.6f} "
        f"{und_stats['min_s']:8.6f}/{und_stats['max_s']:.6f}"
    )
    print(
        f"{'SLS with 11 undulators (BorisSpatial)':36s} "
        f"{und_boris_stats['first_call_s']:12.6f} "
        f"{und_boris_stats['median_s']:12.6f} "
        f"{und_boris_stats['min_s']:8.6f}/{und_boris_stats['max_s']:.6f}"
    )
    print("-" * 72)
    print(f"Median runtime ratio (with spline undulators / bare): {ratio_spline:.3f}x")
    print(f"Median runtime ratio (with BorisSpatial undulators / bare): {ratio_boris:.3f}x")

    tw_bare = line_bare.twiss4d(include_collective=True)

    for place in WIGGLER_PLACES:
        row = tw_bare.rows[place]
        print(place, row.betx, row.bety, row.alfx, row.alfy, row.dx, row.dy)

if __name__ == "__main__":
    main()
