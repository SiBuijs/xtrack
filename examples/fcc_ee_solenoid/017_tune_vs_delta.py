"""Tune (qx, qy) vs momentum offset delta for the FCC-ee solenoid lattices.

Both ways of getting this are Twiss-only -- no tracking is needed to build a
tune-vs-delta chromaticity curve, since a *periodic* Twiss call already
returns the tune of the one-turn map about the closed orbit at whatever
delta0 it's given (`line.twiss(method='4d', delta0=...)`). Tracking a
particle away from that closed orbit and extracting a tune via FFT/NAFF
(e.g. `Line.get_amplitude_detuning_coefficients`) is a different question --
it probes amplitude-dependent detuning at fixed delta, not how the
closed-orbit tune itself shifts with delta, and is noisier/more expensive
for no benefit here.

Compares two ways of getting the linear/second-order chromaticity for each
of qx, qy:

1. **Scan + fit**: sweep delta0 through an off-momentum 4D twiss, record
   qx(delta)/qy(delta), fit a degree-`--fit-order` polynomial (default 2,
   quadratic). With `q(delta) = q0 + dq*delta + 0.5*ddq*delta**2 +
   (1/6)*dddq*delta**3 + ...` (a Taylor expansion), the fit's order-n
   coefficient recovers the order-n chromaticity via `coeff_n * n!`. Raising
   `--fit-order` to 3+ adds higher-order chromaticity terms (dddqx/dddqy,
   ...) for lattices whose nonlinear chromaticity a quadratic doesn't
   capture well (see below) -- there's no direct-Twiss counterpart for
   those (method 2 below only ever produces order 1/2), so they're fit-only.
2. **Direct from Twiss**: a single on-momentum `line.twiss(method='4d')`
   call already carries `tw.dqx`/`tw.dqy`/`tw.ddqx`/`tw.ddqy` -- xtrack
   computes these internally via its own small finite-difference delta
   probe (`xtrack.twiss._get_chromatic_functions`, `delta_chrom` default
   5e-5) whenever the twiss is periodic, which every ordinary closed-ring
   twiss() call in this pipeline already is (`chrom is None and periodic`
   in `twiss_line`).

The two should agree closely (at the default fit order and delta range);
this script plots both together (scanned points + fit curve, one subplot
per plane) with a text box comparing the fit and Twiss-direct chromaticity
values.

Empirically (checked by hand against all three cases before picking
DELTA_MAX below), `line.twiss(method='4d', delta0=...)` on this coupled
(solenoid-on) optics occasionally fails to converge -- either a
`ClosedOrbitSearchError` or a `ValueError` from the linear-normal-form
eigenvector ordering -- at scattered delta0 values, including some sitting
inside an otherwise fine range (not just "large delta0 is unstable"; it's
closer to isolated near-degenerate points). The scan below treats that as
expected rather than a bug: it catches both, skips the offending point
(NaN, reported), and fits through whatever survives.
"""

from __future__ import annotations

from math import factorial
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from xtrack.twiss import ClosedOrbitSearchError

from aperture_study_io import save_tune_study, variant_suffix
from lattice_knobs import robust_twiss, set_lattice_knobs
from solenoid_params import (
    MAIN_SOLENOID_B0,
    add_b0_argument,
    add_max_order_argument,
    field_tag,
    order_tag,
)

plt.close("all")

HERE = Path(__file__).resolve().parent

# Empirically tuned (sb_on, checked by hand): this lattice's chromaticity is
# strongly nonlinear -- ddqy ~ 5.7e4 -- so a quadratic fit only tracks the
# Twiss-direct dqx/dqy/ddqx/ddqy closely if the scan window stays fairly
# tight; a wider window (e.g. the full momentum-acceptance-scale +-1e-2 used
# elsewhere in this directory) picks up quartic-and-higher contamination and
# the fit visibly diverges from the local (delta_chrom-based) Twiss values
# -- a real feature of this optics, not a fitting bug (see module
# docstring). At +-1e-3 the two methods agree to ~1-2%; narrower still
# (+-3e-4) agrees to <1%. +-1e-2 is still a valid --delta-max choice if you
# want to see that higher-order departure on purpose.
DELTA_MAX = 1.0e-3
N_DELTA = 21

# Quadratic (dq, ddq) by default -- matches what a single Twiss call can
# directly cross-check (method 2 above). Raise via --fit-order to add
# higher-order (fit-only) chromaticity terms.
FIT_ORDER = 2


def _taylor_coeffs_from_polyfit(poly: np.ndarray) -> np.ndarray:
    """Convert np.polyfit's highest-degree-first coefficients into Taylor
    coefficients [q0, dq, ddq, dddq, ...] such that
    q(delta) = sum_n coeffs[n] * delta**n / n! -- same convention as
    xtrack.twiss.get_non_linear_chromaticity's dnqx/dnqy."""
    low_to_high = poly[::-1]
    return np.array([low_to_high[n] * factorial(n) for n in range(len(low_to_high))])


def _lattice_paths(tag, order_tag_str=""):
    # order_tag_str only applies to the SplineBoris path -- VariableSolenoid
    # is linear-only and has no transverse-order knob (see 00_overview.md).
    return (
        HERE / f"fccee_z_lcc_splineboris_solenoids_coupling_corrected_{tag}{order_tag_str}.json",
        HERE / f"fccee_z_lcc_varsol_solenoids_coupling_corrected_{tag}.json",
    )


def _build_tune_cases(tag, order_tag_str=""):
    """Build the TUNE_CASES list for a given field_tag (e.g. '2T', '3T') and
    order_tag (e.g. '', '_o2' -- SplineBoris-only, see _lattice_paths)."""
    splineboris_json, varsol_json = _lattice_paths(tag, order_tag_str)
    sb_tag = f"{tag}{order_tag_str}"
    return [
        dict(
            name="sb_on",
            model="SB",
            lattice_json=splineboris_json,
            with_solenoids=True,
            with_correctors=True,
            title=f"SplineBoris ({sb_tag}): solenoids powered + correction scheme",
        ),
        dict(
            name="varsol_on",
            model="VarSol",
            lattice_json=varsol_json,
            with_solenoids=True,
            with_correctors=True,
            title=f"VariableSolenoid ({tag}): solenoids powered + correction scheme",
        ),
        dict(
            name="sb_off",
            model="SB",
            lattice_json=splineboris_json,
            with_solenoids=False,
            with_correctors=False,
            title=f"SplineBoris ({sb_tag}): solenoids unpowered",
        ),
    ]


# Unlike 015 (POL), this scan is pure Twiss -- no tracking -- so sb_off costs
# no more than the other two and is included by default for the bare-machine
# comparison.
DEFAULT_TUNE_CASE_NAMES = ["sb_on", "varsol_on", "sb_off"]


def _order_symbol(n: int, sym: str) -> str:
    """LaTeX label for the order-n delta-derivative of q_sym: dq_x, ddq_x,
    then d^3q_x, d^4q_x, ... beyond that."""
    if n == 1:
        return fr"dq_{sym}"
    if n == 2:
        return fr"ddq_{sym}"
    return fr"d^{n}q_{sym}"


def _plot_tune_vs_delta_figure(
    *,
    delta_values,
    qx,
    qy,
    qx0,
    qy0,
    poly_qx,
    poly_qy,
    fit_order,
    coeffs_x_fit,
    coeffs_y_fit,
    dqx_twiss,
    ddqx_twiss,
    dqy_twiss,
    ddqy_twiss,
    title,
):
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(6.4, 7.6), sharex=True)
    delta_fine = np.linspace(delta_values[0], delta_values[-1], 400)

    # Plot relative to the on-momentum tune (qx0/qy0, from the Method-2
    # Twiss-direct call at delta0=0) -- this only shifts the curve vertically
    # by a constant, so it doesn't change dq/ddq/... at all, just strips the
    # large integer-tune offset (e.g. ~194) that otherwise dominates the axis.
    panels = (
        (ax_x, qx, qx0, poly_qx, coeffs_x_fit, dqx_twiss, ddqx_twiss, "x"),
        (ax_y, qy, qy0, poly_qy, coeffs_y_fit, dqy_twiss, ddqy_twiss, "y"),
    )
    for ax, q, q0, poly, coeffs_fit, dq_tw, ddq_tw, sym in panels:
        ax.plot(delta_values, q - q0, "o", ms=4, color="C0", label="Twiss scan")
        ax.plot(
            delta_fine, np.polyval(poly, delta_fine) - q0, "--", color="C3",
            label=f"order-{fit_order} fit",
        )
        ax.set_ylabel(fr"$q_{sym} - q_{sym}(\delta{{=}}0)$")
        ax.legend(loc="upper left", fontsize=8)

        fit_lines = [f"fit (order-{fit_order} scan):"]
        for n in range(1, fit_order + 1):
            fit_lines.append(fr"  ${_order_symbol(n, sym)} = {coeffs_fit[n]:.5g}$")
        twiss_lines = [
            "Twiss (direct):",
            fr"  ${_order_symbol(1, sym)} = {dq_tw:.5g}$",
            fr"  ${_order_symbol(2, sym)} = {ddq_tw:.5g}$",
        ]
        info_text = "\n".join(fit_lines + twiss_lines)
        ax.text(
            0.98, 0.03, info_text,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=7.5, linespacing=1.4,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )

    ax_y.set_xlabel(r"$\delta$")
    ax_y.tick_params(axis="x", labelrotation=30)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _run_tune_vs_delta(case, *, delta_max, n_delta, fit_order, sexamp, tag):
    lattice_json = case["lattice_json"]
    title = case["title"]

    if not lattice_json.exists():
        raise SystemExit(
            f"Missing lattice file: {lattice_json.name}\n"
            f"Build it first for tag {tag!r} via "
            "004a_build_and_check_solenoids.py -> "
            "004b_install_solenoids_in_fcc_ring.py / "
            "004b_install_varsol_solenoids_in_fcc_ring.py -> "
            "004c_correct_solenoids_in_fcc_ring.py (each accepts --b0 and, "
            "for the SplineBoris path, --max-transverse-order)."
        )

    print(f"\n=== {title} ===")
    print(f"Loading lattice: {lattice_json.name}")
    env = xt.load(lattice_json)
    line = env.fccee_p_ring
    line.cycle("ipa")
    set_lattice_knobs(
        line,
        with_solenoids=case["with_solenoids"],
        with_correctors=case["with_correctors"],
        sext_amp=sexamp,
    )

    line.discard_tracker()
    line.build_tracker()

    # Method 2: chromaticity read directly off a single on-momentum twiss.
    tw0 = robust_twiss(line, twiss_method="twiss", method="4d")
    qx0 = float(tw0.qx)
    qy0 = float(tw0.qy)
    dqx_twiss = float(tw0.dqx)
    dqy_twiss = float(tw0.dqy)
    ddqx_twiss = float(tw0.ddqx)
    ddqy_twiss = float(tw0.ddqy)
    print("Twiss (direct, single on-momentum twiss):")
    print(f"  dqx  = {dqx_twiss:.6g}   ddqx = {ddqx_twiss:.6g}")
    print(f"  dqy  = {dqy_twiss:.6g}   ddqy = {ddqy_twiss:.6g}")

    # Method 1: scan delta0 through an off-momentum 4D twiss and fit. A
    # handful of isolated delta0 points can fail to converge on this coupled
    # optics (see module docstring) -- skip those (NaN) rather than aborting
    # the whole scan. chrom=False since only qx/qy are read here -- the
    # per-point chromatic-derivative probe (2-3 extra sub-twiss evaluations
    # each) that a periodic twiss computes by default is wasted work in this
    # loop; it's only needed once, for the Method-2 call above.
    delta_values = np.linspace(-delta_max, delta_max, n_delta)
    qx = np.full(n_delta, np.nan)
    qy = np.full(n_delta, np.nan)
    n_failed = 0
    for i, dd in enumerate(delta_values):
        try:
            tw = robust_twiss(
                line, twiss_method="twiss", method="4d", delta0=float(dd),
                chrom=False,
            )
        except (ClosedOrbitSearchError, ValueError) as e:
            n_failed += 1
            print(f"  delta0={dd:+.5g}: twiss failed ({type(e).__name__}), skipping point")
            continue
        qx[i] = tw.qx
        qy[i] = tw.qy
    print(
        f"Scanned {n_delta} delta values in [{-delta_max:.4g}, {delta_max:.4g}] "
        f"({n_failed} failed/skipped)"
    )

    valid = ~np.isnan(qx) & ~np.isnan(qy)
    min_points = fit_order + 2  # at least one degree of freedom beyond an exact fit
    if valid.sum() < min_points:
        raise SystemExit(
            f"Only {valid.sum()} delta0 point(s) converged -- need at least "
            f"{min_points} for a degree-{fit_order} fit. Try a smaller "
            "--delta-max, more --n-delta, or a lower --fit-order."
        )
    poly_qx = np.polyfit(delta_values[valid], qx[valid], deg=fit_order)
    poly_qy = np.polyfit(delta_values[valid], qy[valid], deg=fit_order)
    # coeffs_*_fit[n] is the order-n Taylor coefficient: coeffs[1]=dq,
    # coeffs[2]=ddq, coeffs[3]=dddq, ... (q(delta) = sum_n coeffs[n]*delta**n/n!)
    coeffs_x_fit = _taylor_coeffs_from_polyfit(poly_qx)
    coeffs_y_fit = _taylor_coeffs_from_polyfit(poly_qy)
    print(f"Fit (order-{fit_order} scan):")
    print(f"  dqx  = {coeffs_x_fit[1]:.6g}   ddqx = {coeffs_x_fit[2]:.6g}   "
          f"[Twiss: dqx={dqx_twiss:.6g}, ddqx={ddqx_twiss:.6g}]")
    print(f"  dqy  = {coeffs_y_fit[1]:.6g}   ddqy = {coeffs_y_fit[2]:.6g}   "
          f"[Twiss: dqy={dqy_twiss:.6g}, ddqy={ddqy_twiss:.6g}]")
    for n in range(3, fit_order + 1):
        print(f"  d{n}qx = {coeffs_x_fit[n]:.6g}   d{n}qy = {coeffs_y_fit[n]:.6g}   "
              "[fit only, no direct-Twiss reference]")

    fig = _plot_tune_vs_delta_figure(
        delta_values=delta_values,
        qx=qx,
        qy=qy,
        qx0=qx0,
        qy0=qy0,
        poly_qx=poly_qx,
        poly_qy=poly_qy,
        fit_order=fit_order,
        coeffs_x_fit=coeffs_x_fit,
        coeffs_y_fit=coeffs_y_fit,
        dqx_twiss=dqx_twiss,
        ddqx_twiss=ddqx_twiss,
        dqy_twiss=dqy_twiss,
        ddqy_twiss=ddqy_twiss,
        title=title,
    )
    save_tune_study(
        fig=fig,
        model=case["model"],
        with_solenoids=case["with_solenoids"],
        with_correctors=case["with_correctors"],
        n_delta=n_delta,
        delta_max=delta_max,
        delta_values=delta_values,
        qx=qx,
        qy=qy,
        fit_order=fit_order,
        coeffs_x_fit=coeffs_x_fit,
        coeffs_y_fit=coeffs_y_fit,
        dqx_twiss=dqx_twiss,
        ddqx_twiss=ddqx_twiss,
        dqy_twiss=dqy_twiss,
        ddqy_twiss=ddqy_twiss,
        qx0_twiss=qx0,
        qy0_twiss=qy0,
        variant=variant_suffix(sexamp=sexamp),
        sexamp=sexamp,
        field_tag=tag,
    )
    print(f"[{title}] Tune-vs-delta scan complete.")


def _select_cases(case_names, cases_by_name):
    if not case_names:
        return [cases_by_name[name] for name in DEFAULT_TUNE_CASE_NAMES]

    unknown = [name for name in case_names if name not in cases_by_name]
    if unknown:
        valid = ", ".join(cases_by_name)
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Choose from: {valid}")

    return [cases_by_name[name] for name in case_names]


def main():
    parser = argparse.ArgumentParser(
        description="Tune-vs-delta chromaticity scan for FCC-ee solenoid lattices."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE",
        help=(
            "Cases to run (default: sb_on, varsol_on, sb_off). "
            "Available: sb_on, varsol_on, sb_off"
        ),
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available cases and exit.",
    )
    add_b0_argument(parser, default=MAIN_SOLENOID_B0)
    add_max_order_argument(parser)
    parser.add_argument(
        "--delta-max",
        type=float,
        default=DELTA_MAX,
        metavar="DELTA",
        help=f"Scan delta0 over [-DELTA, +DELTA] (default: {DELTA_MAX:g}).",
    )
    parser.add_argument(
        "--n-delta",
        type=int,
        default=N_DELTA,
        metavar="N",
        help=f"Number of delta0 points in the scan (default: {N_DELTA}).",
    )
    parser.add_argument(
        "--fit-order",
        type=int,
        default=FIT_ORDER,
        metavar="N",
        help=(
            f"Polynomial fit order for the delta-scan (default: {FIT_ORDER}, "
            "i.e. dq/ddq). N>=3 adds higher-order chromaticity terms "
            "(dddqx/dddqy, ...) with no direct-Twiss reference -- useful for "
            "lattices whose nonlinear chromaticity a quadratic doesn't "
            "capture well over the chosen --delta-max (see module "
            "docstring). Must be >= 2."
        ),
    )
    parser.add_argument(
        "--sexamp",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Sextupole amplification knob (default: 1.0).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive figure display (data and PDFs are still saved).",
    )
    args = parser.parse_args()

    if args.fit_order < 2:
        raise SystemExit(
            f"--fit-order must be >= 2 (got {args.fit_order}) -- ddq relies "
            "on the quadratic term."
        )

    tag = field_tag(args.b0)
    order_tag_str = order_tag(args.max_transverse_order)
    tune_cases = _build_tune_cases(tag, order_tag_str)
    tune_cases_by_name = {case["name"]: case for case in tune_cases}

    if args.list_cases:
        print(
            f"Field tag: {tag}{order_tag_str} (--b0 {args.b0:g}, "
            f"--max-transverse-order {args.max_transverse_order})")
        for case in tune_cases:
            print(f"{case['name']}: {case['title']}")
        return

    for case in _select_cases(args.cases, tune_cases_by_name):
        _run_tune_vs_delta(
            case,
            delta_max=args.delta_max,
            n_delta=args.n_delta,
            fit_order=args.fit_order,
            sexamp=args.sexamp,
            tag=f"{tag}{order_tag_str}",
        )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
