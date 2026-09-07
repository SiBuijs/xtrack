"""Off-momentum tune footprint of the FCC-ee solenoid rings on a resonance diagram.

Scan the momentum offset ``delta0`` over ``[-DELTA_MAX, +DELTA_MAX]`` through the
2 T and 3 T solenoid-corrected FCC-ee Z rings (periodic 4D twiss -- no tracking),
record ``(qx, qy)`` at each delta, and overlay the resulting tune footprints on a
``(Qx, Qy)`` plane ruled with resonance lines ``a*Qx + b*Qy = c`` for every
integer ``(a, b, c)`` with ``1 <= |a| + |b| <= --max-resonance-order`` (default 3,
sextupole -- the chromatic sextupoles are the highest globally-distributed
multipole in the ring; there are also localised IR octupole/decapole correctors
(``oct1*`` / ``dec1*``), so ``--max-resonance-order 4`` or ``5`` will show those
lines too). The point of the plot is to see which resonance lines the tune
crosses, and at what delta, as the strongly nonlinear vertical chromaticity of
the solenoid-on optics (``ddqy ~ 5.7e4``, see 017_tune_vs_delta.py) sweeps the
tune around.

For each ``--b0`` field case one "corrected" footprint is drawn (solenoids +
correctors ON). One "bare" footprint (solenoids OFF -- the design ring) is drawn
once regardless of ``--b0``, since with the solenoids off the field strength is
irrelevant. ``--no-bare`` drops it.

The tune plane is plotted in **absolute** tune (not fractional / mod 1); the
window is auto-fit to the bounding box of the converged footprint points and can
therefore span several integer tunes. Use ``--qx-range`` / ``--qy-range`` to pin
it by hand, and ``--max-resonance-order 2`` to thin the line set when the
vertical sweep is large.

Isolated ``delta0`` points that fail to converge (``ClosedOrbitSearchError`` /
``ValueError`` from the linear-normal-form eigenvector ordering) are expected on
this coupled optics -- they are recorded as NaN, reported, and skipped, same as
017_tune_vs_delta.py.

The (expensive) scan result is pickled to ``aperture_study_io.DATA_DIR``; rerun
with ``--replot`` (same ``--b0`` / ``--max-transverse-order`` / ``--no-bare`` /
``--n-delta`` / ``--delta-max``) to regenerate the figure without re-twissing.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from xtrack.twiss import ClosedOrbitSearchError

from aperture_study_io import DATA_DIR, PLOT_DIR
from lattice_knobs import robust_twiss, set_lattice_knobs
from solenoid_params import add_max_order_argument, field_tag, order_tag

HERE = Path(__file__).resolve().parent

DELTA_MAX = 0.015
N_DELTA = 61
# Sextupole is the highest globally-distributed multipole (chromatic sextupoles);
# octupoles/decapoles exist only as localised IR correctors, so 3 is the sensible
# default. Bump via --max-resonance-order to also draw those weaker lines.
MAX_RESONANCE_ORDER = 3
DELTA_CMAP = "coolwarm"

# Extra reference delta highlighted on the corrected footprints (in addition to
# the delta = 0 star), drawn as a diamond. Set to None to disable.
MARK_DELTA = 0.0095

# resonance order -> (colour, linewidth, base linestyle, alpha). Orders above
# the table fall back to a thin grey dotted line.
ORDER_STYLE = {
    1: ("#d62728", 1.8, "-", 0.95),
    2: ("#1f77b4", 1.2, "-", 0.90),
    3: ("#2ca02c", 0.9, "-", 0.85),
    4: ("#9467bd", 0.6, ":", 0.55),
    5: ("#8c564b", 0.5, ":", 0.45),
}

# field_tag -> "corrected" scatter marker; the single "bare" footprint is always
# a square. Unlisted field tags fall back in _case_marker().
CORRECTED_MARKERS = {"2T": "o", "3T": "^", "4T": "D", "1T": "v"}


# --------------------------------------------------------------------------- #
# CLI (module level, 004j idiom -- keeps --replot reading cleanly)
# --------------------------------------------------------------------------- #
_parser = argparse.ArgumentParser(
    description="Off-momentum tune footprint on a resonance diagram "
    "(FCC-ee solenoid rings).",
)
_parser.add_argument(
    "--b0",
    type=float,
    nargs="+",
    default=[2.0, 3.0],
    metavar="TESLA",
    help="Main-solenoid field-strength case(s) to scan (default: 2.0 3.0). "
    "Each selects fccee_z_lcc_splineboris_solenoids_coupling_corrected_"
    "{field_tag}{order_tag}.json.",
)
_parser.add_argument(
    "--n-delta",
    type=int,
    default=N_DELTA,
    metavar="N",
    help=f"Number of delta0 points in linspace(-delta_max, delta_max, N) "
    f"(default: {N_DELTA}).",
)
_parser.add_argument(
    "--delta-max",
    type=float,
    default=DELTA_MAX,
    metavar="DELTA",
    help=f"Scan delta0 over [-DELTA, +DELTA] (default: {DELTA_MAX:g}).",
)
_parser.add_argument(
    "--max-resonance-order",
    type=int,
    default=MAX_RESONANCE_ORDER,
    metavar="N",
    help=f"Draw resonance lines a*Qx + b*Qy = c with 1 <= |a|+|b| <= N "
    f"(default: {MAX_RESONANCE_ORDER}; 3 = sextupole, 4 = octupole, "
    "5 = decapole -- the latter two only from localised IR correctors).",
)
add_max_order_argument(_parser)
_parser.add_argument(
    "--no-bare",
    action="store_true",
    help="Skip the solenoids-OFF baseline (design-ring) footprints.",
)
_parser.add_argument(
    "--qx-range",
    type=float,
    nargs=2,
    default=None,
    metavar=("QX_LO", "QX_HI"),
    help="Manual horizontal window override (default: auto-fit to the footprint).",
)
_parser.add_argument(
    "--qy-range",
    type=float,
    nargs=2,
    default=None,
    metavar=("QY_LO", "QY_HI"),
    help="Manual vertical window override (default: auto-fit to the footprint).",
)
_parser.add_argument(
    "--pad",
    type=float,
    default=0.05,
    metavar="TUNE",
    help="Auto-window margin in tune units (default: 0.05).",
)
_parser.add_argument(
    "--replot",
    action="store_true",
    help="Reload the pickled scan for this exact knob set and only redraw "
    "(skips every twiss).",
)
_parser.add_argument(
    "--no-show",
    action="store_true",
    help="Save the PDF without opening an interactive window.",
)
_args = _parser.parse_args()

ORDER_TAG = order_tag(_args.max_transverse_order)
B0_VALUES = list(_args.b0)


# --------------------------------------------------------------------------- #
# Resonance-line geometry
# --------------------------------------------------------------------------- #
def resonance_lines(max_order, qx_lim, qy_lim):
    """Yield (a, b, c, n) for every integer line ``a*Qx + b*Qy = c`` with
    ``n = |a| + |b|`` in ``1..max_order`` that can intersect the rectangle
    ``qx_lim x qy_lim``.

    Each geometric locus is emitted once: (a, b) and (-a, -b) describe the
    same family, so representatives are restricted to the canonical half-plane
    ``a > 0`` or ``(a == 0 and b > 0)``. For a fixed (a, b), ``a*qx + b*qy`` is
    linear, so over the rectangle it ranges between its values at the four
    corners; every integer c in that span gives a line crossing the box.
    """
    qx_lo, qx_hi = sorted(qx_lim)
    qy_lo, qy_hi = sorted(qy_lim)
    corners = (
        (qx_lo, qy_lo),
        (qx_hi, qy_lo),
        (qx_lo, qy_hi),
        (qx_hi, qy_hi),
    )
    for n in range(1, max_order + 1):
        for a in range(0, n + 1):
            b_mag = n - a
            b_iter = (b_mag,) if b_mag == 0 else (b_mag, -b_mag)
            for b in b_iter:
                if a == 0 and b <= 0:
                    continue
                vals = [a * cx + b * cy for cx, cy in corners]
                c_lo = int(np.ceil(min(vals) - 1e-9))
                c_hi = int(np.floor(max(vals) + 1e-9))
                for c in range(c_lo, c_hi + 1):
                    yield a, b, c, n


def _line_box_segment(a, b, c, qx_lim, qy_lim, eps=1e-9):
    """Clip the line ``a*Qx + b*Qy = c`` to the rectangle.

    Returns ``((x0, y0), (x1, y1))`` or ``None`` when the line misses the box
    (or only grazes a single corner). Vertical lines (b == 0) and horizontal
    lines (a == 0) fall out of the branch guards with no special-casing.
    """
    qx_lo, qx_hi = sorted(qx_lim)
    qy_lo, qy_hi = sorted(qy_lim)
    pts = []
    if b != 0:  # crossings of the two vertical edges
        for qx in (qx_lo, qx_hi):
            qy = (c - a * qx) / b
            if qy_lo - eps <= qy <= qy_hi + eps:
                pts.append((qx, min(max(qy, qy_lo), qy_hi)))
    if a != 0:  # crossings of the two horizontal edges
        for qy in (qy_lo, qy_hi):
            qx = (c - b * qy) / a
            if qx_lo - eps <= qx <= qx_hi + eps:
                pts.append((min(max(qx, qx_lo), qx_hi), qy))
    uniq = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-7 and abs(p[1] - q[1]) < 1e-7 for q in uniq):
            uniq.append(p)
    return (uniq[0], uniq[1]) if len(uniq) >= 2 else None


def draw_resonance_lines(ax, max_order, qx_lim, qy_lim):
    """Draw every resonance line up to ``max_order`` clipped to the window.

    Sum resonances (``a*b > 0``) are solid, difference resonances
    (``a*b < 0``) dashed, for orders up to 3; order 4+ keeps its (dotted)
    table linestyle. Returns proxy Line2D handles for an order legend. The
    caller must pin ``set_xlim`` / ``set_ylim`` afterwards -- these segment
    plots are inert under autoscale but nothing else fixes the frame.
    """
    orders_seen = set()
    for a, b, c, n in resonance_lines(max_order, qx_lim, qy_lim):
        seg = _line_box_segment(a, b, c, qx_lim, qy_lim)
        if seg is None:
            continue
        colour, lw, base_ls, alpha = ORDER_STYLE.get(n, ("0.5", 0.5, ":", 0.4))
        ls = "--" if (a * b < 0 and n <= 3) else base_ls
        (x0, y0), (x1, y1) = seg
        ax.plot(
            [x0, x1], [y0, y1],
            color=colour, lw=lw, ls=ls, alpha=alpha, zorder=1,
        )
        orders_seen.add(n)
    handles = []
    for n in sorted(orders_seen):
        colour, lw, base_ls, _ = ORDER_STYLE.get(n, ("0.5", 0.5, ":", 0.4))
        handles.append(
            plt.Line2D([], [], color=colour, lw=lw, ls=base_ls, label=f"order {n}")
        )
    return handles


# --------------------------------------------------------------------------- #
# The delta scan (structurally 017_tune_vs_delta.py::_run_tune_vs_delta)
# --------------------------------------------------------------------------- #
def _scan_case(b0, config, *, n_delta, delta_max, order_tag_str):
    """Off-momentum 4D-twiss tune scan for one (field strength, config) case.

    ``config`` is "corrected" (solenoids + correctors on) or "bare"
    (solenoids off -- the design ring). Returns a dict with the delta grid and
    the qx/qy arrays (NaN where the twiss did not converge).
    """
    tag = field_tag(b0)
    lattice_json = HERE / (
        "fccee_z_lcc_splineboris_solenoids_coupling_corrected_"
        f"{tag}{order_tag_str}.json"
    )
    if not lattice_json.exists():
        raise SystemExit(
            f"Missing lattice: {lattice_json.name}\n"
            f"Build it via 004b_install_solenoids_in_fcc_ring.py --b0 {b0:g} "
            "[--max-transverse-order N] then "
            f"004c_correct_solenoids_in_fcc_ring.py --b0 {b0:g} [--max-transverse-order N]."
        )

    print(f"\n=== {tag}{order_tag_str} ring, {config} ===")
    print(f"Loading lattice: {lattice_json.name}")
    env = xt.load(lattice_json)
    line = env.fccee_p_ring
    line.cycle("ipa")

    solenoids_on = config == "corrected"
    set_lattice_knobs(
        line,
        with_solenoids=solenoids_on,
        with_correctors=solenoids_on,
        sext_amp=1.0,
    )
    line.discard_tracker()
    line.build_tracker()

    delta_values = np.linspace(-delta_max, delta_max, n_delta)
    qx = np.full(n_delta, np.nan)
    qy = np.full(n_delta, np.nan)
    n_failed = 0
    for i, dd in enumerate(delta_values):
        try:
            # chrom=False: only qx/qy are read here, so the per-point
            # chromatic-derivative sub-probe a periodic twiss runs by default
            # is wasted work in this loop.
            tw = robust_twiss(
                line,
                twiss_method="twiss",
                method="4d",
                delta0=float(dd),
                chrom=False,
            )
        except (ClosedOrbitSearchError, ValueError) as exc:
            n_failed += 1
            print(f"  delta0={dd:+.5g}: twiss failed ({type(exc).__name__}), skipping")
            continue
        qx[i] = tw.qx
        qy[i] = tw.qy
        print(f"  delta0={dd:+.5g}: qx={tw.qx:.5f}  qy={tw.qy:.5f}")

    print(
        f"Scanned {n_delta} delta values in [{-delta_max:.4g}, {delta_max:.4g}] "
        f"({n_failed} failed/skipped)"
    )
    label = f"{tag} corrected" if solenoids_on else "bare (design ring)"
    return dict(
        b0=float(b0),
        field_tag=tag,
        config=config,
        label=label,
        delta_values=delta_values,
        qx=qx,
        qy=qy,
        n_failed=n_failed,
    )


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _case_marker(field_tag_str, config):
    if config == "bare":
        return "s"
    return CORRECTED_MARKERS.get(field_tag_str, "D")


def _nearest_delta_index(d, finite, target):
    """Index of the finite scan point closest to ``target`` delta, or None if
    ``target`` is outside the scanned range or the nearest point is further
    than one grid step away / not converged."""
    if target is None or target > d.max() + 1e-12 or target < d.min() - 1e-12:
        return None
    step = float(d[1] - d[0]) if len(d) > 1 else np.inf
    i = int(np.argmin(np.abs(d - target)))
    if not finite[i] or abs(d[i] - target) > 0.5 * abs(step) + 1e-12:
        return None
    return i


def _compute_window(cases, *, pad, qx_range, qy_range):
    """Auto-fit the (Qx, Qy) window to the converged footprint, or use the
    manual overrides. May legitimately span several integer tunes."""
    qx_all = np.concatenate([c["qx"] for c in cases])
    qy_all = np.concatenate([c["qy"] for c in cases])
    finite = np.isfinite(qx_all) & np.isfinite(qy_all)
    if finite.sum() == 0:
        raise SystemExit(
            "No converged (qx, qy) points in any case -- try a smaller "
            "--delta-max or more --n-delta."
        )
    qx_lo, qx_hi = float(qx_all[finite].min()), float(qx_all[finite].max())
    qy_lo, qy_hi = float(qy_all[finite].min()), float(qy_all[finite].max())

    # qx barely moves on this optics (dqx ~ 0.1) -- floor its margin so the
    # frame is not a degenerate sliver.
    px = max(pad, 0.10 * (qx_hi - qx_lo), 0.05)
    py = max(pad, 0.05 * (qy_hi - qy_lo))
    qx_lim = (qx_lo - px, qx_hi + px)
    qy_lim = (qy_lo - py, qy_hi + py)

    if qx_range is not None:
        qx_lim = (min(qx_range), max(qx_range))
    if qy_range is not None:
        qy_lim = (min(qy_range), max(qy_range))

    if (qy_lim[1] - qy_lim[0]) > 3.0:
        print(
            f"  NOTE: qy window spans {qy_lim[1] - qy_lim[0]:.1f} tune units; "
            "higher-order resonance lines will be dense. Consider "
            "--max-resonance-order 2 or --qy-range to zoom."
        )
    return qx_lim, qy_lim


def _plot_tune_diagram(cases, *, max_resonance_order, pad, qx_range, qy_range):
    qx_lim, qy_lim = _compute_window(
        cases, pad=pad, qx_range=qx_range, qy_range=qy_range
    )

    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    # aspect left at "auto": the qx span (~0.01-0.05) is ~100x smaller than
    # the qy span, so "equal" would collapse the plot to a vertical line.

    res_handles = draw_resonance_lines(ax, max_resonance_order, qx_lim, qy_lim)

    delta_max = float(np.max(np.abs(cases[0]["delta_values"])))
    norm = mcolors.Normalize(vmin=-delta_max, vmax=delta_max)
    cmap = plt.get_cmap(DELTA_CMAP)

    marker_handles = []
    for case in cases:
        d = case["delta_values"]
        qx = case["qx"]
        qy = case["qy"]
        finite = np.isfinite(qx) & np.isfinite(qy)
        label = case.get("label") or f"{case['field_tag']} {case['config']}"
        if finite.sum() < 1:
            print(f"  WARNING: {label} has no converged points -- footprint not drawn.")
            continue
        mk = _case_marker(case["field_tag"], case["config"])
        ax.scatter(
            qx[finite], qy[finite],
            c=d[finite], cmap=cmap, norm=norm,
            marker=mk, s=28,
            edgecolors="k" if case["config"] == "corrected" else "none",
            linewidths=0.3,
            zorder=3,
        )
        # flag delta = 0 (star) and delta = MARK_DELTA (diamond) on every
        # footprint (corrected and bare)
        i0 = _nearest_delta_index(d, finite, 0.0)
        if i0 is not None:
            ax.scatter(
                [qx[i0]], [qy[i0]],
                marker="*", s=240,
                facecolor="yellow", edgecolor="k", linewidths=0.6,
                zorder=5,
            )
        i1 = _nearest_delta_index(d, finite, MARK_DELTA)
        if i1 is not None:
            ax.scatter(
                [qx[i1]], [qy[i1]],
                marker="D", s=120,
                facecolor="cyan", edgecolor="k", linewidths=0.6,
                zorder=4,
            )
        marker_handles.append(
            plt.Line2D([], [], marker=mk, ls="none", color="0.3", label=label)
        )
    marker_handles.append(
        plt.Line2D(
            [], [], marker="*", ls="none",
            color="yellow", markeredgecolor="k", label=r"$\delta = 0$",
        )
    )
    if MARK_DELTA is not None:
        marker_handles.append(
            plt.Line2D(
                [], [], marker="D", ls="none",
                color="cyan", markeredgecolor="k",
                label=rf"$\delta = {MARK_DELTA:g}$",
            )
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"momentum offset $\delta$", fraction=0.046, pad=0.03)

    ax.set_xlabel(r"$Q_x$")
    ax.set_ylabel(r"$Q_y$")
    ax.set_xlim(*qx_lim)
    ax.set_ylim(*qy_lim)
    ax.set_title(
        r"FCC-ee solenoid rings: tune footprint vs $\delta \in "
        rf"[{-delta_max:+.3g},\ {delta_max:+.3g}]$"
        "\n"
        f"resonances to order {max_resonance_order}",
        fontsize=10,
    )
    leg1 = ax.legend(
        handles=res_handles, title="resonance order",
        loc="upper left", fontsize=8, framealpha=0.9,
    )
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Persistence / entry point
# --------------------------------------------------------------------------- #
def _tags():
    b0_tag = "".join(field_tag(b0) for b0 in B0_VALUES)
    bare_tag = "" if not _args.no_bare else "_nobare"
    scan_tag = (
        f"n{_args.n_delta}_dmax{('%g' % _args.delta_max).replace('.', 'p')}"
    )
    return b0_tag, ORDER_TAG, bare_tag, scan_tag


def _data_path():
    b0_tag, order_tag_str, bare_tag, scan_tag = _tags()
    return DATA_DIR / (
        f"tune_diagram_vs_delta_{b0_tag}{order_tag_str}{bare_tag}_{scan_tag}.pkl"
    )


def _save_fig(fig, stem):
    out_dir = PLOT_DIR / "Coupling_Studies" / "tune_diagram_vs_delta"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved plot: {path}")
    return path


def main():
    data_path = _data_path()

    if _args.replot:
        if not data_path.exists():
            raise SystemExit(
                f"--replot: no saved scan at {data_path}\n"
                "Run once without --replot (same --b0 / --max-transverse-order "
                "/ --no-bare / --n-delta / --delta-max) to create it."
            )
        print(f"--replot: loading saved scan data from {data_path}")
        with open(data_path, "rb") as f:
            cases = pickle.load(f)
    else:
        scan_kw = dict(
            n_delta=_args.n_delta,
            delta_max=_args.delta_max,
            order_tag_str=ORDER_TAG,
        )
        cases = [_scan_case(b0, "corrected", **scan_kw) for b0 in B0_VALUES]
        if not _args.no_bare:
            # Solenoids off -> field strength is irrelevant, so the bare ring is
            # scanned once (using the first --b0's lattice file).
            cases.append(_scan_case(B0_VALUES[0], "bare", **scan_kw))
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(data_path, "wb") as f:
            pickle.dump(cases, f)
        print(f"Saved scan data: {data_path}")

    if all(
        (np.isfinite(c["qx"]) & np.isfinite(c["qy"])).sum() < 1 for c in cases
    ):
        raise SystemExit(
            "Every case failed to converge at all delta points -- reduce "
            "--delta-max or raise --n-delta."
        )

    plt.close("all")
    fig = _plot_tune_diagram(
        cases,
        max_resonance_order=_args.max_resonance_order,
        pad=_args.pad,
        qx_range=_args.qx_range,
        qy_range=_args.qy_range,
    )

    b0_tag, order_tag_str, bare_tag, scan_tag = _tags()
    stem = (
        f"tune_diagram_vs_delta_{b0_tag}{order_tag_str}{bare_tag}_{scan_tag}"
        f"_ord{_args.max_resonance_order}"
    )
    _save_fig(fig, stem)

    if not _args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
