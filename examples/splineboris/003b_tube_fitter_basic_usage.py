from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from xtrack._temp.splineboris.tube_fitter import TubeFitter


def _falling(n, k):
    '''n * (n-1) * ... * (n-k+1), for nonneg integers; 0 if k > 0 and n < k.'''
    if k == 0:
        return 1
    if n < k:
        return 0
    result = 1
    for i in range(k):
        result *= (n - i)
    return result


def reconstruct_fit_on_line(fitter, der=0, x=0.0, y=0.0):
    '''
    Evaluate d^der/dx^der of the fitted field along the line (x, y, s), for
    all s in ``fitter.s_full``, reconstructed from the fitted scalar
    potential

        psi(x, y, s) = sum_{(p, q)} C_pq(s) x^p y^q,   B = -grad psi

    where C_pq(s) is the degree-1 B-spline with coefficients
    ``fitter.Psi[:, p, q]`` on ``fitter.knots``. The tube fit is global (it
    uses the raw data at every transverse position, not just on axis), so
    this is just the same fit sampled off axis and differentiated -- nothing
    is re-fitted. At x = y = 0 this reproduces TubeFitter's own on-axis
    multipole columns (``_on_axis_multipole_from_psi``) for Bx, By.

    Bs has no transverse dependence beyond what the (p, q) terms carry (the
    dominant on-axis part is fitted separately as ``fitter.Psi_bs``, since a
    constant offset in the potential doesn't show up in Bx/By at all); at
    der=0 that on-axis part is added back in, at der>=1 it drops out anyway
    (it's constant in x).
    '''
    s = fitter.s_full
    k = 1

    bx = np.zeros_like(s)
    by = np.zeros_like(s)
    bs = np.zeros_like(s)
    for (p, q) in fitter.pq_pairs:
        c = BSpline(fitter.knots, fitter.Psi[:, p, q], k)

        # d^der/dx^der acts on the x^p factor in every term below (never on
        # y^q) -- Bx picks up an extra power of x from -d/dx itself, folded
        # into the falling factorial as fall(p, der+1) = p * fall(p-1, der).
        fall_x = _falling(p, der + 1)
        if fall_x:
            bx -= fall_x * c(s) * x ** (p - 1 - der) * y ** q

        fall_p = _falling(p, der)
        if fall_p and q >= 1:
            by -= q * fall_p * c(s) * x ** (p - der) * y ** (q - 1)

        if fall_p:
            bs -= fall_p * c.derivative()(s) * x ** (p - der) * y ** q

    if der == 0 and fitter.Psi_bs is not None:
        bs = bs + BSpline(fitter.knots, fitter.Psi_bs, k)(s)

    return bx, by, bs


def _raw_on_line(fitter, x=0.0, y=0.0):
    '''Raw data rows nearest to the transverse point (x, y), one per s.'''
    idx = fitter.df_raw_data.index
    x_grid = np.unique(idx.get_level_values("X").to_numpy(dtype=float))
    y_grid = np.unique(idx.get_level_values("Y").to_numpy(dtype=float))
    x_near = x_grid[np.argmin(np.abs(x_grid - x))]
    y_near = y_grid[np.argmin(np.abs(y_grid - y))]
    df = fitter.df_raw_data.xs((x_near, y_near), level=["X", "Y"]).sort_index()
    return (x_near, y_near), df


def plot_data_and_fits(fitter, der=0, x=0.0, y=0.0):
    '''
    Plot raw data vs. fit along the line (x, y, s).

    This mirrors TubeFitter.plot_fields, but (a) omits the per-slice vertical
    dashed lines marking the B-spline frame borders, and (b) lets you pick
    the transverse point instead of being fixed to the axis. ``x`` and ``y``
    are in metres (x = 0.5 mm -> x=0.0005).

    On axis (x == y == 0), der>0 reuses TubeFitter's own on-axis columns.
    Off axis, der>0 is reconstructed the same way TubeFitter derives those
    on-axis columns in the first place -- by differentiating the fitted
    potential -- just evaluated away from the axis; the "measured" curve for
    der>0 is therefore, as on axis, the tube-multipole reconstruction rather
    than independently measured data (there is no direct measurement of a
    field derivative), while the "fit" curve is additionally masked by which
    components ``fit()`` judged significant (``component_to_fit``).
    '''
    import matplotlib.pyplot as plt

    s = fitter.s_full
    on_axis = (x == 0.0 and y == 0.0)

    if on_axis:
        def series(df, field):
            try:
                return df[(field, der)].to_numpy()
            except KeyError:
                return np.zeros_like(s)

        bx_raw, by_raw, bs_raw = (series(fitter.df_on_axis_raw, f) for f in ("Bx", "By", "Bs"))
        bx_fit, by_fit, bs_fit = (series(fitter.df_on_axis_fit, f) for f in ("Bx", "By", "Bs"))
        x_near, y_near = 0.0, 0.0
    else:
        if der == 0:
            (x_near, y_near), df_raw = _raw_on_line(fitter, x, y)
            bx_raw = df_raw["Bx"].to_numpy()
            by_raw = df_raw["By"].to_numpy()
            bs_raw = df_raw["Bs"].to_numpy() if "Bs" in df_raw else np.zeros_like(bx_raw)
        else:
            x_near, y_near = x, y
            bx_raw, by_raw, bs_raw = reconstruct_fit_on_line(fitter, der, x_near, y_near)

        bx_fit, by_fit, bs_fit = reconstruct_fit_on_line(fitter, der, x_near, y_near)
        if not fitter.component_to_fit.get(("Bskew", der), False):
            bx_fit = np.zeros_like(s)
        if not fitter.component_to_fit.get(("Bnorm", der), False):
            by_fit = np.zeros_like(s)
        if der != 0 or not fitter.component_to_fit.get(("Bs", 0), False):
            bs_fit = np.zeros_like(s)

    if der == 2:
        x_label, y_label, s_label = (
            r"$\frac{d^2 B_x}{d x^2}$", r"$\frac{d^2 B_y}{d x^2}$", r"$\frac{d^2 B_s}{d x^2}$")
    elif der == 1:
        x_label, y_label, s_label = (
            r"$\frac{d B_x}{d x}$", r"$\frac{d B_y}{d x}$", r"$\frac{d B_s}{d x}$")
    else:
        x_label, y_label, s_label = r"$B_x$", r"$B_y$", r"$B_s$"

    raw_label = "Measured" if der == 0 else "Tube multipoles"
    fit_label = "Fit" if der == 0 else "Exported (to_fit)"

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)
    for ax, raw, fit in [(ax1, bx_raw, bx_fit), (ax2, by_raw, by_fit), (ax3, bs_raw, bs_fit)]:
        ax.plot(s, raw, label=raw_label)
        ax.plot(s, fit, label=fit_label, linestyle="--")

    ax1.set_title(f"Magnetic Field at (X, Y) = ({x_near:g}, {y_near:g}) m")
    ax1.set_ylabel(f"Horizontal Field, {x_label} [T]")
    ax2.set_ylabel(f"Vertical Field, {y_label} [T]")
    ax3.set_ylabel(f"Longitudinal Field, {s_label} [T]")
    ax3.set_xlabel(r"Longitudinal Position, $s$ [m]")
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")
    ax3.legend(loc="upper right")
    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()


'''
Basic usage of TubeFitter.

This script fits the same field map as 003a_fieldfitter_basic_usage.py, but
using the tube approach (global sparse fit over a longitudinal B-spline
basis, see Riemann & Aiba, IPAC2021, TUPAB238) instead of FieldFitter's
sequential Hermite-piecewise regions.

It plots the fit results for each derivative order and the integrated field
along the longitudinal direction.

The raw data only has three transverse x positions, which means the highest
order polynomial that we can fit is 2. This also means that we can only
incorporate up to the second derivative of the field into the fit (sextupole
components).
'''

dz = 0.001  # Step size in the z (longitudinal) direction for numerical differentiation

# Convert the field map to a DataFrame. TubeFitter expects columns Bx, By, Bs
# (FieldFitter uses the equivalent Bskew, Bnorm, Bs naming for the same data).
file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "simona_field_map.txt"
df_raw_data = pd.read_csv(
    file_path, sep="\t", header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
    dtype=float,
).set_index(["X", "Y", "Z"])

deg = 4

n_frames = 4441

# Below, you can either choose n_frames or residual_tol
# If you choose n_frames, the fitter will simply use that number of frames.
# If you choose residual_tol, the fitter will search for the smallest number of frames
# that meets the specified residual tolerance.
# Some reference times (depdendent on the machine and field map (simona_field_map.txt, deg=4 here)):
# n_frames=1000     : takes ~30 seconds     lands at ~5.7e-3 residual
# residual_tol=1e-3 : takes ~10 minutes     lands at n_frames=2324
# If we simply choose n_frames=4441 (the maximum), it takes ~35 seconds and lands at ~6e-6 residual

import time

# Temporarily silence the automatic n_frames-search convergence plot so it
# doesn't pop up a blocking window mid-timing -- remove this line to see it
# again. (Only affects plot_n_frames_search; plot_fields/plot_integrated_fields
# below are untouched.)
TubeFitter.plot_n_frames_search = lambda self, *a, **kw: None

start_time = time.time()

fitter = TubeFitter(
    raw_data=df_raw_data,
    n_frames=n_frames,
    #residual_tol=1e-3,
    distance_unit=dz,
    deg=deg,
    field_tol=1e-3,
    #tube_radius=0.0005,
)
fitter.fit()

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds for {fitter.n_frames} frames")

for der in range(0, deg + 1):
    plot_data_and_fits(fitter, der=der)

# Raw data vs. fit sampled off axis, at x = 0.5 mm, for every derivative
# order. The tube fit is global, so this is the same fit evaluated (and, for
# der>0, differentiated) at a different transverse point -- no re-fitting is
# needed or possible here.
for der in range(0, deg + 1):
    plot_data_and_fits(fitter, der=der, x=0.0005)

fitter.plot_integrated_fields()
