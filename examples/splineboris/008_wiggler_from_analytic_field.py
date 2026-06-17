import math

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
from xtrack.beam_elements.splineboris import Spline4, SplineBoris


def compute_wiggler_constants(
    g: float,
    B_r: float,
    length: float,
    n_periods: int,
    pole_width: float,
):
    """
    Compute wiggler constants and amplitudes.

    Parameters
    ----------
    g : float
        Magnetic gap [m].
    B_r : float
        Reference on-axis field [T].
    length : float
        Total wiggler length [m].
    n_periods : int
        Number of periods.
    pole_width : float
        Pole width w_p [m]; defines k_x = 2*pi / w_p.
    """
    lambda_u = length / n_periods
    k_s = 2.0 * np.pi / lambda_u
    k_x = 2.0 * np.pi / pole_width
    k_y = np.sqrt(k_x**2 + k_s**2)
    B_0 = B_r / np.cosh(np.pi * g / lambda_u)
    return lambda_u, k_s, k_x, k_y, B_0


def _taylor_cos_coeffs(k_x: float, max_order: int):
    """Return coefficients c_n for cos(k_x x) = sum c_n x^n up to n = max_order."""
    coeffs = np.zeros(max_order + 1, dtype=float)
    for m in range(0, max_order // 2 + 1):
        n = 2 * m
        if n > max_order:
            break
        coeffs[n] = ((-1.0) ** m) * k_x**n / float(math.factorial(n))
    return coeffs


def _make_spline4_for_channel(ampl: float, k_s: float, s0: float, s1: float) -> Spline4:
    """
    Build Spline4 Hermite data for f(s) = ampl * sin(k_s s) over [s0, s1].
    """
    ds = s1 - s0
    val_start = ampl * np.sin(k_s * s0)
    val_end = ampl * np.sin(k_s * s1)
    der_start = ampl * k_s * np.cos(k_s * s0)
    der_end = ampl * k_s * np.cos(k_s * s1)
    if ds == 0:
        mean = val_start
    else:
        mean = ampl * (np.cos(k_s * s0) - np.cos(k_s * s1)) / (k_s * ds)
    return Spline4(
        val_start=float(val_start),
        der_start=float(der_start),
        val_end=float(val_end),
        der_end=float(der_end),
        mean=float(mean),
    )


def build_wiggler_splineboris_sequence(
    g: float,
    B_r: float,
    length: float,
    n_periods: int,
    pole_width: float,
    max_order: int = 3,
    segments_per_period: int = 10,
    n_steps_per_segment: int = 20,
):
    """
    Build a list of SplineBoris elements approximating the analytic wiggler field.

    The analytic model used here is

        B_x = -B_0 (k_x / k_y) sin(k_x x) sinh(k_y y) sin(k_s s)
        B_y =  B_0           cos(k_x x) cosh(k_y y) sin(k_s s)
        B_s =  B_0 (k_s / k_y) cos(k_x x) sinh(k_y y) cos(k_s s)

    We specialize to y = 0, so sinh(k_y y) = 0, cosh(k_y y) = 1 and only
    B_y survives:

        B_y(x, 0, s) = B_0 cos(k_x x) sin(k_s s).

    The cos(k_x x) factor is expanded as a Taylor series in x, and the
    resulting x^n multipole coefficients are encoded as SplineBoris by_n
    channels with purely sinusoidal s-dependence.
    """
    lambda_u, k_s, k_x, _k_y, B_0 = compute_wiggler_constants(
        g=g,
        B_r=B_r,
        length=length,
        n_periods=n_periods,
        pole_width=pole_width,
    )

    coeffs = _taylor_cos_coeffs(k_x=k_x, max_order=max_order)

    total_segments = n_periods * segments_per_period
    ds = length / total_segments

    elements = []
    names = []

    for ii in range(total_segments):
        s0 = ii * ds
        s1 = (ii + 1) * ds

        by_channels = []
        for order, c_n in enumerate(coeffs):
            if abs(c_n) == 0.0:
                by_channels.append(None)
                continue
            ampl = B_0 * c_n
            by_channels.append(_make_spline4_for_channel(ampl=ampl, k_s=k_s, s0=s0, s1=s1))

        element = SplineBoris(
            bs=Spline4(0.0, 0.0, 0.0, 0.0, 0.0),
            bx=(),
            by=tuple(by_channels),
            length=ds,
            n_steps=n_steps_per_segment,
        )
        elements.append(element)
        names.append(f"wiggler_sb_{ii:04d}")

    return elements, names, lambda_u, k_s, k_x, B_0


def analytic_field_y(x, s, B_0, k_x, k_s):
    """
    Analytic reference field B_y(x, 0, s) used for validation.
    """
    return B_0 * np.cos(k_x * x) * np.sin(k_s * s)


# Match the parameters of examples/boris_spatial/005_wiggler_field.py
g = 12e-3
B_r = 1.2
length = 2.0
n_periods = 100

pole_width = 2e-2

elements, names, lambda_u, k_s, k_x, B_0 = build_wiggler_splineboris_sequence(
    g=g,
    B_r=B_r,
    length=length,
    n_periods=n_periods,
    pole_width=pole_width,
    max_order=6,
    segments_per_period=10,
    n_steps_per_segment=20,
)

print(f"lambda_u = {lambda_u:.6e} m")
print(f"k_s      = {k_s:.6e} 1/m")
print(f"k_x      = {k_x:.6e} 1/m")
print(f"B_0      = {B_0:.6e} T")
print(f"Total segments: {len(elements)}")

line = xt.Line(elements=elements, element_names=names)
line.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV,
    q0=1,
    p0c=2.7e9,
)

# Field reconstruction check at y = 0.
n_samp_s = 200
s_grid = np.linspace(0.0, length, n_samp_s, endpoint=False)
x_probe = 1.0e-4

by_analytic = analytic_field_y(x=x_probe, s=s_grid, B_0=B_0, k_x=k_x, k_s=k_s)
by_spline = np.zeros_like(by_analytic)

ds = length / len(elements)
for ee, s_val in enumerate(s_grid):
    # Keep sampling robust at segment boundaries: map s to [0, length) first,
    # then compute idx/s_local and clamp tiny floating-point overshoots.
    s_wrapped = s_val % length
    idx = min(int(s_wrapped / ds), len(elements) - 1)
    el = elements[idx]
    s_local = s_wrapped - idx * ds
    s_local = min(max(s_local, 0.0), el.length)
    _, by_val, _ = el.get_field(x=x_probe, y=0.0, s_local=s_local)
    by_spline[ee] = by_val

max_abs_err = np.max(np.abs(by_spline - by_analytic))
rel_err = max_abs_err / max(1e-12, np.max(np.abs(by_analytic)))
print(f"Max |By_spline - By_analytic| = {max_abs_err:.3e} T")
print(f"Relative max error            = {rel_err:.3e}")

plt.close("all")
line.build_tracker(use_prebuilt_kernels=False)
tw = line.twiss(
    betx=1.0,
    bety=1.0,
    include_collective=False,
)

fig_twiss, ax_twiss = plt.subplots(figsize=(10, 4))
ax_twiss.plot(tw.s, tw.betx, label="betx", color="C0")
ax_twiss.plot(tw.s, tw.bety, label="bety", color="C1")
ax_twiss.set_xlabel("s [m]")
ax_twiss.set_ylabel("beta [m]")
ax_twiss.set_title("Twiss through SplineBoris wiggler (betx0=bety0=1 m)")
ax_twiss.grid(True, alpha=0.3)
ax_twiss.legend(loc="best")
fig_twiss.tight_layout()

# Simple single-particle track through the SplineBoris sequence.
p = line.particle_ref.copy()
p.x = 0.0
p.px = 0.0
p.y = 1.0e-4
p.py = 0.0

line.track(p, turn_by_turn_monitor="ONE_TURN_EBE")
mon = line.record_last_track

print("\nTrack end coordinates (SplineBoris line):")
print(f"  x     = {p.x[0]:+.6e} m")
print(f"  px    = {p.px[0]:+.6e}")
print(f"  y     = {p.y[0]:+.6e} m")
print(f"  py    = {p.py[0]:+.6e}")
print(f"  zeta  = {p.zeta[0]:+.6e} m")
print(f"  delta = {p.delta[0]:+.6e}")

# Orbit projections vs s.
fig_orbit, (ax_x, ax_y) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax_x.plot(mon.s[0, :], mon.x[0, :], color="C0", linewidth=1.6)
ax_x.set_ylabel("x [m]")
ax_x.set_title("Orbit through SplineBoris wiggler")
ax_x.grid(True, alpha=0.3)

ax_y.plot(mon.s[0, :], mon.y[0, :], color="C1", linewidth=1.6)
ax_y.set_xlabel("s [m]")
ax_y.set_ylabel("y [m]")
ax_y.grid(True, alpha=0.3)
fig_orbit.tight_layout()

plt.show()

