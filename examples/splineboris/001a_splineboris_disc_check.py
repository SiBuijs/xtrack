"""
Minimal introduction to the `SplineBoris` API.

We build a single `SplineBoris` element with:

- a **longitudinal** field `Bs(s)` (here zero),
- a **normal dipole** component `By0(s)`,
- a **normal quadrupole** component `By1(s) · x`,

and show:

- how to pass multiple Hermite splines via `by=(..., ...)`,
- basic tracking through the element,
- simple field plots using `SplineBoris.get_field(...)`.
"""

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt



# ---------------------------------------------------------------
# SplineBoris field definition
# ---------------------------------------------------------------

length = np.pi/2
n_steps = 100

# Constant longitudinal field (zero in this example)
Bs = xt.Spline4(
    val_start=0.0,
    der_start=0.0,
    val_end=0.0,
    der_end=0.0,
    integral=0.0,
)

# Slightly asymmetric normal dipole field By0(s)
By0_left = xt.Spline4(
    val_start=1.0,
    der_start=0.0,
    val_end=0.0,
    der_end=-1.0,
    integral=1/length,
)

# Slightly asymmetric normal dipole field By0(s)
By0_right = xt.Spline4(
    val_start=0.0,
    der_start=-1.0,
    val_end=-1.0,
    der_end=0.0,
    integral=-1/length,
)

sb_left  = xt.SplineBoris(bs = Bs, bx=(None,), by=(By0_left,), length=length)
sb_right = xt.SplineBoris(bs = Bs, bx=(None,), by=(By0_right,), length=length)

print(sb_left.get_field(0, 0.01, length))
print(sb_right.get_field(0, 0.01, 0))

from xtrack.beam_elements.splineboris_src.spline_B_field_eval_python import hermite_to_polynomial

leftlist = By0_left.as_list()
rightlist = By0_right.as_list()

poly_left = hermite_to_polynomial(s_start=0, s_end=length, coeffs=leftlist)
poly_right = hermite_to_polynomial(s_start=0, s_end=length, coeffs=rightlist)


# Quick plotter for the left/right polynomial profiles.
s_plot = np.linspace(0.0, length, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(s_plot, poly_left(s_plot), label='poly_left')
ax.plot(s_plot, poly_right(s_plot), label='poly_right')
ax.axhline(0.0, color='k', lw=0.8, alpha=0.5)
ax.set_xlabel('s')
ax.set_ylabel('By(s)')
ax.set_title('SplineBoris Hermite -> Polynomial')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()



print(hermite_to_polynomial(s_start=0, s_end=length, coeffs=[1.0, 0.0, 0.0, -1.0, 1/length]))
print(hermite_to_polynomial(s_start=0, s_end=length, coeffs=[0.0, -1.0, -1.0, 0.0, -1/length]))