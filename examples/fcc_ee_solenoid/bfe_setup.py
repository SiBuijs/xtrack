"""SplineBoris -> BFieldExpansion conversion for the FCC-ee IR solenoid study.

The study builds its solenoids as chains of xt.SplineBoris slices: one quartic
Hermite polynomial per slice and per field component, fields in tesla. This
module converts already-built SplineBoris slices into xt.BFieldExpansion
slices carrying the same field, so the analytic path
(spline_boris_setup.build_splineboris_line) and the BNL field-map path
(LongitudinalFitter.to_line) are both covered without touching the library.

Convention (see examples/bfieldexpansion/compare_splineboris_speed.py):
SplineBoris bx[i], by[i], bs are the i-th x-derivatives of Bx, By and the
on-axis Bs at x = y = 0; BFieldExpansion ksc[i], knc[i], ksolc are the same
quantities divided by the reference rigidity, as polynomial coefficients in
the local s (ascending powers, s in m). No factorial or sign conversion.

pkin_const=True keeps SplineBoris' kinetic-momentum convention at the slice
boundaries (claude_notes/01 and 11).
"""
import numpy as np
import xtrack as xt

from xtrack.beam_elements.splineboris_src.spline_B_field_eval_python import (
    hermite_to_polynomial)

# RK4 steps per 2.4 cm slice, from bfe_check_conversion.py (2026-11-03):
# 2 steps cost about as much as SplineBoris' 10 Boris steps and are ~15x
# more accurate (main 2T: 4.6e-13 vs 7.1e-12 max exit error; 3T similar).
DEFAULT_BFE_STEPS = 2


def _hermite_row(coeffs, length, rigidity0):
    poly = hermite_to_polynomial(0.0, length, np.asarray(coeffs, dtype=float))
    return poly.coef / rigidity0


def splineboris_to_bfe(element, *, rigidity0,
                       num_integration_steps=DEFAULT_BFE_STEPS,
                       keep_scale=True):
    """Return a BFieldExpansion carrying the field of one SplineBoris slice.

    ``scale_b`` becomes ``kscale`` (when ``keep_scale``); the coefficient
    arrays themselves are unscaled, as in SplineBoris. ``shift_x/shift_y``
    (field-map offsets) are not supported here and must be zero.
    """
    if not isinstance(element, xt.SplineBoris):
        raise TypeError(f'expected SplineBoris, got {type(element).__name__}')
    if element.shift_x != 0 or element.shift_y != 0:
        raise NotImplementedError(
            'SplineBoris field-map offsets are not converted; use the '
            'standard BFieldExpansion shift_x/shift_y instead.')
    knl = np.asarray(element.knl, dtype=float)
    ksl = np.asarray(element.ksl, dtype=float)
    if np.any(knl != 0) or np.any(ksl != 0):
        raise NotImplementedError('extra SplineBoris knl/ksl are not converted')

    length = float(element.length)
    bs = np.asarray(element.bs, dtype=float)
    bx = np.atleast_2d(np.asarray(element.bx, dtype=float))
    by = np.atleast_2d(np.asarray(element.by, dtype=float))

    scale = getattr(element, 'scale_b', 1.0)
    scale = 1.0 if scale is None else float(scale)

    return xt.BFieldExpansion(
        length=length,
        ksolc=_hermite_row(bs, length, rigidity0),
        ksc=np.array([_hermite_row(row, length, rigidity0) for row in bx]),
        knc=np.array([_hermite_row(row, length, rigidity0) for row in by]),
        kscale=scale if keep_scale else 1.0,
        s_start=0.0,
        num_integration_steps=num_integration_steps,
        pkin_const=True,
    )


def sb_line_to_bfe_line(line, *, rigidity0,
                        num_integration_steps=DEFAULT_BFE_STEPS):
    """Convert every SplineBoris element of ``line``; others are copied."""
    elements = []
    for element in line.elements:
        if isinstance(element, xt.SplineBoris):
            elements.append(splineboris_to_bfe(
                element, rigidity0=rigidity0,
                num_integration_steps=num_integration_steps))
        else:
            elements.append(element.copy())
    out = xt.Line(elements=elements, element_names=list(line.element_names))
    if line.particle_ref is not None:
        out.particle_ref = line.particle_ref.copy()
    return out


def bfe_field_tesla(element, x, y, s_local, *, rigidity0):
    """(Bx, By, Bs) in tesla, including kscale, like SplineBoris.get_field."""
    fv = element.get_field(x, y, s_local)
    return (fv['Bx'] * rigidity0, fv['By'] * rigidity0, fv['Bs'] * rigidity0)


def ksol_integral(element):
    """Integrated Bs/(B rho) of one BFieldExpansion slice, including kscale."""
    if isinstance(element, xt.BFieldExpansion):
        return float(element.ksoll[0])
    raise TypeError(f'not a BFieldExpansion: {type(element).__name__}')
