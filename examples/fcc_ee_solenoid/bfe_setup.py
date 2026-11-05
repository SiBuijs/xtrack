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
        # One allocated entry each so 004b can attach the in-solenoid
        # corrector kicks as knl[0]/ksl[0] expressions (BFE array shapes are
        # fixed at construction). NB: kscale multiplies them too, unlike
        # SplineBoris' scale_b -- see check_bfe_nominal_field().
        knl=[0.0],
        ksl=[0.0],
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


# --- Model-agnostic access to solenoid slices (SplineBoris or BFE) --------- #
# The pipeline scripts use these instead of hard-coding xt.SplineBoris, so the
# same code installs, scales and inspects either element type.

FIELD_ELEMENT_TYPES = (xt.SplineBoris, xt.BFieldExpansion)
FIELD_ELEMENT_TYPE_NAMES = tuple(cls.__name__ for cls in FIELD_ELEMENT_TYPES)


def is_field_element(element):
    """True for a SplineBoris or BFieldExpansion solenoid slice."""
    return isinstance(element, FIELD_ELEMENT_TYPES)


def scale_attr(element):
    """Name of the overall field multiplier: 'scale_b' (SB) or 'kscale' (BFE)."""
    if isinstance(element, xt.BFieldExpansion):
        return 'kscale'
    if isinstance(element, xt.SplineBoris):
        return 'scale_b'
    raise TypeError(f'not a solenoid slice: {type(element).__name__}')


def get_scale(element):
    """Current overall field multiplier (None, as SB may store it, -> 1)."""
    value = getattr(element, scale_attr(element), 1.0)
    return 1.0 if value is None else float(value)


def coefficient_attrs(element):
    """(skew, normal) transverse-coefficient arrays: ('bx', 'by') for SB
    (Hermite data, 5 per row) or ('ksc', 'knc') for BFE (polynomial
    coefficients). Both are linear in the field, so scaling or zeroing a
    whole row means the same thing for either."""
    if isinstance(element, xt.BFieldExpansion):
        return ('ksc', 'knc')
    if isinstance(element, xt.SplineBoris):
        return ('bx', 'by')
    raise TypeError(f'not a solenoid slice: {type(element).__name__}')


def transverse_order(element):
    """Highest transverse multipole row stored (SB multipole_order - 1)."""
    if isinstance(element, xt.BFieldExpansion):
        return element.order
    return element.multipole_order - 1


def ksol_integral(element, *, rigidity0):
    """Integrated Bs/(B rho) of one slice, including its field multiplier."""
    if isinstance(element, xt.BFieldExpansion):
        return float(element.ksoll[0])
    if isinstance(element, xt.SplineBoris):
        return get_scale(element) * element.bs[4] * element.length / rigidity0
    raise TypeError(f'not a solenoid slice: {type(element).__name__}')


def bs_mean_tesla(element, *, rigidity0):
    """Mean Bs over one slice in T, including its field multiplier."""
    return ksol_integral(element, rigidity0=rigidity0) * rigidity0 / element.length


def field_tesla(element, x, y, s_local, *, rigidity0):
    """(Bx, By, Bs) in T at local s, for either element type."""
    if isinstance(element, xt.BFieldExpansion):
        return bfe_field_tesla(element, x, y, s_local, rigidity0=rigidity0)
    return tuple(element.get_field(x, y, s_local))


def check_bfe_nominal_field(line, *, tol=1e-12):
    """Refuse BFE slices whose corrector kick would be rescaled by kscale.

    BFieldExpansion multiplies knl/ksl by kscale, SplineBoris' scale_b does
    not. 004b puts the in-solenoid correctors on the main-solenoid slices as
    knl[0]/ksl[0], so BFE reproduces SB only where those slices sit at
    kscale = 1 (on_sol = main_b_scale = 1). Decision 2026-11-03: --model bfe
    is valid at nominal field only; main_b_scale scans and solenoid-off cases
    with live correctors stay on SplineBoris. Raises ValueError otherwise.
    """
    bad = []
    for name in line.element_names:
        element = line[name]
        if not isinstance(element, xt.BFieldExpansion):
            continue
        kicks = np.r_[np.asarray(element.knl), np.asarray(element.ksl)]
        if np.any(kicks != 0) and abs(element.kscale - 1.0) > tol:
            bad.append((name, float(element.kscale)))
    if bad:
        name, kscale = bad[0]
        raise ValueError(
            f'{len(bad)} BFieldExpansion slices carry corrector kicks at '
            f'kscale != 1 (first: {name}, kscale={kscale:.6g}); BFE scales '
            'knl/ksl with kscale, SplineBoris does not. --model bfe is only '
            'valid with on_sol = main_b_scale = 1.')
    return True


# Boris steps per slice for the integrator='boris' cases (bfeboris_on in
# 009/010/018): the same step count as SplineBoris' production setting, so
# BFE-Boris is the same algorithm at the same resolution as SB. Boris is
# second order; RK4 at DEFAULT_BFE_STEPS is the more accurate (and, per
# unit accuracy, cheaper) choice, see claude_notes/13 step 6.
BORIS_STEPS = 10

# Integrator and steps per slice of the 009/010/018 BFE cases. The two cases
# share the RK4-corrected BFE lattice, so they differ only in the integrator.
BFE_CASE_INTEGRATORS = {
    "BFE": ("rk4", DEFAULT_BFE_STEPS),
    "BFEBoris": ("boris", BORIS_STEPS),
}


def bfe_lattice_path(here, tag, order_tag_str="", output_tag_str=""):
    """The 004c --model bfe lattice (same name pattern as SplineBoris')."""
    return here / (f"fccee_z_lcc_bfe_solenoids_coupling_corrected_"
                   f"{tag}{order_tag_str}{output_tag_str}.json")


def set_bfe_integrator(line, integrator, num_steps):
    """Set the integrator and steps per slice of every BFieldExpansion slice.

    Call before build_tracker. Returns the number of slices changed.
    """
    n = 0
    for name in line.element_names:
        element = line[name]
        if isinstance(element, xt.BFieldExpansion):
            element.integrator = integrator
            element.num_integration_steps = num_steps
            n += 1
    if n == 0:
        raise ValueError("set_bfe_integrator: no BFieldExpansion slices in the line")
    return n


def prepare_case_line(line, model):
    """Apply a 009/010/018 case's BFE integrator and check the nominal field.

    No-op for the non-BFE models.
    """
    if model not in BFE_CASE_INTEGRATORS:
        return
    integrator, num_steps = BFE_CASE_INTEGRATORS[model]
    n = set_bfe_integrator(line, integrator, num_steps)
    check_bfe_nominal_field(line)
    print(f"  BFieldExpansion: {n} slices, integrator={integrator!r}, "
          f"{num_steps} steps per slice")
