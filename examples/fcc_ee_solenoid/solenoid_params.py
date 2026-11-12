"""Shared physical/geometric parameters for the detector solenoid + local
compensation scheme, used across the "current" pipeline
(004a -> 004b[_varsol] -> 004c -> 004d -> 009/010/014).

Single source of truth: these values used to be independently hardcoded in
004a_build_and_check_solenoids.py, 004b_install_solenoids_in_fcc_ring.py, and
004b_install_varsol_solenoids_in_fcc_ring.py, which let them silently drift
out of sync (e.g. B0 changed in one file but not the others). Import from
here instead of re-hardcoding.

Not used by the legacy/superseded 000a/000b/001a/001b path (see
claude_notes/00_overview.md) -- those scripts keep their own independent
copies of these numbers.
"""

# Tilt of the main detector solenoid w.r.t. the beam axis [rad].
THETA = -0.015

# Main detector solenoid: radius [m] and on-axis field strength [T]. The
# half-length depends on which field-strength case is selected -- see
# half_length_for_b0() below.
MAIN_SOLENOID_A = 0.13
MAIN_SOLENOID_B0 = 3.0

# Compensation solenoids (one on each side of the IP): length [m], radius
# [m], and field strength [T]. B0 here is unscaled -- 004a computes a scale
# factor automatically so the net integrated field (main + 2x compensation)
# cancels, regardless of MAIN_SOLENOID_B0.
COMP_SOLENOID_LENGTH = 1.5
COMP_SOLENOID_A = 0.03
COMP_SOLENOID_B0 = 1.0

# Distance of each compensation solenoid from the IP [m].
COMP_SOLENOID_DISTANCE_FROM_IP = 12.0

# s-range (from the IP-side end of the main solenoid) over which the first
# dipole corrector is overlaid with the main solenoid slices, on each side
# [m]. ds_start depends on the field-strength case (see
# corrector_ds_start_for_b0() below); ds_end is the same for every case.
MAIN_SOLENOID_CORRECTOR_DS_END = 2.29

# Marker offset from the IP for the compensation-solenoid orbit/optics
# correctors [m], and the (thin, isthick=False) corrector length [m].
COMPENSATION_CORRECTOR_MARKER_DS = 11.95
COMPENSATION_CORRECTOR_LENGTH = 1.0

# Main-solenoid half-length [m] (full length = 2x this) and first-corrector
# ds_start [m], keyed by MAIN_SOLENOID_B0 -- the 3T detector design is
# physically longer (2.6 m full length) than the 2T one (2.46 m) and its
# first corrector sits farther out. Selected at runtime via --b0 (see
# add_b0_argument below), so 004a/004b[_varsol] build/install each case with
# its own correct geometry instead of one case silently borrowing the
# other's numbers.
_MAIN_SOLENOID_HALF_LENGTH_BY_B0 = {
    2.0: 1.23,
    3.0: 1.30,
}
_MAIN_SOLENOID_CORRECTOR_DS_START_BY_B0 = {
    2.0: 1.23,
    3.0: 1.40,
}


def _lookup_by_b0(table: dict, b0: float, what: str) -> float:
    try:
        return table[b0]
    except KeyError:
        raise ValueError(
            f"No known {what} for B0={b0!r} T -- add an entry to "
            f"solenoid_params.py (known B0 values: {sorted(table)}).")


def half_length_for_b0(b0: float = MAIN_SOLENOID_B0) -> float:
    """Main-solenoid half-length [m] (full length = 2x this) for the given
    on-axis field strength -- see _MAIN_SOLENOID_HALF_LENGTH_BY_B0 above."""
    return _lookup_by_b0(
        _MAIN_SOLENOID_HALF_LENGTH_BY_B0, b0, 'main-solenoid half-length')


def corrector_ds_start_for_b0(b0: float = MAIN_SOLENOID_B0) -> float:
    """s-offset [m] from the IP-side end of the main solenoid where the
    first dipole corrector starts, for the given on-axis field strength --
    see _MAIN_SOLENOID_CORRECTOR_DS_START_BY_B0 above."""
    return _lookup_by_b0(
        _MAIN_SOLENOID_CORRECTOR_DS_START_BY_B0, b0, 'corrector ds_start')


# Module-level defaults for the default MAIN_SOLENOID_B0 -- kept for any
# caller that wants the default case's geometry without going through the
# CLI-selected B0.
MAIN_SOLENOID_HALF_LENGTH = half_length_for_b0()
MAIN_SOLENOID_CORRECTOR_DS_START = corrector_ds_start_for_b0()


# Name prefix of the zero-length trim quadrupoles that
# 004c_correct_solenoids_in_fcc_ring.py installs at the centre of each of the
# six bends framing an IP (see its "Mid-bend trim quadrupoles" section). They
# are xt.Quadrupole, so the element_type == 'Quadrupole' scans that 004c and
# 004f-004j use to discover coupling-corrector host quads would otherwise pick
# them up. A zero-length Quadrupole keeps k1s but yields ksl = 0, so each one
# would be an exactly-dead skew knob padding the ~84-knob coupling solve and
# adding a spurious exact zero to 004g's singular-value spectrum. Every one of
# those scans therefore filters this prefix out.
BEND_MID_QUAD_PREFIX = 'qbmid_'


def field_tag(b0: float = MAIN_SOLENOID_B0) -> str:
    """Filename-safe tag for a main-solenoid field strength, e.g. 3.0 -> '3T'."""
    text = f"{b0:g}".replace(".", "p").replace("-", "m")
    return f"{text}T"


# Tag identifying the main-solenoid field strength (MAIN_SOLENOID_B0) this
# module is currently configured for. Threaded into filenames across the
# 004a-004d/009-015 pipeline and aperture_study_io.py so lattices/studies
# built at different field strengths don't silently overwrite each other.
FIELD_TAG = field_tag()


def add_b0_argument(parser, *, default: float = MAIN_SOLENOID_B0) -> None:
    """Add a ``--b0`` CLI flag for selecting the main-solenoid field strength.

    Lets build/run scripts pick a field strength at runtime instead of
    hand-editing MAIN_SOLENOID_B0 above -- the chosen value is turned into a
    field_tag() to build or select the matching tagged lattice files (e.g.
    ``2.0`` -> ``2T``, ``3.0`` -> ``3T``). Only selects/tags files; it does
    not itself validate that a build for that value exists on disk.
    """
    parser.add_argument(
        "--b0",
        type=float,
        default=default,
        metavar="TESLA",
        help=(
            "Main-solenoid field strength in tesla, selects which "
            f"field_tag()-tagged lattice files to build/use (default: "
            f"{default:g} T -> tag {field_tag(default)!r})."
        ),
    )


# Default cap on the transverse multipole order (0..4) baked into each
# SplineBoris solenoid slice by build_splineboris_line() in
# spline_boris_setup.py -- this directly sets each installed xt.SplineBoris
# element's multipole_order, i.e. how many bx/by polynomial terms the Boris
# pusher evaluates per step. Lower orders drop higher multipole content
# (order 2 = sextupole, the order the 004b `sext_amp` knob scales) in
# exchange for cheaper tracking. Kept in sync with the module-level default
# in 004a_build_and_check_solenoids.py (the only place it is actually built).
MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE = 4


def order_tag(
        max_order: int = MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE) -> str:
    """Filename-safe tag for a non-default SplineBoris transverse-order cap,
    e.g. 4 (the default) -> '', 2 -> '_o2'. Silent at the default so builds
    that never touch this flag keep their original (untagged) filenames."""
    if max_order == MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE:
        return ""
    return f"_o{max_order}"


# Solenoid models selectable with --model. Each name is also the token used in
# the lattice filenames (fccee_z_lcc_{model}_solenoids_..., temp_fcc_ee_lcc_
# {model}_solenoids_...). The *bfe models are the SplineBoris cases with every
# slice converted to xt.BFieldExpansion (bfe_setup.py); they share the field,
# templates and knobs of their SplineBoris counterpart.
SOLENOID_MODELS = ("splineboris", "bfe", "varsol", "partialmap", "partialmap_bfe")
# Models built from the analytic 004a templates at a --max-transverse-order
# cap: only these carry order_tag() in their filenames. VarSol is linear-only
# and the field map's order is fixed by the 004b fit.
ORDER_TAGGED_MODELS = ("splineboris", "bfe")
# The 2 T BNL field map (004b_install_partial_2T_map_in_fcc_ring.py).
FIELD_MAP_MODELS = ("partialmap", "partialmap_bfe")


def is_bfe_model(model: str) -> bool:
    """True for the models whose solenoids are xt.BFieldExpansion slices."""
    return model in ("bfe", "partialmap_bfe")


def model_order_tag(
        model: str,
        max_order: int = MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE) -> str:
    """order_tag() for the models that carry it, '' for the others."""
    return order_tag(max_order) if model in ORDER_TAGGED_MODELS else ""


def add_model_argument(parser, *, default: str = "splineboris",
                       choices=SOLENOID_MODELS) -> None:
    """Add the ``--model`` CLI flag shared by the 004 pipeline scripts."""
    parser.add_argument(
        "--model",
        choices=list(choices),
        default=default,
        help=(
            f"Solenoid model (default: {default}). bfe = the SplineBoris "
            "solenoids converted to BFieldExpansion; partialmap[_bfe] = the "
            "2 T BNL field map up to 2.4 m from each IP, needs --b0 2.0."
        ),
    )


def add_max_order_argument(
        parser, *,
        default: int = MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE) -> None:
    """Add a ``--max-transverse-order`` CLI flag capping the SplineBoris
    transverse multipole order baked into the solenoid field model.

    Mirrors add_b0_argument(): the chosen value is turned into an
    order_tag() to build or select the matching tagged lattice files. Only
    selects/tags files; it does not itself validate that a build for that
    value exists on disk. Valid range is 0..4 (0..MAX_TRANSVERSE_DERIVATIVE_
    ORDER, the fixed field-extraction order in 004a -- raising that would be
    needed to go higher).
    """
    parser.add_argument(
        "--max-transverse-order",
        type=int,
        default=default,
        metavar="N",
        help=(
            "Max transverse multipole order (0..4) baked into each "
            "SplineBoris solenoid slice, selects which order_tag()-tagged "
            f"lattice files to build/use (default: {default})."
        ),
    )
