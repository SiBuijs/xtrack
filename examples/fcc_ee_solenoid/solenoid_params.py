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
