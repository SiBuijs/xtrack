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

# Main detector solenoid: half-length [m] (full length = 2 * this), radius
# [m], and on-axis field strength [T].
MAIN_SOLENOID_HALF_LENGTH = 1.23
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
# [m].
MAIN_SOLENOID_CORRECTOR_DS_START = 1.23
MAIN_SOLENOID_CORRECTOR_DS_END = 2.29

# Marker offset from the IP for the compensation-solenoid orbit/optics
# correctors [m], and the (thin, isthick=False) corrector length [m].
COMPENSATION_CORRECTOR_MARKER_DS = 11.95
COMPENSATION_CORRECTOR_LENGTH = 1.0


def field_tag(b0: float = MAIN_SOLENOID_B0) -> str:
    """Filename-safe tag for a main-solenoid field strength, e.g. 3.0 -> '3T'."""
    text = f"{b0:g}".replace(".", "p").replace("-", "m")
    return f"{text}T"


# Tag identifying the main-solenoid field strength (MAIN_SOLENOID_B0) this
# module is currently configured for. Threaded into filenames across the
# 004a-004d/009-015 pipeline and aperture_study_io.py so lattices/studies
# built at different field strengths don't silently overwrite each other.
FIELD_TAG = field_tag()
