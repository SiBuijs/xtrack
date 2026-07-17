# 015_spin_polarization.py — radiative spin polarization study

Written 2026-07-17. Companion to `00_overview.md`. New script, not part of the
000-014 pipeline numbering discussed elsewhere, but structurally a sibling of
`014_emittance_evolution.py` (same `*_CASES`/CLI pattern) reading the same two
final corrected lattices (`01_lattice_construction_000_004d.md`).

## Physics background

Standard radiative-polarization formalism (A. Chao; also Barber/Ellison-type
references): the Sokolov-Ternov spin-flip synchrotron-radiation effect drives
polarization up toward an asymptotic value `P_inf`, on a timescale `tau_pol`,
while quantum energy-diffusion coupled to spin precession via `dn/ddelta`
(the invariant-spin-field's chromatic derivative) *depolarizes* on a
timescale `tau_depol`. Combined:
`P_eq = P_inf / (1 + tau_pol/tau_depol)`.

## Where these quantities live in xtrack

All computed inside `xtrack.twiss._get_spin_polarization` (in `twiss.py`,
`polarization_analysis=True`, which forces `spin=True` and
`radiation_integrals=True` internally — a single `line.twiss(...)` call with
both `radiation_analysis=True` and `polarization_analysis=True` works,
confirmed working together in this codebase's xtrack 0.106.1):
- `tw.spin_polarization_inf_no_depol` — **this is `P_inf`** (eq. 8.36 in the
  user's reference material; the user didn't know the attribute name, this is
  it). Formula: `8/(5*sqrt(3)) * alpha_minus_co/alpha_plus_co`.
- `tw.spin_t_pol_component_s` — **`tau_pol`**, the pure Sokolov-Ternov
  buildup time (seconds), from `alpha_plus_co` alone (no depolarization
  contribution).
- `tw.spin_t_depol_component_s` — Twiss's own analytic estimate of
  `tau_depol` (seconds), from the `dn/ddelta`-derived depolarizing term
  `alpha_plus - alpha_plus_co`.
- `tw.spin_polarization_eq` — Twiss's own directly-computed equilibrium
  polarization (uses the full `alpha_plus`/`alpha_minus`, i.e. already
  includes the depolarizing correction — not literally eq. 8.37's two-time-
  constant combination, but should closely agree with it; the script
  computes eq. 8.37 independently and prints both for comparison).
- `tw.spin_t_pol_buildup_s` — combined buildup time `1/alpha_plus` (not used
  by name in the script, informational).
- `tw.spin_tune_fractional` — spin tune, printed for diagnostics only.
- Reference standalone script predating this Twiss integration:
  `examples/spin/spin.py` — computes the same `alpha_plus_co`/`alpha_minus_co`
  /`pol_inf` formulas by hand on a LEP lattice; useful cross-check if the
  Twiss internals ever need re-deriving.

## What the script does

Mirrors `014_emittance_evolution.py`'s structure almost exactly (same
`POL_CASES` list with `sb_on`/`varsol_on`/`sb_off`, same
`--cases/--list-cases/--n-turns/--n-part/--sexamp/--no-show` CLI, `sb_off`
excluded from the default case set): builds the ring with
`set_lattice_knobs`, does one `robust_twiss(twiss_method="twiss6d",
radiation_analysis=True, polarization_analysis=True, strengths=True)` to get
both the equilibrium-emittance quantities (for bunch generation) and the spin
quantities above, generates a matched Gaussian bunch **at full equilibrium
emittance** (no `/3` factor like 014 — unlike emittance damping, here the
bunch should already sit at its steady-state energy spread since that's what
sets the depolarization rate, not a transient), sets
`particles.spin_x=0, spin_y=1, spin_z=0` for all particles (full polarization
along y, `P(0)=1`), switches to `configure_radiation(model="quantum")` and
tracks.

Spin tracking activates automatically per-particle whenever
`spin_x/y/z != 0` (see `track_magnet_radiation.h`) — no separate "enable
spin" flag on the line is needed, unlike e.g. `XTRACK_MULTIPOLE_NO_SYNRAD`
which `_get_spin_polarization` itself toggles internally for its own
element-by-element R-matrix probing (not something this script touches).

Per-turn polarization: `P(t) = |<spin_vec>|` — mean spin_x/y/z over (alive)
particles, then vector magnitude. Full polarization = all unit spin vectors
aligned (`P=1`); full depolarization = random directions (`P->0`).

Depolarization is fit as a **straight line** (`_fit_linear_depolarization`,
`np.polyfit` degree 1): `P(n) ~= P0 + slope*n`, `tau_depol = -1/slope`
(turns), converted to seconds via `tw.t_rev0`. This is deliberately *not* an
exponential fit (unlike 014's damping-rate fit) — per the user's framing,
`tau_depol` is always vastly larger than any feasible tracked turn count
(confirmed empirically, see below), so only the linear (leading-order) term
of the true exponential decay is ever resolvable from tracking.

Final combination: `p_eq_derived = p_inf / (1 + tau_pol_s/fit_tau_depol_s)`
(eq. 8.37, using Twiss `P_inf`/`tau_pol` + the tracking-fit `tau_depol`),
printed and plotted alongside Twiss's own direct `spin_polarization_eq` as a
sanity-check comparison — same "fit vs Twiss" comparison spirit as 014's
emittance/damping-rate boxes.

## Empirical numbers (sanity-checked against the real lattice before writing the script)

`fccee_z_lcc_splineboris_solenoids_coupling_corrected.json`, `twiss6d` with
`radiation_analysis=True, polarization_analysis=True` costs about the same
extra ~45-50s as 014's `radiation_analysis=True` alone (the added spin R-matrix
probing is cheap relative to the element-by-element radiation R-matrix that
`radiation_analysis` already computes) — not a significant extra cost per
case.

- **`sb_off`** (solenoids off): `P_inf=0.922`, `tau_pol=848000 s`,
  `tau_depol (Twiss)=2.18e10 s` (~691 years) — depolarization is
  astronomically slow with no solenoid coupling, hence excluded from the
  default case set (same reasoning 014 uses to exclude it, but here the
  margin is even larger).
- **`sb_on`** (solenoids on, corrected): `P_inf=0.887`, `tau_pol=816000 s`,
  `tau_depol (Twiss)=960 s` (~3.18e6 turns), `P_eq (Twiss)=0.00104` — the
  solenoid-induced coupling amplifies `dn/ddelta` enough to drop `tau_depol`
  by ~7 orders of magnitude relative to `sb_off`, which is the physically
  interesting comparison this whole study exists to make.

Tracking speed on this line (SplineBoris-heavy, ~21500 elements) was ~200
particle-turns/sec on the dev machine *when otherwise idle*; under
concurrent CPU load (e.g. a simultaneous `014` run) it can be far slower — a
20-particle/50-turn smoke test took several minutes wall-clock under load
during development. At `tau_depol ~ 3.18e6` turns for `sb_on`, the script's
014-matched defaults (`N_TURNS=10_000`, `N_PART=1000`) only cover ~0.3% of
one depolarization time — a real measurement needs a deliberately larger
`--n-turns`/`--n-part` than those defaults to pull the linear-fit slope out
of finite-N noise; a smoke test with tiny `--n-turns --n-part` will produce a
`fit_tau_depol_s` wildly off from the Twiss value (confirmed: a 20p/50t smoke
run gave `fit_tau_depol_s=7.1e3 s` vs the true `960 s`) — this is expected
noise-floor behavior, not a bug, and resolves with enough turns/particles.
No specific "enough" value has been empirically established yet — a good
next step if this script gets used for real physics conclusions would be a
convergence scan (increasing `--n-turns`/`--n-part` until `fit_tau_depol_s`
stabilizes against the Twiss value for `sb_on`).

## Saved data

`aperture_study_io.py` gained `save_pol_study` (new `StudyTag = "POL"`),
following the exact same `make_study_stem`/`DATA_DIR`+`PLOT_DIR` pattern as
`save_emitt_study`. Like EMIT, there is **no replot support** in
`replot_from_npz`/`012_replot_aperture_from_data.py` for POL files — rerun
the script to regenerate the plot.

Not wired into `013_run_da_and_ma.py` (that orchestrator's scope wasn't
extended to include this new study — run `015_spin_polarization.py`
directly).
