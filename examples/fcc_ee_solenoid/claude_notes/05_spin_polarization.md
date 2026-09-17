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

~~Not wired into `013_run_da_and_ma.py`~~ — **superseded 2026-09-17, see
below.** 015 itself is still not wired into 013, but the combined script
`018_emittance_and_polarization.py` is, and it covers the same study.

---

# 018_emittance_and_polarization.py — the 014+015 merge (2026-09-17)

014 and 015 each ran their own full tracking pass (1000p x 10000t, quantum
radiation) over the same lattice, and 270 of their 503 lines were
byte-identical. They differed in exactly three places:

| | 014 | 015 |
|---|---|---|
| Twiss | `radiation_analysis=True` | + `polarization_analysis=True` (strict superset) |
| Bunch | `eq_nemitt/3` | full equilibrium |
| Spin IC | not set | `spin_y = 1.0` |

018 does **one** twiss, **one** bunch, **one** `line.track()` and **one**
turn-by-turn monitor, then runs both analyses off that monitor and calls both
`save_emitt_study` and `save_pol_study` unchanged. 013's tracking stage now
points at 018 (`EMITT_POL_SCRIPT`), so a default 013 run yields DA + MA +
EMIT + POL. **014 and 015 are deliberately left byte-untouched** so existing
results stay reproducible.

## Why spin is free when it is off

`magnet_spin` (`xtrack/beam_elements/elements_src/track_magnet_radiation.h`)
early-returns when all three spin components are zero, and `spin_flag` is
hardcoded to `1` in the magnet kernels — so 014 already walked the spin code
path, it just no-opped. `magnet_spin` also consumes no random numbers and
never writes back to the orbit. Consequence, verified: **at a fixed `--seed`
018's emittance arrays are bit-identical to a 014 run.**

## The spin-IC trap (easy to get wrong)

`twiss(spin=True)` writes the invariant spin field onto `tw.particle_on_co`
(`xtrack/twiss.py`, `_find_spin_fixed_point`) and `build_particles` copies
`particle_ref`'s spin into every generated particle
(`xpart/build_particles.py:447-449`). So **after a polarization twiss the
generated bunch is NOT spin-zero**, and a `--no-pol` mode that merely
"doesn't set spin" would silently do full spin tracking. 018 therefore zeroes
spin explicitly in the `--no-pol` branch, and drops
`polarization_analysis` from the twiss as well.

The same mechanism explains the shape of the P(n) curve: 015's `spin_y=1`
*overwrites* the matched `n0` direction, so the bunch relaxes back onto it
in the first turns. That relaxation is ~the entire 1e-5 drop seen over a
10 000-turn run.

## Bunch IC and the fit window

018 uses **014's eq/3 bunch** (`--bunch-emitt-divisor`, default 3), so the
damping-rate fit still has a transient to fit (it works: fitted alpha_x
7.17e-4 vs Twiss 7.73e-4). The polarization fit compensates with
`--pol-fit-start-turn` (default 2000, matching 016's `--turn-start`), which
drops both the spin-IC relaxation and the bulk of the damping transient —
tau_z ~ 600 turns, so by turn 2000 the longitudinal plane, which drives
dn/ddelta depolarization, is within ~1% of equilibrium.

This is defensible mainly because **the tracked tau_depol is noise-dominated
anyway.** Measured on `POL_Sol_On_SB_3T_1000p_10000t`: total P drop 1.0e-5
over the run, nearly all of it in turn 0->1; from turn 1000-9000 P falls
5.5e-7 against 4.4e-7 turn-to-turn scatter, i.e. SNR ~ 1. Moving the fit
start from 0 to 6000 swings tau_depol from 8.9e9 to 4.0e10 turns, and the
fitted 2.7e6 s disagrees with Twiss's analytic 960 s by ~2800x (P_eq derived
0.68 vs Twiss 0.00104). **Do not treat a non-reproducing `fit_tau_depol_*`
as a merge bug** — it is the pre-existing noise floor, and it is why 016
exists.

## Provenance tagging (`__bunchdiv<F>`)

Filenames did not record the bunch IC, so a POL file made from an eq/3 bunch
would have been indistinguishable on disk from one 015 made from an
equilibrium bunch. 018 tags each study against *its own parent's* divisor:

| run | EMIT stem | POL stem |
|---|---|---|
| default (divisor 3) | untagged = 014's | `…__bunchdiv3p0` |
| `--bunch-emitt-divisor 1` | `…__bunchdiv1p0` | untagged = 015's |

So an untagged file always means "same physics the legacy script would have
written". `EMIT_`/`POL_` prefixes are unchanged, so 016's `glob("POL_*.npz")`
still finds everything and its PDF stem inherits the tag.
`--bunch-emitt-divisor 1 --no-emitt` reproduces 015 exactly.

`aperture_study_io.py` gained two additive, backward-compatible kwargs:
`save_pol_study(fit_turn_start=0, bunch_emitt_divisor=None)` and
`save_emitt_study(bunch_emitt_divisor=None)`, stored in the npz (`np.nan` =
"not recorded", which is what 014/015 write since they don't pass it).
`fit_turn_start=0` is literally correct for 015's full-range fit.

## Determinism

Neither parent seeded anything, but the whole chain runs off the global numpy
RNG: bunch generation uses `np.random.*`, and the per-particle
quantum-radiation seeds come from `np.random.randint` in
`particles._init_random_number_generator`, called at the first `track()`
because `configure_radiation('quantum')` sets `line._needs_rng`. So
`np.random.seed()` before bunch generation makes the whole run reproducible —
exposed as `--seed`.

This works **only because radiation is still `'mean'` at twiss time**, so the
twiss's internal probe tracking draws no `np.random` numbers. Moving
`configure_radiation('quantum')` above the twiss would silently break it.

## Flags

`--no-emitt` / `--no-pol` (mutually exclusive) select halves; the bunch IC is
independent of which halves run, so `--no-emitt` stays a strict subset of the
combined run. Also `--pol-fit-start-turn`, `--bunch-emitt-divisor`, `--seed`.
A guard rejects `--pol-fit-start-turn >= --n-turns` *before* tracking, which
short smoke runs will hit (use e.g. `--pol-fit-start-turn 20`).
013 forwards these as `--emitt-no-pol`, `--emitt-pol-only`,
`--emitt-pol-fit-start-turn`, `--emitt-bunch-divisor`, `--emitt-seed`.
