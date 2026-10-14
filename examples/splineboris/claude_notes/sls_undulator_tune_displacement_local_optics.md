# `007b_sls_undulator_tune_displacement_local_optics.py` -- same offset scan, phase-advance tune shift, undulator never inserted in the ring

Summary of
`examples/splineboris/007b_sls_undulator_tune_displacement_local_optics.py`
as of 2026-08-25 (`f4e25b0cc`, the commit that added this file), for
orientation before further work on it. This is a note about the code, not a
design doc -- re-read the source for anything load-bearing. Companion notes:
[sls_undulator_tune_displacement_007a.md](sls_undulator_tune_displacement_007a.md)
(007a, the sibling this file was forked from -- read that one first, this
note only covers how/why 007b differs from it),
[tube_fitter_to_line.md](tube_fitter_to_line.md),
[tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md).

## Current state (2026-10-14) -- read this first

The rest of this note describes the file as added on 2026-08-25. Since then
(`21d1bf1e5`, `a8e5ecc17`, `d7343fade`, 2026-10-06..13) the **main result
changed method**, and several "differences from 007a" below no longer hold.

- **The tune shift is now exact, from a one-turn matrix**, not from local
  phase advance. Nothing is inserted into the ring. Per offset, each
  placement's standalone `undulator` is twissed from the bare-ring optics at
  its start, and the 4x4 one-turn map at the first undulator entrance is
  built as `M = ... T_next @ D^-1 @ M_und ...`. Here `M_und` is the
  undulator's own R matrix, `D^-1` removes the drift of length `l_wig` the
  undulator replaces (the replaced span holds only zero-strength
  placeholders), and `T_next` is the bare-ring transfer to the next
  placement (for the last one: `M_bare,one-turn @ T_first->last^-1`, with
  `M_bare` from `W_matrix` and `lnf.Rot2D`). The eigen-tunes are assigned to
  the nearest bare fractional tune (`one_turn_tune_shift`). This should
  reproduce 007a's closed-ring Twiss exactly, and `deltaq{x,y}_list` now
  holds it.
- **The local phase advance is only a diagnostic, and it is not the tune
  shift.** `deltaq{x,y}_local_list` = `mu_und - mu_drift`, with
  `mu_drift = arctan2(L, beta0 - alpha0 L)/2pi`. With the entrance optics
  frozen it only counts `(1/4pi) int K beta (1 - cos 2psi) ds`, about 1/8 of
  the shift here; the rest builds up in the rest of the ring. So point 2
  below ("tune shift is read off phase advance") is **wrong for the current
  file**. The code cites `claude_notes/local_phase_advance_vs_tune_shift.md`
  for this, which is **not in this checkout** (see `00_overview.md`).
- **The 007a vs 007b cross-check now exists** (it was "not done" below).
  `load_ring_tune_shift_007a()` reads 007a's cache `data/{place}_{model}.npz`
  and overlays its closed-ring Twiss tune shift (grey open markers), if it
  was scanned at the same offsets.
- **Fit and models as in 007a now**: `get_undulator_fit()` with
  `TubeFitter` + `LongitudinalFitter`, SB/MK via
  `LongitudinalFitter.to_line()`/`to_multipole_line()`, `FIELD_TOL` at
  export, `field_focusing()` for SB, offsets at the measured bump
  amplitudes, and `CASES = ['ars11_uind_0610_1']`. See the 007a note's
  current-state section. The same dead `MultipoleEdge` filter and comment
  are here too.
- **Figures: 2 per placement**, `{place}_tune_shift_MK_vs_SB_local_optics.pdf`
  (one-turn ΔQ, local Δμ, 007a ring ΔQ, quadratic fit, formula curves,
  measurements) and `{place}_tune_shift_vs_closed_gap_zeroed_local_optics.pdf`.
  The per-case 8-figure set below is gone.
- **Paths**: `OUT_DIR` is the portable `$SLS_UNDULATOR_PLOT_DIR` /
  `~/cernbox/Pictures/SLS_Undulator_Studies` (no `local_optics` subfolder
  any more; the suffix is in the filenames). `data_007b/` was removed from git
  and gitignored on 2026-10-08 (`08afe66f7`), so `--replot` needs a local
  run first. Cache arrays are now shaped `(n_offsets, n_placements, n_pts)`,
  without the `_pp` suffixes described below.

## Why this file exists

007a inserts the undulator into the full SLS ring and reads `ΔQx`/`ΔQy` off
the ring's Twiss (`tw.qx`/`tw.qy`), which mixes in ring-wide effects: the
betatron-coupling closest-tune-approach resonance (the reason 007a needs its
`|C-|` correction machinery at all) and beta-beat feedback around the whole
lattice. 007b instead asks a narrower question -- how much phase advance
does the undulator *by itself* contribute, isolated from the rest of the
ring entirely -- by Twissing only the standalone undulator line, seeded with
the bare ring's local optics (`betx`/`bety`/`alfx`/`alfy` at the placement
point) as initial conditions, and reading the tune-shift contribution
straight off the phase advance (`mux`/`muy`) it adds relative to what that
same physical span of bare ring would have contributed. Created the same day
as the `MultipoleEdge` addition to `to_multipole_line()` (see
[tube_fitter_to_multipole_line.md](tube_fitter_to_multipole_line.md)'s
"Update" section) -- likely motivated by wanting a coupling-free cross-check
of whether that edge-kick change was moving 007a's Twiss-vs-formula residual
for the right physical reason, rather than via ring-wide coupling artifacts.

## What's structurally identical to 007a

`get_tube_fitter()`/module-level field-map fit/cache (`n_frames=4441,
deg=multipole_order-1=2, field_tol=1e-4`), `WIGGLER_CASES` (3 placements) x
`MODEL_LABELS` (`SB`/`MK`, 6 cases total), the undulator build + 4-corrector
match at zero offset, the `_multipole_field_bx_by()` field-reconstruction
helper for MK, the `hor_off_list` scan (`n_tunes=30`, ±0.5 mm), the
per-offset corrector re-match (same rationale as 007a -- fixed correctors
would leak a residual orbit distortion into the phase-advance calculation),
the `x_deflection_baseline` local-track correction, the analytic
`ΔQ = 1/(4π) ∮ ΔK1(s)·β(s) ds` cross-check (same formula, same sign
convention, same `mult_K1`/`mult_K2`/`mult_K2_skew` from a dedicated
`to_multipole_line()` build, same thick-only `mult_names` filtering of the
`MultipoleEdge` kicks -- see
[sls_undulator_tune_displacement_007a.md](sls_undulator_tune_displacement_007a.md)'s
"`MultipoleEdge` thick-filtering fix" section, present in 007b from its
creation rather than added later), `get_case_data()`/`--replot`
caching pattern, and the overall `compute_case()`/`plot_case()` split.

## What's structurally different

**1. The undulator is never inserted into `line_sls`.** `env`/`line_sls` are
loaded only to get `tw_no_undulator` (bare-ring Twiss) for `qx_0`/`qy_0` and
for the local optics at each placement. The standalone `undulator` line
(with its 4 matched correctors) is Twissed directly via
`undulator.twiss4d(betx=ic['betx'], bety=ic['bety'], alfx=ic['alfx'],
alfy=ic['alfy'], include_collective=True)` (`_twiss_undulator(ic)`), once per
placement, both for the field-along-trajectory sampling and inside the
offset scan.

**2. Tune shift is read off phase advance, not `tw.qx`/`tw.qy`.** Per
placement `ic`, per offset:
```
dqx_total += tw_p.mux[-1] - (mux_bare_end - mux_bare_start)
dqy_total += tw_p.muy[-1] - (muy_bare_end - muy_bare_start)
```
where `mux_bare_start`/`mux_bare_end` are `tw_no_undulator.mux` interpolated
at the placement's start/end `s`. For the `'both'` case this is a sum over
two *independent* Twiss problems (each undulator gets its own initial
conditions from the bare ring, at its own placement) -- no cross-talk
between the two placements, unlike 007a where both undulator copies sit in
the same ring and see each other's effect on the closed orbit/optics
implicitly.

**3. No coupling correction, no `c_minus`.** This is the direct consequence
of point 2: there is no periodic/closed-ring quantity to compute `c_minus`
from when Twissing an open, non-periodic standalone line seeded with
arbitrary initial conditions, so 007a's entire "Coupling correction of the
analytic cross-check" machinery (`_coupling_corrected_tunes()`,
`deltaqx_resid_corr_beta0`, the `tune_shift_coupling_corrected` figure) has
no counterpart here -- confirmed by the module docstring inside
`compute_case()` itself, which spells this out explicitly. Only the
uncorrected `deltaqx_formula_list`/`deltaqx_formula_pert_list` (unperturbed
vs. perturbed beta) exist; **there is no perturbed-beta reference beta
either in the sense 007a has it** -- `betx_pert_at_s`/`bety_pert_at_s` here
come from the local (per-placement, per-offset) `betx_scan_pp`/
`bety_scan_pp`, i.e. the standalone undulator's own perturbed beta seeded
from the bare-ring initial condition, not a ring-wide perturbed beta.

**4. Per-placement array shape, not per-scan flat arrays.** Because there
can be 1 or 2 independent placements sharing nothing but the offset scan
loop, most scan-quantity arrays carry an extra leading placement axis:
`orbit_scan_x_pp`/`betx_scan_pp`/etc. all come out shape `(n_placements,
n_offsets, n_pts)` (the `_pp` suffix = "per placement"), and `orbit_scan_s_pp`
is `(n_placements, n_pts)` -- since each placement's undulator gets its own
local `s` grid (in local coordinates, offset by `ic['s0']` for plotting
against the bare-ring `s` axis). Contrast with 007a's flat `orbit_scan_x`
etc. (`(n_offsets, n_pts)`, one shared ring-wide `s` grid regardless of how
many undulators are inserted). `data['n_placements']` is saved alongside so
`plot_case()` knows how many placement-loops to draw without recomputing it
from `undulator_s_ranges.shape`.

**5. Separate cache dir and output subfolder** -- `DATA_DIR =
examples/splineboris/data_007b/` (not 007a's `data/`; the file's own
comment explains why: "compute_case()'s return schema here is entirely
different... sharing 007a's DATA_DIR would either collide on filenames or
leave stale/incompatible keys behind"), `OUT_DIR = .../SLS_Undulator_Studies/
local_optics` (same hardcoded-per-`simonfan` base as 007a -- see 007a's
note's "Hardcoded output path" section, equally broken on this machine), and
saved filenames get an extra `_local_optics` suffix
(`{place_label}_{model_label}_{suffix}_local_optics.pdf`) so they can't
collide with 007a's output even if `OUT_DIR` were ever unified.

**6. One fewer figure per case.** 007a's 9 possible figures minus
`tune_shift_coupling_corrected` (no coupling correction to plot, per point 3
above) = 8 possible, same MK-model exclusions as 007a
(`fig_orbit`/`fig_orbit_scan`/`fig_tune_shift_diff` dropped for MK,
`fig_beta_scan` dropped for SB) and same SB/MK field-row count
(`n_field_rows = 3 if model_label == 'SB' else 2`, `Bs` row only for SB).
Every field/orbit/beta plot now overlays all `n_placements` traces on one
axes (looped `for i in range(n_placements)`) rather than the single-undulator
assumption 007a's flat-array plotting code had -- legend labels are only
attached to `i == 0`'s traces to avoid a legend entry per placement.

## Data on disk

`data_007b/` already has cached `.npz` for all 6 cases (`ars11_uind_0210_1`,
`ars11_uind_0610_1`, `both` x `SB`, `MK`) as of this note, committed in the
same `f4e25b0cc` commit that added the script -- so `--replot` works
out-of-the-box here without needing a fresh full run first, unlike 007a
whose `data/` dir doesn't currently exist on this machine at all (see
[sls_undulator_tune_displacement_007a.md](sls_undulator_tune_displacement_007a.md)'s
"Not done / open"). As with 007a, the cache key is only `{place_label}_
{model_label}` with no parameter fingerprint, so changing scan parameters
(`n_tunes`, `hor_off_list` bounds, `multipole_order`, ...) will silently
serve stale data unless `data_007b/` is deleted by hand first.

## Not done / open

- Output path (`OUT_DIR`) needs the same portable `PLOT_DIR`/env-var fix
  007a's note flags before this script can save PDFs on this machine.
- Same `--replot` staleness caveat as 007a (see above).
- No cross-check yet between 007a's ring-inserted `ΔQ` and 007b's
  phase-advance `ΔQ` for the same case (e.g. how much of 007a's residual
  after coupling-correction is explained by 007b's coupling-free number) --
  this comparison seems to be exactly what motivated 007b's existence but
  wasn't seen being made explicitly in either script as of this note.
