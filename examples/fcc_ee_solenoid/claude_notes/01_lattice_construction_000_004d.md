# 000a-004d: building and installing the detector solenoids

Written 2026-07-10. Companion to `00_overview.md`. Covers the "legacy" local/
non-local install scripts (000a/000b/001a/001b) and the "current" SplineBoris/
VariableSolenoid pipeline (004a-004d), plus the two orphaned analysis scripts
002/003.

## Legacy path: 000a/000b (local) and 001a/001b (non-local)

Both pairs follow the same 2-step pattern: script `a` builds+installs raw
solenoid slices (no correction), script `b` loads that output and adds
orbit/optics correction, tilted doublets, and consolidated knobs.

### 000a_local_install_solenoids_and_correctors.py
- Input `fccee_z_lcc.json` -> output `temp_fcc_ee_lcc_local_solenoid.json`.
- Builds the main solenoid field directly from 3 `SolenoidField` instances
  computed *analytically in the tilted solenoid's own frame* and rotated into
  the beam frame by hand (sin/cos of `theta`), rather than using
  `TiltedSolenoid` (that helper class didn't exist yet at this point in
  history — 001a is the first script to use it).
  - Main: `SolenoidField(L=1.23*2, a=0.13, B0=2., z0=0)`.
  - Two extra "field_comp_sol_right/left" compensation contributions
    *co-located inside the same main-solenoid slice region*
    (`SolenoidField(L=0.8, a=0.13, B0=1, z0=+-1.63)`) — this is the "local"
    part of the name: compensation is local to the main solenoid, not a
    separate solenoid 12 m away.
  - Slices into 200 `xt.VariableSolenoid` elements (`sol_slice_{ii}_{ip}`)
    over `s in linspace(-2.399, 2.399, 201)`, with `ks_profile` built as a
    *vars expression* combining on/off knobs (`on_sol_{ip}`,
    `on_comp_sol_{left,right}_{ip}`) and a `field_comp_sol_{left,right}_{ip}`
    scale knob, so the balance can be retuned later via knobs, not rebuilding.
  - Scales `field_comp_sol_{right,left}_{ip}` so integrated `ks*length` of
    compensation exactly cancels the main solenoid's, split 50/50 left/right.
  - Inserts one first-order horizontal/vertical corrector
    (`acbh1_sol_{left,right}_{ip}`) distributed as a `knl[0]`/`ksl[0]` term
    spread proportionally over the slices between `s_ip+1.23` and
    `s_ip+2.29` (i.e. embedded inside the outer part of the main solenoid
    itself, not a separate element) — this pattern (`ds_start=1.23`,
    `ds_end=2.29`) recurs unchanged through 001a/004b/004c.
  - Adds `dy_match_{l,r}_{ip}` markers at `+-11.95 m` and
    `corr_sol_{left,right}_{ip}` (thin `xt.Multipole`, length 1) just outside
    them — anchor points for the orbit-correction knobs added in 000b.

### 000b_local_correction.py
- Input `temp_fcc_ee_lcc_local_solenoid.json` -> output
  `fccee_z_lcc_local_solenoid.json`.
- Per-IP `config` dict hardcodes which doublet quads
  (`qd0a{l,r}.N`, `qf1a-d{l,r}.N` etc, `N` = 0..3 one per IP) participate in
  optics correction / doublet tilt / orbit correctors — this exact `config`
  dict (IP names -> quad name lists) is copied verbatim into 001b and again
  (slightly reformatted) into 004c. If you need to change which quads
  correct which IP, all three copies must be kept in sync (there is no shared
  source of truth for this config across scripts).
- For each IP in turn (solenoids OFF at all other IPs while correcting this
  one): tilt doublet quads by `+-ksol_l_main_solenoid/4` each
  (`rot_s_rad`, driven by `on_rot_doublet_{left,right}_{ip}` knobs), then
  build two `line.match_knob(...)`:
  - `on_sol_orbit_corr_{ip}`: 24 corrector vars (`acbh1..6`/`acbv1..6` left
    and right), targets `x=px=y=py=dy=dpy=0` at the `dy_match_l/r` markers,
    `init_at=ip` with `betx/bety` pinned to the solenoid-off Twiss.
  - `on_sol_optics_corr_{ip}`: normal-quad `k1` knobs on
    `quad_for_optics_correction`, targets restoring `betx/bety/alfx/alfy/dx/dpx`
    at the straight-section boundaries (`end_ds_start_straight_{ip}` /
    `end_straight_start_ds_{ip}`) to the solenoid-off values.
  - Solved twice each, interleaved, then `generate_knob()`'d (freezes a linear
    knob-response instead of re-solving every time). No coupling-correction
    knob at this stage (that's new in 004c).
  - All resulting sub-knobs get chained to one `on_sol_corr_{ip}` knob per IP.
- Final sanity twiss with everything on (`tw_on_corr`) and a `two_on_corr`
  reusing `tw_off` as the `init` to check phase-advance consistency; 3 debug
  plots (`betx2/bety1`, orbit, `muy` error) shown but not saved.

### 001a_non_local_install_solenoids_and_correctors.py
- Input `fccee_z_lcc.json` -> output
  `temp_fcc_ee_lcc_non_local_solenoid.json`.
- First script to use `TiltedSolenoid` (from `tilted_solenoid.py`) for the
  main field, instead of manual rotation — same physical solenoid,
  cleaner code.
- Compensation solenoids are now **physically separate elements 12 m from the
  IP** (`SolenoidField(L=1.5, a=0.03, B0=1., z0=0)`, sliced over
  `linspace(-1, 1, 51)`), placed via
  `env.place(line_comp_solenoid_left, anchor='end', at=-12, from_=ip_name)`
  and the mirrored one at `at=+12` — this is the "non-local" naming, and it's
  the geometry that 004a/004b/004c continue (`COMP_SOLENOID_DISTANCE_FROM_IP
  = 12.0`), i.e. **004a-004d is a refined continuation of the 001a/001b
  approach, not of 000a/000b.**
- Same corrector-embedding-in-main-solenoid and `dy_match`/`corr_sol` marker
  pattern as 000a.

### 001b_non_local_correction.py
- Same structure as 000b (same `config` dict, same two `match_knob` calls,
  same doublet-tilt trick), operating on `on_comp_sol_{ip}` (single knob, not
  left/right separately, since compensation is now one coherent element pair)
  instead of `on_comp_sol_{left,right}_{ip}`.
- Output `fccee_z_lcc_non_local_solenoid.json`.

## 002_analysis_and_plots.py / 003_a_look_at_non_linear_components.py

- **002**: loads `fccee_z_lcc_non_local_solenoid.json` (legacy path — not the
  current corrected lattice!), slices IP regions finely for plotting, computes
  `tw6d`/`tw4d` with `radiation_integrals=True, polarization_analysis=True`,
  plus a `mean`-radiation twiss for energy-loss-per-length. Produces two
  multi-panel figures (optics+Bs+y+Dy+coupling; optics+Bs+dE/ds+spin) around
  `ipg`. Purely diagnostic, not part of any downstream dependency.
- **003**: standalone sanity check, *not* tied to any saved lattice — builds a
  bare `TiltedSolenoid`, computes transverse-field derivatives via finite
  differences (`compute_pure_field_derivatives`/`compute_pure_by_derivatives`)
  and asserts the central-difference `k1s` estimate matches the order-1 "pure"
  derivative extraction to `atol=1e-8`. Useful as a template for validating
  `SolenoidField`/`TiltedSolenoid` derivative machinery if that ever changes.

## Current path: 004a → 004b (×2) → 004c → 004d

### 004a_build_and_check_solenoids.py
The template-building script; everything downstream reads its output
`004_solenoid_lines.json`. Delegates the actual math to `spline_boris_setup.py`
(see `00_overview.md`); this script is mostly config + orchestration + plots.

- Two physical field models: `main_field_model = TiltedSolenoid(L=2.46,
  a=0.13, B0=2.0, theta=-0.015)`, `comp_field_model = SolenoidField(L=1.5,
  a=0.03, B0=1.0, z0=0)`.
- `extract_tapered_field_data(...)` (from `spline_boris_setup.py`) is the
  heavy lifting: samples `bx`/`by`/`bs` and their transverse derivatives up to
  `MAX_TRANSVERSE_DERIVATIVE_ORDER=4` (fixed, always 4, regardless of the
  `--max-transverse-order` flag below) on `MAIN_SOLENOID_S_AXIS =
  linspace(-2.399, 2.399, 201)` (main) / `COMP_SOLENOID_S_AXIS =
  linspace(-1, 1, 201)` (comp), applies a quintic smoothstep taper
  (`TAPER_LENGTH=0.15 m`) so fields go exactly to 0 (value+slope+curvature) at
  the array ends, then fits `LSQUnivariateSpline`s in `s` per slice boundary
  to get consistent start/end values+derivatives+slice-means for every
  transverse-derivative order — this is what eventually becomes each
  `xt.Spline4`'s `val_start/der_start/val_end/der_end/mean`.
- **`--max-transverse-order` (added 2026-08-03, default 4)**: caps
  `MAX_TRANSVERSE_DERIVATIVE_ORDER_FOR_SPLINE`, the *separate* constant
  actually passed to `build_splineboris_line` further down (see next bullet)
  — controls how many `bx`/`by` orders get baked into each installed
  `xt.SplineBoris` (i.e. `element.multipole_order`), not how many are
  extracted from the field map. Lower = fewer polynomial terms evaluated per
  Boris step = cheaper tracking, at the cost of losing higher multipole
  content (order 2 = sextupole). Tagged via `order_tag()` into
  `OUTPUT_LINES_JSON` (silent/untagged at the default 4) and threaded through
  004b/004c/004d/009/010/014/015/013 the same way `--b0`/`field_tag` is —
  see `00_overview.md`'s `solenoid_params.py` entry for the full chain.
- `comp_scale_b = -main_bs_integral / comp_bs_integral_unscaled / 2.0`: the
  key charge-balancing computation, done **here** (not in 004b/004c).
- Builds 4 templates via `build_splineboris_line`/`build_variable_solenoid_line`:
  `main_solenoid`, `compensation_solenoid` (SplineBoris, with
  `sextupole_amplification_factor` support built in but applied at 1.0 here —
  actual runtime scaling happens later via the `sext_amp` knob in 004b, see
  below), `main_solenoid_varsol`, `compensation_solenoid_varsol`
  (VariableSolenoid, linear-only, no sextupole knob).
- Saves all 4 (`.to_dict()`) plus a `metadata` block (build settings +
  `comp_scale_b` + integrals) to `004_solenoid_lines.json`.
- Remainder of the script (majority of its ~790 lines) is **checks/plots
  only**, not used downstream: builds a local 3-element system (comp_left +
  main + comp_right, `assemble_three_solenoid_system`) for both SplineBoris
  and VariableSolenoid, twiss-checks orbit/coupling starting at the IP with
  `betx=0.09, bety=0.0007`, computes `symplectic_error` for main/comp
  SplineBoris lines individually, and produces many field-map-vs-SplineBoris
  comparison plots (raw fields, s-derivatives up to order 5, transverse
  derivatives up to order 4, one "mixed derivative" spec
  `d^2 Bx/dx^2 ds^2`). If you need to re-validate the SplineBoris fit quality
  after changing `extract_tapered_field_data`, this is the script to rerun.

### 004b_install_solenoids_in_fcc_ring.py (SplineBoris) and
### 004b_install_varsol_solenoids_in_fcc_ring.py (VariableSolenoid)
Both: input `fccee_z_lcc.json` + `004_solenoid_lines.json` -> output
`temp_fcc_ee_lcc_{splineboris,varsol}_solenoids.json`. No correction yet
(orbit/optics knobs generated by 004c). Structurally near-identical to
001a but **cloning pre-built templates** instead of rebuilding fields inline:

- Loads templates, for each IP clones `main_solenoid` /
  `compensation_solenoid[_varsol]` element-by-element
  (`template_element.copy()`) into fresh `env.elements[...]` entries named
  `sol_slice_{ip}_*` / `comp_sol_slice_{left,right}_{ip}_*`, so each IP gets
  its own independent copy (needed since knobs like `on_sol_{ip}` must gate
  each IP separately).
  - SplineBoris variant: knob applied via `scale_b = template_scale_b *
    on_sol_or_comp_ref` (SplineBoris exposes a `scale_b` multiplier
    directly). **Sextupole term hookup**: if
    `cloned_element.multipole_order >= 3`, `bx[2,k]`/`by[2,k]` (the
    order-2/sextupole Spline4 coefficients, `k=0..4`) get multiplied by the
    global `sext_amp` env var — this is the actual runtime location of the
    "sextupole amplification" feature; 004a's
    `sextupole_amplification_factor` param is a no-op at 1.0 baked into the
    template, and the *real* per-run knob is `env['sext_amp']` set here (and
    later toggled by `lattice_knobs.set_lattice_knobs`).
  - VariableSolenoid variant: knob applied via
    `_apply_solenoid_knob_to_element` multiplying `ks_profile[0:2]` and
    `knl[0]`/`ksl[0]` by the knob ref directly (no `scale_b` attribute on
    this element type). No sextupole knob (VariableSolenoid is linear-only).
- Same insertion geometry as 001a (`anchor='center'` for main at `s_ip`,
  comp left/right at `+-12 m`), same embedded first-order correctors
  (`acbh1/acbv1_sol_{left,right}_{ip}` distributed over `s_ip+-[1.23,2.29]`),
  same `dy_match_{l,r}` / `corr_sol_{left,right}` markers.
- SplineBoris variant only: after installing all 4 IPs, does a standalone
  diagnostic — turns all solenoids/comp on, walks the line table for
  `element_type == 'SplineBoris'`, sums `bs_mean*length` split into
  main-solenoid vs compensation contributions, and plots the ring's `Bs(s)`
  profile as stitched continuous regions (returning to 0 between disjoint
  solenoid regions). Confirms integrated `Bs` cancels as designed.

### 004c_correct_solenoids_in_fcc_ring.py
`--model splineboris|varsol` (default splineboris) selects
`temp_fcc_ee_lcc_{model}_solenoids.json` as input and writes
`fccee_z_lcc_{model}_solenoids_coupling_corrected.json` — **this output is
the final lattice used by 009/010/011/013/014.**
- Same hardcoded per-IP `config` (`quad_for_optics_correction`,
  `doublet_quad_{left,right}`, `corr_{1..4}_{left,right}_on_quad`) as
  000b/001b, copied a third time.
- `measure_ksol_l_main_solenoid`: model-aware integral (handles both
  `VariableSolenoid.ks_profile.mean()*length` and SplineBoris's
  `scale_b*bs[4]*length/rigidity0` — note index `[4]` on the Spline4 picks
  the slice mean, consistent with `Spline4(mean=...)` argument order used
  when building it in `spline_boris_setup.py`).
- Adds a **third correction stage beyond 000b/001b's orbit+optics**:
  `on_sol_coupling_corr_{ip}` — skew-quad (`k1s`) knobs on every quadrupole
  from the straight-section start to the IP and from the IP to the
  straight-section end (`k1s_{name}_sol_coupling_corr`), targeting
  `betx2=bety1=0`, `alfx2=alfy1=0`, `dy=dpy=0` at both straight-section
  boundaries. This is the extra linear-coupling/vertical-dispersion cleanup
  that the legacy 000b/001b path didn't have — likely why the "current" path
  is preferred for physics results despite being more code.
  Note: `opt_orbit`'s targets in 004c use `start/end =
  dy_match_l/r_{ip}` (identical to 000b/001b); the optics/coupling matches use
  `start/end = end_ds_start_straight_{ip}/end_straight_start_ds_{ip}` (also
  identical). Correction order per IP: orbit -> optics -> coupling -> orbit ->
  optics (iterate for consistency) -> generate all 3 knobs.
- Final consolidated `on_sol_corr_{ip}` knob now also drives
  `on_sol_coupling_corr_{ip}` in addition to the 000b/001b set.
- Same before/after `tw4d`(`zero_at='ipg'`) diagnostic plots as 000b/001b.

### 004d_analysis_and_plots.py
- Loads `fccee_z_lcc_splineboris_solenoids_coupling_corrected.json` **only**
  (no varsol equivalent script) and reproduces the same two multi-panel
  figures as 002 (optics/Bs/y/Dy/coupling and optics/Bs/dE-ds/spin), plus a
  radiation-mean twiss (`compensate_radiation_energy_loss`). This is the
  "current path" analog of 002 — if you want an up-to-date diagnostic plot of
  the corrected lattice, use 004d, not 002.
- Prints `tw4d`/`tw6d` tunes (`qx, qy, qs`) as a final sanity check.
