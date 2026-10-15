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
`004_solenoid_lines_{FIELD_TAG}{ORDER_TAG}.json` (e.g. `_2T`, `_3T`,
`_3T_o2`; this note wrote the pre-2026-07-24 untagged name
`004_solenoid_lines.json` below). Delegates the actual math to `spline_boris_setup.py`
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
- Remainder of the script (majority of its ~1240 lines as of 2026-10-14) is **checks/plots
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
Both: input `fccee_z_lcc.json` + `004_solenoid_lines_{FIELD_TAG}{ORDER_TAG}.json`
-> output `temp_fcc_ee_lcc_splineboris_solenoids_{FIELD_TAG}{ORDER_TAG}{OUT_TAG}.json`
/ `temp_fcc_ee_lcc_varsol_solenoids_{FIELD_TAG}.json`. `OUT_TAG` comes from
the SplineBoris variant's `--output-tag` (e.g. `mainscale`), which must be
passed to 004c too. No correction yet
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
- **Field-scale knobs (added 2026-08-27 to 09-01; their note
  `07_main_b_scale_scans.md` is missing from this checkout, see `00_overview.md`).** Main solenoid
  strength = `on_sol_{ip} * main_b_scale`; each compensation solenoid =
  `on_comp_sol_{ip} * comp_b_scale * comp_b_scale_{left,right}_{ip}`. All
  default to 1.0 and 004c leaves them there, so the correction is solved at
  design field. `comp_b_scale` exists in both variants; `main_b_scale` and the
  per-side `comp_b_scale_{side}_{ip}` exist only in the SplineBoris variant.
  `lattice_knobs.set_lattice_knobs` sets `comp_b_scale`/`main_b_scale` if
  present. Scanned by 004f (`comp_b_scale`), 004h/004j (`main_b_scale`, on
  the `--output-tag mainscale` lattices); 004h/004i also float the per-side
  knobs in the coupling re-solve.
- Same insertion geometry as 001a (`anchor='center'` for main at `s_ip`,
  comp left/right at `+-12 m`), same embedded first-order correctors
  (`acbh1/acbv1_sol_{left,right}_{ip}` distributed over
  `s_ip+-[ds_start, 2.29]`, with `ds_start = corrector_ds_start_for_b0(b0)`:
  1.23 m at 2 T, 1.40 m at 3 T),
  same `dy_match_{l,r}` / `corr_sol_{left,right}` markers.
- SplineBoris variant only: after installing all 4 IPs, does a standalone
  diagnostic — turns all solenoids/comp on, walks the line table for
  `element_type == 'SplineBoris'`, sums `bs_mean*length` split into
  main-solenoid vs compensation contributions, and plots the ring's `Bs(s)`
  profile as stitched continuous regions (returning to 0 between disjoint
  solenoid regions). Confirms integrated `Bs` cancels as designed.

### 004c_correct_solenoids_in_fcc_ring.py

*(Section rewritten 2026-10-14. The correction was restructured on
2026-09-15/16, and the original 2026-07-10 description of a single
full-straight match no longer applies.)*

Input `temp_fcc_ee_lcc_{model}_solenoids_{FIELD_TAG}{ORDER_TAG}{OUT_TAG}.json`,
output `fccee_z_lcc_{model}_solenoids_coupling_corrected_{...}.json` (varsol
has no `ORDER_TAG`). **This output is the lattice used by 004d, 009/010/011/
013/014/015/018.** CLI: `--model splineboris|varsol`, `--b0`,
`--max-transverse-order`, `--output-tag` (must match 004b's, e.g.
`mainscale`), and tuning flags `--optics-step`, `--optics-rcond`, `--ips`
(a subset of IPs; a partial run does **not** write the output lattice),
`--no-chromaticity`, `--chromaticity-points` (default 21).

- **Same hardcoded per-IP `config` dict** as 000b/001b (no shared source of
  truth), now with two more fields per IP: `bend_for_mid_quad_correction`
  (three bends upstream and three downstream, e.g. `b0cl.3 b0bl.3 b0al.3` /
  `b1ra.0 b1rb.0 b1rc.0` at ipa) and `sext_left`/`sext_right`, the sdm1
  chromatic sextupole nearest the IP (`sdm1l.7`/`sdm1r.0` at ipa).
- **Mid-bend trim quads (one-time, before any optics).** Each of those six
  bends per IP is cut in half (`cut_at_s` on the sorted `s_center`s; the
  bends must arrive as unsliced `RBend`s) and a zero-length
  `xt.Multipole` named `qbmid_<bend>` goes at the cut, 24 in the ring. The
  strength goes on `knl[1]` (k1 is dead at zero length). They are
  Multipoles, not Quadrupoles, so `unique_quadrupoles()` and the 004f-004j
  host scans do not pick them up as skew hosts (those scripts also skip
  the `qbmid_` prefix explicitly).
- **Bare reference:** one `tw0 = line.twiss4d()` with all solenoids off;
  `init_ip[ip] = tw0.get_twiss_init(ip)`. Both halves of every IP start
  from the bare optics at the IP.
- **Doublet tilt** by `+-ksol_l_main_solenoid/4` exactly as before
  (`measure_ksol_l_main_solenoid` is model-aware: `ks_profile.mean()*length`
  for VarSol, `scale_b*bs[4]*length/rigidity0` for SplineBoris, `[4]` = the
  Spline4 slice mean).
- **Per IP, per side (`SIDES = ('left', 'right')`)**, three matches, all
  `init=init_ip[ip]`. The right half runs IP → boundary with targets at
  `xt.END`; the left half runs boundary → IP (backward twiss) with targets
  at `xt.START`:
  - **orbit** `on_sol_orbit_corr_{side}_{ip}`: 12 correctors
    (`acbh/acbv1..6_sol_{side}_{ip}`), `x=px=y=py=dy=dpy=0` at
    `dy_match_{l,r}_{ip}`.
  - **optics** `on_sol_optics_corr_{side}_{ip}`: 15 knobs, the 12 normal
    quads of that side (`k1_<quad>_sol_corr`, k1) plus 3 mid-bend trims
    (`k1_qbmid_<bend>_sol_corr`, k1l). Targets: betx/bety (tol 1e-5),
    alfx/alfy (`OPTICS_ALF_TOL=1e-7`), dx/dpx (1e-8) at the straight-section
    boundary, plus betx/bety at `sext_{side}` with `tag='sext'` (a `dy` target
    there is commented out). 9 targets in all.
    **The QD0 → sdy1 phase advance is deliberately *not* targeted** (it was
    tried, see notes 08/09). It is only measured, via
    `doublet_to_sdy1_range()`.
  - **coupling** `on_sol_coupling_corr_{side}_{ip}`: `k1s` on every
    Quadrupole of that half (~40 knobs). Targets at the boundary, from
    `coupling_edge_targets()`: since 2026-10-15 the four normalised y/py
    components of mode 1 in the W matrix = 0 (`COUPLING_AMP_TOL = 1e-5`) plus
    `dy=0` (1e-7) and `dpy=0` (1e-9). Before that (`--coupling-targets beta`,
    legacy): `betx2=bety1=0`, `alfx2=alfy1=0`, `dy=0` (5e-5), `dpy=0` (1e-7),
    which leaked coupling into the arcs -- see note 11.
- **Solve order per side:** orbit 1/3 → optics 1/3 (staged) → coupling 1/2 →
  orbit 2/3 → optics 2/3 → coupling 2/2 → orbit 3/3 → optics 3/3.
  `solve_and_report()` wraps each solve with a banner (knob, range, pass).
  Solver settings, all explained in long comments and in note 10:
  `OPTICS_RCOND = 1e-6`, `OPTICS_STEP = 1e-7`, `OPTICS_BROYDEN = False`,
  `OPTICS_STAGE_SEXT = True` (pass 1 first solves with `sext` disabled, then
  enabled), knob limits off (`OPTICS_K1_LIMIT_* = None`; they made things
  worse), coupling `rcond=3e-3`.
- **Summary before `generate_knob()`**: per knob, end penalty, `max|knob|`
  (a converged half-straight needs |k1| ~ 3e-4..3e-3; far more means it
  wandered off in a weak direction) and targets in tolerance. This must run
  before `generate_knob()`, because `target_status()` afterwards corrupts
  the generated state.
- **Knob tree:** `on_sol_corr_{ip}` drives `on_comp_sol_{ip}`,
  `on_rot_doublet_{left,right}_{ip}` and `on_sol_{orbit,optics,coupling}_corr_{ip}`.
  Each of those drives its two `..._corr_{side}_{ip}` children, which are
  the generated knobs. The per-IP names downstream scripts use are
  unchanged. Children are wired as string *expressions*, so overriding one
  with a number detaches it; assign the string back to re-attach (004cc's
  `set_correction_state`, 004d's `_twiss_without_correctors`).
- **End-of-run reports** on the full ring: betx/bety at each targeted sdm1
  bare → corrected, IP orbit, **QD0 ↔ sdy1 mux/muy bare → corrected** (an
  error of 1e-6 in muy is worth ~25 in Q''y), and Q'/Q'' before/after via
  `report_nonlinear_chromaticity()`, which uses
  `examples/nonlinear_tunes/detuning.py`. Note that its `d2qx`/`d2qy` are
  **Q''/4**. The docstring of `report_nonlinear_chromaticity` wrongly says
  Q''/2; see note 10.
- Figures: betx2/bety1 and x/y over ±20 m, `tw_off` + muy error, and
  figure 4: Q(δ) with fit, plus its quadratic part only, bare vs corrected.

Related: `004cc_correct_solenoids_in_fcc_ring.py` is a one-IP, right-half
sandbox of the same correction (notes 09/10), and `004l_sextupole_target_ab.py`
A/Bs the optics match with and without the `sext` targets (see
`00_overview.md`).

### 004d_analysis_and_plots.py

*(Section rewritten 2026-10-14. The script was largely rewritten on 2026-09-22
to 09-24 for the presentation, so the 2026-07-10 description of two
002-style figures and the 2026-09-18 description of three beta figures no
longer apply.)*

SplineBoris only. It loads
`fccee_z_lcc_splineboris_solenoids_coupling_corrected_{FIELD_TAG}{ORDER_TAG}.json`
for `--b0`/`--max-transverse-order`, **and also always loads both the 2T and
3T lattices** (`COMPARISON_TAGS`) for the comparison figures, so both must
exist. It works on a shallow copy of the line, cut every 0.2 m over 2.4-11 m
either side of each IP for smooth plots. `FIGURE_SCALE = 2.0` doubles every
figure, enlarging fonts and ticks but deliberately not legends or line
widths.

Twisses: `tw_off` (6d, solenoids and corrections off), `tw4d` (on, with
`polarization_analysis`, `radiation_integrals`, `coupling_edw_teng=True`),
`tw` (6d, on, polarization), `two` (open twiss from the bare start),
`tw_nocorr` (on, but with the straight's orbit/optics/coupling corrector
knobs at zero, keeping the compensation solenoids and doublet rotation on;
see `_twiss_without_correctors`), and `tw_rad` (mean radiation +
`compensate_radiation_energy_loss`, for dE/ds).

Figures, in order:
- fig1/fig2: the original 002-style multi-panel views (optics/Bs/y/Dy/
  coupling, optics/Bs/dE-ds/spin).
- **2T vs 3T betx2/bety1** (`_betx2_bety1_fig`) over the IR (±20 m, with
  solenoid/corrector region markers from `_compute_ir_markers`) and the full
  straight (±1400 m) (fig3/fig4). Each panel has a `_coupling_summary_text`
  box with the closest-tune-approach `c_minus`. Ring-averaged RDTs used to be
  in that box and are now only printed at the end.
- **The same with correctors off** (`_nocorr_fig`): fig4a/4b beta, fig4c/4d
  coupling. These are separate figures because the coupling is ~30x larger
  there.
- **Coupling RDTs** |f1001|, |f1010| from `tw4d` (`_coupling_rdt_fig`,
  fig5/fig6).
  The long block comment above `RDT_CASES` is worth reading. There are 1208
  NaN points in eight blocks spanning the **compensation-solenoid bodies**,
  traced to W being in kinetic momenta inside solenoids, which is not
  symplectic. So Edwards-Teng (and the betx/bety/betx2/bety1 read off the
  same W) are not canonical inside solenoid bodies.
- **beta_x/beta_y with and without solenoids** (`_beta_comparison_fig`),
  `BETA_COMPARISON_RANGES` = IR ±20 m and full straight ±1400 m, **both
  linear**. The beta-beat panel was removed; `max |dbetx/betx|` and
  `max |dbety/bety|` over the straight are printed at the end instead,
  together with ring-wide f1001/f1010 statistics.

Saved PDFs, to `PLOT_DIR / 'Coupling_Studies'`: 8 combined figures
(`betas_{IR,straight}[_coroff].pdf`, `coupledbetas_{IR,straight}[_coroff].pdf`),
and 14 single-field-case versions in `Coupling_Studies/plots/`
(`*_2T.pdf`/`*_3T.pdf`).

- **Linear vs log beta axes (2026-09-18, still relevant).** `tw_off.plot()`
  in fig1 is linear, and its default colours are `betx` black, `bety` red and
  `dx`/`dy` blue/green on a twin right axis, so the flat green line is `dy`,
  not a beta. On linear the IP waist collapses onto zero; on log it opens
  into a deep V that dominates the eye while the `bety` double hump flattens.
  Same numbers either way, checked on the 3T lattice at `ipa`:

  ```
         s     betx_off     bety_off      betx_on      bety_on
   -14.000         2844         5217         2845         5228
    -4.500        386.9    1.517e+04        386.4    1.516e+04
     0.000         0.09       0.0007      0.08999       0.0007
     4.500        386.9    1.517e+04          387    1.516e+04
  ```

- **The boxes at the compensation solenoids are real, not a plotting
  artifact.** Between roughly ±12.5 and ±13.6 m, `betx` jumps 2895 -> 7631
  with `bety` dropping to match, with sharp edges at the slice boundaries
  (`comp_sol_slice_{left,right}_ipa_*`). This note originally (2026-09-18)
  read it as the Edwards-Teng mode beta partially swapping mode identity
  inside the coupled region, so that `betx` there is beta_x1 and not the
  physical envelope. The later RDT analysis above found that W itself is in
  kinetic momenta and not symplectic exactly in these slices. **Which of the
  two explains the box has not been checked.** Either way, do not read betx
  there as a canonical beta. It is also what the final `max |dbetx/betx|`
  print picks up: ~164% over ipa's straight is this box, not a beta-beat the
  correction failed to fix.
