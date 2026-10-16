# To do: FCC-ee IR solenoid compensation paper

Canonical copy: `examples/fcc_ee_solenoid/paper/TODO.md` on branch `fcc_nonlocal_paper`. The claude.ai project syncs from here. Tick an item only once its result is logged in `RESULTS.md` (see `../CLAUDE.md`).

**Labels:** `SIM` simulation to run · `DERIVE` equations to work out · `READ` literature/research · `ASK` information from others · `CHECK` verification · `FIG` figure/table · `WRITE` text

## Critical path (suggested order)

- [x] **Field map was public:** both copies of `Field_map_2T_BNL_August_14_2026.csv` were purged from every branch's history on 2026-10-14 → `.gitignore` rule added; 006b reads `$FCC_SOLENOID_FIELD_MAP`
- [ ] `ASK` Tell V. Teotia the map was publicly visible in the fork from 2026-08-28 until 2026-10-14, and ask whether it may be shared at all (e.g. alongside the paper)
- [x] Commit `claude_notes/` 08–10, which hold the Q″y diagnosis → commit bd4d5793
- [x] Re-run 004c → 004d at 2 T and 3 T with the current, frozen solver settings and log the baseline in `RESULTS.md` (all numbers there are `stale`) → `fccee_z_lcc_splineboris_solenoids_coupling_corrected_{2T,3T}.json` at e749d4795, logs in `data/logs_2026-10-15/` (RESULTS L12–L22; |C⁻|, εy and the IP/edge numbers stay `prelim` until the measurement scripts are committed)
- [x] Decisive test: is 3 T fine **without** the mid-bend trim quads, now that the solver is fixed? (§4a) → **No, and neither is 2 T**: true Q″y +24998 (3T) / +10411 (2T) without trims vs +4668 / −1248 with; left-half optics matches stop at 4/8 targets. `fccee_z_lcc_splineboris_solenoids_coupling_corrected_{2T,3T}_notrims.json`, RESULTS L23–L26, note 12
- [ ] Validate the integrator (Appendix) and work out the linearized-vs-analytic derivation in parallel (§3)
- [ ] Re-run the 2 T / 3 T analytic results with the frozen setup (§4)
- [ ] Build the field-map pipeline and produce the 2 T map results (§5)
- [ ] Write

## Repo audit (2026-10-14): what already exists and what it changes

- [x] `CHECK` Field map out of the repo: both copies purged from history, the map lives in `~/cernbox/Solenoid_Field_Maps/`, and 006b loads it through `$FCC_SOLENOID_FIELD_MAP` (2026-10-14)
- [ ] `ASK` Optional: ask GitHub Support to clear cached views of the old commits (needs the "first changed commit" from the purge output)
- [x] `WRITE` Track `claude_notes/` so the GitHub sync to the claude.ai project can include them → bd4d5793
- [x] `CHECK` Fix 004c's docstring, which says `d2q` is Q″/2; per note 10 the printed `d2qx`/`d2qy` is **Q″/4**. Make sure every Q″ in the paper is the true d²Q/dδ² → fixed 2026-10-15; RESULTS L19–L27 all quote true Q″
- [ ] `CHECK` **Reconcile the story** of the slides with notes 08–10 and the comments in 004c:
    - The sdm1 beta-beat (βy at sdm1r.0: 1.47 m bare → 2.87 m at 2 T → 310 m at 3 T) is **not** the Q″y driver.
    - The driver is the **QD0 → sdy1 vertical phase advance** (the LCC −I pair), which the half-straight match leaves free.
    - The irreproducibility came from a **rank-deficient optics match**: 15 knobs, 9 targets, condition number 3.7e7. xdeps' default `rcond = 1e-14` inverts the two near-null directions.
    - Fix: `OPTICS_RCOND = 1e-6` plus a staged pass 1, with no new targets.
    - This replaces the "beta-beat at sextupole → mid-bend quads" narrative of the September slides.
- [x] `SIM` Mid-bend trims off vs on (rcond fix active, 2 T and 3 T): compare Q″y, MA, max|k1|. If they aren't needed, the paper proposes **no new hardware** → RESULTS L23–L26, L28–L29
    - Q″y, max|k1| and convergence done 2026-10-15 (RESULTS L23–L26): the trims **are** needed at both fields.
    - DA and MA (mean radiation, 5000 turns) done 2026-10-16 (L28–L29): DA unaffected. Without trims the 3T MA keeps every |δ| ≤ 0.0095 only up to x̂ = 2.9σx (7.3σx with trims). At 2T the trims make no difference to the MA at this grid resolution. Quantum reruns are listed under §4
- [ ] `FIG` Singular values of the optics-match Jacobian (the gap between σ₇ ≈ 9.6 and σ₈ ≈ 4.9e-2), from `004g_coupling_svd_diagnostic.py` or 004c's diagnostics
- [ ] `FIG` Q″y vs QD0→sdy1 phase error: the calibration (dQ″y/dμ = 2.42e7 left, 2.48e7 right, per unit tune of phase error, true Q″) and the 8 measured points. Note 10 reports agreement to ~0.2%. This explains the irreproducibility in one plot
- [ ] `SIM` Show the sign flips of Q″y across repeated runs with `rcond = 1e-14`, and the stability with `1e-6`
- [x] `CHECK` Coupling leaked into the arcs (|C⁻| 3.9e-4 / 8.1e-4) because the coupling targets βx2/βy1/αx2/αy1 see only the position part, quadratically. 004c now targets the W-matrix mode-1 components + tighter Dy/Dpy → e749d4795, note 11
- [x] `CHECK` Commit the measurement scripts used for RESULTS L12–L18, L21–L22 (|C⁻|, Chao εy, closed-ring ET at the edges, IP/edge residuals) as a numbered script, so those rows can become `final` → `004n_figures_of_merit.py` (f071fb929); the rows are now `final`
- [x] `CHECK` 004cc still carries a value-shift bug in its sextupole targets (note 10): fix it or retire 004cc → already fixed: the `sext` targets read betx/bety/dy at sdm1 correctly (checked 2026-10-15, 004cc lines 319–321); the bug was never committed (note 10). Note 10's other caveat stands: 004cc's chain B has no rcond
- [ ] `WRITE` Scope decision: polarization (015/016/018 exist; note 05 says the tracked τ_depol has SNR ≈ 1). In the paper, or in the outlook?
    - Decided 2026-10-16: keep running it together with the emittance study (018 tracks both in one go, at no extra cost). Whether the paper mentions it is decided later
- [ ] `CHECK` εy definition: 014/018 use single-plane CS emittances, which are only approximate with coupling (00_overview, caveats). Switch to 4×4 eigen-emittances for the paper numbers

---

## 0. Framing and setup

- [ ] `WRITE` Choose the target journal (PRAB vs NIM A) and check its length, format and figure requirements
- [ ] `WRITE` Settle a title that reflects the field-map result
- [ ] `WRITE` State the research question: is non-local compensation feasible at 2 T and 3 T, and with a realistic 2 T field map?
- [ ] `WRITE` Write an explicit list of contributions (Xsuite implementation with exact fields; 3 T case; nonlinear dynamics; MA diagnosis and fix; field-map result)
- [ ] `WRITE` Remove all "local vs non-local comparison" framing; cite [1] for the local scheme as background only
- [ ] `WRITE` Define the figures of merit used throughout: $|C^-|$, coupled β ratios, εy, DA, MA, $\xi^{(1)}$ and $\xi^{(2)}$
    - [ ] State the chromaticity convention ($\xi^{(2)} = d^2Q/d\delta^2$ vs a Taylor coefficient). Watch out: 004c/004j print `d2qy` = Q″/4 (note 10); check Twiss `ddqy` separately
- [ ] `CHECK` Record the lattice version (commit), energy mode and beam parameters (E, Bρ, εx, tunes, crossing angle) used for every result
- [ ] `WRITE` Write the εy budget paragraph from the information you already have (the IR solenoid's allowed share, and how 1.3 pm compares)
- [ ] `READ` Find and state the DA requirement (injection) and its source
- [ ] `READ` Find and state the MA requirement (injection δ = 0.0095, beamstrahlung tails, lifetime) and its source
    - [ ] Also the injection amplitude (in σx) at which δ = 0.0095 has to survive: L29's verdict depends on it (3T without trims fails above 2.9σx)

## 1. Literature and prior art

- [ ] `READ` Boscolo, Ciarma & Burkhardt 2025 [1]: extract their 2 T numbers (layout, fringe fields, SR power, εy) as benchmarks
- [ ] `READ` FCC FSR Vol. 2 [2]: εy budget, IR layout and the IR sextupole scheme (local chromatic correction, crab sextupoles)
    - [ ] Identify the role of sdm1r.0 and which optics conditions (β, α, phase advance) it relies on
- [ ] `READ` Prior art on IR solenoid compensation and field-map-based modelling at SuperKEKB, CEPC, EIC and BEPCII, so the field-map novelty claim is stated precisely
- [ ] `READ` DAΦNE/KLOE non-local compensation: the original reference
- [ ] `READ` Generalized gradients / field-map fitting from surface data: Venturini & Dragt (NIM A, 1999); Mitchell & Dragt (PRSTAB, 2010)
- [ ] `READ` Boris algorithm properties: Qin et al., "Why is Boris algorithm so good?" (Phys. Plasmas, 2013), plus follow-up work on whether it is symplectic
- [ ] `READ` Xsuite SplineBoris source: which integration scheme, its order, s-based or t-based, and how radiation and quantum excitation are implemented
- [ ] `READ` Mais–Ripken coupled lattice functions: the original reference (e.g. Willeke & Ripken, 1989)
- [ ] `READ` A reference for the first-order $|C^-|$ integral *including the solenoid term*
- [ ] `READ` Boscolo's method for SR power from fringe fields (analytic, or a code such as BDSIM/Geant4)
- [ ] `READ` Polarization effects of detector solenoids in FCC-ee (for the outlook only)

## 2. Background: solenoid effects and non-local scheme (→ §2)

- [ ] `DERIVE` Rotation angle $\theta = \frac{1}{2B\rho}\int B_z ds$ and focusing $k \propto B_z^2$; reproduce 8.09 mrad (2 T) and 12.8 mrad (3 T)
- [ ] `CHECK` θ(3 T)/θ(2 T) = 1.58 rather than 1.5: document the geometry difference (length, radius, profile) between the 2 T and 3 T models
- [ ] `DERIVE` Rotated final-focus quadrupole: $k_{1n} \to k_{1n} \cos(-2\theta_R)$, $k_{1s} = k_{1n} \sin(-2\theta_R)$; define $\theta_R$ precisely
- [ ] `WRITE` Coupling formalism: coupled β ratios $\beta_{x2}/\beta_x$, $\beta_{y1}/\beta_y$ and the closest tune approach $|C^-|$
- [ ] `WRITE` εy formalism: the evolution law $\varepsilon_y(n)$ and the fit procedure
- [ ] `DERIVE` Decompose εy into its coupling and vertical-dispersion ($D_y$) contributions (radiation-integral form)

## 3. Magnetic field modelling (→ §3)

### Linearized vs analytic model

- [ ] `CHECK` Document exactly what the "linearized model" in your code contains (does it include the crossing-angle projection into the beam frame?)
    - Repo: the linearized model = **VarSol** (`xt.VariableSolenoid`: a linear `ks_profile` plus one dipole kick per slice); lattices exist via `004b_install_varsol_…` and `004c --model varsol`. So SB vs VarSol *is* the analytic-vs-linearized comparison
- [ ] `DERIVE` Off-axis expansion of an axisymmetric solenoid to order $r^5$:
      $B_r = -\frac{r}{2} B_z' + \frac{r^3}{16} B_z''' - \dots$ and $B_z(r,s) = B_z - \frac{r^2}{4} B_z'' + \frac{r^4}{64} B_z'''' - \dots$
- [ ] `DERIVE` Transform to the beam frame: offset $x_0(s) \approx \theta_c s$ (θc = 15 mrad) plus tilt; expand to linear order in (x, y) around the reference trajectory
- [ ] `DERIVE` Identify the feed-down terms the linearized model drops (∝ $x_0 B_z''$, $x_0^2 B_z'''$, …) and classify them as normal or skew quadrupole-like
- [ ] `DERIVE` Insert those terms into the first-order $|C^-|$ integral and estimate the extra coupling at 2 T and 3 T
- [ ] `SIM` Expansion-order scan in SplineBoris: $|C^-|$ and coupled β ratios vs truncation order, uncorrected and corrected
    - Repo: `--max-transverse-order N` is already threaded through 004a→004d and 013 (only `3T_o2` is built so far); build o0–o4 for 2 T and 3 T
- [ ] `SIM` Toggle individual feed-down terms on and off to confirm which ones drive the extra coupling
    - Repo: `004e_sextupole_window_study.py` (zeroing one order in a window) is a template for this
- [ ] `CHECK` Compare the derived estimate with the tracking result (aim for one figure that shows both)

### Field representation

- [ ] `DERIVE` Write out the relation between bpmeth's $a_m$, $b_m$, $b_z$ and the scalar-potential expansion; state the truncation order and the resulting order of the Maxwell residual
- [ ] `SIM` Field-space comparison along both beams' reference trajectories: linearized vs analytic (vs map), for $B_z$, $B_x$, $B_y$ and gradients
    - Repo: start from `004aa_splineboris_field_content.py`, 004a's checks and `006_compare_individual_solenoid_models.py`

### Field map (2 T)

- [ ] `ASK` V. Teotia: grid extent and resolution, coordinate system, symmetry assumptions, and what the map includes (detector solenoid, compensation/screening solenoids, iron yoke, FF quads?)
    - Repo (006b docstring): the map is **one side only** (z = 0–6 m from the IP), an 11 × 11 grid of ±10 mm (2 mm spacing) with 1 cm planes, in the beam frame (Bx ≈ 0.015 Bz on axis)
    - So the compensation solenoids (at 12 m) are not in the map, and the other side has to be mirrored. Still to ask: is the screening solenoid included, is the field mirror-symmetric, is there iron/saturation, and how large is the noise?
- [ ] `ASK` V. Teotia: is iron saturation modelled, and what are the estimated numerical noise and errors of the map?
- [ ] `ASK` Is a 3 T field map planned or obtainable?
- [ ] `CHECK` The map covers both beam trajectories at ±15 mrad over the full region of non-negligible field
- [ ] `SIM` Map → SplineBoris pipeline: compare derivative-based on-axis fitting with a surface (generalized-gradient) fit; choose one and justify it
    - Repo: `TubeFitter` + `LongitudinalFitter` (`xtrack/_temp/splineboris/`) already do the surface fit (`examples/splineboris/006b_fcc_ee_solenoid.py`, `010_solenoid_tubefitter_comparison.py`). What remains is to document the accuracy (`003c_accuracy_scan.py` as a template) and install the map in the ring through 004b/004c
- [ ] `SIM` Reconstruction residual $|B_{map} - B_{fit}|$ inside the good-field region (DA envelope around both trajectories)
- [ ] `SIM` Multipole decomposition of the map along s: the axisymmetric part vs non-axisymmetric terms (dipole, normal and skew quadrupole, …)
- [ ] `SIM` Fit the axisymmetric part of the map with the analytic model (effective a, L, B0) and compare
- [ ] `CHECK` Maxwell consistency (div B, curl B) of the map and of the reconstruction
- [ ] `CHECK` Left/right (IP-side) asymmetries in the map that break the symmetry of the correction

## Appendix: numerical validation of the integrator

- [ ] `CHECK` Confirm SplineBoris's scheme and order from the source (expect second order → error ∝ h²)
- [ ] `SIM` Convergence scans in step size, number of spline segments, polynomial order and expansion order
    - [ ] Observables: one-pass map elements, Qx, Qy, $|C^-|$, εy, $\xi^{(2)}$
    - [ ] Log-log plot with the operating point marked; show the numerical error is far below the effects reported
- [ ] `SIM` Symplecticity and volume preservation, no radiation: one-pass Jacobian M at several amplitudes (on axis, ~10σ, DA edge)
    - Repo: `symplectic_error()` in `spline_boris_setup.py` and `007_splineboris_tapered_solenoid_set_symplecticity.py`; extend these to several amplitudes and to the full ring
    - [ ] Report $\lVert M^T J M - J \rVert$ and $\det M - 1$; compare with a thick-element reference
- [ ] `SIM` Long-term tracking without radiation (≥ 10⁴–10⁵ turns): emittance and amplitude drift must be ≪ the physical damping rate (αy ≈ 7.7×10⁻⁴ per turn)
- [ ] `SIM` Benchmark: long, nearly uniform solenoid vs Xsuite's exact thick `Solenoid` map
- [ ] `SIM` Benchmark: SplineBoris linear transfer matrix vs integration of the linearized model
- [ ] `SIM` Benchmark: tracked energy loss through the IR vs the radiation integral of $B_\perp^2$ along the reference trajectory
- [ ] `SIM` Optional benchmark: 2 T linear optics vs the MAD-X results of [1]
- [ ] `CHECK` Quantum excitation in SplineBoris is consistent with the rest of the lattice (the εy tracking depends on it)

## 4a. Matching procedure (prerequisite for §4 and §5)

- [ ] `WRITE` Document the current matching: knobs (7 FF quads per side with skew components, 4 with dipole components, 5 (+3) optics quads, skew correctors), targets, solver, tolerances, knob limits
- [ ] `READ` Extract the IR sextupole optics conditions (β, α, phase advance to the FF and between pairs) from the lattice documentation
- [ ] `WRITE` Largely answered already by notes 08–10 (rank deficiency + `rcond`, see Repo audit). Turn that into the paper's methods paragraph; the items below are now optional supporting evidence
- [ ] `SIM` Multi-start study with the original targets: N random initial knob sets at 2 T and at 3 T
    - [ ] Record βy and phase advance at sdm1r.0, $\xi_y^{(2)}$, knob strengths and convergence; plot histograms
- [ ] `SIM` Decisive test: add the sextupole conditions as explicit targets and match with the **existing knobs only**, at 2 T and 3 T; record residuals and knobs at their limits
    - Repo: `004l_sextupole_target_ab.py` already A/Bs the sdm1 targets. Per note 09, the condition that matters is the QD0→sdy1 phase (plus the −I strength condition), and targeting it directly failed. Rephrase this test as "mid-bend trims off vs on, with the rcond fix" (Repo audit)
- [ ] `SIM` Repeat with the mid-bend quadrupoles; show convergence and determinism
- [ ] `SIM` Try alternatives to the mid-bend quads (existing quads further out, sextupole retuning) to show they are necessary
- [ ] `CHECK` Knob strengths vs hardware limits (FF skew components, mid-bend quad gradients), 2 T vs 3 T
- [ ] `DERIVE` Physical argument for why only 3 T needs extra knobs (perturbation ∝ B², 2.25× larger; knob range consumed)
- [ ] `SIM` *(Only if the trims-off test shows they're needed)* Feasibility of the mid-bend quads: effect on εx, dispersion, damping partition and SR; impact on the other energy modes
- [ ] `ASK` *(Only if needed)* Discuss the mid-bend quads with the lattice designers (space in the bends, acceptability)
- [ ] `CHECK` Freeze the matching as a scripted, deterministic procedure, identical for the analytic and map cases

## 4. Results: analytic 2 T and 3 T (→ §4)

Run each item for: solenoids off · 2 T · 3 T without mid-bend quads (to show the problem) · 3 T with mid-bend quads

- [ ] `SIM` Linear optics: β, Δβ/β and dispersion in the IR and the full straight; residual beta-beat around the ring
- [ ] `SIM` Orbit and vertical dispersion $D_y$ in the IR and around the ring
- [ ] `SIM` Coupling: $\beta_{x2}/\beta_x$ and $\beta_{y1}/\beta_y$ along the straight; $|C^-|$ corrected vs uncorrected
- [ ] `SIM` εy from Twiss and from tracking, with damping times (**the 2 T value is still missing**); decompose into coupling vs $D_y$ contributions
- [ ] `SIM` DA with radiation (5000 turns, the default; the particle count follows from the grid in 009/010), **including 3 T after the mid-bend fix**
    - Mean radiation done for SB-2T/3T with and without trims (L28–L29). Open:
    - [ ] Solenoids off (`sb_off`) DA and MA at 5000 turns: the existing baseline runs used 10⁴ turns
    - [ ] Quantum radiation (`--radiation quantum`) for all of the above, solenoids off included
    - Repo: `013_run_da_and_ma.py --b0 … [--max-transverse-order …]` runs DA + MA + emittance + polarization in one go (via 018). Pass `--seed` for reproducibility
- [ ] `SIM` MA vs requirement, for all four configurations
- [ ] `SIM` $\xi^{(1)}$ and $\xi^{(2)}$ in both planes, **including the baseline values for solenoids off and 2 T**
- [ ] `SIM` Tune footprint vs δ for all configurations
- [ ] `SIM` Optional: amplitude detuning and frequency-map analysis to identify the resonances limiting MA
- [ ] `SIM` If quantum excitation is stochastic, run several seeds for DA, MA and εy and report the spread
- [ ] `DERIVE` Skew-sextupole hypothesis: re-derive the RDT ∝ $\int \beta_y^{3/2} \partial_x^2 B_x ds$, document the cancellation, and keep it as a short "ruled out" paragraph
    - Repo: 004a's β-weighted d²Bx/dx² diagnostic and `004e_sextupole_window_study.py`
- [ ] `SIM` SR from the fringe fields: first reproduce Boscolo's 2 T power, then compute 3 T (power and critical energy)

## 5. Results: 2 T field map (→ §5)

- [ ] `SIM` Replace the analytic detector solenoid (and the compensation solenoids, if the map includes them) with the map
- [ ] `SIM` Orbit distortion from the non-axisymmetric dipole components; correct with the existing dipole correctors
- [ ] `SIM` Apply the frozen matching procedure; record convergence and compare knob strengths with analytic 2 T
- [ ] `SIM` The full figure-of-merit set from §4 (one new column in the summary table)
- [ ] `SIM` Attribution: axisymmetric part only → add non-axisymmetric multipoles one at a time → full map
- [ ] `SIM` If the correction fails or is only partial: identify the responsible component and the extra knobs needed
- [ ] `SIM` SR power with the map fields, compared with analytic 2 T
- [ ] `SIM` Sensitivity to map uncertainty: vary the fit truncation and add noise; check that the figures of merit are stable

## 6. Discussion and outlook (→ §6–7)

- [ ] `FIG` 2×2 comparison (field strength × field realism) with 3 T map as the missing corner
- [ ] `FIG` Candidate: $|C^-|$, εy and Q″ vs main-solenoid field scale for 2 T and 3 T, from `004j_main_b_scale_suite.py`. It shows the scheme's sensitivity and headroom
- [ ] `WRITE` Argument for the 3 T map: the realism penalty at 2 T vs the margins at 3 T (an argument, not a claim)
- [ ] `WRITE` Decide whether to include a scaled 2 T map as a rough 3 T indicator (caveats: saturation, different geometry)
- [ ] `WRITE` Limitations: no errors or misalignments, single energy mode, no polarization, no beam-beam, idealized compensation solenoids
- [ ] `WRITE` Outlook: 3 T field map, DA and MA with errors and misalignments, polarization

## 7. Figures and tables

- [ ] `FIG` IR layout schematic: both beams, detector/compensation/screening solenoids, FF quads, mid-bend quads, sdm1r.0
- [ ] `FIG` Field comparison (linearized / analytic / map) along the beam trajectory
- [ ] `FIG` $|C^-|$ vs expansion order with the analytic estimate overlaid
- [ ] `FIG` IR optics, corrected vs uncorrected (2 T, 3 T, map)
- [ ] `FIG` Coupled β ratios along the straight
- [ ] `FIG` εy evolution, tracking vs Twiss (compact, one panel per configuration or overlaid)
- [ ] `FIG` DA comparison (contours overlaid in one plot rather than separate panels)
- [ ] `FIG` MA comparison
- [ ] `FIG` Tune footprint vs δ, before and after the mid-bend quads
    - Styling done 2026-10-16 (004k, e867e7033). The "before" figure still uses the September lattice (`f89a90b2c`); consider the `_notrims` lattices instead, so before/after differ only in the trims and not also in the solver fix
    - [ ] Replot into `~/cernbox/Pictures/FCC_Solenoid_Paper/Tune_Diagrams/` (`004k --replot`)
- [ ] `FIG` β at sdm1r.0 before and after, plus the multi-start histogram
- [ ] `FIG` Appendix: convergence log-log plot and symplecticity error
- [ ] `FIG` Table 1: machine and solenoid parameters (E, Bρ, θc, a, L, B0, θ, compensation-solenoid parameters)
- [ ] `FIG` Table 2: figures of merit for every configuration (off / 2 T / 3 T before / 3 T after / 2 T map)
- [ ] `FIG` Clean the plot titles of internal labels ("ipa:", commit hashes, "frac_lost"); use consistent styles and units, and colour-blind-safe palettes

## 8. Housekeeping

- [ ] `CHECK` Replace "two orders of magnitude" with the exact factor (7.88×10⁻⁴ → 4.36×10⁻², ≈ ×55)
- [ ] `CHECK` Fix the typo $\beta_{y2}/\beta_y$ → $\beta_{y1}/\beta_y$ (slide 20)
- [ ] `CHECK` Tag the code and lattice versions behind every figure; keep the configs needed to reproduce them
    - Since 584a6859e new figures and `.npz` data carry their commit, dirty flag, host and command line (`pdfinfo … | grep Subject`). Older figures in the paper folder are documented in `~/cernbox/Pictures/FCC_Solenoid_Paper/README.md`; keep it up to date as figures are added
    - DA/MA/EMIT/POL/TUNE figures can be redrawn from data with `012_replot_aperture_from_data.py` (ac434e92c); figure styling lives in `study_plots.py`
- [ ] `CHECK` Push `fcc_nonlocal_paper` and pull it on xsuite-sietse before the next remote runs (provenance stamping and replot changes of 2026-10-16)
- [ ] `WRITE` Code and data availability statement
- [ ] `WRITE` Acknowledgements (T. Pieloni, M. Seidel, V. Teotia for the field map; CHART funding)
- [ ] `ASK` Check the FCC / CERN internal publication approval procedure and timeline
- [ ] `ASK` Co-author review round with G. Iadarola before submission
