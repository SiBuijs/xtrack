# boris_spatial claude_notes — overview

Session-handoff notes for `examples/boris_spatial/`, on branch `fcc_da_ma`,
written 2026-07-20. This directory studies `xtrack.BorisSpatialIntegrator`
(`xtrack/boris.py`) — a pure-Python spatial (not time-domain) Boris pusher
for magnetic-only fields, used to validate xtrack's built-in solenoid
tracking against direct field-map integration. Read this file first; add
more numbered files here as the study grows.

**Treat as a snapshot, not ground truth**: re-check the source if a
referenced file/function no longer exists, and check `git log`/`git status`
in this directory for what's changed since 2026-07-20.

## Script pipeline

- `000_conventional_boris.py`, `001_spatial_boris.py`,
  `001a_spacial_boris_solenoid_lin_fringe.py`, `002_check_zeta_against_drift.py`:
  earlier exploratory/validation scripts (not touched this session).
- `linear_fringe_solenoid.py` — `LinearFringeSolenoid(B0, s1, s2, s3, s4)`:
  a hand-built, piecewise-linear-taper on-axis `Bz(s)` field with
  `Bx = -x/2 dBz/ds`, `By = -y/2 dBz/ds`. Simple but "unphysical" (not a
  real coil's field) — used in `004_study_convergence.py`.
- `004_study_convergence.py` — sweeps `n_steps` for `BorisSpatialIntegrator`
  through `LinearFringeSolenoid`, checks exit-position error and the
  integrator's own symplectic deviation (`‖RᵀSR−S‖`, via `line.get_R_matrix`)
  both scale as `1/n_steps²`, as expected for the leapfrog-like scheme. Also
  has a field-component plot (`fig4`: Bx,By,Bz vs s at the tracked particle's
  transverse offset).
- `005_study_symplectic_error.py` — see below; still evolving.
- `006_check_boris_corrected.py` — tests a `g(theta)=(theta/2)cot(theta/2)`
  drift/kick-rescaling hypothesis (new `xtrack.BorisSpatialCorrected`
  class) against plain `BorisSpatialIntegrator`; see
  `02_g_theta_correction_hypothesis.md` — **result: hypothesis not
  supported**, corrected version's symplectic error came out ~2-3x larger,
  same `1/n_steps²` order.

## 005: symplectic-error study — current state and findings

**Goal** (from the user): quantify `BorisSpatialIntegrator`'s symplectic
error against something independently known to be symplectic, using the
`R`-matrix method from 004 (`‖RᵀSR−S‖` via `line.get_R_matrix`).

**Field model history in this script:**
1. Started with `LinearFringeSolenoid` + a `VariableSolenoid`-chain
   reference (built by slicing `xt.VariableSolenoid` with `integrator=
   'yoshida4'` to match the taper breakpoints). This converged nicely but
   turned out to be a weak comparison: `VariableSolenoid`'s C code
   ([variable_solenoid.h](../../../xtrack/beam_elements/elements_src/variable_solenoid.h#L38-L44))
   computes `ax=-0.5*ks*(y-y0)`, `ay=0.5*ks*(x-x0)` — i.e. **the same
   paraxial linear-taper field model** as `LinearFringeSolenoid`, not an
   elliptic-integral/exact finite-solenoid field. It's a legitimate
   *symplectic-integration* reference (composition of exact sub-maps via
   Lie–Trotter/Yoshida splitting) but not an independently-modeled physical
   field, so the comparison was somewhat circular.
2. Switched to `SolenoidField` (`xtrack/_temp/boris_and_solenoid_map/solenoid_field.py`)
   — the elliptic-integral closed-form field of a real finite circular
   solenoid (Hampton et al., *AIP Advances* 10, 065320 (2020)), already used
   in `tests/test_boris_spatial.py`. The `VariableSolenoid` reference was
   removed for now (deferred — see "Next step" below); currently 005 only
   computes Boris's own `sympl_error` vs `n_steps`, no external reference.

**Params currently in the script:** `p0`: electron, `mass0=xt.ELECTRON_MASS_EV`,
`q0=1`, `x=-1e-3, y=1e-3, px=-1e-3`; `sf = SolenoidField(L=4, a=0.3, B0=1.5,
z0=20)`; `s_start=0, s_end=30`; `n_steps_vect = [200, 500, ..., 100000]`
(same as 004). `energy0` is the key knob for the finding below — it's
currently set to `45.6e9/1000` (45.6 MeV, copied from
`test_boris_spatial.py`), which puts the trajectory in a **strongly-bending**
regime (see below); full FCC-ee energy (`45.6e9`, no `/1000`) gives a
paraxial regime instead.

### Key finding: `SolenoidField` is correct; the symplectic-error floor is real and amplitude-dependent

While testing at `energy0=45.6e9/1000` (45.6 MeV), `sympl_error` **did not
converge to zero** as `n_steps → ∞`; it converged (cleanly, as `1/n_steps²`,
confirmed by curve fit) to a nonzero floor of **~28** — four to five orders
of magnitude larger than the ~1e-7–1e-9 floor seen with `LinearFringeSolenoid`
in 004.

Investigation trail (don't need to repeat this — findings below are settled):
- **First hypothesis (wrong): field bug.** A finite-difference check of
  `div B` on `SolenoidField.get_field` gave `~0.05` (nonzero) at points along
  the trajectory. This turned out to be a **testing artifact**: the check
  used a Python `int` for `z`, which trips a real but narrow dtype bug at
  [solenoid_field.py:76-77](../../../xtrack/_temp/boris_and_solenoid_map/solenoid_field.py#L76-L77) —
  `Bx = 0 * z` / `By = 0 * z` should be `0 * x` / `0 * y`. As written, `Bx`/`By`
  inherit `z`'s dtype; an integer-dtype `z` silently truncates `Bx`,`By` to 0
  wherever `|B|<1`. **This never triggers during actual particle tracking**
  (coordinates are always `float64`), so it does not explain the symplectic
  error. Worth a one-line fix regardless, but low priority.
- With proper `float` inputs, `div B ≈ 1e-9` (numerical noise). Independently
  cross-checked `Br`/`Bz` against a from-scratch brute-force Biot–Savart
  calculation (superposition of circular current loops using the paper's
  single-loop Eqs. 17/19, integrated over the solenoid length) — agreement
  to 6 significant figures. **`SolenoidField`'s formula is correctly
  implemented**, matching Hampton et al. exactly.
- Checked `pz` (longitudinal momentum fraction) along the reference
  trajectory and all six `get_R_matrix` finite-difference-perturbed
  trajectories: stays `~0.998` throughout, never near zero — rules out a
  `pz→0` singularity in the spatial parameterization (`x += px/pz · dz`) as
  the cause.
- **Amplitude test (the informative one):** reran with `energy0=45.6e9` (full
  FCC-ee energy, no `/1000`) instead of 45.6 MeV, same `B0=1.5T, a=0.3m`.
  This is far more paraxial (Larmor radius ~100m vs. `a=0.3m`, trajectory
  stays nearly straight). Result: `sympl_error ≈ 0.0017`, **flat** already
  from `n_steps=1000` (not decreasing further with more steps — likely
  already dominated by `get_R_matrix`'s own finite-difference floor, not by
  Boris's discretization at all). That's ~4 orders of magnitude smaller than
  the strongly-bending-regime floor.

**RESOLVED.** The `zeta`-bookkeeping hypothesis above was wrong and has been
refuted analytically: `boris.py`'s `zeta` update is provably *exact*, term
for term, matching xtrack's canonical `Drift_single_particle_exact` formula
(`track_drift.h:34`) — shown by direct algebraic substitution (`P =
gamma*m_kg*beta*c` cancels exactly). Confirmed by an FD-step-size sweep
(varying `get_R_matrix`'s `steps` dict by 5 orders of magnitude changed the
~28 floor by <0.2%) and by printing `RᵀSR−S` itself: the `zeta` row/column
were *exactly* zero; the defect was concentrated almost entirely in the
`(px,py)` entry (28.0 out of 28.17 total).

**Actual root cause: canonical vs. kinetic transverse momentum.**
`BorisSpatialIntegrator` integrates Newton's law directly, so its internal
`px`, `py` are *kinetic* momentum, but xtrack's `p.px`, `p.py` elsewhere
mean *canonical* momentum, related by `canonical = kinetic + q*A/P0`
(confirmed against xtrack's own convention,
`track_magnet.template.h:97`: `kin_px = px - ax`). For a solenoid, `A_x,
A_y` (hence `a_x = q*A_x/P0`) are nonzero anywhere the coil's field/fringe
hasn't fully decayed — even at `s_end=30`, 8 m past the coil edge, `Bz ≈
3e-4 T` (vs `B0=1.5T`), giving a tiny but nonzero `a_x, a_y ~ 1e-4`. Boris
never applied this correction, so the *stored* `p.px, p.py` were silently
kinetic, not canonical. This tiny gauge offset gets hugely amplified by the
strongly-bending trajectory's own sensitivity (`R[1,3]≈108`), producing an
O(1)-O(10) defect that is exactly `n_steps`-independent (it's a bookkeeping
omission, not a discretization error — this is *why* it didn't shrink like
`h²`, which is what tipped off the wrong turn: the user correctly expected
the true per-step Boris defect to scale as `h³` local / `h²` global and
vanish, and it does, once this offset is corrected — see below).

Verified decisively: manually replicating `get_R_matrix`'s 13-particle FD
scheme through the Boris element and adding `a_x = q*A_x/P0`, `a_y =
q*A_y/P0` to the raw output `px, py` before building `R` dropped the
symplectic error from **28.17 → 0.00023** — five orders of magnitude, using
only the entry/exit correction (no change to the `n_steps` loop itself).

**Vector potential added to `solenoid_field.py`.** `SolenoidField` didn't
expose `A`, only `B`. Added
`SolenoidField.get_vector_potential(x, y, z, n_quad=200)`
implementing Hampton et al. Appendix C, Eq. (C26)/(C31) (`Aθ` via
Gauss-Legendre quadrature over the source angle `θs`; purely azimuthal,
`Az=0`, converted to Cartesian `Ax = -Aθ*y/r`, `Ay = Aθ*x/r`). Validated:
numerical `curl(A)` matches `get_field`'s `B` to 6+ significant figures at
several test points (inside, near-edge, far downstream).

**Fix applied to `BorisSpatialIntegrator` (`boris.py`).** Added an optional
`vector_potential_callable` constructor argument. If given,
`track()` now converts `p.px, p.py` canonical→kinetic at entry
(`_shift_canonical_kinetic(..., sign=-1)`) before the leapfrog loop (which
still operates entirely in kinetic-momentum space, unchanged), then
kinetic→canonical at exit (`sign=+1`). Backward compatible: if
`vector_potential_callable=None` (default), behavior is unchanged.

**Confirmed fixed end-to-end** via `line.get_R_matrix(...)` with
`vector_potential_callable=sf.get_vector_potential`: the floor is gone and
`sympl_error` now shrinks cleanly with `n_steps` (5.7 → 0.70 → 0.021 →
0.0013 → 0.00005 for n_steps = 200 → 1000 → 5000 → 20000 → 100000) —
consistent with real `1/n²` convergence to zero, exactly the h² scaling
expected for a 2nd-order symplectic-like leapfrog. `005_study_symplectic_error.py`
now plots both the uncorrected (floored) and corrected (converging) curves
for comparison.

See `01_symplectic_error_mechanism.md` for a detailed, numerically-verified
walkthrough of *why* this bookkeeping gap produces an O(10)-O(30) symplectic
violation instead of something imperceptibly small — the short version is
two multiplicative amplifications: the tiny (~1e-6) fringe-field gauge term
gets chain-ruled through the R-matrix's own large bending-induced position
sensitivities (`R ~ O(10)-O(100)`, normal solenoid optics) into a ~1%-level
defect confined to the `px,py` rows, which then gets amplified a second time
by the symplectic check's bilinear (`RᵀSR`) structure.

### Next step

The original ask — find a *known-symplectic reference integrator* to
compare Boris against — is still not done; 005 only compares Boris against
itself (corrected vs. uncorrected). Worth doing now that the field-model
question (`SolenoidField` vs `LinearFringeSolenoid`/`VariableSolenoid`) and
the canonical-momentum question are both settled, so a real reference
comparison (e.g. a paraxial `VariableSolenoid`-chain, or a finer look at
the residual sub-floor once corrected) would be measuring genuine
integration error rather than being confounded by either issue.

Separately, still unresolved (flagged, not fixed): `solenoid_field.py`'s
`get_field` has a minor dtype bug at `Bx = 0*z` / `By = 0*z` (should be
`0*x`/`0*y`) — never triggers with real float64 particle tracking, low
priority. And `xtrack/beam_elements/elements_src/track_splineboris.h` (a
compiled, production spatial-Boris element) appears to have the same
kinetic/canonical `px` ambiguity — it writes kinetic `px` straight to
`LocalParticle_set_px` without an `ax` correction — but this wasn't traced
through its wrapping `track_magnet_edge.h` (`model==3`) logic; worth an
independent look if `SplineBoris` is used anywhere for real tracking.
