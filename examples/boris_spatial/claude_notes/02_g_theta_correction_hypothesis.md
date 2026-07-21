# g(theta) drift/kick rescaling hypothesis — implemented, tested, does not help

Follow-up to `00_overview.md`/`01_symplectic_error_mechanism.md`. The user
proposed (via a from-scratch Lie/BCH derivation in a linked "Sympy
Commutators" note, not stored in this repo) that rescaling the drift (`D`)
and kick (`K`) sub-steps of the D-K-R-K-D Strang splitting in
`step_spatial_boris_B` (`xtrack/boris.py`) by `g(theta) = (theta/2)*cot(theta/2)`,
`theta = q*Bz*dz/pz` (the per-step Bz-rotation angle), would reduce the
integrator's symplectic error — because their BCH expansion of the
uncorrected splitting gives `D`, `K` coefficients `1 - theta^2/12 + ...`
instead of 1, and rescaling by `g(theta)` exactly restores coefficient 1
for both (leaving only a residual `T`-commutator term that can't be
absorbed this way).

**Implemented as requested**, following the user's literal 3-step recipe,
in `xtrack/boris.py`:
- `_g_theta(theta)`: `(theta/2)*cot(theta/2)`, series-expanded near
  `theta=0` to avoid the removable `0/0`.
- `step_spatial_boris_B_corrected(...)`: (1) predictor half-drift + field
  eval at that midpoint, (2) predictor half-kick from that field to get
  `pz` at the kick point → `theta`, `g(theta)`; (3) the *real* D-K-R-K-D
  step, re-evaluating the field at the `g`-corrected midpoint, with drift
  and kick (not rotation) sub-steps scaled by `g`.
- `BorisSpatialCorrected(BorisSpatialIntegrator)`: identical class, only
  swaps in the corrected step function via a new `_step_fn` class
  attribute (refactored `BorisSpatialIntegrator` to dispatch through
  `self._step_fn` instead of calling `step_spatial_boris_B` directly, so
  both classes share `__init__`/`track`/canonical-momentum handling
  unchanged). Exported from `xtrack/__init__.py`.
- `006_check_boris_corrected.py`: same setup/sweep as `005` (45.6 MeV
  electron, `SolenoidField(L=4, a=0.3, B0=1.5, z0=20)`, `s: 0→30`),
  compares `sympl_error` of `BorisSpatialIntegrator` vs
  `BorisSpatialCorrected`, both with `vector_potential_callable` set (so
  the canonical/kinetic bug from `00_overview.md` isn't a confound).

**Result: the hypothesis does not hold, at least as tested.**
`BorisSpatialCorrected`'s symplectic error is **~1.8–2.7x *larger*** than
plain `BorisSpatialIntegrator`'s at every sampled `n_steps` from 200 to
100000, and **both still scale as `1/n_steps²`** (same convergence order,
just a different prefactor):

```
n_steps=    200  plain=5.72e+00  corrected=1.05e+01  ratio=1.84
n_steps=   1000  plain=7.04e-01  corrected=1.92e+00  ratio=2.73
n_steps=   5000  plain=2.15e-02  corrected=5.22e-02  ratio=2.43
n_steps=  20000  plain=1.34e-03  corrected=3.20e-03  ratio=2.39
n_steps= 100000  plain=5.22e-05  corrected=1.30e-04  ratio=2.49
```

(`_g_theta` itself checked independently against direct/series formulas —
not a bug in the scale factor.)

**Candidate explanation (not verified to the depth of `01_...md` — worth
scrutiny before trusting):** each of `D`, `K`, `R` in
`step_spatial_boris_B` is already an *exact* solution of its own
sub-Hamiltonian (kick/rotation with position frozen at the field-eval
point, drift with momentum frozen) — a standard fact from geometric
integration is that a composition of exact sub-flows is *unconditionally*
an exact composition of symplectomorphisms, regardless of step size or how
well the composed map's BCH log matches the true (unsplit) generator. If
so, the plain D-K-R-K-D map should already be exactly symplectic
(analytically) for any `n_steps`, and the `~1/n²`-decaying "symplectic
error" both 005 and 006 measure via `get_R_matrix`'s finite-difference `R`
is plausibly dominated by the FD estimator's own truncation noise rather
than a real defect of the analytic map — in which case the BCH-matching
trick has nothing genuine to cancel, and `g(theta)` only perturbs the map's
*accuracy* (vs. the true continuous flow) and the predictor/corrector field
evaluation (two field calls at slightly different midpoints per step,
vs. one in the plain scheme) adds asymmetry that plausibly explains the
worse FD-measured number. Also notable: `theta ≈ 2.96` in this test case
(strongly-bending regime, deliberately chosen in `005` to make the earlier
canonical-momentum bug visible), so `g(theta) ≈ 0.135` — a large,
non-perturbative rescaling, not a subtle correction; worth re-testing at
small `theta` (paraxial/full-energy regime) if this is revisited, and worth
independently confirming/refuting the "already unconditionally symplectic"
claim above (e.g. by checking `R^T S R = S` to much higher precision than
the FD-based `get_R_matrix`, via an analytic/autodiff Jacobian of one
`step_spatial_boris_B` call) before concluding the splitting structure
itself is the reason this hypothesis fails.

**Not changed:** `step_spatial_boris_B` (the original, still the default
via `BorisSpatialIntegrator`) is untouched; `BorisSpatialCorrected` is
purely additive/experimental.

## Follow-up: added the T-operator correction — still doesn't help

The user's derivation also gives the full BCH exponent of the
*uncorrected* D-K-R-K-D map, including the residual term left out above:

```
exp(h/2 D) exp(h Z_KR) exp(h/2 D)
    = exp( h R + h g(theta) (D + K) + p_T T ),
p_T = (h^2 / P_z) * g(theta)(1 - g(theta)) / theta,
T = q*Bx*d_x + q*By*d_y  (a position-shift generator)
```

Implemented in `step_spatial_boris_B_corrected` (`xtrack/boris.py`): after
the existing g(theta)-rescaled D-K-R-K-D step, subtract `p_T * q * Bx`
from `x1` and `p_T * q * By` from `y1` (i.e. apply `exp(-p_T T)`) using
the field at the corrected mid-step, to cancel the residual `p_T T` term
from the map's merged generator. `p_T`'s `g(1-g)/theta` factor is
series-expanded near `theta=0` (`_pT_coeff`) same as `_g_theta`.
Applied once at the end of the step (not symmetrized around the D-K-R-K-D
block — simplicity over exactness, worth revisiting if this line of
attack is pursued further).

**Result: still doesn't help, at the tested (large) theta.** Re-ran
`006_check_boris_corrected.py`:

```
n_steps=    200  plain=5.724e+00  corrected(+T)=2.809e+00  ratio=0.491
n_steps=   1000  plain=7.038e-01  corrected(+T)=2.346e+00  ratio=3.332
n_steps=   5000  plain=2.150e-02  corrected(+T)=6.156e-02  ratio=2.863
n_steps=  20000  plain=1.337e-03  corrected(+T)=3.727e-03  ratio=2.788
n_steps= 100000  plain=5.221e-05  corrected(+T)=1.495e-04  ratio=2.863
```

Notable: at `n_steps=200` the T-correction actually *undercuts* plain
Boris (ratio 0.49 < 1) — the only sampled point where the corrected
scheme wins. At every other `n_steps` it's ~2.8-3.3x worse than plain,
i.e. essentially unchanged from (very slightly worse than) the g(theta)-only
result above. Still `1/n_steps²` scaling throughout, no order improvement.
Doesn't change the standing candidate explanation above (D,K,R being exact
sub-flows may already make the plain splitting unconditionally symplectic,
so the FD-measured "error" here may not be a real defect these correction
terms can fix) — if anything the T-correction's consistent extra penalty
at large `n_steps` reinforces it, though the `n=200` flip is odd enough to
be worth a second look before treating that as settled.

Also unresolved from before: the g(theta) scaling convention itself
(multiply sub-step amplitude by `g(theta)` vs. divide by it) was never
independently re-derived/checked against the BCH formula above; worth
scrutiny if this hypothesis gets revisited, since the sign/direction of
that scaling determines whether it's supposed to shrink or grow the
sub-steps relative to the uncorrected map.

Separately: fixed an unrelated int64-overflow bug in
`006_check_boris_corrected.py`'s `~1/N^4` reference plotting line
(`n_steps_vect[2:]` defaulted to `int64` under `np.array(...)`, silently
wrapping at `100000**4`, ~10^20, well past int64 max, producing an
artificial flatline in the plotted reference curve at the last point).
Fixed by casting to `float` before exponentiating; purely cosmetic, no
effect on the actual measured `sympl_error` data.

## Why doesn't it help? — ranked list of candidate reasons

Discussed with the user after the T-correction result above. Most to
least likely:

1. **The metric may not be measuring a real defect.** `D`, `K`, `R` are
   each exact flows of their own sub-Hamiltonian, so their composition is
   *unconditionally* symplectic regardless of step size (standard
   splitting-method fact — see candidate explanation above). If true, the
   `~1/n²` "symplectic error" `get_R_matrix` reports is dominated by its
   own finite-difference truncation noise, not a real defect — in which
   case there's nothing genuine for `g(theta)`/`T` to cancel, and the
   extra arithmetic they add (predictor pass, `cot`, divisions) just gives
   the FD differentiator more noise to pick up.
2. **theta ≈ 2.96 is far outside the perturbative regime the correction
   was derived for.** `g(theta)` and the BCH expansion behind it assume
   small `theta`. Near `theta ≈ pi` (`cot(theta/2) → 0`), `g(theta)`
   drops to ~0.135 — a huge, non-perturbative rescaling — and the
   higher-order BCH terms the derivation neglects are almost certainly
   *not* negligible there, so canceling only the `g(theta)(D+K)` and
   `p_T T` terms doesn't guarantee net improvement.
3. **Scaling-direction ambiguity, never independently re-derived.** The
   code multiplies the D,K sub-step amplitude by `g(theta)`; if the
   uncorrected step's merged coefficient is already `g(theta)`, restoring
   it to 1 requires *dividing* by `g(theta)`, not multiplying — multiplying
   could push the effective coefficient toward `g(theta)^2`, further from
   the true flow rather than closer.
4. **The T-correction isn't symmetrized** (applied once at the end
   instead of split before/after the D-K-R-K-D block), breaking the
   time-reversal symmetry that lets Strang-type splittings cancel error at
   even orders — plausibly explains the non-monotonic `n_steps=200`
   improvement vs. worse-everywhere-else pattern.
5. **Possible misreading of the `T` operator's action.** `T = qBx*d_x +
   qBy*d_y` was implemented as "shift x,y directly." In Lie-operator beam
   dynamics (Dragt/Forest-style) such differential-operator notation is
   often shorthand for a Poisson-bracket generator whose actual
   phase-space action isn't necessarily "shift this coordinate directly" —
   without the source derivation note this can't be fully ruled out, and
   would mean the correction perturbs the wrong variables.
6. **Extra field evaluations / predictor-corrector asymmetry** in
   `step_spatial_boris_B_corrected` vs. the single-field-eval plain scheme
   add their own step-size-dependent error structure, independent of the
   `g(theta)`/`T` theory itself.

Most likely #1 + #2 together explain most of the effect (nothing real to
fix, using a formula outside its validity range); #3 and #4 are concrete,
checkable implementation choices worth revisiting first if this is
pursued further.

## Follow-up: tested the #3 scaling-direction flip (1/g instead of g) — worse, not better

Reasoning (worked out with the user): `theta` is fixed entirely by the
rotation `R` (`Bz`, `dz`, `pz`), independent of how large the `D`, `K`
sub-steps are made. Since every term in the BCH tower that produces
`g(theta)` carries exactly one factor of `D` or `K`, scaling `D, K` by a
prefactor `c` should scale the merged `(D+K)` coefficient linearly:
`coefficient = c * g(theta)`. The original implementation set `c =
g(theta)`, giving merged coefficient `g(theta)^2` — moving *away* from
the target coefficient 1 (matching `D+K+R`), not towards it. Setting
`c = 1/g(theta)` instead should give coefficient exactly 1.

Implemented by changing the D,K sub-step scale factor in
`step_spatial_boris_B_corrected` from `g` to `ginv = 1/g` (4 call sites);
the T-correction (`p_T`, step 5) was dropped for this test, since `p_T`
was derived for `c=1` and its `c`-dependence hasn't been rederived — so
it can't be trusted alongside `c=1/g(theta)`. Re-ran the same
`n_steps` sweep:

```
n_steps=    200  plain=5.724e+00  corrected(1/g)=4.340e+01  ratio=7.582
n_steps=   1000  plain=7.038e-01  corrected(1/g)=3.162e+00  ratio=4.492
n_steps=   5000  plain=2.150e-02  corrected(1/g)=8.758e-02  ratio=4.074
n_steps=  20000  plain=1.337e-03  corrected(1/g)=5.385e-03  ratio=4.028
n_steps= 100000  plain=5.221e-05  corrected(1/g)=2.148e-04  ratio=4.115
```

**Worse than the original multiply-by-g version** (which was ~1.8-2.7x
plain; `1/g` is ~4.0-7.6x plain), despite `1/g` being the
coefficient-matching-correct direction per the argument above. Also
notable: `theta` is *not* constant across this sweep (`theta ∝ dz ∝
1/n_steps`, so `theta` shrinks from large at `n_steps=200` down to the
per-step-tiny/perturbative regime at `n_steps=100000`) — yet the ratio
stays roughly flat (~4-4.5x) once `n_steps >= 1000`, i.e. the correction
is consistently worse across a wide range of `theta`, including well
inside the small-`theta`/perturbative regime where #2 ("outside the
regime it was derived for") shouldn't apply. That weakens #2 as a
sufficient explanation on its own and shifts weight back towards #1
(the FD `sympl_error` metric may not be measuring a real, fixable
defect) or towards the missing `p_T T` term being load-bearing at *any*
`c` (i.e. the coefficient-1 condition on `(D+K)` alone, without the
correct `T` correction, isn't enough to get closer to the true flow) —
consistent with #4/#5 as well, still unresolved.

Not changed: `step_spatial_boris_B` (plain, default via
`BorisSpatialIntegrator`) untouched.

## Follow-up 2: rederived T-correction for c=1/g, re-enabled — small improvement, still worse than plain

The user supplied a rederived formula for the T-correction under
`c = 1/g(theta)`: since `p_T` at `c=1` already carries one net factor of
`c` from the BCH bookkeeping, and `T`'s coefficient should scale the same
way the `(D+K)` coefficient does, the `c=1/g` version needs an extra
`1/g^2` relative to the `c=1` formula:

    cT = p_T / g^2 = (dz^2/pz_kick0) * _pT_coeff(theta, g) / g^2
    x1 -= cT * q * Bx0
    y1 -= cT * q * By0

(using the predictor-pass `Bx0, By0, pz_kick0` — consistent with where
`theta`/`g` are evaluated; `cT` is `O(h^3)` so the exact evaluation point
only affects higher orders). Implemented as step (5) in
`step_spatial_boris_B_corrected`, re-enabling `_pT_coeff` (previously
unused after Follow-up 1).

Sign was not derivable with confidence from the operator bookkeeping
alone (too many sign conventions in play between the notes' generator
definitions and the code's kick sign), so checked empirically against
both signs:

```
n_steps=    200  plain=5.724e+00  corrected(1/g + T, "-")=5.077e+01  ratio=8.870
n_steps=   1000  plain=7.038e-01  corrected(1/g + T, "-")=2.751e+00  ratio=3.908
n_steps=   5000  plain=2.150e-02  corrected(1/g + T, "-")=7.838e-02  ratio=3.646
n_steps=  20000  plain=1.337e-03  corrected(1/g + T, "-")=4.853e-03  ratio=3.630
n_steps= 100000  plain=5.221e-05  corrected(1/g + T, "-")=1.933e-04  ratio=3.703

n_steps=    200  plain=5.724e+00  corrected(1/g + T, "+")=3.676e+01  ratio=6.423
n_steps=   1000  plain=7.038e-01  corrected(1/g + T, "+")=3.573e+00  ratio=5.076
n_steps=   5000  plain=2.150e-02  corrected(1/g + T, "+")=9.684e-02  ratio=4.504
n_steps=  20000  plain=1.337e-03  corrected(1/g + T, "+")=5.911e-03  ratio=4.421
n_steps= 100000  plain=5.221e-05  corrected(1/g + T, "+")=2.359e-04  ratio=4.519
```

`"-"` (i.e. `x1 -= cT*q*Bx0`, matching `+p_T T` with `T = q*Bx*d_x +
q*By*d_y` in the notes' convention) wins at every `n_steps` except 200,
and is the sign now in the code. Comparing to Follow-up 1 (1/g alone, no
T): adding the T-correction with the right sign shaves the ratio from
~4.0-4.5x down to ~3.6-3.9x at `n_steps >= 1000` — a small, consistent
improvement, but the correction is still worse than plain Boris
everywhere, and still worse than the original multiply-by-g version
(~1.8-2.7x, no T-correction). At `n_steps=200` the "-" corrected version
is *worse* than "1/g alone" was (8.87x vs 7.58x) — `theta` is largest
there (least perturbative), consistent with `cT`'s `O(h^3)` derivation
being least trustworthy at large `theta`.

Net effect of both follow-ups: getting the `(D+K)` coefficient to exactly
1 and adding the (now correctly-signed) `T`-term correction is closer to
the true generator than the original `g(theta)` multiply, but empirically
overall worse than not correcting at all. This is a second data point
against "the multiply-by-g version was accidentally better because it
left more of the BCH tower uncorrected" and continues to weigh in favor
of #1 (the FD `sympl_error` metric not reflecting a real, fixable defect)
over #2/#4/#5 as the dominant explanation — though none of these are
conclusively settled.

`BorisSpatialCorrected` now uses `1/g(theta)` for the D,K sub-step
scaling plus the `-cT*q*Bx0`/`-cT*q*By0` T-correction (step 5 in
`step_spatial_boris_B_corrected`); `_pT_coeff` is back in active use.
