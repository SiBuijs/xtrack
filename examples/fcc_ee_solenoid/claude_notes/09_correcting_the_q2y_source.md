# Trying to correct the Q''y source — what failed and why (2026-09-16)

Follow-up to `08_second_order_chromaticity_source.md`, which diagnosed *where*
the large Q''y comes from. This note records the attempts to *fix* it. All of
them failed, and the measurement of why is the useful part.

> **Status 2026-09-17 — read `10_conditioning_the_optics_match.md` first.**
> The problem was solved, but not by targeting anything. It was a conditioning
> problem: the match is rank-deficient and `xdeps` was inverting its near-null
> directions, so *which* null-space point the solver reached was arbitrary
> run-to-run. `OPTICS_RCOND = 1e-6` plus a staged pass-1 solve fixed it with no
> new targets.
>
> One section below is **wrong**: "Why the solver never found it — one
> hypothesis checked and killed" argues truncation is not the problem because
> `rcond=1e-14` sits far below the condition number. The mechanism it then
> describes (step amplified by 1/σ, solver leaves the linear regime) is right;
> the conclusion is backwards. That amplification is exactly what truncation
> prevents. See note 10.
>
> The null-space measurement, the response matrix recipe, the W reference
> section, and the two "where to go next" items remain valid and useful.

**Verdict up front.** The local chromatic correction's −I sextupole pair needs
two conditions to cancel the doublet's chromatic kick: the right **phase**
(QD0 → sdy1 = 0.5) and the right **strength** (q_S = Σ βy·k2L·Dx matching
q_F = Σ βy·k1L). The 004c/004cc half-straight optics match constrains
**neither**. Worse, within the null space of its existing targets the two
conditions trade against each other: the knob direction that fixes the phase
drifts the optics at the sdy sextupoles enough to break the strength match by
more than it gains. So the phase target recorded as "the fix" at the end of
note 08 does not work, and neither does targeting W directly.

## Correction to note 08

Note 08's section "Fix (implemented 2026-09-15, not yet run at time of
writing)" should be read as **attempted and failed**. Two independent reasons,
both measured below:
1. It targets `mux` as well as `muy`. The horizontal phase is essentially
   uncontrollable by these knobs (0.05% of its gradient survives the linear
   targets), so that match can never converge whatever the solver does.
2. Even the vertical one, achieved in isolation, makes W *worse*, not better.

## What 004cc looks like now

`004cc_correct_solenoids_in_fcc_ring.py` was the sandbox for all of this: one
IP, right half straight only (IP → `end_straight_start_ds_<ip>`), so it runs in
minutes rather than the full 004c.

- **`IP_NAME` at the top** selects the IP. Element names are resolved by
  *family* from the cut half straight (`elements_in_straight()`), so `qd0ar`
  gives `qd0ar.0` at ipa and `qd0ar.3` at ipj. Verified against 004c's
  hardcoded `config` dict for all four IPs, including the irregular
  `sext_right` (`sdm1r.0/.2/.4/.6`).
- **Chromatic phase advance section** at the end: the 004dd calculation
  (quadratic fit of mu(s, δ)) restricted to the straight, plus `wx_chrom` /
  `wy_chrom`, plus the QD0 → sdy1 phase, printed and plotted per case.
- **Per-knob decomposition**: `PHASE_CASES` = bare / corrected / no orbit /
  no optics / no coupling. `set_correction_state()` overrides one
  `on_sol_<part>_corr_<ip>` knob at a time and writes the expression back
  afterwards (assigning the string re-attaches it to `on_sol_corr_<ip>`;
  verified).

- **Two independent correction chains (added 2026-09-21/22).** The whole
  orbit / optics / coupling chain is now `solve_correction_chain()`, run twice
  with every vary knob reset to zero in between (`reset_vary_knobs()`):
  - **Chain A** has the `sext` targets (betx/bety/dy at sdm1r) off and solves
    the optics with `rcond=1e-6`. Without `rcond` it drifted off its own
    boundary targets (betx at the straight end 1.4% off).
  - **Chain B** has the `sext` targets on and uses no `rcond`. It is the
    original code path, and the generated knobs come from it.

  Chain A only feeds the beta comparison: `BETA_CASES`, a printed table at
  sdm1r, and two `BETA_PLOT_RANGES` panels.

  Inside each chain the optics solve is still staged on `wchrom`. Linear
  targets come first, then W, because W runs to 1e6 in trial states and
  otherwise dominates the merit.

  **Staging the sext targets the way 004c's `OPTICS_STAGE_SEXT` does
  (boundary-only first, then continue with `sext` on) was tried here on
  2026-09-21 and failed.** The boundary-only solution (bety ~ 537 m at
  sdm1r) is a trap: the run ended 72% off on betx at the straight end with
  `wy_chrom` 127 instead of 2. That is why the chains are independent. Why
  staging works in 004c but not here is not established; the code comment
  only notes that this match also carries the W targets, which 004c's does
  not.

### Gotchas found while making it IP-switchable
- A stray `sext_for_chromaticity_correction = 'sdm1r.0'` sat in the middle of
  the correction section and silently overwrote the derived name → `KeyError`
  at ipj. A test that only ran the setup section missed it. The test that
  catches this class of bug runs the **whole file** with
  `xdeps.optimize.optimize.Optimize.solve` stubbed to a no-op, so every
  optimizer is constructed (that is where target lookups happen) without the
  long matches.
- All four half straights are optically near-identical: bare W_y at the end is
  2.013 (ipa) / 2.056 (ipj), and μy across the straight agrees to 6 digits. Any
  per-IP difference comes from the correction, not the geometry.

## Where the damage comes from (confirmed at knob level)

ipa, right side, 3T. Phase error = corrected − bare of μy(qd0ar.0 → sdy1r.0):

| case | μy phase error | W_y at end of straight |
|---|---|---|
| bare | — (reference 0.49999233) | 2.013 |
| corrected | −4.75e−5 | 4.678 |
| no orbit | −4.73e−5 | 5.203 |
| **no optics** | **+2.2e−7** | 11.904 |
| no coupling | −4.75e−5 | 4.905 |

- Switch off **only** the optics correction and the phase error vanishes into
  noise. Orbit and coupling correction are innocent to three digits. This is
  the knob-level confirmation of note 08's attribution.
- −4.75e−5 matches note 08's independently measured ipa-right value (−4.0e−5).
- The W leak is quantitatively explained by the phase error:
  `4π q_S ε = 4π × 5560 × 4.75e−5 = 3.3` against an observed excess over bare
  of `4.678 − 2.013 = 2.7` (ratio 0.8, the same factor the lumped model gives
  elsewhere).
- Note the counter-intuitive row: the optics correction *reduces* W_y
  (11.9 → 4.7) while introducing the phase error. Without it βy at the sdy
  sextupoles is wrong, i.e. a **strength** mismatch, which costs more than the
  phase error does. The optics match is doing real work; it just leaves the
  remaining degree of freedom unconstrained.

## Attempt 1 — target `wy_chrom` directly, single stage

`chrom=True` passes through `match()` into the twiss (`ActionTwiss` stores
kwargs), so `wx_chrom`/`wy_chrom` at `xt.END` are valid targets. Cost: three
twisses per merit evaluation instead of one.

**Failed badly.** Penalty trace: start 40.35 → 7.7e6 → 7.3e8 → back to 40.35.
`take_best` returned the *starting* point, so `generate_knob` produced a null
knob and there was no optics correction at all (`corrected` and `no optics`
came out byte-identical in the diagnostics — that is the tell).

Cause: W is unbounded and strongly nonlinear. In the trial states between the
uncorrected optics and the solution, `wy_chrom` at the end of the straight runs
to 1e6 and beyond; it then dominates the merit function and the Gauss-Newton
step becomes meaningless. β misbehaves by factors, W by six orders of
magnitude.

## Attempt 2 — same target, two stages

```python
opt_optics.disable(target='wchrom')
opt_optics.solve()                    # linear targets only
opt_optics.enable(target='wchrom')
opt_optics.solve()                    # now W is O(10), not O(1e6)
opt_optics.target_status()
```

Solver behaves: stage 1 ends at 2.56e−7, stage 2 at 2.89 (not 40.35), and the
failure mode is now safe — the worst `take_best` can do is return the stage-1
solution. All nine linear targets met to 1e−7…1e−9. But:

| target | current | target | residue |
|---|---|---|---|
| wx_chrom | 8.191 | 8.088 | 0.104 |
| wy_chrom | **4.904** | **2.013** | **2.891** |

W_y is stuck at ~4.9 — **the same value the match reaches without the W target
at all** (4.678). 786 model calls bought nothing. The phase error likewise did
not move (−4.751e−5 → −4.773e−5).

## The measurement that explains it — response matrix and null space

Rather than guess whether this was a solver problem or a degrees-of-freedom
problem, build the Jacobian and look. Method (≈16 twisses, seconds):

```python
# after converging the LINEAR targets only
vary_names = [vv.name for vv in opt_optics.vary]   # 15 aux knobs from match_knob
def observables():
    tw = line_ds.twiss4d(init=init_ip)
    return np.array([tw['betx', name_end], tw['bety', name_end],
                     tw['alfx', name_end], tw['alfy', name_end],
                     tw['dx', name_end], tw['dpx', name_end],
                     tw['betx', sext], tw['bety', sext], tw['dy', sext],
                     tw['muy', sdy1] - tw['muy', qd0],     # row 9
                     tw['mux', sdy1] - tw['mux', qd0]])    # row 10
# central differences, STEP = 1e-6, -> J (11 x 15)
A = J[:9]                                    # the linear targets
u, s, vt = np.linalg.svd(A)
N = vt[9:].T                                 # null space, 15 x 6
g_null = N @ (N.T @ J[9])                    # phase gradient projected
dk = g_null * (needed / np.dot(g_null, g_null))   # min-norm knob move
```

Note `match_knob` varies *auxiliary* variables named
`<knob>_from_<knob_name>`, not the raw `k1_*_sol_corr` knobs — perturb the ones
`opt.vary` reports, not the element knobs, which are driven by expressions.

### Results (ipa right, 3T, at the converged linear solution)

Singular values of the 9×15 linear block:
`[1.441e6, 3.773e5, 8.482e3, 4.941e3, 1.346e3, 3.401e1, 9.632, 4.872e-2, 3.924e-2]`
— rank 9, **null space dimension 6**, condition number ~3.7e7.

| | fraction of gradient surviving the constraints | min-norm trims vs existing |
|---|---|---|
| μy(qd0 → sdy1) | 2.54e−2 | 5.3× |
| μx(qd0 → sdy1) | 4.87e−4 | 69× |

So the phase is formally controllable but nearly degenerate with the linear
targets. The min-norm direction for Δμy = +4.773e−5:

| quad | k1 | required dk1 | relative |
|---|---|---|---|
| qd3r.0 | 0.014966 | +2.23e−03 | **14.9%** |
| qf2r.0 | −0.007711 | +5.79e−04 | 7.5% |
| qd4r.0 | −0.019924 | −2.50e−04 | 1.3% |
| qf1ar.0 | 0.100000 | +8.70e−04 | 0.87% |
| doublet quads | | | all < 1% |

### Why the solver never found it — one hypothesis checked and killed

It is **not** least-squares truncation: `xdeps` `SVD.lstsq` defaults to
`rcond=1e-14` (`xdeps/optimize/matrixutils.py:7`), far below the 3.7e7
condition number, so the weak directions are not being discarded. The opposite:
along a direction with a tiny singular value the Gauss-Newton step is amplified
by 1/σ, the solver jumps far outside the linear regime, and the penalty
explodes — which is exactly the 7.7e6 / 7.3e8 excursion of attempt 1. Also
worth knowing: `Optimize.solve` defaults to `broyden=False`, so 004cc was
always using the robust full-Jacobian mode and never had 004c's broyden stall.

## The result that kills the phase-target approach

Apply the predicted direction at several scales and measure (validation run):

```
 scale    muy error    predicted       wy  dbetx/betx  dbety/bety   dbetx_s/b
  0.25   -3.540e-05   -3.580e-05    4.622   +9.34e-04   -4.46e-04   -2.93e-03
  0.50   -2.227e-05   -2.387e-05    6.558   +3.83e-03   -1.79e-03   -1.16e-02
  1.00   +6.371e-06   -4.222e-09   18.966   +1.68e-02   -7.17e-03   -4.55e-02
  1.50   +3.811e-05   +2.386e-05   40.570   +4.36e-02   -1.62e-02   -9.90e-02
```

- The **phase moves as predicted** (within 7% at half scale), so the null-space
  algebra is right and the phase *is* correctable with these knobs.
- But **W_y goes 4.9 → 19 → 41**. Correcting the phase makes the leak four
  times worse.
- Reason: the null space holds the nine *targeted* quantities to first order,
  and βy/Dx at the sdy sextupoles are not among them. They drift (βx at sdm1
  moves 4.6% at full scale), which changes q_S. A 0.4% drift in βy at sdy is
  ~20 units of q_S against the 5560 being cancelled — the size of the W_y
  observed at scale 1. Consistent, though not yet directly measured (the scan
  did not record βy/Dx at sdy1/sdy2 — worth adding if this is revisited).

## Where to go next (not yet tried)

1. **Constrain the sextupoles too**: add βy and Dx at sdy1 and sdy2 to the
   optics-match targets, alongside the muy phase target (drop the mux one).
   That is 14 targets against 15 knobs — tight, quite possibly
   over-constrained, but cheap to test and it answers the feasibility question
   immediately.
2. **Add a knob**: let the sdy sextupole strengths themselves vary. That is the
   natural handle on the strength condition, which nothing currently touches.
   Cost: those sextupoles also set the ring's global Q'y, so the change has to
   be compensated globally.
3. If both fail, the honest conclusion is that a half-straight match with β/α/D
   targets at the straight edge cannot preserve the local chromatic correction,
   and the correction scheme itself needs rethinking.

## Reference: what W is

Montague chromatic functions, from B. W. Montague, *Linear Optics for Improved
Chromaticity Correction*, CERN LEP Note 165 (1979). Definition as xtrack
implements it (`xtrack/twiss.py:2084-2091`, citing the MAD8 Physical Methods
Manual §6.3; identical to the MAD-X User's Guide TWISS table):

```
b_chrom = (dβ/dδ)/β,   a_chrom = dα/dδ − α(dβ/dδ)/β,   W = sqrt(a² + b²)
```

Useful facts:
- W is the amplitude of the beta-beat *per unit δ*: |Δβ/β| ≈ W·δ. W_y ≈ 5e3
  inside the doublet means βy changes by a factor of several at δ = 1e−3.
- (b, a) behaves as a vector: it rotates by 2Δμ between perturbations, and a
  thin gradient error adds β·ΔK to the `a` component. Per unit δ, a quadrupole
  contributes q = βK and a sextupole in dispersion q = βK2L·Dx. MAD also
  outputs the phase Φ = arctan(a/b); xtrack computes it but leaves it
  commented out.
- Q'' is the overlap of the residual beat with the chromatic kicks:
  `Q'' ≈ −(1/4π) Σ q_i b_i`. For the doublet, ΔQ''y ≈ q_F·ΔW/(4π) ≈ 440·ΔW.
  Do **not** use that 440 as a general W → Q'' conversion: the bare ring leaves
  W_y ≈ 2 at every straight yet has Q''y = −149, because the sum is coherent
  and the bare lattice is designed so the terms cancel.
- Tolerance framing that makes the small numbers intelligible: for
  |Q''y| < 100 you need |ε| < 4e−6 in the QD0 → sdy1 phase. The bare lattice
  sits at 8e−6 from exactly 0.5; ipa-right corrected is 12× out of tolerance
  and ipj-left (note 08) is 175× out.
