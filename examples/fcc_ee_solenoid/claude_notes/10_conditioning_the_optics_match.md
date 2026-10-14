# Conditioning the half-straight optics match — the fix that worked (2026-09-16/17)

Third note in the Q''y series, after `08_second_order_chromaticity_source.md`
(where the Q''y comes from) and `09_correcting_the_q2y_source.md` (attempts to
target the phase directly, all failed).

**Verdict up front.** The Q''y problem was never really a physics problem, it
was a **conditioning** problem. The half-straight optics match is
rank-deficient (15 knobs, 8–9 targets), the solver was inverting its
near-null directions, and which point in the null space it landed on was
effectively arbitrary — that is why the sign and magnitude of Q''y flipped
between otherwise identical runs. Two changes to the *solver*, with no new
targets and no new physics, made the result reproducible and small:

1. `OPTICS_RCOND = 1e-6` — truncate the weak singular directions.
2. **Staged pass-1 solve** — solve the straight-boundary targets first with the
   sextupole targets disabled, then re-enable and solve the full set.

Neither touches what is being matched. Both change *which* member of the
solution family the solver picks.

## Why the null space matters even when every target is met

The optics match has 15 vary knobs (12 quad + 3 mid-bend trims per half
straight) against 8 targets (β/α/D at the straight boundary, βx/βy at sdm1).
That leaves ~7 unconstrained directions. Every point in that 7-D family
satisfies the match equally well. They are **not** equally good lattices: the
targets say nothing about the straight's interior, which is where Q'' is
actually built, and nothing about the QD0 → sdy1 phase.

Demonstration (two ipg solutions, both meeting all targets):

| | βy at sdm1 | μy(QD0→sdy1) error | max abs knob |
|---|---|---|---|
| step 1e−6, no rcond | **6306.7 m** | **+0.49** (π → 2π!) | — |
| step 1e−7, rcond 1e−6 | 8.07 m | −3.7e−4 | 2.9e−3 |

Same targets, same tolerances, same penalty class. "The target is met" is
simply not a statement about the ring.

This also retroactively explains the Q''y sign flips recorded in note 08
(+15,400 / +18,500 one run, −9,355 an older one): an unconstrained null-space
coordinate, sampled afresh each run.

## Fix 1 — `OPTICS_RCOND = 1e-6`

Singular values of the linear block span `[1.4e6 … 3.9e-2]`, condition number
3.7e7, with a clean gap of ~200 between the 7th (9.632) and the 8th (4.872e-2).
`xdeps` `SVD.lstsq` defaults to `rcond=1e-14` (`xdeps/optimize/matrixutils.py:7`),
i.e. essentially no truncation, so those two near-null directions get inverted
and the Gauss-Newton step is multiplied by 1/σ.

What that looked like in the run of 2026-09-16 (ipd left, pass 3):

```
penalty 0.546 -> 80.04 -> backtrack only to 0.722 (still uphill, accepted)
        -> 7240.7
```

Over three passes the trims drifted to **26×** their proper size (max |k1|
7.7e-2 against ~2.9e-3 for a good solution). The ring came out with qx=170.26
(bare 194.16), qy=148.16 (bare 170.20), max βx 9089, and no closed orbit at
δ=±1e-3 at all.

Note this **corrects note 09's "one hypothesis checked and killed" section**,
which argued truncation was *not* the problem because `rcond=1e-14` is far
below the condition number. The diagnosis of the mechanism there is right —
along a tiny-σ direction the step is amplified by 1/σ and leaves the linear
regime — but the conclusion drawn from it was backwards. That amplification is
precisely what truncation prevents, and turning truncation on is what fixed it.

With `rcond=1e-6`: all 8 optics matches 9/9 in tolerance, max abs knob
5.4e-4 … 1.5e-3, one incomplete match in the whole run (orbit, penalty 8.2e-9),
ring qx=194.160016 / qy=170.200006, δ=±1e-3 both closing. **d2qy = +682.6**
(bare −37.66), d2qx = 9.92.

### Side effect: tolerances and rcond fight each other

Truncation puts a floor on the achievable residual. `OPTICS_ALF_TOL` had to be
**loosened** 1e-8 → 1e-7 because the solve could no longer deliver 1e-8 and was
reporting INCOMPLETE at residues of 2–5e-8 — a relative beta-beat of the same
1e-8, physically meaningless. Tightening tolerances below the truncation floor
buys nothing and costs false alarms.

## Fix 2 — the staged solve (`OPTICS_STAGE_SEXT`)

```python
opt.disable(target='sext')   # only the 6 straight-boundary targets active
opt.solve(...)               # stage 1
opt.enable(target='sext')
opt.solve(...)               # stage 2, full set
```

Implemented as `solve_optics(opt, label, staged, ...)` in 004c, applied to
**pass 1 only**. Passes 2 and 3 start already near a solution, so disabling
`sext` there would pull βx/βy at sdm1 back off target for nothing and cost an
extra Jacobian per side. Targets are re-enabled in a `finally` — the optimizer
is reused by later passes and then frozen by `generate_knob()`, so an exception
in stage 1 must not leave it permanently missing targets.

**Why it works.** With truncation each Newton step is minimum-norm, so the point
you land on is minimum-norm *relative to where you started*. Stage 1 lands on a
small-trim point near the bare lattice; stage 2 then starts from there instead
of from the unconverged solenoid optics.

Measured on ipg (the half-straight that had blown up):

| | unstaged | staged |
|---|---|---|
| ipg left, max abs knob | 4.54e-03 | **4.69e-05** |
| ipg left, μy error | −2.6e-03 | **+5.3e-05** |
| ipg right, max abs knob | — | 1.82e-04 |
| ipg right, μy error | — | −3.1e-06 |

97× on the trims, 49× on the phase error. For scale, 004c's own comment calls
"a few 1e-4 to 3e-3" normal for a converged half-straight — 4.7e-05 is below
even that, i.e. the staged solve is not merely avoiding the runaway, it is
finding a genuinely minimal correction.

## The calibration check that validates the whole picture

ipg-only run: `d2qy = +264.98`, i.e. `d(d2qy) = +302.64` against bare.

Note 08's calibration (ΔQ''y ≈ 2.42e7 per unit QD0→sdy1 phase error per side),
applied to the two measured errors:

```
(5.3e-05 - 3.1e-06) x 2.42e7 = +1210  true Q''y   ->  +302 in d2q convention
measured:                                             +302.64
```

Agreement to ~0.2%, on a number that had been swinging by tens of thousands
between runs. **This is the main result**: ring Q''y is now a predictable,
linear function of the eight QD0→sdy1 phase errors, rather than a lottery. That
predictability is worth more than the value itself — it means the phase table
004c prints at the end is a sufficient diagnostic, and a full chromaticity
sweep is only needed for confirmation.

## Dead ends measured along the way — do not retry as written

### Knob limits (`OPTICS_K1_LIMIT_*`) — rejected

The idea was ±1e-2 / ±2e-3 bounds that "only bite on a runaway". Measured:

| limits | rcond | ipg left | ipg right |
|---|---|---|---|
| on | off | converges | **stuck at 28.6** |
| on | on | converges | **stuck at 28.1** |
| off | on | converges | converges |

Limits alone break ipg right, and they still break it *with* rcond. The first
unregularised Newton step hits a bound, and the solve then crawls along it.
Both constants are left in 004c set to `None` with the table as a comment.

### `step = 1e-6` — rejected

004cc uses `step=1e-6` and appeared to converge better than 004c's `1e-7`,
which is what prompted trying it. It does converge — to penalty 2.9e-6 — while
putting βy at sdm1l.3 at **6306.7 m** and shifting μy(QD0→sdy1) by **+0.49**
(π → 2π). It converges *into* the null space, invisibly, because nothing in
004cc prints knob magnitudes. Left at 1e-7.

This is also the resolution of "why does 004cc run fine when 004c doesn't":
it does not. It was failing silently the whole time.

### `bety` at sdm1 as a target — worked as asked, wrong answer

`d2qy` went +682.6 → **−18863.9**, 28× worse and sign-flipped. And yet the
target itself did exactly what it was supposed to:

- βy at sdm1 held at bare values (22.6 m → 1.4704 m).
- Per-side μy errors improved ~10×, to −1.5e-5 … −4.6e-5 on **seven of eight**
  half straights.
- The eighth (ipg left) landed on a different branch: μy error −2.6e-3, trims
  7× bigger — while reporting **10/10 targets in tolerance at penalty 3.3e-6,
  the best-converged of all eight matches**. That one side accounts for
  essentially the whole ring number (phase-sum model: −68,600 vs measured
  −75,500, 9%).

Two lessons. First, penalty is not quality: the best-converged match produced
the worst lattice. Second, and less obvious — in the earlier run the left and
right phase errors had *opposite* signs (−3.7e-4, +3.9e-4) and largely
cancelled around the ring; with this target they were all negative and added.
Feeding seven good sides plus a hypothetically well-behaved ipg left through
the calibration still gives ≈ −1500, worse than +683 despite every individual
error being 10× smaller. **Part of the old "good" number was cancellation, not
correctness.** Watch the signed sum, not the individual magnitudes.

(This experiment predates the staged solve. It has not been re-run with
staging on, and staging is exactly the thing that fixed ipg left's branch
selection — so it is not a closed question, just an untried combination.)

## Would tightening tolerances / more staging cycles help further?

Analysed, not exhaustively measured. Short answer: probably not.

- **Tolerances: no.** The phase error is orthogonal to the target space.
  Evidence: ipg left and right finished at penalties within 2× of each other
  (8.56e-7, 4.88e-7), both 8/8 in tolerance, yet their μy errors differ by 17×.
  If phase error tracked target residual that could not happen. Plus the
  rcond-floor conflict above.
- **More back-and-forth: diminishing.** Staging works by changing the *starting
  point*, and with truncation you then sit at the min-norm point relative to it.
  A second cycle has nothing left to remove. Cheap to falsify (one extra solve
  per side) but expect a plateau, not a trend.
- **What would actually move it**, both untried:
  1. βy and Dx targets at **sdy1/sdy2** — the strength half of the −I
     condition (note 09's "where to go next" item 1). The current βy target is
     at sdm1, where β is 1.5–6.5 m: chromatically irrelevant, and it does
     nothing for the sdy sextupoles at βy ≈ 1e4 m.
  2. A **muy-only** phase target as a third stage. Note 09's failure targeted
     mux *and* muy; mux is uncontrollable with these knobs (0.05% of gradient
     survives), so including it poisons the solve.
  - Cheap decisive test before spending runs: take one Jacobian of
    μy(QD0→sdy1) against the 15 knobs and check whether the phase-correcting
    direction lies in the subspace `rcond=1e-6` **keeps** or the part it
    **truncates**. If truncated, rcond itself is the blocker and no amount of
    staging or tolerance work can help — go to option 1.

## Diagnostics added to 004c

- **`max|knob|` column** in the per-IP summary. The single most useful line in
  the log: a converged half-straight needs |k1| of a few 1e-4 to 3e-3, and
  anything far above means the solve wandered off in a weak direction rather
  than correcting anything. Null-space excursions are invisible in the penalty
  and obvious here.
- **`solve_and_report` counts only active targets.** `target_status()` lists
  disabled targets with `tol_met=False`, so without this every staged stage-1
  solve announces "2/8 targets outside tolerance" for the two it was
  deliberately not solving.
- **`--ips LIST`** runs a subset of IPs (~4 min instead of ~25). Sets
  `WRITE_OUTPUT_LATTICE = False` so a partial run prints
  `Partial run (--ips ...): NOT writing <file>` instead of overwriting the
  corrected JSON the downstream scripts read.
- **`--optics-step H` / `--optics-rcond R`** (0 disables truncation) for tuning
  the matching setup without editing the file.
- **Figure 4, Q(δ) with fit.** Uses the sweep `get_nonlinear_chromaticity`
  already ran, so it costs no extra twiss. Top row is Q vs δ; bottom row is
  `Q(δ) − (Q0 + Q'δ)`, which is the *only* view where the curvature is visible
  — over δ=±1e-3 the linear term moves Qy by ~5e-3 and the quadratic by ~3e-4,
  so the top row looks like a straight line however large Q'' is. Plotted from
  the same coefficients that produce the printed `d2q`, so the curve cannot
  disagree with the number.
  `report_nonlinear_chromaticity` now also returns `deltas`, `qx`, `qy` and the
  plain polynomial coefficients (`derivatives * n!`).

## Conventions (unchanged, repeated because they keep biting)

- True Q'' = d²Q/dδ². `detuning.py`'s `q*_derivatives[2]`, which 004c/004j print
  as `d2qx`/`d2qy`, is **Q''/4**. Verified against synthetic data: feeding in
  true Q''y = 4×264.98 recovers `d2qy = 264.98`. 004c's own docstring still says
  "index 2 is Q''/2" — that is wrong, and figure 4's legend spells out both.
- Bare 3T ring: `d2qy = −37.6591`, i.e. true Q''y ≈ −151.

## State of the files

- **004c** — has both fixes, the diagnostics above, and the staged solve. The
  `sext` targets are `betx`/`bety` at sdm1 (`dy` commented out by the user).
  Phase targets remain removed; `doublet_to_sdy1_range()` and the end-of-run
  phase report are deliberately **kept** as measurement.
- **004cc** — *(updated 2026-10-14)* The value-shift bug this note
  described (`betx=tw0['bety', sext]`, `bety=tw0['dpy', sext]`, i.e. βy targeted
  to 1.96e-12 m) is **gone**. The `sext` targets read `betx`/`bety`/`dy` at
  sdm1r correctly. No commit ever contained the bug (checked from `6cbf2d368`
  to `HEAD`), so it only ever lived in the uncommitted working tree on
  2026-09-16/17. For reference, the bare values there are betx 6.457586, bety
  1.471115, dy −1.473e-11, dpy 1.961e-12.
  Since 2026-09-21/22, 004cc runs the correction as **two independent chains**,
  each starting with every vary knob at zero: chain A has the `sext` targets
  off and uses `rcond=1e-6` on the optics solve; chain B has them on and uses
  no `rcond`. The knobs come from chain B, and chain A exists only for a
  comparison figure. All vary knobs still use `step=1e-6`. So chain B still has
  the no-`rcond` conditioning problem described above, and **W or phase numbers
  from 004cc may still come from an arbitrary null-space point**. See
  `09_correcting_the_q2y_source.md` ("What 004cc looks like now") for why
  004c's staged solve does not carry over.

## Outcome — the delivered lattice

Measured directly from `fccee_z_lcc_splineboris_solenoids_coupling_corrected_3T.json`
as it sits on disk (mtime 2026-09-16 19:11), via `set_lattice_knobs` + a
21-point sweep:

| | qx | qy | Q'x | Q'y | d2qx | d2qy | max βx | max βy |
|---|---|---|---|---|---|---|---|---|
| bare | 194.160148 | 170.199898 | 12.0836 | 5.2134 | 9.7453 | **−37.6591** | 3038.3 | 15167.6 |
| corrected | 194.160099 | 170.200098 | 11.9918 | 5.6321 | 7.0838 | **+1166.5014** | 7646.2 | 15162.9 |

That is true Q''y ≈ +4666, against ≈ −151 bare. Small enough that the momentum
acceptance study (009) ran on it and chromaticity was not the limiting factor,
which is where this chapter was closed.

Provenance: committed as `e4836d884` "Seems we have finally improved enough
for safe injection. Now an MA calculation.", together with the 004c that
contains both fixes — so this **is** the staged + truncated result. For
comparison along the way: +682.6 with `rcond` only (no staging), −18863.9 with
the `bety`-at-sdm1 target, and a broken ring (no closed orbit at δ=±1e-3) with
neither.

One caveat worth knowing before building on it:

- **max βx roughly 2.5× bare** (7646 vs 3038) in the corrected ring, while
  max βy is essentially unchanged. The ring closes and both tunes are right to
  6 digits, so this is not a broken lattice, but a horizontal beta blowup of
  that size in the corrected state has not been explained or checked against
  aperture anywhere in these notes. Worth a look if horizontal DA/MA ever
  disappoints.

The two untried options in the section above are the thread to pull if Q''
becomes limiting again.
