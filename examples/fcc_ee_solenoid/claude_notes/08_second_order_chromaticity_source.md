# Source of the large vertical second-order chromaticity after 004c

Written 2026-09-15. Companion to `00_overview.md`.

> **Status 2026-09-17.** The *diagnosis* in this note holds and was confirmed
> to 0.2% — see note 10. The 2.42e7 per-side calibration below is the single
> most useful number in the series. Its **"Fix" section is superseded twice
> over**: note 09 shows the phase-target fix fails, and note 10 shows the real
> problem was conditioning of the match (rank deficiency + no SVD truncation),
> not the choice of targets. Read 10 first if you are here to fix something.

The investigation was done with scratch scripts that were not committed; the
key recipes are reproduced below. It applies to the corrected lattices that
004c wrote at 21:11 (3T) and 21:18 (2T) that day. In that version the optics
match targets `betx`, `dpy`, `dy` at the `sdm1` sextupole; the label still
says betx/bety.

## Conventions

- **Q''** means the true derivative d²Q/dδ². `detuning.py`'s
  `q*_derivatives[2]` stores Q''/4, and that is what 004c/004j print as
  `d2qy`.
- **Quick measurement:** a 5-point stencil over δ = ±h, ±2h with
  h = 2.5e-4. Use `twiss4d(delta0=δ)` with `co_guess` = δ times the
  on-momentum dispersion at the start. It takes ~8 s on the full ring and
  agrees with the 21-point fit to <1%.

## Numbers

| lattice | Q'y | Q''y |
|---|---|---|
| bare (solenoids and corrections off) | 5.24 | −149 |
| 3T corrected | 5.41 | **+18,497** |
| 2T corrected | — | **+15,419** |
| older 3T lattice (004dd cache, 19:18) | — | −9,355 |

The older lattice had the opposite sign. The sign depends on the correction
run; see "Why the phase error appears" below.

The effect is additive over IPs. Turning on the solenoid and correction at
one IP only gives this ΔQ''y for 3T:

| IP | ΔQ''y |
|---|---|
| ipa | +873 |
| ipd | +2,250 |
| ipg | −151 |
| ipj | +15,855 |

These are knob toggles at ipd, using `on_sol_{ip}` and `on_sol_corr_{ip}`
plus the child knobs:
- **Coupling correction off, or orbit correction off:** Q''y unchanged.
- **Optics correction off (both sides):** Q''y = −67, close to bare, even
  though Q'y = 7.74.
- **Solenoid + orbit + coupling correction, no optics correction:** at any
  IP, Q''y = −65 to −89.

So neither the solenoid field nor the coupling or orbit correction causes
it. **The optics re-match does.**

Pitfall: "corrections on, solenoid off" is NOT a meaningful state. The orbit
correctors then create a large orbit through the IR sextupoles, and the optics
is wrecked (βy at sdy drops to ~1300 m, Dx reaches ~10 m). Don't use it.

## IR structure that matters (same at all four IPs, 3T bare)

- **Final-focus (FF) doublet:** QD0 (qd0a/b/c) and QF1 (qf1a–d), at 2.4–12 m
  from the IP. βy goes up to ~15,000 m, and μy ≈ ±0.25 from the IP.
  - qf1c and qf1d have k1 = 0 in the bare lattice; the optics trims switch
    them on.
- **Vertical local-chromatic-correction (LCC) sextupoles:** a −I pair on each
  side at βy = 10,370 m.
  - Left: `sdy1l` (−371 m, μy −0.75) and `sdy2l` (−728 m, μy −1.25),
    Dx ≈ 0.28 m.
  - Right: `sdy1r` (+179 m, +0.75) and `sdy2r` (+440 m, +1.25),
    Dx ≈ 0.39 m.
  - Each family has 4 slices of 0.15 m.
  - Chromatic kick per side: q_S = −Σ βy·k2l·Dx ≈ +5,400 (left) and
    +5,560 (right). This cancels the FF's q_F = Σ βy·k1L ≈ −5,500.
- **Horizontal LCC:** `sfx1l`/`sfx2l` at βx = 1,500 m, plus crab
  sextupoles `scrab*`. They were not involved here.
- **`sdm1l`/`sdm1r`** (the `sext_left`/`sext_right` targets in 004c) sit
  at β ≈ 1.4–6.5 m, so q ≈ 0.1. **They are chromatically irrelevant.**
  - Their βy changes after correction (up to +92% at ipg right) do not
    track Q''y.
  - The 004c mid-bend comment blaming βy at sdm1r for Q'' described a
    symptom, not the cause.
- **Compensation solenoids** sit at 12–14 m, right after QF1. That is the
  same phase as the FF, so to first order they do not change the QD0→sdy1
  phase (measured ~3e-7).
- Between the FF and sdy, W_y (`wy_chrom`) is about 4,000 by design. In the
  bare arcs it is about 4.5 rms.

## Mechanism

The vertical LCC works only if the phase advance from QD0 to sdy1 is exactly
0.5 (π), with sdy2 a further 0.5 on. The FF chromatic β-beat then has a node
at the sextupoles.

With a phase error Δμ (in tune units) on one side, second-order perturbation
theory in the chromatic gradient gives:
- **Pair term:** the double-sum term of the FF/sextupole pair produces
  **ΔQ'' ≈ q_F·q_S·Δμ ≈ q²·Δμ**.
- **Linear in Δμ, with its sign.** A magnitude mismatch between FF and
  sextupoles enters Q'' only quadratically, though it enters Q' linearly.
  This is why the optics-off state changes Q' by +2.5 but leaves Q'' alone.
- **Local.** There is no resonance denominator, so the working point does not
  matter.
- **Per side.** Summed over all pairs (FF_L, FF_R, S_L, S_R), the FF_L–FF_R
  phase across the IP cancels, leaving q²·(e_L + e_R).

In the user's sextupole picture this is still a sextupole δ² term: the gradient
k2·Dx·δ is weighted by βy(δ). The second δ enters through the chromatic β-beat
at the sextupole, not through Dy² or second-order dispersion. (1/β)∂β/∂δ
(`by_chrom`) at sdy goes from 1.5 in the bare ring to ±25 at ipj and 7–11 at
ipd.

## Evidence

Calibrated coefficient, measured with trombones in the bare ring (recipe
below): **dQ''y/dμ = 2.42e7 on the left, 2.48e7 on the right**, per unit
tune of phase error. The thin-lens estimate q² ≈ 3e7 is about 20% high.

Measured 3T errors in the QD0→sdy1 phase (corrected − bare), with the
predictions from the calibrated coefficient:

| IP | left | right | predicted ΔQ''y | measured |
|---|---|---|---|---|
| ipa | +7.8e-5 | −4.0e-5 | +910 | +873 |
| ipd | +1.88e-4 | −9.1e-5 | +2,296 | +2,250 |
| ipg | +1.07e-4 | −1.07e-4 | −61 | −151 |
| ipj | **+7.13e-4** | −6.1e-5 | +15,736 | +15,855 |

- **2T** (same coefficient): predicted 15,083, measured 15,419. The
  dominant sides are ipj left (+4.0e-4) and ipd left (+2.6e-4).
- **Causal test (3T corrected):** set trombones to minus the measured errors
  on all 8 sides.
  - Q''y goes from 18,497 to **−100**, and Q'y stays at 5.409.
  - With only ipj's two trombones set, Q''y = 2,987, which is the sum of the
    other IPs.
  - W_y in the arcs goes from rms 38 back to 5.

### Phase-reference pitfall

Do NOT measure "IP → sdy1". With βy* = 0.8 mm, μy at the IP marker jumps by
~6e-4 for a waist shift of a few µm.
- That jump shows up with opposite signs on the two sides.
- Per-IP sums are unaffected, but the left/right attribution is wrong. For
  ipj, "IP→sdy1" wrongly puts the error mostly on the right.

Reference the phase to the FF instead:
- **Right side:** `mu(sdy1r slices mean) - mu('qd0ar.N')`.
- **Left side:** `mu('sol_slice_{ip}_000') - mu(sdy1l slices mean)`. That
  element is at the QD0L exit, s = −2.399 m.
- Take both modulo qy. The bare value is 0.5000 in each case.

## Why the phase error appears

The 004c half-straight optics match has 15 knobs: 12 quad trims plus 3
mid-bend thin quads. It has 9 targets:
- β, α, Dx, Dpx at the straight edge;
- βx, Dy, Dpy at sdm1.

No quads are trimmed between sdy1 and the straight edge, so β and α at sdy are
restored (βy to <0.1%). The phase advance QD0→sdy1, however, is a null-space
direction, and the result depends on the solver path. That explains the sign
flip between runs.

At ipj left the solution is extreme:
- **FF trims** with βy·Δk1L up to about ±30 that mostly cancel: qf1al.2
  −3.9%, qd0cl.2 −1.8%, and qf1cl.2/qf1dl.2 switched on.
- **Mid-bend trim** `qbmid_b0bl.2` with βy·k1l ≈ −2.1.
- **Local phase excursion:** the error reaches −0.10 at sdm1l.5 (βy +47%)
  and comes back to +7.1e-4 at sdy1l.

### Attempted fix (implemented 2026-09-15) — FAILED, see note 09

**2026-09-16: this does not work. Do not restore it without reading
`09_correcting_the_q2y_source.md` first.** Two measured reasons: (1) the `mux`
half of it is uncontrollable by these knobs (0.05% of its gradient survives the
other targets), so the match can never converge; (2) even the `muy` half, when
achieved in isolation, makes the W leak ~4x *worse*, because the knob direction
that fixes the phase drifts βy/Dx at the sdy sextupoles and so breaks the
sextupole *strength* half of the −I cancellation. The description below is kept
as the record of what was tried.

004c's optics match now has
`xt.TargetRelPhaseAdvance(mu, bare_value, start, end, tol=1e-6, weight=10)`
for mux and muy.
- **Range:** QD0R entrance → first sdy1r slice on the right, first sdy1l
  slice → QD0L exit on the left. It comes from `doublet_to_sdy1_range()`;
  the element after `qd0al.N` is the marker `bpm_qd0al.N`.
- **Target value:** the bare ring value, `np.mod(tw0[mu, end] - tw0[mu, start], q)`.
  It is ≈ 0.5000 for muy (0.49999 right, 0.50001 left) and ≈ 0.49 for mux.
- **Checked beforehand:**
  - Evaluated on the open half-straight twiss from the IP init (backward on
    the left), the target reproduces the ring value to 1e-12.
  - weight=10 is passed explicitly, because a callable target otherwise gets
    weight 1.
- **Match size:** 15 knobs against 11 targets now.
- **New printout:** after correction, 004c prints the corrected − bare
  QD0↔sdy1 phase for all 8 sides. 1e-6 in muy is worth ~25 in Q''y.
- If a future session reads this: check the 004c log and Q''y from that run
  before assuming the fix worked.

## How the 004k and 004dd outputs relate

- **004k:** the corrected Qy(δ) is a steep upward parabola. It reaches 170.5
  near |δ| ≈ 5e-3 at 3T, which explains the NaN gaps.
- **004dd:** μy''(s) shows where the tune shift is *realised*, not its
  source. Corrected − bare, accumulated between straight edges:

| segment | ΔQ''y |
|---|---|
| each of the 4 arcs | +2,800 to +4,100 |
| each straight | +700 to +1,500 |

- Inside each arc cell μy'' oscillates by thousands. This is a bounded
  oscillation; the secular trend is what counts.
- The jumps at IP markers and at sdm1 (low β) are artefacts.
- With the trombones, every segment is below 30. So the arc accumulation is
  the W_y leaking out of the IRs beating around the ring.
- The older cache shows the same pattern: −1,718 per arc and −583 per
  straight.
- **Quick diagnostic:** `line.twiss4d(chrom=True).wy_chrom` in the arcs.
  About 4.5 rms is healthy; about 40 means the LCC phase is broken somewhere.

## Recipes

Pure vertical phase trombone. `LineSegmentMap` with `qy` only; with `qy=0`
it is the identity, so insert it once and set its fields later. The kernel
reads `qy`, `betx[0/1]` etc. at track time.

```python
env.elements[name] = xt.LineSegmentMap(qx=0.0, qy=0.0, betx=1.0, bety=1.0)
line.insert([env.place(name, at=(s_ip + sign * 20.0) % C)])  # drift between comp sol and bends

tw = line.twiss4d()                         # in the state being tested
el = line[name]
for a in ('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx', 'dy', 'dpy'):
    getattr(el, a)[0] = getattr(el, a)[1] = tw[a, name]
for a, c in (('x_ref', 'x'), ('px_ref', 'px'), ('y_ref', 'y'), ('py_ref', 'py')):
    getattr(el, a)[0] = getattr(el, a)[1] = tw[c, name]
el.qy = dmu        # adds dmu to the phase from QD0 to sdy1 (right) / sdy1 to QD0 (left)
```

- **Placement:** ±20 m from the IP is past the compensation solenoid, where
  Dx ≈ 1e-4 and residual coupling is small (betx2 ≈ 0.1 m).
- **Per-IP toggles:** set `on_sol_{ip}` and `on_sol_corr_{ip}` for the
  chosen IP and zero them for the others. The saved JSON keeps the knob chain
  `on_sol_corr_{ip}` → `on_sol_{kind}_corr_{ip}` →
  `on_sol_{kind}_corr_{side}_{ip}`.
  - `set_lattice_knobs` overwrites those expressions with floats. After
    toggling a child knob by hand, restore it with
    `line.vars[k] = line.vars[parent]`.
