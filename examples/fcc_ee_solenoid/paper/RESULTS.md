# Results log: numbers the paper quotes

One row per number. Only rows with status **final** may appear in the paper. See "Production rules" in `../CLAUDE.md` for how rows get added.

**Status:** `final` = frozen pipeline, committed code · `prelim` = current code, not yet frozen · `stale` = older pipeline or solver settings, must be re-run.

**Case key:** `off` = solenoids + correctors off (`sb_off`) · `SB-2T`, `SB-3T` = SplineBoris analytic · `VS-2T`, `VS-3T` = VariableSolenoid (linear) · `MAP-2T` = BNL field map · suffix `-oN` = `--max-transverse-order N` (default 4)

## Summary table (Table 2 in the paper)

| Quantity | off | SB-2T | SB-3T | MAP-2T | Notes |
|---|---|---|---|---|---|
| \|C⁻\| corrected | 3.3e-15 (stale) | — | 7.88e-4 (stale) | — | uncorrected SB-3T: 4.36e-2 (stale) |
| εy eq. (Twiss) [m·rad] | 4.7e-15 (stale) | — | 1.32e-12 (stale) | — | state the emittance definition |
| εy eq. (tracking) [m·rad] | 3.9e-15 (stale) | — | 1.36e-12 (stale) | — | |
| Q″y (true d²Q/dδ²) | ≈ −151 (bare 3T ring, note 10) | — | 5.7e4 → 4.7e3 (stale, convention unverified) | — | 004c prints Q″/4; see notes 08–10 |
| DA | — | — | — | — | report in σ, not lost fraction |
| MA at injection | — | — | — | — | requirement: δ = 0.0095 |
| SR power from fringe fields [W] | n/a | — | — | — | benchmark against Boscolo et al. (2 T) |

## Full log

| ID | Quantity | Case | Value | Unit | Script + args | Commit | Output file | Seed | Date | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L01 | \|C⁻\| corrected | SB-3T | 7.88e-4 | – | (slides, 24 Sep) | f89a90b2c? | – | – | 2026-09-24 | stale | before the rcond/staging fix |
| L02 | \|C⁻\| correctors off | SB-3T | 4.36e-2 | – | (slides) | ? | – | – | 2026-09-24 | stale | ratio ≈ 55, not "two orders of magnitude" |
| L03 | εy eq. (Twiss) | SB-3T | 1.317e-12 | m·rad | (slides) | ? | – | – | 2026-09-24 | stale | |
| L04 | εy eq. (tracking fit) | SB-3T | 1.361e-12 | m·rad | (slides) | ? | – | ? | 2026-09-24 | stale | |
| L05 | εy eq. (Twiss) | off | 4.745e-15 | m·rad | (slides) | ? | – | – | 2026-09-24 | stale | |
| L06 | αy damping | SB-3T | 7.73e-4 | 1/turn | (slides) | ? | – | – | 2026-09-24 | stale | also the criterion for numerical damping |
| L07 | Q″y | SB-3T | 5.66e4 | – | (slides, Twiss direct) | ? | – | – | 2026-09-24 | stale | the notes report 1.5e4–1.8e4 for the 004c rings on 15 Sep, so this needs to be reconciled |
| L08 | Q″y after fix | SB-3T | 4.7e3 | – | (slides) | ? | – | – | 2026-09-24 | stale | |
| L09 | MA lost fraction | SB-3T before / after / off | 0.748 / 0.637 / 0.632 | – | (slides) | ? | – | ? | 2026-09-24 | stale | replace with δ-acceptance |
| L10 | dQ″y/dμ (QD0→sdy1 phase) | bare ring | 2.42e7 left / 2.48e7 right | per unit tune of phase error (true Q″) | trombone calibration (note 08) | ? | – | – | 2026-09-15 | prelim | candidate key figure; note 10 says it predicts the ring Q″y to ~0.2% |
| L11 | Q″y, ipg-only correction (Δ vs bare) | SB-3T | +302.64 (printed `d2qy`, = Q″/4) | – | 004c (note 10) | ? | – | – | 2026-09-17 | prelim | true ΔQ″y ≈ +1210, matches the calibration |
