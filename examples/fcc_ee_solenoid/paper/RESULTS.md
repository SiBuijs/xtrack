# Results log: numbers the paper quotes

One row per number. Only rows with status **final** may appear in the paper. See "Production rules" in `../CLAUDE.md` for how rows get added.

**Status:** `final` = frozen pipeline, committed code · `prelim` = current code, not yet frozen · `stale` = older pipeline or solver settings, must be re-run.

**Case key:** `off` = solenoids + correctors off (`sb_off`) · `SB-2T`, `SB-3T` = SplineBoris analytic · `VS-2T`, `VS-3T` = VariableSolenoid (linear) · `MAP-2T` = BNL field map · suffix `-oN` = `--max-transverse-order N` (default 4)

## Summary table (Table 2 in the paper)

| Quantity | off | SB-2T | SB-3T | MAP-2T | Notes |
|---|---|---|---|---|---|
| \|C⁻\| corrected | 4.5e-17 (prelim, L12) | 4.41e-6 (prelim, L13) | 7.43e-6 (prelim, L14) | — | uncorrected SB-2T: 0.150 (L15); uncorrected SB-3T has no closed orbit (L16) |
| Edwards–Teng coupling at the straight edges (normalised, max of 8) | 0 | 6.5e-8 (prelim, L17) | 3.8e-7 (prelim, L17) | — | the paper's statement of "no leak"; note 11 |
| εy eq. (Twiss, Chao 6D eigen-emittance) [m] | 4.745e-15 (prelim, L18) | 7.22e-14 (prelim, L18) | 1.247e-12 (prelim, L18) | — | normal-mode emittance from the 6D radiative twiss |
| εy eq. (tracking) [m·rad] | 3.9e-15 (stale) | — | 1.36e-12 (stale) | — | |
| Q′x / Q′y | 12.084 / 5.213 (final, L19) | 12.053 / 5.404 (final, L19) | 11.997 / 5.636 (final, L19) | — | 21-point fit, δ = ±1e-3 |
| Q″x / Q″y (true d²Q/dδ²) | 39.0 / −150.6 (final, L20) | 34.5 / −1248.2 (final, L20) | 29.1 / +4668.5 (final, L20) | — | = 4 × printed d2q (note 10) |
| DA | — | — | — | — | report in σ, not lost fraction |
| MA at injection | — | — | — | — | requirement: δ = 0.0095 |
| SR power from fringe fields [W] | n/a | — | — | — | benchmark against Boscolo et al. (2 T) |

## Full log

| ID | Quantity | Case | Value | Unit | Script + args | Commit | Output file | Seed | Date | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L01 | \|C⁻\| corrected | SB-3T | 7.88e-4 | – | (slides, 24 Sep) | 59d8b1986 | – | – | 2026-09-24 | stale | reproduced exactly from that commit's lattice on 2026-10-15; superseded by L14 |
| L02 | \|C⁻\| correctors off | SB-3T | 4.36e-2 | – | (slides) | ? | – | – | 2026-09-24 | stale | ratio ≈ 55, not "two orders of magnitude" |
| L03 | εy eq. (Twiss) | SB-3T | 1.317e-12 | m·rad | (slides) | 59d8b1986 | – | – | 2026-09-24 | stale | reproduced (Chao, 1.3167e-12) from that commit's lattice on 2026-10-15; superseded by L18 |
| L04 | εy eq. (tracking fit) | SB-3T | 1.361e-12 | m·rad | (slides) | ? | – | ? | 2026-09-24 | stale | |
| L05 | εy eq. (Twiss) | off | 4.745e-15 | m·rad | (slides) | ? | – | – | 2026-09-24 | stale | |
| L06 | αy damping | SB-3T | 7.73e-4 | 1/turn | (slides) | ? | – | – | 2026-09-24 | stale | also the criterion for numerical damping |
| L07 | Q″y | SB-3T | 5.66e4 | – | (slides, Twiss direct) | ? | – | – | 2026-09-24 | stale | the notes report 1.5e4–1.8e4 for the 004c rings on 15 Sep, so this needs to be reconciled |
| L08 | Q″y after fix | SB-3T | 4.7e3 | – | (slides) | ? | – | – | 2026-09-24 | stale | |
| L09 | MA lost fraction | SB-3T before / after / off | 0.748 / 0.637 / 0.632 | – | (slides) | ? | – | ? | 2026-09-24 | stale | replace with δ-acceptance |
| L10 | dQ″y/dμ (QD0→sdy1 phase) | bare ring | 2.42e7 left / 2.48e7 right | per unit tune of phase error (true Q″) | trombone calibration (note 08) | ? | – | – | 2026-09-15 | prelim | candidate key figure; note 10 says it predicts the ring Q″y to ~0.2% |
| L11 | Q″y, ipg-only correction (Δ vs bare) | SB-3T | +302.64 (printed `d2qy`, = Q″/4) | – | 004c (note 10) | ? | – | – | 2026-09-17 | prelim | true ΔQ″y ≈ +1210, matches the calibration |
| L12 | \|C⁻\| | off | 4.47e-17 | – | 4D twiss of the 3T (and 2T) corrected lattice, solenoids + correctors off; scratch `measure_fom.py` (uncommitted) | e749d4795 | `fccee_z_lcc_splineboris_solenoids_coupling_corrected_{2T,3T}.json`; log `data/logs_2026-10-15/baseline_fom_{2T,3T}.log` | n/a | 2026-10-15 | prelim | prelim only because the measurement script is not committed |
| L13 | \|C⁻\| corrected | SB-2T | 4.41e-6 | – | 004a/004b `--b0 2.0` (at 27b0bb9d2, code unchanged since) → 004c `--b0 2.0` (defaults) → `measure_fom.py` | e749d4795 | `fccee_z_lcc_splineboris_solenoids_coupling_corrected_2T.json`; logs `data/logs_2026-10-15/baseline_004c_2T.log`, `baseline_fom_2T.log` | n/a | 2026-10-15 | prelim | first run with the W-matrix coupling targets (note 11); legacy targets gave 3.87e-4 |
| L14 | \|C⁻\| corrected | SB-3T | 7.43e-6 | – | as L13 with `--b0 3.0` | e749d4795 | `…_corrected_3T.json`; `baseline_004c_3T.log`, `baseline_fom_3T.log` | n/a | 2026-10-15 | prelim | supersedes L01 (7.88e-4, legacy targets; 8.07e-4 when re-run this morning at 27b0bb9d2) |
| L15 | \|C⁻\| correctors off | SB-2T | 0.150 | – | as L13, solenoids on, all correction knobs off (`set_lattice_knobs(with_correctors=False)`: also compensation solenoids and doublet rotation off) | e749d4795 | `baseline_fom_2T.log` | n/a | 2026-10-15 | prelim | not the slides' definition of "uncorrected" (L02); define before quoting |
| L16 | \|C⁻\| correctors off | SB-3T | no closed orbit | – | as L15 at 3T | e749d4795 | `baseline_fom_3T.log` | n/a | 2026-10-15 | prelim | L02 (4.36e-2) is not reproducible with this definition; it presumably kept some correction on |
| L17 | Edwards–Teng coupling matrix at the straight edges, normalised G_y R G_x⁻¹ with the bare Floquet matrices, max over the 8 edges | SB-2T / SB-3T | 6.45e-8 / 3.77e-7 | – | closed-ring `twiss4d(coupling_edw_teng=True)`; scratch `et_edges.py` (uncommitted) | e749d4795 | `data/logs_2026-10-15/baseline_et_{2T,3T}.log` | n/a | 2026-10-15 | prelim | legacy targets: 5.1e-3 / 1.0e-2. Edges are field-free, so ET is valid there (note 11) |
| L18 | εy equilibrium, Chao (6D radiative twiss, `configure_radiation('mean')` + `compensate_radiation_energy_loss`, recipe of 004j) | off / SB-2T / SB-3T | 4.745e-15 / 7.222e-14 / 1.2468e-12 | m | `measure_fom.py` | e749d4795 | `baseline_fom_{2T,3T}.log` | n/a | 2026-10-15 | prelim | εx 7.196e-10 / 7.191e-10 (2T/3T), 7.199e-10 off. A second evaluation with the line cycled mid-arc gives 7.220e-14 / 1.2466e-12 (0.03% spread). Legacy targets: 8.91e-14 / 1.320e-12 |
| L19 | Q′x / Q′y (fit) | off / SB-2T / SB-3T | 12.0836 / 5.2134; 12.0534 / 5.4037; 11.9967 / 5.6362 | – | 004c's own end-of-run report, 21 points, δ = ±1e-3 | e749d4795 | `baseline_004c_{2T,3T}.log` | n/a | 2026-10-15 | final | twiss `dqx/dqy`: 12.081 / 5.431 (2T), 12.024 / 5.666 (3T) |
| L20 | Q″x / Q″y, true d²Q/dδ² (= 4 × printed d2q) | off / SB-2T / SB-3T | 38.98 / −150.64; 34.53 / −1248.17; 29.10 / +4668.49 | – | 004c's report, printed d2qx/d2qy: 9.7453 / −37.6591; 8.6336 / −312.0419; 7.2758 / 1167.1217 | e749d4795 | `baseline_004c_{2T,3T}.log` | n/a | 2026-10-15 | final | twiss `ddqy`: −148.6 / −1244.6 / +4695.9 (consistent; supersedes L07/L08). Unchanged by the coupling-target change to 4 digits |
| L21 | IP and straight-edge match, corrected − bare, worst over 4 IPs / 8 edges: ΔDy, ΔDpy, Δβ/β, Δα, coupled β ratio, Δ(x,y) | SB-2T | IP: 4.7e-11 m, 1.6e-7, 1.1e-7, 2.4e-7, 7.5e-13, 8.3e-12 m; edge: 1.3e-7 m, 3.3e-10, 1.1e-7, 1.9e-7, 8.0e-15, 7.0e-10 m | – | scratch `collect_ab.py` (uncommitted) | e749d4795 | `data/coupling_target_ab_2026-10-15/baseline_2T.npz` | n/a | 2026-10-15 | prelim | target was < 1e-5 everywhere (met) |
| L22 | as L21 | SB-3T | IP: 2.2e-10 m, 5.7e-7, 4.8e-8, 2.2e-6, 6.0e-12, 7.7e-11 m; edge: 6.2e-7 m, 1.8e-9, 7.9e-8, 1.8e-7, 1.8e-13, 6.7e-9 m | – | as L21 | e749d4795 | `…/baseline_3T.npz` | n/a | 2026-10-15 | prelim | as L21 |
