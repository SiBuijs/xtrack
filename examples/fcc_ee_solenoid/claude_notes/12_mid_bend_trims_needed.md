# The mid-bend trims are needed, at 2T and at 3T (2026-10-15)

Answers the paper's "decisive test" (TODO §4a): with the conditioning fix of
note 10 and the coupling targets of note 11, can the correction work **without**
the mid-bend trim quadrupoles? **No, at either field.** So the paper cannot say
"no new hardware" for this correction scheme.

Runs: 004c `--b0 {2.0,3.0} --no-midbend-trims --output-tag notrims
--input-tag ""` at cfcf507f6 (everything else default: sdm1 βx/βy targets,
staged pass 1, rcond 1e-6, W-matrix coupling targets), then 004n. Lattices
`fccee_z_lcc_splineboris_solenoids_coupling_corrected_{2T,3T}_notrims.json`
(committed), logs in `data/logs_2026-10-15/004c_notrims_*.log`. Numbers in
paper/RESULTS.md L23–L26.

| | 2T trims | 2T no trims | 3T trims | 3T no trims |
|---|---|---|---|---|
| true Q″y | −1248 | **+10411** | +4668 | **+24998** |
| QD0→sdy1 μy error, left / right half | −9.7e-6 / −1.5e-6 | **+1.1e-4** / −1.2e-6 | +5.2e-5 / −3.4e-6 | **+2.7e-4** / −3.0e-6 |
| optics matches in tolerance | 8/8 | right 4/4, **left 0/4** (4/8 targets) | 8/8 | right 4/4, **left 0/4** |
| max \|k1\| quad trims [m⁻²] | 4.3e-5 | 1.5e-4 | 1.35e-4 | 4.1e-4 |
| \|C⁻\| | 4.4e-6 | 4.9e-6 | 7.4e-6 | 1.1e-5 |
| εy [m] | 7.220e-14 | 7.221e-14 | 1.2466e-12 | 1.2470e-12 |
| worst edge ΔDy [m] | 1.3e-7 | 1.7e-6 | 6.2e-7 | 1.1e-5 |
| 004c runtime | 3.5 min | 13.5 min | 3.5 min | 13.5 min |

- **The left half is the problem.** Without trims its optics match cannot meet
  the boundary + sdm1 targets together: it ends at Dpx 2.4e-6 / 4.4e-6 at the
  left edge (tol 1e-8), αx ~1e-6 (tol 1e-7), βx at sdm1 ~2e-5 m off (tol
  1e-5), after ~5300 model calls. The right halves converge as before.
- **Q″y is the same mechanism as notes 08/10.** The left-half QD0→sdy1 phase
  error grows ~10× (2T) / ~5× (3T), and the note-08 calibration (2.42e7 per
  unit phase error per side) predicts the ring ΔQ″y to 0.3% (2T) / 3% (3T,
  phase errors are printed to two digits only).
- **Coupling and εy do not care.** The coupling match is independent of the
  trims; its small degradation (some matches at 4/6) is the knock-on of the
  unconverged optics.
- Before this test the expectation (TODO, Repo audit) was that the trims might
  only be needed at 3T. That is wrong: 2T without trims has a larger |Q″y|
  than 3T with them.

Not done: MA on the `_notrims` lattices (013/009 cannot read tagged lattices
yet), and the "alternatives to the mid-bend quads" study of TODO §4a.
