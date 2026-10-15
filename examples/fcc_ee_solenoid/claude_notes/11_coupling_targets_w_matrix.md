# Coupling targets: from βx2/βy1 to the W matrix (2026-10-15)

Fourth note on the 004c correction, after 08–10 (the Q''y story). This one is
about the **coupling** match, and why the corrected rings leaked coupling into
the arcs although every coupling match reported "all targets in tolerance".

**Verdict up front.** The legacy edge targets βx2 = βy1 = αx2 = αy1 = 0 do not
constrain the coupling completely, and they constrain it only quadratically.
Replacing them with the four y/py components of mode 1 in the W matrix
(normalised, linear), and tightening the Dy/Dpy tolerances, reduced the leak
into the arcs by 9–10 orders of magnitude and brought everything at the IPs and
straight edges below 1e-6, with Q'' and the skew-knob sizes unchanged. This is
now 004c's default (`--coupling-targets wmatrix`); `beta` is kept as a legacy
option to reproduce older lattices.

All numbers below: closed-ring 4D twiss, solenoids + corrections on minus both
off, same lattice. Runs on HEAD `27b0bb9d2` plus the uncommitted 004c edits of
the day; logs and the test diffs are in `data/logs_2026-10-15/`, raw data in
`data/coupling_target_ab_2026-10-15/`, figures in
`~/cernbox/Pictures/FCC_Solenoid_Studies/Coupling_Target_Test_2026-10-15/`.

## Symptom

With the legacy targets: |C⁻| 3.9e-4 (2T) / 8.1e-4 (3T), βx2/βx in the arcs
2.9e-5 / 1.2e-4, and at the IPs and edges Δβ/β and Δα of 1e-5..1.7e-4. The
β/α errors were not optics errors: at every point Δβx/βx ≈ −βx2/βx and
Δβy/βy ≈ −βy1/βy, i.e. the main-mode β is reduced by exactly the share that
leaked into the other plane, and the size matches (|C⁻|/(2ΔQ))² with
ΔQ ≈ 0.04. **Dead end:** restoring the bare tunes with the arc families
(kqf2/kqd1, ~1e-7 relative) changed none of these numbers (ratios 0.98–1.03);
a pure tune error cannot move β/α anyway.

## Why the legacy targets miss it

Coupling at a point has 4 degrees of freedom. Write mode 1's eigenvector as
(x, x', y, y'):

- βy1 ∝ |y part|² — position only, and **squared**: a tolerance τ admits an
  amplitude √τ, and the Jacobian is flat at zero.
- αy1 ∝ (y part)·(y' part) — vanishes with the position part whatever the
  angle part is, so it adds nothing at the solution.
- Nothing constrains the angle part alone. The blind direction is exactly "a
  thin skew quadrupole at the edge marker": it mixes angles, not positions.

Measured on the open half-straight twiss the match itself uses (normalised):
mode 1 → y 3e-5..8e-5, mode 1 → y' **4e-4..7e-4**, with αy1 ~ 1e-8. The
vertical mode's x part was also 1e-4..4e-4 despite βx2 "in tolerance"
(the quadratic problem).

**Dead end:** adding γx2/γy1 (squared angle part) with tolerances scaled to an
amplitude of 1e-5. No coupling match converged (2/8 in tolerance each), runs
were ~3× slower, and the ring was worse than the legacy one (|C⁻| 6.3e-4 /
1.0e-3). Side observation: because it kept iterating, it drove the edge ΔDy
to ~1e-7, which showed the knobs can reach that.

## The fix

`coupling_edge_targets()` in 004c: the y and py rows of the two mode-1 columns
of W at the edge, in the bare normalised coordinates (ŷ = y/√βy,
p̂y = (αy y + βy py)/√βy), all four = 0 with `COUPLING_AMP_TOL = 1e-5`. Plus
dy/dpy at the edge with `COUPLING_DY_TOL = 1e-7`, `COUPLING_DPY_TOL = 1e-9`
(legacy 5e-5 / 1e-7).

| | 2T legacy | 2T new | 3T legacy | 3T new |
|---|---|---|---|---|
| βx2/βx in the arcs | 2.9e-5 | 1.2e-14 | 1.2e-4 | 2.1e-13 |
| \|C⁻\| | 3.9e-4 | 4.4e-6 | 8.1e-4 | 7.4e-6 |
| leak of the full straight (normalised map) | 8.7e-4 | 1.3e-8 | 1.8e-3 | 9.1e-8 |
| closed-ring Edwards–Teng R at the edges (normalised) | 5.1e-3 | 6.5e-8 | 1.0e-2 | 3.8e-7 |
| ΔDpy at the IPs | 9.3e-6 | 1.6e-7 | 9.2e-5 | 5.7e-7 |
| ΔDy at the edges [m] | 5.6e-6 | 1.3e-7 | 6.1e-5 | 6.2e-7 |
| Δβ/β at IPs / edges | 1.6e-5 / 2.7e-5 | 1.1e-7 / 1.1e-7 | 8.9e-5 / 1.2e-4 | 4.8e-8 / 7.8e-8 |
| εy (Chao, 6D) [m] | 8.91e-14 | 7.22e-14 | 1.320e-12 | 1.247e-12 |
| d2qy printed (= Q''y/4) | −312.01 | −312.04 | 1167.21 | 1167.12 |

All 16 coupling matches converge per case, max|k1s| ≤ 1.1e-6, runtime
unchanged (~4.5 min per case).

## Why W and not the Edwards–Teng matrix

ET R = 0 at the edge is the textbook form of the same condition, and was
tried (`edwards_teng` variant, normalised G_y R G_x⁻¹). The ring came out the
same as with W, but it left 1.5e-4 (2T) / 2.5e-4 (3T) of **angle** coupling
at the IPs, against ~1e-5 with W. Cause:

- The IP sits **inside the solenoid**, and SplineBoris tracks **kinetic**
  momenta (confirmed by the user; note 01's 004d section already records the
  same fact for W inside the compensation-solenoid bodies, where it causes
  NaN Edwards–Teng points). Maps that start or end at the IP are
  therefore not symplectic in these coordinates: max|MᵀJM − J| = 1.31e-2
  (2T) / 1.96e-2 (3T) for IP → 12 m, i.e. exactly ks = B0/Bρ; 0.41 / 0.61
  for 12 m → IP. Between two field-free points (12 m before → 12 m after)
  it is 3e-9 / 6e-9, so the integrator itself is fine (the end tapering
  works).
- init_ip is the bare, field-free Twiss. The W propagated from it is then
  non-symplectic at the edge by ~1e-4. xtrack's ET routine inverts W with the
  symplectic shortcut (`S.T @ W.T @ S`) and inherits that error. The mode-1
  components of W need no inverse.
- The same 1e-4 appears on the open twiss as an x/px part of mode 2 at the
  edge. It is an artifact of the start point, not edge coupling: the closed
  ring is clean at the edges (table above).
- At the IP itself, the remaining mode 2 → x' term of 1–2e-4 is the
  solenoid's ±(ks/2)(y, x) kinetic-momentum term (≈ 2·ks/2·√(βx*βy*), scales
  with B0) and cancels across the straight. It cannot be removed from
  outside the solenoid.

For the paper: describe the condition as "the Edwards–Teng coupling matrix
vanishes at the straight-section edges" and **quote the closed-ring ET at the
(field-free) edges**, which is valid there; mention the W implementation in
one sentence. "Uncoupled at the IP" is best stated for the position part
(coordinate-independent): the coupled β ratios at the IPs are 7.5e-13 (2T) /
6.0e-12 (3T).

Possible later cleanup (not needed for the results): start the half matches
from the bare Twiss converted into the solenoid's kinetic coordinates at the
IP (one shear with ks/2). Then W at the edges is symplectic and W and ET
become exactly equivalent.

## Diagnostics worth keeping

- **Leak of a half straight from its transfer map**: R from
  `tw.get_R_matrix(a, b)` (closed ring, line cycled mid-arc so no straight
  straddles s = 0), normalised with the bare W at both ends; the norm of the
  off-diagonal 2×2 blocks is unitless and linear in the coupling.
- **Coupled β ratios are blind to the angle part**, like the targets were.
  Check the W components (position and angle) or the normalised ET matrix.
- Penalty "in tolerance" proved nothing here, again (cf. note 10).
