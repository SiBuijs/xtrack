# 17 — IDEA main + screening solenoid from the Opera on-axis curves (2026-11-15)

Scripts: `001c_plot_IDEA_solenoid_beam_axis_raw.py`,
`001d_construct_main_and_screening_solenoid.py`. Data:
`IDEA_solenoid_beam_axis_raw.json`. All committed by the user in `e37aad453`
/ `ca07d6615`. Digitizer: `17_scripts/` (see below).

**Exploratory, not part of the paper's frozen pipeline.** No paper numbers.
The BNL map is out of scope for now (user, 2026-11-15).

## Goal

Build the IR solenoid elements (BFieldExpansion / SplineBoris) from the IDEA
main and screening solenoid fields. This replaces the crude analytic coil
models used so far (`TiltedSolenoid` sheets in 004a, 009/009a and 010a). The
case the user wants to study is the screening starting at **±1.40 m**, the
IDEA field's own edge.

## Data: `IDEA_solenoid_beam_axis_raw.json`

- **Source.** A screenshot of an Opera plot "Bz (T) vs Z (m)", z = 0–6 m, one
  side of the IP, 2 T at the IP. It has three curves: Main Solenoid Only
  (green), Screening Only (red) and Both Energized (black). The user provided
  it on 2026-11-15. The screenshot is **not stored** in the repo; ask the user
  to re-digitize.
- **Format.** Per curve: `z`, `Bz`, `sigma_Bz` and `flag` (`visible`,
  `partial`, `hidden_under_<curve>`, `under_logo`). It also holds the
  calibration (residuals ≤ 0.7 px; 1 px = 3.4 mm in z, 2.2 mT in Bz) and
  notes. The data are raw: one point per pixel column, one per pixel row in
  steep parts, so **z repeats there**. Merge repeated z before `np.interp`.
- **Digitizer.** `17_scripts/17a_digitize_idea_opera_plot.py <png> <out>`,
  then `17b_build_idea_json.py <out> IDEA_solenoid_beam_axis_raw.json`. It
  uses colour masks, separates the black curve from the (black) gridlines by
  thickness, and handles occlusion with a band of known thickness.
  - Fixed bug (2026-11-15): a fully hidden point must lie within
    |Δt|/2 + 1 px of the curve covering it. Before the fix, 3 green points
    at 5.32–5.33 m took red's value.
- **Facts established:**
  - Screening is plotted **positive**. Both = Main − Screening_plotted to 5 mT
    rms (with a + sign it is off by 1.7 T rms), so the screening field is −S.
  - **Not exactly additive.** G − S − B averages −8.7 mT on 1.85–2.6 m. The
    Opera model has iron, and the user says the sum need not be exact.
  - Green is hidden under red at 1.9–4.1 m, so its values there are red's,
    ±1 px.
  - The Opera logo covers z ≥ 5.735 m. B and S end at 5.73 m.
  - The ripples in B beyond ~2.05 m are in S. At 4.1–5.0 m, S + B reproduces
    G to ~1 mT. The screening overshoots the main field (S/G up to 1.25 at
    4.45 m) and stops cancelling it at ~4.75 m (S/G back to 0.5), so B dips
    to −25 mT at 4.45 m and rises to +35 mT at 4.75 m.

## 001c

001c plots the raw data. Lines join the digitized points; open circles mark
points that were not fully visible in the screenshot. The bottom panel
zooms in, with a ±`sigma_Bz` band.

## 001d steps

1. **Ripple removal (idealized screening, the user's choice).** B is cleaned
   beyond `Z_CUT = 2.05` m: an exponential tail fitted on 1.70–2.05 m
   (λ = 13.0 cm), with a cosine blend over 2.00–2.05 m. Then S := G − B_clean.
   - This is option 2 of the user's ranking. Cleaning B is well defined,
     because the ripple sits on a ~0 baseline.
   - It also removes the screening's end, so the cleaned S cancels G out
     to 6 m.
   - ∫B beyond 2.05 m drops from 15.3 to 0.6 mT·m; ∫B over 0–5.73 m goes
     from 2.751 to 2.736 T·m.
2. **Spline fits with `LongitudinalFitter`** (`xtrack/_temp/splineboris/`):
   component `("Bs", 0)`, data mirrored about the IP, ±6 m, `end_condition=
   "zero"`, `preserve_integral=True`.
   - **G**, raw points, **0.2 m elements**: 0.65 mT rms at z < 1.2 m, 1.1 mT
     at 1.7–5.73 m, 5.6 mT under the logo; |Bs''''| ≤ 74 T/m⁴. At 0.1 m the
     spline fits the digitization noise (Bs'''' ±600 T/m⁴); at 0.3 m it
     biases the fall-off near 3 m.
   - **S**, cleaned S on the 1 mm grid, **0.1 m elements** (set by the user):
     at the edge 7 mT rms (20 max), at z < 1.2 m 1.2 mT rms (4.4 max). With
     0.2 m it misses the edge by up to 71 mT and rings back to the IP
     (41 mT). Below 0.1 m the last element (5.95–6.0 m) has no data and the
     fit fails.
   - **Analytic fits of G fail** because the iron makes G fall off faster than
     any air coil can:
     - one `SolenoidField`: ±35 mT (B0 2.042 T, L 5.782 m, a 0.795 m);
     - a thick coil collapses to a sheet;
     - two sheets: 15–17 mT;
     - fitting z ≤ 2 m only: 0.9 mT there, but 0.5–0.8 T off at 3–4 m;
     - a chain of 28 coaxial 0.25 m coils (a = 0.3–0.6 m, linear least
       squares) fits to 0.7 mT. Tested, not implemented.
3. **Beam-frame coefficients** A_m = ∂ₓ^{m−1}B_x at x = y = 0, with A_0 = B_s,
   to O(θ²). The formulas were derived by the user, not re-checked here (see
   notes 15–16 for the A_m formalism):
   - A_0 = b − θ²(S b/2 + S² b′/4)
   - A_1 = θ(b + S b′/2)
   - A_2 = −b′/2 + θ²(3/2 b′ + 5/4 S b″ + 3/16 S² b‴)
   - A_3 = −θ(3/2 b″ + 3/8 S b‴)
   - A_4 = 3/8 b‴ − θ²(15/4 b‴ + 27/16 S b⁗)

   `TILT_ANGLE = 0.015`. The b⁽⁵⁾ term is **dropped** (user's choice): it is
   zero inside quartic elements, and would be ≲ 40 T/m³, ~5 % of A_4.
   θ = 0 reproduces the solenoid-frame a_m exactly. Total = main − screening.
   Max |A_m| of the total: A_0 1.994 T, A_1 0.077 T, A_2 4.48 T/m,
   A_3 15.8 T/m², A_4 769 T/m³.

Figures (`PLOT_DIR`): `IDEA_solenoid_beam_axis_raw.pdf` (001c),
`IDEA_solenoid_beam_axis_ripple_removal.pdf`,
`IDEA_solenoid_beam_axis_spline_fits.pdf` and
`IDEA_solenoid_beam_frame_coefficients.pdf` (001d).

## Edge positions (50 % points of the fits)

| | 50 % point |
|---|---|
| Main | ±2.89 m (effective length 5.79 m; same as the one-sheet fit's half-length) |
| Screening start | ±1.40 m (half its max: 1.397 m; S/G = 0.5: 1.403 m) |
| Total | ±1.397 m, so the beam-visible edge is set by the screening |
| Raw screening end | 4.75 m. The idealized S has no end of its own |

For comparison:
- **010a:** screening coils start at `sol_half_length = 1.3` m (local coils
  1.3–1.9 m). The total's 50 % point is at 1.23–1.24 m, for
  `B0_local = −2.3685` or −2.7685 T × B0/3. First corrector in 010a: 1.78–1.965
  and 2.015–2.2 m.
- **`solenoid_params.py`:** 3 T has half-length 1.30 m and first corrector
  1.40–2.29 m; 2 T has 1.23 m and 1.23–2.29 m.

## Comparison with the user's "Bz_cs / Bx_cs" figure (origin not recorded)

- **Same physics.** A_1 = θ(b + S b′/2) with θ = −0.015, applied to its own
  Bz, reproduces its Bx to 1.0 mT rms.
- **Sign.** Its Bx matches **θ_b = −0.015** in the A_m formulas. 001d uses
  +0.015, so its A_1 and A_3 have the opposite sign.
  - **Corrected 2026-11-15:** θ_b is not `THETA`. θ_b is the solenoid
    axis' angle from the beam (B_x = +θ_b B_s at the IP). `THETA`
    (`TiltedSolenoid`, 004a, 010a) is the beam's angle from the solenoid
    axis, B_x = −sin(`THETA`) B_s.
  - So the ring's `THETA = −0.015` means θ_b = +0.015, which is what
    001d/001e use. The 004a 2 T template has B_x(IP) = +0.030 T and a
    negative edge peak; IDEA's B_x has the same signs.
  - This figure therefore has the opposite tilt to every ring model.
  - 001e now sets `TILT_ANGLE = −THETA` explicitly (in
    `idea_solenoid_field.py`).
- **Edge.** Its 50 % point is at ±1.23 m (= the 2 T half-length in
  `solenoid_params.py`), against 1.40 m for ours. Max slope 7.5–7.8 T/m
  against 8.6 T/m for ours.
- **Bx edge peak.** +57.8 mT at 1.24 m, against 77.5 mT at 1.41 m for ours.
- **Tail.** 10–25 mT at 2.0–2.2 m (uncertain: the line is under the By line
  there), against ~3 mT for ours.

## 001e: moving the screening edge (2026-11-15)

`001e_vary_screening_length.py` is standalone (001d's loading and step 1 are
copied, by the user's choice). It rebuilds S for edges (S/G = 0.5) at
`SCREENING_EDGES = 1.20…1.60 m`, on a single 0.1 m grid for S and the total.
The method is in its docstring. Key points:
- **Move the fraction S/G, not S.** Past its edge S follows G, which falls by
  up to ~1 T/m at 3–4 m. Shifting S itself by 0.2 m leaves up to 0.28 T
  uncancelled.
- **Symmetric pair of edges.** S/G = ρ(z − z_e) + ρ(−z − z_e). The inner
  tail of ρ past the IP is an exponential with a self-consistent e-folding
  length of 42.9 cm. S(0) is 11.5 / 6.8 / 4.2 mT for edges at
  1.20 / 1.40 / 1.60 m.
- **Match point z_m on a node; it moves with the edge** (user's choice).
  Until 2026-11-15 z_m = edge + 1.0 m (`MATCH_DISTANCE`). Since then it is
  the node past z_p + `TAPER_LENGTH`, with z_p the peak of S for ε = 0
  (see the over-screening section). That is edge + 1.0 m for 1.20/1.25 m and
  edge + 0.9 m for the other edges; at ε = 0 the total changes by ≤ 1.6 mT.
  For |z| ≥ z_m the B-spline coefficients are G's, so the
  total is identically 0 and the elements stop at ±z_m. G's 0.2 m spline on
  the 0.1 m grid is exact (nested knots). The rest is a least-squares fit,
  integral preserved; it is C3 by construction.
- **The edge always sits on a node.** An edge between nodes fits about
  twice as badly at the edge (prototype, 1.23 m: 13/35 mT rms/max against
  5.5/18). Hence 0.05 m steps via an offset grid; see below.
- **5 cm steps (implemented in 001e, 2026-11-15).** `SCREENING_EDGES`
  runs 1.20–1.60 m in 0.05 m steps. Each edge picks its grid
  (`grid_for_edge`). Each case holds `element_boundaries` on |z| ≤ z_m, with
  the IP split. Edges at odd
  multiples of 0.05 m use a 0.1 m grid offset by 0.05 m (`S_END` = 6.05,
  121 elements). G is projected onto it, with G = 0 beyond ±6 m (15.5 µT
  max error, against exact on the aligned grid). The element straddling the
  IP is split at s = 0 into two 0.05 m elements. The split is exact: a
  quartic on half the interval is a quartic, and Spline4 and
  BFieldExpansion take the element's own length.
  - Prototype for edges 1.25, 1.35 and 1.45 m: edge fit 5.5–5.7 / 18 mT,
    as on the nodes. SplineBoris elements (and the Hermite → polynomial
    conversion that `bfe_setup` uses) reproduce the spline to 4e-14 T.
    There are 2·z_m/0.1 + 1 elements, e.g. 46 for 1.25 m.
  - Still needed:
    - an element builder over `element_boundaries`, because
      `LongitudinalFitter.to_line` assumes a uniform `delta`;
    - in 004b, the corrector intervals selected by s must fall on element
      boundaries.
  - BFieldExpansion elements could not be built on 2026-11-15: the prebuilt
    kernels predate the BFieldExpansion change of 2026-11-05 and segfault.
- **Results (2 T).**
  - S/G = 0.5 lands on the node to 0.2 mm.
  - Fit − target at the edge: 5.4–5.7 mT rms (18 max).
  - ∫B_s total over 0–6 m: +1.88 T·m per m of edge shift (2.729 T·m at
    1.40 m).
  - max |A_1| grows from 0.066 T (1.20 m) to 0.089 T (1.60 m).
  - The 1.40 m case equals 001d with its edge moved by 3.7 mm (1.404 →
    1.40 m), to 1.4 mT rms. Beyond z_m, 001d's total has 1.2 mT rms of
    G/S fit mismatch, where 001e has 0.
- **Real field on the beam axis** (real-field figure, `TILT_ANGLE` = +0.015).
  - B_s: main positive, screening negative (−S), total = main + screening.
  - B_x of the total:
    - +θ·B_s in the body (0.030 T at 2 T, 0.037 T at 2.5 T);
    - a sharp negative peak at ±edge, of size max |A_1|: 0.066 → 0.089 T
      for edges 1.20 → 1.60 m at 2 T (0.082 → 0.111 T at 2.5 T);
    - exactly 0 beyond ±z_m.
  - Beyond z_m, main and screening each still carry B_x (S·b′/2 grows with
    S), but they cancel exactly. Their small wiggles at 3.5–5.5 m are the
    fit's ripple in G′, amplified by S·b′/2. They cancel too.
  - These signs are the ring's (θ_b = −`THETA`, see the Sign bullet
    above): the same as the 004a templates and 010a.
- **A_0 formula bug (found 2026-11-15, note 18).** 001d's A_0 = b − θ²(s b/2
  + s² b′/4) is not even dimensionally consistent. Rederived, it is
  b − θ²(b/2 + s b′ + s² b″/4); against `TiltedSolenoid` (analytic 2 T) the
  error falls from 5.1 to 0.06 mT.
  - Fixed in `idea_solenoid_field.py`, so it applies to 001e. 001d is
    unchanged, because the user asked to leave it alone.
  - A_1..A_4 rederive correctly at O(θ²). At the sharp IDEA edge, though,
    s·θ ≈ 2 cm is not small against the edge width, and the O(θ³)
    remainder of A_1 is ~2.5 mT.
  - The ring elements therefore use exact coefficients (note 18).
- **Field strength.** `MAIN_FIELD` (set at the top; the user set 2.5 T on
  2026-11-15, and 2 T reproduces the data) scales the digitized
  curves by `MAIN_FIELD / OPERA_FIELD` (2 T) when they are loaded. The
  screening follows the main field, and all later steps are linear, so every
  field, integral and residual scales and the edges stay put. Checked at
  3 T: exactly ×1.5. Linear scaling ignores iron saturation.
- Figures:
  - `IDEA_solenoid_screening_edge_scan_components_<field_tag>.pdf` (e.g.
    `_2T`): main, screening (as plotted, +S) and total, one panel per edge,
    z = 0–6 m.
  - `IDEA_solenoid_screening_edge_scan_real_field_<field_tag>.pdf`: the real
    field on the beam axis, beam frame, S = ±6 m. B_s = A_0 and B_x = A_1
    (with `TILT_ANGLE`), for main, screening (−S) and total, one row per
    edge.
  - The fit residuals and max |A_m| are printed only.

## 001e: over-screening, as in 010a (2026-11-15)

- **Goal.** Mimic 010a, where the local solenoid (1.3–1.9 m, −0.79 B0) next to
  the main coil pulls the total down to −0.67 B0. Its negative region is
  −0.36 m × B0, i.e. −1.07 T·m at 3 T (computed from 010a's analytic fields).
  Make the IDEA screening stronger so that the total overshoots below zero,
  while S still matches G before the main solenoid's own edge (G = G(0)/2 at
  ~2.9 m).
- **Chosen (user):** scale S by k(z) = 1 + ε·w(z) (`SCREENING_OVERSHOOT` = ε,
  `TAPER_LENGTH`).
  - First version: w = 1 for |z| ≤ z_m − `TAPER_LENGTH`, then a septic (C3)
    smoothstep to 0 at z_m = edge + `MATCH_DISTANCE` (user: "should that not
    depend on z_m and the edge"). Superseded the same day by the
    peak-anchored taper, below.
  - The edge keeps its measured shape, scaled by 1 + ε. Its half-plateau
    point (S/G = (1 + ε)/2) stays on the node.
- **Taper ≥ 0.4 m.** The 0.1 m quartic cannot follow a shorter transition:
  - a 0.3 m taper gives 0.17 T fit error;
  - a hard switch at z_s leaves 0.8 T, and the integral constraint pushes
    that error into S(0) (+54 mT).
- **Alternatives prototyped** (edge 1.40 m, 2.5 T, each lowering ∫B_s,total
  by 0.90 T·m, i.e. 010a's negative region × B0). Scratch script
  `overshoot_options.py`, not kept; figure
  `IDEA_solenoid_screening_overshoot_options_2p5T.pdf`.

  | | strength | z_m | min total | max B_x edge / outer |
  |---|---|---|---|---|
  | scale S, taper 1.9→2.4 m (chosen) | ε = 0.52 | 2.4 m | −1.14 T | 0.165 / 0.076 T |
  | extra anti-solenoid 1.4–2.0 m, measured edges at both ends | ΔB = 1.49 T | 2.6 m | −1.34 T | 0.180 / 0.090 T |
  | prescribed negative lobe in the total (0.4/0.5 m ramps) | 1.63 T | 2.4 m | −1.62 T | 0.133 / 0.110 T |

  The anti-solenoid's outer fringe needs ~0.6 m to die out, so its z_m moves
  out by 0.2 m. The lobe's ramp, which cannot be shorter than 0.4 m, moves
  the total's zero crossing out to 1.53 m.
- First-version results (2.5 T, ε = 0.5, taper 0.5 m, z_m = edge + 1.0 m):
  min total −1.12 to −1.05 T; ∫B_s total 0.85–0.88 T·m below ε = 0; B_x
  edge peak 0.14–0.18 T, opposite-sign peak 0.067–0.075 T. Figures
  `..._2p5T_eps0p50.pdf` (now stale).

### Peak-anchored taper (2026-11-15, current)

- **Why.** With the taper ending at z_m (the user had set
  `MATCH_DISTANCE` = 1.2 m locally), the total sat flat at about −1.05 T
  from the S peak (1.76 m at edge 1.40 m) to the taper start (2.10 m), and
  rose only in the taper. That gave a shoulder in B_s and a B_x tail ("the
  extra kink"). The user asked for the matching to follow the peak of the
  field instead, aligned with the peak for ε = 0.
- **How.** z_p = peak of the ε = 0 target G·[ρ(z − e) + ρ(−z − e)]: the vertex
  of a parabola over ±`PEAK_WINDOW` (0.10 m) around the argmax.
  - The peak is flat (±5 cm costs 2–5 mT), so the raw argmax steps by up to
    3 cm between edges, on vertices of the digitized ρ.
  - z_p = edge + 0.411 m (1.20 m) … edge + 0.338 m (1.60 m), within 1.3 cm
    of the fitted ε = 0 spline's peak.
  - w = 1 for |z| ≤ z_p, then the septic taper over exactly `TAPER_LENGTH`
    (settable, ≥ 0.4 m). z_m is the next node; S = G·ρ between the taper end
    and z_m (< 0.1 m).
  - The ε = 0 field and w both have zero slope at z_p, so the target's peak
    stays at z_p for any ε (fitted peaks within 1.4 cm). `MATCH_DISTANCE` is
    gone.
  - Note: z_m itself cannot be the peak. S = G there, so there would be no
    overshoot at the peak.
- **Prototype comparison** (edge 1.40 m, 2.5 T; figure
  `IDEA_solenoid_screening_overshoot_peak_anchor_2p5T.pdf`, scratch scripts
  not kept):

  | | taper, z_m [m] | min total | ∫ 0–6 m / negative part [T·m] | max \|B_x\| / opposite-sign |
  |---|---|---|---|---|
  | z_m = edge + 1.2, ε = 0.5 | 2.10–2.60 | −1.08 T | 2.35 / −0.89 | 0.162 / 0.076 T |
  | peak-anchored, ε = 0.5 | 1.78–2.28, 2.30 | −1.07 T | 2.69 / −0.55 | 0.162 / 0.065 T |
  | peak-anchored, ε = 0.73 | same | −1.59 T | 2.35 / −0.85 | 0.193 / 0.096 T |
  | 010a at 2.5 T | — | −1.67 T | — / −0.90 | — |

  - The lobe is shorter, so the same integral needs a larger ε.
  - A 0.6 m taper needs ε 0.68 (instead of 0.73) and gives a 0.075 T
    opposite-sign peak, with z_m 0.1 m further out.
  - The user chose 0.5 m, as long as it stays settable.
- **ε = 0.75** (user: "whatever you prefer"; chosen by me). It reproduces
  010a's lobe:
  - at edge 1.40 m: −0.650 B0 and −0.348 m × B0, against 010a's −0.67 B0
    and −0.358 m × B0;
  - at edge 1.30 m, 010a's own local-solenoid start: −0.661 B0 and
    −0.364 m.
- **Results (2.5 T, ε = 0.75, taper 0.5 m):**
  - Same elements as ε = 0 (44–50). The total is 0 beyond z_m, and the edge
    sits on its node to 0.2 mm.
  - The total crosses zero 1.6 cm past the edge.
  - Min total −1.67 T (−0.669 B0, edge 1.20 m) to −1.56 T (−0.625 B0,
    1.60 m), 3–4 cm past z_p. Negative part −0.95 to −0.78 T·m.
  - ∫B_s total over 0–6 m: 1.77–2.89 T·m, i.e. 0.99–1.17 T·m below ε = 0
    (2.328 vs 3.411 T·m at 1.40 m).
  - S(0) ×1.67–1.82; total(0) 3.5–10.5 mT lower.
  - Fit − target: 38–40 mT max at the edge, 5–19 mT in the taper.
  - B_x (A_1):
    - edge peak 0.17–0.22 T (0.08–0.11 T at ε = 0);
    - an opposite-sign peak of 0.094–0.108 T about 0.26 m past z_p,
      mid-taper;
    - A_2–A_4 ×1.75.
- Figures (`_eps0p75`):
  - `IDEA_solenoid_screening_edge_scan_components_2p5T_eps0p75.pdf`;
  - `..._real_field_2p5T_eps0p75.pdf`.

## Open items / next steps

- ~~Decide the sign convention of `TILT_ANGLE`~~: resolved 2026-11-15.
  +0.015 = −`THETA` is the ring's convention (Sign bullet above).
- ~~Turn A_0..A_4 into elements and install them~~: done 2026-11-15 as
  BFieldExpansion with exact coefficients, installed and corrected through
  004c (note 18).
- 001d still has the wrong A_0 θ² term (see "A_0 formula bug"). Fix it there
  only if the user agrees to touch 001d.
- ε = 0.75 (010a's lobe at edge 1.40 m) is a default, not a physics
  choice. Revisit it once the corrections run: ε could instead come from a
  target ∫B_s, since the field is linear in ε. Over-screening adds an
  opposite-sign B_x peak (~0.1 T) mid-taper, which the corrector layout has
  to cover.
- Multi-coil chain, if an analytic 3D field is ever needed.
