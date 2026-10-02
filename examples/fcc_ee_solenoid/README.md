# FCC-ee detector solenoid: lattice construction and correction

What this folder does: take the FCC-ee Z-pole lattice, install a tilted detector
solenoid plus its compensation scheme around each of the four interaction
points, correct the damage it does to orbit, optics and betatron coupling, and
plot the result.

Everything here stops at that point. There is no aperture, emittance,
polarization or parameter-scan code on this branch — only the chain that builds
a corrected lattice and looks at it.

---

## The physics problem

The machine is FCC-ee at the Z pole: 45.6 GeV e+/e−, lattice
`fccee_z_lcc.json` with local-chromaticity-correction optics. That file comes
from outside this folder; nothing here produces it.

Each of the four IPs (`ipa`, `ipd`, `ipg`, `ipj`) gets a detector solenoid
**tilted by θ = −0.015 rad** relative to the beam axis, because the beams cross
at an angle. The tilt is what makes this hard: a solenoid aligned with the beam
is a clean x–y coupler, but a tilted one also steers the beam and perturbs
dispersion.

Geometry, all from `solenoid_params.py`:

| | 2 T case | 3 T case |
|---|---|---|
| main solenoid half-length | 1.23 m | 1.30 m |
| radius | 0.13 m | 0.13 m |
| first corrector starts at | 1.23 m | 1.40 m |
| corrector region ends at | 2.29 m | 2.29 m |

Two compensation solenoids sit 12.0 m either side of each IP, 1.5 m long,
radius 0.03 m. Their field is scaled at build time so the net ∫B·ds over the
triplet cancels — that scale (`comp_scale_b`) is computed in 004a, not later.

Two element models are carried in parallel the whole way:

- **SplineBoris** (`xt.SplineBoris`) — the field is given as quartic spline
  profiles of `bs` along s plus per-multipole-order `bx`/`by`, pushed with a
  Boris integrator. Captures multipole content up to order 4. Slower, more
  faithful. Referred to as **SB**.
- **VariableSolenoid** (`xt.VariableSolenoid`) — a linear `ks_profile` ramp plus
  one dipole kick per slice. Cheap, linear only, no multipole content.
  Referred to as **VarSol**.

Both are built at 2 T and 3 T, giving four corrected lattices in total.

---

## Quick start

From this directory, on the `xsuite` environment. Set `MPLBACKEND=Agg` if you
do not want plot windows — several scripts end in `plt.show()`.

```bash
# 1. Build the isolated solenoid element templates (~45 s each)
python 004a_build_and_check_solenoids.py --b0 2.0
python 004a_build_and_check_solenoids.py --b0 3.0

# 2. Install them into the ring (~35-40 s each)
python 004b_install_solenoids_in_fcc_ring.py        --b0 2.0
python 004b_install_solenoids_in_fcc_ring.py        --b0 3.0
python 004b_install_varsol_solenoids_in_fcc_ring.py --b0 2.0
python 004b_install_varsol_solenoids_in_fcc_ring.py --b0 3.0

# 3. Correct orbit, optics and coupling at all four IPs
#    (~5-10 min each; the four are independent and can run in parallel)
python 004c_correct_solenoids_in_fcc_ring.py --model splineboris --b0 2.0
python 004c_correct_solenoids_in_fcc_ring.py --model splineboris --b0 3.0
python 004c_correct_solenoids_in_fcc_ring.py --model varsol      --b0 2.0
python 004c_correct_solenoids_in_fcc_ring.py --model varsol      --b0 3.0

# 4. Plot
python 004d_analysis_and_plots.py --b0 3.0
```

Using a lattice afterwards:

```python
import xtrack as xt
env = xt.Environment.from_json(
    'fccee_z_lcc_splineboris_solenoids_coupling_corrected_3T.json')
line = env['fccee_p_ring']
tw = line.twiss4d()        # qx 194.1601, qy 170.2001, C- 7.98e-4
```

---

## The scripts

### 004a – 004d: the current chain

**`004a_build_and_check_solenoids.py`** builds the solenoid field models in
isolation, outside any lattice. It evaluates the analytic tilted field
(`tilted_solenoid.TiltedSolenoid`, wrapping an elliptic-integral solenoid field),
extracts spline coefficients from it, assembles the main + two compensation
solenoids into a three-element system, computes the compensation scale that
cancels the net field integral, and checks symplecticity. Writes four line
templates — SB and VarSol versions of the main and compensation solenoids — plus
a metadata block recording every build setting.

Flags: `--b0` picks the field strength and hence the filename tag.
`--max-transverse-order` caps the multipole order baked into the SplineBoris
elements (default 4; lower is cheaper to track and tags the output `_o2` etc.).

**`004b_install_solenoids_in_fcc_ring.py`** (SB) and
**`004b_install_varsol_solenoids_in_fcc_ring.py`** (VarSol) clone those
templates into the real ring at each IP, insert the dipole correctors, and
create the on/off knobs. Output is an *uncorrected* ring with solenoids
installed but switched off.

**`004c_correct_solenoids_in_fcc_ring.py`** is the heart of the study. Per IP,
and separately for the left and right half-straight:

1. **Doublet tilt.** Rotate the final-focus doublet by half the solenoid
   rotation angle, in opposite senses either side of the IP. This is a
   feed-forward step, not a fit — the angle is computed directly from the
   solenoid strength.
2. **Orbit.** Match the closed orbit back to the bare machine's using 12
   dipole correctors per side — 6 horizontal and 6 vertical.
3. **Optics.** Restore betx, bety, alfx, alfy, dx, dpx at the straight-section
   boundary (~1360 m from the IP) using the final-focus quads plus mid-bend trim
   quadrupoles, with betx/bety at the chromatic sextupole as additional targets.
4. **Coupling.** Zero the x–y coupling using 42 skew quadrupole trims per
   side.

It also reports second-order chromaticity before and after, and the vertical
phase advance from the doublet to the vertical sextupole pair.

Flags: `--model`, `--b0`, `--max-transverse-order`, `--output-tag`,
`--no-chromaticity` (saves ~44 full-ring twisses), `--chromaticity-points`,
`--optics-step`, `--optics-rcond`, and `--ips` to correct a subset.

> **`--ips` does not write output.** A partial run leaves the other IPs'
> solenoids off, so the lattice is not a valid corrected ring. The script
> detects this and refuses to save. Use it only for tuning the match.

**`004d_analysis_and_plots.py`** twisses the corrected SB lattice and plots it.
Six figures appear on screen:

1. Ring overview: beta, dispersion, Bs, vertical orbit y, vertical dispersion
   D_y, and the normalised coupled betas, over the ±20 m IR window
2. The same window with energy loss dE/ds and the three spin components
3. betx2/betx and bety1/bety in the IR, 2 T and 3 T overlaid on one axes
4. The same ratios over the full straight section, one panel per field
5. `ipa` beta functions with and without solenoids, IR window
6. The same over the full straight section

It writes 4 combined PDFs to `$FCC_SOLENOID_PLOT_DIR/Coupling_Studies` (default
`~/cernbox/Pictures/FCC_Solenoid_Studies`) and 14 single-field PDFs to
`Coupling_Studies/plots`.

> 004d always loads the untagged 2 T **and** 3 T corrected SB lattices for its
> comparison panels, regardless of `--b0`, which only selects the primary case.
> Both must exist. Note also that the comparison paths do not carry the order
> tag, so `--max-transverse-order 2` mixes orders in one figure.

### 000 – 003: the earlier path, kept for reference

These predate the 004 chain and are not on its dependency path. They install a
simpler solenoid model — `TiltedSolenoid` slices directly, without the spline
field extraction — and correct it two different ways. Based on Boscolo, Ciarma
and Burkhardt, [CDS 2948247](https://cds.cern.ch/record/2948247), NIM A 1083
(2026) 171135.

| script | what it does |
|---|---|
| `000a_local_install_solenoids_and_correctors.py` | install solenoids + correctors, **local** correction scheme |
| `000b_local_correction.py` | run the local correction |
| `001a_non_local_install_solenoids_and_correctors.py` | install solenoids + correctors, **non-local** scheme |
| `001b_non_local_correction.py` | run the non-local correction |
| `002_analysis_and_plots.py` | twiss and plot the result. Currently reads the non-local lattice; the local one is a commented-out line at the top |
| `003_a_look_at_non_linear_components.py` | standalone: finite-difference the tilted field to expose its non-linear content. Reads no lattice |

The branch name `fcc_nonlocal_solenoid` refers to the 001 non-local scheme.

### Helper modules

| module | used by | contents |
|---|---|---|
| `solenoid_params.py` | 004a–004d | single source of truth for geometry; `field_tag`/`order_tag` filename helpers; `--b0`/`--max-transverse-order` argparse helpers; `PLOT_DIR` |
| `spline_boris_setup.py` | 004a | the field-extraction and SplineBoris/VarSol line-building logic |
| `tilted_solenoid.py` | 001a, 003_a, 004a | `TiltedSolenoid`: the analytic field model, rotated into and out of the tilted frame |

`004c` additionally reaches into `../nonlinear_tunes/` for
`detuning.get_nonlinear_chromaticity`. Do not delete that folder.

---

## The knobs

This is the part most worth knowing. A corrected lattice carries **1636
variables**: 472 inherited from the base FCC-ee lattice (the `ksd*` sextupole
families, arc quad families, RF phases, geometry angles) and 1164 created by
004b and 004c.

### What a human actually touches

Per IP there are exactly **two independent top-level knobs**:

```python
line['on_sol_ipa']      = 0 or 1   # the main detector solenoid field
line['on_sol_corr_ipa'] = 0 or 1   # everything corrective at this IP
```

`on_sol_corr_{ip}` is a single master. 004c wires everything else to hang off
it, so turning it on enables the compensation solenoid, the doublet tilt and all
three corrections at once:

```
on_sol_corr_{ip}
├── on_comp_sol_{ip}                 the compensation solenoid field
├── on_rot_doublet_left_{ip}         final-focus doublet tilt
├── on_rot_doublet_right_{ip}
├── on_sol_orbit_corr_{ip}
│   ├── on_sol_orbit_corr_left_{ip}
│   └── on_sol_orbit_corr_right_{ip}
├── on_sol_optics_corr_{ip}
│   ├── on_sol_optics_corr_left_{ip}
│   └── on_sol_optics_corr_right_{ip}
└── on_sol_coupling_corr_{ip}
    ├── on_sol_coupling_corr_left_{ip}
    └── on_sol_coupling_corr_right_{ip}
```

Note the compensation solenoid is **not** independent of the correction — it is
part of it. To get the bare machine you need `on_sol_{ip} = 0` as well, since
the main field is deliberately left on its own switch.

All switches are at 1 in the files as saved, i.e. every lattice on disk is the
fully corrected machine.

> **Assigning a number to a mid-level switch breaks its link.** `line['on_sol_
> optics_corr_ipa'] = 0` replaces the expression tying it to `on_sol_corr_ipa`
> with a constant, and it will no longer follow the master afterwards. That is
> usually what you want for a one-off comparison, but it does not undo itself.

One value knob rather than a switch:

```python
line['phi_rot_doublet_ipa']   # doublet tilt angle [rad], 0.01279 at 3 T
```

It is set to half the solenoid rotation angle, which is itself half the full
rotation — i.e. `(ksol * L / 2) / 2`.

### The full inventory

| count | knobs | what they are |
|---:|---|---|
| 96 | `acb[hv]<n>_sol_{side}_{ip}` | orbit corrector strengths, 6 per plane per IP-side |
| 96 | `…_from_on_sol_orbit_corr_{side}_{ip}` | the intermediate nodes xtrack creates when a knob is defined |
| 12 | `on_sol_orbit_corr_{ip}`, `…_{side}_{ip}` | orbit switches |
| 120 | `k1_q*.<n>_sol_corr` | optics quad trims, 15 per IP-side |
| 120 | `…_from_on_sol_optics_corr_{side}_{ip}` | nodes |
| 12 | `on_sol_optics_corr_*` | optics switches |
| 336 | `k1s_q*.<n>_sol_coupling_corr` | coupling skew-quad trims, 42 per IP-side |
| 336 | `…_from_on_sol_coupling_corr_{side}_{ip}` | nodes |
| 12 | `on_sol_coupling_corr_*` | coupling switches |
| 4 | `phi_rot_doublet_{ip}` | doublet tilt angle |
| 8 | `on_rot_doublet_{side}_{ip}` | doublet tilt switches |
| 4 | `on_sol_corr_{ip}` | the per-IP master |
| 4 | `on_sol_{ip}` | main solenoid field |
| 4 | `on_comp_sol_{ip}` | compensation solenoid field |

**1164 total.** Roughly half are the `*_from_on_sol_*` nodes, which are
bookkeeping — xtrack generates one per knob automatically and you never address
them directly. Of what remains, 552 are per-magnet strengths the matches wrote,
and only 56 are switches.

### Knobs that used to exist and no longer do

If you are reading older notes, a presentation, or a lattice from before
2026-10-02, these four appear and are now **gone**:

| removed knob | what it did |
|---|---|
| `sext_amp` | scaled every SplineBoris order-2 (sextupole) spline coefficient at runtime |
| `comp_b_scale` | global multiplier on the compensation solenoid field |
| `main_b_scale` | global multiplier on the main solenoid field |
| `comp_b_scale_{side}_{ip}` | per-side compensation multipliers |

They existed only so the deleted scan scripts could sweep them, and all four sat
at their nominal 1.0 in every committed lattice, so removing them changed no
optics quantity. Two consequences worth knowing:

- **Sextupole content is now fixed at build time.** `bx[2,k]` and `by[2,k]` are
  installed as plain element data rather than as expressions on `sext_amp`. If
  you want to scale the solenoid's sextupole content, change
  `SEXTUPOLE_AMPLIFICATION_FACTOR` in 004a and rebuild. There is no runtime
  knob.
- **The SplineBoris lattices got much lighter.** `sext_amp` alone appeared in
  ~24000 expressions, about 46 % of the expression graph. Dropping it cut those
  files from ~15 MB to ~12.6 MB.

If you need any of these back, re-add them in 004b and re-run 004b and 004c.

---

## The JSON files

15 files, 135 MB. Producer and consumer for each:

| file | MB | written by | read by |
|---|---:|---|---|
| `fccee_z_lcc.json` | 8.7 | **nothing — external input** | 000a, 001a, both 004b |
| `004_solenoid_lines_2T.json` | 0.7 | 004a `--b0 2.0` | both 004b `--b0 2.0` |
| `004_solenoid_lines_3T.json` | 0.7 | 004a `--b0 3.0` | both 004b `--b0 3.0` |
| `temp_fcc_ee_lcc_splineboris_solenoids_2T.json` | 12.2 | 004b | 004c |
| `temp_fcc_ee_lcc_splineboris_solenoids_3T.json` | 12.2 | 004b | 004c |
| `temp_fcc_ee_lcc_varsol_solenoids_2T.json` | 10.2 | 004b_varsol | 004c |
| `temp_fcc_ee_lcc_varsol_solenoids_3T.json` | 10.2 | 004b_varsol | 004c |
| `fccee_z_lcc_splineboris_solenoids_coupling_corrected_2T.json` | 12.6 | 004c | **004d** |
| `fccee_z_lcc_splineboris_solenoids_coupling_corrected_3T.json` | 12.6 | 004c | **004d** |
| `fccee_z_lcc_varsol_solenoids_coupling_corrected_2T.json` | 10.8 | 004c | nothing |
| `fccee_z_lcc_varsol_solenoids_coupling_corrected_3T.json` | 10.8 | 004c | nothing |
| `temp_fcc_ee_lcc_local_solenoid.json` | 10.0 | 000a | 000b |
| `fccee_z_lcc_local_solenoid.json` | 10.1 | 000b | 002, commented out |
| `temp_fcc_ee_lcc_non_local_solenoid.json` | 9.7 | 001a | 001b |
| `fccee_z_lcc_non_local_solenoid.json` | 9.8 | 001b | 002 |

Naming:

- `temp_*` — an intermediate. Solenoids installed, not yet corrected.
- `*_coupling_corrected_*` — 004c output. The deliverable.
- `_2T` / `_3T` — field strength. 2.5 T would tag `_2p5T`.
- `_o2` — a non-default `--max-transverse-order`. SplineBoris only; VarSol is
  linear so the tag is deliberately left off its filenames.

All lattices are saved as an `xt.Environment` holding one line, `fccee_p_ring`,
so load them with `xt.Environment.from_json` and index the line out.

### Are they all necessary?

**No.** Only one file is irreplaceable, and about half the bulk is intermediates.

**Keep — genuinely irreplaceable (8.7 MB)**

`fccee_z_lcc.json`. No script in this repo produces it. If it is lost it has to
come from whoever built the LCC optics.

**Keep — expensive to rebuild (25.2 MB)**

The two SB corrected lattices. 5-10 minutes of matching each, and they are
what 004d reads.

**Keep, but nothing reads them (21.6 MB)**

The two VarSol corrected lattices. They are 004c's output for `--model varsol`
and the reason the VarSol model exists, but no surviving script loads them —
004d only plots SB. Worth keeping as the cheap linear-only comparison lattice;
just be aware they are currently write-only.

**Droppable — pure intermediates (64.5 MB, 48 % of the total)**

The six `temp_*` files. Each is regenerated in well under a minute from its
input:

| | rebuild cost |
|---|---|
| `temp_*_splineboris_*` | 004b, ~39 s |
| `temp_*_varsol_*` | 004b_varsol, ~34 s |
| `temp_*_local_solenoid` | 000a |
| `temp_*_non_local_solenoid` | 001a |

Deleting them costs you one 004b run before any future 004c run. The
`004_solenoid_lines_*` templates (1.4 MB for both) are also regenerable, in ~45 s
each, but they are small enough not to bother.

**Droppable — effectively unused (10.1 MB)**

`fccee_z_lcc_local_solenoid.json`. Its only reader is a commented-out line at
the top of 002. Either delete it, or uncomment that line if the local-scheme
comparison still matters.

A note on disk versus repo: these files are tracked in git, so deleting them
frees working-tree space but not repository history.

---

## Known rough edges

- **The 3 T second-order chromaticity is large.** The corrected rings come out
  near d2qy ≈ +1170 at 3 T against about −38 with solenoids off. The mechanism
  is understood: the half-straight optics match leaves the vertical phase
  advance from the doublet to the vertical sextupole pair free, and a beta bump
  lands on the chromatic sextupole. The 2 T case shows the same effect more
  mildly, around −310. Not resolved.
- **004d mixes orders in its comparison panels** when run with
  `--max-transverse-order 2`, because the comparison filenames omit the order
  tag.
- **Coupling RDTs have holes in the IR.** `f1001` and `f1010` come back NaN
  inside the compensation solenoid bodies — 1208 of 32573 points on the
  corrected 3 T ring, in eight blocks. The cause is traced to `tw4d.W_matrix`
  being reported in kinetic rather than canonical momenta inside the field,
  which makes the Edwards-Teng discriminant non-invariant. `betx`, `bety`,
  `betx2` and `bety1` in those same slices inherit the same convention, so read
  them as "is the coupling corrected here", not as canonical Twiss functions.
  The long comment above the `COUPLING RDT VALIDITY NOTE` anchor in 004d has the
  full derivation.
