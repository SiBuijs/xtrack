# FCC-ee IR solenoid study — paper branch (`fcc_nonlocal_paper`)

This folder holds the simulations for a journal paper on **non-local compensation of the FCC-ee detector solenoid**. Launch Claude Code from this folder so that this file is loaded.

## What the paper shows

There are two comparisons, each with the same correction scheme, knobs and procedure:
- **Field strength:** 2 T analytic vs 3 T analytic, which shows how much headroom the scheme has.
- **Field realism:** 2 T analytic vs the 2 T BNL field map. This is the headline result.

The 3 T field map is the missing case and is only *discussed*. Comparing the local and non-local schemes is **not** a goal.

## Read before working

- `paper/TODO.md`: the paper's task list. Read it when picking up work. It is the canonical copy; the claude.ai project syncs from it.
- `paper/RESULTS.md`: every number the paper will quote, with provenance.
- `claude_notes/00_overview.md`: the pipeline, the shared helpers and the history of past decisions. Read the topic note it points to before changing a script.

Mention these paths in backticks (as here) rather than as `@`-imports, so they aren't loaded into every session.

## Setup facts

- Z-pole, 45.6 GeV, lattice `fccee_z_lcc.json` (LCC optics), solenoids at all 4 IPs (`ipa`, `ipd`, `ipg`, `ipj`), tilt θ = −0.015 rad.
- Element models:
  - **SB** = `xt.SplineBoris`, the full field;
  - **VarSol** = `xt.VariableSolenoid`, the linear/"linearized" model.

  `sb_off` is the bare-machine baseline built from the same JSON.
- Pipeline:
  - 004a templates → 004b install → 004c correct → 004d analysis;
  - 013 runs DA + MA + emittance + polarization (via 018).
- Case flags: `--b0 {2.0,3.0}`, `--max-transverse-order N` (default 4), `--model {splineboris,varsol}`. Tags are threaded into every filename (`field_tag`, `order_tag` in `solenoid_params.py`).
- Raw data goes to `data/` (gitignored). Paper figures (DA, MA, EMIT, POL, TUNE, 004k tune diagrams) go to `~/cernbox/Pictures/FCC_Solenoid_Paper/<topic>/` (`aperture_study_io.paper_figure_dir`), all other PDFs to `~/cernbox/Pictures/FCC_Solenoid_Studies`.

## Conventions (use these everywhere, in code and text)

- Figures of merit: |C⁻|, βx2/βx and βy1/βy, εy, DA (in σ), MA (δ at the injection amplitude), Q′ and Q″.
- Q″ always means the true d²Q/dδ². Always say which plane. **Pitfall:** the `d2qx`/`d2qy` that 004c/004j print (`detuning.py`, `q*_derivatives[2]`) is **Q″/4** (see note 10; 004c's docstring said Q″/2 until 2026-10-15). `004n_figures_of_merit.py` stores the true Q″ as `q2x`/`q2y`. Check the convention of any other source before quoting it.
- With coupling present, single-plane Courant–Snyder emittances are approximate. For the paper's εy, use the 4×4 covariance eigen-emittances, or state which definition was used.
- SI units. Fields in T, lengths in m.

## Production rules

1. A paper number comes only from the **frozen pipeline**:
   - 004c with its default solver settings (`OPTICS_RCOND = 1e-6`, staged pass 1);
   - a committed state of the code, never uncommitted edits;
   - corrected-lattice JSONs are never edited by hand.
2. Every run that feeds the paper:
   - records the script, its arguments, the git commit (`git rev-parse --short HEAD`), the seed and the output file;
   - when a run fixes no seed (e.g. 018 without `--seed`), says so explicitly.
3. **Logging a result** (do all four steps, in this order):
   - a. Check the output file exists and the numbers are sane against the previous value in `paper/RESULTS.md`.
   - b. Add or update the row in `paper/RESULTS.md`, with provenance.
   - c. Tick the item in `paper/TODO.md` and append `→ <output path>`. Never tick an item without a result file.
   - d. If the result changes the understanding, add a short dated note in `claude_notes/` and update the index in `00_overview.md`.
4. Heavy runs (DA, MA, tracking, scans):
   - run in the background with output logged to a file;
   - report the expected runtime before starting anything longer than ~10 minutes.
5. Before reporting a physics conclusion, re-check it against the numbers in the output files rather than recalling it. Contradictions between `claude_notes/` and code comments: the newer dated source wins, and the conflict gets flagged.

## Repository rules

- This is a fork of `xsuite/xtrack`.
  - Changes to library code under `xtrack/` (SplineBoris, TubeFitter, …) go in separate, focused commits, so they can later be upstreamed without the paper files.
  - Say so whenever a task touches `xtrack/`.
- **Field maps are never committed.** They were purged from history on 2026-10-14, and `.gitignore` now has `**/Field_map_*.csv`.
  - 006b reads the BNL map from `$FCC_SOLENOID_FIELD_MAP`, falling back to `~/cernbox/Solenoid_Field_Maps/`. New scripts load it the same way.
  - Don't add new files over ~1 MB without asking.
- Commit messages: imperative, say which script and which case (2 T/3 T, SB/VarSol, order) changed.
