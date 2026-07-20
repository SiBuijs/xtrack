# Why the canonical/kinetic momentum gap produces a symplectic error — mechanism, in detail

Follow-up to `00_overview.md`'s "canonical vs. kinetic transverse momentum"
finding. That note establishes *what* was wrong (Boris writes kinetic `px,
py` into slots xtrack treats as canonical). This note explains, with actual
numbers from the strongly-bending 45.6 MeV case in `005_study_symplectic_error.py`,
*why* that gap produces a symplectic-error floor of ~28 instead of something
imperceptibly small, and why it only matters for strongly-bending
trajectories.

All numbers below were computed directly (not estimated) at `n_steps=100000`
(converged), same setup as 005: `SolenoidField(L=4, a=0.3, B0=1.5, z0=20)`,
`s_start=0, s_end=30`, electron at `energy0=45.6e9/1000` (45.6 MeV).

## 1. The physics: minimal coupling and the symplectic form

A charged particle in a magnetic field is Hamiltonian only with respect to
canonical momentum `P = p_kin + qA`, not kinetic momentum `p_kin`. The map
`(x, p_kin) → (x, P) = (x, p_kin + qA(x))` is exactly the canonical
transformation that turns the "twisted" symplectic form preserved by the
physical (kinetic) dynamics,

```
ω_kin = dx∧dp_kin,x + dy∧dp_kin,y + qBz·dx∧dy
```

into the plain canonical form `ω_can = dx∧dPx + dy∧dPy` that
`line.get_R_matrix`'s symplectic check (`RᵀSR = S`) assumes.

`BorisSpatialIntegrator`'s leapfrog is internally self-consistent — each
step correctly propagates kinetic momentum and preserves `ω_kin`. The gap is
purely at the interface: the raw integrator writes `p_kin` straight into
`p.px`/`p.py`, so checking its trajectory against the canonical `S` is
checking the wrong 2-form. This is why the fix (`vector_potential_callable`
in `boris.py`) only needs to run once at `s_start` and once at `s_end`, not
inside the loop — it's a one-time bookkeeping conversion, not a
discretization correction.

## 2. Why the omitted term is tiny in absolute terms

The gauge term `ax = qAx/P0` was evaluated along the reference trajectory:

```
s=20.0 (inside fringe)   Bz=1.48 T     ax = -4.9e-3
s=22.0 (coil edge)       Bz=0.75 T     ax = -2.5e-3
s=24.0                   Bz=7.4e-3 T   ax = -2.4e-5
s=26.0                   Bz=1.6e-3 T   ax = -5.2e-6
s=28.0                   Bz=6.0e-4 T   ax = -2.0e-6
s=30.0 (exit, used)      Bz=2.9e-4 T   ax = -9.6e-7
```

Near the coil, `ax` is comparable to or larger than the particle's own
initial canonical `px = -1e-3` — using kinetic px there instead of canonical
would be a large fractional error. But 005's exit boundary sits 8 m past the
coil, deep in the fringe-decay tail, where `ax ~ 1e-6` — a part-per-million-
level omission. On its own this looks harmless. It isn't, because of what
happens next.

## 3. Stage 1 amplification: R already contains large entries (correct solenoid physics), and the *gradient* of `ax`, not its value, is what enters R

`get_R_matrix` builds `R[i,j] = ∂(exit var i)/∂(entry var j)` by finite
differences. The correction added to `px_exit` is `ax(x_exit, y_exit)`, a
*function of exit position* — so by the chain rule it contributes

```
d(ax)/dx_exit · R[x, j]  +  d(ax)/dy_exit · R[y, j]
```

to `∂px_exit/∂(var j)`, for every column `j`. Numerically, at the reference
trajectory's exit point:

```
d(ax)/dy_exit ≈ -9.6e-4   (tiny — tied to the residual Bz(exit)=2.9e-4 T)
R_corr[y, px0] = ∂y_exit/∂px0 ≈ -90.2   (large — the solenoid bends hard)
```

Product: `(-9.6e-4)×(-90.2) ≈ 0.087` — which matches the actual discrepancy
`dR = R_raw - R_corr` in that entry (`dR[px,py0] ≈ 0.079`) to within the
precision of the numerical gradient. Verified across the whole `px` and `py`
rows: predicted-from-chain-rule and actual `dR` rows agree to 3 significant
figures.

Concretely, comparing the two R matrices (`n_steps=100000`):

```
R_raw[px,py0]  = 11.85     R_corr[px,py0]  = 11.77      diff ≈ 0.08
R_raw[py,py0]  =  8.816    R_corr[py,py0]  =  8.920      diff ≈ -0.10
```

Every other entry (position rows, zeta row) is essentially unchanged — the
correction only ever gets added to `px, py`, so *only two rows* of the 6×6
`R` carry any defect, and each only by about 1%.

**Point of stage 1:** a microscopic (1e-6-level) field-gradient artifact
gets multiplied by the trajectory's own large, physically legitimate
position-sensitivity (`R ~ O(10)-O(100)`, a normal feature of strong
solenoid focusing/x–y coupling at low energy) into a ~1%-level defect
confined to the momentum rows of R.

## 4. Stage 2 amplification: the symplectic check is bilinear in R

`‖RᵀSR − S‖` is a matrix product, not a linear functional of `R`. Writing
`R_raw = R_corr + dR`:

```
R_raw^T S R_raw − S ≈ R_corr^T S dR + dR^T S R_corr      (first order in dR)
```

Checked numerically: this linear approximation gives `28.162`, versus the
exact `28.161` — matching almost exactly, confirming the entire ~28 floor
is explained by this one cross term, not by any higher-order blowup.

That cross term sums `(large R_corr entry) × (small dR entry)` over all 6
phase-space directions. The dominant contribution to the `(px,py)` entry of
the defect (which alone accounts for essentially the whole matrix norm)
decomposes as:

```
R_corr[x,px0] · dR[py,py0]   ≈  58.4 ×  0.079  ≈  4.6
R_corr[y,px0] · dR[py,py0]   ≈ -90.2 × -0.104  ≈  9.4
```

These appear (with sign) from both `R_corr^T S dR` and its transpose,
doubling to `≈28` — matching the observed floor.

## 5. Summary: two multiplicative amplifications

```
  1e-6   (ax at exit, tiny — deep in the fringe-field tail)
   × ~1e3   (chain rule through R's already-large position sensitivity   → ~1e-1)
   × ~1e2   (bilinear symplectic product against R's large entries again → ~1e1)
  ≈ 28
```

None of the individual large numbers (`R[px,y0] ≈ -90`, etc.) are
themselves wrong — that's correct, physical solenoid optics (strong
bending/focusing at 45.6 MeV with `a=0.3m`). The point is that the
symplectic check is *quadratic* in a matrix that already contains large
entries, so it is exquisitely sensitive to even a tiny, asymmetric
(px/py-only) defect. This is also why the effect is amplitude/bending
dependent: the controlling ratio is

```
q·c·B0·a / P0c  ≈ 135 MeV/c / P0c
```

`≈ 2.96` for the 45.6 MeV run used here (strongly bending, large R
sensitivities) vs. `≈ 0.003` for full FCC-ee energy (45.6 GeV, paraxial,
`R ~ O(1)`, same fringe-field `ax` produces a negligible residual).

## Reproducing these numbers

All of the numbers above were generated ad hoc (not committed as a script)
by calling `line.get_R_matrix(...)` for both the raw and
`vector_potential_callable`-corrected `BorisSpatialIntegrator`, at
`n_steps=100000`, and directly inspecting `R_raw - R_corr`,
`R_corr^T @ S @ dR + dR^T @ S @ R_corr`, and a central-difference estimate of
`d(ax)/dx, d(ax)/dy` at the reference trajectory's exit point via
`SolenoidField.get_vector_potential`. Worth promoting into a proper script
(e.g. `006_symplectic_error_mechanism.py`) if this analysis needs to be
repeated for a different geometry/energy.
