# 13 — Boris integrator in a curved frame (side topic)

Started 2026-11-04. Not part of the paper's critical path: a derivation of the
spatial Boris (drift–kick–rotation) splitting on a reference frame with
**constant** curvature h, extending the straight-frame derivation the user
wrote (Lorentz force → re-parametrization in z → vector field
F = D + K + R → Strang splitting). Worked through step by step; each step is
appended here as it is done. TODO item: `paper/TODO.md`, section
"Side topic: Boris integrator in a curved frame".

## Step 1 — Lorentz force in the curved frame (2026-11-04)

### Frame

Planar Frenet–Serret frame along a reference curve R(s) with constant
curvature h (bending in the horizontal plane, MAD/Xsuite convention: x
points outward, the reference bends towards −x for h > 0):

$$
\mathbf{r} = \mathbf{R}(s) + x \hat{\mathbf{e}}_x(s) + y \hat{\mathbf{e}}_y ,
\qquad
\frac{d\mathbf{R}}{ds} = \hat{\mathbf{e}}_s ,
\qquad
\frac{d\hat{\mathbf{e}}_s}{ds} = -h \hat{\mathbf{e}}_x ,
\qquad
\frac{d\hat{\mathbf{e}}_x}{ds} = h \hat{\mathbf{e}}_s ,
\qquad
\frac{d\hat{\mathbf{e}}_y}{ds} = 0 .
$$

(e_x, e_y, e_s) is right-handed (e_x × e_y = e_s), so the cross product has
the same component formula as in (x, y, z). Write P = P_x e_x + P_y e_y + P_s e_s
and B = B_x e_x + B_y e_y + B_s e_s (local components).

### Kinematics

R and e_x depend on t only through s, so by the chain rule

$$
\dot{\mathbf{r}} = \frac{d\mathbf{R}}{ds}\dot{s} + \dot{x} \hat{\mathbf{e}}_x + x \frac{d\hat{\mathbf{e}}_x}{ds}\dot{s} + \dot{y} \hat{\mathbf{e}}_y
= (1 + hx) \dot{s} \hat{\mathbf{e}}_s + \dot{x} \hat{\mathbf{e}}_x + \dot{y} \hat{\mathbf{e}}_y
$$

The first term (Ṙ = ṡ e_s) is the motion of the reference point. The h x ṡ e_s
term is the rotation of e_x: at fixed x > 0 the particle sits on a circle of
radius 1/h + x, so it moves faster than its reference point.

so with P = γm ṙ:

$$
\dot{x} = \frac{P_x}{\gamma m} , \qquad
\dot{y} = \frac{P_y}{\gamma m} , \qquad
\dot{s} = \frac{P_s}{\gamma m (1 + hx)} .
$$

The only change w.r.t. the straight frame is the metric factor 1 + hx in ṡ:
P_s is the momentum along e_s, but the arc length on the reference advances
more slowly than the particle moves when x > 0.

### Dynamics

The vector equation dP/dt = (q/γm) P × B holds in any frame, but the basis
vectors now rotate as the particle moves along s:

$$
\dot{\hat{\mathbf{e}}}_x = h\dot{s} \hat{\mathbf{e}}_s , \qquad
\dot{\hat{\mathbf{e}}}_s = -h\dot{s} \hat{\mathbf{e}}_x , \qquad
\dot{\hat{\mathbf{e}}}_y = 0 .
$$

Differentiating P = Σ P_i e_i gives the extra terms P_x ė_x + P_s ė_s, so

$$
\begin{aligned}
\dot{P}_x &= \frac{q}{\gamma m}\left(P_y B_s - P_s B_y\right) + \frac{h P_s^2}{\gamma m (1 + hx)} \cr
\dot{P}_y &= \frac{q}{\gamma m}\left(P_s B_x - P_x B_s\right) \cr
\dot{P}_s &= \frac{q}{\gamma m}\left(P_x B_y - P_y B_x\right) - \frac{h P_x P_s}{\gamma m (1 + hx)}
\end{aligned}
$$

t and E are unchanged: ṫ = 1, Ė = 0.

### Checks

- **h → 0** recovers the straight-frame equations exactly.
- **|P| is conserved**: P_x Ṗ_x + P_s Ṗ_s picks up h P_s² P_x − h P_x P_s² = 0
  from the new terms, and the magnetic terms vanish as before. So
  P_s = sqrt(P² − P_x² − P_y²) can still replace the P_s equation in step 2.
- **Reference orbit**: x = y = 0, P = P_s e_s, B = B_y e_y. Then Ṗ_x = 0 requires
  h = q B_y / P, i.e. h = 1/ρ with Bρ = P/q. The frame follows the ideal
  trajectory in a dipole with the correct sign.
- **Numerical check**: integrating these equations in a uniform lab field
  (arbitrary direction, h = 0.7, off-axis start) and mapping back to the lab
  frame agrees with direct Cartesian integration to 1e-13 in position and
  momentum (scratch script, not committed).

### Interpretation: curvature is an extra rotation

The rotating basis has angular velocity Ω = −h ṡ e_y (ė_i = Ω × e_i). In
components, the equation of motion is therefore

$$
\dot{\mathbf{P}} = \frac{q}{\gamma m} \mathbf{P}\times\mathbf{B} - \boldsymbol{\Omega}\times\mathbf{P}
= \frac{q}{\gamma m} \mathbf{P}\times\left(\mathbf{B} - \frac{h P_s}{q (1 + hx)} \hat{\mathbf{e}}_y\right)
$$

So the curvature acts like an extra **vertical** field
B_h = −h P_s / (q (1 + hx)), which cancels the dipole field of the reference
trajectory (B_y = h P/q at x = 0, P_s = P). This is a rotation of
(P_x, P_s) about e_y. It is the frame-curvature analogue of the R operator,
but in the P_x–P_s plane instead of P_x–P_y, and its rate depends on the state
(P_s and x), not only on the field. The fictitious terms are exactly those
needed to keep this rotation norm-preserving.

## Step 2 — s as the independent variable (2026-11-04, written by the user, checked)

d/dt = ṡ d/ds with ṡ = P_s / (γm (1 + hx)). Primes are d/ds:

$$
\begin{aligned}
x' &= (1 + hx) \frac{P_x}{P_s} , \qquad y' = (1 + hx) \frac{P_y}{P_s} , \qquad s' = 1 \cr
t' &= (1 + hx) \frac{\gamma m}{P_s} , \qquad E' = 0 \cr
P_x' &= (1 + hx) \frac{q}{P_s} \left(P_y B_s - P_s B_y\right) + h P_s \cr
P_y' &= (1 + hx) \frac{q}{P_s} \left(P_s B_x - P_x B_s\right) \cr
P_s' &= (1 + hx) \frac{q}{P_s} \left(P_x B_y - P_y B_x\right) - h P_x
\end{aligned}
$$

Rearranged, P_x' = (1 + hx)(q/P_s) P_y B_s − (1 + hx) q B_y + h P_s: on the
reference orbit the last two terms cancel for h = q B_y / P. Review points on
the user's LaTeX draft (2026-11-04): the vector form must use e_y (not e_x);
r is the particle position, not "the basis vectors", and depends on t;
"momenta" → velocities; lab XZ-plane; state the sign convention of h; one
sentence on where the h terms come from; list s′, t′, E′.

## Step 3 — vector field and the D, K, R operators (2026-11-04)

With P_s = sqrt(P² − P_x² − P_y²) eliminated (its s-equation, −h P_x included,
follows from P_x′ and P_y′):

$$
F = (1 + hx)\left(\frac{P_x}{P_s}\partial_x + \frac{P_y}{P_s}\partial_y\right) + \partial_s
+ \left[q(1 + hx)\left(\frac{P_y}{P_s}B_s - B_y\right) + hP_s\right]\partial_{P_x}
+ q(1 + hx)\left(B_x - \frac{P_x}{P_s}B_s\right)\partial_{P_y}
$$

(The user's first draft had a spurious +hP_x in the ∂_{P_y} term and no ∂_s.)
Split F = D + K + R with the curvature term in the drift:

$$
\begin{aligned}
D &= (1 + hx)\left(\frac{P_x}{P_s}\partial_x + \frac{P_y}{P_s}\partial_y\right) + \partial_s + hP_s\partial_{P_x} \cr
K &= q(1 + hx)\left(B_x\partial_{P_y} - B_y\partial_{P_x}\right) \cr
R &= \frac{q(1 + hx)B_s}{P_s}\left(P_y\partial_{P_x} - P_x\partial_{P_y}\right)
\end{aligned}
$$

D is the exact field-free motion in the curved frame (a straight lab line;
Forest's "polar drift"). K and R leave x frozen, so 1 + hx only rescales them
and they stay exactly solvable. Putting hP_s into K instead would make K
momentum-dependent and lose its exact solution.

## Step 4 — the curved Boris step (2026-11-04)

Strang splitting as in the straight frame:
exp(Δs/2 D) exp(Δs/2 K) exp(Δs R) exp(Δs/2 K) exp(Δs/2 D).

**Drift (changed).** P_x is no longer constant: (P_x, P_s) rotate by the frame
angle φ = h ℓ over a drift of length ℓ, P_y is constant. With ρ = 1/h,
intersecting the straight lab line with the rotated plane gives

$$
\begin{aligned}
P_x^{+} &= P_x\cos\varphi + P_s\sin\varphi , \qquad P_s^{+} = P_s\cos\varphi - P_x\sin\varphi , \qquad P_y^{+} = P_y \cr
x^{+} &= (\rho + x)\frac{P_s}{P_s^{+}} - \rho , \qquad
y^{+} = y + (\rho + x)\frac{\sin\varphi}{P_s^{+}}P_y , \qquad
t^{+} = t + (\rho + x)\frac{\sin\varphi}{P_s^{+}}\gamma m
\end{aligned}
$$

valid while P_s⁺ > 0; h → 0 gives the straight drift.

**Kick:** P_x −= q(1 + hx)B_y Δs/2, P_y += q(1 + hx)B_x Δs/2 (fields at the
half-drifted position), then recompute P_s.
**Rotation:** angle ω = q(1 + hx)B_s Δs/P_s,
P_x ← P_x cos ω + P_y sin ω, P_y ← −P_x sin ω + P_y cos ω.

**Checks (scratch script, not committed).** The drift map agrees with direct
integration of D to 3e-15. The full step in a non-uniform field (h = 0.4) is
second order: the error drops ×4 per halving of Δs (2.1e-3 → 8.2e-6 for
n = 10 → 160).

**Caveat: the reference orbit is not exact.** In a pure design dipole
(qB_y = hP), the D and K substeps don't cancel exactly: x = −5e-5 after 10
steps of 0.1 m at h = 0.4 (O(φ³) per step). Fix if needed: put the design
dipole into the drift (the exact sector-bend map, a circle in a uniform field,
which Xsuite already uses for bends) and kick only with B_y − hP/q.
