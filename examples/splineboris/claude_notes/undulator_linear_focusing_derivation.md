# Linear focusing on a wiggling orbit: derivation of $K_x$, $K_y$

> **Superseded for the $K_x$/$K_y$ derivation (2026-10-15).** The canonical
> derivation is now `undulator_focusing_writeup.md`, which is a fuller rewrite
> and corrects a missing $y$ in the second Hill equation. 007a, 007b and 007c
> cite that file. This note is kept for the material the write-up does not
> carry: the SB vs MK Twiss comparison (section 7), the collected list of
> approximations (section 8) and the implementation plan (section 9).

Context: `007a_sls_undulator_tune_displacement.py`, first-order tune-shift
cross-check
$$\Delta Q_{x,y} = \frac{1}{4\pi}\oint K_{x,y}(s)\,\beta_{x,y}(s)\,ds .$$
Goal: the correct $K_x(s)$, $K_y(s)$ for a 3D undulator field, including the
longitudinal field $B_s$ and the fact that the closed orbit wiggles.

Conventions: positive charge, on momentum ($\delta = 0$),
$B\rho = p/q$, $s$ the independent variable along a straight section,
$' = d/ds$. Positive $K$ is focusing in that plane.

---

## 1. Exact equations of motion

Lorentz force, $\dot{\mathbf p} = q\,\mathbf v\times\mathbf B$. With
$v_x = v_s x'$, $v_y = v_s y'$ and $d/dt = v_s\, d/ds$:

$$\frac{dp_x}{ds} = \frac{q}{v_s}\left(v_y B_s - v_s B_y\right) = q\left(y' B_s - B_y\right),$$
$$\frac{dp_y}{ds} = \frac{q}{v_s}\left(v_s B_x - v_x B_s\right) = q\left(B_x - x' B_s\right).$$

The transverse momenta are $p_x = p\,x'/h$, $p_y = p\,y'/h$ with
$h = \sqrt{1 + x'^2 + y'^2}$, and $p$ is constant in a magnetic field, so

$$\left(\frac{x'}{h}\right)' = \frac{y' B_s - B_y}{B\rho},\qquad
\left(\frac{y'}{h}\right)' = \frac{B_x - x' B_s}{B\rho}. \tag{1}$$

This is exact.

**Paraxial approximation.** The wiggle angles here are
$\sim 2\text{--}3\times10^{-4}$ rad, so $h = 1 + O(x'^2)$ and
$(x'/h)' = x'' + O(x'^2 x'')$. To first order in the angles:

$$x'' = \frac{y' B_s - B_y}{B\rho},\qquad y'' = \frac{B_x - x' B_s}{B\rho}. \tag{2}$$

Sanity checks:
- dipole $B_y>0$: $x''<0$ (standard convention);
- normal quad $B_y = G x$, $B_x = G y$: $x'' = -(G/B\rho)\,x$,
  $y'' = +(G/B\rho)\,y$, i.e. $K_x = +K_1$, $K_y = -K_1$.

---

## 2. Linearize about the closed orbit

Let $(x_w(s), y_w(s))$ be the (wiggling) closed orbit, which itself satisfies
(2), and write

$$x = x_w + u,\qquad y = y_w + v .$$

Expand every field component to first order about the orbit,
$B(x_w+u,\,y_w+v,\,s) \approx B + u\,\partial_x B + v\,\partial_y B$, with all
fields and gradients evaluated **on the orbit** $(x_w, y_w, s)$. Substitute
into (2) and subtract the orbit's own equation:

$$u'' = \frac{1}{B\rho}\Big[\big(y_w'\,\partial_x B_s - \partial_x B_y\big)\,u
\;+\;\big(y_w'\,\partial_y B_s - \partial_y B_y\big)\,v \;+\; B_s\, v'\Big],$$

$$v'' = \frac{1}{B\rho}\Big[\big(\partial_y B_x - x_w'\,\partial_y B_s\big)\,v
\;+\;\big(\partial_x B_x - x_w'\,\partial_x B_s\big)\,u \;-\; B_s\, u'\Big].$$

(The $B_s v'$ and $-B_s u'$ terms come from $y'B_s$ and $x'B_s$ with
$y' = y_w' + v'$ and $x' = x_w' + u'$.)

---

## 3. Keep the diagonal (focusing) terms

- The $v$ term in $u''$, the $u$ term in $v''$, and the $B_s v'$, $-B_s u'$
  terms **couple** the planes. The last two are Guignard's $\mp M y'$,
  $\pm M x'$ (rotation) terms. Coupling shifts the eigentunes only at second
  order (via the distance to the difference resonance), so it is dropped for
  a first-order tune shift.
- What remains has the form $u'' + K_x u = 0$, $v'' + K_y v = 0$ with

$$\boxed{K_x = \frac{\partial_x B_y - y_w'\,\partial_x B_s}{B\rho},\qquad
K_y = \frac{x_w'\,\partial_y B_s - \partial_y B_x}{B\rho}}
\qquad\text{(on the orbit)}. \tag{3}$$

For a straight orbit ($x_w' = y_w' = 0$) this reduces to $K_x = +K_1$,
$K_y = -K_1$. The $B_s$ terms are new. On a straight orbit they would be
pure coupling. On a wiggling orbit, $x_w' \neq 0$ multiplies the
*transverse gradient* of $B_s$, which makes them diagonal focusing.

**Relation to Guignard's $M^2/4$.** His focusing term $\tfrac14 M^2$, with
$M \propto B_s$, is second order in $B_s$ itself. For a planar or elliptical
undulator $B_s \approx 0$ on the orbit (on axis in the map:
$\text{rms}\,B_s \approx 5\times10^{-8}$ T), so that term vanishes. The
focusing here comes instead from his *first*-order $M x'$ term, evaluated
with $M \propto y\,\partial_y B_s$ and a nonzero orbit slope $x_w'$.

---

## 4. Useful identities (vacuum field, $\nabla\times\mathbf B = 0$, $\nabla\cdot\mathbf B = 0$)

$$\partial_y B_s = \partial_s B_y,\qquad \partial_x B_s = \partial_s B_x,\qquad
\partial_x B_y = \partial_y B_x, \tag{4}$$

and each Cartesian component is harmonic
($\nabla^2\mathbf B = \nabla(\nabla\cdot\mathbf B) - \nabla\times\nabla\times\mathbf B = 0$):

$$\partial_y^2 B_x = -\,\partial_x^2 B_x - \partial_s^2 B_x . \tag{5}$$

---

## 5. What the $B_s$ terms integrate to

Assume the field vanishes, or the orbit slope is zero, at the undulator ends,
so all boundary terms in the partial integrations vanish.

**Vertical, from $B_y$.** Using (4) and the orbit equation
$x_w'' = -B_y/B\rho$ (plus small terms):

$$\int \frac{x_w'\,\partial_y B_s}{B\rho}\,ds = \int \frac{x_w'\,\partial_s B_y}{B\rho}\,ds
= -\int \frac{x_w''\,B_y}{B\rho}\,ds = +\int \frac{B_y^2}{(B\rho)^2}\,ds .$$

This is the textbook natural vertical focusing, $\langle B_y^2\rangle/(B\rho)^2$.

**Horizontal, from $B_x$.** Using $y_w'' = +B_x/B\rho$:

$$\int \frac{-y_w'\,\partial_x B_s}{B\rho}\,ds = -\int \frac{y_w'\,\partial_s B_x}{B\rho}\,ds
= +\int \frac{y_w''\,B_x}{B\rho}\,ds = +\int \frac{B_x^2}{(B\rho)^2}\,ds .$$

This is what is currently in the code as $K_{s,x}$, $K_{s,y}$.

---

## 6. The piece the multipole $\delta K_1$ misses

The quad-like terms in (3), $\partial_x B_y$ and $\partial_y B_x$, are the
**same** gradient by (4), entering $K_x$ with $+$ and $K_y$ with $-$. On the
orbit, expand it around $y = 0$, where the fit is defined:

$$\partial_x B_y(x_w, y_w) = \partial_y B_x(x_w, y_w)
= \underbrace{\partial_x B_y(x_w, 0)}_{(B_y,1) + x_w (B_y,2) + \dots}
\;+\; y_w\,\partial_y^2 B_x + O(y_w^2).$$

By (5), $\partial_y^2 B_x = -\partial_x^2 B_x - \partial_s^2 B_x$.

- The $-\partial_x^2 B_x = -(B_x,2)$ part is what the multipole model has, as
  the skew-sextupole feed-down $-K_{2,\text{skew}}\,y$ in `deltaK1`.
- The $-\partial_s^2 B_x$ part is **absent** from any 2D multipole description.
  Its integral, partially integrating twice with $y_w'' = B_x/B\rho$:

$$\int \frac{-y_w\,\partial_s^2 B_x}{B\rho}\,ds = -\int \frac{y_w''\,B_x}{B\rho}\,ds
= -\int \frac{B_x^2}{(B\rho)^2}\,ds .$$

So the missing piece contributes

$$\int K_x \supset -\int\frac{B_x^2}{(B\rho)^2}ds\quad\text{(cancels } K_{s,x}\text{ exactly)},\qquad
\int K_y \supset +\int\frac{B_x^2}{(B\rho)^2}ds .$$

**Net effect:** the natural focusing of the *horizontal* field component $B_x$
ends up in the **vertical** plane, not the horizontal one, given the field
model's transverse profile (set by the fitted $x$-derivatives at $y = 0$).
The analogous $B_y$ expansion needs no Laplace step: $\partial_x^2 B_y$ at
$y = 0$ is the fitted $(B_y,2)$ itself. So the vertical $K_{s,y}$ term has no
hidden counterpart.

Synthetic check (pure-$B_x$ undulator, 0.4 T, $\lambda_u$ = 36 mm, 0.72 m,
everything via `SplineBoris.get_field` on the wiggling orbit), per metre:

| term | value |
|---|---|
| $\int K_{s,x}\,ds$ | $+5.7895\times10^{-4}$ |
| $\int \partial_x B_y(\text{orbit})/B\rho\,ds$ | $-5.7895\times10^{-4}$ |
| sum | $+1.9\times10^{-10}$ |
| $\int B_x^2/(B\rho)^2\,ds$ | $+5.7895\times10^{-4}$ |

---

## 7. Comparison with the SB vs MK Twiss (0610, $\Delta x = 0$)

The field map is elliptical: on axis $\text{rms}\,B_x = 0.21$ T and
$\text{rms}\,B_y = 0.59$ T, so $\langle B_x^2\rangle/\langle B_y^2\rangle = 0.131$.
MK (mean-field thick multipoles) has none of the natural focusing, so
SB − MK should equal the natural focusing alone.

| plane | prediction | SB − MK Twiss |
|---|---|---|
| x | $\langle B_x^2\rangle$ term cancels: $\approx 0$ | $1.8\times10^{-6}$ |
| y | $\langle B_y^2\rangle$: $2.907\times10^{-3}$, plus $\langle B_x^2\rangle$: $\approx 0.383\times10^{-3}$, total $3.29\times10^{-3}$ | $3.24\times10^{-3}$ |

The $\langle B_x^2\rangle$-in-$y$ value is the computed $\Delta Q_x$ from
$K_{s,x}$ ($4.37\times10^{-4}$) rescaled by $\bar\beta_y/\bar\beta_x = 3.90/4.45$.

---

## 8. Approximations, in one place

1. **Paraxial**, first order in the wiggle angle. Drops the $h$,
   $x'^2$, $x'y'$ factors in (1), including small velocity-dependent
   terms such as $2x_w'B_y\,u'$.
2. **First-order field expansion** about the orbit (linear optics).
3. **Coupling dropped**: tunes are shifted by it only at second order.
4. **First-order perturbation theory in $K$.** Only the period-average of $K$
   enters. The part of $K$ oscillating at the undulator period
   contributes at second order. This is small, but it is not tested by the
   checks above.
5. **Boundary terms** in the partial integrations are assumed zero. That is
   fine when the corrector-matched orbit has $x_w' \approx 0$ outside the field.

---

## 9. Implementation plan (not implemented yet)

For the SB case, evaluate (3) directly instead of `deltaK1` + $K_s$:
- orbit $x_w, y_w$ and slopes $x_w', y_w'$ at sub-element resolution
  (slopes and positions integrated through each element from its Twiss
  entrance values, using $B_x$, $B_y$ from `get_field`);
- $\partial_x B_y$, $\partial_y B_x$, $\partial_x B_s$, $\partial_y B_s$ by
  central finite differences of `SplineBoris.get_field` at $(x_w, y_w)$
  (exact, since the field is polynomial in $x$, $y$);
- $K_{x,y}$ averaged per element, then integrated against $\beta$ as now.

This automatically includes $K_1$, all feed-downs, the 3D ($\partial_s^2$)
terms and the $B_s$ terms, using exactly the field the tracking sees. MK
keeps the multipole `deltaK1` formula, which is exact for its own (2D) model.
