# 15 — True and pseudo multipoles of the tilted solenoid, and their RDTs

Written 2026-11-10. It collects two claude.ai discussions that had only lived in chat:

- "Beam-frame coefficients for tilted solenoid" (2–3 Nov 2026): the true/pseudo split.
- "Where we stand" (5 Nov 2026): RDTs, literature, and the role of pseudo terms in the RDTs.

Notation follows the Overleaf text (Schüren framework):

- beam frame $(X, Y, S)$, $\zeta = X + iY$, crossing angle $\theta_b$;
- $A_m = \partial_X^{m-1}B_X$ on the beam axis, and $B_m = 0$;
- $b_S = A_0'$, with $B = -\nabla\Phi$.

Related TODO items: `paper/TODO.md` §3, "Beam-frame coefficients $A_m$", and the chromatic coupling study.

Status markers used below:

- **[derived]**: algebra shown in the chat;
- **[checked]**: verified symbolically or numerically;
- **[argued]**: a structural argument, not computed;
- **[estimate]**: an order-of-magnitude estimate.

## 1. Expansion around the beam axis: true vs pseudo

Write the potential around the beam axis as a double series in $\zeta$ and $\bar\zeta$:

$$
-\Phi = \sum_{\ell\geq0}\frac{(-1)^\ell}{4^\ell(\ell!)^2}T_0^{(2\ell)}(\zeta\bar\zeta)^\ell + \sum_{q\geq1}\sum_{\ell\geq0}\frac{(-1)^\ell q!}{4^\ell\ell!(q+\ell)!}T_q^{(2\ell)}\left(\zeta^q + \bar\zeta^q\right)(\zeta\bar\zeta)^\ell .
$$

The two kinds of term:

- **$\ell = 0$: true multipoles.** These are $T_q(\zeta^q + \bar\zeta^q) = 2T_qR^q\cos q\varphi$. They are free data, not constrained by Laplace's equation, and fixed by the magnet.
- **$\ell\geq1$: pseudo-multipoles.** Each carries a factor $(\zeta\bar\zeta)^\ell = R^{2\ell}$. They are fixed multiples of $T_q^{(2\ell)}$, the S-derivatives of the true coefficient on the same diagonal.

Only cosines appear, so all terms are skew. This follows from midplane symmetry ($B_\varphi = 0$, hence $B_m = 0$).

**Link to the $A_m$** [derived, checked]. On the midplane $\zeta = \bar\zeta = X$, which gives

$$
\frac{A_p}{p!} = \sum_{\ell=0}^{\lfloor p/2\rfloor} w_{p-2\ell}\frac{(-1)^\ell(p-2\ell)!}{4^\ell\ell!(p-\ell)!}T_{p-2\ell}^{(2\ell)}, \qquad w_0 = 1, \quad w_{q\geq1} = 2 .
$$

Solving order by order:

| p | relation | true part | pseudo part |
|---|---|---|---|
| 0 | $T_0 = A_0$ | on-axis potential | — |
| 1 | $T_1 = \frac12A_1$ | skew dipole | — |
| 2 | $T_2 = \frac14\left(A_2 + \frac12b_S'\right) = \frac14g_S$ | skew quad $g_S$ | radial $-\frac12b_S'$ (from $T_0$) |
| 3 | $T_3 = \frac{1}{12}\left(A_3 + \frac34A_1''\right)$ | skew sextupole | $-\frac34A_1''$ (from $T_1$) |

Here $g_S = A_2 + \frac12b_S'$ is the same quantity as $g_s$ in the project doc (C.3), where $\partial_XB_X = F + 2g_s$ and $\partial_YB_Y = F$.

Verification: all $c_{ab}$ with $a + b\leq4$ were computed from an exact tilted test potential ($\theta_b = 0.2$, $S = 1.3$). The recursion, the $A_p$ relation for p = 0–4, and the $T_2$ and $T_3$ formulas hold to working precision.

**Leading orders in $\theta_b$** [checked: strict expansion, error scaling verified numerically].

- $A_3 = -\frac{3\theta_b}{8}(4b_s'' + Sb_s''') + O(\theta_b^3)$.
- The $O(\theta_b)$ part of $A_3$ is entirely the pseudo part $-\frac34A_1''$. The true skew sextupole $T_3$ is $O(\theta_b^3)$.
- In general the true coefficient $T_q$ is $O(\theta_b^q)$.

The strict forms of $A_0$–$A_2$ to $O(\theta_b^2)$, with $b_s$ at $S$:

$$
A_0 = a_0 - \theta_b^2\left(\frac S2b_s + \frac{S^2}{4}b_s'\right), \qquad A_1 = \theta_b\left(b_s + \frac S2b_s'\right) + O(\theta_b^3),
$$

$$
A_2 = -\frac12b_s' + \theta_b^2\left(\frac32b_s' + \frac54Sb_s'' + \frac{3}{16}S^2b_s'''\right) + O(\theta_b^4).
$$

**Normalisation for RDT codes** [derived 2026-11-10, by hand]. The true part $2T_qR^q\cos q\varphi$ of $-\Phi$ gives $B_Y + iB_X = 2qT_qi\zeta^{q-1}$. That is a pure skew multipole in the Xsuite manual's convention (eq. 4.12), with

$$
B\rho\hat K_{q-1} = 2q!T_q .
$$

Check: q = 1 gives $A_1$, and q = 2 gives $4T_2 = g_S$.

## 2. Binomial view: what makes a term pseudo

$\partial_X^p = (\partial_\zeta + \partial_{\bar\zeta})^p$ splits $A_p$ into two kinds of term:

- **Pure terms**, $(\partial_\zeta^p + \partial_{\bar\zeta}^p)(-\Phi)$: the true multipoles.
- **Mixed terms**, containing $\partial_\zeta\partial_{\bar\zeta} = \frac14\nabla_\perp^2$: the pseudo-multipoles.

$4\partial_\zeta\partial_{\bar\zeta}\Phi = \nabla_\perp^2\Phi = -\nabla_\perp\cdot B_\perp = \partial_SB_S$. So each mixed pair measures the transverse divergence of the field, which Maxwell ties to the longitudinal variation [derived].

In gradient-matrix language:

- the trace (divergence) is pseudo;
- the symmetric traceless part is true;
- the antisymmetric (curl) part is absent, because $B = -\nabla\Phi$.

## 3. Properties of pseudo terms

1. **No independent strength.** Every pseudo term is a fixed multiple of $T_q^{(2\ell)}$, so the field near the axis is determined by the $T_q(S)$ alone. [derived generally; checked for degree ≤ 4]
2. **Parent's angular pattern.** Pseudo terms on diagonal q go as $R^{2\ell+q}\cos q\varphi$. Only the q = 0 terms are rotationally symmetric. [two-line identity]
3. **They integrate to zero.** $\nabla_\perp^2\int\Phi dS = [B_S] = 0$ for an isolated field. The integrated transverse field is therefore a pure 2D multipole field, and $\int A_m dS = \int A_m^{\rm t}dS$, e.g. $\int A_2^{\rm p} = -\frac12[A_0'] = 0$ and $\int A_3^{\rm p} = -\frac34[A_1'] = 0$. [proven for an isolated field; checked numerically, two-charge model, $\theta_b = 0.2$]
   - This holds for the whole field, not for a single fringe: at the entrance alone the radial term gives a real kick, which the exit undoes.
4. **Their dynamical effect needs longitudinal variation.** There are three routes:
   - (a) weighted integrals: integrating by parts moves the S-derivatives onto the optics functions (§5);
   - (b) second order in the field: in $x'' = ky' + \frac{k'}{2}y$, the term $\frac{k'}{2}y$ is the radial pseudo part, and together with $ky'$ it produces the solenoid focusing $(k/2)^2$ [derived, checked];
   - (c) the trajectory changing within the fringe.
5. **Rotation.** The rotating frame $u = we^{-i\theta}$ only generates terms linear in $w$. It therefore removes exactly the longitudinal field plus the linear q = 0 pseudo term, i.e. the Larmor structure. Two kinds of term stay in that frame:
   - everything the tilt introduces (q ≥ 1, true and pseudo);
   - the nonlinear q = 0 pseudo terms. [argued; that the nonlinear q = 0 terms survive unchanged is not checked]

Draft sentence for the paper (from the 3 Nov chat):

> "The pseudo-multipoles carry no independent strength: by Laplace's equation, they are fixed by the longitudinal derivatives of the true multipole coefficients and share their angular dependence. Integrated along the beam line, they vanish, so that the integrated field contains only true multipoles. Pseudo-multipoles therefore affect the beam only through the longitudinal variation of the field relative to the trajectory and the optics, or at second order in the field, as in the focusing of the solenoid itself."

## 4. Mapping onto RDTs (Xsuite physics manual, ch. 4)

Eq. (4.8) of the manual (implemented in `xt.rdt_first_order_perturbation`, `xtrack/rdt.py`) takes integrated normal and skew strengths `k{n}l` and `k{n}sl` per element. For the solenoid's true part:

- per slice, `k{q-1}sl` $= 2q!T_q dS/B\rho$;
- all `k{n}l` are 0.

The manual's $\Omega(r+t+1)$ selector then keeps only RDTs with $r + t$ odd [derived from eq. 4.8]:

| coefficient | multipole | RDTs (conjugates omitted) | meaning |
|---|---|---|---|
| $T_1$ ($A_1$) | skew dipole | $f_{0010}$, $f_{0001}$ | vertical orbit |
| $T_2$ ($g_S$) | skew quad | $f_{1001}$ ($Q_x - Q_y$), $f_{1010}$ ($Q_x + Q_y$) | difference and sum coupling; $f_{1001}$ is the RDT behind $C^-$ |
| $T_3$ | skew sextupole | $f_{2001}$, $f_{2010}$, $f_{1110}$, $f_{0021}$, $f_{0030}$ | $2Q_x\mp Q_y$, $Q_y$, $3Q_y$ |
| $T_4$ | skew octupole | $f_{3001}$, $f_{1030}$, $f_{1012}$, $f_{0031}$, … | 4th-order coupling resonances |

**Consequence** [derived]. Detuning needs $p = q$ and $r = t$, so $r + t$ is even, which a skew term can't supply. The solenoid fringe therefore gives no first-order amplitude detuning and no first-order chromaticity from any $A_m$.

**Feed-down** (manual eq. 4.20) [derived]. With the offset $\tilde b = (x_0 + D_x\delta) + i(y_0 + D_y\delta)$, one step of feed-down from a skew term $i\hat K_{m+1}$ gives:

- **real (horizontal) offset**: still skew. E.g. $A_3$ with $D_x$ gives a δ-dependent $f_{1001}$, i.e. chromatic coupling;
- **imaginary (vertical) offset**: $i\cdot i = -1$, so the term becomes normal. E.g. $A_3$ with $D_y$ or $y_0$ gives a normal gradient: β-beat ($f_{2000}$, $f_{0020}$), and with δ also Q′.

The vertical orbit and $D_y$ are what turn the solenoid's skew field into normal terms.

Not in eq. (4.20), and only partly in the code:

- the $1/(1+\delta)$ scaling of the strength;
- the δ-dependence of β and φ at the source. Near the IP this route is likely the dominant one for $g_S$, because the Montague $W$ is large there [argued].

**Limits of the 2D formula:**

- it needs the true coefficients as input, not the raw $A_m$. E.g. the trace part of $A_2$ is not a skew quadrupole;
- the longitudinal field $b_S$ is not in eq. (4.8) at all, so it needs the solenoid formula of §5;
- the 2D vertical feed-down neglects the $\partial_S^2$ terms that a 3D field brings in through Laplace's equation. This is probably small for the small $y_0$ and $D_y$ here [unchecked].

## 5. How pseudo terms enter the RDTs

**True coefficients** enter as strength times β: $h_{pqrt}\propto T_q\beta_x^{(p+q)/2}\beta_y^{(r+t)/2}$.

**Pseudo terms** enter as the parent strength times derivatives of the optics [argued in general; checked for the linear solenoid term]. A pseudo term is $T'(S)$ times a polynomial, and it contributes $\int T'wdS$ to $f_{pqrt}$, where

$$
w = \beta_x^{(p+q)/2}\beta_y^{(r+t)/2}e^{i[(p-q)\phi_x + (r-t)\phi_y]} .
$$

Integration by parts gives $-\int Tw'dS$. With $\beta' = -2\alpha$ and $\phi' = 1/\beta$, the ratio $w'/w$ is made of $\alpha/\beta$ and $i/\beta$.

**Check with the classic solenoid coupling term** [derived by hand]. In the symmetric gauge, the coupling part of the Hamiltonian, $\frac k2(yp_x - xp_y)$, becomes in normalised coordinates

$$
\frac k2\sqrt{\beta_x\beta_y}\left[\frac{\tilde y\tilde p_x}{\beta_x} - \frac{\tilde x\tilde p_y}{\beta_y} - \tilde x\tilde y\left(\frac{\alpha_x}{\beta_x} - \frac{\alpha_y}{\beta_y}\right)\right] .
$$

This has the structure of the classic driving term $\propto\sqrt{\beta_x\beta_y}S\left[\left(\frac{\alpha_x}{\beta_x} - \frac{\alpha_y}{\beta_y}\right) - i\left(\frac{1}{\beta_x}\mp\frac{1}{\beta_y}\right)\right]e^{i(\mu_x\pm\mu_y)}$ (Guignard; De Ninno & Fanelli, eq. 17). Van der Schueren et al. (IPAC'26) find the same fingerprint: $b_1'$ enters $h_{1011}$ and $h_{0111}$ with factors $(\alpha_x\pm i)$. The radial pseudo field never appears explicitly in this gauge; it is absorbed into the momentum-dependent terms.

**Gauge dependence** [argued]. Which pieces look pseudo depends on the gauge, i.e. on the canonical transformation. Two consequences:

- the one-turn RDTs observed outside the field region are gauge-invariant;
- $f_{pqrt}(s)$ evaluated inside the solenoid region depends on the representation. Xsuite evaluates it at every element, so values inside the solenoid have to be read with care.

**Size in the FCC-ee IR** [estimate]. Near the IP, $\beta\approx S^2/\beta^*$, so $\alpha/\beta\approx-1/S$ and the weights are powers of $S$. For example, $\sqrt{\beta_x\beta_y}\propto S^2$ gives $w'/w\approx2/S$ for $f_{1001}$.

- pseudo/true $\sim$ (fringe length)/$S$, which is of order one for the fringe at $S\sim1$–2.5 m;
- so pseudo terms **cannot** be dropped from the RDTs here, even though they integrate to zero in the field.

**Opportunity** [not done]. Because the drift-region weights are powers of $S$, the integration by parts can be done in closed form. Each pseudo term then becomes the parent $T_q$ times a known power of $S$. Combined with the leading-order $C_m\propto\frac{1}{S^m}\frac{d^m}{dS^m}[S^{2m}b^{(m-1)}]$ (project doc C.4), this could give the solenoid fringe's $f_{1001}$ as a closed-form integral over $b(s)$. That would be an analytic result not found in the literature checked so far.

## 6. Literature

- **S. Van der Schueren, R. De Maria, H. Huttunen, M. Migliorati, V. Korchevnyuk**, "Linear and Nonlinear Effects of Fringe Fields on the Single-Particle Dynamics in the Helium Light Ion Compact Synchrotron", IPAC'26, WEP5087.
  - First-order RDTs for s-dependent fields specified by $a_n$, $b_n$, $b_s$ (same notation as ours).
  - Derivative terms ($b_1'$) appear with $(\alpha_x\pm i)$ weights.
  - It is a dipole case: $b_s$ is defined but does not appear in the extracted RDT table.
  - The table was garbled in extraction, so check the coefficients against the PDF.
- **G. Guignard**, "The general theory of all sum and difference resonances in a three-dimensional magnetic field in a synchrotron", CERN 76-06. Also Guignard & Hagel, CERN 92-01; P. J. Bryant, CERN 94-01. Coupling coefficients including solenoids.
- **G. De Ninno, D. Fanelli**, "Multiple resonance compensation for betatron coupling and its equivalence with matrix method", arXiv:physics/9907040. Convenient statement of the solenoid driving term (eq. 17).
- **Y. Cai, Y. Nosochkov**, "Dynamical Effects Due to Fringe Field of the Magnet in Circular Accelerators", PAC'05 MPPE025, arXiv:physics/0505159.
  - Hard-edge solenoid fringe generator in the axial gauge: octupole-like and rotationally invariant.
  - The off-axis solenoid case is only mentioned qualitatively ("additional x-y couplings"). This is the gap our $A_m$ work fills.
- **A. Franchi et al.**: the RDT formalism used in the Xsuite manual (its refs. [26, 27]).
- **Venturini & Dragt** (NIM A, 1999); **Newton & Wolski** (Liverpool): generalized gradients. Newton & Wolski use the term "pseudo-multipole", but without a link to RDTs.

## 7. Open items

- [ ] Check the Van der Schueren et al. table against the PDF, and ask Silke whether a solenoid ($b_s$) version exists.
- [ ] $f_{1001}$ from $g_S$ alone, compared with the uncorrected $|C^-|$ difference VarSol vs BFE order 1 (first concrete test of §4–5).
- [ ] Closed-form drift-region integral of §5.
- [ ] Explicit check that the nonlinear q = 0 pseudo terms survive the rotation unchanged.
- [ ] Solve the $A_p$ relation at p = 4: the true skew octupole and the two pseudo pieces of $A_4$.
