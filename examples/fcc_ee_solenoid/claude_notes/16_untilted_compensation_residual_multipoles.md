# 16 — What untilted compensation solenoids leave behind

Written 2026-11-12. Question (from the user): the detector solenoid is tilted by $\theta_b$ with respect to the beam, and the two compensation solenoids are aligned with the beam (untilted). Their strength is set so that $\int B_S\,dS = 0$ along the beam axis, which undoes the Larmor rotation. Which of the beam-frame coefficients $A_m$ of the detector are **not** cancelled by the untilted compensators, i.e. what is $A_m - a_m$?

This is exactly the setup of the analytic lattice model: `TiltedSolenoid` for the detector, and compensation solenoids of radius 3 cm at 12 m from the IP, aligned with the beam (`solenoid_params.py`, `00_overview.md` constants section).

Notation follows note 15. Status markers: **[derived]**, **[checked]** (symbolically or numerically), **[argued]**.

## 0. Short answer

1. **The compensation strength is the same as without a tilt.** $\int B_S\,dS$ along *any* straight line through the bore equals $\Phi \equiv \int b\,dz$ on the solenoid axis (Ampère's law), whatever $\theta_b$ is. The same holds for each half, IP → outside. [derived, checked]
2. **An untilted solenoid only has the rotationally symmetric part ($q = 0$).** So the compensators cancel the longitudinal field and its pseudo-terms, and nothing else. Every true multipole of the detector with $q\geq1$ (skew dipole, skew quad $g_S$, skew sextupole, …) survives **unchanged**. [derived]
3. **Integrated over the beam line, only one thing is left: a uniform skew-dipole field**
   $$\int (A_1 - a_1)\,dS = \tan\frac{\theta_b}{2}\,\Phi ,$$
   i.e. a net vertical kick. Every higher integrated coefficient, $\int(A_m - a_m)\,dS$ for $m\geq2$, is **exactly zero**, to all orders in $\theta_b$. [derived, checked against exact Biot–Savart fields]
4. **Locally (as functions of $S$) everything is still there**, starting at $O(\theta_b)$ for odd $m$ and $O(\theta_b^2)$ for the tilt part of even $m$ (table in §5). These local terms integrate to zero, but near the IP they still drive RDTs through the optics weights (note 15 §5).

For FCC-ee ($\theta_b = 15$ mrad, $B\rho = 152.1$ T m), the integrated kick is about 0.24 mrad at 2 T and 0.38 mrad at 3 T (§7). That is the vertical orbit that `on_sol_orbit_corr_{ip}` has to correct.

## 1. Setup and notation

**Frames.**
- Detector frame $(x, y, z)$: the solenoid axis is $z$, and the solenoid centre is at the IP (origin).
- Beam frame $(X, Y, S)$: the beam axis is a straight line through the IP, at angle $\theta \equiv \theta_b$ to $z$ in the horizontal plane:
  $$x = X\cos\theta - S\sin\theta, \qquad y = Y, \qquad z = S\cos\theta + X\sin\theta .$$
  On the beam axis ($X = Y = 0$): $z = S\cos\theta$ and $x = -S\sin\theta$. Short-hands: $c = \cos\theta$, $s = \sin\theta$.

**Detector solenoid.** It has on-axis field $b(z) = B_z(0,0,z)$. Define $a(z) = \int_{-\infty}^{z} b$, so $a' = b$, and the flux-like integral
$$\Phi = \int_{-\infty}^{\infty} b\,dz, \qquad \Phi_+ = \int_0^\infty b\,dz .$$

**Compensation solenoids.** Both are coaxial with the beam axis. Their combined on-axis field is $-c(S)$, with $C(S) = \int_{-\infty}^S c$. Lower-case $a_m$ are the beam-frame coefficients of the untilted solenoid with field $+c(S)$. So the compensators contribute $-a_m$, and the residual is
$$\Delta_m \equiv A_m - a_m .$$

**Coefficients.** As in note 15, with $B = -\nabla\Phi_{\rm mag}$:
- $A_0 = -\Phi_{\rm mag}$ on the beam axis, so $B_S = A_0'$ on the axis;
- $A_m = \partial_X^{m-1}B_X = \partial_X^m(-\Phi_{\rm mag})$ on the axis, for $m\geq1$;
- $B_m = 0$ (midplane symmetry).

**Assumptions.**
- The beam axis is straight through the field region (it is a drift around the IP).
- Fields are isolated: $b$, $c$ and all their derivatives vanish at $\pm\infty$.
- The near-axis (paraxial) series is used for the local formulas. The integrated results of §3 and §6 are also proven without it (Ampère, analyticity) and checked against exact fields.

## 2. Step 1 — coefficients of an untilted solenoid

The axisymmetric potential around its own axis is
$$-\Phi_{\rm mag} = \sum_{\ell\geq0}\frac{(-1)^\ell}{4^\ell(\ell!)^2}\,C^{(2\ell)}(S)\,(X^2+Y^2)^\ell .$$
On the midplane, $a_m = \partial_X^m(\cdot)|_{X=0}$ picks out $m!$ times the coefficient of $X^m$. Only even powers exist, so:
$$a_{2\ell+1} = 0, \qquad a_{2\ell} = \frac{(-1)^\ell(2\ell)!}{4^\ell(\ell!)^2}\,c^{(2\ell-1)}(S)\quad(\ell\geq1), \qquad a_0 = C(S).$$
Explicitly:

| $m$ | $a_m$ |
|---|---|
| 0 | $C(S)$ |
| 1 | 0 |
| 2 | $-\frac12c'$ |
| 3 | 0 |
| 4 | $\frac38c'''$ |
| 6 | $-\frac{5}{16}c^{(5)}$ |

In note 15's language, an untilted solenoid has only $T_0 = C$. Every $a_{2\ell}$ with $\ell\geq1$ is a **pseudo** term of $T_0$, and it has no true multipoles at all. **[derived]**

## 3. Step 2 — the compensation condition does not depend on the tilt

**Claim.** $\displaystyle\int_{-\infty}^{\infty}B_S\,dS\Big|_{\text{tilted line}} = \Phi$ for every $\theta$.

**Proof 1 (Ampère).** Close the straight line with a half-circle at infinity. The field of the coil falls off like a dipole, $|B|\sim R^{-3}$, while the arc length grows like $R$, so the arc contributes nothing. The loop links the coil current once, so $\oint B\cdot dl = \mu_0NI$. The untilted line gives the same loop, so the integral is $\Phi$. This needs only that the line passes through the bore, not through the windings. **[derived]**

**Proof 2 (series).** $\int B_S\,dS = A_0(+\infty) - A_0(-\infty)$. At both ends the off-axis terms vanish (derivatives of $b$ decay), so this is $a(\infty) - a(-\infty) = \Phi$. **[derived]**

**Per side.** At the IP the beam axis is *on* the solenoid axis ($r = 0$), so $A_0(0) = a(0)$ exactly. Hence $\int_0^\infty B_S\,dS = a(\infty) - a(0) = \Phi_+$, also for every $\theta$. **[derived, checked to $O(\theta^5)$]**

**Consequence.** The compensators need $\int c\,dS = \Phi$ (and $\int_0^\infty c = \Phi_+$ per side), exactly as without a tilt. The tilt does change the *profile* of $B_S$ along the beam (§5, $\Delta_0'$), but not its integral.

Numerically: for a single current loop and a thick coil, at $\theta$ = 0.015, 0.2 and 0.5, $\int B_S\,dS/\Phi = 1$ to $3\times10^{-6}$. The deviation is the truncation of the integral at $|S| = 400$. **[checked]**

## 4. Step 3 — what an untilted compensator can and cannot cancel

In note 15's expansion, the field near the beam axis is fixed by the true coefficients $T_q(S)$. The $A_p$ are linear combinations of them (note 15 §1, extended here to $p = 4$):

| $p$ | $A_p$ in terms of $T_q$ |
|---|---|
| 0 | $T_0$ |
| 1 | $2T_1$ |
| 2 | $4T_2 - \frac12T_0''$ |
| 3 | $12T_3 - \frac32T_1''$ |
| 4 | $48T_4 - 4T_2'' + \frac38T_0''''$ |

The $p = 4$ row closes the last open item of note 15 §7. It follows from the general relation there, and it was checked symbolically against the exact tilted potential: $A_4 - (-4T_2'' + \frac38T_0'''')$ comes out $O(\theta^4)$, as $48T_4$ must. **[derived, checked]**

These relations are linear, so they also hold for the residual. The compensators contribute only to $T_0$, by §2. Therefore:
$$T_0^{\rm res} = A_0 - a_0, \qquad T_q^{\rm res} = T_q^{\rm det}\quad(q\geq1).$$

**In words:** the untilted compensator cancels the longitudinal field and the rotationally symmetric pseudo-terms that come with it, i.e. the Larmor structure. Every multipole that the tilt creates ($q\geq1$, true and pseudo) passes through untouched. **[derived]**

## 5. Step 4 — the residuals as functions of $S$

**Exact detector coefficients.** Restricted to the midplane, the detector potential is $F(x,z) = \sum_\ell\frac{(-1)^\ell}{4^\ell(\ell!)^2}a^{(2\ell)}(z)x^{2\ell}$. Moving along $X$ in the beam frame is the directional derivative $D = c\,\partial_x + s\,\partial_z$, so $A_m(S) = D^mF$ evaluated at $(x, z) = (-Ss, Sc)$:
$$A_m(S) = \sum_{j=0}^{m}\sum_{\ell\geq j/2}\binom{m}{j}c^j s^{m-j}\,\frac{(-1)^\ell(2\ell)!}{4^\ell(\ell!)^2(2\ell-j)!}\,a^{(2\ell+m-j)}(Sc)\,(-Ss)^{2\ell-j} .$$

**Residuals.** Expanding in $\theta$ (sympy, `16_scripts/16a_tilted_solenoid_series.py`, re-expanding $a^{(n)}(Sc)$ about $S$) and subtracting §2 gives the list below. All functions are evaluated at $S$, primes are $d/dS$, and the upper-case terms are the leading orders. **[checked]** The $A_0$–$A_3$ terms agree with note 15; the $O(\theta^3)$ and $O(\theta^4)$ terms are new.

$$\boxed{\Delta_0 = A_0 - a_0 = \big[a(S) - C(S)\big] - \theta^2\Big(\tfrac{S}{2}b + \tfrac{S^2}{4}b'\Big) + \theta^4\Big(\tfrac{S}{24}b + \tfrac{5S^2}{24}b' + \tfrac{S^3}{8}b'' + \tfrac{S^4}{64}b'''\Big) + O(\theta^6)}$$

The residual longitudinal field is $\Delta_0' = (b - c) - \theta^2\left(\frac12b + Sb' + \frac14S^2b''\right) + O(\theta^4)$. Its integral is zero by §3. Its running integral $\Delta_0(S)/(2B\rho)$ is the local rotation angle, which no longer cancels point by point.

$$\boxed{\Delta_1 = A_1 = \theta\Big(b + \tfrac{S}{2}b'\Big) - \theta^3\Big(\tfrac16b + \tfrac56Sb' + \tfrac12S^2b'' + \tfrac1{16}S^3b'''\Big) + O(\theta^5)}$$

$$\boxed{\Delta_2 = A_2 - a_2 = -\tfrac12(b - c)' + \theta^2\Big(\tfrac32b' + \tfrac54Sb'' + \tfrac3{16}S^2b'''\Big) + O(\theta^4)}$$

$$\boxed{\Delta_3 = A_3 = -\theta\Big(\tfrac32b'' + \tfrac38Sb'''\Big) + \theta^3\Big(\tfrac{11}{4}b'' + \tfrac{23}{8}Sb''' + \tfrac34S^2b'''' + \tfrac5{96}S^3b^{(5)}\Big) + O(\theta^5)}$$

$$\boxed{\Delta_4 = A_4 - a_4 = \tfrac38(b - c)''' - \theta^2\Big(\tfrac{15}{4}b''' + \tfrac{27}{16}Sb'''' + \tfrac5{32}S^2b^{(5)}\Big) + O(\theta^4)}$$

The $\theta^0$ parts of $\Delta_0$, $\Delta_2$ and $\Delta_4$ are the untilted pseudo-terms of the net on-axis field $b - c$. They vanish only if $c$ equals $b$ point by point, which it doesn't: the compensators sit 12 m away. The odd residuals are entirely detector, because $a_1 = a_3 = 0$.

**The true part**, which is what the compensator cannot touch (§4), has a closed form at leading order **[checked]**:
$$T_q^{\rm det} = \frac{\theta^q}{4^q(q!)^2\,S^q}\,\frac{d^q}{dS^q}\Big[S^{2q}\,b^{(q-1)}(S)\Big] + O(\theta^{q+2}) .$$
This is the $C_m$ form of the project doc (C.4), now with the normalisation. Explicitly:
- $2T_1 = \frac{\theta}{2S}(S^2b)'$;
- $g_S = 4T_2 = \frac{\theta^2}{16S^2}(S^4b')''$;
- $12T_3 = \frac{\theta^3}{192S^3}(S^6b'')'''$;
- $48T_4 = \frac{\theta^4}{3072S^4}(S^8b''')''''$.

## 6. Step 5 — integrated residuals over the whole line

**Main result.** For a coaxial coil with flux $\Phi$, the field integrated along any straight line through the bore, with direction $\hat S$, is
$$\int B\,dS = \Phi\,\frac{\hat z + \hat S}{1 + \hat z\cdot\hat S} .$$
The integrated field vector lies along the bisector of the solenoid axis and the beam direction. Its components are:
- $\int B_S = \Phi$;
- $\int B_X = \Phi\tan\frac\theta2$;
- $\int B_Y = 0$;

and they are the same for **every** parallel line through the bore. So the integrated transverse field is uniform, with no gradient of any order. **[derived, checked]**

**Derivation, step by step.**

1. Write $D$ in terms of the solenoid-axis derivative and the derivative along the beam, $T = \partial_S = -s\,\partial_x + c\,\partial_z$. Since $\partial_z = s\,D + c\,T$,
   $$D = \frac{\partial_z - c\,T}{s} .$$
2. $T$-derivatives integrate to boundary terms. So
   $$\int A_1\,dS = \frac1s\Big[\int\partial_zF\,dS - c\,[F]_{-\infty}^{\infty}\Big], \qquad [F] = \Phi .$$
3. On the line, $\partial_zF = \sum_\ell k_\ell\,b^{(2\ell)}(Sc)\,(Ss)^{2\ell}$, with $k_\ell = \frac{(-1)^\ell}{4^\ell(\ell!)^2}$. Substitute $u = Sc$ and integrate by parts $2\ell$ times, using $\int u^nb^{(n)}du = (-1)^nn!\,\Phi$:
   $$\int\partial_zF\,dS = \frac{\Phi}{c}\sum_\ell\frac{(-1)^\ell(2\ell)!}{4^\ell(\ell!)^2}\tan^{2\ell}\theta = \frac{\Phi}{c}(1 + \tan^2\theta)^{-1/2} = \Phi .$$
   This is the binomial series of $(1 + t^2)^{-1/2}$. So $\int B_z\,dS = \Phi$, which is a separate statement from $\int B_S\,dS = \Phi$.
4. Combine:
   $$\int A_1\,dS = \Phi\,\frac{1 - \cos\theta}{\sin\theta} = \Phi\tan\frac\theta2 .$$
   Expanded, this is $\Phi(\theta/2 + \theta^3/24 + \dots)$, which matches the integral of the $\Delta_1$ series term by term.
5. For $m\geq2$: $D^m = s^{-m}\sum_j\binom mj(-c)^jT^j\partial_z^{m-j}$. The terms with $j\geq1$ integrate to $[T^{j-1}\partial_z^{m-j}F] = 0$, because derivatives of the field vanish at $\pm\infty$; only $j = 1$, $m = 1$ would leave $[F]$. The $j = 0$ term gives $\int u^{2\ell}b^{(2\ell+m-1)}du = (2\ell)!\,[b^{(m-2)}] = 0$. Hence
   $$\int A_m\,dS = 0 \qquad (m\geq2), \quad\text{to all orders in }\theta .$$

**Second argument (no series).**
- Let $G = \int B_\perp\,dS$ over a family of parallel lines, as a 2D field in $(X, Y)$. It is curl-free, since $\int(\nabla\times B)_S = 0$, and divergence-free, since $\nabla_\perp\cdot G = -[B_S] = 0$. So $G_X - iG_Y$ is an analytic function of $\zeta = X + iY$ (note 15 §3.3: the integrated field is a pure 2D multipole field).
- Every midplane line crosses the solenoid axis, at a different $z$. The derivation above never used *where* it crosses, so $G_X(X, 0) = \Phi\tan\frac\theta2$ on a whole segment.
- An analytic function that is constant on a segment is constant everywhere. So $G = (\Phi\tan\frac\theta2, 0)$ for every line through the bore. **[argued]**

**Numerical check [checked].** The test fields are exact current-loop fields (elliptic integrals), with $\mu_0I = 1$: a single loop, and a thick coil of 15 loops (radii 0.8–1.2, $z_0\in[-0.6, 0.6]$). The test lines have $\theta$ = 0.015, 0.2 and 0.5, and offsets $(X_0, Y_0)\in\{0, 0.1, 0.2\}\times\{0, 0.1\}$.
- $\int B_S$ and $\int B_z$ equal 1;
- $\int B_X/\tan(\theta/2)$ equals 1, to $3\times10^{-6}$, the same for all offsets (so no gradient);
- $\int B_Y < 10^{-13}$.

Script: `16_scripts/16c_current_loop_check.py`.

**Applied to the residual.** The compensators are coaxial with the beam ($\theta = 0$ for them). Their integrated contribution is therefore $-\Phi_c\tan 0 = 0$ for $m = 1$, and 0 for $m\geq2$. With $\Phi_c = \Phi$:

| residual | integrated over the whole line |
|---|---|
| $\int\Delta_0'\,dS = \int(B_S^{\rm det} - c)\,dS$ | $0$ (compensation condition, §3) |
| $\int\Delta_1\,dS$ | $\Phi\tan\frac{\theta}{2}$ |
| $\int\Delta_m\,dS$, $m\geq2$ | $0$ exactly |
| $\int T_q^{\rm det}\,dS$, $q\geq2$ (true part) | $0$ exactly |

**Generalisation.** For compensators tilted by $\theta_c$ relative to the beam, the leftover is $\Phi(\tan\frac{\theta}{2} - \tan\frac{\theta_c}{2})$. So compensators **coaxial with the detector** ($\theta_c = \theta$) would leave no integrated kick at all. The leftover kick exists *because* the compensators are aligned with the beam. **[derived]**

## 7. Step 6 — per side (IP → outside)

There is one compensator on each side, and the beam optics differ on the two sides of a drift, so the half-line integrals are also useful. Let the compensator field and its derivatives vanish at the IP (true here: it sits at 12 m). Then $\int_0^\infty a_m\,dS = 0$ for $m\geq1$, and the per-side residual is the detector's own half-integral. This was done exactly or to $O(\theta^5)$ (`16_scripts/16b_half_line_integrals.py`) with $\int_0^\infty S^nb^{(k)}dS$ rules; $b^{(k)}(0)$ is at the IP.

| | $\int_0^\infty(\cdot)\,dS$ | leading order |
|---|---|---|
| $B_S$ | $\Phi_+$ (exact) | — |
| $\Delta_1 = A_1$ | $\Phi_+\tan\frac\theta2$ (exact) | $\frac\theta2\Phi_+$ |
| $\Delta_2$ | $b(0)\,\dfrac{\cos^2\theta + \cos\theta - 1}{1 + \cos\theta}$ | $b(0)\left(\frac12 - \frac58\theta^2\right)$ |
| $g_S = 4T_2$ (true skew quad) | $-b(0)\,\dfrac{(1 - \cos\theta)(2 + \cos\theta)}{2(1 + \cos\theta)}$ | $-\frac38\theta^2\,b(0)$ |
| $\Delta_3 = A_3$ | $\propto b'(0)$: $\frac98\theta b'(0) + O(\theta^3)$ | 0 for a symmetric $b$ |
| $\Delta_4$ | $\propto b''(0)$: $-\frac38b''(0) + \frac{19}{8}\theta^2b''(0) + \dots$ | — |

Notes on the table:
- The $\frac12b(0)$ in $\Delta_2$ and the $b''(0)$ in $\Delta_4$ are pseudo-terms. They appear only because the line is cut at the IP, where the field is maximal, and the other half cancels them. They are not a physical "kick at the IP".
- The true skew quad is odd in $S$ for a symmetric $b$: $-\frac38\theta^2b(0)$ on one side and $+\frac38\theta^2b(0)$ on the other. Its total is zero, but it does not drop out of $f_{1001}$, because the phase advance is odd about the IP (note 15 §5).

## 8. Numbers for the FCC-ee model

Inputs: $\theta_b = 15$ mrad and $B\rho = 45.6\,{\rm GeV}/c = 152.1$ T m. Take $\Phi\approx B_0L$, which is exact for a current-sheet solenoid if $B_0$ is the long-solenoid field. With $L = 2.46$ m (2 T) and 2.60 m (3 T):

| case | $\Phi$ [T m] | $\Phi\tan(\theta_b/2)$ [T m] | vertical kick [mrad] | per side [mrad] |
|---|---|---|---|---|
| 2 T | 4.92 | 0.0369 | 0.24 | 0.12 |
| 3 T | 7.80 | 0.0585 | 0.38 | 0.19 |

- Per-side true skew quad, $\frac38\theta_b^2b(0)/B\rho$: about $1.1\times10^{-6}$ m⁻¹ at 2 T, $1.7\times10^{-6}$ m⁻¹ at 3 T, integrated. This is small, but it sits at the high-β positions near the IP.
- Both sides kick the same way, because $A_1$ is even in $S$ for a symmetric $b$. So the kick does not cancel across the IP. It is the source of the IR vertical orbit, and of the vertical dispersion that comes with it, that the orbit correctors (`on_sol_orbit_corr_{ip}`) remove. **[argued, not compared with the lattice yet]**

## 9. Open items

- [ ] Compare the 0.24 mrad (2 T) with the corrector strengths that 004c actually sets in `on_sol_orbit_corr_{ip}`. They should match to the extent that the orbit correction is local.
- [ ] Check that 004a's numerically integrated `main_integral` (along the tilted beam axis) equals $B_0L$, independent of θ, as §3 says. If it is evaluated over a finite window, the difference is the window truncation.
- [ ] RDT size of the local residuals in §5 (feeds note 15 §7, the $f_{1001}$ item).
