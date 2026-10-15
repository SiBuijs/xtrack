# Why the phase advance across the undulator is not the tune shift

Context: `007b_sls_undulator_tune_displacement_local_optics.py` Twisses the
standalone undulator from the bare-ring optics $(\beta_0, \alpha_0)$ at its
entrance and takes

$$
\Delta\mu_{\rm local} = \mu_{\rm und} - \mu_{\rm drift}
$$

(phase advance across the undulator minus that of the drift it replaces) as
the tune shift. For 0610 this gives a value about 8× too small, with the
wrong sign of the slope:

| 0610, MK | closed-ring $\Delta Q_x$ (007a) | local $\Delta\mu_x$ (007b) | closed-ring $\Delta Q_y$ | local $\Delta\mu_y$ |
|---|---|---|---|---|
| $\delta x = -0.3$ mm | +3.44e-3 | +0.58e-3 | −2.86e-3 | −0.56e-3 |
| $\delta x = 0$       | +3.47e-3 | +0.43e-3 | −3.00e-3 | −0.42e-3 |
| $\delta x = +0.3$ mm | +3.54e-3 | +0.29e-3 | −3.13e-3 | −0.28e-3 |

This is not a bug in the script; the quantity itself is a different one.
Below: what $\Delta\mu_{\rm local}$ actually measures (§1–3), why the tune
shift is bigger (§4), the numerical checks (§5), and how to get the tune
without inserting the undulator in the ring (§6). Phases $\mu$ are in turns
(as xtrack's `mux`/`muy`), $\psi$ in radians.

---

## 1. Ingredients

Courant–Snyder optics $(\beta, \alpha)$, $\gamma = (1+\alpha^2)/\beta$. A
section with $2\times 2$ transfer matrix $M$ transports them as

$$
\beta_1 = M_{11}^2\beta_0 - 2M_{11}M_{12}\alpha_0 + M_{12}^2\gamma_0,
\qquad
\alpha_1 = -M_{11}M_{21}\beta_0 + (M_{11}M_{22}+M_{12}M_{21})\alpha_0 - M_{12}M_{22}\gamma_0,
$$

and the phase advance across it (radians), for incoming optics $(\beta,\alpha)$, is

$$
\tan\psi = \frac{M_{12}}{M_{11}\beta - M_{12}\alpha}. \tag{1}
$$

A thin focusing lens of integrated strength $KL$ is
$M = \begin{pmatrix}1 & 0\\ -KL & 1\end{pmatrix}$, so from the formulas
above it **keeps $\beta$ and changes only $\alpha$**:

$$
\beta \to \beta, \qquad \alpha \to \alpha + KL\,\beta. \tag{2}
$$

## 2. One thin kick inside a fixed-entrance section

Put a thin lens $KL$ somewhere inside the undulator. Split the undulator
into the part before the kick and the part after it, with phase advance
$\psi$ from the kick to the **undulator exit**.

- Before the kick nothing changes: same entrance optics, same elements,
  same phase advance.
- The kick itself has zero length, so it adds no phase directly. It only
  changes $\alpha$ at that point, by (2).
- After the kick, the same downstream elements are traversed, but with
  the new $\alpha$. By (1), the phase advance depends on $\alpha$:

$$
\frac{\partial\psi}{\partial\alpha}
= \cos^2\psi\,\frac{\partial\tan\psi}{\partial\alpha}
= \cos^2\psi\,\frac{M_{12}^2}{(M_{11}\beta - M_{12}\alpha)^2}
= \cos^2\psi\,\tan^2\psi = \sin^2\psi .
$$

So with $\delta\alpha = KL\beta$:

$$
\delta\psi = KL\,\beta\,\sin^2\psi
\quad\Longrightarrow\quad
\boxed{\;\delta\mu_{\rm local} = \frac{KL\,\beta}{2\pi}\sin^2\psi
= \frac{KL\,\beta}{4\pi}\,\bigl(1-\cos 2\psi\bigr)\;} \tag{3}
$$

The last step uses $2\sin^2\psi = 1-\cos2\psi$. This holds for any optics
downstream of the kick, not only drifts.

Two limiting cases make the meaning clear:

- **Kick at the exit** ($\psi = 0$): $\delta\mu_{\rm local} = 0$. The kick
  bends the beam envelope ($\alpha$ changes), but nothing is left inside the
  section for that change to turn into phase.
- **Kick a quarter-period before the exit** ($\psi = \pi/2$):
  $\delta\mu_{\rm local} = KL\beta/2\pi$, twice the tune shift (see §4).

## 3. The whole undulator

Summing (3) over the undulator, with $K(s)$ the local focusing
($K_1 + K_2 x + \dots$, or the full $K_x$, $K_y$ of
`undulator_linear_focusing_derivation.md`) and $\psi(s)$ the phase advance
from $s$ to the undulator exit:

$$
\Delta\mu_{\rm local}
= \frac{1}{4\pi}\int_{\rm und} K(s)\,\beta(s)\,\bigl(1-\cos 2\psi(s)\bigr)\,ds . \tag{4}
$$

Compare this with the first-order tune shift

$$
\Delta Q = \frac{1}{4\pi}\int_{\rm und} K(s)\,\beta(s)\,ds . \tag{5}
$$

The only difference is the weight $1-\cos 2\psi(s)$.

For 0610 the undulator is $L = 2.22$ m long and the bare phase advance
across it is about $0.08$ turns, i.e. $\psi \le 0.5$ rad. Over
$0 \le \psi \le 0.5$ the weight $1-\cos2\psi$ runs from 0 at the exit to
about 0.49 at the entrance, and averages about 0.17. That is the factor ≈ 8
in the table.

**Why the slope flips sign.** The slope $d\Delta Q/d\delta x$ comes mostly
from the sextupole feed-down $K_2\,x$. Its unweighted integral almost
cancels between the two halves of the undulator (MK, horizontal):

| | $\int K_2 \beta_x\,ds$ | with weight $1-\cos2\psi$ |
|---|---|---|
| first half  | −55.3 | −0.8 |
| second half | +53.4 | +6.8 |

The weighted sum no longer cancels, so the slope of (4) has nothing to do
with that of (5). It is a different average over the same field.

## 4. Where the rest of the tune shift goes

The tune is the phase advance per turn **along the periodic optics**, the
$(\beta, \alpha)$ that the one-turn map maps onto itself. Holding the
entrance optics at their bare values gives a periodic solution only for the
bare ring. Once the undulator is in, the kick in (2) leaves the beam with
the wrong $\alpha$ at the undulator exit. In the closed ring that mismatch
does not stay local:

- the periodic optics change all around the ring (the beta beat),
  including at the undulator entrance;
- so the phase advance through **the rest of the ring** (~39 betatron
  periods horizontally, ~15 vertically) changes as well.

$\Delta\mu_{\rm local}$ keeps only the piece accumulated inside the
undulator, with the entrance pinned. The tune shift is the sum of all
pieces, and the sum does not depend on where the kick sits. This follows
directly from the one-turn matrix. At the kick, the bare one-turn matrix is
$M_0 = \cos\mu_0\,I + \sin\mu_0\,J$ with
$J = \begin{pmatrix}\alpha & \beta\\ -\gamma & -\alpha\end{pmatrix}$ and
$\mu_0 = 2\pi Q$. Adding the lens there:

$$
\operatorname{Tr}\!\left[M_0\begin{pmatrix}1 & 0\\ -KL & 1\end{pmatrix}\right]
= 2\cos\mu_0 - KL\,(M_0)_{12}
= 2\cos\mu_0 - KL\,\beta\sin\mu_0 .
$$

With $\operatorname{Tr} = 2\cos(\mu_0+\delta\mu) \approx 2\cos\mu_0 - 2\,\delta\mu\sin\mu_0$:

$$
\delta\mu = \tfrac12 KL\beta \ \text{rad}
\quad\Longrightarrow\quad
\Delta Q = \frac{KL\,\beta}{4\pi},
$$

with no $\psi$ anywhere. The "where in the undulator" information that
(3) depends on is just how the shift is split between the undulator and
the rest of the ring. The tune only cares about the total.

In short:

> $\Delta\mu_{\rm local}$ is the part of the tune shift that happens to be
> accumulated inside the undulator when its entrance optics are frozen. The
> remainder is accumulated in the rest of the ring through the beta beat the
> undulator drives. For a short insertion ($\psi \ll 1$) that remainder is
> almost everything.

## 5. Numerical checks

**Thin-kick toy** (pure matrices; drift $L = 1.9$ m,
$\beta_0 = 3.5$, $\alpha_0 = -0.19$, $KL = 10^{-4}$, kick at $s_k$):

| $s_k$ [m] | $\psi$ [rad] | local $\delta\mu$ (matrices) | Eq. (3) | $KL\beta/4\pi$ |
|---|---|---|---|---|
| 0.0 | 0.457 | 1.0861e-05 | 1.0860e-05 | 2.79e-05 |
| 0.5 | 0.319 | 5.8968e-06 | 5.8961e-06 | 3.00e-05 |
| 1.0 | 0.193 | 2.4369e-06 | 2.4367e-06 | 3.32e-05 |
| 1.5 | 0.080 | 4.8133e-07 | 4.8132e-07 | 3.77e-05 |
| 1.9 | 0     | 0          | 0          | 4.21e-05 |

**Real undulator (MK, 0610).** Eq. (4) with the to_multipole_line()
$K_1 + K_2 x$ and the bare (drift-propagated) $\beta(s)$, $\psi(s)$, against
the local Twiss of 007b:

| $\delta x$ | Eq. (5) x | Eq. (4) x | local Twiss $\Delta\mu_x$ | Eq. (5) y | Eq. (4) y | local Twiss $\Delta\mu_y$ |
|---|---|---|---|---|---|---|
| −0.3 mm | +3.450e-3 | +5.750e-4 | +5.750e-4 | −2.894e-3 | −5.656e-4 | −5.556e-4 |
| 0       | +3.492e-3 | +4.322e-4 | +4.295e-4 | −3.025e-3 | −4.252e-4 | −4.174e-4 |
| +0.3 mm | +3.543e-3 | +2.907e-4 | +2.885e-4 | −3.164e-3 | −2.860e-4 | −2.776e-4 |

| slope | Eq. (5) | Eq. (4) | local Twiss |
|---|---|---|---|
| $d\Delta Q_x/d\delta x$ | +0.156 | −0.474 | −0.478 |
| $d\Delta Q_y/d\delta x$ | −0.450 | +0.466 | +0.463 |

Eq. (4) reproduces the local Twiss to about 1–2 %, and Eq. (5) reproduces
the closed-ring Twiss of 007a. Both effects are understood.

(Side note: the undulator is 2.22 m long, longer than the 1.908 m
placeholder span, so it reaches 0.31 m into the following drift. 007b
therefore subtracts the exact drift phase advance
$\arctan2(L, \beta_0 - \alpha_0 L)/2\pi$ rather than interpolating the
bare-ring `mux`, which is off by 1.6e-4 here because `mux` is linear in $s$
only between Twiss rows.)

## 6. Getting the tune without inserting the undulator

Keep the "Twiss initial conditions at the undulator entrance" idea, but
close the loop with matrices instead of reading off the phase advance:

$$
M_{\rm turn} = M_{\rm rest}\,M_{\rm und},
\qquad
M_{\rm rest} = M_{\rm bare,\,one\text{-}turn}\;M_{\rm drift}^{-1},
$$

all at the undulator entrance. $M_{\rm rest}$ is the bare ring minus the
drift the undulator replaces, and $M_{\rm und}$ is the undulator's transfer
matrix at this offset. Then

$$
Q = \frac{1}{2\pi}\arccos\!\left(\tfrac12\operatorname{Tr} M_{\rm turn}\right)
$$

per plane in the uncoupled $2\times2$ case, or the eigen-tunes of the
$4\times4$ matrix if the skew terms matter. This is exact, not first order,
so it should reproduce 007a's closed-ring Twiss. $\Delta\mu_{\rm local}$
can stay in the figure as a diagnostic.
