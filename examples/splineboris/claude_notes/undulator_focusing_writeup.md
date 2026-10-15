# Linear focusing of the X11MA undulator

The tune shift $\Delta Q_{x,y}$ in the presence of quadrupole errors $K_{x,y}(s)$ is
$$\Delta Q_{x,y} = \frac{1}{4\pi}\oint K_{x,y}(s)\,\beta_{x,y}(s)\,ds .$$
The $K_{x,y}(s)$ enters starting from Hill's Equation as the term that is linear in $x$ and $y$:
$$x'' + K_x(s)\,x = 0\quad\quad\quad y''+ K_y(s)\,y = 0$$
**Correction: the $y$ was missing in the second Hill's equation.**

In the undulator, we have three separate orbits:
- The reference orbit, which is a straight line through the undulator.
- The closed orbit (CO), which follows a Lissajous-like trajectory through the undulator.
- The particle orbit, which in general deviates from the CO.

If we want to study the optics of the undulator, we need to study the effects of small perturbations w.r.t. the CO, not just the reference orbit.

## 1. Equation of Motion
We start with the Lorentz Force:
$$\dot{\mathbf{p}} = q \mathbf{v} \times \mathbf{B}$$
Per transverse component:
$$\frac{\mathrm{d} p_x}{\mathrm{d} t} = q\left( v_yB_s - v_s B_y \right)$$
$$\frac{\mathrm{d} p_y}{\mathrm{d} t} = q\left( v_sB_x - v_x B_s \right)$$
If we reparametrize to $s$, we have: $\mathrm{d}/\mathrm{d}t = \dot{s} \ \mathrm{d}/\mathrm{d}s$. Now, $\dot{s} = v_s$ and letting primes indicate derivatives w.r.t. $s$ we can write:
$$\frac{\mathrm{d} p_x}{\mathrm{d} s} = \frac{q}{v_s} \left( v_yB_s - v_s B_y \right)$$
$$\frac{\mathrm{d} p_y}{\mathrm{d} s} = \frac{q}{v_s} \left( v_sB_x - v_x B_s \right)$$
Then, with $v_x = v_s x'$ and $v_y = v_s y'$, we can write:
$$p_x' = q \left( y'B_s - B_y \right)$$
$$p_y' = q \left( B_x - x' B_s \right)$$
In the absence of curvature, the momenta are related to the angles by
$$p_x = \gamma m v_x \quad \Rightarrow \quad p_x' = \gamma m v_x' = \gamma m v_s' x' + \gamma m v_s x''$$
and similar for $p_y$. In the **paraxial approximation**, we note that
$$v_s = \sqrt{v^2 - v_x^2 - v_y^2} \approx v \left(1 - \frac{v_x^2 + v_y^2}{2v^2}\right)$$
This shows that, to first order in the angles, $v_s = v$ is constant and we can omit $v_s'$. Thus:
$$p_x' \approx \gamma m v\, x''$$
$$p_y' \approx \gamma m v\, y''$$
$$p_s \approx \gamma m v = p$$
The equations of motion then become
$$x'' = \frac{q}{p} (y'B_s - B_y)$$
$$y'' = \frac{q}{p} (B_x - x'B_s)$$
We identify $p/q = B\rho$ as the rigidity and finally write

**Correction: the rigidity is $B\rho = p/q$, so $q/p = 1/B\rho$ (not $q/p$ itself).**
$$x'' = \frac{1}{B\rho} (y'B_s - B_y)$$
$$y'' = \frac{1}{B\rho} (B_x - x'B_s)$$
In a normal focusing quadrupole, we'd have
$$B_y = G x \quad\quad\quad B_x = G y$$
where $G = \partial B_y/\partial x$.

**Correction: with $B_y = -Gx$, $\partial B_y/\partial x$ would be $-G$ and the horizontal plane would come out defocusing. For $G = \partial_x B_y > 0$ the field is $B_y = Gx$, $B_x = Gy$ (curl-free: $\partial_y B_x = \partial_x B_y$).**

We would then get:
$$x'' + \frac{G}{B\rho}x = 0$$
$$y'' - \frac{G}{B\rho}y = 0$$
and we identify
$$K_x = -K_y = \frac{G}{B\rho}$$
which has units of $1/\mathrm{m}^2$.

## 2. Linearize Around the Orbit
We will study small perturbations around the CO. To this end, we write $x$ and $y$ as the coordinates of the CO, $x_\mathrm{CO}$ and $y_\mathrm{CO}$, with the addition of a small perturbation w.r.t. this orbit, $u$ and $w$:
$$x = x_\mathrm{CO} + u$$
$$y = y_\mathrm{CO} + w$$

The EoMs then become:
$$x_\mathrm{CO}'' + u'' = \frac{1}{B\rho} \left( (y_\mathrm{CO}' + w') B_s - B_y \right)$$
$$y_\mathrm{CO}'' + w'' = \frac{1}{B\rho} \left( B_x - (x_\mathrm{CO}' + u') B_s \right)$$
Note that the magnetic fields are evaluated at the offset coordinates as well. We can then write:
$$B_i(x_\mathrm{CO}+u,\,y_\mathrm{CO}+w,\,s) \approx B_i + u\,\partial_x B_i + w\,\partial_y B_i$$
where $B_i$ and its derivatives on the right-hand side are evaluated **on the closed orbit**, $(x_\mathrm{CO}, y_\mathrm{CO}, s)$.

**Correction: the arguments read $(x_w+u,\,y_w+v)$; in this notation they are $(x_\mathrm{CO}+u,\,y_\mathrm{CO}+w)$. That the derivatives are taken on the CO, not on the reference orbit, is the key to Sections 4 and 5.**

This gives us
$$x_\mathrm{CO}'' + u'' = \frac{1}{B\rho} \left( (y_\mathrm{CO}' + w') (B_s + u\partial_xB_s + w\partial_yB_s) - B_y - u\partial_xB_y - w\partial_yB_y \right)$$
$$y_\mathrm{CO}'' + w'' = \frac{1}{B\rho} \left( B_x + u\partial_xB_x + w\partial_yB_x - (x_\mathrm{CO}' + u') (B_s + u\partial_xB_s + w\partial_yB_s) \right)$$
**Correction: the second equation had $u\,\partial_xB_y + w\,\partial_yB_y$; the expansion of $B_x$ gives $u\,\partial_xB_x + w\,\partial_yB_x$ (as you already had after the subtraction below).**

We can subtract the EoM of the CO, which leaves us with
$$u'' = \frac{1}{B\rho} \left(  w' (B_s + u\partial_xB_s + w\partial_yB_s) + y_\mathrm{CO}' (u\partial_xB_s + w\partial_yB_s) - u\partial_xB_y - w\partial_yB_y \right)$$
$$w'' = \frac{1}{B\rho} \left( u\partial_xB_x + w\partial_yB_x - u' (B_s + u\partial_xB_s + w\partial_yB_s) - x_\mathrm{CO}' (u\partial_xB_s + w\partial_yB_s) \right)$$
To linear order we drop the products of perturbations ($w'u$, $w'w$, ...). Of the remaining terms, those proportional to $w$ in the $u$-equation (and vice versa) only couple the planes, and those proportional to $u'$, $w'$ multiply $B_s$ on the CO, which is negligible here. We can identify that the terms proportional to $u$ and $w$ in their respective EoMs are the focusing terms. These are:
$$K_u = \frac{1}{B\rho} \left( \partial_x B_y - y_\mathrm{CO}' \partial_x B_s \right)$$
$$K_w = \frac{1}{B\rho} \left( x_\mathrm{CO}' \partial_y B_s - \partial_y B_x \right)$$
These are the usual quadrupole focusing terms, with the addition of longitudinal fields.

## 3. Integrated Effect
If we want to know the effect over the entire undulator, we calculate:
$$\int_0^L K_u\ \mathrm{d}s = \frac{1}{B\rho} \int_0^L \left( \partial_x B_y - y_\mathrm{CO}' \partial_x B_s \right)\ \mathrm{d}s = k_1L - \frac{1}{B\rho} \int_0^L y_\mathrm{CO}'\partial_x B_s \ \mathrm{d}s$$
Here:
$$k_1 = \frac{1}{B\rho L} \int_0^L \partial_x B_y\big|_\mathrm{CO}\ \mathrm{d}s$$
is the integrated normal quadrupole strength **seen by the closed orbit**.

**Correction: $\partial_x B_y$ is evaluated on the CO (Section 2), so this $k_1$ is not the quadrupole content of the field map on the reference orbit. For the X11MA map the on-axis quadrupole is zero, yet $k_1$ on the CO is the largest term of all (Sections 4 and 6).**

We are more interested in the other integral. From Maxwell's Equations in vacuum ($\nabla\times\mathbf{B} = 0$), we have
$$\partial_x B_s = \partial_s B_x \quad\quad\quad \partial_y B_s = \partial_s B_y$$
so we write:
$$\int_0^L K_u\ \mathrm{d}s = k_1L - \frac{1}{B\rho} \int_0^L y_\mathrm{CO}'\partial_s B_x\ \mathrm{d}s$$
Then, from integration by parts, we can write:
$$\int_0^L y_\mathrm{CO}'\partial_s B_x\ \mathrm{d}s = \left[ y_\mathrm{CO}' B_x \right]_0^L - \int_0^L y_\mathrm{CO}'' B_x\ \mathrm{d}s $$
**Correction: the boundary term is $[y_\mathrm{CO}' B_x]$, not $[y_\mathrm{CO}' \partial_s^2 B_x]$ ($u = y_\mathrm{CO}'$, $\mathrm{d}v = \partial_s B_x\,\mathrm{d}s$, so $v = B_x$).**

Two observations:
- The field vanishes outside the undulator (and the correctors make $y_\mathrm{CO}' = 0$ at its ends), so the first term vanishes. **Correction: it is $y_\mathrm{CO}'B_x$ that must vanish at the edges, not $y_\mathrm{CO}$ and the field gradient.**
- In our EoM, $y_\mathrm{CO}'' \approx B_x/B\rho$ to zeroth order. With this, we can write:
$$\int_0^L y_\mathrm{CO}'\,\partial_s B_x\ \mathrm{d}s = - \frac{1}{B\rho} \int_0^L B_x^2 \ \mathrm{d}s  = - \frac{L\,\langle B_x^2\rangle}{B\rho}, \qquad \langle B^2\rangle \equiv \frac{1}{L}\int_0^L B^2\,\mathrm{d}s$$
**Correction: the left-hand side is $\int y_\mathrm{CO}'\,\partial_s B_x$ (not $\int y_\mathrm{CO}' B_x$), and a factor $L$ was missing.**

Therefore, we get:
$$\int_0^L K_u\ \mathrm{d}s = k_1L + \frac{L\,\langle B_x^2\rangle}{(B\rho)^2}$$
$$\int_0^L K_w\ \mathrm{d}s = - k_1L + \frac{L\,\langle B_y^2\rangle}{(B\rho)^2}$$
**Correction: the natural-focusing terms carry $(B\rho)^2$: one $1/B\rho$ from $K$ and one from $y_\mathrm{CO}''$ (dimension check: $\int K\,\mathrm{d}s$ is in $1/\mathrm{m}$).** The $-k_1L$ in the vertical plane follows from $\partial_y B_x = \partial_x B_y$ (curl-free field).

For $k_1 > 0$, the sign before $k_1$ indicates that we have the usual quadrupole focusing (defocusing) in the horizontal (vertical) plane. The second term is the undulator's *natural focusing*: the longitudinal field met by the wiggling CO focuses the plane perpendicular to the field component that drives the wiggle.

## 4. Field Expansion Around the Reference Orbit
To see what $k_1$ on the CO contains, we need the field off the reference orbit. SplineBoris uses the midplane expansion of `bpmeth` (`GeneralVectorPotential`), whose inputs are the fields on the midplane $y = 0$ as polynomials in $x$, with on-axis coefficients that depend on $s$:
$$B_y(x, 0, s) = b_1(s) + b_2(s)\,x + \tfrac12 b_3(s)\,x^2 + \dots \qquad B_x(x, 0, s) = a_1(s) + a_2(s)\,x + \tfrac12 a_3(s)\,x^2 + \dots$$
so $b_1 = B_y$, $b_2 = \partial_x B_y$, $b_3 = \partial_x^2 B_y$ on axis (and the same for $a_n$ and $B_x$), plus the on-axis $b_s(s) = B_s$. These are the components the TubeFitter extracts from the field map. Everything off the midplane follows from Laplace's equation, $\nabla^2\phi = 0$ with $\mathbf{B} = \nabla\phi$, through the recurrence $\phi_{i+2} = -(\partial_x^2 + \partial_s^2)\phi_i$ in $\phi = \sum_i \phi_i\, y^i/i!$. To second order in $x$ and $y$:
$$
\begin{aligned}
B_x &= a_1 + a_2\,x + b_2\,y + \tfrac12 a_3\,x^2 + b_3\,xy - \tfrac12\big(a_3 + a_1''\big)\,y^2 \\[2pt]
B_y &= b_1 + b_2\,x - \big(a_2 + b_s'\big)\,y + \tfrac12 b_3\,x^2 - \big(a_3 + a_1''\big)\,xy - \tfrac12\big(b_3 + b_1''\big)\,y^2 \\[2pt]
B_s &= b_s + a_1'\,x + b_1'\,y + \tfrac12 a_2'\,x^2 + b_2'\,xy - \tfrac12\big(a_2' + b_s''\big)\,y^2
\end{aligned}
$$
Two things stand out:
- The vertical curvature of $B_y$ is not an input. It follows from Laplace's equation:
$$\partial_y^2 B_y = -\big(\partial_x^2 B_y + \partial_s^2 B_y\big) = -\big(b_3 + b_1''\big)$$
and similarly $\partial_y^2 B_x = -(a_3 + a_1'')$.
- The linear terms of $B_s$ are the $s$-derivatives of the on-axis dipole fields, $\partial_x B_s = a_1'$ and $\partial_y B_s = b_1'$: the Maxwell relations of Section 3. $B_s$ contains no $b_3$ to this order.

## 5. Focusing on the Closed Orbit
We now evaluate $K_u$ and $K_w$ on the CO, keeping terms linear in $(x_\mathrm{CO}, y_\mathrm{CO})$. From the expansion,
$$\partial_x B_y\big|_\mathrm{CO} = b_2 + b_3\,x_\mathrm{CO} - \big(a_3 + a_1''\big)\,y_\mathrm{CO}, \qquad \partial_x B_s\big|_\mathrm{CO} \approx a_1', \qquad \partial_y B_s\big|_\mathrm{CO} \approx b_1'$$
and $\partial_y B_x = \partial_x B_y$. The CO wiggles in phase with the field ($x_\mathrm{CO} \propto b_1$, $y_\mathrm{CO} \propto a_1$), so the feed-down terms $b_3 x_\mathrm{CO}$ and $a_3 y_\mathrm{CO}$ do *not* average out over the undulator, even when the map has no quadrupole ($b_2 = 0$). This is the $k_1$ of Section 3:
$$k_1 L = \frac{1}{B\rho}\int_0^L \Big( b_2 + b_3\,x_\mathrm{CO} - \big(a_3 + a_1''\big)\,y_\mathrm{CO} \Big)\,\mathrm{d}s .$$
Integrating the $B_s$ terms by parts as in Section 3, but now keeping them in terms of the CO position ($\int y_\mathrm{CO}'\,a_1'\,\mathrm{d}s = -\int y_\mathrm{CO}\,a_1''\,\mathrm{d}s$ and $\int x_\mathrm{CO}'\,b_1'\,\mathrm{d}s = -\int x_\mathrm{CO}\,b_1''\,\mathrm{d}s$), the $a_1''$ terms cancel in the horizontal plane and we get
$$B\rho\int_0^L K_u\,\mathrm{d}s = \int_0^L \Big( b_2 + b_3\,x_\mathrm{CO} - a_3\,y_\mathrm{CO} \Big)\,\mathrm{d}s$$
$$B\rho\int_0^L K_w\,\mathrm{d}s = \int_0^L \Big( -b_2 - \big(b_3 + b_1''\big)\,x_\mathrm{CO} + \big(a_3 + a_1''\big)\,y_\mathrm{CO} \Big)\,\mathrm{d}s$$
With Section 4 the combinations in brackets are curvatures of the field, which gives a compact form:
$$B\rho\int_0^L K_u\,\mathrm{d}s = \int_0^L \Big( b_2 + x_\mathrm{CO}\,\partial_x^2 B_y - y_\mathrm{CO}\,\partial_x^2 B_x \Big)\,\mathrm{d}s$$
$$B\rho\int_0^L K_w\,\mathrm{d}s = \int_0^L \Big( -b_2 + x_\mathrm{CO}\,\partial_y^2 B_y - y_\mathrm{CO}\,\partial_y^2 B_x \Big)\,\mathrm{d}s$$
The longitudinal field is what turns the curvature in one plane into the other. For example, in the vertical plane the feed-down gives $-x_\mathrm{CO}\,\partial_x^2 B_y$ and the $B_s$ term gives $-x_\mathrm{CO}\,\partial_s^2 B_y$; by Laplace's equation, their sum is $x_\mathrm{CO}\,\partial_y^2 B_y$. In words: **the horizontal wiggle (driven by $B_y$) focuses in proportion to the curvature of $B_y$ in each plane, and likewise the vertical wiggle with $B_x$.**

### Sinusoidal approximation
For a field $\propto \cos(k_u s)$ with $k_u = 2\pi/\lambda_u$, the CO equations $x_\mathrm{CO}'' = -B_y/B\rho$ and $y_\mathrm{CO}'' = B_x/B\rho$ give
$$x_\mathrm{CO} \approx \frac{b_1}{k_u^2 B\rho}, \qquad y_\mathrm{CO} \approx -\frac{a_1}{k_u^2 B\rho},$$
and therefore (with $b_2 = 0$)
$$\int_0^L K_u\,\mathrm{d}s \approx \frac{1}{k_u^2 (B\rho)^2}\int_0^L \big( b_1 b_3 + a_1 a_3 \big)\,\mathrm{d}s$$
$$\int_0^L K_w\,\mathrm{d}s \approx -\frac{1}{k_u^2 (B\rho)^2}\int_0^L \Big( b_1\big(b_3 + b_1''\big) + a_1\big(a_3 + a_1''\big) \Big)\,\mathrm{d}s$$
**Check: a flat-pole planar undulator.** $a_n = 0$, no horizontal curvature ($b_3 = 0$), and $b_1'' = -k_u^2 b_1$. Then $\int K_u = 0$ and $\int K_w = L\langle B_y^2\rangle/(B\rho)^2$: the textbook natural focusing, entirely vertical. This is the only case in which the $k_1L = 0$ reading of Section 3 is right.

## 6. Application to the X11MA Undulator
For the X11MA field map (placement `ars11_uind_0610_1`, 2.7 GeV, $B\rho = 9.006$ T m, $\lambda_u = 36$ mm) the terms evaluated along the actual wiggling CO are (in $10^{-3}\,\mathrm{m}^{-1}$, undulator on axis):

| term | $\int K_u\,\mathrm{d}s$ | $\int K_w\,\mathrm{d}s$ |
|---|---|---|
| on-axis quadrupole, $b_2$ | 0 | 0 |
| feed-down of the horizontal wiggle, $x_\mathrm{CO}\,\partial_x^2 B_y$ | +10.48 | −10.48 |
| feed-down of the vertical wiggle, $y_\mathrm{CO}\,\partial_y^2 B_x$ | −1.84 | +1.84 |
| $\to$ $k_1 L$ on the CO | **+8.64** | **−8.64** |
| $B_s$ on the wiggle, $L\langle B_{x,y}^2\rangle/(B\rho)^2$ | +1.24 | +9.50 |
| **total** | **+9.88** | **+0.85** |

The $B_s$ terms reproduce $L\langle B_x^2\rangle/(B\rho)^2 = 1.24$ and $L\langle B_y^2\rangle/(B\rho)^2 = 9.50$ exactly, and $\int (K_u + K_w)\,\mathrm{d}s = L\langle B_x^2 + B_y^2\rangle/(B\rho)^2$, as required by Section 3. Weighted with $\beta$, the totals give $\Delta Q_x = 3.45\times10^{-3}$ and $\Delta Q_y = 2.5\times10^{-4}$, within about 1 % (horizontal) and 2 % (vertical) of the exact tune shift from the closed-ring Twiss ($3.47\times10^{-3}$ and $2.4\times10^{-4}$).

Two observations:
- **The natural focusing ends up in the "wrong" planes.** $k_1 L$ on the CO is nearly as large as the vertical natural focusing, with opposite sign, so almost all focusing is horizontal. The reason is the map's transverse curvature: on axis, $\partial_x^2 B_y \approx 1.07\,k_u^2 B_y$ while $\partial_y^2 B_y \approx -0.08\,k_u^2 B_y$, i.e. $B_y$ grows away from the axis horizontally and hardly at all vertically. This is the opposite of a flat-pole planar undulator and reflects the strongly non-planar (elliptical) geometry of the device. The same holds for $B_x$ with the planes swapped ($\partial_y^2 B_x \approx 0.86\,k_u^2 B_x$, $\partial_x^2 B_x \approx -0.18\,k_u^2 B_x$).
- **The vertical plane nearly cancels.** In the sinusoidal approximation the $B_y$ part of the vertical focusing is $\propto -\int b_1(b_3 + b_1'')$. The $B_s$ term contributes $-b_1 b_1''/k_u^2 \to +L\langle B_y^2\rangle/(B\rho)^2$ regardless of the field's transverse shape, while the feed-down contributes $-b_1 b_3/k_u^2$. For this map $b_3 \approx -b_1''$, i.e. $\partial_y^2 B_y \approx 0$, so the two almost cancel. This is not built into the expansion: $B_s$ contains no $b_3$. It is a property of the fitted inputs $b_1(s)$ and $b_3(s)$. Note that $\partial_y^2 B_y$ is not measured directly (the map spans only $\pm 0.2$ mm vertically, where a $k_u^2$ curvature changes $B_y$ by $\sim 6\times10^{-4}$). It follows from Laplace's equation applied to the well-constrained horizontal curvature ($\pm 2$ mm) and the on-axis $s$-dependence.

A model built from 2D multipoles of the on-axis field (the multipole-kick model, MK) contains the same feed-down terms, but no $B_s$. It therefore keeps the full $-k_1L = -9.4\times10^{-3}\,\mathrm{m}^{-1}$ in the vertical plane, which gives $\Delta Q_y \approx -3.0\times10^{-3}$ instead of $+2.4\times10^{-4}$ and a correspondingly larger vertical beta beat.
