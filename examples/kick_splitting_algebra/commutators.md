# Commutators of Beam Optics Operators D, K, R

Phase space coordinates $(x, y, z, P_x, P_y)$ with dependent momentum
$P_z = \sqrt{P^2 - P_x^2 - P_y^2}$ ($P$ fixed).

Operators (acting on a generic phase-space function):

$$D = \frac{P_x}{P_z}\partial_x + \frac{P_y}{P_z}\partial_y + \partial_z$$

$$K = -q B_y\partial_{P_x} + q B_x\partial_{P_y}$$

$$R = \frac{q B_z}{P_z}\left(P_y\partial_{P_x} - P_x\partial_{P_y}\right)$$

Field components $B_x, B_y, B_z$ depend on $(x, y)$ only.

**Scope:** 1st and 2nd order commutators only (12 total).

# First-order commutators

## [D, K]

$$ [D, K] = [D,K]_0 + [D,K]_1 + [D,K]_{\nabla B} $$

**Leading order** ($[D,K]_0$, always nonzero):

$$ [D,K]_0 = \frac{q \left(- B_{x} \partial_y + B_{y} \partial_x\right)}{P_z} $$

**Paraxial correction** ($[D,K]_1$, vanishes for $P_\perp \ll P_z$):

$$ [D,K]_1 = \frac{q \left(P_{x}^{2} B_{y} \partial_x - P_{x} P_{y} B_{x} \partial_x + P_{x} P_{y} B_{y} \partial_y - P_{y}^{2} B_{x} \partial_y\right)}{P_z^{3}} $$

**Field gradient term** ($[D,K]_{\nabla B}$, vanishes for uniform field):

$$ [D,K]_{\nabla B} = \frac{q \left(P_{x} \partial_x B_{x} \partial_{P_y} - P_{x} \partial_x B_{y} \partial_{P_x} + P_{y} \partial_y B_{x} \partial_{P_y} - P_{y} \partial_y B_{y} \partial_{P_x}\right)}{P_z} $$

## [D, R]

$$ [D, R] = [D,R]_0 + [D,R]_1 + [D,R]_{\nabla B} $$

**Leading order** ($[D,R]_0$, always nonzero):

$$ [D,R]_0 = 0 $$

**Paraxial correction** ($[D,R]_1$, vanishes for $P_\perp \ll P_z$):

$$ [D,R]_1 = \frac{q \left(- P_{x} \partial_y + P_{y} \partial_x\right) B_{z}}{- P^{2} + P_{x}^{2} + P_{y}^{2}} $$

**Field gradient term** ($[D,R]_{\nabla B}$, vanishes for uniform field):

$$ [D,R]_{\nabla B} = \frac{q \left(P_{x}^{2} \partial_x B_{z} \partial_{P_y} - P_{x} P_{y} \partial_x B_{z} \partial_{P_x} + P_{x} P_{y} \partial_y B_{z} \partial_{P_y} - P_{y}^{2} \partial_y B_{z} \partial_{P_x}\right)}{- P^{2} + P_{x}^{2} + P_{y}^{2}} $$

## [K, R]

$$ [K, R] = [K,R]_0 + [K,R]_1 + [K,R]_{\nabla B} $$

**Leading order** ($[K,R]_0$, always nonzero):

$$ [K,R]_0 = \frac{q^{2} \left(B_{x} \partial_{P_x} + B_{y} \partial_{P_y}\right) B_{z}}{P_z} $$

**Paraxial correction** ($[K,R]_1$, vanishes for $P_\perp \ll P_z$):

$$ [K,R]_1 = \frac{q^{2} \left(P_{x}^{2} B_{y} \partial_{P_y} - P_{x} P_{y} B_{x} \partial_{P_y} - P_{x} P_{y} B_{y} \partial_{P_x} + P_{y}^{2} B_{x} \partial_{P_x}\right) B_{z}}{P_z^{3}} $$

**Field gradient term** ($[K,R]_{\nabla B}$, vanishes for uniform field):

$$ [K,R]_{\nabla B} = 0 $$

# Second-order commutators

## [D, [D, K]]

$$ [D, [D, K]] = [D,D,K]_0 + [D,D,K]_1 + [D,D,K]_{\nabla B} $$

**Leading order** ($[D,D,K]_0$, always nonzero):

$$ [D,D,K]_0 = 0 $$

**Paraxial correction** ($[D,D,K]_1$, vanishes for $P_\perp \ll P_z$):

$$ [D,D,K]_1 = 0 $$

**Field gradient term** ($[D,D,K]_{\nabla B}$, vanishes for uniform field):

$$ [D,D,K]_{\nabla B} = \frac{q \left(P^{2} P_{x}^{2} \partial_x^{2} B_{x} \partial_{P_y} - P^{2} P_{x}^{2} \partial_x^{2} B_{y} \partial_{P_x} - 2 P^{2} P_{x} P_{y} \partial_{P_x} \partial_x \partial_y B_{y} + 2 P^{2} P_{x} P_{y} \partial_{P_y} \partial_x \partial_y B_{x} - 2 P^{2} P_{x} \partial_x B_{x} \partial_y + 2 P^{2} P_{x} \partial_x B_{y} \partial_x + P^{2} P_{y}^{2} \partial_y^{2} B_{x} \partial_{P_y} - P^{2} P_{y}^{2} \partial_y^{2} B_{y} \partial_{P_x} - 2 P^{2} P_{y} \partial_y B_{x} \partial_y + 2 P^{2} P_{y} \partial_y B_{y} \partial_x - P_{x}^{4} \partial_x^{2} B_{x} \partial_{P_y} + P_{x}^{4} \partial_x^{2} B_{y} \partial_{P_x} + 2 P_{x}^{3} P_{y} \partial_{P_x} \partial_x \partial_y B_{y} - 2 P_{x}^{3} P_{y} \partial_{P_y} \partial_x \partial_y B_{x} + 2 P_{x}^{3} \partial_x B_{x} \partial_y - P_{x}^{2} P_{y}^{2} \partial_x^{2} B_{x} \partial_{P_y} - P_{x}^{2} P_{y}^{2} \partial_y^{2} B_{x} \partial_{P_y} + P_{x}^{2} P_{y}^{2} \partial_x^{2} B_{y} \partial_{P_x} + P_{x}^{2} P_{y}^{2} \partial_y^{2} B_{y} \partial_{P_x} - 2 P_{x}^{2} P_{y} \partial_x B_{x} \partial_x + 2 P_{x}^{2} P_{y} \partial_y B_{x} \partial_y + 2 P_{x}^{2} P_{y} \partial_x B_{y} \partial_y + 2 P_{x} P_{y}^{3} \partial_{P_x} \partial_x \partial_y B_{y} - 2 P_{x} P_{y}^{3} \partial_{P_y} \partial_x \partial_y B_{x} - 2 P_{x} P_{y}^{2} \partial_y B_{x} \partial_x - 2 P_{x} P_{y}^{2} \partial_x B_{y} \partial_x + 2 P_{x} P_{y}^{2} \partial_y B_{y} \partial_y - P_{y}^{4} \partial_y^{2} B_{x} \partial_{P_y} + P_{y}^{4} \partial_y^{2} B_{y} \partial_{P_x} - 2 P_{y}^{3} \partial_y B_{y} \partial_x\right)}{P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}} $$

## [D, [D, R]]

$$ [D, [D, R]] = [D,D,R]_0 + [D,D,R]_1 + [D,D,R]_{\nabla B} $$

**Leading order** ($[D,D,R]_0$, always nonzero):

$$ [D,D,R]_0 = 0 $$

**Paraxial correction** ($[D,D,R]_1$, vanishes for $P_\perp \ll P_z$):

$$ [D,D,R]_1 = 0 $$

**Field gradient term** ($[D,D,R]_{\nabla B}$, vanishes for uniform field):

$$ [D,D,R]_{\nabla B} = \frac{q \left(- P_{x}^{3} \partial_x^{2} B_{z} \partial_{P_y} + P_{x}^{2} P_{y} \partial_x^{2} B_{z} \partial_{P_x} - 2 P_{x}^{2} P_{y} \partial_{P_y} \partial_x \partial_y B_{z} + 2 P_{x}^{2} \partial_x B_{z} \partial_y - P_{x} P_{y}^{2} \partial_y^{2} B_{z} \partial_{P_y} + 2 P_{x} P_{y}^{2} \partial_{P_x} \partial_x \partial_y B_{z} - 2 P_{x} P_{y} \partial_x B_{z} \partial_x + 2 P_{x} P_{y} \partial_y B_{z} \partial_y + P_{y}^{3} \partial_y^{2} B_{z} \partial_{P_x} - 2 P_{y}^{2} \partial_y B_{z} \partial_x\right)}{P_z^{3}} $$

## [D, [K, R]]

$$ [D, [K, R]] = [D,K,R]_0 + [D,K,R]_1 + [D,K,R]_{\nabla B} $$

**Leading order** ($[D,K,R]_0$, always nonzero):

$$ [D,K,R]_0 = \frac{q^{2} \left(B_{x} \partial_x + B_{y} \partial_y\right) B_{z}}{- P^{2} + P_{x}^{2} + P_{y}^{2}} $$

**Paraxial correction** ($[D,K,R]_1$, vanishes for $P_\perp \ll P_z$):

$$ [D,K,R]_1 = \frac{q^{2} \left(- P_{x}^{2} B_{x} \partial_x - P_{x}^{2} B_{y} \partial_y - P_{y}^{2} B_{x} \partial_x - P_{y}^{2} B_{y} \partial_y\right) B_{z}}{P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}} $$

**Field gradient term** ($[D,K,R]_{\nabla B}$, vanishes for uniform field):

$$ [D,K,R]_{\nabla B} = \frac{q^{2} \left(P^{2} P_{x} B_{x} \partial_x B_{z} \partial_{P_x} + P^{2} P_{x} B_{y} \partial_x B_{z} \partial_{P_y} + P^{2} P_{x} B_{z} \partial_x B_{x} \partial_{P_x} + P^{2} P_{x} B_{z} \partial_x B_{y} \partial_{P_y} + P^{2} P_{y} B_{x} \partial_y B_{z} \partial_{P_x} + P^{2} P_{y} B_{y} \partial_y B_{z} \partial_{P_y} + P^{2} P_{y} B_{z} \partial_y B_{x} \partial_{P_x} + P^{2} P_{y} B_{z} \partial_y B_{y} \partial_{P_y} - P_{x}^{3} B_{x} \partial_x B_{z} \partial_{P_x} - P_{x}^{3} B_{z} \partial_x B_{x} \partial_{P_x} - P_{x}^{2} P_{y} B_{x} \partial_x B_{z} \partial_{P_y} - P_{x}^{2} P_{y} B_{x} \partial_y B_{z} \partial_{P_x} - P_{x}^{2} P_{y} B_{y} \partial_x B_{z} \partial_{P_x} - P_{x}^{2} P_{y} B_{z} \partial_x B_{x} \partial_{P_y} - P_{x}^{2} P_{y} B_{z} \partial_y B_{x} \partial_{P_x} - P_{x}^{2} P_{y} B_{z} \partial_x B_{y} \partial_{P_x} - P_{x} P_{y}^{2} B_{x} \partial_y B_{z} \partial_{P_y} - P_{x} P_{y}^{2} B_{y} \partial_x B_{z} \partial_{P_y} - P_{x} P_{y}^{2} B_{y} \partial_y B_{z} \partial_{P_x} - P_{x} P_{y}^{2} B_{z} \partial_y B_{x} \partial_{P_y} - P_{x} P_{y}^{2} B_{z} \partial_x B_{y} \partial_{P_y} - P_{x} P_{y}^{2} B_{z} \partial_y B_{y} \partial_{P_x} - P_{y}^{3} B_{y} \partial_y B_{z} \partial_{P_y} - P_{y}^{3} B_{z} \partial_y B_{y} \partial_{P_y}\right)}{P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}} $$

## [K, [D, K]]

$$ [K, [D, K]] = [K,D,K]_0 + [K,D,K]_1 + [K,D,K]_{\nabla B} $$

**Leading order** ($[K,D,K]_0$, always nonzero):

$$ [K,D,K]_0 = 0 $$

**Paraxial correction** ($[K,D,K]_1$, vanishes for $P_\perp \ll P_z$):

$$ [K,D,K]_1 = \frac{q^{2} \left(- P^{2} P_{x} B_{x}^{2} \partial_x + 2 P^{2} P_{x} B_{x} B_{y} \partial_y - 3 P^{2} P_{x} B_{y}^{2} \partial_x - 3 P^{2} P_{y} B_{x}^{2} \partial_y + 2 P^{2} P_{y} B_{x} B_{y} \partial_x - P^{2} P_{y} B_{y}^{2} \partial_y + P_{x}^{3} B_{x}^{2} \partial_x - 2 P_{x}^{3} B_{x} B_{y} \partial_y + 3 P_{x}^{2} P_{y} B_{x}^{2} \partial_y + 4 P_{x}^{2} P_{y} B_{x} B_{y} \partial_x - 2 P_{x}^{2} P_{y} B_{y}^{2} \partial_y - 2 P_{x} P_{y}^{2} B_{x}^{2} \partial_x + 4 P_{x} P_{y}^{2} B_{x} B_{y} \partial_y + 3 P_{x} P_{y}^{2} B_{y}^{2} \partial_x - 2 P_{y}^{3} B_{x} B_{y} \partial_x + P_{y}^{3} B_{y}^{2} \partial_y\right)}{P_z \left(P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}\right)} $$

**Field gradient term** ($[K,D,K]_{\nabla B}$, vanishes for uniform field):

$$ [K,D,K]_{\nabla B} = \frac{2 q^{2} \left(P^{2} B_{x} \partial_y B_{x} \partial_{P_y} - P^{2} B_{x} \partial_y B_{y} \partial_{P_x} - P^{2} B_{y} \partial_x B_{x} \partial_{P_y} + P^{2} B_{y} \partial_x B_{y} \partial_{P_x} - P_{x}^{2} B_{x} \partial_y B_{x} \partial_{P_y} + P_{x}^{2} B_{x} \partial_y B_{y} \partial_{P_x} + P_{x} P_{y} B_{x} \partial_x B_{x} \partial_{P_y} - P_{x} P_{y} B_{x} \partial_x B_{y} \partial_{P_x} - P_{x} P_{y} B_{y} \partial_y B_{x} \partial_{P_y} + P_{x} P_{y} B_{y} \partial_y B_{y} \partial_{P_x} + P_{y}^{2} B_{y} \partial_x B_{x} \partial_{P_y} - P_{y}^{2} B_{y} \partial_x B_{y} \partial_{P_x}\right)}{P_z^{3}} $$

## [K, [D, R]]

$$ [K, [D, R]] = [K,D,R]_0 + [K,D,R]_1 + [K,D,R]_{\nabla B} $$

**Leading order** ($[K,D,R]_0$, always nonzero):

$$ [K,D,R]_0 = \frac{q^{2} \left(B_{x} \partial_x + B_{y} \partial_y\right) B_{z}}{- P^{2} + P_{x}^{2} + P_{y}^{2}} $$

**Paraxial correction** ($[K,D,R]_1$, vanishes for $P_\perp \ll P_z$):

$$ [K,D,R]_1 = \frac{2 q^{2} \left(- P_{x}^{2} B_{y} \partial_y + P_{x} P_{y} B_{x} \partial_y + P_{x} P_{y} B_{y} \partial_x - P_{y}^{2} B_{x} \partial_x\right) B_{z}}{P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}} $$

**Field gradient term** ($[K,D,R]_{\nabla B}$, vanishes for uniform field):

$$ [K,D,R]_{\nabla B} = \frac{q^{2} \left(P^{2} P_{x} B_{x} \partial_x B_{z} \partial_{P_x} - P^{2} P_{x} B_{x} \partial_y B_{z} \partial_{P_y} + 2 P^{2} P_{x} B_{y} \partial_x B_{z} \partial_{P_y} - P^{2} P_{x} B_{z} \partial_y B_{x} \partial_{P_y} + P^{2} P_{x} B_{z} \partial_y B_{y} \partial_{P_x} + 2 P^{2} P_{y} B_{x} \partial_y B_{z} \partial_{P_x} - P^{2} P_{y} B_{y} \partial_x B_{z} \partial_{P_x} + P^{2} P_{y} B_{y} \partial_y B_{z} \partial_{P_y} + P^{2} P_{y} B_{z} \partial_x B_{x} \partial_{P_y} - P^{2} P_{y} B_{z} \partial_x B_{y} \partial_{P_x} - P_{x}^{3} B_{x} \partial_x B_{z} \partial_{P_x} + P_{x}^{3} B_{x} \partial_y B_{z} \partial_{P_y} + P_{x}^{3} B_{z} \partial_y B_{x} \partial_{P_y} - P_{x}^{3} B_{z} \partial_y B_{y} \partial_{P_x} - 2 P_{x}^{2} P_{y} B_{x} \partial_x B_{z} \partial_{P_y} - 2 P_{x}^{2} P_{y} B_{x} \partial_y B_{z} \partial_{P_x} - P_{x}^{2} P_{y} B_{y} \partial_x B_{z} \partial_{P_x} + P_{x}^{2} P_{y} B_{y} \partial_y B_{z} \partial_{P_y} - P_{x}^{2} P_{y} B_{z} \partial_x B_{x} \partial_{P_y} + P_{x}^{2} P_{y} B_{z} \partial_x B_{y} \partial_{P_x} + P_{x} P_{y}^{2} B_{x} \partial_x B_{z} \partial_{P_x} - P_{x} P_{y}^{2} B_{x} \partial_y B_{z} \partial_{P_y} - 2 P_{x} P_{y}^{2} B_{y} \partial_x B_{z} \partial_{P_y} - 2 P_{x} P_{y}^{2} B_{y} \partial_y B_{z} \partial_{P_x} + P_{x} P_{y}^{2} B_{z} \partial_y B_{x} \partial_{P_y} - P_{x} P_{y}^{2} B_{z} \partial_y B_{y} \partial_{P_x} + P_{y}^{3} B_{y} \partial_x B_{z} \partial_{P_x} - P_{y}^{3} B_{y} \partial_y B_{z} \partial_{P_y} - P_{y}^{3} B_{z} \partial_x B_{x} \partial_{P_y} + P_{y}^{3} B_{z} \partial_x B_{y} \partial_{P_x}\right)}{P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}} $$

## [K, [K, R]]

$$ [K, [K, R]] = [K,K,R]_0 + [K,K,R]_1 + [K,K,R]_{\nabla B} $$

**Leading order** ($[K,K,R]_0$, always nonzero):

$$ [K,K,R]_0 = 0 $$

**Paraxial correction** ($[K,K,R]_1$, vanishes for $P_\perp \ll P_z$):

$$ [K,K,R]_1 = \frac{q^{3} \left(- P^{2} P_{x} B_{x}^{2} \partial_{P_y} - 2 P^{2} P_{x} B_{x} B_{y} \partial_{P_x} - 3 P^{2} P_{x} B_{y}^{2} \partial_{P_y} + 3 P^{2} P_{y} B_{x}^{2} \partial_{P_x} + 2 P^{2} P_{y} B_{x} B_{y} \partial_{P_y} + P^{2} P_{y} B_{y}^{2} \partial_{P_x} + P_{x}^{3} B_{x}^{2} \partial_{P_y} + 2 P_{x}^{3} B_{x} B_{y} \partial_{P_x} - 3 P_{x}^{2} P_{y} B_{x}^{2} \partial_{P_x} + 4 P_{x}^{2} P_{y} B_{x} B_{y} \partial_{P_y} + 2 P_{x}^{2} P_{y} B_{y}^{2} \partial_{P_x} - 2 P_{x} P_{y}^{2} B_{x}^{2} \partial_{P_y} - 4 P_{x} P_{y}^{2} B_{x} B_{y} \partial_{P_x} + 3 P_{x} P_{y}^{2} B_{y}^{2} \partial_{P_y} - 2 P_{y}^{3} B_{x} B_{y} \partial_{P_y} - P_{y}^{3} B_{y}^{2} \partial_{P_x}\right) B_{z}}{P_z \left(P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}\right)} $$

**Field gradient term** ($[K,K,R]_{\nabla B}$, vanishes for uniform field):

$$ [K,K,R]_{\nabla B} = 0 $$

## [R, [D, K]]

$$ [R, [D, K]] = [R,D,K]_0 + [R,D,K]_1 + [R,D,K]_{\nabla B} $$

**Leading order** ($[R,D,K]_0$, always nonzero):

$$ [R,D,K]_0 = 0 $$

**Paraxial correction** ($[R,D,K]_1$, vanishes for $P_\perp \ll P_z$):

$$ [R,D,K]_1 = \frac{q^{2} \left(P_{x}^{2} B_{x} \partial_x - P_{x}^{2} B_{y} \partial_y + 2 P_{x} P_{y} B_{x} \partial_y + 2 P_{x} P_{y} B_{y} \partial_x - P_{y}^{2} B_{x} \partial_x + P_{y}^{2} B_{y} \partial_y\right) B_{z}}{P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}} $$

**Field gradient term** ($[R,D,K]_{\nabla B}$, vanishes for uniform field):

$$ [R,D,K]_{\nabla B} = \frac{q^{2} \left(- P^{2} P_{x} B_{x} \partial_y B_{z} \partial_{P_y} + P^{2} P_{x} B_{y} \partial_x B_{z} \partial_{P_y} - P^{2} P_{x} B_{z} \partial_x B_{x} \partial_{P_x} - P^{2} P_{x} B_{z} \partial_y B_{x} \partial_{P_y} - P^{2} P_{x} B_{z} \partial_x B_{y} \partial_{P_y} + P^{2} P_{x} B_{z} \partial_y B_{y} \partial_{P_x} + P^{2} P_{y} B_{x} \partial_y B_{z} \partial_{P_x} - P^{2} P_{y} B_{y} \partial_x B_{z} \partial_{P_x} + P^{2} P_{y} B_{z} \partial_x B_{x} \partial_{P_y} - P^{2} P_{y} B_{z} \partial_y B_{x} \partial_{P_x} - P^{2} P_{y} B_{z} \partial_x B_{y} \partial_{P_x} - P^{2} P_{y} B_{z} \partial_y B_{y} \partial_{P_y} + P_{x}^{3} B_{x} \partial_y B_{z} \partial_{P_y} + P_{x}^{3} B_{z} \partial_x B_{x} \partial_{P_x} + P_{x}^{3} B_{z} \partial_y B_{x} \partial_{P_y} - P_{x}^{3} B_{z} \partial_y B_{y} \partial_{P_x} - P_{x}^{2} P_{y} B_{x} \partial_x B_{z} \partial_{P_y} - P_{x}^{2} P_{y} B_{x} \partial_y B_{z} \partial_{P_x} + P_{x}^{2} P_{y} B_{y} \partial_y B_{z} \partial_{P_y} + P_{x}^{2} P_{y} B_{z} \partial_y B_{x} \partial_{P_x} + 2 P_{x}^{2} P_{y} B_{z} \partial_x B_{y} \partial_{P_x} + P_{x} P_{y}^{2} B_{x} \partial_x B_{z} \partial_{P_x} - P_{x} P_{y}^{2} B_{y} \partial_x B_{z} \partial_{P_y} - P_{x} P_{y}^{2} B_{y} \partial_y B_{z} \partial_{P_x} + 2 P_{x} P_{y}^{2} B_{z} \partial_y B_{x} \partial_{P_y} + P_{x} P_{y}^{2} B_{z} \partial_x B_{y} \partial_{P_y} + P_{y}^{3} B_{y} \partial_x B_{z} \partial_{P_x} - P_{y}^{3} B_{z} \partial_x B_{x} \partial_{P_y} + P_{y}^{3} B_{z} \partial_x B_{y} \partial_{P_x} + P_{y}^{3} B_{z} \partial_y B_{y} \partial_{P_y}\right)}{P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}} $$

## [R, [D, R]]

$$ [R, [D, R]] = [R,D,R]_0 + [R,D,R]_1 + [R,D,R]_{\nabla B} $$

**Leading order** ($[R,D,R]_0$, always nonzero):

$$ [R,D,R]_0 = 0 $$

**Paraxial correction** ($[R,D,R]_1$, vanishes for $P_\perp \ll P_z$):

$$ [R,D,R]_1 = \frac{q^{2} \left(P_{x} \partial_x + P_{y} \partial_y\right) P_z B_{z}^{2}}{P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}} $$

**Field gradient term** ($[R,D,R]_{\nabla B}$, vanishes for uniform field):

$$ [R,D,R]_{\nabla B} = \frac{2 q^{2} P_z \left(P_{x}^{2} \partial_y B_{z} \partial_{P_y} - P_{x} P_{y} \partial_x B_{z} \partial_{P_y} - P_{x} P_{y} \partial_y B_{z} \partial_{P_x} + P_{y}^{2} \partial_x B_{z} \partial_{P_x}\right) B_{z}}{P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}} $$

## [R, [K, R]]

$$ [R, [K, R]] = [R,K,R]_0 + [R,K,R]_1 + [R,K,R]_{\nabla B} $$

**Leading order** ($[R,K,R]_0$, always nonzero):

$$ [R,K,R]_0 = \frac{q^{3} \left(- B_{x} \partial_{P_y} + B_{y} \partial_{P_x}\right) B_{z}^{2}}{- P^{2} + P_{x}^{2} + P_{y}^{2}} $$

**Paraxial correction** ($[R,K,R]_1$, vanishes for $P_\perp \ll P_z$):

$$ [R,K,R]_1 = \frac{2 q^{3} \left(P_{x}^{2} B_{x} \partial_{P_y} - P_{x} P_{y} B_{x} \partial_{P_x} + P_{x} P_{y} B_{y} \partial_{P_y} - P_{y}^{2} B_{y} \partial_{P_x}\right) B_{z}^{2}}{P^{4} - 2 P^{2} P_{x}^{2} - 2 P^{2} P_{y}^{2} + P_{x}^{4} + 2 P_{x}^{2} P_{y}^{2} + P_{y}^{4}} $$

**Field gradient term** ($[R,K,R]_{\nabla B}$, vanishes for uniform field):

$$ [R,K,R]_{\nabla B} = 0 $$
