# Commutators of Beam Optics Operators D, K, R

Phase space coordinates $(x, y, z, P_x, P_y)$ with $P_z$ treated as constant.
Operators (acting on a generic phase-space function):
$$D = \frac{P_x}{P_z}\partial_x + \frac{P_y}{P_z}\partial_y + \partial_z$$

$$K = -q B_y\partial_{P_x} + q B_x\partial_{P_y}$$

$$R = \frac{q B_z}{P_z}\left(P_y\partial_{P_x} - P_x\partial_{P_y}\right)$$

Field components $B_x, B_y, B_z$ are treated as constants (no $\nabla B$ terms).

**Scope:** 1st and 2nd order commutators only (12 total).

# First-order commutators

## [D, K]

$$ [D, K] = \frac{q \left(- B_{x} \partial_y + B_{y} \partial_x\right)}{P_{z}} $$

## [D, R]

$$ [D, R] = \frac{B_{z} q \left(P_{x} \partial_y - P_{y} \partial_x\right)}{P_{z}^{2}} $$

## [K, R]

$$ [K, R] = \frac{B_{z} q^{2} \left(B_{x} \partial_{P_x} + B_{y} \partial_{P_y}\right)}{P_{z}} $$

# Second-order commutators

## [D, [D, K]]

$$ [D, [D, K]] = 0 $$

## [D, [D, R]]

$$ [D, [D, R]] = 0 $$

## [D, [K, R]]

$$ [D, [K, R]] = \frac{B_{z} q^{2} \left(- B_{x} \partial_x - B_{y} \partial_y\right)}{P_{z}^{2}} $$

## [K, [D, K]]

$$ [K, [D, K]] = 0 $$

## [K, [D, R]]

$$ [K, [D, R]] = \frac{B_{z} q^{2} \left(- B_{x} \partial_x - B_{y} \partial_y\right)}{P_{z}^{2}} $$

## [K, [K, R]]

$$ [K, [K, R]] = 0 $$

## [R, [D, K]]

$$ [R, [D, K]] = 0 $$

## [R, [D, R]]

$$ [R, [D, R]] = \frac{B_{z}^{2} q^{2} \left(P_{x} \partial_x + P_{y} \partial_y\right)}{P_{z}^{3}} $$

## [R, [K, R]]

$$ [R, [K, R]] = \frac{B_{z}^{2} q^{3} \left(B_{x} \partial_{P_y} - B_{y} \partial_{P_x}\right)}{P_{z}^{2}} $$
