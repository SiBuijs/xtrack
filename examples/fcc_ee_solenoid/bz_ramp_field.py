import numpy as np

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


class LinearBzRamp:
    """Exact Maxwell-consistent field with Bz linear in s.

    On-axis Bz(z) = slope * (z - z0); the accompanying Br = -(slope/2) * r
    makes div(B) = 0 and curl(B) = 0 exactly for all r (not just paraxially),
    so this can be superposed onto any other vacuum field (e.g. the main
    detector solenoid) and the sum stays Maxwell-consistent.
    """

    def __init__(self, slope, z0=0.0):
        self.slope = slope
        self.z0 = z0

    def get_field(self, x, y, z):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        bz = self.slope * (z - self.z0) * np.ones_like(x + y + z)
        br_over_r = -0.5 * self.slope
        bx = br_over_r * x * np.ones_like(y + z)
        by = br_over_r * y * np.ones_like(x + z)

        return bx, by, bz

    def compute_pure_field_derivatives(
            self, s, direction, step, component, max_order=4, min_order=1):
        return SolenoidField.compute_pure_field_derivatives(
            self, s=s, direction=direction, step=step, component=component,
            max_order=max_order, min_order=min_order)
