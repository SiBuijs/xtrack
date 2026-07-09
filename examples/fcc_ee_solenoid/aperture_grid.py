"""Initial-condition grids for aperture studies.

Vendored from ``fcc_92/004_tutorial_cap_meeting/gen_grid.py`` so the
fcc_ee_solenoid examples do not depend on an external checkout.
"""

from __future__ import annotations

import numpy as np
import xpart as xp
import xtrack as xt


def initial_conditions_grid(
    *,
    study: str,
    energy_spread: float,
    nn_y_r: int,
    max_y_r: float,
    delta_initial_values: np.ndarray,
    min_y_r: float = 0.0,
    min_x_theta: float | None = None,
    max_x_theta: float | None = None,
    nn_x_theta: int = 1,
) -> xt.Table:
    """Polar grid in normalized coordinates, replicated over momentum offsets."""
    if study != "MA":
        raise ValueError(f"Only study='MA' is supported, got {study!r}")

    if min_x_theta is None:
        min_x_theta = np.pi / 4
    if max_x_theta is None:
        max_x_theta = np.pi / 4

    x_normalized, y_normalized, _, _ = xp.generate_2D_polar_grid(
        r_range=(min_y_r, max_y_r),
        theta_range=(min_x_theta, max_x_theta),
        nr=nn_y_r,
        ntheta=nn_x_theta,
    )

    delta_initial_values = np.asarray(delta_initial_values)
    num_delta = int(delta_initial_values.size)
    num_spatial = int(x_normalized.size)

    x_normalized = np.tile(x_normalized, num_delta)
    y_normalized = np.tile(y_normalized, num_delta)
    delta_init = np.repeat(delta_initial_values, num_spatial)

    num_particles = num_delta * num_spatial

    out = xt.Table(
        dict(
            id=np.arange(num_particles, dtype=int),
            x_normalized=x_normalized,
            y_normalized=y_normalized,
            delta_init=delta_init,
        ),
        index="id",
    )
    out.nn_x_theta = nn_x_theta
    out.nn_y_r = nn_y_r
    out.num_delta = num_delta
    out.num_particles = num_particles

    return out
