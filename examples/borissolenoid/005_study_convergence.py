"""
Convergence study for ``BorisSolenoid`` as a function of ``n_steps``.

For a range of Boris substep counts we track an off-axis electron through a
single solenoid element and compare the exit position to the finest-step
result. We also evaluate the one-turn transfer matrix from finite differences
and report its symplectic deviation and determinant, comparing
``BorisSolenoid`` and ``BorisSpatialIntegrator`` (same 3D field) to
``VariableSolenoid`` (symplectic paraxial slices from the on-axis ``Bz``
profile).

Notes on the symplectic metric
------------------------------
The quantity ``||R^T S R - S||_2`` is computed correctly, but it should not be
read as an integration error that vanishes with ``n_steps``. It is reported
normalised by ``||R_{4d}||_2^2`` so that values from elements with different
transfer-matrix strength can be compared. The Boris /
field-aligned helical map is volume-preserving (``det(R) ~ 1``) but not
symplectic. For the Hampton analytical solenoid field used here, the deviation
saturates at a finite value even for very large ``n_steps`` while the particle
trajectory still converges. This is unlike the softer ``LinearFringeSolenoid``
used in ``examples/boris_spatial/004_study_convergence.py``, where the same
metric does decrease with step count.

``BorisSpatialIntegrator`` is a collective (Python-tracked) element. When
calling ``get_R_matrix`` on a line that contains it, pass
``include_collective=True``; otherwise xtrack silently substitutes drifts and
the symplectic metric is computed for the wrong map.

``VariableSolenoid`` uses a symplectic thick-lens integrator on linearly
varying ``ks(s)`` slices (on-axis field plus integrated transverse kicks). Its
normalised symplectic deviation should be near machine zero.
"""

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

S = xt.linear_normal_form.S
S4 = S[:4, :4]


def normalized_symplectic_errors(r_matrix):
    se = np.linalg.norm(r_matrix.T @ S @ r_matrix - S, ord=2)
    se4 = np.linalg.norm(
        r_matrix[:4, :4].T @ S4 @ r_matrix[:4, :4] - S4, ord=2
    )
    r4_norm = np.linalg.norm(r_matrix[:4, :4], ord=2)
    return se, se4, se / r4_norm ** 2, se4 / r4_norm ** 2


def build_variable_solenoid_line(*, field, length, brho, n_slices):
    s_axis = np.linspace(0.0, length, n_slices + 1)
    bx_axis, by_axis, bz_axis = field.get_field(
        np.zeros_like(s_axis),
        np.zeros_like(s_axis),
        s_axis,
    )
    ks = bz_axis / brho
    k0 = by_axis / brho
    k0s = bx_axis / brho
    elements = []
    for ii in range(n_slices):
        ds = s_axis[ii + 1] - s_axis[ii]
        elements.append(
            xt.VariableSolenoid(
                length=ds,
                ks_profile=[ks[ii], ks[ii + 1]],
                knl=[0.5 * (k0[ii] + k0[ii + 1]) * ds],
                ksl=[0.5 * (k0s[ii] + k0s[ii + 1]) * ds],
            )
        )
    return xt.Line(elements=elements)

p0 = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV,
    q0=1,
    energy0=50e6,
    x=[-5e-3],
    px=0,
    y=2e-3,
    delta=0,
)

L_coil = 4.0
a = 0.3
B0 = 1.0
z0 = 5.0
length = 10.0

solenoid_field = SolenoidField(L=L_coil, a=a, B0=B0, z0=z0)
brho = p0.rigidity0[0]

# Cap slice count: VariableSolenoid R-matrix cost scales with slice number.
VARSOL_MAX_SLICES = 5000

n_steps_vect = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

sympl_error = []
sympl_error_4d = []
sympl_error_norm = []
sympl_error_4d_norm = []
sympl_error_taper_norm = []
sympl_error_4d_taper_norm = []
sympl_error_spatial_norm = []
sympl_error_4d_spatial_norm = []
sympl_error_varsol_norm = []
sympl_error_4d_varsol_norm = []
det_R = []
det_R_taper = []
x = []
y = []
x_taper = []
y_taper = []
for n_steps in n_steps_vect:
    print("n_steps=", n_steps)
    solenoid = xt.BorisSolenoid(
        L_coil=L_coil,
        a=a,
        B0=B0,
        z0=z0,
        length=length,
        n_steps=n_steps,
    )
    line = xt.Line(elements=[solenoid])
    RR = line.get_R_matrix(particle_on_co=p0.copy())["R_matrix"]

    se, se4, se_norm, se4_norm = normalized_symplectic_errors(RR)
    sympl_error.append(se)
    sympl_error_4d.append(se4)
    sympl_error_norm.append(se_norm)
    sympl_error_4d_norm.append(se4_norm)

    solenoid_taper = xt.BorisSolenoid(
        L_coil=L_coil,
        a=a,
        B0=B0,
        z0=z0,
        length=length,
        n_steps=n_steps,
        taper=True,
    )
    line_taper = solenoid_taper.to_tapered_line()
    RR_taper = line_taper.get_R_matrix(particle_on_co=p0.copy())["R_matrix"]
    _, _, se_taper_norm, se4_taper_norm = normalized_symplectic_errors(RR_taper)
    sympl_error_taper_norm.append(se_taper_norm)
    sympl_error_4d_taper_norm.append(se4_taper_norm)

    spatial_integrator = xt.BorisSpatialIntegrator(
        fieldmap_callable=solenoid_field.get_field,
        s_start=0,
        s_end=length,
        n_steps=n_steps,
    )
    line_spatial = xt.Line(elements=[spatial_integrator])
    # BorisSpatialIntegrator is a Python (collective) element; without this
    # flag get_R_matrix replaces it by a drift and the symplectic check is wrong.
    RR_spatial = line_spatial.get_R_matrix(
        particle_on_co=p0.copy(),
        include_collective=True,
    )["R_matrix"]
    _, _, se_spatial_norm, se4_spatial_norm = normalized_symplectic_errors(
        RR_spatial
    )
    sympl_error_spatial_norm.append(se_spatial_norm)
    sympl_error_4d_spatial_norm.append(se4_spatial_norm)

    n_varsol_slices = min(n_steps, VARSOL_MAX_SLICES)
    line_varsol = build_variable_solenoid_line(
        field=solenoid_field,
        length=length,
        brho=brho,
        n_slices=n_varsol_slices,
    )
    RR_varsol = line_varsol.get_R_matrix(particle_on_co=p0.copy())["R_matrix"]
    _, _, se_varsol_norm, se4_varsol_norm = normalized_symplectic_errors(
        RR_varsol
    )
    sympl_error_varsol_norm.append(se_varsol_norm)
    sympl_error_4d_varsol_norm.append(se4_varsol_norm)

    p_boris = p0.copy()
    solenoid.track(p_boris)
    x.append(p_boris.x[0])
    y.append(p_boris.y[0])
    p_taper = p0.copy()
    line_taper.track(p_taper)
    x_taper.append(p_taper.x[0])
    y_taper.append(p_taper.y[0])
    det_R.append(np.linalg.det(RR))
    det_R_taper.append(np.linalg.det(RR_taper))

err = np.sqrt((np.array(x) - x[-1]) ** 2 + (np.array(y) - y[-1]) ** 2)
err_taper = np.sqrt(
    (np.array(x_taper) - x_taper[-1]) ** 2
    + (np.array(y_taper) - y_taper[-1]) ** 2
)

plt.close("all")
fig1 = plt.figure(1, figsize=(6.4, 4.8))
plt.loglog(n_steps_vect[:-1], err[:-1], "-o", label="BorisSolenoid")
plt.loglog(
    n_steps_vect[:-1], err_taper[:-1], "-s", label="BorisSolenoid tapered"
)
plt.loglog(
    n_steps_vect[:-1],
    err[0] * n_steps_vect[0] ** 2 * 1 / np.array(n_steps_vect[:-1]) ** (2),
    "--",
    label=r"~ 1/$N_\text{steps}^2$",
)
plt.xlabel("Number of steps")
plt.ylabel("Error on exit position (m)")
plt.xlim(n_steps_vect[0] / 2, n_steps_vect[-1])
plt.legend()

fig2 = plt.figure(2, figsize=(6.4, 4.8))
plt.loglog(
    n_steps_vect,
    np.abs(sympl_error_norm),
    "-o",
    label=r"BorisSolenoid, $||R^T S R - S||_2 / ||R_{4d}||_2^2$",
)
plt.loglog(
    n_steps_vect,
    np.abs(sympl_error_4d_norm),
    "-s",
    label=r"BorisSolenoid, $||R_{4d}^T S_{4d} R_{4d} - S_{4d}||_2 / ||R_{4d}||_2^2$",
)
plt.loglog(
    n_steps_vect,
    np.abs(sympl_error_taper_norm),
    "-^",
    label=r"BorisSolenoid tapered, $||R^T S R - S||_2 / ||R_{4d}||_2^2$",
)
plt.loglog(
    n_steps_vect,
    np.abs(sympl_error_4d_taper_norm),
    "-v",
    label=r"BorisSolenoid tapered, $||R_{4d}^T S_{4d} R_{4d} - S_{4d}||_2 / ||R_{4d}||_2^2$",
)
plt.loglog(
    n_steps_vect,
    np.abs(sympl_error_spatial_norm),
    "--o",
    label=r"BorisSpatial, $||R^T S R - S||_2 / ||R_{4d}||_2^2$",
)
plt.loglog(
    n_steps_vect,
    np.abs(sympl_error_4d_spatial_norm),
    "--s",
    label=r"BorisSpatial, $||R_{4d}^T S_{4d} R_{4d} - S_{4d}||_2 / ||R_{4d}||_2^2$",
)
plt.loglog(
    n_steps_vect,
    np.abs(sympl_error_varsol_norm),
    "-.^",
    label=rf"VariableSolenoid, $||R^T S R - S||_2 / ||R_{{4d}}||_2^2$ ($\leq${VARSOL_MAX_SLICES} slices)",
)
plt.loglog(
    n_steps_vect,
    np.abs(sympl_error_4d_varsol_norm),
    "-.x",
    label=rf"VariableSolenoid, $||R_{{4d}}^T S_{{4d}} R_{{4d}} - S_{{4d}}||_2 / ||R_{{4d}}||_2^2$",
)
plt.xlabel("Number of steps")
plt.ylabel("Normalised symplectic deviation")
plt.xlim(n_steps_vect[0] / 2, n_steps_vect[-1])
plt.legend(fontsize=8)

fig3 = plt.figure(3, figsize=(6.4, 4.8))
plt.loglog(n_steps_vect, np.abs(np.abs(det_R) - 1), "-o", label="BorisSolenoid")
plt.loglog(
    n_steps_vect,
    np.abs(np.abs(det_R_taper) - 1),
    "-s",
    label="BorisSolenoid tapered",
)
plt.xlabel("Number of steps")
plt.ylabel(r"$| |\det R| - 1 |$ (volume preservation)")
plt.legend()

plt.show()
