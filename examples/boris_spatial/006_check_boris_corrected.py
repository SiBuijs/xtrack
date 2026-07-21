import numpy as np

import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from scipy.constants import c as clight
from scipy.constants import e as qe

# Tests the g(theta) = (theta/2)*cot(theta/2) drift/kick rescaling
# hypothesis in `xtrack.BorisSpatialCorrected` (see claude_notes/) against
# plain `xtrack.BorisSpatialIntegrator`, both with the canonical/kinetic
# vector-potential correction (see 005) so that bug is not a confound.
#
# Also checks orbit accuracy (not just symplecticity) of both methods
# against a fixed-step RK4 reference integration of the exact spatial
# equations of motion (see `rk4_deriv`/`rk4_track` below).

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                energy0=45.6e9 / 1000,
                x=[-1e-3,],
                px=-1e-3,
                y=1e-3,
                delta=0)

sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)
s_start = 0
s_end = 30

S = xt.linear_normal_form.S


def rk4_deriv(z, x, y, px, py, P, q, field_fn):
    # Exact spatial equations of motion for a magnetic-only field,
    # parametrized by z (path length along the reference trajectory),
    # with kinetic momentum p = (px, py, pz), |p| = P constant:
    #   dr/dz = p / pz,   dp/dz = (q/pz) * (p x B)
    pz = np.sqrt(np.maximum(P**2 - px**2 - py**2, 0.0))
    Bx, By, Bz = field_fn(np.array([x]), np.array([y]), np.array([z]))
    dxdz = px / pz
    dydz = py / pz
    dpxdz = q * (py * Bz[0] / pz - By[0])
    dpydz = q * (Bx[0] - px * Bz[0] / pz)
    return dxdz, dydz, dpxdz, dpydz


def rk4_track(x0, y0, z0, px0, py0, P, q, z_end, n_steps, field_fn):
    # Fixed-step classical RK4 integration of `rk4_deriv`, used as an
    # independent ("ground truth") reference trajectory, not derived from
    # the D-K-R-K-D splitting the Boris integrators are built on.
    dz = (z_end - z0) / n_steps
    x, y, px, py, z = x0, y0, px0, py0, z0
    for _ in range(n_steps):
        k1 = rk4_deriv(z, x, y, px, py, P, q, field_fn)
        k2 = rk4_deriv(z + dz / 2, x + dz / 2 * k1[0], y + dz / 2 * k1[1],
                        px + dz / 2 * k1[2], py + dz / 2 * k1[3], P, q, field_fn)
        k3 = rk4_deriv(z + dz / 2, x + dz / 2 * k2[0], y + dz / 2 * k2[1],
                        px + dz / 2 * k2[2], py + dz / 2 * k2[3], P, q, field_fn)
        k4 = rk4_deriv(z + dz, x + dz * k3[0], y + dz * k3[1],
                        px + dz * k3[2], py + dz * k3[3], P, q, field_fn)
        x += dz / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y += dz / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        px += dz / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        py += dz / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        z += dz
    return x, y, z, px, py


# --- Kinetic initial condition, matching what BorisSpatialIntegrator.track
# --- feeds to the step function internally (canonical -> kinetic shift).
charge_coulomb0 = p0.charge[0] * qe
P0_0 = p0.p0c[0] * qe / clight
P_0 = P0_0 * (1 + p0.delta[0])
Px0_canonical = p0.px[0] * P0_0
Py0_canonical = p0.py[0] * P0_0
Ax0, Ay0, _ = sf.get_vector_potential(
    np.array([p0.x[0]]), np.array([p0.y[0]]), np.array([s_start], dtype=float))
Px0_kin = Px0_canonical - charge_coulomb0 * Ax0[0]
Py0_kin = Py0_canonical - charge_coulomb0 * Ay0[0]

n_steps_rk4_ref = 100_000
x_ref, y_ref, z_ref, _, _ = rk4_track(
    p0.x[0], p0.y[0], s_start, Px0_kin, Py0_kin, P_0, charge_coulomb0,
    s_end, n_steps_rk4_ref, sf.get_field)
print(f'RK4 reference (n_steps={n_steps_rk4_ref}): x={x_ref:.9e}  y={y_ref:.9e}')

n_steps_vect = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

sympl_error_boris = []
sympl_error_boris_corrected = []
orbit_error_boris = []
orbit_error_boris_corrected = []
for n_steps in n_steps_vect:
    print('n_steps=', n_steps)
    integrator = xt.BorisSpatialIntegrator(
        fieldmap_callable=sf.get_field,
        vector_potential_callable=sf.get_vector_potential,
        s_start=s_start, s_end=s_end, n_steps=n_steps)
    line = xt.Line(elements=[integrator])
    R_obj = line.get_R_matrix(particle_on_co=p0.copy(),
                               include_collective=True)
    R_boris = R_obj['R_matrix']
    se = np.linalg.norm(R_boris.T @ S @ R_boris - S, ord=2)
    sympl_error_boris.append(se)

    p_boris = p0.copy()
    integrator.track(p_boris)
    orbit_error_boris.append(
        np.sqrt((p_boris.x[0] - x_ref)**2 + (p_boris.y[0] - y_ref)**2))

    integrator_corrected = xt.BorisSpatialCorrected(
        fieldmap_callable=sf.get_field,
        vector_potential_callable=sf.get_vector_potential,
        s_start=s_start, s_end=s_end, n_steps=n_steps)
    line_corrected = xt.Line(elements=[integrator_corrected])
    R_obj_corrected = line_corrected.get_R_matrix(particle_on_co=p0.copy(),
                                                    include_collective=True)
    R_boris_corrected = R_obj_corrected['R_matrix']
    se_corrected = np.linalg.norm(
        R_boris_corrected.T @ S @ R_boris_corrected - S, ord=2)
    sympl_error_boris_corrected.append(se_corrected)

    p_boris_corrected = p0.copy()
    integrator_corrected.track(p_boris_corrected)
    orbit_error_boris_corrected.append(
        np.sqrt((p_boris_corrected.x[0] - x_ref)**2
                 + (p_boris_corrected.y[0] - y_ref)**2))

sympl_error_boris = np.array(sympl_error_boris)
sympl_error_boris_corrected = np.array(sympl_error_boris_corrected)
orbit_error_boris = np.array(orbit_error_boris)
orbit_error_boris_corrected = np.array(orbit_error_boris_corrected)

import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 4.8))
plt.loglog(n_steps_vect, sympl_error_boris, '-o',
           label='BorisSpatialIntegrator')
plt.loglog(n_steps_vect, sympl_error_boris_corrected, '-o',
           label=r'BorisSpatialCorrected ($g(\theta)$ drift/kick rescaling)')
n_steps_ref = np.array(n_steps_vect[2:], dtype=float)
plt.loglog(n_steps_vect[2:],
           sympl_error_boris[2] * n_steps_vect[2]**2 / n_steps_ref**2,
           '--', label=r'~ 1/$N_\text{steps}^2$')
plt.loglog(n_steps_vect[2:],
           sympl_error_boris[2] * n_steps_vect[2]**4 / n_steps_ref**4,
           ':', label=r'~ 1/$N_\text{steps}^4$')
plt.xlabel('Number of steps')
plt.ylabel(r'Symplectic deviation $||R^TSR-S||$')
plt.title('BorisSpatialIntegrator vs BorisSpatialCorrected')
plt.legend()
plt.grid(True)

fig2 = plt.figure(2, figsize=(6.4, 4.8))
plt.loglog(n_steps_vect, orbit_error_boris, '-o',
           label='BorisSpatialIntegrator')
plt.loglog(n_steps_vect, orbit_error_boris_corrected, '-o',
           label=r'BorisSpatialCorrected ($g(\theta)$ drift/kick rescaling)')
plt.loglog(n_steps_vect[2:],
           orbit_error_boris[2] * n_steps_vect[2]**2 / n_steps_ref**2,
           '--', label=r'~ 1/$N_\text{steps}^2$')
plt.xlabel('Number of steps')
plt.ylabel(f'Orbit error vs. RK4 reference ($N={n_steps_rk4_ref}$) [m]')
plt.title('Orbit accuracy: Boris methods vs. RK4 ground truth')
plt.legend()
plt.grid(True)

plt.show()
