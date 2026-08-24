"""
Symplectic error of BorisSolenoid (kinetic vs. canonical transverse momentum)
as a function of integration window length, at constant steps-per-unit-length.

Companion to examples/boris_spatial/005_study_symplectic_error.py (which used
the pure-Python BorisSpatialIntegrator + SolenoidField). Same study, but using
the compiled BorisSolenoid element and its `canonical` flag instead of a
separate vector_potential_callable.
"""

import numpy as np

import xtrack as xt

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                energy0=45.6e9 / 1000,
                x=[-1e-3,],
                px=-1e-3,
                y=1e-3,
                delta=0)

S = xt.linear_normal_form.S

L_coil = 4.0
a = 0.3
B0 = 1.5

s_start = 0
windows = [15, 30, 60]  # element length; solenoid re-centered in each window

# Reference: these step counts apply to a window of `reference_length`;
# for other window lengths, n_steps is scaled so steps-per-unit-length is constant.
reference_length = 30
n_steps_vect_ref = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
steps_per_length_vect = np.array(n_steps_vect_ref) / reference_length

sympl_error_boris_per_window = {}
sympl_error_boris_corrected_per_window = {}
for s_end in windows:
    z0 = (s_start + s_end) / 2
    length = s_end - s_start

    sympl_error_boris = []
    sympl_error_boris_corrected = []
    for steps_per_length in steps_per_length_vect:
        n_steps = int(round(steps_per_length * length))
        print(f's_end={s_end}, steps_per_length={steps_per_length:.3g}, n_steps={n_steps}')

        solenoid = xt.BorisSolenoid(L_coil=L_coil, a=a, B0=B0, z0=z0,
                                     length=length, n_steps=n_steps,
                                     canonical=False)
        line = xt.Line(elements=[solenoid])
        R_boris = line.get_R_matrix(particle_on_co=p0.copy())['R_matrix']
        se = np.linalg.norm(R_boris.T @ S @ R_boris - S, ord=2)
        sympl_error_boris.append(se)

        solenoid_corrected = xt.BorisSolenoid(L_coil=L_coil, a=a, B0=B0, z0=z0,
                                               length=length, n_steps=n_steps,
                                               canonical=True)
        line_corrected = xt.Line(elements=[solenoid_corrected])
        R_boris_corrected = line_corrected.get_R_matrix(
            particle_on_co=p0.copy())['R_matrix']
        se_corrected = np.linalg.norm(
            R_boris_corrected.T @ S @ R_boris_corrected - S, ord=2)
        sympl_error_boris_corrected.append(se_corrected)

    sympl_error_boris_per_window[s_end] = np.array(sympl_error_boris)
    sympl_error_boris_corrected_per_window[s_end] = np.array(sympl_error_boris_corrected)

import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 4.8))
for s_end in windows:
    sympl_error_boris = sympl_error_boris_per_window[s_end]
    sympl_error_boris_corrected = sympl_error_boris_corrected_per_window[s_end]
    line, = plt.loglog(steps_per_length_vect, sympl_error_boris, '-o',
               label=rf'window [{s_start}, {s_end}], kinetic $p_x,p_y$')
    plt.loglog(steps_per_length_vect, sympl_error_boris_corrected, '-o',
               color=line.get_color(), markerfacecolor='none',
               label=rf'window [{s_start}, {s_end}], canonical $p_x,p_y$')
    plt.loglog(steps_per_length_vect[2:],
               sympl_error_boris_corrected[2] * steps_per_length_vect[2]**2
               / steps_per_length_vect[2:]**2, '--', color=line.get_color())
plt.loglog([], [], 'k--', label=r'~ 1/$N_\text{steps/length}^2$')
plt.xlabel('Steps per unit length')
plt.ylabel(r'Symplectic deviation $||R^TSR-S||$')
plt.title('BorisSolenoid Symplectic Error vs. Integration Window')
plt.legend()
plt.grid(True)

plt.show()
