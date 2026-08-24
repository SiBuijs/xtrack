import numpy as np

import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                energy0=45.6e9 / 1000,
                x=[-1e-3,],
                px=-1e-3,
                y=1e-3,
                delta=0)

S = xt.linear_normal_form.S

s_start = 0
windows = [15, 30, 60]  # s_end values; solenoid re-centered in each window

# Reference: these step counts apply to a window of `reference_length`;
# for other window lengths, n_steps is scaled so steps-per-unit-length is constant.
reference_length = 30
n_steps_vect_ref = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]#, 500000, 1000000]
steps_per_length_vect = np.array(n_steps_vect_ref) / reference_length

sympl_error_boris_per_window = {}
sympl_error_boris_corrected_per_window = {}
for s_end in windows:
    z0 = (s_start + s_end) / 2
    sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=z0)
    length = s_end - s_start

    sympl_error_boris = []
    sympl_error_boris_corrected = []
    for steps_per_length in steps_per_length_vect:
        n_steps = int(round(steps_per_length * length))
        print(f's_end={s_end}, steps_per_length={steps_per_length:.3g}, n_steps={n_steps}')
        integrator = xt.BorisSpatialIntegrator(fieldmap_callable=sf.get_field,
                                                s_start=s_start,
                                                s_end=s_end,
                                                n_steps=n_steps)
        line = xt.Line(elements=[integrator])
        R_obj = line.get_R_matrix(particle_on_co=p0.copy(),
                                   include_collective=True)
        R_boris = R_obj['R_matrix']
        se = np.linalg.norm(R_boris.T @ S @ R_boris - S, ord=2)
        sympl_error_boris.append(se)

        integrator_corrected = xt.BorisSpatialIntegrator(
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
plt.title('Boris Symplectic Error vs. Integration Window')
plt.legend()
plt.grid(True)

plt.show()
