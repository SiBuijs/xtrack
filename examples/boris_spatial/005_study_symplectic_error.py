import numpy as np

import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

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

n_steps_vect = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

sympl_error_boris = []
sympl_error_boris_corrected = []
for n_steps in n_steps_vect:
    print('n_steps=', n_steps)
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

sympl_error_boris = np.array(sympl_error_boris)
sympl_error_boris_corrected = np.array(sympl_error_boris_corrected)

import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 4.8))
plt.loglog(n_steps_vect, sympl_error_boris, '-o', label=rf'BorisSpatialIntegrator, kinetic $p_x,p_y$')
plt.loglog(n_steps_vect, sympl_error_boris_corrected, '-o',
           label=rf'BorisSpatialIntegrator, canonical $p_x,p_y$')
plt.loglog(n_steps_vect[2:],
           sympl_error_boris_corrected[2] * n_steps_vect[2]**2
           / np.array(n_steps_vect[2:])**2, '--', label=r'~ 1/$N_\text{steps}^2$')
plt.xlabel('Number of steps')
plt.ylabel(r'Symplectic deviation $||R^TSR-S||$')
plt.title('Boris Symplectic Error Through Analytical Solenoid')
plt.legend()
plt.grid(True)

plt.show()
