"""
Symplectic error study for the SplineBoris representation of a solenoid.

Mirrors examples/boris_spatial/005_study_symplectic_error.py: sweeps the
number of integration steps and looks at how the one-turn map's deviation
from symplecticity shrinks. Here the field is not tracked directly through
BorisSpatialIntegrator (as in 005) but through the piecewise-polynomial
SplineBoris representation built from FieldFitter/SplineBorisSequence, using
the same analytical solenoid as examples/splineboris/006_solenoid.py.

The BorisSpatialIntegrator curve from 005 (canonical px, py) is kept as a
gold-standard reference, since it tracks the analytical field directly
instead of a piecewise-polynomial fit of it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack._temp.splineboris.field_fitter import FieldFitter
from xtrack._temp.splineboris.splineboris_sequence import SplineBorisSequence

# ----------------------------------------------------------------------
# Reference particle and solenoid (same as boris_spatial/005 and
# splineboris/006)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Fit the analytical field to piecewise polynomials (as in 006_solenoid.py).
# This is the expensive, n_steps-independent part of the pipeline, so the
# fit parameters are cached to disk and only recomputed if missing.
# ----------------------------------------------------------------------
multipole_order = 2
dx = 0.001
dy = 0.001
n_fit_points = 5000

RESULTS_DIR = Path(__file__).resolve().parent / "_008_results"
RESULTS_DIR.mkdir(exist_ok=True)
fit_pars_path = RESULTS_DIR / "solenoid_fit_pars.csv"

if fit_pars_path.exists():
    print(f"Loading cached solenoid field fit from {fit_pars_path}")
    df_fit_pars = pd.read_csv(
        fit_pars_path,
        index_col=list(SplineBorisSequence.FIELD_FIT_INDEX_COLUMNS),
    )
else:
    print(f"No cached fit found at {fit_pars_path}, fitting solenoid field...")
    z_point_count = n_fit_points + 1
    x_axis = np.linspace(-multipole_order * dx / 2, multipole_order * dx / 2, multipole_order + 1)
    y_axis = np.linspace(-multipole_order * dy / 2, multipole_order * dy / 2, multipole_order + 1)
    z_axis = np.linspace(s_start, s_end, z_point_count)
    x_grid, y_grid, z_grid = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    bx, by, bz = sf.get_field(x_grid.ravel(), y_grid.ravel(), z_grid.ravel())

    df_raw_data = pd.DataFrame(
        np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), bx, by, bz]),
        columns=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
    ).set_index(["X", "Y", "Z"])

    fitter = FieldFitter(
        raw_data=df_raw_data,
        xy_point=(0, 0),
        distance_unit=1,
        min_region_size=10,
        deg=multipole_order - 1,
        field_tol=1e-4,
    )
    df_fit_pars = fitter.df_fit_pars
    fitter.save_fit_pars(fit_pars_path)
    print(f"Saved field fit to {fit_pars_path}")

# ----------------------------------------------------------------------
# Sweep steps_per_point and measure the symplectic error of the resulting
# SplineBoris line (n_steps grows with steps_per_point).
# ----------------------------------------------------------------------
steps_per_point_vect = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

n_steps_splineboris = []
sympl_error_splineboris = []
for steps_per_point in steps_per_point_vect:
    print('steps_per_point=', steps_per_point)
    seq = SplineBorisSequence(
        df_fit_pars=df_fit_pars,
        multipole_order=multipole_order,
        steps_per_point=steps_per_point,
    )
    line = seq.to_line()

    total_steps = sum(int(elem.n_steps) for elem in seq.elements)
    n_steps_splineboris.append(total_steps)

    R_obj = line.get_R_matrix(particle_on_co=p0.copy(), include_collective=True)
    R_splineboris = R_obj['R_matrix']
    se = np.linalg.norm(R_splineboris.T @ S @ R_splineboris - S, ord=2)
    sympl_error_splineboris.append(se)

n_steps_splineboris = np.array(n_steps_splineboris)
sympl_error_splineboris = np.array(sympl_error_splineboris)

# ----------------------------------------------------------------------
# Gold-standard reference: BorisSpatialIntegrator tracking the analytical
# field directly, with the canonical (vector-potential corrected) momenta
# (same curve as in boris_spatial/005_study_symplectic_error.py).
# ----------------------------------------------------------------------
n_steps_vect = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

sympl_error_boris_corrected = []
for n_steps in n_steps_vect:
    print('n_steps (BorisSpatialIntegrator)=', n_steps)
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

sympl_error_boris_corrected = np.array(sympl_error_boris_corrected)

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 4.8))
plt.loglog(n_steps_splineboris, sympl_error_splineboris, '-o',
           label='SplineBoris solenoid (canonical $p_x,p_y$)')
plt.loglog(n_steps_vect, sympl_error_boris_corrected, '-o',
           label='BorisSpatialIntegrator, canonical $p_x,p_y$ (reference)')
plt.loglog(n_steps_vect[2:],
           sympl_error_boris_corrected[2] * n_steps_vect[2]**2
           / np.array(n_steps_vect[2:])**2, '--', label=r'~ 1/$N_\text{steps}^2$')
plt.xlabel('Number of steps')
plt.ylabel(r'Symplectic deviation $||R^TSR-S||$')
plt.title('SplineBoris Symplectic Error Through Analytical Solenoid')
plt.legend()
plt.grid(True)

plt.show()
