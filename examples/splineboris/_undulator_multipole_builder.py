import numpy as np
import xtrack as xt
from xtrack._temp.splineboris.splineboris_sequence import SplineBorisSequence


def _compute_brho(p_ref):
    p0c_eV = float(np.atleast_1d(p_ref.p0c)[0])
    q0 = abs(float(np.atleast_1d(p_ref.q0)[0]))
    # B*rho [T*m] = p[GeV/c] / (0.299792458 * |q/e|)
    return (p0c_eV * 1e-9) / (0.299792458 * q0)


def _extract_multipole_strengths(seq, s_positions, multipole_order, brho, dx=1e-6):
    n_eval = multipole_order + 2
    x_eval = np.linspace(-n_eval // 2 * dx, n_eval // 2 * dx, n_eval)
    n_orders = multipole_order

    ds = np.diff(s_positions)
    s_mid = 0.5 * (s_positions[:-1] + s_positions[1:])
    n_slices = len(s_mid)

    knl = np.zeros((n_slices, n_orders))
    ksl = np.zeros((n_slices, n_orders))

    for i, s in enumerate(s_mid):
        by_vals = np.array([seq.get_field(x_pt, 0.0, s)[1] for x_pt in x_eval])
        bx_vals = np.array([seq.get_field(x_pt, 0.0, s)[0] for x_pt in x_eval])

        coeff_by = np.polyfit(x_eval, by_vals, n_orders - 1)
        coeff_bx = np.polyfit(x_eval, bx_vals, n_orders - 1)

        for n in range(n_orders):
            dby_n = np.polyval(np.polyder(coeff_by, m=n), 0.0)
            dbx_n = np.polyval(np.polyder(coeff_bx, m=n), 0.0)
            knl[i, n] = dby_n * ds[i] / brho
            ksl[i, n] = dbx_n * ds[i] / brho

    return knl, ksl, ds


def build_multipole_kick_undulator(
    env,
    p_ref,
    df_fit_pars,
    multipole_order=3,
    shift_x=0.0,
    shift_y=0.0,
    n_slices=None,
    name_prefix="und_kick",
):
    seq = SplineBorisSequence(
        df_fit_pars=df_fit_pars,
        multipole_order=multipole_order,
        steps_per_point=1,
        shift_x=shift_x,
        shift_y=shift_y,
    )

    if n_slices is None:
        n_slices = sum(int(ee.n_steps) for ee in seq.elements)

    df_reset = df_fit_pars.reset_index()
    s_start = float(df_reset["s_start"].min())
    s_end = float(df_reset["s_end"].max())
    s_positions = np.linspace(s_start, s_end, n_slices + 1)

    brho = _compute_brho(p_ref)
    knl, ksl, ds = _extract_multipole_strengths(
        seq=seq,
        s_positions=s_positions,
        multipole_order=multipole_order,
        brho=brho,
    )

    element_names = []
    for ii in range(len(ds)):
        if ii == 0:
            drift_length = ds[ii] / 2
            drift_name = f"{name_prefix}_drift_entry"
        else:
            drift_length = (ds[ii - 1] + ds[ii]) / 2
            drift_name = f"{name_prefix}_drift_{ii}"

        kick_name = f"{name_prefix}_kick_{ii}"

        env.new(drift_name, xt.Drift, length=float(drift_length))
        env.new(
            kick_name,
            xt.Multipole,
            knl=knl[ii, :].tolist(),
            ksl=ksl[ii, :].tolist(),
        )
        element_names.extend([drift_name, kick_name])

    exit_name = f"{name_prefix}_drift_exit"
    env.new(exit_name, xt.Drift, length=float(ds[-1] / 2))
    element_names.append(exit_name)

    line = xt.Line(env=env, element_names=element_names)
    line.particle_ref = p_ref.copy()
    return line, seq.length
