import time

import numpy as np

import xtrack as xt


QE = 1.602176634e-19
C_LIGHT = 299792458.0


def _const_spline(value, length):
    return xt.Spline4(
        val_start=value,
        der_start=0.0,
        val_end=value,
        der_end=0.0,
        integral=value * length,
    )


def _build_line(tracking_method, length, n_steps):
    element = xt.SplineBoris(
        bs=_const_spline(0.0, length),
        by=(
            _const_spline(0.85, length),
            xt.Spline4(0.10, 0.0, -0.05, 0.0, 0.01),
        ),
        bx=(
            _const_spline(0.15, length),
            xt.Spline4(-0.08, 0.0, 0.04, 0.0, -0.008),
        ),
        length=length,
        n_steps=n_steps,
        tracking_method=tracking_method,
    )
    line = xt.Line(elements=[element], element_names=["sb"])
    line.particle_ref = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1,
        p0c=3.0e9,
    )
    line.build_tracker(use_prebuilt_kernels=False)
    return line


def _track_and_time(line, particle, num_turns):
    t0 = time.perf_counter()
    line.track(particle, num_turns=num_turns, turn_by_turn_monitor=True)
    elapsed = time.perf_counter() - t0
    return elapsed, line.record_last_track


def _stack_states(record):
    return np.vstack([
        np.asarray(record.x[0]),
        np.asarray(record.px[0]),
        np.asarray(record.y[0]),
        np.asarray(record.py[0]),
        np.asarray(record.zeta[0]),
    ])


def _theory_defect_first_turn(sb, particle_ref, initial, n_steps):
    h = sb.length / n_steps
    x0 = initial["x"]
    y0 = initial["y"]
    px0_norm = initial["px"]
    py0_norm = initial["py"]
    delta0 = initial["delta"]

    p0_si = float(np.ravel(np.asarray(particle_ref.p0c))[0]) * QE / C_LIGHT
    p_si = p0_si * (1.0 + delta0)
    px_si = px0_norm * p0_si
    py_si = py0_norm * p0_si
    pz_si = np.sqrt(max(p_si * p_si - px_si * px_si - py_si * py_si, 0.0))
    q_si = float(np.ravel(np.asarray(particle_ref.q0))[0]) * QE

    bx, by, bz = sb.get_field(x0, y0, 0.5 * sb.length)
    kappa = q_si * bz / pz_si if pz_si != 0 else 0.0
    pref = (h**3) / 12.0

    dx_step = pref * (-(kappa * kappa / pz_si) * px_si + (q_si * kappa * bx / pz_si))
    dy_step = pref * (-(kappa * kappa / pz_si) * py_si + (q_si * kappa * by / pz_si))
    dpx_step_si = pref * (q_si * kappa * kappa * by)
    dpy_step_si = pref * (-q_si * kappa * kappa * bx)

    # Crude accumulated prediction over one element pass (n_steps identical defects).
    dx_turn = n_steps * dx_step
    dy_turn = n_steps * dy_step
    dpx_turn = (n_steps * dpx_step_si) / p0_si
    dpy_turn = (n_steps * dpy_step_si) / p0_si
    return np.array([dx_turn, dpx_turn, dy_turn, dpy_turn], dtype=float), (bx, by, bz, kappa)


def main():
    length = 0.5
    n_steps = 24
    num_turns = 2000

    initial = dict(
        x=2.2e-4,
        px=1.5e-4,
        y=-1.1e-4,
        py=0.9e-4,
        zeta=3e-4,
        delta=1.0e-3,
    )

    line_boris = _build_line(xt.SplineBoris.METHOD_BORIS, length=length, n_steps=n_steps)
    line_helical = _build_line(xt.SplineBoris.METHOD_HELICAL, length=length, n_steps=n_steps)

    p_boris = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1,
        p0c=3.0e9,
        **initial,
    )
    p_helical = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1,
        p0c=3.0e9,
        **initial,
    )

    boris_time, boris_record = _track_and_time(line_boris, p_boris, num_turns=num_turns)
    helical_time, helical_record = _track_and_time(line_helical, p_helical, num_turns=num_turns)

    boris_states = _stack_states(boris_record)
    helical_states = _stack_states(helical_record)
    diff = helical_states - boris_states
    first_turn_diff = diff[:4, 0]
    theory_first_turn, (bx0, by0, bz0, kappa0) = _theory_defect_first_turn(
        line_boris["sb"], line_boris.particle_ref, initial, n_steps
    )

    rms_delta = np.sqrt(np.mean(diff * diff, axis=1))
    final_delta = np.abs(diff[:, -1])

    labels = ("x", "px", "y", "py", "zeta")

    print("Benchmark: SplineBoris Boris vs Helical")
    print(f" turns={num_turns}, n_steps={n_steps}, length={length} m")
    print(f" method_flag_boris={line_boris['sb'].tracking_method}, method_flag_helical={line_helical['sb'].tracking_method}")
    print(f" runtime_boris   = {boris_time:.6f} s")
    print(f" runtime_helical = {helical_time:.6f} s")
    print(f" speed_ratio_helical_over_boris = {helical_time / boris_time:.3f}")
    print(" state deltas (helical - boris):")
    for ll, rr, ff in zip(labels, rms_delta, final_delta):
        print(f"  {ll:>4s}: rms={rr:.6e}, final={ff:.6e}")
    print(" first-turn defect check (helical - boris vs DeltaM estimate):")
    print(f"  field(midpoint@start): Bx={bx0:.6e} T, By={by0:.6e} T, Bz={bz0:.6e} T, kappa={kappa0:.6e} 1/m")
    for name, obs, pred in zip(("x", "px", "y", "py"), first_turn_diff, theory_first_turn):
        ratio = obs / pred if pred != 0 else np.nan
        print(f"  {name:>4s}: observed={obs:.6e}, predicted={pred:.6e}, obs/pred={ratio:.6e}")

    # Keep teardown explicit; avoids noisy context-buffer warnings on interpreter exit.
    line_boris.discard_tracker()
    line_helical.discard_tracker()


if __name__ == "__main__":
    main()