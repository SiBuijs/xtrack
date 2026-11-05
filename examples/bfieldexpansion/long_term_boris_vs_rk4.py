"""Long-term behaviour of the Boris and RK4 integrators of BFieldExpansion.

The spatial Boris step preserves phase-space volume exactly: each of its
substeps (drift or sector bend, kick, rotation about Bs) has unit Jacobian
determinant, for any step size. It is not symplectic, however, when the
longitudinal field varies along s: the 4D volume is kept, but area can move
between the two transverse planes. The classical RK4 step is neither volume
preserving nor symplectic, but its errors decrease as ds**4.

This example closes one curved BFieldExpansion (s-dependent solenoid,
quadrupole and skew quadrupole on top of the design dipole) into a ring with
a linear one-turn map (xt.LineSegmentMap, no radiation, no RF), and tracks a
small 4D particle cloud for many turns. Every --chunk turns it records the
two eigen-emittances of the cloud and their product, the 4D volume
sqrt(det Sigma). For a linear map, the eigen-emittances of a cloud are
exactly invariant if the map is symplectic, and the volume is invariant if
the map preserves volume, so any change is an integrator effect, independent
of the number of particles. The cloud is small, to keep the nonlinear part of
the expansion negligible.

Before tracking, each configuration's one-turn matrix (from twiss4d) is
analysed: det(M) - 1, the symplecticity defect |M^T J M - J|, and the moduli
of its eigenvalues. An eigenvalue with |lambda| != 1 makes its mode's
emittance change by |lambda|**(2 N) after N turns; the prediction is printed
next to the tracked result.

Defaults (1e5 turns, 100 particles; Boris 20 steps and RK4 5 steps, which
cost about the same per pass, and RK4 20 steps) take about 15 minutes on one
CPU core.

Usage:
    python long_term_boris_vs_rk4.py [--num-turns N] [--num-particles P]
        [--chunk C] [--no-plot] [--save-prefix PATH]
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

import xtrack as xt

P0C = 2e9  # protons
H, LENGTH = 0.4, 1.0
CONFIGS = (('boris', 20), ('rk4', 5), ('rk4', 20))
GEOMETRIC_EMITTANCES = (2e-10, 1e-10)  # mode 1, mode 2

J4 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])


def make_element(integrator, num_integration_steps):
    # Design dipole + quadrupole and skew-quadrupole profiles + a solenoid
    # ramp (whose radial fringe field makes Boris non-symplectic). No
    # sextupole, to keep the one-turn map close to linear at small amplitude.
    return xt.BFieldExpansion(
        length=LENGTH, h=H, k0=H,
        knc=[[0.0], [0.3, -0.2, 0.1]],
        ksc=[[0.], [0.05, 0.05]],
        ksolc=[0.1, 0.3, -0.3],
        integrator=integrator, num_integration_steps=num_integration_steps)


def make_line(integrator, num_integration_steps):
    arc = xt.LineSegmentMap(length=10., qx=0.31, qy=0.27,
                            betx=3., bety=2.)
    line = xt.Line(elements={'bfe': make_element(integrator, num_integration_steps),
                             'arc': arc})
    line.particle_ref = xt.Particles(p0c=P0C)
    return line


def eigen_emittances(coords):
    """Eigen-emittances (ascending) and 4D volume of a cloud (4, n)."""
    sigma = np.cov(coords)
    eigs = np.linalg.eigvals(sigma @ J4)
    emit = np.sort(np.abs(eigs.imag))[::2]
    return emit, np.sqrt(np.linalg.det(sigma))


def analyse_one_turn_matrix(tw):
    M = tw.R_matrix[:4, :4]
    moduli = np.sort(np.abs(np.linalg.eigvals(M)))
    return dict(det_m1=np.linalg.det(M) - 1,
                sympl=np.abs(M.T @ J4 @ M - J4).max(),
                moduli=moduli)


def run_config(integrator, num_integration_steps, num_turns, num_particles,
               chunk, seed=2026):
    line = make_line(integrator, num_integration_steps)
    line.build_tracker()
    tw = line.twiss4d()
    if not (np.isfinite(tw.qx) and np.isfinite(tw.qy)):
        raise RuntimeError('one-turn map is not stable')
    analysis = analyse_one_turn_matrix(tw)

    rng = np.random.default_rng(seed)
    beta_gamma = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
    particles = line.build_particles(
        particle_on_co=tw.particle_on_co, W_matrix=tw.W_matrix[0],
        x_norm=rng.normal(size=num_particles), px_norm=rng.normal(size=num_particles),
        y_norm=rng.normal(size=num_particles), py_norm=rng.normal(size=num_particles),
        nemitt_x=GEOMETRIC_EMITTANCES[0] * beta_gamma,
        nemitt_y=GEOMETRIC_EMITTANCES[1] * beta_gamma,
        method='4d')

    def snapshot():
        coords = np.array([particles.x, particles.px, particles.y, particles.py])
        return eigen_emittances(coords)

    turns, emits, volumes = [0], [], []
    emit, volume = snapshot()
    emits.append(emit)
    volumes.append(volume)
    start = perf_counter()
    for _ in range(num_turns // chunk):
        line.track(particles, num_turns=chunk)
        if np.any(particles.state <= 0):
            raise RuntimeError(f'particles lost with {integrator} n={num_integration_steps}')
        turns.append(turns[-1] + chunk)
        emit, volume = snapshot()
        emits.append(emit)
        volumes.append(volume)
    elapsed = perf_counter() - start
    return dict(turns=np.array(turns), emits=np.array(emits),
                volumes=np.array(volumes), analysis=analysis, elapsed=elapsed,
                qx=tw.qx, qy=tw.qy)


def make_plot(results):
    import matplotlib.pyplot as plt

    colors = {('boris', 20): '#eb6834', ('rk4', 5): '#2a78d6', ('rk4', 20): '#1baf7a'}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3), sharex=True,
                            constrained_layout=True)
    panels = ((lambda r: r['emits'][:, 0], 'eigen-emittance, mode 1'),
              (lambda r: r['emits'][:, 1], 'eigen-emittance, mode 2'),
              (lambda r: r['volumes'], '4D volume = product of both'))
    for ax, (get, title) in zip(axs, panels):
        for key, r in results.items():
            values = get(r)
            ax.plot(r['turns'], values / values[0] - 1, lw=2,
                    color=colors.get(key), label=f'{key[0]}, {key[1]} steps')
        ax.set_title(title, fontsize=11, loc='left')
        ax.set_xlabel('turn')
        ax.grid(True, color='#e5e5e2', lw=0.8)
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    axs[0].set_ylabel('relative change since turn 0')
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Curved BFieldExpansion closed by a linear map, no radiation: '
                 'small 4D cloud', fontsize=11)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num-turns', type=int, default=100_000)
    parser.add_argument('--num-particles', type=int, default=100)
    parser.add_argument('--chunk', type=int, default=500,
                        help='turns between emittance snapshots (default: 500)')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--save-prefix', type=Path)
    args = parser.parse_args()
    if args.num_turns < args.chunk or args.chunk < 1:
        parser.error('need num_turns >= chunk >= 1')

    results = {}
    for integrator, nstep in CONFIGS:
        print(f'{integrator} n={nstep}: tracking {args.num_particles} particles '
              f'for {args.num_turns} turns...', flush=True)
        r = run_config(integrator, nstep, args.num_turns, args.num_particles,
                       args.chunk)
        results[(integrator, nstep)] = r
        a = r['analysis']
        n_turns = r['turns'][-1]
        # Each mode's emittance scales as |lambda|^(2N). The moduli come in
        # complex-conjugate pairs; they are sorted by size, not matched to
        # the emittance modes (which are sorted by emittance).
        predicted = np.sort(a['moduli'][::2] ** (2 * n_turns) - 1)
        measured_emit = r['emits'][-1] / r['emits'][0] - 1
        measured_vol = r['volumes'][-1] / r['volumes'][0] - 1
        print(f'   Qx={r["qx"]:.6f} Qy={r["qy"]:.6f}; tracking time {r["elapsed"]:.0f} s')
        print(f'   one-turn matrix: det-1={a["det_m1"]:+.2e}, '
              f'|M^T J M - J|={a["sympl"]:.2e}, |lambda|-1={a["moduli"] - 1}')
        print(f'   after {n_turns} turns: emittance change {measured_emit} '
              f'(predicted from |lambda|^2N - 1, unmatched: {predicted}), volume change {measured_vol:+.2e}',
              flush=True)

    if not args.no_plot or args.save_prefix:
        fig = make_plot(results)
        if args.save_prefix:
            args.save_prefix.parent.mkdir(parents=True, exist_ok=True)
            path = args.save_prefix.with_name(args.save_prefix.name + '.png')
            fig.savefig(path, dpi=150)
            print(f'Saved {path}')
        if not args.no_plot:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == '__main__':
    main()
