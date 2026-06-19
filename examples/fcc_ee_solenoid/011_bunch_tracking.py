from pathlib import Path

import matplotlib.pyplot as plt
import xtrack as xt

plt.close("all")

HERE = Path(__file__).resolve().parent
# Same SplineBoris lattice as 010a_dynamic_aperture_splineboris.py
# (from 004b_install_solenoids_in_fcc_ring.py and
# 004c_correct_solenoids_in_fcc_ring.py).
INPUT_LATTICE_JSON = (
    HERE / "fccee_z_lcc_splineboris_solenoids_coupling_corrected.json"
)

IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]


def _set_solenoid_knobs(line, *, with_solenoids, with_correctors):
    for ip_name in IP_NAMES:
        if f"on_sol_{ip_name}" in line.vars:
            line[f"on_sol_{ip_name}"] = float(with_solenoids)

        for corr_knob in (
            f"on_sol_corr_{ip_name}",
            f"on_comp_sol_{ip_name}",
            f"on_rot_doublet_left_{ip_name}",
            f"on_rot_doublet_right_{ip_name}",
            f"on_sol_orbit_corr_{ip_name}",
            f"on_sol_optics_corr_{ip_name}",
            f"on_sol_coupling_corr_{ip_name}",
        ):
            if corr_knob in line.vars:
                line[corr_knob] = float(with_correctors)


print(f"\n=== SplineBoris: solenoids powered + correction scheme ===")
print(f"Loading lattice: {INPUT_LATTICE_JSON.name}")
env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring
line.cycle("ipa")
_set_solenoid_knobs(
    line,
    with_solenoids=True,
    with_correctors=True,
)

tw = line.twiss6d(strengths=True)

print(
    f"Tune: Qx={tw['qx']:.6f}, Qy={tw['qy']:.6f}, "
    f"Qs={tw['qs']:.6f}"
)
print(
    f"Chromaticity: dQx/ddelta={tw['dqx']:.4f}, "
    f"dQy/ddelta={tw['dqy']:.4f}"
)
print(
    f"Beta at start: betx={tw['betx'][0]:.3f} m, "
    f"bety={tw['bety'][0]:.3f} m"
)
print(
    f"Dispersion at start: dx={tw['dx'][0]*1e3:.3f} mm, "
    f"dy={tw['dy'][0]*1e3:.3f} mm"
)

tw.plot("delta", lattice=True)
plt.show()
