from pathlib import Path

import xtrack as xt
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
INPUT_LATTICE_SOL_JSON = (
    HERE / "fccee_z_lcc_splineboris_solenoids_coupling_corrected.json"
)
INPUT_LATTICE_NO_SOL_JSON = HERE / "fccee_z_lcc.json"

IP_NAMES = ["ipa", "ipd", "ipg", "ipj"]

# Set to True for the solenoid lattice, False for the bare ring.
WITH_SOLENOIDS = True

if WITH_SOLENOIDS:
    lattice_json = INPUT_LATTICE_SOL_JSON
    with_correctors = True
    title = "Ring with solenoids + correction scheme"
else:
    lattice_json = INPUT_LATTICE_NO_SOL_JSON
    with_correctors = False
    title = "Ring without solenoids (no solenoid correctors)"

print(f"\n=== {title} ===")
print(f"Loading lattice: {lattice_json.name}")
env = xt.load(lattice_json)
line = env.fccee_p_ring
line.cycle("ipa")

for ip_name in IP_NAMES:
    if f"on_sol_{ip_name}" in line.vars:
        line[f"on_sol_{ip_name}"] = float(WITH_SOLENOIDS)

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

tw = line.twiss()

tw.plot('delta', lattice=True)
plt.show()
