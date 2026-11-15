"""
Plot the digitized on-axis Bz(z) of the IDEA main and screening solenoids
(IDEA_solenoid_beam_axis_raw.json): Main Solenoid Only, Screening Only and
Both Energized, z = 0..6 m on one side of the IP.

Top: the three curves as in the source plot. Bottom: zoom on the tails
(|Bz| <= 0.05 T) with the per-point digitization uncertainty. Points that
were not fully visible in the source plot (partial, hidden under another
curve, under the Opera logo) are marked with open circles.

The 'Screening Only' curve is drawn as plotted in the source (positive);
the screening solenoid's own on-axis field is -Bz of that curve (see the
notes in the JSON).
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from aperture_study_io import PLOT_DIR, pdf_metadata

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / 'IDEA_solenoid_beam_axis_raw.json'

with open(DATA_FILE) as fid:
    data = json.load(fid)

curves = data['curves']
labels = {
    'main_only': 'Main Solenoid Only',
    'screening_only': 'Screening Only (as plotted; field is $-B_z$)',
    'both': 'Both Energized',
}

fig, (ax_full, ax_zoom) = plt.subplots(
    2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

for key in ['main_only', 'screening_only', 'both']:
    cc = curves[key]
    z = np.array(cc['z'])
    bz = np.array(cc['Bz'])
    sigma = np.array(cc['sigma_Bz'])
    not_visible = np.array(cc['flag']) != 'visible'
    color = cc['plot_colour']

    for ax in (ax_full, ax_zoom):
        ax.plot(z, bz, '-', color=color, lw=1.5, label=labels[key])
        ax.plot(z[not_visible], bz[not_visible], 'o', ms=2.5, mfc='none',
                mec=color, mew=0.6)
    ax_zoom.fill_between(z, bz - sigma, bz + sigma, color=color, alpha=0.25,
                         lw=0)

ax_full.plot([], [], 'o', ms=4, mfc='none', mec='gray',
             label='not fully visible in source plot')
ax_full.set_ylabel(r'$B_z$ [T]')
ax_full.set_ylim(-0.05, 2.1)
ax_full.legend(loc='upper right', fontsize=9)
ax_full.set_title('IDEA solenoids, on-axis $B_z$ (digitized)')

ax_zoom.axhline(0, color='gray', lw=0.8)
ax_zoom.set_ylim(-0.05, 0.05)
ax_zoom.set_ylabel(r'$B_z$ [T]')
ax_zoom.set_xlabel('z [m]')
ax_zoom.set_xlim(0, 6)

for ax in (ax_full, ax_zoom):
    ax.grid(True, alpha=0.4)

fig.tight_layout()

PLOT_DIR.mkdir(parents=True, exist_ok=True)
out_path = PLOT_DIR / 'IDEA_solenoid_beam_axis_raw.pdf'
fig.savefig(out_path, bbox_inches='tight', metadata=pdf_metadata())
print(f'Saved {out_path}')

plt.show()
