"""Second stage of the IDEA digitization (note 17): turn the raw output of
17a_digitize_idea_opera_plot.py into IDEA_solenoid_beam_axis_raw.json
(metadata, flags, under_logo points, rounding).

Usage: python 17b_build_idea_json.py <17a output .json> <IDEA_solenoid_beam_axis_raw.json>
"""
import json, sys, numpy as np
d = json.load(open(sys.argv[1])); out_path = sys.argv[2]
cal = d['_calibration']
legend = {'main_only': ('Main Solenoid Only', 'green'),
          'screening_only': ('Screening Only', 'red'),
          'both': ('Both Energized', 'black')}
curves = {}
for k, (lab, col) in legend.items():
    z, b, s, f = (np.array(d[k][q]) for q in ('z', 'Bz', 'sigma_Bz', 'flag'))
    o = np.lexsort((-b if k != 'screening_only' else b, z))   # z, then along the curve
    z, b, s, f = z[o], b[o], s[o], f[o].astype(object)
    xpx = z * cal['px_per_m'] + cal['x0_px']
    ul = (xpx >= cal['logo_x_px'][0]) & (xpx <= cal['logo_x_px'][1])
    f[ul] = 'under_logo'
    s[ul] = np.maximum(s[ul], 3.0 / cal['px_per_T'])
    curves[k] = {'legend': lab, 'plot_colour': col,
                 'z': [round(float(v), 5) for v in z], 'Bz': [round(float(v), 5) for v in b],
                 'sigma_Bz': [round(float(v), 5) for v in s], 'flag': [str(v) for v in f]}
    print(k, len(z), f'z {z.min():.4f}..{z.max():.4f}', {u: int((f == u).sum()) for u in np.unique(f)},
          'non-monotonic z steps:', int((np.diff(z) < 0).sum()), 'repeated z:', int((np.diff(z) == 0).sum()))
out = {
  'description': ('On-axis longitudinal field Bz(z) of the IDEA detector main solenoid and '
                  'screening solenoid, digitized from an Opera plot "Bz (T) vs Z (m)" with three '
                  'curves: Main Solenoid Only, Screening Only, Both Energized. One side of the IP, '
                  'z = 0..6 m. Raw digitization: one point per pixel column (per pixel row in steep '
                  'parts), not smoothed or resampled.'),
  'source': {'figure': 'Opera plot, Bz (T) vs Z (m), z = 0..6 m (2 T at z = 0)',
             'provided_by': 'S. Buijsman, 2026-11-15 (screenshot, 2021 x 1205 px)',
             'axis': ('not stated on the plot; file named "beam_axis" as requested. '
                      'The 15 mrad tilt between detector and beam axes changes on-axis Bz only at '
                      'second order in the offset.'),
             'digitized': '2026-11-15'},
  'units': {'z': 'm', 'Bz': 'T', 'sigma_Bz': 'T'},
  'curves': curves,
  'flags': {
    'visible': 'centre of the fully visible line',
    'partial': ('line partly covered by another curve, a gridline or the logo; centre from the '
                'visible edge(s) and the line thickness, sigma_Bz includes the ambiguity'),
    'hidden_under_screening_only': 'line fully covered by the red curve; Bz taken from it',
    'hidden_under_both': 'line fully covered by the black curve; Bz taken from it',
    'under_logo': ('inside the Opera logo box (z >= 5.735 m): only slivers between the letters '
                   'are visible; sigma_Bz >= 3 px')},
  'notes': [
    ('Sign: "Screening Only" is plotted positive, but Both = Main - Screening_plotted '
     '(rms 5 mT, max 19 mT over z = 0..5.75 m; Main + Screening_plotted is off by 1.7 T rms). '
     'The screening solenoid\'s on-axis field is therefore -Bz of the screening_only curve, '
     'assuming linear superposition (the Opera model may contain iron).'),
    ('Main Solenoid Only is hidden under Screening Only for z ~ 1.9-4.1 m (the curves coincide '
     'there within the line width); see the flags.'),
    ('Gap: the Opera logo box covers z >= 5.735 m. Both Energized and Screening Only end at '
     'z = 5.73 m (black curve and black letters cannot be separated; the box darkens the red '
     'line); Main Solenoid Only has a few under_logo points up to z = 5.95 m. All three curves '
     'are below ~0.01 T there and overlap each other within ~8 px for z > 5.5 m.'),
    ('Sampling: one point per pixel column, and one per pixel row where the curve is steep '
     '(Both: z ~ 1.2-1.6 m; Screening Only: ~1.25-1.6 and ~2.5-3.15 m; Main Solenoid Only: '
     '~2.55-3.1 m). z is non-decreasing but repeats there (several rows share the '
     'same column centre); Bz is monotonic within those runs. Interpolate Bz(z) only after '
     'merging repeated z, or treat the points as a parametric curve.'),
    ('Accuracy: 1 px = 3.4 mm in z and 2.2 mT in Bz. Calibration residuals <= 0.7 px. '
     'Line thickness ~6 px (~13 mT); visible points are good to ~0.5 px (~1 mT), '
     'see sigma_Bz per point. In the steep edge (z ~ 1.3-1.5 m) a 1 px error in z '
     'is up to ~30 mT in Bz.')],
  'digitization': {
    'method': ('Colour masks per curve (green/red/black); black curve separated from the '
               '(black) gridlines by a 3x3 morphological opening; column scan where '
               '|dBz/dz| is small, row scan where the curve is steep; occlusion handled by '
               'fitting a band of the measured line thickness that covers all visible pixels '
               'and extends only into adjacent non-background pixels.'),
    'calibration': {'x_px_at_z0': round(cal['x0_px'], 3), 'px_per_m': round(cal['px_per_m'], 4),
                    'y_px_at_Bz0': round(cal['y0_px'], 3), 'px_per_T': round(cal['px_per_T'], 4),
                    'gridlines_z': cal['z_lines'], 'gridlines_x_px': [round(v, 2) for v in cal['x_lines_px']],
                    'gridlines_Bz': cal['bz_lines'], 'gridlines_y_px': [round(v, 2) for v in cal['y_lines_px']],
                    'max_abs_residual_px_x_y': [round(v, 2) for v in cal['max_abs_residual_px']]},
    'line_thickness_px': {k: round(d[k]['thickness_px'], 2) for k in legend}},
}
json.dump(out, open(out_path, 'w'), indent=1)
import os; print('wrote', out_path, os.path.getsize(out_path) // 1024, 'kB')
